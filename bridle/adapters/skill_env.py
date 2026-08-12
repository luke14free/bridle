"""bridle.adapters.skill_env — a compiled `RewardPlan`, running inside a ManiSkill env.

WHAT THIS IS: the `RewardPlan --> ManiSkill env` arrow of `skill.yaml --schema--> SkillSpec
--compile()--> RewardPlan --> env --> preflight --> PPO`. Everything to the LEFT of this file
decides; this file only MEASURES (read a quantity off the simulator) and APPLIES (fold the plan's
ops over those readings). That split is why `bridle/skill/**` is stdlib-only and testable in
milliseconds on a laptop, and why torch is confined to `bridle/adapters/`.

THE PER-TERM ARITHMETIC IS NOT RE-IMPLEMENTED HERE. `compile.evaluate_plan`'s `_VALUE` table already
spells out what every one of the 9 terms computes, and a second transcription in torch is how the
two halves of a reward stop agreeing — the stdlib evaluator a unit test asserts on and the CUDA
evaluator PPO actually trains against would drift apart silently, each one "correct". So this module
CALLS that table (`_TERM_VALUE` below) with batched values and only supplies (a) the readings and
(b) the ordered fold's `torch.where` selection. The readings are handed over as RAW tensors: see the
note above `_values_for` for the wrapper that used to be needed and the compile.py fix that removed
the need for it.

WHAT IS NOT SHARED, AND WHY: the fold loop itself. `evaluate_plan` selects with
`_where(c, a, b) = c*a + (1-c)*b`, which is right for a scalar and right for a tensor, but the plan
for this file specifies `torch.where` — an actual elementwise select, with no `0 * inf = nan` corner
and no arithmetic on the accumulator for rows that do not apply. The ORDER, the kinds, and the
condition/level split all come from the plan, so the only thing transcribed is the three-line
selection, and `test_skillcompile.py`'s fold fixtures pin the semantics it has to match.

BATCHING IS THE BINDING CONSTRAINT (phase2-decisions, global constraints). Every reading is a `(N,)`
tensor over up to 4096 parallel environments, and there is no Python loop over environments and no
Python `if` on a batched condition anywhere below — a `bool()` on a batch takes ONE branch for all
4096 envs at once, which is not a slower right answer but a different reward.

TORCH IS IMPORTED LAZILY, INSIDE THE FUNCTIONS THAT NEED IT, the same way `adapters/preflight.py`
does it — and for the same measured reason. A module-scope `import torch` made this file
unimportable on a box without torch, so no file in `bridle/tests/` could import it (checked
2026-08-13: `grep -rl adapters.skill_env bridle/tests/` was empty), and the two key-set asserts at
the bottom of the MEASURE/PREDICATE tables — free guards that cost nothing to run — never ran in the
suite that actually runs. The import is cached after the first call, so the per-call cost is a dict
lookup, not a re-import.
THE PREDICATE MINI-LANGUAGE LIVES NEXT DOOR, in `bridle/adapters/skill_predicates.py`: the `ast`
whitelist, the `all[...]`/`any[...]` desugarer, argument resolution and the sixteen predicates. It
is a parser/evaluator rather than an env adapter, it re-implements the discipline `expr.py` already
owns, and it is the part that grows when quantifiers land. It knows nothing about an env beyond the
`MeasureContext` handed to it, so the dependency runs one way — this module imports it, never the
reverse — which is why `SkillEnvError` and `_norm` are defined there and re-exported here.
"""
import dataclasses
import warnings as warn

from bridle.adapters.skill_predicates import (
    PREDICATE_FNS, SkillEnvError, _desugar_brackets, _eval_predicate_text, _f, _norm, _up_axis,
)
from bridle.skill.compile import RewardPlan
from bridle.skill.compile import _CONDITION_LEVEL as _TERM_CONDITION_LEVEL
from bridle.skill.compile import _SCOPE_REACH
from bridle.skill.compile import _VALUE as _TERM_VALUE
from bridle.skill.compile import _bind_text
from bridle.skill.vocab import MEASURES, PREDICATES, Sign

__all__ = [
    "MEASURE_FNS", "PREDICATE_FNS", "SIGNED_MEASURES", "EnvBinding", "DEFAULT_BINDING",
    "MeasureContext", "StateSlots", "SkillEnvError", "build_reward_fn", "build_success_fn",
    "build_reset_fn", "SkillRuntime",
]
#: Re-exported so `skill_env.PREDICATE_FNS` and `except skill_env.SkillEnvError` keep resolving to
#: the SAME objects after the split — this is a file move, not a second implementation. Named here
#: rather than left as bare imports so a linter does not delete them as unused.
_REEXPORTED = (PREDICATE_FNS, SkillEnvError, _desugar_brackets, _eval_predicate_text, _f, _norm,
               _up_axis)


# ── how a skill's nouns map onto one env ────────────────────────────────────────────────────────

@dataclasses.dataclass(frozen=True)
class EnvBinding:
    """Which attribute of THIS env each abstract noun in the vocabulary reads.

    The `scene:` block is parsed and fingerprinted but NOT synthesised into an env class in this
    phase (phase2-decisions, scope limit), so nothing connects `scene.held` to `env.cube` yet. Until
    that exists the binding is supplied by the caller, and every default below is the SO100
    `grasp_cube` family's actual attribute name (`mani_skill/envs/tasks/digital_twins/so100_arm/
    grasp_cube.py`), not a guess.

    A field defaulting to None is a reading the vocabulary gives the author no channel to specify —
    `joint_qpos` names "a named joint" but a measure reference is a bare string with nowhere to put
    the name. Those RAISE `SkillEnvError` unless bound, rather than picking a joint.
    """

    #: The held/target object actor. `Actor.merge(cubes, name="cube")`, grasp_cube.py:406.
    held: str = "cube"
    #: (N,3) goal point. grasp_cube.py:429.
    goal: str = "target_pos"
    #: (N,) or scalar half-extent of the held object, the offset between a surface and a resting
    #: object's CENTER z. grasp_cube.py:332.
    half: str = "cube_half_sizes"
    #: Top surface z of the DESTINATION seat. grasp_cube.py:418 (`platform_top_z` = 0.03).
    seat_top: str = "platform_top_z"
    #: Surface the held object's OWN natural resting height is measured from — the table at z=0 in
    #: every env in this corpus (`_initialize_episode` spawns the cube at `xyz[:,2] = half`,
    #: grasp_cube.py:574). A skill whose object rests on the platform binds `platform_top_z` here.
    #: See `_m_height_above_resting` for why this is a binding and not a constant.
    resting_surface_z: float | str = 0.0
    #: A goal frozen once, mid-episode, that does NOT track the live scene — descend_stack's
    #: `_stack_goal`. Absent on an env that has no stacking phase, in which case the static goal is
    #: seeded from the live seat at reset, which for a bolted-down platform IS the frozen goal.
    static_goal: str | None = "_stack_goal"
    #: Finger links whose net contact force `contact_force` sums. reach_env.py:121.
    finger_links: tuple = ("Fixed_Jaw", "Moving_Jaw")
    #: A NON-held scene object, for `scene_object_xy_drift`. No default: "the scene object" is not a
    #: thing an env has one of.
    scene_object: str | None = None
    #: Which joint `joint_qpos` reads — an index into `get_qpos()`, or a joint name.
    joint: int | str | None = None
    #: Target yaw for `yaw_diff_mod_symmetry`: a number, or an env attribute holding (N,). 0.0 is
    #: the platform's yaw in every env of this family (compact_grasp_env._coord_scene builds it as
    #: `yaw=torch.zeros(N)`), so it is a measured default rather than a placeholder.
    goal_yaw: float | str = 0.0
    #: Rotational symmetry of the held object, radians. pi/2 = a box, which is reach_grab's
    #: `_cube_yaw_folded` and compact_grasp's box branch (`remainder(raw + pi/4, pi/2) - pi/4`).
    yaw_symmetry: float = 1.5707963267948966


DEFAULT_BINDING = EnvBinding()


# ── per-env buffers ─────────────────────────────────────────────────────────────────────────────

class StateSlots:
    """Per-env buffers for everything a reward reads that is not in the current frame.

    Four kinds live here and they are all the same problem: ProgressPotential's previous distance,
    an `at_reset` measure's anchor pose, `latched`/`sustained`'s accumulators, and
    `action_delta_norm`'s previous action.

    ── THE PARTIAL-RESET RULE, AND THE CRASH IT PREVENTS ──
    `seed()` writes ONLY the rows named by `env_idx`. ManiSkill's own setters write only the scene's
    CURRENT reset mask, so they expect `(len(env_idx), ...)`; handing one a full `(num_envs, ...)`
    tensor raises

        shape mismatch: value tensor of shape [4096, 6] cannot be broadcast to indexing result of
        shape [1, 6]

    the instant ONE env resets on its own — which is every step under `--partial-reset`. The worked
    example of the correct convention is `primitives/descend_to_target/descend_env.py`'s
    `_initialize_episode` (the 2026-08-12 fix, with that exact message quoted at the call site).

    The second half is quieter and worse. Seeding a WHOLE tensor when one env restarts also
    overwrites the 4095 envs that did not: every in-flight ProgressPotential buffer jumps to its
    neighbour's distance, and the next step's `weight * (prev - measure)` pays out a one-step
    progress spike that no policy earned. `ProgressPotential.doc` states the rule
    ("seed ONLY the resetting rows' buffer entries") as something the framework must own; this is
    where it is owned.

    ALLOCATION-TIME SEEDING IS NOT AN EXCEPTION TO THAT RULE. `slot()` fills the whole buffer from
    `init` the first time it is asked for one, because at that moment no env has any history — there
    is nothing to erase. Every write after that goes through `seed(env_idx)`.
    """

    def __init__(self, num_envs=None, device=None, dtype=None):
        #: `dtype=None` means float32 and is resolved HERE rather than in the signature, because a
        #: `dtype=torch.float32` default would evaluate at class-definition time and drag the
        #: module-scope torch import back in (see the module docstring).
        import torch
        #: Both may be None until `ensure(env)` sees a live env. That is what lets ONE `StateSlots`
        #: be shared by the reward/success/reset callables before any of them has an env: built
        #: separately they would each allocate their own, and a `latched` success in the reward fold
        #: would be a different latch from the one the success criterion accumulates.
        self.num_envs = None if num_envs is None else int(num_envs)
        self.device = device
        self.dtype = torch.float32 if dtype is None else dtype
        self._buffers = {}

    @classmethod
    def for_env(cls, env):
        env = getattr(env, "unwrapped", env)
        return cls(env.num_envs, env.device)

    def ensure(self, env):
        env = getattr(env, "unwrapped", env)
        if self.num_envs is None:
            self.num_envs, self.device = int(env.num_envs), env.device
        elif self.num_envs != int(env.num_envs):
            raise SkillEnvError(f"these buffers were allocated for {self.num_envs} environments and "
                                f"this env has {int(env.num_envs)} — one StateSlots per env")
        return self

    def slot(self, name, init, width=None):
        """The buffer called `name`, allocated and seeded from `init()` on first request.

        `init` is a CALLABLE, not a value, so the seeding read only happens on the step the buffer is
        first needed — a plan that never reads a slot never touches the simulator for it.
        """
        import torch
        buffer = self._buffers.get(name)
        if buffer is None:
            seed = init()
            shape = (self.num_envs,) if width is None else (self.num_envs, width)
            buffer = torch.empty(shape, dtype=self.dtype, device=self.device)
            buffer.copy_(torch.as_tensor(seed, device=self.device, dtype=self.dtype).expand(shape))
            self._buffers[name] = buffer
        return buffer

    def has(self, name):
        return name in self._buffers

    def seed(self, name, env_idx, value):
        """Write `value` into the rows named by `env_idx` and NOWHERE else. See the class docstring
        for the crash and the spurious-progress-spike this is the fix for."""
        import torch
        buffer = self._buffers.get(name)
        if buffer is None:
            return
        buffer[env_idx] = torch.as_tensor(value, device=buffer.device, dtype=buffer.dtype)

    def fill_rows(self, name, env_idx, value):
        """Constant write, rows only — same partial-reset rule as `seed`."""
        buffer = self._buffers.get(name)
        if buffer is not None:
            buffer[env_idx] = float(value)

    def names(self):
        return tuple(self._buffers)

    def fresh_rows(self, name, elapsed):
        """A 0/1 mask of the rows that have NOT yet been advanced at this elapsed step.

        A stateful predicate (`sustained`, and any accumulator) can legitimately be read twice in one
        control step — once from `evaluate()` for the success criterion, once from
        `compute_dense_reward()` as a gate — and advancing its streak on both reads counts a single
        step twice, which makes `sustained(grasped, k=3)` latch in 2 steps. The guard is elementwise
        against the env's own per-env step counter, so it stays correct under partial reset where the
        counters differ across the batch.
        """
        import torch
        if elapsed is None:
            return None
        elapsed = elapsed.to(dtype=self.dtype, device=self.device)
        last = self.slot(f"{name}.tick", init=lambda: elapsed - 1.0)
        fresh = (last != elapsed).to(self.dtype)
        last.copy_(torch.where(fresh > 0, elapsed, last))
        return fresh


# ── the reading context ─────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class MeasureContext:
    """Everything a measure or predicate may read, for ONE control step of ONE env batch."""

    env: object
    obs: object
    action: object
    info: dict
    slots: StateSlots
    binding: EnvBinding = DEFAULT_BINDING
    #: Which builder made this context, named so a refusal can say which call has no action to read.
    #: `build_reset_fn` and `build_success_fn` both evaluate with `action=None`, and a message that
    #: names only one of them sends the reader to the wrong line (see `_action`).
    where: str = "this evaluation"

    def __post_init__(self):
        self.env = getattr(self.env, "unwrapped", self.env)
        self._measures = {}
        self._predicates = {}

    # ── env accessors ───────────────────────────────────────────────────────
    def attr(self, name, what):
        value = getattr(self.env, name, None)
        if value is None:
            raise SkillEnvError(
                f"{what} needs `env.{name}` and {type(self.env).__name__} does not have it. Bind a "
                f"different attribute on EnvBinding, or use a measure this env can supply")
        return value

    @property
    def held(self):
        return self.attr(self.binding.held, "the held/target object")

    @property
    def goal(self):
        return self.attr(self.binding.goal, "the goal point")

    @property
    def object_p(self):
        return self.held.pose.p

    @property
    def half(self):
        return self._as_batch(self.attr(self.binding.half, "the held object's half-extent"))

    @property
    def num_envs(self):
        return self.env.num_envs

    @property
    def device(self):
        return self.env.device

    @property
    def elapsed(self):
        return getattr(self.env, "elapsed_steps", None)

    def _as_batch(self, value):
        """Any scalar/(N,)/(N,1) env attribute, as a `(N,)` tensor on the env's device."""
        import torch
        t = torch.as_tensor(value, device=self.device, dtype=torch.float32)
        if t.dim() == 0:
            return t.expand(self.num_envs)
        return t.reshape(self.num_envs)

    def resolve_length(self, value, what):
        """A binding field that is either a number or the name of an env attribute."""
        import torch
        if isinstance(value, str):
            return self._as_batch(self.attr(value, what))
        return torch.full((self.num_envs,), float(value), device=self.device, dtype=torch.float32)

    # ── cached reads ────────────────────────────────────────────────────────
    def measure(self, name):
        if name not in self._measures:
            fn = MEASURE_FNS.get(name)
            if fn is None:
                raise SkillEnvError(f"no measure named {name!r} — legal: {', '.join(sorted(MEASURES))}")
            self._measures[name] = _check_batch(fn(self), name, self.num_envs)
        return self._measures[name]

    def predicate(self, text):
        if text not in self._predicates:
            self._predicates[text] = _eval_predicate_text(self, text)
        return self._predicates[text]


def _check_batch(value, name, num_envs):
    """Every reading is `(N,)` float. A measure that returns `(N,1)` or `(N,3)` broadcasts silently
    into an `(N,N)` reward the moment it meets another row, so the shape is asserted at the source."""
    import torch
    if not torch.is_tensor(value):
        raise SkillEnvError(f"measure {name!r} returned {type(value).__name__}, not a tensor")
    value = value.reshape(-1) if value.dim() == 2 and value.shape[1] == 1 else value
    if value.dim() != 1 or value.shape[0] != num_envs:
        raise SkillEnvError(f"measure {name!r} returned shape {tuple(value.shape)}, expected "
                            f"({num_envs},) — one reading per parallel environment")
    return value.float()


# ── measures ────────────────────────────────────────────────────────────────────────────────────
# One function per `vocab.MEASURES` key, each returning `(N,)`. SIGN AND FRAME ARE THE CONTRACT:
# a `Sign.SIGNED` measure returns the signed difference and never its magnitude (see
# `_m_height_above_seat_live`), and a `Frame.AT_RESET` / `Frame.STATIC_GOAL` measure reads a
# `StateSlots` anchor rather than the live scene.


def _m_tcp_to_object(ctx):
    return _norm(ctx.object_p - ctx.attr("agent", "the tcp").tcp_pos)


def _m_object_to_goal_xy(ctx):
    return _norm(ctx.object_p[..., :2] - ctx.goal[..., :2])


def _m_object_to_goal_z(ctx):
    return (ctx.object_p[..., 2] - ctx.goal[..., 2]).abs()


def _m_object_to_goal_xy_plus_z(ctx):
    """L1 composite `||dxy|| + |dz|`, NOT a 3D norm — move_over_bin's `_placement_distance` exactly,
    feeding its ProgressPotential at weight 10.0. The env's own method is called when it exists
    (grasp_cube.py:737) because that version routes the z-goal upward while xy is still far, and a
    plain `xy + |dz|` substitute would be a different potential over the same name."""
    placement = getattr(ctx.env, "_placement_distance", None)
    if callable(placement):
        return placement(ctx.object_p, ctx.goal, ctx.half)
    return _m_object_to_goal_xy(ctx) + _m_object_to_goal_z(ctx)


def _m_height_above_resting(ctx):
    """SIGNED: object z minus its OWN natural resting height (the surface beneath it, plus its half
    extent). `binding.resting_surface_z` defaults to the table at z=0 — `_initialize_episode` spawns
    the cube at `xyz[:,2] = cube_half_sizes` (grasp_cube.py:574), so that is the resting height lift's
    Ramp is measured against.

    IT IS A BINDING AND NOT A CONSTANT because the corpus uses one word for two surfaces: descend's
    source calls `platform_top_z + cube_half_sizes` "resting" (descend_env.py:184) and
    `Contract.release.height_above_resting` is descend's 0.015 hover above the PLATFORM. A skill
    whose object's resting surface is the platform binds `resting_surface_z="platform_top_z"`; the
    seat-relative reading has its own name (`height_above_seat_live`) and is never silently
    substituted here."""
    _warn_default_resting_frame(ctx)
    surface = ctx.resolve_length(ctx.binding.resting_surface_z, "the resting surface height")
    return ctx.object_p[..., 2] - (surface + ctx.half)


#: The `(type(env), resting_surface_z, seat_top)` triples this process has already decided about.
#:
#: WHY THE GUARD IS MEMOISED AND NOT JUST RE-EVALUATED. `_warn_default_resting_frame` is called from
#: inside `_m_height_above_resting`, so it ran once per `compute_dense_reward` for any plan naming
#: that measure — and its condition does `float(ctx._as_batch(seat).max())`, which SYNCHRONISES CUDA:
#: a host-device round trip inserted into the reward hot path of every step of every rollout. The
#: f-string argument to `warn.warn` was also built on every call, whatever the warning filter did
#: with it afterwards. Measured 2026-08-13 on the CPU fake: 5 reward steps -> 5 evaluations of the
#: guard, 5 seat reads. Nothing in the key can change within a rollout (the env's type is fixed, and
#: the binding is a frozen dataclass captured at build time), so the decision is taken once and the
#: hot path costs one set lookup after that.
_WARNED_RESTING_FRAME = set()


def _warn_default_resting_frame(ctx):
    """The default `resting_surface_z=0.0` is the TABLE. Say so, out loud, when the env has a seat.

    A document that writes `height_above_resting_in(0.01)` against descend and does not pass
    `resting_surface_z="platform_top_z"` measures the table frame and gets a band that is 3 cm out —
    `scripts/probe_skill_env.py` has to pass exactly that binding for its G4 check to agree with
    descend's own gate (35/35), and before this warning nothing said so. A silent wrong frame is the
    defect class the `Frame` tag on every measure exists to prevent, so it is not silent.

    It is a warning and not a refusal because the table frame is CORRECT for the measure's own cited
    source — lift's Ramp, whose object rests on the table (grasp_cube.py:574 spawns the cube at
    `xyz[:,2] = cube_half_sizes`). Refusing would reject the one lineage the measure was ported from.
    The condition is narrow on purpose: it fires only when the env publishes a raised seat, i.e. only
    when there are two candidate surfaces and the default silently picked one.

    IT DECIDES ONCE PER `(env type, resting_surface_z, seat_top)`, not once per step — see
    `_WARNED_RESTING_FRAME` for the CUDA sync that made the per-step version a hot-path cost. Warning
    once is also what `warnings`' own default filter would do with the message; what it would NOT do
    is skip building it.
    """
    if ctx.binding.resting_surface_z != 0.0:
        return
    key = (type(ctx.env), ctx.binding.resting_surface_z, ctx.binding.seat_top)
    if key in _WARNED_RESTING_FRAME:
        return
    #: Recorded BEFORE the seat is read, so the "this env publishes no seat" answer is memoised too:
    #: that branch is the one that pays the sync and then says nothing.
    _WARNED_RESTING_FRAME.add(key)
    seat = getattr(ctx.env, ctx.binding.seat_top, None)
    if seat is None or float(ctx._as_batch(seat).max()) <= 0.0:
        return
    warn.warn(
        f"`height_above_resting` is being measured against the TABLE (EnvBinding."
        f"resting_surface_z=0.0), but {type(ctx.env).__name__} publishes a raised seat "
        f"`{ctx.binding.seat_top}`. If the band you mean is above that seat, pass "
        f"EnvBinding(resting_surface_z={ctx.binding.seat_top!r}) — or use the measure that names "
        f"the seat frame, `height_above_seat_live`.", stacklevel=2)


def _seat_resting_z(ctx):
    return ctx._as_batch(ctx.attr(ctx.binding.seat_top, "the destination seat")) + ctx.half


def _m_height_above_seat_live(ctx):
    """SIGNED, and the sign is the whole point: + = above the seat, - = pressed INTO it.

    This one reading feeds two rows that disagree about direction — descend's hover attractor
    `2.5*(1 - tanh(6*|sdz - hover|))` and its crush penalty `-3.0*clamp(-sdz, min=0)`. Returning a
    magnitude here makes `clamp(-sdz, min=0)` identically zero, so the crush penalty still trains,
    still logs, and contributes nothing — and that term exists because pressing the cube to dz=0
    broke 16/16 grasps (2026-06-04, descend_env.py:82-87). `descend_env.py:185` computes exactly
    `cp[..., 2] - (platform_top_z + cube_half_sizes)`."""
    return ctx.object_p[..., 2] - _seat_resting_z(ctx)


def _m_height_above_seat_static_goal(ctx):
    """The same quantity against a goal FROZEN once, which does not track the live scene.

    descend_stack grades its reward against `self._stack_goal` while its `evaluate()` gates success
    on the LIVE top — one quantity, two frames, which is why `Measure.frame` exists (vocab.py,
    correction 2). When the env publishes such a goal it is read from there; otherwise the goal is
    frozen into a slot.

    WHEN THE FREEZE HAPPENS, exactly: `build_reset_fn` PRE-ALLOCATES this slot from the post-restore
    state whenever `plan.measures_needed` contains this measure, so the freeze point is the reset. It
    is not "at reset" by construction — `StateSlots.slot()` freezes on first READ, and `build_reset_fn`
    used to re-seed only a slot that already existed, so before 2026-08-13 episode 1's freeze point
    was the first `compute_dense_reward` call instead. Identical for a bolted-down platform, wrong for
    a live stack top that the arm has already begun to disturb. A plan reached through the free
    `build_reward_fn` with no reset hook installed still freezes on first read; there is nowhere else
    to put it."""
    import torch
    published = getattr(ctx.env, ctx.binding.static_goal or "", None)
    if published is not None:
        goal = torch.as_tensor(published, device=ctx.device, dtype=torch.float32)
        goal = goal[..., 2] if goal.dim() >= 2 and goal.shape[-1] == 3 else goal
        return ctx.object_p[..., 2] - ctx._as_batch(goal)
    frozen = ctx.slots.slot("frame.static_goal_seat_z", init=lambda: _seat_resting_z(ctx))
    return ctx.object_p[..., 2] - frozen


def _m_object_z(ctx):
    return ctx.object_p[..., 2]


def _m_gripper_qpos(ctx):
    """SIGNED. Closed sits ~-0.73, opening drifts toward 0 — descend_env.py:186 reads exactly
    `get_qpos()[..., -1]`, the last (jaw) joint."""
    return ctx.attr("agent", "the robot").robot.get_qpos()[..., -1]


def _m_contact_force(ctx):
    """Net contact force summed over the finger-pad links, in newtons. reach_env.py:121-122:
    `get_net_contact_forces(["Fixed_Jaw","Moving_Jaw"]).norm(dim=-1).sum(dim=-1)`."""
    robot = ctx.attr("agent", "the robot").robot
    forces = robot.get_net_contact_forces(list(ctx.binding.finger_links))
    return forces.norm(dim=-1).sum(dim=-1)


def _m_object_xy_drift_from_reset(ctx):
    """Frame.AT_RESET: xy displacement since the episode reset pose. The anchor is a slot seeded per
    env_idx, never as a whole tensor — a whole-tensor capture zeroes the accumulated drift of every
    env that did not restart (`vocab.MEASURES` says so at this measure)."""
    anchor = ctx.slots.slot("frame.object_xy0", init=lambda: ctx.object_p[..., :2], width=2)
    return _norm(ctx.object_p[..., :2] - anchor)


def _scene_object(ctx):
    name = ctx.binding.scene_object
    if name is None:
        raise SkillEnvError(
            "`scene_object_xy_drift` reads a NON-held scene object and nothing says which one: an "
            "env has no canonical 'the scene object'. Set EnvBinding(scene_object='<attr>') — e.g. "
            "'target_zone' on the SO100 platform envs")
    return ctx.attr(name, "the scene object")


def _m_scene_object_xy_drift(ctx):
    obj = _scene_object(ctx)
    anchor = ctx.slots.slot(f"frame.scene_xy0.{ctx.binding.scene_object}",
                            init=lambda: obj.pose.p[..., :2], width=2)
    return _norm(obj.pose.p[..., :2] - anchor)


def _m_object_linear_velocity(ctx):
    return _norm(ctx.held.linear_velocity)


def _m_object_angular_velocity(ctx):
    """CLAUDE.md gotcha (2): a GRASPED cube reads ~22 rad/s here while visibly rotating ~0.45 rad/s —
    ~98% contact-solver noise. VelocityPenalty shapes on it at weight 0.05 knowingly; nothing may
    gate SUCCESS on it."""
    return _norm(ctx.held.angular_velocity)


def _action(ctx):
    """The current action, or a refusal that names WHICH call has none.

    Two builders evaluate without an action and the fix differs between them, so the message cannot
    name just one: `build_success_fn` has no action because a success criterion is evaluated from
    `evaluate()` (move the row out of `success:`), and `build_reset_fn` has none because a reset
    happens between steps (a stateful row over an action measure cannot be re-anchored there —
    `action_delta_norm`'s buffer re-seeds itself on the next step instead). `MeasureContext.where`
    carries the caller so the reader is sent to the right line.
    """
    import torch
    if ctx.action is None:
        raise SkillEnvError(
            f"this row reads the action (`action_norm`/`action_delta_norm`) and {ctx.where} "
            f"evaluates without one. From `build_success_fn`: move the row out of the `success:` "
            f"criterion. From `build_reset_fn`: an action-valued row cannot be re-anchored at reset "
            f"— drop `reseed_on_restore`, or use a measure of the scene rather than of the action")
    return torch.as_tensor(ctx.action, device=ctx.device, dtype=torch.float32)


def _m_action_norm(ctx):
    return _norm(_action(ctx))


def _m_action_delta_norm(ctx):
    """`||a_t - a_{t-1}||` — the jerk-LIKE variant nine source files describe as "the jerk penalty"
    and none of them compute (no env has ever stored a previous action). Ships at chassis weight 0.0
    so enabling it is a deliberate sweep, not a silent parity break; the buffer it needs is why it
    could not just be added to the existing rows."""
    action = _action(ctx)
    previous = ctx.slots.slot("frame.prev_action", init=lambda: action, width=action.shape[-1])
    return _norm(action - previous)


def _yaw(q):
    """Raw planar yaw from a wxyz quaternion `(N,4)`. compact_grasp_env.py:81-82, verbatim."""
    import torch
    return torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                       1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))


def _m_yaw_diff_mod_symmetry(ctx):
    """Symmetry-reduced |yaw(object) - yaw(goal)|, folded into a half-period either side of zero.

    The FOLD IS ON THE DIFFERENCE, not on each yaw separately: folding the two angles independently
    and subtracting can report a full period of error for two orientations that are the same pose.
    `binding.yaw_symmetry` defaults to pi/2 (a box), which is reach_grab's `_cube_yaw_folded` and
    compact_grasp's box branch (`remainder(raw + pi/4, pi/2) - pi/4`). Not wired as a DistancePull
    target anywhere in the corpus — a 5-DoF arm generally cannot reach a full-pose target, so
    shaping on it generates reward hacking rather than behaviour (vocab.MEASURES)."""
    import torch
    goal_yaw = ctx.resolve_length(ctx.binding.goal_yaw, "the goal yaw")
    period = float(ctx.binding.yaw_symmetry)
    delta = _yaw(ctx.held.pose.q) - goal_yaw
    return (torch.remainder(delta + period / 2.0, period) - period / 2.0).abs()


def _m_joint_pos_margin_to_limit(ctx):
    """Smallest distance from any joint to its nearest hardware limit — the safety margin. Zero of
    the 99 audited reward rows use it (Amendment-A addition), so there is no source instance to copy;
    the reading is `min over joints of min(q - lower, upper - q)`, clamped at 0 because the measure
    is declared a MAGNITUDE and a joint driven past its limit by the solver is at margin 0, not at a
    negative distance."""
    import torch
    robot = ctx.attr("agent", "the robot").robot
    q = robot.get_qpos()
    limits = robot.get_qlimits()
    margin = torch.minimum(q - limits[..., 0], limits[..., 1] - q)
    return torch.clamp(margin.min(dim=-1).values, min=0.0)


def _m_joint_qpos(ctx):
    """SIGNED raw angle of ONE joint. Which joint is an `EnvBinding` field, not a measure parameter,
    because a measure is referenced by a bare string (`measure: joint_qpos`) with nowhere to put a
    name — a vocabulary gap, reported rather than papered over with a default index."""
    which = ctx.binding.joint
    if which is None:
        raise SkillEnvError(
            "`joint_qpos` reads a NAMED joint and the vocabulary gives a measure reference nowhere "
            "to name one. Set EnvBinding(joint=<index or joint name>), or use `gripper_qpos` if the "
            "joint you mean is the gripper")
    robot = ctx.attr("agent", "the robot").robot
    q = robot.get_qpos()
    if isinstance(which, int):
        return q[..., which]
    names = [j.name for j in robot.active_joints]
    if which not in names:
        raise SkillEnvError(f"EnvBinding(joint={which!r}) is not an active joint of this robot — "
                            f"legal: {', '.join(names)}")
    return q[..., names.index(which)]


MEASURE_FNS = {
    "tcp_to_object": _m_tcp_to_object,
    "object_to_goal_xy": _m_object_to_goal_xy,
    "object_to_goal_z": _m_object_to_goal_z,
    "object_to_goal_xy_plus_z": _m_object_to_goal_xy_plus_z,
    "height_above_resting": _m_height_above_resting,
    "height_above_seat_live": _m_height_above_seat_live,
    "height_above_seat_static_goal": _m_height_above_seat_static_goal,
    "object_z": _m_object_z,
    "gripper_qpos": _m_gripper_qpos,
    "contact_force": _m_contact_force,
    "object_xy_drift_from_reset": _m_object_xy_drift_from_reset,
    "scene_object_xy_drift": _m_scene_object_xy_drift,
    "object_linear_velocity": _m_object_linear_velocity,
    "object_angular_velocity": _m_object_angular_velocity,
    "action_norm": _m_action_norm,
    "action_delta_norm": _m_action_delta_norm,
    "yaw_diff_mod_symmetry": _m_yaw_diff_mod_symmetry,
    "joint_pos_margin_to_limit": _m_joint_pos_margin_to_limit,
    "joint_qpos": _m_joint_qpos,
}

#: Not "these look complete" — the key sets are compared. A measure in the vocabulary with no
#: implementation here would surface as a `KeyError` mid-rollout, after the GPU was spent; a
#: measure implemented here that the vocabulary does not declare is a name no document can write.
assert set(MEASURE_FNS) == set(MEASURES), (
    f"MEASURE_FNS and vocab.MEASURES disagree: missing "
    f"{sorted(set(MEASURES) - set(MEASURE_FNS))}, extra {sorted(set(MEASURE_FNS) - set(MEASURES))}")

#: The measures whose SIGN a caller may assert on. Kept as a derived set so a sign change in the
#: vocabulary shows up here rather than in a stale hand-written list.
SIGNED_MEASURES = frozenset(n for n, m in MEASURES.items() if m.sign is Sign.SIGNED)

#: Measures whose implementation ABOVE depends on the vocabulary still declaring them SIGNED, with
#: the measured consequence of it not doing so. This is what `SIGNED_MEASURES` is FOR — without a
#: consumer it was a derived set nobody read, which is the same as not deriving it.
#:
#: The failure it guards is a two-step one, and each step looks reasonable alone: the vocabulary
#: relabels a measure MAGNITUDE, and a later reader "fixes" the implementation to match by returning
#: `.abs()`. `_m_height_above_seat_live`'s docstring records what that costs — descend's crush
#: penalty `-3.0*clamp(-sdz, min=0)` becomes identically zero, so the term still trains, still logs,
#: and contributes nothing, and pressing the cube to dz=0 broke 16/16 grasps on 2026-06-04.
_SIGN_LOAD_BEARING = {
    "height_above_seat_live": "descend's crush penalty `-3.0*clamp(-sdz, min=0)` is identically "
                              "zero against a magnitude; pressing to dz=0 broke 16/16 grasps",
    "height_above_resting": "the `height_above_resting_in` band is `0 <= h <= band`, and a "
                            "magnitude makes the lower bound unfalsifiable",
    "gripper_qpos": "descend's grip-hold hinge penalises `qpos - (-0.6)` above zero; a magnitude "
                    "inverts which side of the jaw travel is penalised",
}
assert not (set(_SIGN_LOAD_BEARING) - SIGNED_MEASURES), (
    f"these measures are implemented as signed differences and the vocabulary no longer declares "
    f"them SIGNED: "
    f"{ {n: _SIGN_LOAD_BEARING[n] for n in sorted(set(_SIGN_LOAD_BEARING) - SIGNED_MEASURES)} }")


# ── the fold ────────────────────────────────────────────────────────────────────────────────────

_SUCCESS_KEYS = ("success", "success_latched")


def _custom_row(ctx, target):
    """Tier 3: an imported `module:function`, which the stdlib evaluator refuses by construction
    ("only the adapter can call it", `compile._v_custom`). Called with the same four arguments a
    ManiSkill `compute_dense_reward` receives, so an existing env method can be lifted into a skill
    document without rewriting it."""
    module_name, _, function_name = target.partition(":")
    import importlib
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise SkillEnvError(f"custom row {target!r}: cannot import {module_name!r} ({exc})") from exc
    fn = getattr(module, function_name, None)
    if not callable(fn):
        raise SkillEnvError(f"custom row {target!r}: {module_name!r} has no callable "
                            f"{function_name!r}")
    return _check_batch(fn(ctx.env, ctx.obs, ctx.action, ctx.info), target, ctx.num_envs)


def _success_value(ctx, plan):
    """The document's success criterion, as a 0/1 float `(N,)`.

    `info["success"]` FIRST, because that is the truth the env publishes and the one the deployed
    envs' `compute_dense_reward` reads (`descend_env.py:210`) — recomputing it here from the same
    state would be a second implementation of the same predicate, free to disagree. Only when the
    env publishes none is the op's own `condition` (the criterion, params already bound into it by
    `compile._lower_term_row`) evaluated.

    THIS IS NOT A CLAIM THAT THE DOCUMENT'S SUCCESS EQUALS THE ENV'S. It is the opposite: reading
    `info["success"]` is what makes the two agree by CONSTRUCTION on this path, which is why
    `scripts/probe_skill_env.py`'s reward-parity numbers (max abs diff 1.2e-07 / 2.4e-07 vs
    `descend_env.compute_dense_reward`) establish that the replace/floor MECHANICS agree — both sides
    consume the same `info["success"]` — and establish nothing about whether the document's
    `success:` line is the same predicate the env publishes. On the §4 fixture it measurably is not:
    the same probe's G5 finds `height_above_resting_in` and descend's `low` gate disagreeing on 29/29
    rows below the seat, because the predicate carries a `>= 0` lower bound the gate lacks. Task 6
    measures that disagreement separately; a preflight assert comparing `build_success_fn` against
    the env's own `evaluate()` is what would close it.
    """
    published = ctx.info.get("success") if isinstance(ctx.info, dict) else None
    if published is not None:
        return _f(published, ctx)
    condition = next((op.params["condition"] for op in plan.ops if "condition" in op.params), None)
    if condition is None:
        raise SkillEnvError("this plan needs a success signal and neither the env's info dict nor "
                            "the plan carries one")
    return ctx.predicate(condition)


def _values_for(ctx, plan):
    """Everything `compile._VALUE` will ask for, keyed exactly as it asks for it: measure names,
    the gate/predicate STRINGS as written in the document, the success signals, and one entry per
    state slot. Only what the plan declares is read — a measure no row names costs no sim call.

    THE VALUES ARE RAW TENSORS, and until 2026-08-13 they could not be. `compile.py`'s helpers are
    branch-free (`_where(c,a,b) = c*a + (1-c)*b`) precisely so one fold serves a Python float and a
    CUDA batch, but every one of them builds its condition from a comparison, a torch comparison
    yields a BOOL tensor, and

        >>> 1 - torch.tensor([True, False])
        RuntimeError: Subtraction, the `-` operator, with a bool tensor is not supported.

    so `compile._relu(torch.tensor([-1.0, 2.0]))` raised and `HingePenalty`/`Ramp` could not fold a
    bare tensor at all. This file carried a `_B` wrapper whose comparisons returned 0.0/1.0 floats to
    close that from the outside. `compile._numeric` (`c * 1`, commit f1da01b) closes it at the
    source, for every caller rather than for this one, so the wrapper was deleted: measured over the
    plans in `bridle/tests/test_skill_env_fold.py`, raw == wrapped BITWISE on all 62 dumped tensors
    (fold outputs, slot buffers, all 19 measures, all 15 evaluable predicates, and the whole
    `_values_for` payload). `_numeric` yields int64 where the wrapper yielded float32 for the
    CONDITION; that never reaches a result, because both branches of every `_where` in the fold are
    floats and int64 * float32 promotes to float32 — also measured, over every selection the suite
    performs.
    """
    import torch
    values = {}
    for name in plan.measures_needed:
        values[name] = ctx.measure(name)
    for op in plan.ops:
        for key in ("predicate", "gate"):
            text = op.params.get(key)
            if isinstance(text, str):
                values[text] = ctx.predicate(text)
        if op.fn_key == "expr":
            for name in op.params["expr"].names:
                if name in PREDICATES and name not in values:
                    values[name] = ctx.predicate(name)
        if op.stateful:
            slot = op.params["slot"]
            # `.clone()`: `_advance_state` writes the NEW measure into this same buffer after the
            # fold, and a live view would turn `prev - measure` into `measure - measure` = 0 for any
            # caller that evaluated the two in the other order.
            values[slot] = ctx.slots.slot(
                slot, init=lambda o=op: ctx.measure(o.params["measure"])).clone()

    needs_success = any(op.fn_key == "SuccessBonus" for op in plan.ops)
    if needs_success:
        success = _success_value(ctx, plan)
        values["success"] = success
        latch = ctx.slots.slot("success_latched", init=lambda: torch.zeros_like(success))
        latch.copy_(torch.maximum(latch, success))
        values["success_latched"] = latch.clone()
    return values


def _advance_state(ctx, plan):
    """Per-step buffer updates that belong to the adapter, not to the pure evaluator.

    ProgressPotential's `prev <- measure` is written AFTER the fold has read the old value, which is
    what makes the row a potential difference rather than a constant zero. `action_delta_norm`'s
    buffer follows the same order for the same reason.
    """
    for op in plan.ops:
        if op.stateful:
            slot = ctx.slots.slot(op.params["slot"], init=lambda o=op: ctx.measure(o.params["measure"]))
            slot.copy_(ctx.measure(op.params["measure"]))
    if ctx.slots.has("frame.prev_action") and ctx.action is not None:
        ctx.slots.slot("frame.prev_action", init=lambda: _action(ctx),
                       width=_action(ctx).shape[-1]).copy_(_action(ctx))


def _check_plan(plan):
    """Refuse, before any GPU is spent, a plan this fold cannot run — including one whose SCOPE it
    would otherwise honour by accident.

    The scope check is the point. `compile._SCOPE_REACH` is deliberately ONE table, read by
    `_HONOURED` (what compiles) and by `evaluate_plan` (what runs), so that a scope cannot be
    declared legal which the fold ignores — its own comment names the failure it exists to stop,
    `scope: all` accepted and quietly folded as `preceding`. This adapter is the third reader, and
    until 2026-08-13 it was not a reader at all: it folded against `acc` directly and validated only
    `fn_key`, so the moment a second entry landed in that table the plan would compile clean,
    `evaluate_plan` would honour the new scope, and the GPU fold would silently treat it as
    `preceding`. Measured with a second entry spliced into the table: `evaluate_plan` returned 0.0
    where this fold returned -5.0.
    """
    for op in plan.ops:
        if op.fn_key == "custom":
            # A custom row skips the mode/scope validation below because its VALUE is opaque — it is
            # an imported `module:function`, not a term with a condition/level split. Skipping the
            # validation is not the same as accepting any mode: the fold adds a custom row
            # unconditionally, so a declared `mode: replace` would be silently folded as `add`, which
            # is the same defect class as the scope one this function was written for. Refuse it.
            if op.kind != "add":
                raise SkillEnvError(
                    f"custom row {op.params.get('target')!r} declares `mode: {op.kind}` and this "
                    f"fold can only ADD one. A replace/floor row needs a (condition, level) split "
                    f"and a tier-3 row is one opaque number — express the criterion as a "
                    f"`SuccessBonus`/`PredicateBonus` row with `mode: {op.kind}`, and keep the "
                    f"custom row for the value it computes")
            continue
        if op.fn_key not in _TERM_VALUE:
            raise SkillEnvError(f"no evaluator for reward row kind {op.fn_key!r}")
        if op.kind != "add" and op.fn_key not in _TERM_CONDITION_LEVEL:
            raise SkillEnvError(f"row kind {op.kind!r} needs a condition/level split and "
                                f"{op.fn_key!r} has none")
        # `scope` is None for a plain `add` — the accumulator IS the preceding rows, so an add row
        # reaches nothing and carries no scope (`compile._lower_term_row`).
        if op.kind != "add" and op.scope not in _SCOPE_REACH:
            raise SkillEnvError(
                f"reward row {op.fn_key!r} declares scope={op.scope!r}, which this fold does not "
                f"implement — legal: {', '.join(sorted(_SCOPE_REACH))}. A scope reaches back into "
                f"the accumulator and every reader of it has to agree what it reaches; defaulting "
                f"an unknown one to `preceding` is the silent divergence "
                f"`compile._SCOPE_REACH` exists to prevent")


def build_reward_fn(plan: RewardPlan, *, binding: EnvBinding = DEFAULT_BINDING, slots=None):
    """`plan` -> `fn(env, obs, action, info) -> (N,) Tensor`, the UNSCALED dense reward.

    UNSCALED on purpose: `plan.scale` is the NORMALIZED path's divisor
    (`compute_normalized_dense_reward` returns `compute_dense_reward(...)/12.0`, grasp_cube.py:838),
    so a numerical-parity comparison against a deployed `compute_dense_reward` compares this
    (phase2-decisions §4). Divide by `plan.scale` in `compute_normalized_dense_reward`, not here.

    THE FOLD IS ORDERED, NOT A SUM. `acc = op(acc)` in document order: descend's row 8 is
    `SuccessBonus{mode: replace}` and row 9 is `ActionPenalty`, so the success step pays
    `12.0 - 0.001*||a||` (descend_env.py:210-212) — not 12.0, and not the sum of every row. The
    selection is `torch.where` on a per-env condition; a Python `if` there would take one branch for
    all 4096 environments, which is a different reward, not a slower one.

    WHAT A replace/floor ROW OPERATES OVER COMES FROM `compile._SCOPE_REACH`, not from this file —
    the same table `_HONOURED` builds its legal scope set from and `evaluate_plan` folds against.
    `bridle/tests/test_skill_env_fold.py` splices a second entry into that table and asserts the two
    folds still agree, which is the only thing that keeps the third reader honest.
    """
    import torch
    _check_plan(plan)
    state = slots

    def reward_fn(env, obs, action, info):
        nonlocal state
        base = getattr(env, "unwrapped", env)
        if state is None:
            state = StateSlots()
        state.ensure(base)
        ctx = MeasureContext(base, obs, action, info, state, binding, where="build_reward_fn")
        values = _values_for(ctx, plan)

        acc = torch.zeros(ctx.num_envs, dtype=torch.float32, device=ctx.device)
        for op in plan.ops:
            # A Python `if` on `op.kind` is safe and a Python `if` on a VALUE is not: the kind is
            # fixed at compile time and identical for all 4096 envs, while the condition differs per
            # env and therefore goes through torch.where below.
            if op.fn_key == "custom":
                acc = acc + _custom_row(ctx, op.params["target"])
                continue
            if op.kind == "add":
                acc = acc + _TERM_VALUE[op.fn_key](op.params, values)
                continue
            condition, level = _TERM_CONDITION_LEVEL[op.fn_key](op.params, values)
            condition = condition.to(torch.bool)
            level = _fold_level(acc, level, op)
            # `reached` is what the row's mode operates OVER. `_check_plan` has already refused a
            # scope this table has no entry for, so the lookup cannot KeyError here.
            reached = _SCOPE_REACH[op.scope](acc)
            acc = (torch.where(condition, level, reached) if op.kind == "replace"
                   else torch.where(condition, torch.maximum(reached, level), acc))

        _advance_state(ctx, plan)
        return acc

    return reward_fn


def _fold_level(acc, level, op):
    """The replace/floor level, broadcast to the batch — or a refusal that names why it could not be.

    `_CONDITION_LEVEL` returns a compile-time scalar today (`SuccessBonus.value`,
    `PredicateBonus.weight`), and `float(level)` on a per-environment tensor raises `ValueError: only
    one element tensors can be converted to Python scalars` — a message that names torch's conversion
    rule and not the reward row that broke, which is the wrong end of the stack for the author this
    file writes its messages for.

    `torch.is_tensor(level)` reads the level RAW, and so does the caller's `condition`. It did not
    always: while `_values_for` wrapped its readings, `condition` went through an unwrapping `_raw()`
    that `level` never did, so a wrapped level would have slipped past this check. Deleting the
    wrapper (see `_values_for`) removed the asymmetry rather than exposing it — there is now one
    representation for both.
    """
    import torch
    if torch.is_tensor(level) and level.numel() != 1:
        raise SkillEnvError(
            f"reward row {op.fn_key!r} produced a PER-ENVIRONMENT level of shape "
            f"{tuple(level.shape)} for a `mode: {op.kind}` row, and this fold's level is a "
            f"compile-time scalar (SuccessBonus `value`, PredicateBonus `weight`). A per-env level "
            f"needs `torch.where(condition, level, reached)` with `level` broadcast instead of "
            f"`torch.full_like` — implement that in `_fold_level` rather than letting "
            f"`float(level)` decide")
    return torch.full_like(acc, float(level))


def build_success_fn(spec, *, binding: EnvBinding = DEFAULT_BINDING, slots=None):
    """`spec` -> `fn(env, info) -> (N,) BoolTensor`, the document's success criterion.

    `spec.success` still carries its `params.X` references — `parse_spec` checks that each one is
    DECLARED and leaves binding to compile time — so they are bound here through
    `compile._bind_text`, the same substitution `compile._lower_term_row` puts on the SuccessBonus
    op's `condition`. One substitution, so the criterion the reward folds and the criterion
    `evaluate()` publishes cannot be two different numbers.
    """
    criterion = _bind_text("success", spec.success, spec.params)
    state = slots

    def success_fn(env, info):
        nonlocal state
        base = getattr(env, "unwrapped", env)
        if state is None:
            state = StateSlots()
        state.ensure(base)
        ctx = MeasureContext(base, None, None, info, state, binding, where="build_success_fn")
        return ctx.predicate(criterion) > 0.5

    return success_fn


def build_reset_fn(plan: RewardPlan, *, binding: EnvBinding = DEFAULT_BINDING, slots=None):
    """`fn(env, env_idx)` — call it from `_initialize_episode`, AFTER the state restore.

    Every buffer is re-anchored to the post-restore state of the resetting rows ONLY. Two distinct
    failures are being avoided and both are recorded:

      * the crash — ManiSkill's setters write only the current reset mask, so a full-tensor write
        raises `value tensor of shape [4096, 6] cannot be broadcast to indexing result of shape
        [1, 6]` the first time one env resets alone, i.e. every step under `--partial-reset`
        (descend_env.py:119-126);
      * the silent one — re-seeding all rows erases the in-flight potential of every env that did
        NOT restart and pays each of them a one-step progress spike on the next step
        (`ProgressPotential.doc`).

    Re-anchoring after the restore rather than before is `reseed_on_restore`, and it is why descend's
    own `_initialize_episode` re-seeds `prev_place_dist` from the true state at line 137 instead of
    trusting the pre-restore value.

    A SLOT THE PLAN NEEDS IS ALLOCATED HERE, NOT ON FIRST READ. `StateSlots.slot()` freezes its
    anchor the first time something asks for it, and this hook used to re-seed only a slot that
    already existed — so on episode 1 the freeze point of `height_above_seat_static_goal` was the
    first `compute_dense_reward` call, one action after the reset. Identical for a bolted-down
    platform, wrong for a live stack top the arm has already begun to disturb, and the docstring of
    that measure claimed the reset. `plan.measures_needed` and `plan.state_slots` say exactly which
    slots the plan will ask for, so they are allocated from the post-restore state instead of being
    guessed at.

    TWO SLOTS KEEP FIRST-READ FREEZING, and they are both reached through the `undisturbed`
    predicate, which the plan does not enumerate: `frame.object_up0` (the reset orientation) and
    `frame.object_xy0` (the reset xy). `object_xy0`'s allocation is gated on
    `"object_xy_drift_from_reset" in plan.measures_needed`, and `undisturbed` reads that measure
    through `ctx.measure(...)` rather than through a row, so a plan whose only reader is the
    predicate leaves the gate false. Both are therefore frozen at the first step that reads them,
    one action after the reset, and both are re-seeded rows-only once they exist — which is what
    keeps a partial reset correct for them even though the freeze point is late. Closing the gap
    needs the plan to enumerate the measures its PREDICATES read; until it does, this is stated
    rather than silent, and `[E]` in `bridle/tests/test_skill_env_fold.py` pins the re-seeding for
    both slots.
    """
    state = slots

    def reset_fn(env, env_idx):
        nonlocal state
        base = getattr(env, "unwrapped", env)
        if state is None:
            state = StateSlots()
        state.ensure(base)
        ctx = MeasureContext(base, None, None, {}, state, binding, where="build_reset_fn")

        # Accumulators go back to zero for the restarting rows: a latched success or a sustained
        # streak carried across an episode boundary is a success the new episode did not earn. The
        # step-guard ticks go to -1 rather than 0, because 0 is a real `elapsed_steps` value and a
        # tick equal to it would mark the new episode's first step as already advanced.
        for name in state.names():
            if name.endswith(".tick"):
                state.fill_rows(name, env_idx, -1.0)
            elif name.startswith("pred.") or name == "success_latched":
                state.fill_rows(name, env_idx, 0.0)

        def anchor(name, reader, width=None, allocate=False):
            """Re-READ one buffer from the post-restore state, rows only.

            `allocate=True` also creates it when the plan says it will be read. Allocation fills the
            WHOLE buffer, which the `StateSlots` docstring records as the one place that is not an
            exception to the partial-reset rule: at allocation no env has history to erase. Every
            write after it, including the `seed` below, is rows-only.

            ONE `reader()` CALL, NOT TWO. The allocate path used to pass `reader` to `slot()` and
            then call `reader()` again for the `seed`, so every newly allocated slot cost two sim
            reads per reset — of the same post-restore state, since nothing steps in between.
            """
            if state.has(name):
                reading = reader()
            elif allocate:
                reading = reader()
                state.slot(name, init=lambda: reading, width=width)
            else:
                return
            state.seed(name, env_idx, reading[env_idx])

        needed = plan.measures_needed
        anchor("frame.object_xy0", lambda: ctx.object_p[..., :2], 2,
               allocate="object_xy_drift_from_reset" in needed)
        anchor("frame.object_up0", lambda: _up_axis(ctx.held.pose.q), 3)
        anchor("frame.static_goal_seat_z", lambda: _seat_resting_z(ctx), None,
               # ...only when the env does not publish one. When it does, the measure reads the
               # published goal and never touches this slot, and allocating it would freeze a value
               # nothing reads (and pay a sim read per reset for it).
               allocate=("height_above_seat_static_goal" in needed
                         and getattr(base, binding.static_goal or "", None) is None))
        anchor(f"frame.scene_xy0.{binding.scene_object}",
               lambda: _scene_object(ctx).pose.p[..., :2], 2,
               allocate="scene_object_xy_drift" in needed)
        for op in plan.ops:
            if op.stateful:
                # A stateful row over an action measure raises here, from `_action`, naming
                # `build_reset_fn` — a reset happens between steps and has no action to anchor to.
                # That refusal is deliberate: leaving last episode's action in the buffer pays the
                # new episode's first step a potential difference across an episode boundary.
                anchor(op.params["slot"], lambda o=op: ctx.measure(o.params["measure"]),
                       allocate=True)

    return reset_fn


class SkillRuntime:
    """The three callables plus the ONE `StateSlots` they must share.

    Built separately, each builder allocates its own buffers, and a `latched` success in the reward
    fold would then be a different latch from the one the success criterion accumulates — two
    truths, which is the failure mode the whole `Frame` distinction exists to prevent. This is the
    normal way to wire a skill into an env; the free functions exist for callers who want one piece.
    """

    def __init__(self, plan: RewardPlan, spec=None, *, binding: EnvBinding = DEFAULT_BINDING,
                 slots=None):
        self.plan = plan
        self.spec = spec
        self.binding = binding
        #: Created UNSIZED and handed to all three builders, which size it from the first env they
        #: see (`StateSlots.ensure`). Passing `None` instead would let each builder allocate its own.
        self.slots = slots if slots is not None else StateSlots()
        self.reward = build_reward_fn(plan, binding=binding, slots=self.slots)
        self.success = (build_success_fn(spec, binding=binding, slots=self.slots)
                        if spec is not None else None)
        self.reset = build_reset_fn(plan, binding=binding, slots=self.slots)
