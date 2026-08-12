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
(b) the ordered fold's `torch.where` selection. See `_B` for the one adaptation that makes calling it
with tensors possible at all.

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
"""
import ast
import dataclasses
import operator

import torch

from bridle.skill.compile import RewardPlan
from bridle.skill.compile import _CONDITION_LEVEL as _TERM_CONDITION_LEVEL
from bridle.skill.compile import _VALUE as _TERM_VALUE
from bridle.skill.compile import _bind_text
from bridle.skill.vocab import MEASURES, PREDICATES, Sign

__all__ = [
    "MEASURE_FNS", "PREDICATE_FNS", "EnvBinding", "DEFAULT_BINDING", "MeasureContext",
    "StateSlots", "SkillEnvError", "build_reward_fn", "build_success_fn", "build_reset_fn",
    "SkillRuntime",
]


class SkillEnvError(Exception):
    """A reading this adapter cannot take against the env it was handed.

    Same shape of message as `SpecError`/`CompileError` — what was asked for, what is missing, what
    to write instead — because the reader is the same 27-30B author, one tier further down. A measure
    that cannot be read RAISES rather than returning a plausible substitute: a wrong-but-finite
    number trains a policy, logs clean, and is indistinguishable from a right one.
    """


# ── the batched value ───────────────────────────────────────────────────────────────────────────
# `compile.py`'s helpers are written branch-free (`_where(c,a,b) = c*a + (1-c)*b`) specifically so
# the same code can fold a Python float and a CUDA tensor. It ALMOST works. It does not, and the
# reason is one line of torch:
#
#     >>> 1 - torch.tensor([True, False])
#     RuntimeError: Subtraction, the `-` operator, with a bool tensor is not supported.
#
# `_max`, `_min`, `_clamp` and `_relu` all build their condition from a comparison (`a > b`), and a
# torch comparison yields a BOOL tensor, so `compile._relu(torch.tensor([-1.0, 2.0]))` raises today
# (measured 2026-08-13). Reported as a defect in compile.py; NOT worked around by transcribing the
# term math, which is the whole thing this file is trying not to do.
#
# `_B` closes it from the outside: a thin value wrapper whose comparisons return a 0.0/1.0 FLOAT
# tensor instead of a bool one. Every helper in `compile.py` and `expr.py` then runs verbatim over
# batched tensors, and the arithmetic PPO trains against is literally the arithmetic the CPU unit
# tests assert on.

_ARITHMETIC = {"add": operator.add, "sub": operator.sub, "mul": operator.mul,
               "truediv": operator.truediv, "pow": operator.pow}
_COMPARISON = {"lt": operator.lt, "le": operator.le, "gt": operator.gt, "ge": operator.ge,
               "eq": operator.eq, "ne": operator.ne}


class _B:
    """One batched `(N,)` reading, wrapped so scalar-shaped code can operate on it elementwise."""

    __slots__ = ("t",)

    def __init__(self, t):
        self.t = t

    def __repr__(self):
        return f"_B({tuple(self.t.shape)}, {self.t.dtype})"

    def __abs__(self):
        return _B(self.t.abs())

    def __neg__(self):
        return _B(-self.t)

    def __pos__(self):
        return _B(self.t)

    # `tanh`/`exp`/`log`/`sqrt` are METHOD dispatch in both `compile._dispatch` and
    # `expr._method_or_math` (`getattr(x, "tanh", None)`), which is the mechanism that keeps one
    # expression string meaning one thing for a float and for a tensor. Providing them keeps `_B` on
    # the tensor branch rather than falling through to `math.tanh`, which would raise on a batch.
    def tanh(self):
        return _B(self.t.tanh())

    def exp(self):
        return _B(self.t.exp())

    def log(self):
        return _B(self.t.log())

    def sqrt(self):
        return _B(self.t.sqrt())


def _raw(x):
    return x.t if isinstance(x, _B) else x


def _install_operators(cls):
    for name, fn in _ARITHMETIC.items():
        def forward(self, other, fn=fn):
            return _B(fn(self.t, _raw(other)))

        def reflected(self, other, fn=fn):
            return _B(fn(_raw(other), self.t))

        setattr(cls, f"__{name}__", forward)
        setattr(cls, f"__r{name}__", reflected)
    for name, fn in _COMPARISON.items():
        def compare(self, other, fn=fn):
            # ...to the operand's own float dtype, NOT to bool: a bool result is exactly what makes
            # `1 - c` raise, and a 0.0/1.0 float multiplies and subtracts like the scalar case.
            return _B(fn(self.t, _raw(other)).to(self.t.dtype))

        setattr(cls, f"__{name}__", compare)


_install_operators(_B)
# Defining `__eq__` drops the default `__hash__`; `_B` is only ever a dict VALUE, never a key.
_B.__hash__ = None


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

    def __init__(self, num_envs=None, device=None, dtype=torch.float32):
        #: Both may be None until `ensure(env)` sees a live env. That is what lets ONE `StateSlots`
        #: be shared by the reward/success/reset callables before any of them has an env: built
        #: separately they would each allocate their own, and a `latched` success in the reward fold
        #: would be a different latch from the one the success criterion accumulates.
        self.num_envs = None if num_envs is None else int(num_envs)
        self.device = device
        self.dtype = dtype
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
        buffer = self._buffers.get(name)
        if buffer is None:
            seed = init()
            seed = seed.t if isinstance(seed, _B) else seed
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
        buffer = self._buffers.get(name)
        if buffer is None:
            return
        value = value.t if isinstance(value, _B) else value
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
        t = torch.as_tensor(value, device=self.device, dtype=torch.float32)
        if t.dim() == 0:
            return t.expand(self.num_envs)
        return t.reshape(self.num_envs)

    def resolve_length(self, value, what):
        """A binding field that is either a number or the name of an env attribute."""
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


def _norm(v):
    return torch.linalg.norm(v, dim=-1)


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
    surface = ctx.resolve_length(ctx.binding.resting_surface_z, "the resting surface height")
    return ctx.object_p[..., 2] - (surface + ctx.half)


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
    frozen into a slot at reset, which for a bolted-down platform IS the frozen goal rather than a
    stand-in for one."""
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
    if ctx.action is None:
        raise SkillEnvError(
            "this row reads the action (`action_norm`/`action_delta_norm`) and none was supplied — "
            "build_success_fn evaluates without one. Move the row out of the success criterion")
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


# ── predicates ──────────────────────────────────────────────────────────────────────────────────
# A predicate field is a bare name or a nested call over existing names (`spec._check_predicate`),
# and `success:` additionally uses the bracket form the design doc's §4 example writes,
# `all[a, b, c]`. Both are parsed with `ast` against a whitelist — never `eval` — for the same reason
# `expr.py` is: the author is a 27-30B model and `().__class__.__bases__` has to be a parse-time
# refusal rather than something that depends on it never being tried.
#
# Every predicate returns a 0.0/1.0 FLOAT tensor, not a bool one, so it can be multiplied into a
# gate and subtracted from 1 without the bool-tensor arithmetic error `_B` exists to avoid.

#: `ast.List`/`ast.Tuple` are admitted for ONE reason: `and_`/`or_` declare `terms: list[predicate]`,
#: so `and_(terms=[grasped, above_z(z=0.06)])` is a spelling the vocabulary's own type invites even
#: though every chassis writes the positional form. Nothing else is: no attribute access, no
#: subscript, no arithmetic — same whitelist discipline as `expr.py`, for the same reason.
_PRED_NODES = frozenset({ast.Expression, ast.Call, ast.Name, ast.Constant, ast.Load, ast.keyword,
                         ast.UnaryOp, ast.USub, ast.UAdd, ast.List, ast.Tuple})

_BRACKET_SUGAR = {"all": "and_", "any": "or_"}


def _desugar_brackets(text):
    """`all[a, b]` -> `and_(a, b)`, `any[...]` -> `or_(...)`.

    The `success:` grammar is the one thing `spec.py` explicitly does NOT parse ("that grammar
    belongs to whoever evaluates it"), and the document form in the design doc §4 and in the
    acceptance fixture is the bracket one. It is sugar and nothing more: the bracket lowers to the
    `and_`/`or_` that already exist in PREDICATES, so there is one set of semantics, not two.
    """
    out, depth_stack = [], []
    i = 0
    while i < len(text):
        matched = None
        for word, replacement in _BRACKET_SUGAR.items():
            if text.startswith(word + "[", i):
                matched = (word, replacement)
                break
        if matched:
            out.append(matched[1] + "(")
            depth_stack.append(len(out))
            i += len(matched[0]) + 1
            continue
        char = text[i]
        if char == "[":
            depth_stack.append(None)
            out.append(char)
        elif char == "]":
            opened = depth_stack.pop() if depth_stack else None
            out.append(")" if opened is not None else "]")
        else:
            out.append(char)
        i += 1
    return "".join(out)


class _Args:
    """The arguments of one predicate call, resolvable by keyword OR position.

    `and_(grasped, above_z(z=0.06))` writes its operands positionally and `above_z(z=0.06)` writes
    its own by keyword; both spellings appear in `vocab.CHASSIS`, so both resolve here.
    """

    def __init__(self, ctx, node, source):
        self.ctx = ctx
        self.source = source
        self.positional = list(node.args) if isinstance(node, ast.Call) else []
        self.keyword = {kw.arg: kw.value for kw in node.keywords} if isinstance(node, ast.Call) else {}

    def _node(self, index, name):
        if name in self.keyword:
            return self.keyword[name]
        if index is not None and index < len(self.positional):
            return self.positional[index]
        return None

    def predicate(self, index, name):
        node = self._node(index, name)
        if node is None:
            raise SkillEnvError(f"{self.source}: `{name}` is required and names a predicate")
        return _eval_predicate_node(self.ctx, node, self.source)

    def all_predicates(self):
        nodes, queue = [], list(self.positional) + [self.keyword[k] for k in sorted(self.keyword)]
        for node in queue:
            # `and_(terms=[a, b])` and `and_(a, b)` are the same conjunction; a declared
            # `list[predicate]` is flattened rather than being a second grammar.
            nodes.extend(node.elts if isinstance(node, (ast.List, ast.Tuple)) else [node])
        if not nodes:
            raise SkillEnvError(f"{self.source}: needs at least one operand predicate")
        return [_eval_predicate_node(self.ctx, n, self.source) for n in nodes]

    def number(self, index, name, default=None, required=False):
        node = self._node(index, name)
        if node is None:
            if required:
                raise SkillEnvError(f"{self.source}: `{name}` is required and is a number")
            return default
        value = _literal(node)
        if value is None:
            raise SkillEnvError(
                f"{self.source}: `{name}` has to be a number here, and {ast.unparse(node)!r} is not "
                f"one. A `params.X` reference is substituted before this point; an expression over "
                f"scene attributes (`bin.inner_radius - 0.3*object.half_size`) needs the scene "
                f"binding this phase does not build (phase2-decisions, scope limit)")
        return float(value)

    def flag(self, index, name, default):
        node = self._node(index, name)
        if node is None:
            return default
        if isinstance(node, ast.Constant) and isinstance(node.value, bool):
            return node.value
        raise SkillEnvError(f"{self.source}: `{name}` is true or false, got {ast.unparse(node)!r}")

    def identifier(self, index, name, default=None):
        node = self._node(index, name)
        if node is None:
            return default
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        raise SkillEnvError(f"{self.source}: `{name}` names a point, got {ast.unparse(node)!r}")


def _literal(node):
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        inner = _literal(node.operand)
        if inner is None:
            return None
        return -inner if isinstance(node.op, ast.USub) else inner
    return None


def _f(mask, ctx):
    """A bool tensor as the 0.0/1.0 float every predicate returns."""
    return mask.to(dtype=torch.float32) if torch.is_tensor(mask) else \
        torch.full((ctx.num_envs,), float(mask), device=ctx.device)


def _p_grasped(ctx, args):
    """PRIVILEGED sim ground truth (`agent.is_grasping`, contact + force + angle). Allowed as a
    training-time gate; never in a deployed switching rule — the zero-privilege rule, CLAUDE.md."""
    held = ctx.info.get("is_grasped") if isinstance(ctx.info, dict) else None
    if held is None:
        held = ctx.attr("agent", "the grasp predicate").is_grasping(ctx.held)
    return _f(held, ctx)


def _p_not_grasped(ctx, args):
    return 1.0 - _p_grasped(ctx, args)


def _p_above_z(ctx, args):
    return _f(ctx.measure("object_z") > args.number(0, "z", required=True), ctx)


def _p_below_height(ctx, args):
    return _f(ctx.measure("object_z") < args.number(0, "z", required=True), ctx)


def _anchor_xy(ctx, name):
    if name is None:
        return ctx.goal[..., :2]
    value = getattr(ctx.env, name, None)
    if value is None:
        raise SkillEnvError(f"anchor {name!r} is not an attribute of {type(ctx.env).__name__}; the "
                            f"scene block is not bound to env objects in this phase, so an anchor "
                            f"has to be an env attribute holding an (N,3) or (N,2) point")
    point = value.pose.p if hasattr(value, "pose") else torch.as_tensor(value, device=ctx.device)
    return point[..., :2]


def _p_within_radius(ctx, args):
    anchor = _anchor_xy(ctx, args.identifier(0, "anchor"))
    radius = args.number(1, "radius_expr", required=True)
    return _f(_norm(ctx.object_p[..., :2] - anchor) < radius, ctx)


def _p_in_cylinder(ctx, args):
    """The container-interior test. `in_cylinder` declares radius and floor but NO anchor
    (vocab.PREDICATES), so the anchor is the goal point — the only centre an unqualified container
    test can mean in this corpus."""
    radius = args.number(0, "radius", required=True)
    floor = args.number(1, "floor", default=0.0)
    inside = _norm(ctx.object_p[..., :2] - _anchor_xy(ctx, None)) < radius
    return _f(inside & (ctx.object_p[..., 2] > floor), ctx)


def _p_at_rest(ctx, args):
    """Either bound may be omitted. NEVER gate on angular alone for a grasped object — CLAUDE.md
    gotcha (2), ~98% contact-solver noise — but that is a document-level choice this cannot police."""
    linear = args.number(0, "linear")
    angular = args.number(1, "angular")
    if linear is None and angular is None:
        raise SkillEnvError("at_rest: give at least one of `linear` / `angular`; with neither it is "
                            "a check that can never fail")
    ok = torch.ones(ctx.num_envs, dtype=torch.bool, device=ctx.device)
    if linear is not None:
        ok = ok & (ctx.measure("object_linear_velocity") < linear)
    if angular is not None:
        ok = ok & (ctx.measure("object_angular_velocity") < angular)
    return _f(ok, ctx)


def _up_axis(q):
    """The object's own +z axis in world coordinates, from a wxyz quaternion `(N,4)`."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.stack([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], dim=-1)


def _p_undisturbed(ctx, args):
    """Moved less than `drift` and tilted less than `tilt` SINCE RESET — so the tilt is measured
    against the reset orientation's own up-axis, not against world vertical: an object that spawns
    on its side is not "tilted" until something tips it."""
    drift = args.number(0, "drift", required=True)
    tilt = args.number(1, "tilt", required=True)
    up = _up_axis(ctx.held.pose.q)
    anchor = ctx.slots.slot("frame.object_up0", init=lambda: up, width=3)
    cosine = torch.clamp((up * anchor).sum(dim=-1), -1.0, 1.0)
    ok = (ctx.measure("object_xy_drift_from_reset") < drift) & (torch.acos(cosine) < tilt)
    return _f(ok, ctx)


def _p_height_above_resting_in(ctx, args):
    """`height_above_resting` in [0, band]. descend uses a band INSTEAD of an at-rest gate because a
    held cube being positioned is never stationary and the at-rest gate never latched (eval
    2026-06-03: descend_low_once=1.0 while obj_at_rest=0.06). Reads the measure it NAMES — if the
    band is meant against the destination seat, bind `resting_surface_z` (see
    `_m_height_above_resting`) rather than expecting this predicate to switch measures silently."""
    band = args.number(0, "band", required=True)
    height = ctx.measure("height_above_resting")
    return _f((height >= 0.0) & (height <= band), ctx)


def _p_and(ctx, args):
    out = None
    for term in args.all_predicates():
        out = term if out is None else out * term
    return out


def _p_or(ctx, args):
    """De Morgan rather than `max`, so the result stays a product of floats: `1 - prod(1 - p)`."""
    out = None
    for term in args.all_predicates():
        complement = 1.0 - term
        out = complement if out is None else out * complement
    return 1.0 - out


def _p_not(ctx, args):
    return 1.0 - args.predicate(0, "term")


def _p_sustained(ctx, args):
    """`predicate` has held for `k` steps.

    `consecutive=True` (7 primitives): one failing step resets the streak. `consecutive=False`
    (grab/sphere_grab): the count ACCUMULATES and never resets on a slip. That is not cosmetic — the
    cumulative version false-passed flaky grips before 2026-06-25 (vocab.PREDICATES).

    The streak advances at most once per control step even when the same predicate is read from both
    `evaluate()` and `compute_dense_reward()` — see `StateSlots.fresh_rows`."""
    value = args.predicate(0, "predicate")
    k = args.number(1, "k", default=1.0)
    consecutive = args.flag(2, "consecutive", True)
    name = f"pred.sustained.{args.source}"
    streak = ctx.slots.slot(name, init=lambda: torch.zeros_like(value))
    advanced = (streak + 1.0) * value if consecutive else streak + value
    fresh = ctx.slots.fresh_rows(name, ctx.elapsed)
    streak.copy_(advanced if fresh is None else torch.where(fresh > 0, advanced, streak))
    return _f(streak >= k, ctx)


def _p_latched(ctx, args):
    """OR-accumulated: once true it stays true for the episode. move_to_target/move_over_bin's
    success — the bonus it feeds pays every remaining step. Idempotent, so it needs no step guard."""
    value = args.predicate(0, "predicate")
    latch = ctx.slots.slot(f"pred.latched.{args.source}", init=lambda: torch.zeros_like(value))
    latch.copy_(torch.maximum(latch, value))
    return latch.clone()


def _unimplemented_quantifier(name, over_doc):
    def fn(ctx, args):
        raise SkillEnvError(
            f"`{name}` quantifies over a COLLECTION ({over_doc}) and this phase does not build one: "
            f"the `scene:` block is parsed and fingerprinted but is not synthesised into env objects "
            f"(phase2-decisions, scope limit), so there is nothing to enumerate. Write the criterion "
            f"over the single held object, or wait for the scene-generation phase")
    return fn


PREDICATE_FNS = {
    "grasped": _p_grasped,
    "not_grasped": _p_not_grasped,
    "above_z": _p_above_z,
    "below_height": _p_below_height,
    "within_radius": _p_within_radius,
    "in_cylinder": _p_in_cylinder,
    "at_rest": _p_at_rest,
    "undisturbed": _p_undisturbed,
    "height_above_resting_in": _p_height_above_resting_in,
    "and_": _p_and,
    "or_": _p_or,
    "not_": _p_not,
    "sustained": _p_sustained,
    "latched": _p_latched,
    "forall": _unimplemented_quantifier("forall", "e.g. 'bricks_in_bin'"),
    "for_n": _unimplemented_quantifier("for_n", "e.g. 'bricks_in_bin'"),
}

assert set(PREDICATE_FNS) == set(PREDICATES), (
    f"PREDICATE_FNS and vocab.PREDICATES disagree: missing "
    f"{sorted(set(PREDICATES) - set(PREDICATE_FNS))}, extra "
    f"{sorted(set(PREDICATE_FNS) - set(PREDICATES))}")


def _eval_predicate_node(ctx, node, source):
    if isinstance(node, ast.Name):
        name = node.id
    elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        name = node.func.id
    else:
        raise SkillEnvError(f"{source}: {ast.unparse(node)!r} is not a predicate — write a bare name "
                            f"from the vocabulary, or a call over them like "
                            f"`and_(grasped, above_z(z=0.06))`")
    fn = PREDICATE_FNS.get(name)
    if fn is None:
        raise SkillEnvError(f"{source}: unknown predicate {name!r} — legal: "
                            f"{', '.join(sorted(PREDICATES))}")
    # The unparsed call text is the slot identity for a stateful predicate: two `sustained(...)`
    # rows over different operands must not share one streak counter, and the same one read twice
    # must not become two.
    return fn(ctx, _Args(ctx, node, ast.unparse(node)))


def _eval_predicate_text(ctx, text):
    source = _desugar_brackets(text)
    try:
        tree = ast.parse(source, mode="eval")
    except SyntaxError as exc:
        raise SkillEnvError(f"predicate {text!r} does not parse: {exc}") from exc
    for node in ast.walk(tree):
        if type(node) not in _PRED_NODES:
            raise SkillEnvError(
                f"predicate {text!r} contains {type(node).__name__}, which a predicate expression "
                f"may not: it is a bare name or a call over names, nothing else")
    return _eval_predicate_node(ctx, tree.body, text)


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
    state slot. Only what the plan declares is read — a measure no row names costs no sim call."""
    values = {}
    for name in plan.measures_needed:
        values[name] = _B(ctx.measure(name))
    for op in plan.ops:
        for key in ("predicate", "gate"):
            text = op.params.get(key)
            if isinstance(text, str):
                values[text] = _B(ctx.predicate(text))
        if op.fn_key == "expr":
            for name in op.params["expr"].names:
                if name in PREDICATES and name not in values:
                    values[name] = _B(ctx.predicate(name))
        if op.stateful:
            slot = op.params["slot"]
            # `.clone()`: `_advance_state` writes the NEW measure into this same buffer after the
            # fold, and a live view would turn `prev - measure` into `measure - measure` = 0 for any
            # caller that evaluated the two in the other order.
            values[slot] = _B(ctx.slots.slot(
                slot, init=lambda o=op: ctx.measure(o.params["measure"])).clone())

    needs_success = any(op.fn_key == "SuccessBonus" for op in plan.ops)
    if needs_success:
        success = _success_value(ctx, plan)
        values["success"] = _B(success)
        latch = ctx.slots.slot("success_latched", init=lambda: torch.zeros_like(success))
        latch.copy_(torch.maximum(latch, success))
        values["success_latched"] = _B(latch.clone())
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
    for op in plan.ops:
        if op.fn_key == "custom":
            continue
        if op.fn_key not in _TERM_VALUE:
            raise SkillEnvError(f"no evaluator for reward row kind {op.fn_key!r}")
        if op.kind != "add" and op.fn_key not in _TERM_CONDITION_LEVEL:
            raise SkillEnvError(f"row kind {op.kind!r} needs a condition/level split and "
                                f"{op.fn_key!r} has none")


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
    """
    _check_plan(plan)
    state = slots

    def reward_fn(env, obs, action, info):
        nonlocal state
        base = getattr(env, "unwrapped", env)
        if state is None:
            state = StateSlots()
        state.ensure(base)
        ctx = MeasureContext(base, obs, action, info, state, binding)
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
                acc = acc + _raw(_TERM_VALUE[op.fn_key](op.params, values))
                continue
            condition, level = _TERM_CONDITION_LEVEL[op.fn_key](op.params, values)
            condition = _raw(condition).to(torch.bool)
            level = torch.full_like(acc, float(level))
            acc = (torch.where(condition, level, acc) if op.kind == "replace"
                   else torch.where(condition, torch.maximum(acc, level), acc))

        _advance_state(ctx, plan)
        return acc

    return reward_fn


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
        ctx = MeasureContext(base, None, None, info, state, binding)
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
    """
    state = slots

    def reset_fn(env, env_idx):
        nonlocal state
        base = getattr(env, "unwrapped", env)
        if state is None:
            state = StateSlots()
        state.ensure(base)
        ctx = MeasureContext(base, None, None, {}, state, binding)

        # Accumulators go back to zero for the restarting rows: a latched success or a sustained
        # streak carried across an episode boundary is a success the new episode did not earn. The
        # step-guard ticks go to -1 rather than 0, because 0 is a real `elapsed_steps` value and a
        # tick equal to it would mark the new episode's first step as already advanced.
        for name in state.names():
            if name.endswith(".tick"):
                state.fill_rows(name, env_idx, -1.0)
            elif name.startswith("pred.") or name == "success_latched":
                state.fill_rows(name, env_idx, 0.0)
        # Anchors and potentials are re-READ from the post-restore state, per slot, rows only.
        for name, reader in (("frame.object_xy0", lambda: ctx.object_p[..., :2]),
                             ("frame.object_up0", lambda: _up_axis(ctx.held.pose.q)),
                             ("frame.static_goal_seat_z", lambda: _seat_resting_z(ctx))):
            if state.has(name):
                state.seed(name, env_idx, reader()[env_idx])
        if state.has(f"frame.scene_xy0.{binding.scene_object}"):
            state.seed(f"frame.scene_xy0.{binding.scene_object}", env_idx,
                       _scene_object(ctx).pose.p[..., :2][env_idx])
        for op in plan.ops:
            if op.stateful and state.has(op.params["slot"]):
                state.seed(op.params["slot"], env_idx,
                           ctx.measure(op.params["measure"])[env_idx])

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
