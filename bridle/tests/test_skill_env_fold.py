"""The two folds, pinned together. CPU tensors, no GPU, no simulator.

WHY THIS EXISTS. A compiled `RewardPlan` is folded twice, by two different pieces of code:

  * `bridle.skill.compile.evaluate_plan` — stdlib, scalar, what `test_skillcompile.py` asserts on
    and what `bridle plan` prints;
  * `bridle.adapters.skill_env.build_reward_fn` — torch, batched, what PPO actually trains against.

Nothing pinned them together. `bridle/tests/` contained no file that imported the adapter at all
(measured 2026-08-13: `grep -rl adapters.skill_env bridle/tests/` printed nothing), because the
adapter imported torch at module scope; the only coverage was a GPU probe in the other repo. Two
evaluators of one reward that no test compares is how the CPU one a unit test proves correct and the
CUDA one a policy is trained against drift apart, each of them "correct".

The drift was already live and this test is what found it reproducible: `evaluate_plan` folds a
replace/floor row against `_SCOPE_REACH[op.scope](acc)`, and the adapter folded against `acc`
directly. They agreed only because that table has one entry. Spliced with a second entry (below,
"scope drift"), `evaluate_plan` returned 0.0 where the adapter returned -5.0 — a plan that compiles
clean, that the stdlib evaluator honours, and that the GPU silently folds as `preceding`.

WHAT ELSE IS COVERED HERE RATHER THAN ON THE GPU. The GPU probe folds descend, and descend has no
ProgressPotential row (`prev_place_dist` appears only in its `_initialize_episode`, never in
`compute_dense_reward`), no `mode: floor` row, no `Ramp`, no `expr:` and no `custom:`. So
`plan.state_slots` was empty there and the stateful branch of `_values_for`, its load-bearing
`.clone()`, and `_advance_state`'s write ordering were executed by no assertion at all. All of that
is cheaper and better tested here, against a fake env of CPU tensors, than against a simulator.

WHAT IS NOT COVERED HERE, deliberately: ground-truth VALUES for the ten measures that need a real
robot (`contact_force`, `joint_qpos`, ...). This file uses a fake env, so a measure here is only as
true as the fake. Those belong in `scripts/probe_skill_env.py` against ManiSkill, and are listed as
outstanding in the task report.

THE PLANS ARE BUILT AS `Op`s DIRECTLY, not compiled from a document. The unit under test is the
FOLD, and its input is a plan; going through `parse_spec`/`compile_spec` would add a dependency on
the schema for no extra coverage of the thing being pinned (`test_skillcompile.py` already covers
document -> plan). It also keeps this file green while the schema is being edited.

Run: python -m pytest bridle/tests/test_skill_env_fold.py
     PYTHONPATH=. python bridle/tests/test_skill_env_fold.py
"""
import math
import sys
from types import MappingProxyType

FAILS = []


def check(name, cond, note=""):
    tail = f"  — {note}" if note else ""
    if cond:
        print(f"  PASS  {name}{tail}")
    else:
        print(f"  FAIL  {name}{tail}")
        FAILS.append(name)


def raises(exc, fn, *a, **k):
    """Returns the exception INSTANCE (truthy) or False, so a check can also read the message."""
    try:
        fn(*a, **k)
    except exc as e:
        return e
    except Exception:
        return False
    return False


# ── the fake env ────────────────────────────────────────────────────────────────────────────────
# Enough of the SO100 `grasp_cube` surface for the measures this file folds, in CPU tensors. It is a
# stand-in for the simulator's READINGS and nothing more: every property below is a number this test
# sets, so a check here proves the FOLD, never the physics.

N = 4          # four parallel environments — enough for a partial reset to have rows on both sides
DEV = "cpu"


class _Pose:
    def __init__(self, p, q):
        self.p, self.q = p, q


class _Actor:
    def __init__(self, torch, p):
        self.pose = _Pose(p, torch.tensor([[1.0, 0.0, 0.0, 0.0]] * N))
        self.linear_velocity = torch.zeros(N, 3)
        self.angular_velocity = torch.zeros(N, 3)


class _Robot:
    def __init__(self, torch):
        self._q = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, -0.70]] * N)

    def get_qpos(self):
        return self._q


class _Agent:
    def __init__(self, torch, tcp):
        self.robot = _Robot(torch)
        self.tcp_pos = tcp

    def is_grasping(self, obj):
        import torch
        return torch.tensor([True, True, False, True])


class FakeEnv:
    """A batch of 4 envs whose readings this test dictates. `unwrapped` is itself, as ManiSkill's is
    for a raw env, so the adapter's `getattr(env, "unwrapped", env)` resolves to this object."""

    def __init__(self):
        import torch
        self.num_envs, self.device = N, torch.device(DEV)
        self.cube = _Actor(torch, torch.tensor([[0.25, 0.05, 0.100],
                                                [0.25, 0.05, 0.050],
                                                [0.20, 0.00, 0.043],
                                                [0.30, 0.10, 0.020]]))
        self.target_pos = torch.tensor([[0.25, 0.05, 0.045]] * N)
        self.cube_half_sizes = torch.full((N,), 0.014)
        self.platform_top_z = 0.03
        self.agent = _Agent(torch, torch.tensor([[0.25, 0.05, 0.12]] * N))
        self.elapsed_steps = torch.zeros(N)

    @property
    def unwrapped(self):
        return self

    def move(self, dz):
        """One "step": shift every cube by `dz` and tick the per-env step counter."""
        self.cube.pose.p[:, 2] += dz
        self.elapsed_steps = self.elapsed_steps + 1.0


class CountingEnv(FakeEnv):
    """A `FakeEnv` that counts reads of the two attributes whose READ COUNT is itself a property
    under test.

      * `platform_top_z` — the resting-frame warning's guard reads it and takes `.max()`, which
        synchronises CUDA on a real device. It must be consulted once per binding, not once per
        reward step (`skill_env._WARNED_RESTING_FRAME`).
      * `cube` — every `reader()` call inside `build_reset_fn`'s `anchor()` is one sim read, and the
        allocate path used to make two of them per newly allocated slot.

    Counting is the only way to see either: both are invisible in the fold's OUTPUT, which is why
    each was reviewed as correct while costing a per-step round trip.
    """

    def __init__(self):
        self.seat_reads = self.cube_reads = 0
        super().__init__()

    @property
    def platform_top_z(self):
        self.seat_reads += 1
        return self._seat

    @platform_top_z.setter
    def platform_top_z(self, value):
        self._seat = value

    @property
    def cube(self):
        self.cube_reads += 1
        return self._cube

    @cube.setter
    def cube(self, value):
        self._cube = value


def action_of(torch, scale=1.0):
    return torch.tensor([[0.1, -0.2, 0.3, 0.0, 0.05, -0.1]] * N) * scale


# ── plan construction ───────────────────────────────────────────────────────────────────────────

def op(kind, fn_key, scope=None, stateful=False, **params):
    from bridle.skill.compile import Op
    return Op(kind=kind, scope=scope, fn_key=fn_key, params=MappingProxyType(params),
              stateful=stateful)


def plan_of(*ops, measures=(), slots=()):
    from bridle.skill.compile import RewardPlan
    return RewardPlan(ops=tuple(ops), scale=12.0, measures_needed=frozenset(measures),
                      state_slots=tuple(slots), warnings=())


# ── the parity harness ──────────────────────────────────────────────────────────────────────────

def fold_both(plan, env, action, info, ref_slots, adapter_fn):
    """Fold one step through BOTH evaluators over the SAME readings, and return (adapter, reference).

    The reference side deliberately runs in the order `collect values -> advance state -> evaluate`.
    That order is what makes `_values_for`'s `.clone()` load-bearing: the slot buffer it hands out is
    the same tensor `_advance_state` overwrites, so without the clone a ProgressPotential row folds
    `measure - measure = 0` for any caller that evaluates after advancing. The comment on that
    `.clone()` names exactly this caller; before this test there was none.

    `evaluate_plan` is called once PER ENVIRONMENT with plain Python floats. That is not a
    concession — it is the point. The stdlib evaluator is the scalar one that `test_skillcompile.py`
    asserts on, and comparing it row-by-row against the batched torch fold is what proves the batched
    one computes the same reward for every environment rather than a broadcast that happens to look
    right in aggregate.
    """
    from bridle.adapters import skill_env as SE
    from bridle.skill.compile import evaluate_plan

    # The three builders size their buffers from the first env they see; this harness stands in for
    # one of them, so it has to do the same.
    ref_slots.ensure(env)
    ctx = SE.MeasureContext(env, None, action, info, ref_slots, SE.DEFAULT_BINDING, where="test")
    values = SE._values_for(ctx, plan)
    SE._advance_state(ctx, plan)
    # `_values_for` hands over RAW tensors. It used to hand over `skill_env._B` wrappers, and this
    # line read `v.t[i]`; `compile._numeric` (f1da01b) made the wrapper unnecessary and it was
    # deleted, which is what makes `float(v[i])` the whole unwrapping needed here.
    reference = [evaluate_plan(plan, {k: float(v[i]) for k, v in values.items()})
                 for i in range(N)]
    got = adapter_fn(env, None, action, info)
    return got, reference


def _warnings_off():
    """`height_above_resting_in` reads the table-frame measure and warns about it on purpose (see
    `_warn_default_resting_frame`); block [H] is where that warning is asserted on, so it is silenced
    where it is merely incidental rather than being weakened at the source."""
    import contextlib
    import warnings
    stack = contextlib.ExitStack()
    stack.enter_context(warnings.catch_warnings())
    warnings.simplefilter("ignore")
    return stack


def agree(got, reference, tol=2e-6):
    return (tuple(got.shape) == (N,)
            and all(abs(float(got[i]) - reference[i]) <= tol for i in range(N)))


def gap(got, reference):
    return max(abs(float(got[i]) - reference[i]) for i in range(N))


# ── the checks ──────────────────────────────────────────────────────────────────────────────────

def custom_row_fn(env, obs, action, info):
    """Target of the tier-3 `custom:` row below — the four arguments a ManiSkill
    `compute_dense_reward` receives, which is the calling convention `_custom_row` defines."""
    import torch
    return torch.full((env.num_envs,), 7.0)


CUSTOM_TARGET = "bridle.tests.test_skill_env_fold:custom_row_fn"


def descend_shaped_plan():
    """descend's shape: gated shaping rows, then `SuccessBonus{mode: replace}`, then the action
    penalty. The ordering is the property — the success step pays `12.0 - 0.001*||a||`, not 12.0
    (descend_env.py:210-212)."""
    return plan_of(
        op("add", "PredicateBonus", weight=1.0, predicate="grasped", scope=None),
        op("add", "DistancePull", weight=1.5, measure="object_to_goal_xy",
           kernel="one_minus_tanh", k=4.0, setpoint=0.0, axes=None, gate="grasped"),
        op("add", "DistancePull", weight=2.5, measure="height_above_seat_live",
           kernel="one_minus_tanh", k=6.0, setpoint=0.015, axes=None, gate="grasped"),
        op("add", "HingePenalty", weight=3.0, measure="height_above_seat_live", threshold=0.0,
           side="below", gate="grasped", enabled_if=None),
        op("add", "VelocityPenalty", body="held", linear_weight=0.3, angular_weight=0.05),
        op("replace", "SuccessBonus", scope="preceding", value=12.0, predicate_ref="per_step",
           condition="grasped"),
        op("add", "ActionPenalty", weight=0.001, norm="l2", measure="action_norm"),
        measures=("object_to_goal_xy", "height_above_seat_live", "object_linear_velocity",
                  "object_angular_velocity", "action_norm"))


def run_checks():
    try:
        import torch
    except ImportError as exc:                                      # pragma: no cover
        # NOT a silent skip. This used to `print(...)` and `return`, so `main()` went on to print
        # "0 failure(s)" and exit 0 for a run that had executed nothing — "cannot verify" rendered
        # as "verified", in the harness of a project whose standing rule is that it must not be.
        # torch is present in the project venv, so its absence means the run was pointed at the
        # wrong interpreter, which is a result worth failing over rather than skipping past.
        check("torch is importable, so this file can run at all", False,
              f"{exc} — this file needs torch for CPU tensors (no GPU and no simulator). Run it "
              f"with /home/luca/robotics/maniskill/.venv/bin/python")
        return
    from bridle.adapters import skill_env as SE
    from bridle.skill import compile as C

    # ── [A] the two folds agree, row by row, over every op kind ──────────────────────────────────
    print("\n[A] build_reward_fn == evaluate_plan, per environment")
    plan = descend_shaped_plan()
    env, action = FakeEnv(), action_of(torch)
    reward = SE.build_reward_fn(plan)
    got, ref = fold_both(plan, env, action, {}, SE.StateSlots(), reward)
    check("A1 the descend shape agrees on all 4 envs", agree(got, ref),
          f"max abs diff {gap(got, ref):.3e}, rewards {[round(float(x), 5) for x in got]}")

    # ...and the fold is ORDERED: the success row replaces, then the action penalty subtracts.
    info = {"success": torch.tensor([True, False, True, False])}
    env2 = FakeEnv()
    got2, ref2 = fold_both(plan, env2, action, info, SE.StateSlots(), SE.build_reward_fn(plan))
    paid = float(got2[0])
    expected = 12.0 - 0.001 * float(torch.linalg.norm(action[0]))
    check("A2 mode=replace agrees, and the success row is not the last word",
          agree(got2, ref2) and abs(paid - expected) < 1e-6,
          f"paid {paid:.6f}, expected {expected:.6f} (12.0 would be wrong)")

    # ── [B] mode: floor — named alongside `replace` in the brief, measured by neither probe ──────
    # The accumulator has to straddle the floor level, or `floor` and `replace` compute the same
    # number and the check cannot tell them apart. The ramp below gives 6.4 / 2.4 / 1.84 / 0.0 and
    # the floor level is 2.0, so two envs are above it, one below, and one is not grasped at all.
    print("\n[B] mode: floor")
    floor_plan = plan_of(
        op("add", "Ramp", weight=8.0, measure="object_z", floor=0.02, cap=0.12, normalize=True,
           gate=None),
        op("floor", "PredicateBonus", scope="preceding", weight=2.0, predicate="grasped"),
        measures=("object_z",))
    env3 = FakeEnv()
    got3, ref3 = fold_both(floor_plan, env3, action, {}, SE.StateSlots(),
                           SE.build_reward_fn(floor_plan))
    check("B1 floor agrees with evaluate_plan", agree(got3, ref3),
          f"{[round(float(x), 4) for x in got3]}")
    check("B2 floor is `max(acc, level)` and not `replace`: it LEAVES an accumulator already above "
          "the level",
          abs(float(got3[0]) - 6.4) < 1e-5 and abs(float(got3[3]) - 2.0) < 1e-5,
          f"6.4 stays {float(got3[0]):.4f} (replace would pay 2.0); 0.0 is raised to "
          f"{float(got3[3]):.4f}")
    check("B3 ...and it does not fire at all where the predicate is false",
          abs(float(got3[2]) - 1.84) < 1e-5,
          f"ungrasped row keeps its ramp value {float(got3[2]):.4f}, not the 2.0 floor")

    # ── [C] Ramp, both `normalize` settings ──────────────────────────────────────────────────────
    # The un-normalized case is the one that matters: `normalize: false` multiplies the weight by the
    # raw climbed distance in metres instead of by a 0..1 fraction, which is the 25x compact_grasp
    # scale bug. A fold that ignores the flag looks identical on a 1.0-span ramp.
    print("\n[C] Ramp, normalized and not")
    for normalize in (True, False):
        ramp_plan = plan_of(
            op("add", "Ramp", weight=8.0, measure="object_z", floor=0.02, cap=0.12,
               normalize=normalize, gate=None),
            measures=("object_z",))
        env4 = FakeEnv()
        got4, ref4 = fold_both(ramp_plan, env4, action, {}, SE.StateSlots(),
                               SE.build_reward_fn(ramp_plan))
        check(f"C:{'normalized' if normalize else 'raw'} agrees", agree(got4, ref4),
              f"{[round(float(x), 4) for x in got4]}")
    # C1/C2 above would BOTH pass a fold that ignored `normalize` entirely, because they compare the
    # two evaluators against each other and both read the same flag. C3 is what makes the flag
    # load-bearing: the two settings have to be different numbers.
    n_plan = plan_of(op("add", "Ramp", weight=8.0, measure="object_z", floor=0.02, cap=0.12,
                        normalize=True, gate=None), measures=("object_z",))
    r_plan = plan_of(op("add", "Ramp", weight=8.0, measure="object_z", floor=0.02, cap=0.12,
                        normalize=False, gate=None), measures=("object_z",))
    a_n = SE.build_reward_fn(n_plan)(FakeEnv(), None, action, {})
    a_r = SE.build_reward_fn(r_plan)(FakeEnv(), None, action, {})
    check("C3 normalize=false is a DIFFERENT reward — the raw climb in metres, 10x smaller here "
          "(the un-normalized case is the 25x compact_grasp scale bug)",
          abs(float(a_n[0]) - float(a_r[0]) / 0.10) < 1e-5 and float(a_n[0]) != float(a_r[0]),
          f"normalized {float(a_n[0]):.4f} vs raw {float(a_r[0]):.4f}")

    # ── [D] ProgressPotential: the stateful row nothing exercised ────────────────────────────────
    print("\n[D] ProgressPotential — the stateful branch, the clone, and the write order")
    prog = plan_of(
        op("add", "ProgressPotential", stateful=True, weight=5.0, measure="object_to_goal_xy",
           gate=None, reseed_on_restore=True, gamma=1.0, terminal_zero=False,
           slot="reward[0].prev_object_to_goal_xy"),
        measures=("object_to_goal_xy",), slots=("reward[0].prev_object_to_goal_xy",))
    env5, ref_slots, adapter_slots = FakeEnv(), SE.StateSlots(), SE.StateSlots()
    reward5 = SE.build_reward_fn(prog, slots=adapter_slots)
    g1, r1 = fold_both(prog, env5, action, {}, ref_slots, reward5)
    check("D1 step 1 pays nothing: the buffer was seeded from this same reading",
          agree(g1, r1) and all(abs(float(x)) < 1e-7 for x in g1), f"{[float(x) for x in g1]}")

    # Move the cubes off-centre, so the distance to the goal GROWS by a known amount.
    env5.cube.pose.p[:, 0] += 0.01
    env5.elapsed_steps = env5.elapsed_steps + 1.0
    g2, r2 = fold_both(prog, env5, action, {}, ref_slots, reward5)
    moved_away = float(g2[0])
    check("D2 step 2 pays `weight * (prev - measure)`, and the two folds agree",
          agree(g2, r2) and moved_away < -1e-4,
          f"reward {moved_away:.5f} (negative: the object moved AWAY), max abs diff {gap(g2, r2):.3e}")
    check("D3 the reference fold read the PRE-advance value (this is what `.clone()` protects) — a "
          "live view would have folded `measure - measure` = 0",
          abs(r2[0]) > 1e-4, f"reference {r2[0]:.5f}")

    # ...and moving back TOWARD the goal pays a positive potential of the same magnitude.
    env5.cube.pose.p[:, 0] -= 0.01
    env5.elapsed_steps = env5.elapsed_steps + 1.0
    g3, r3 = fold_both(prog, env5, action, {}, ref_slots, reward5)
    check("D4 the potential is signed: returning pays back what leaving cost",
          agree(g3, r3) and abs(float(g3[0]) + moved_away) < 1e-5,
          f"{float(g3[0]):.5f} vs {-moved_away:.5f}")

    # `action_delta_norm` is the OTHER buffer `_advance_state` owns, and it pins the write ORDER
    # rather than the clone: the measure has to be read against the PREVIOUS step's action, so the
    # `prev_action <- action` write must happen after the fold. Advance first and every step reads
    # zero — the row still trains and still logs, and contributes nothing.
    delta = plan_of(op("add", "ActionPenalty", weight=1.0, norm="l2", measure="action_delta_norm"),
                    measures=("action_delta_norm",))
    env_d, slots_d = FakeEnv(), SE.StateSlots()
    reward_d = SE.build_reward_fn(delta, slots=slots_d)
    a1, a2 = action_of(torch, 1.0), action_of(torch, 3.0)
    d1, dr1 = fold_both(delta, env_d, a1, {}, SE.StateSlots(), reward_d)
    check("D5 step 1 of `action_delta_norm` is zero: prev was seeded from this same action",
          agree(d1, dr1) and all(abs(float(x)) < 1e-7 for x in d1), f"{[float(x) for x in d1]}")
    d2, dr2 = fold_both(delta, env_d, a2, {}, SE.StateSlots(), reward_d)
    check("D6 step 2 is `-||a2 - a1||`, so the previous action survived the fold "
          "(`_advance_state` writes AFTER it)",
          abs(float(d2[0]) + float(torch.linalg.norm(a2[0] - a1[0]))) < 1e-6,
          f"{float(d2[0]):.5f} vs -{float(torch.linalg.norm(a2[0] - a1[0])):.5f}")

    # ── [E] the partial-reset write mask, on a REAL stateful slot ────────────────────────────────
    # This is the crash class the `StateSlots` docstring is written about, and until now it was
    # measured only on an anchor slot on the GPU, never on a potential buffer.
    print("\n[E] partial reset writes ONLY the resetting rows")
    slot_name = "reward[0].prev_object_to_goal_xy"
    before = adapter_slots.slot(slot_name, init=lambda: torch.zeros(N)).clone()
    env5.cube.pose.p[:, 0] += 0.05                    # every env's reading changes...
    SE.build_reset_fn(prog, slots=adapter_slots)(env5, torch.tensor([0, 2]))
    after = adapter_slots.slot(slot_name, init=lambda: torch.zeros(N))
    fresh = SE.MeasureContext(env5, None, None, {}, SE.StateSlots(),
                              SE.DEFAULT_BINDING).measure("object_to_goal_xy")
    check("E1 the reset rows were re-anchored to the post-restore reading",
          all(abs(float(after[i] - fresh[i])) < 1e-7 for i in (0, 2)),
          f"rows 0,2 -> {[round(float(after[i]), 5) for i in (0, 2)]}")
    check("E2 the rows that did NOT reset still hold their old value (a whole-tensor write fails here)",
          all(abs(float(after[i] - before[i])) < 1e-7 for i in (1, 3)),
          f"rows 1,3 unchanged at {[round(float(before[i]), 5) for i in (1, 3)]}")
    check("E3 ...and they are not merely equal by luck: the fresh reading differs from them",
          all(abs(float(after[i] - fresh[i])) > 1e-4 for i in (1, 3)),
          f"fresh {[round(float(fresh[i]), 5) for i in (1, 3)]}")

    # A slot the plan declares is now allocated BY the reset, not on first read — so the freeze point
    # of an `at_reset`/`static_goal` frame is the reset even in episode 1.
    static_plan = plan_of(op("add", "DistancePull", weight=1.0,
                             measure="height_above_seat_static_goal", kernel="one_minus_tanh",
                             k=4.0, setpoint=0.0, axes=None, gate=None),
                          measures=("height_above_seat_static_goal",))
    env6, s6 = FakeEnv(), SE.StateSlots()
    SE.build_reset_fn(static_plan, slots=s6)(env6, torch.arange(N))
    # "Frozen AT the reset" is a claim about a VALUE, so the value is what is asserted: the seat's
    # resting z as it stood at the reset, `platform_top_z + cube_half_sizes` = 0.03 + 0.014.
    at_reset = 0.03 + 0.014
    frozen = s6.slot("frame.static_goal_seat_z", init=lambda: torch.full((N,), -99.0))
    check("E4 a slot the plan declares is frozen AT the reset, not at the first reward call — and "
          "the frozen NUMBER is the post-restore seat reading, not merely a slot that exists",
          s6.has("frame.static_goal_seat_z")
          and all(abs(float(frozen[i]) - at_reset) < 1e-7 for i in range(N)),
          f"{[round(float(x), 5) for x in frozen]} vs seat+half = {at_reset}")
    # ...and it does not track the live scene afterwards, which is the whole reason the frame exists:
    # descend_stack grades against a goal frozen once while `evaluate()` gates on the live top.
    env6.platform_top_z = 0.09
    ctx6 = SE.MeasureContext(env6, None, None, {}, s6, SE.DEFAULT_BINDING, where="test")
    after = ctx6.measure("height_above_seat_static_goal")
    check("E5 ...and the measure still reads that frozen z after the seat moves 6 cm, so STATIC_GOAL "
          "is a frame and not an alias for the live one",
          all(abs(float(after[i]) - (float(env6.cube.pose.p[i, 2]) - at_reset)) < 1e-6
              for i in range(N)),
          f"{[round(float(x), 5) for x in after]} — the live frame would read "
          f"{[round(float(env6.cube.pose.p[i, 2]) - (0.09 + 0.014), 5) for i in range(N)]}")

    # `frame.object_xy0` is the OTHER slot that keeps first-read freezing, and until now only
    # `frame.object_up0` was documented as such: its allocation is gated on
    # `"object_xy_drift_from_reset" in plan.measures_needed`, and the `undisturbed` predicate reaches
    # that measure through `ctx.measure(...)`, which the plan does not enumerate. The late freeze is
    # documented; a STALE anchor surviving a partial reset would not be, so that is what is pinned.
    env_u, s_u = FakeEnv(), SE.StateSlots()
    s_u.ensure(env_u)
    ctx_u = SE.MeasureContext(env_u, None, action, {}, s_u, SE.DEFAULT_BINDING, where="test")
    ctx_u.predicate("undisturbed(drift=0.01, tilt=0.3)")      # the first READ is the freeze
    frozen_xy = s_u.slot("frame.object_xy0", init=lambda: torch.zeros(N, 2), width=2).clone()
    env_u.cube.pose.p[:, 0] += 0.05
    SE.build_reset_fn(plan_of(op("add", "PredicateBonus", weight=1.0, predicate="grasped")),
                      slots=s_u)(env_u, torch.tensor([0, 2]))
    now_xy = s_u.slot("frame.object_xy0", init=lambda: torch.zeros(N, 2), width=2)
    check("E6 `frame.object_xy0` freezes on first read (the plan cannot enumerate it — `undisturbed` "
          "reaches its measure through the predicate) and is still re-seeded ROWS-ONLY once it "
          "exists, so a partial reset leaves neither a stale anchor nor a wiped one",
          all(abs(float(now_xy[i, 0]) - float(env_u.cube.pose.p[i, 0])) < 1e-7 for i in (0, 2))
          and all(abs(float(now_xy[i, 0]) - float(frozen_xy[i, 0])) < 1e-7 for i in (1, 3)),
          f"reset rows -> {[round(float(now_xy[i, 0]), 5) for i in (0, 2)]}, "
          f"untouched rows -> {[round(float(now_xy[i, 0]), 5) for i in (1, 3)]}")

    # Each `anchor()` call is a sim read, and the allocate path used to make two of them.
    drift_plan = plan_of(op("add", "DistancePull", weight=1.0,
                            measure="object_xy_drift_from_reset", kernel="one_minus_tanh", k=4.0,
                            setpoint=0.0, axes=None, gate=None),
                         measures=("object_xy_drift_from_reset",))
    env_c = CountingEnv()
    env_c.cube_reads = 0
    SE.build_reset_fn(drift_plan, slots=SE.StateSlots())(env_c, torch.arange(N))
    check("E7 allocating an anchor at reset costs ONE sim read of the post-restore state, not two",
          env_c.cube_reads == 1, f"{env_c.cube_reads} read(s) of `cube` for one allocated anchor")

    # ── [F] tier 2 (`expr`) and tier 3 (`custom`) ────────────────────────────────────────────────
    print("\n[F] expr and custom")
    from bridle.skill import expr as E
    expr_plan = plan_of(
        op("add", "expr", expr=E.parse("2.0 * grasped * (1 - tanh(4.0 * object_to_goal_xy))"),
           bindings={}),
        measures=("object_to_goal_xy",))
    env7 = FakeEnv()
    got7, ref7 = fold_both(expr_plan, env7, action, {}, SE.StateSlots(),
                           SE.build_reward_fn(expr_plan))
    check("F1 a tier-2 `expr:` row folds identically on both sides", agree(got7, ref7),
          f"{[round(float(x), 5) for x in got7]}, max abs diff {gap(got7, ref7):.3e}")
    check("F2 ...and the bare name `grasped` inside it resolved to the PREDICATE's per-env 0/1, "
          "not to a constant",
          abs(float(got7[2])) < 1e-9 and float(got7[0]) > 0.5,
          f"ungrasped env -> {float(got7[2])}, grasped env -> {float(got7[0]):.4f}")

    custom_plan = plan_of(
        op("add", "PredicateBonus", weight=1.0, predicate="grasped"),
        op("add", "custom", target=CUSTOM_TARGET),
        measures=())
    got8 = SE.build_reward_fn(custom_plan)(FakeEnv(), None, action, {})
    check("F3 a tier-3 `custom:` row is called with (env, obs, action, info) and ADDS to the fold",
          abs(float(got8[0]) - 8.0) < 1e-6 and abs(float(got8[2]) - 7.0) < 1e-6,
          f"grasped {float(got8[0])}, ungrasped {float(got8[2])}")
    # `raises(Exception, ...)` + a grep for "custom" would be satisfied by a KeyError that merely
    # mentioned the row, which is a crash and not a refusal. The type and the REASON are asserted.
    refusal = raises(C.CompileError, C.evaluate_plan, custom_plan, {"grasped": 1.0})
    check("F4 ...and the stdlib evaluator REFUSES it with a CompileError that says why tier 3 is "
          "opaque to it, not with a KeyError that happens to name the row",
          bool(refusal) and "only the adapter can call it" in str(refusal), str(refusal)[:70])

    # A `custom` row is skipped by `_check_plan`'s mode/scope validation because its value is opaque
    # — but the fold ADDS it unconditionally, so a declared `mode: replace` would be silently folded
    # as `add`: a declared mode ignored, the same defect class as the scope one [G] is about.
    moded_custom = plan_of(op("replace", "custom", scope="preceding", target=CUSTOM_TARGET))
    err_c = raises(SE.SkillEnvError, SE._check_plan, moded_custom)
    check("F5 a `custom` row that declares a MODE is refused, not quietly folded as `add`",
          bool(err_c) and "mode: replace" in str(err_c) and CUSTOM_TARGET in str(err_c),
          str(err_c)[:96])
    check("F6 ...and a plain `add` custom row still passes the same check",
          SE._check_plan(plan_of(op("add", "custom", target=CUSTOM_TARGET))) is None)

    # ── [G] the scope the fold used to drop ──────────────────────────────────────────────────────
    # `compile._SCOPE_REACH` is ONE table read by `_HONOURED` (what compiles) and by `evaluate_plan`
    # (what runs), precisely so a scope cannot be declared legal that the fold ignores. The adapter
    # is the third reader. Splicing a second entry in is the only way to see whether it reads it,
    # because with one entry every possible scope is the identity.
    print("\n[G] a second scope: does the batched fold honour it?")
    C._SCOPE_REACH["zeroed"] = lambda acc: acc * 0.0
    try:
        scoped = plan_of(
            op("add", "ActionPenalty", weight=1.0, norm="l2", measure="action_norm"),
            op("replace", "SuccessBonus", scope="zeroed", value=12.0, predicate_ref="per_step",
               condition="grasped"),
            measures=("action_norm",))
        env9 = FakeEnv()
        got9, ref9 = fold_both(scoped, env9, action, {"success": torch.tensor([1.0, 0.0, 0.0, 1.0])},
                               SE.StateSlots(), SE.build_reward_fn(scoped))
        check("G1 the batched fold honours a scope other than `preceding`", agree(got9, ref9),
              f"adapter {[round(float(x), 4) for x in got9]} vs evaluate_plan "
              f"{[round(x, 4) for x in ref9]}")
        # Non-vacuity, stated as a number: with `scope: preceding` the not-success rows keep the
        # accumulator (`-||a||`, the row above). With `zeroed` they must not.
        preceding = -float(torch.linalg.norm(action[1]))
        check("G2 ...and honouring it CHANGED the answer, so G1 is not a tautology",
              abs(float(got9[1])) < 1e-9 and abs(preceding) > 1e-3
              and abs(float(got9[1]) - preceding) > 1e-3,
              f"the not-success row folds to zeroed(acc)=0.0, where `preceding` would keep "
              f"acc={preceding:.4f}")

        # ...and the SAME splice over a `floor` row, which is the arm G1/G2 leave unpinned. `floor`
        # has TWO arms — `torch.where(cond, max(reached, level), acc)` — and only the true one goes
        # through the scope. Under the sole registered entry `preceding` the false arm's `acc` and
        # `reached` are the same object, so swapping them is a NO-OP mutation that block [B] cannot
        # see: [B] runs with only `preceding` registered. Measured before this check was added:
        # changing `skill_env.py`'s false arm from `acc` to `reached` left all 39 checks green.
        # With a second scope spliced in they differ, and `evaluate_plan` (`_where(condition,
        # _max(reached, level), acc)`) is the authority on which one is right.
        floor_scoped = plan_of(
            op("add", "ActionPenalty", weight=1.0, norm="l2", measure="action_norm"),
            op("floor", "PredicateBonus", scope="zeroed", weight=2.0, predicate="grasped"),
            measures=("action_norm",))
        env10 = FakeEnv()
        got10, ref10 = fold_both(floor_scoped, env10, action, {}, SE.StateSlots(),
                                 SE.build_reward_fn(floor_scoped))
        check("G5 a `floor` row under a second scope agrees too — the arm G1 does not reach",
              agree(got10, ref10),
              f"adapter {[round(float(x), 4) for x in got10]} vs evaluate_plan "
              f"{[round(x, 4) for x in ref10]}")
        check("G6 ...and its FALSE arm is the accumulator, not the scope's reach: the ungrasped row "
              "keeps `acc`, where folding `reached` would zero it",
              abs(float(got10[2]) - preceding) < 1e-6 and abs(preceding) > 1e-3,
              f"ungrasped row keeps acc={float(got10[2]):.4f} (zeroed(acc)=0.0 would be wrong); "
              f"the grasped rows floor to {float(got10[0]):.4f}")
    finally:
        del C._SCOPE_REACH["zeroed"]

    unknown = plan_of(
        op("replace", "SuccessBonus", scope="all", value=12.0, predicate_ref="per_step",
           condition="grasped"))
    err = raises(SE.SkillEnvError, SE._check_plan, unknown)
    check("G3 an UNKNOWN scope is refused loudly, naming the legal set, instead of defaulting",
          bool(err) and "'all'" in str(err) and "preceding" in str(err), str(err)[:90])
    check("G4 an `add` row still carries no scope and is not caught by that check",
          SE._check_plan(plan_of(op("add", "PredicateBonus", weight=1.0,
                                    predicate="grasped"))) is None)

    # ── [H] the residue of the review: messages, sugar, frames ───────────────────────────────────
    print("\n[H] refusals and sugar")
    # Reached through its OWNER. `skill_env` re-exports `_desugar_brackets` and this file was its
    # only consumer, so the re-export was being kept alive by the test that could have named the
    # owner instead. The re-export stays (external callers in the other repo import from
    # `skill_env`), and H1b pins that it is the same object rather than a second copy.
    from bridle.adapters import skill_predicates as SP
    check("H1 `all[`/`any[` sugar is anchored at a word boundary",
          SP._desugar_brackets("all[a, b]") == "and_(a, b)"
          and SP._desugar_brackets("any[a]") == "or_(a)"
          and SP._desugar_brackets("overall[x]") == "overall[x]"
          and SP._desugar_brackets("many[q]") == "many[q]",
          "overall[x] used to rewrite to overand_(x), many[q] to mor_(q)")
    check("H1b the names `skill_env` re-exports after the file split are the SAME objects, so an "
          "`except skill_env.SkillEnvError` still catches what `skill_predicates` raises",
          SE.SkillEnvError is SP.SkillEnvError and SE.PREDICATE_FNS is SP.PREDICATE_FNS
          and SE._desugar_brackets is SP._desugar_brackets and SE._norm is SP._norm,
          "SkillEnvError, PREDICATE_FNS, _desugar_brackets, _norm")

    # Both builders evaluate with `action=None`, so the message has to name WHICH one — and the
    # only way to see that it does is to raise it from both and compare. A hardcoded
    # "build_success_fn" passes any check that merely greps for both names, because the message
    # spells out the fix for both cases further down.
    def action_refusal(where):
        ctx = SE.MeasureContext(FakeEnv(), None, None, {}, SE.StateSlots(), SE.DEFAULT_BINDING,
                                where=where)
        return str(raises(SE.SkillEnvError, SE._action, ctx))

    from_reset, from_success = action_refusal("build_reset_fn"), action_refusal("build_success_fn")
    check("H2 an action read with no action names the CALLER, not always build_success_fn",
          "and build_reset_fn evaluates" in from_reset
          and "and build_success_fn evaluates" in from_success
          and from_reset != from_success,
          from_reset[:88])

    lvl = raises(SE.SkillEnvError, SE._fold_level, torch.zeros(N), torch.arange(N).float(),
                 op("replace", "SuccessBonus", scope="preceding", value=1.0))
    check("H3 a per-environment `level` fails by NAMING the cause, not as a torch conversion error",
          bool(lvl) and "PER-ENVIRONMENT" in str(lvl), str(lvl)[:80])

    import warnings as _w
    # The guard decides once per `(env type, resting_surface_z, seat_top)` and remembers, so the
    # memo is cleared here rather than letting this check depend on nothing earlier in the file
    # having folded `height_above_resting` against a `FakeEnv` first.
    SE._WARNED_RESTING_FRAME.clear()
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        SE.build_reward_fn(plan_of(op("add", "DistancePull", weight=1.0,
                                      measure="height_above_resting", kernel="one_minus_tanh",
                                      k=4.0, setpoint=0.0, axes=None, gate=None),
                                   measures=("height_above_resting",)))(FakeEnv(), None, action, {})
    check("H4 measuring `height_above_resting` in the TABLE frame against an env with a raised seat "
          "warns instead of silently picking a frame",
          any("resting_surface_z" in str(c.message) for c in caught),
          str(caught[0].message)[:80] if caught else "no warning emitted")

    with _w.catch_warnings(record=True) as quiet:
        _w.simplefilter("always")
        bound = SE.dataclasses.replace(SE.DEFAULT_BINDING, resting_surface_z="platform_top_z")
        SE.build_reward_fn(plan_of(op("add", "DistancePull", weight=1.0,
                                      measure="height_above_resting", kernel="one_minus_tanh",
                                      k=4.0, setpoint=0.0, axes=None, gate=None),
                                   measures=("height_above_resting",)),
                           binding=bound)(FakeEnv(), None, action, {})
    check("H5 ...and it is silent once the frame is stated, so it is a signal and not noise",
          not any("resting_surface_z" in str(c.message) for c in quiet),
          f"{len(quiet)} warning(s)")

    # H4 proves the warning fires; it cannot see that the CONDITION behind it was being re-evaluated
    # every single step. `float(ctx._as_batch(seat).max())` synchronises CUDA, and the f-string was
    # built whatever the warning filter then did with it — a host-device round trip per
    # `compute_dense_reward`, for a condition that cannot change inside a rollout. The seat read is
    # the observable proxy, and it is exact: nothing else in this plan touches `platform_top_z`.
    SE._WARNED_RESTING_FRAME.clear()
    counting = CountingEnv()
    hot = SE.build_reward_fn(plan_of(op("add", "DistancePull", weight=1.0,
                                        measure="height_above_resting", kernel="one_minus_tanh",
                                        k=4.0, setpoint=0.0, axes=None, gate=None),
                                     measures=("height_above_resting",)))
    with _w.catch_warnings(record=True) as repeats:
        _w.simplefilter("always")
        for _ in range(5):
            hot(counting, None, action, {})
    check("H6 ...and the guard behind it is evaluated ONCE per binding, not once per reward step: "
          "its condition synchronises CUDA, so a per-step version is a round trip in the hot path",
          counting.seat_reads == 1 and len(repeats) == 1,
          f"{counting.seat_reads} seat read(s) and {len(repeats)} warning(s) over 5 reward steps "
          f"(5 and 5 before this fix)")

    # ── [I] the free import-time guards, which now run in this suite at all ──────────────────────
    # HONEST LABELLING FIRST. I1-I3 CANNOT FAIL INDEPENDENTLY: each restates a module-scope assert
    # that already ran when `SE` was imported at the top of this function, so a divergent key set
    # takes the import down before any of them is evaluated, and they would be reported as an
    # aborted run rather than as three red checks. They are kept because they print the three key
    # sets and their sizes, which is what a reader wants when the import DOES abort. The coverage
    # that is real here is the import itself: before this file existed, no test in `bridle/tests/`
    # imported the adapter, so none of those asserts ran in the suite that actually runs.
    print("\n[I] import-time guards (I1-I3 restate module asserts; I4/I5 are what makes them bite)")
    from bridle.skill.vocab import MEASURES, PREDICATES
    check("I1 (restated) every vocabulary measure has an implementation and vice versa",
          set(SE.MEASURE_FNS) == set(MEASURES), f"{len(MEASURES)} measures")
    check("I2 (restated) every vocabulary predicate has an implementation and vice versa",
          set(SE.PREDICATE_FNS) == set(PREDICATES), f"{len(PREDICATES)} predicates")
    check("I3 (restated) the measures this file folds as SIGNED are still declared SIGNED upstream",
          not (set(SE._SIGN_LOAD_BEARING) - SE.SIGNED_MEASURES),
          ", ".join(sorted(SE._SIGN_LOAD_BEARING)))

    # I4/I5: the guards BITE. The vocabulary is grown in a FRESH process, before the adapter is
    # imported, with a name it can never legitimately gain — and the import has to fail. This is
    # exactly the defect that made this file red on 2026-08-12: `vocab` gained
    # `below_resting_height`, `skill_predicates` had no implementation, and the key-set assert took
    # `skill_env` and every importer of it down at import. A subprocess is not decoration here: the
    # property is about IMPORT time, and both modules are already imported in this one.
    import os
    import subprocess
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    probe = ("from bridle.skill import vocab\n"
             "table = getattr(vocab, {table!r})\n"
             "table[{name!r}] = next(iter(table.values()))\n"
             "import bridle.adapters.skill_env\n"
             "print('IMPORTED — the guard did not fire')\n")
    env_sub = dict(os.environ, PYTHONPATH=root)
    for table, name, tag, label in (
            ("PREDICATES", "predicate_added_tomorrow", "I4", "PREDICATE_FNS and vocab.PREDICATES"),
            ("MEASURES", "measure_added_tomorrow", "I5", "MEASURE_FNS and vocab.MEASURES")):
        done = subprocess.run([sys.executable, "-c", probe.format(table=table, name=name)],
                              capture_output=True, text=True, cwd=root, env=env_sub)
        tail = done.stderr.strip().splitlines()[-1] if done.stderr.strip() else done.stdout.strip()
        check(f"{tag} a vocabulary that grows a {table[:-1].lower()} nothing implements takes the "
              f"adapter's IMPORT down, naming the missing key",
              done.returncode != 0 and label in tail and name in tail, tail[:110])

    # ── [J] every predicate is REACHED, not just registered ──────────────────────────────────────
    # `PREDICATE_FNS` having 16 entries says nothing about whether 16 of them run: the key-set assert
    # below compares names. This calls each one through the real parse -> desugar -> evaluate path
    # and insists on a (N,) float. It is also the check that makes a file move like the
    # skill_env -> skill_predicates split provable: a helper left behind surfaces here as a
    # NameError on the one predicate that used it, not as a crash in a training run six hours in.
    print("\n[J] every predicate evaluates")
    spellings = {
        "grasped": "grasped",
        "not_grasped": "not_grasped",
        "above_z": "above_z(z=0.06)",
        "below_height": "below_height(z=0.06)",
        "within_radius": "within_radius(anchor=target_pos, radius_expr=0.05)",
        "in_cylinder": "in_cylinder(radius=0.05, floor=0.01)",
        "at_rest": "at_rest(linear=0.02, angular=0.5)",
        "undisturbed": "undisturbed(drift=0.01, tilt=0.3)",
        "height_above_resting_in": "height_above_resting_in(band=0.01)",
        "below_resting_height": "below_resting_height(band=0.01)",
        "and_": "and_(grasped, above_z(z=0.06))",
        "or_": "or_(grasped, above_z(z=0.06))",
        "not_": "not_(grasped)",
        "sustained": "sustained(grasped, k=2, consecutive=True)",
        "latched": "latched(grasped)",
        "forall": "forall(grasped, over=bricks)",
        "for_n": "for_n(grasped, over=bricks, n=2)",
    }
    check("J0 a spelling is exercised for every registered predicate",
          set(spellings) == set(SE.PREDICATE_FNS), sorted(set(spellings) ^ set(SE.PREDICATE_FNS)))
    env_p, slots_p = FakeEnv(), SE.StateSlots()
    slots_p.ensure(env_p)
    ctx_p = SE.MeasureContext(env_p, None, action, {}, slots_p, SE.DEFAULT_BINDING, where="test")
    bad, refused = [], []
    with _warnings_off():
        for name, text in spellings.items():
            try:
                v = ctx_p.predicate(text)
            except SE.SkillEnvError as exc:
                refused.append(name)
                if "scene" not in str(exc):
                    bad.append(f"{name}: refused for the wrong reason ({exc})")
                continue
            if not (v.dtype.is_floating_point and tuple(v.shape) == (N,)
                    and float(v.min()) >= 0.0 and float(v.max()) <= 1.0):
                bad.append(f"{name}: {v.dtype} {tuple(v.shape)}")
    check("J1 every implemented predicate returns a (N,) float in [0,1]", not bad, "; ".join(bad))
    check("J2 the two quantifiers are the only refusals, and they refuse on the scene block",
          sorted(refused) == ["for_n", "forall"], f"refused: {sorted(refused)}")

    # J1 above only proves each predicate RETURNS something; `below_resting_height` exists precisely
    # because it disagrees with `height_above_resting_in` BELOW the resting surface, so a check that
    # never evaluates a negative height proves nothing about the 17th predicate. The heights below
    # are `z - cube_half_sizes` against the default table frame (resting_surface_z=0.0):
    #     z = 0.010  0.020  0.014  0.100   ->   h = -0.004  +0.006   0.000  +0.086
    # With band=0.01 the two predicates differ on env 0 and NOWHERE else, which is the measured
    # 2026-08-12 gap (37/4456 states on this component; 64/64 of the states below the seat).
    env_neg = FakeEnv()
    env_neg.cube.pose.p[:, 2] = torch.tensor([0.010, 0.020, 0.014, 0.100])
    slots_n = SE.StateSlots()
    slots_n.ensure(env_neg)
    ctx_n = SE.MeasureContext(env_neg, None, action, {}, slots_n, SE.DEFAULT_BINDING, where="test")
    with _warnings_off():
        h = ctx_n.measure("height_above_resting")
        in_band = ctx_n.predicate("height_above_resting_in(band=0.01)")
        below = ctx_n.predicate("below_resting_height(band=0.01)")
    check("J3 `below_resting_height` is UNBOUNDED BELOW: a cube pressed 4 mm INTO the surface is "
          "still low, where `height_above_resting_in` calls the same state a failure",
          float(h[0]) < 0.0 and float(below[0]) == 1.0 and float(in_band[0]) == 0.0,
          f"h={float(h[0]):+.4f} m -> below_resting_height={float(below[0])}, "
          f"height_above_resting_in={float(in_band[0])}")
    check("J4 ...and above the surface they agree, so J3 is the ONE measured disagreement and not "
          "two unrelated predicates",
          [float(x) for x in below[1:]] == [1.0, 1.0, 0.0]
          and [float(x) for x in in_band[1:]] == [1.0, 1.0, 0.0],
          f"below={[float(x) for x in below]}, in_band={[float(x) for x in in_band]}")


def _run_and_collect():
    """Run the checks and turn an ABORT into a recorded failure.

    A raise from inside `run_checks` means every check after it never ran, and a run that ends in a
    bare traceback has no verdict line at all. Same rule as the missing-torch branch at the top: a
    run that could not verify must not be reportable as one that did. Measured: with
    `compile._numeric`'s `c * 1` removed, the fold raises at the first check and the file used to
    end on a traceback with no count.
    """
    try:
        run_checks()
    except Exception as exc:                                        # noqa: BLE001
        import traceback
        traceback.print_exc()
        FAILS.append(f"run_checks ABORTED: {type(exc).__name__}: {exc}")


def test_bridle():
    """pytest entry point — the same checks, reported as one assertion.

    The standalone `main()` below stays the primary interface: the project venv has no pytest, and a
    test you cannot run without installing something is a test that stops being run.
    """
    FAILS.clear()
    _run_and_collect()
    assert not FAILS, f"{len(FAILS)} check(s) failed: {FAILS}"


def main():
    _run_and_collect()
    print(f"\n{len(FAILS)} failure(s)")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
