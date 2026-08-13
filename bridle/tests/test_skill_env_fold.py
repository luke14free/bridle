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

GROUND-TRUTH VALUES, AND THE FOUR THAT ARE STILL NOT HERE. Blocks [K] and [L] pin what each measure
and each predicate actually READS, per environment — 15 of the 19 measures and 15 of the 17
predicates. Until 2026-08-13 nothing did: `fold_both` reads the batch once and hands the same values
to both evaluators, so a wrong reading is wrong identically on both sides and they agree. Measured
that day: mutating `_m_height_above_seat_live` to `.abs()` left all 21 bridle test files green, and
so did `_p_not` not negating and `_p_or` returning `_p_and`. Nineteen such mutations were run against
[K]/[L] afterwards and all nineteen go red.

Four measures remain uncovered and are declared as a partition rather than omitted (K1), because a
fake env cannot fabricate their reading: `contact_force` (`robot.get_net_contact_forces`),
`joint_pos_margin_to_limit` (`robot.get_qlimits`), and `joint_qpos` / `scene_object_xy_drift` (both
refuse without an `EnvBinding` naming what they read). Those belong in `scripts/probe_skill_env.py`
against ManiSkill.

THE PLANS ARE BUILT AS `Op`s DIRECTLY, not compiled from a document. The unit under test is the
FOLD, and its input is a plan; going through `parse_spec`/`compile_spec` would add a dependency on
the schema for no extra coverage of the thing being pinned (`test_skillcompile.py` already covers
document -> plan). It also keeps this file green while the schema is being edited.

Run: PYTHONPATH=. python bridle/tests/test_skill_env_fold.py     (the project venv has no pytest)
     python -m pytest bridle/tests/test_skill_env_fold.py
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

    # E8-E12 `frame.prev_action` ACROSS AN EPISODE BOUNDARY (2026-08-13 review, I5). It is allocated
    # by `_m_action_delta_norm` and advanced by `_advance_state`, and `build_reset_fn` re-anchored
    # `*.tick`, `pred.*`, `success_latched`, four named `frame.*` slots and the `op.stateful` slots —
    # none of which is this one, because `ActionPenalty` is `stateful=False`. So the first step of
    # every new episode paid `||a_1 - a_last_of_the_previous_episode||`, and under `--partial-reset`
    # the resetting rows paid it against the DYING episode's action. Measured before the fix with
    # weight=1.0, an episode at a=1.0, a partial reset of rows [0, 2], and a new episode at a=5.0:
    # ALL FOUR rows paid -10.583 = -||5-1|| * sqrt(7). `_action`'s own refusal exists to stop exactly
    # this and covers only the stateful-row path.
    #
    # The buffer cannot be re-READ at reset (there is no action between steps), so the fix is a
    # companion 0/1 mask and these checks are about the two sides of it: the resetting rows must pay
    # ZERO, and the rows that did not reset must keep paying their REAL delta on the same step —
    # a "fix" that zeroed the whole batch would satisfy the first half alone.
    delta_plan = plan_of(op("add", "ActionPenalty", weight=1.0, norm="l2",
                            measure="action_delta_norm"),
                         measures=("action_delta_norm",))
    env_d, slots_d = FakeEnv(), SE.StateSlots()
    delta_reward = SE.build_reward_fn(delta_plan, slots=slots_d)
    delta_reset = SE.build_reset_fn(delta_plan, slots=slots_d)
    old_a, new_a = torch.full((N, 7), 1.0), torch.full((N, 7), 5.0)
    first = [float(v) for v in delta_reward(env_d, None, old_a, {})]
    delta_reward(env_d, None, old_a, {})
    delta_reset(env_d, torch.tensor([0, 2]))
    across = [float(v) for v in delta_reward(env_d, None, new_a, {})]
    carried = -float(torch.linalg.norm(new_a[0] - old_a[0]))     # -10.583, the defect's own number
    check("E8 the resetting rows pay NOTHING for the action gap across an episode boundary",
          all(abs(across[i]) < 1e-7 for i in (0, 2)),
          f"rows 0,2 -> {[round(across[i], 4) for i in (0, 2)]} (was {round(carried, 3)})")
    check("E9 ...and the rows that did NOT reset still pay their real delta on that same step, so "
          "this is a re-anchor and not a batch-wide mute",
          all(abs(across[i] - carried) < 1e-4 for i in (1, 3)),
          f"rows 1,3 -> {[round(across[i], 4) for i in (1, 3)]}")
    after_reset = [float(v) for v in delta_reward(env_d, None, torch.full((N, 7), 6.0), {})]
    expected = -float(torch.linalg.norm(torch.full((7,), 1.0)))
    check("E10 ...and the NEXT step pays a real delta for every row, so the mask is one step wide",
          all(abs(v - expected) < 1e-4 for v in after_reset),
          f"{[round(v, 4) for v in after_reset]} vs {round(expected, 4)}")
    check("E11 ...and the very first step of the very first episode was already zero (nothing to "
          "diff against), which is why only the boundary was broken", all(abs(v) < 1e-7 for v in first))
    check("E12 ...and a plan that never reads `action_delta_norm` allocates neither buffer, so the "
          "reset stays free for the 15 primitives that do not use it",
          not adapter_slots.has(SE._PREV_ACTION) and not adapter_slots.has(SE._PREV_ACTION_OK))

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

    # F7-F9 `_check_plan`'s own docstring says "Refuse, before any GPU is spent", and it did not
    # RESOLVE a tier-3 target: `"no.such.module:nope"` built a callable and raised at step 1, after
    # the env was created (measured 2026-08-13). Importing is the only way to know a tier-3 row
    # exists — the compiler is stdlib and the target is an arbitrary module — so it belongs at the
    # one moment before any GPU work where an import is cheap.
    for label, target, fragment in (
            ("an unimportable module", "no.such.module:nope", "cannot import"),
            ("a name the module does not have", "math:no_such_fn", "no callable"),
            ("a name that exists but is not callable", "math:pi", "no callable"),
            ("a target with no colon at all", "math", "`module:function`")):
        err_t = raises(SE.SkillEnvError, SE._check_plan,
                       plan_of(op("add", "custom", target=target)))
        check(f"F7 a custom row naming {label} is refused before a step is taken",
              bool(err_t) and fragment in str(err_t) and target in str(err_t), str(err_t)[:90])
    check("F8 ...and a resolvable target is still accepted, so this is a refusal and not a ban on "
          "tier 3", SE._check_plan(plan_of(op("add", "custom", target="math:cos"))) is None)
    # ...and the check and the step-1 call go through ONE resolver, so a plan `_check_plan` accepted
    # cannot fail to resolve later. Compared by qualified name rather than by identity: run
    # standalone this file is `__main__`, and `bridle.tests.test_skill_env_fold` is then a second
    # import of it with its own function objects — an `is` check would be asserting the entry point.
    check("F9 ...and `_check_plan` and `_custom_row` resolve through the SAME resolver, so a plan "
          "the check accepted cannot fail to resolve at step 1",
          SE._resolve_custom(CUSTOM_TARGET).__qualname__ == custom_row_fn.__qualname__
          and SE._custom_row.__code__.co_names.count("_resolve_custom") == 1)

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

    # ── [K] absolute VALUES for every measure a fake env can compute exactly ─────────────────────
    # WHY THIS BLOCK EXISTS, and what was wrong with the apparatus without it. Blocks [A]-[G]
    # compare the two folds against EACH OTHER: `fold_both` reads the batch ONCE through
    # `_values_for` and hands the same dict to both evaluators, so a measure that returns the wrong
    # number returns the same wrong number to both and they agree. Block [J] checks predicates for
    # dtype, shape and range and never for TRUTH. Measured 2026-08-13: mutating
    # `_m_height_above_seat_live` to return `.abs()` — turning the signed reading into a magnitude —
    # left ALL 21 bridle test files green. That is design §1 correction 1's exact defect: the crush
    # penalty is `-3.0*clamp(-sdz, min=0)`, which becomes identically zero, so the term that exists
    # because pressing the cube to dz=0 broke 16/16 grasps (2026-06-04) silently vanishes while
    # training and logging look normal. Nine lines of docstring directly above that return say so;
    # nothing executed the claim.
    print("\n[K] measure values, pinned absolutely")
    #: The seat: `platform_top_z + cube_half_sizes` = 0.03 + 0.014. Two of the four cubes below it
    #: ON PURPOSE — a pin taken entirely above the seat is one `.abs()` cannot fail.
    SEAT = 0.03 + 0.014
    env_k = FakeEnv()
    env_k.cube.pose.p[:] = torch.tensor([[0.25, 0.05, 0.100],     # dxy 0,     seat +0.056
                                         [0.28, 0.05, 0.044],     # dxy 0.03,  seat  0.000
                                         [0.25, 0.09, 0.030],     # dxy 0.04,  seat -0.014
                                         [0.31, 0.13, 0.020]])    # dxy 0.10,  seat -0.024
    slots_k = SE.StateSlots()
    slots_k.ensure(env_k)
    ctx_k = SE.MeasureContext(env_k, None, action, {}, slots_k, SE.DEFAULT_BINDING, where="test")

    #: `||[0.1, -0.2, 0.3, 0.0, 0.05, -0.1]||` — `action_of`'s vector, spelled out rather than
    #: recomputed from it, so this pin does not move when that helper does.
    A_NORM = math.sqrt(0.1**2 + 0.2**2 + 0.3**2 + 0.0**2 + 0.05**2 + 0.1**2)
    #: Every expected number is arithmetic on the FIXTURE constants above and reads nothing from the
    #: adapter. tcp sits at (0.25, 0.05, 0.12); the goal at (0.25, 0.05, 0.045).
    pinned = {
        "tcp_to_object": [0.02,
                          math.sqrt(0.03**2 + 0.000**2 + 0.076**2),
                          math.sqrt(0.00**2 + 0.040**2 + 0.090**2),
                          math.sqrt(0.06**2 + 0.080**2 + 0.100**2)],
        "object_to_goal_xy": [0.0, 0.03, 0.04, 0.10],
        "object_to_goal_z": [0.055, 0.001, 0.015, 0.025],
        # L1 composite `||dxy|| + |dz|`, NOT a 3D norm — env 3 reads 0.125, a 3D norm would read
        # sqrt(0.10^2 + 0.025^2) = 0.1031.
        "object_to_goal_xy_plus_z": [0.055, 0.031, 0.055, 0.125],
        # SIGNED, table frame (resting_surface_z=0.0): z - cube_half_sizes.
        "height_above_resting": [0.086, 0.030, 0.016, 0.006],
        # SIGNED, seat frame. THE ONE THIS BLOCK IS FOR — see K3.
        "height_above_seat_live": [0.056, 0.000, -0.014, -0.024],
        # No `_stack_goal` published by the fake, so the goal freezes on this first read at the
        # live seat and the two frames coincide HERE. That they then diverge is E4/E5's subject.
        "height_above_seat_static_goal": [0.056, 0.000, -0.014, -0.024],
        "object_z": [0.100, 0.044, 0.030, 0.020],
        # SIGNED: the jaw joint, `get_qpos()[..., -1]`. Negative is closed; a magnitude here would
        # invert descend's `HingePenalty{threshold: -0.6, side: above}` grip term.
        "gripper_qpos": [-0.70, -0.70, -0.70, -0.70],
        "object_linear_velocity": [0.0, 0.0, 0.0, 0.0],
        "object_angular_velocity": [0.0, 0.0, 0.0, 0.0],
        "action_norm": [A_NORM] * N,
        # Identity quaternions against goal_yaw=0.0.
        "yaw_diff_mod_symmetry": [0.0, 0.0, 0.0, 0.0],
        # Both seed their anchor from the FIRST read, so zero here is the definition and not a
        # verdict; K4 below is where these two actually bite.
        "action_delta_norm": [0.0, 0.0, 0.0, 0.0],
        "object_xy_drift_from_reset": [0.0, 0.0, 0.0, 0.0],
    }
    #: Not pinned here, each with the reading a fake env cannot fabricate. This is a PARTITION, not
    #: a list of exceptions: K1 asserts the two halves cover `MEASURE_FNS` exactly, so a measure
    #: added tomorrow has to be pinned or declared unpinnable, and cannot simply be absent.
    deferred = {
        "contact_force": "needs robot.get_net_contact_forces over the finger-pad links",
        "joint_pos_margin_to_limit": "needs robot.get_qlimits()",
        "joint_qpos": "needs EnvBinding(joint=...); refuses by design without one",
        "scene_object_xy_drift": "needs EnvBinding(scene_object=...); refuses by design without one",
    }
    check("K1 every measure is either pinned to a value here or declared unpinnable with a reason",
          set(pinned).isdisjoint(deferred)
          and set(pinned) | set(deferred) == set(SE.MEASURE_FNS),
          f"{len(pinned)} pinned + {len(deferred)} deferred vs {len(SE.MEASURE_FNS)} implemented; "
          f"unaccounted: {sorted(set(SE.MEASURE_FNS) ^ (set(pinned) | set(deferred)))}")

    wrong = []
    with _warnings_off():                      # `height_above_resting` warns about its frame — [H4]
        for name, want in pinned.items():
            got_m = ctx_k.measure(name)
            if not all(abs(float(got_m[i]) - want[i]) <= 1e-6 for i in range(N)):
                wrong.append(f"{name}: {[round(float(x), 6) for x in got_m]} != {want}")
    check("K2 every pinned measure reads the value the fixture makes it, per environment",
          not wrong, "; ".join(wrong) or f"{len(pinned)} measures")

    # K3 restates part of K2 on purpose. K2 would go red for `.abs()` too, but it would go red as
    # "one of fifteen numbers moved"; this one names the mutation and the 16/16 grasps behind it, so
    # the next reader knows which property they are looking at rather than which line they broke.
    seat_live = [float(x) for x in ctx_k.measure("height_above_seat_live")]
    check("K3 `height_above_seat_live` is SIGNED: a cube pressed INTO the seat reads NEGATIVE, and "
          "the magnitude of this reading is a different vector — `.abs()` here makes descend's "
          "crush penalty `-3.0*clamp(-sdz, min=0)` identically zero (16/16 grasps, 2026-06-04)",
          min(seat_live) < 0.0
          and sum(1 for v in seat_live if v < 0.0) == 2
          and [abs(v) for v in seat_live] != [round(v, 9) for v in seat_live],
          f"{[round(v, 4) for v in seat_live]} — abs() would read "
          f"{[round(abs(v), 4) for v in seat_live]}")

    # K4: the two measures whose first read is zero by construction, read a SECOND time after the
    # thing they track has moved. Without this they are pinned at a number any broken implementation
    # returning a constant zero would also produce.
    env_k.cube.pose.p[:, 0] += 0.03
    env_k.cube.pose.p[:, 1] += 0.04                       # 3-4-5: the xy drift is exactly 0.05
    ctx_k2 = SE.MeasureContext(env_k, None, action_of(torch, 2.0), {}, slots_k,
                               SE.DEFAULT_BINDING, where="test")
    drift = [float(x) for x in ctx_k2.measure("object_xy_drift_from_reset")]
    delta = [float(x) for x in ctx_k2.measure("action_delta_norm")]
    check("K4 `object_xy_drift_from_reset` measures FROM the seeded anchor (moved 3cm x, 4cm y) and "
          "`action_delta_norm` measures from the previous action (doubled, so the delta is `a`)",
          all(abs(v - 0.05) <= 1e-6 for v in drift) and all(abs(v - A_NORM) <= 1e-6 for v in delta),
          f"drift={[round(v, 5) for v in drift]} (want 0.05), "
          f"delta={[round(v, 5) for v in delta]} (want {A_NORM:.5f})")

    # ── [L] the boolean predicates, as TRUTH TABLES ──────────────────────────────────────────────
    # [J] proves each predicate returns an (N,) float in [0,1]. It cannot see a wrong answer, and
    # measured 2026-08-13 it did not: `_p_not` returning its argument unchanged, and `_p_or`
    # returning `_p_and`, both left all 21 test files green. Every table below is chosen so the
    # answer differs per environment — a predicate that is constant, inverted, or confused with its
    # sibling produces a different vector, not a differently-shaped one.
    print("\n[L] predicate truth tables")
    env_l = FakeEnv()
    env_l.cube.pose.p[:] = torch.tensor([[0.25, 0.05, 0.100],      # dxy 0.00, above 0.06
                                         [0.28, 0.05, 0.044],      # dxy 0.03
                                         [0.25, 0.09, 0.030],      # dxy 0.04
                                         [0.31, 0.13, 0.020]])     # dxy 0.10
    # Rest for envs 0 only: env 1 is moving, env 2 is spinning, env 3 is moving fast.
    env_l.cube.linear_velocity = torch.tensor([[0.0, 0.0, 0.00], [0.0, 0.0, 0.05],
                                               [0.0, 0.0, 0.00], [0.3, 0.0, 0.00]])
    env_l.cube.angular_velocity = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
                                                [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
    slots_l = SE.StateSlots()
    slots_l.ensure(env_l)
    ctx_l = SE.MeasureContext(env_l, None, action, {}, slots_l, SE.DEFAULT_BINDING, where="test")

    def truth(ctx, text):
        return [float(x) for x in ctx.predicate(text)]

    #: `_Agent.is_grasping` is fixed at [T, T, F, T], and above_z(0.06) is [1, 0, 0, 0] on the z
    #: column above — two DIFFERENT vectors, which is what makes and_/or_ distinguishable at all.
    tables = {
        "grasped": [1.0, 1.0, 0.0, 1.0],
        "not_grasped": [0.0, 0.0, 1.0, 0.0],
        "above_z(z=0.06)": [1.0, 0.0, 0.0, 0.0],
        # Strict `>`: env 1 sits exactly AT 0.044 and is not above it.
        "above_z(z=0.044)": [1.0, 0.0, 0.0, 0.0],
        "above_z(z=0.019)": [1.0, 1.0, 1.0, 1.0],
        "not_(grasped)": [0.0, 0.0, 1.0, 0.0],
        "not_(above_z(z=0.06))": [0.0, 1.0, 1.0, 1.0],
        "not_(not_(grasped))": [1.0, 1.0, 0.0, 1.0],
        "and_(grasped, above_z(z=0.06))": [1.0, 0.0, 0.0, 0.0],
        "or_(grasped, above_z(z=0.06))": [1.0, 1.0, 0.0, 1.0],
        "and_(grasped, above_z(z=0.019))": [1.0, 1.0, 0.0, 1.0],
        "or_(not_grasped, above_z(z=0.06))": [1.0, 0.0, 1.0, 0.0],
        # Three arguments, not two: a fold that stops after the first pair reads [1, 0, 0, 0] here.
        "and_(grasped, above_z(z=0.019), not_(above_z(z=0.06)))": [0.0, 1.0, 0.0, 1.0],
        "or_(above_z(z=0.06), not_grasped, at_rest(linear=0.02, angular=0.5))": [1.0, 0.0, 1.0, 0.0],
        "below_height(z=0.044)": [0.0, 0.0, 1.0, 1.0],
        # Strict `<` on the xy distance to the goal: 0.00, 0.03, 0.04, 0.10.
        "within_radius(anchor=target_pos, radius_expr=0.035)": [1.0, 1.0, 0.0, 0.0],
        "within_radius(anchor=target_pos, radius_expr=0.05)": [1.0, 1.0, 1.0, 0.0],
        # The strictness itself, probed at radius ZERO. Env 0's cube xy is the goal literal, so its
        # distance is exactly 0.0 and `<=` would admit it where `<` does not — a boundary the other
        # two radii cannot see, because every non-zero distance here runs through a sqrt and lands
        # a few ULP off the round decimal it was built from.
        "within_radius(anchor=target_pos, radius_expr=0.0)": [0.0, 0.0, 0.0, 0.0],
        # `in_cylinder` takes its centre from the goal and applies BOTH the radius and the floor:
        # radius alone is [1,1,1,0], floor=0.035 alone is [1,1,0,0].
        "in_cylinder(radius=0.05, floor=0.01)": [1.0, 1.0, 1.0, 0.0],
        "in_cylinder(radius=0.05, floor=0.035)": [1.0, 1.0, 0.0, 0.0],
        # BOTH bounds are applied: linear alone is [1,0,1,0], angular alone is [1,1,0,1].
        "at_rest(linear=0.02, angular=0.5)": [1.0, 0.0, 0.0, 0.0],
        "at_rest(linear=0.02)": [1.0, 0.0, 1.0, 0.0],
        "at_rest(angular=0.5)": [1.0, 1.0, 0.0, 1.0],
    }
    bad_truth = [f"{text} -> {truth(ctx_l, text)} != {want}"
                 for text, want in tables.items() if truth(ctx_l, text) != want]
    check("L1 every boolean predicate answers the truth table its fixture dictates",
          not bad_truth, "; ".join(bad_truth) or f"{len(tables)} spellings")

    # L2 names the two mutations the reviewer ran, because L1 going red says "a table moved" and a
    # reader deserves to know which property that is.
    check("L2 `not_` NEGATES (it is not the identity) and `or_` is not `and_`",
          truth(ctx_l, "not_(grasped)") != truth(ctx_l, "grasped")
          and truth(ctx_l, "or_(grasped, above_z(z=0.06))")
              != truth(ctx_l, "and_(grasped, above_z(z=0.06))"),
          f"not_(grasped)={truth(ctx_l, 'not_(grasped)')}, "
          f"or_={truth(ctx_l, 'or_(grasped, above_z(z=0.06))')}, "
          f"and_={truth(ctx_l, 'and_(grasped, above_z(z=0.06))')}")

    # L3-L5: `latched` and `sustained` are the two predicates whose answer is a function of the
    # HISTORY, so a single read cannot pin either. Each step below is a fresh MeasureContext over
    # the same StateSlots — that is what a control step is — and `env.move(0.0)` ticks
    # `elapsed_steps`, which is the guard `StateSlots.fresh_rows` uses to advance a streak at most
    # once per step.
    def step_l(texts):
        c = SE.MeasureContext(env_l, None, action, {}, slots_l, SE.DEFAULT_BINDING, where="test")
        return [truth(c, t) for t in texts]

    LATCH = "latched(above_z(z=0.06))"
    first = step_l([LATCH])[0]
    env_l.move(-0.05)                       # every cube drops 5cm: nothing is above 0.06 any more
    dropped = step_l([LATCH, "above_z(z=0.06)"])
    env_l.cube.pose.p[1, 2] = 0.090         # env 1 rises above the line for the first time
    risen = step_l([LATCH])[0]
    check("L3 `latched` is OR-accumulated over the episode: it holds env 0 true after the cube "
          "falls back below the line, and admits env 1 the step it first crosses",
          first == [1.0, 0.0, 0.0, 0.0]
          and dropped == [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
          and risen == [1.0, 1.0, 0.0, 0.0],
          f"{first} -> {dropped[0]} (live {dropped[1]}) -> {risen}")

    # `_p_grasped` prefers `info["is_grasped"]` when the dict carries it, which is what makes the
    # streak drivable step by step on a fake whose `is_grasping` is a constant.
    env_s, slots_s = FakeEnv(), SE.StateSlots()
    slots_s.ensure(env_s)
    K2_, K3_ = "sustained(grasped, k=2, consecutive=True)", "sustained(grasped, k=3, consecutive=True)"

    def step_s(held, texts):
        env_s.move(0.0)                                            # tick elapsed_steps
        c = SE.MeasureContext(env_s, None, action,
                              {"is_grasped": torch.tensor(held)}, slots_s,
                              SE.DEFAULT_BINDING, where="test")
        return [truth(c, t) for t in texts]

    HOLD = [True, True, False, True]
    s1 = step_s(HOLD, [K2_, K3_])
    s2 = step_s(HOLD, [K2_, K3_])
    s3 = step_s(HOLD, [K2_, K3_])
    check("L4 `sustained`'s `k` IS the threshold: k=2 is false after one step and true after two, "
          "k=3 is still false after two and true after three",
          s1 == [[0.0] * N, [0.0] * N]
          and s2 == [[1.0, 1.0, 0.0, 1.0], [0.0] * N]
          and s3 == [[1.0, 1.0, 0.0, 1.0], [1.0, 1.0, 0.0, 1.0]],
          f"step1 k2={s1[0]} k3={s1[1]}; step2 k2={s2[0]} k3={s2[1]}; step3 k2={s3[0]} k3={s3[1]}")

    env_c, slots_c = FakeEnv(), SE.StateSlots()
    slots_c.ensure(env_c)
    CONS = "sustained(grasped, k=2, consecutive=True)"
    CUML = "sustained(grasped, k=2, consecutive=False)"

    def step_c(held):
        env_c.move(0.0)
        c = SE.MeasureContext(env_c, None, action, {"is_grasped": torch.tensor(held)}, slots_c,
                              SE.DEFAULT_BINDING, where="test")
        return [truth(c, CONS), truth(c, CUML)]

    step_c([True, True, False, True])                    # env 1 has held for one step
    step_c([True, False, False, True])                   # ...and slips
    c3 = step_c([True, True, False, True])               # ...and re-grips
    check("L5 `consecutive` is not cosmetic: after a slip the consecutive streak restarts from zero "
          "while the cumulative one carries its earlier steps — the cumulative form false-passed "
          "flaky grips before 2026-06-25 (vocab.PREDICATES)",
          c3[0] == [1.0, 0.0, 0.0, 1.0] and c3[1] == [1.0, 1.0, 0.0, 1.0],
          f"consecutive={c3[0]}, cumulative={c3[1]} on env 1 (held, slipped, re-gripped)")

    # L6: `undisturbed`'s drift arm, which is also a two-read property — the anchor is seeded on the
    # first read, so a single read can only ever return "true" and pins nothing.
    env_u, slots_u = FakeEnv(), SE.StateSlots()
    slots_u.ensure(env_u)
    UND = "undisturbed(drift=0.01, tilt=0.3)"
    settled = truth(SE.MeasureContext(env_u, None, action, {}, slots_u, SE.DEFAULT_BINDING,
                                      where="test"), UND)
    env_u.cube.pose.p[:2, 0] += 0.05                    # envs 0 and 1 are shoved 5cm, 2 and 3 are not
    moved = truth(SE.MeasureContext(env_u, None, action, {}, slots_u, SE.DEFAULT_BINDING,
                                    where="test"), UND)
    check("L6 `undisturbed` measures drift SINCE the reset anchor, not against zero: the two envs "
          "shoved 5cm past a 1cm budget go false and the two that did not move stay true",
          settled == [1.0] * N and moved == [0.0, 0.0, 1.0, 1.0],
          f"{settled} -> {moved}")

    # L0 LAST, so it is a statement about what the block above actually did. Same rule as K1: this
    # is a PARTITION over `PREDICATE_FNS`, so a predicate added tomorrow lands in one of the three
    # lists or the check goes red — it cannot simply be missing, which is how `_p_not` and `_p_or`
    # came to have no truth table at all.
    pinned_here = {"grasped", "not_grasped", "above_z", "below_height", "within_radius",
                   "in_cylinder", "at_rest", "and_", "or_", "not_", "latched", "sustained",
                   "undisturbed"}
    pinned_in_j = {"height_above_resting_in", "below_resting_height"}     # J3/J4, the 37/64 gap
    refuse_here = {"forall", "for_n"}                                     # J2, unimplemented
    check("L0 every predicate has a truth table here, a value pin in [J], or is a declared refusal",
          len(pinned_here | pinned_in_j | refuse_here) == 17
          and pinned_here | pinned_in_j | refuse_here == set(SE.PREDICATE_FNS),
          f"unaccounted: "
          f"{sorted(set(SE.PREDICATE_FNS) ^ (pinned_here | pinned_in_j | refuse_here))}")

    # L1 THE COVERAGE GUARD ITSELF (2026-08-13 review, I4). The import-time assert in
    # `skill_predicates` is set EQUALITY over KEYS, so it reported 17/17 for 15 evaluable predicates
    # — it measured key presence, not behaviour, which is the same defect class as the
    # advertised-but-unimplemented predicate it exists to prevent. It now CALLS every entry with a
    # null context and partitions on what comes back. These checks are what stop that call being
    # replaced by a flag lookup, which would put the declaration back where the drift was.
    import bridle.adapters.skill_predicates as SP
    check("L1 the guard finds the stubs by CALLING them, and finds exactly the two",
          SP._STUBS == {"forall", "for_n"}, f"found {sorted(SP._STUBS)}")
    check("...and it does not simply call everything a stub: 15 of the 17 are evaluable",
          len(SP.PREDICATE_FNS) - len(SP._STUBS) == 15)
    check("...and an evaluator dies of something OTHER than SkillEnvError on a null ctx, which is "
          "the distinction the guard rests on",
          not SP._behaves_as_stub(SP.PREDICATE_FNS["grasped"])
          and SP._behaves_as_stub(SP.PREDICATE_FNS["forall"]))
    check("...and a hand-written function that merely LOOKS unimplemented is judged by its "
          "behaviour, not by its flag",
          SP._behaves_as_stub(lambda ctx, args: (_ for _ in ()).throw(SP.SkillEnvError("no")))
          and not SP._behaves_as_stub(lambda ctx, args: 1.0))

    # ── [M] DOCUMENT -> PLAN -> ADAPTER, end to end ──────────────────────────────────────────────
    # THE INTERFACE CONTRACT NOTHING OWNED (2026-08-13 review, I8). Every other block in this file
    # builds `Op`s by hand, deliberately — it is testing the FOLD, and hand-built ops are how a fold
    # test stops being a compiler test. `test_skillcompile.py` never imports the adapter. So the one
    # thing neither file could see was the seam BETWEEN them: what the compiler promises the adapter.
    #
    # `plan.measures_needed` is that promise. The adapter reads exactly the measures the plan names
    # and hands them to the evaluator as its whole environment, so a measure the compiler forgets to
    # name is not a slower fold, it is `ExprError: undefined name` at step 0. Measured: deleting the
    # `expr` branch of `compile._measures_of` left ALL 21 test files green and produced exactly that
    # at the first control step.
    #
    # The document below is the acceptance fixture plus one tier-2 row whose measure appears NOWHERE
    # else in it — which M3 checks, because an expression over a measure some other row already
    # needs cannot detect a dropped branch (tried first with `height_above_seat_live`: the fold was
    # bit-identical with the branch deleted).
    print("\n[M] document -> plan -> adapter")
    from bridle.skill.compile import compile_spec
    from bridle.skill.spec import parse_spec
    from bridle.tests.test_skillspec import descend_doc
    e2e_doc = dict(descend_doc())
    e2e_doc["reward"] = list(e2e_doc["reward"]) + [
        {"expr": "0.5 * (1 - tanh(4 * tcp_to_object))",
         "why": "a tier-2 row whose only measure is named inside the expression — the shape that "
                "`_measures_of` has a dedicated branch for."}]
    with _warnings_off():
        e2e_plan = compile_spec(parse_spec(e2e_doc), horizon=64)
    check("M1 a real document compiles to a plan whose `measures_needed` includes a measure named "
          "ONLY inside an `expr:` row", "tcp_to_object" in e2e_plan.measures_needed,
          f"needed: {sorted(e2e_plan.measures_needed)}")
    hand_written = {op.params.get("measure") for op in e2e_plan.ops}
    check("M2 ...and that measure is reachable through no other row, so M1 cannot pass by accident",
          "tcp_to_object" not in hand_written, f"named by a row: {sorted(n for n in hand_written if n)}")
    def attempt(fn):
        """The value, or whatever it raised — one broken call is one failed check, not an aborted
        run. The mutation this block exists for RAISES, so catching it here is what lets the checks
        after it still report."""
        try:
            return fn()
        except BaseException as exc:      # noqa: BLE001 — see docstring
            return exc

    env_m, action_m = FakeEnv(), action_of(torch)
    with _warnings_off():
        folded_e2e = attempt(lambda: SE.build_reward_fn(e2e_plan, slots=SE.StateSlots())(
            env_m, None, action_m, {}))
    check("M3 the adapter FOLDS that plan without raising — the promise `measures_needed` makes is "
          "kept end to end", torch.is_tensor(folded_e2e), f"{folded_e2e!r}")
    if torch.is_tensor(folded_e2e):
        with _warnings_off():
            got_m, ref_m = fold_both(e2e_plan, FakeEnv(), action_m, {}, SE.StateSlots(),
                                     SE.build_reward_fn(e2e_plan, slots=SE.StateSlots()))
        check("M4 ...and it agrees with the stdlib evaluator on all 4 envs, so the document means "
              "one thing on both sides of the seam", agree(got_m, ref_m),
              f"max abs diff {gap(got_m, ref_m):.3e}, rewards {[round(float(x), 5) for x in got_m]}")
    # The success criterion crosses the same seam and is what I2 found nothing checking. It is a
    # STRING carried from the document through `_lower_term_row` into the adapter's predicate
    # evaluator, so a name the schema tier accepts must be one this evaluator can look up.
    with _warnings_off():
        success_m = attempt(lambda: SE.build_success_fn(parse_spec(e2e_doc),
                                                        slots=SE.StateSlots())(FakeEnv(), {}))
    check("M5 ...and the document's own `success:` line evaluates through the adapter to a (N,) "
          "boolean, with no env-published success to fall back on",
          torch.is_tensor(success_m) and tuple(success_m.shape) == (N,)
          and success_m.dtype == torch.bool, f"{success_m!r}")

    # ── [N] the two per-step caches: they must not cache a REFUSAL ───────────────────────────────
    # A criterion is fixed at compile time and re-parsed every control step (~31 us for three
    # clauses), and `_custom_row` re-imported its module every step. Both are memoised now. The
    # property worth a test is not the speed — it is that a memoised failure would make a malformed
    # document refuse ONCE and then silently succeed, or refuse with a stale message.
    print("\n[N] the hot-path caches")
    criterion = "sustained(grasped, k=2, consecutive=True)"
    SP._PARSED_PREDICATES.pop(criterion, None)
    env_n, slots_n = FakeEnv(), SE.StateSlots()
    slots_n.ensure(env_n)
    ctx_n = SE.MeasureContext(env_n, None, action_of(torch), {}, slots_n, SE.DEFAULT_BINDING,
                              where="test")
    first_n = ctx_n.predicate(criterion)
    tree_n = SP._PARSED_PREDICATES.get(criterion)
    check("N1 a criterion is parsed once and the tree is kept", tree_n is not None)
    env_n2, slots_n2 = FakeEnv(), SE.StateSlots()
    slots_n2.ensure(env_n2)
    ctx_n2 = SE.MeasureContext(env_n2, None, action_of(torch), {}, slots_n2,
                               SE.DEFAULT_BINDING, where="test")
    check("N2 ...and the second evaluation reuses that exact tree and returns the same values",
          SP._PARSED_PREDICATES.get(criterion) is tree_n
          and all(abs(float(a - b)) < 1e-9 for a, b in zip(first_n, ctx_n2.predicate(criterion))))
    # `typo_name` is in this list and behaves differently on purpose: its tree PARSES and passes the
    # whitelist, so the tree is cached and the refusal comes from `_eval_predicate_node` on every
    # evaluation. That is the point of the split below — what must not be memoised is the REFUSAL,
    # and only the first two fail before a tree exists to store.
    for bad, fragment, cacheable in (("grasped + 1", "BinOp", False),
                                     ("grasped(", "does not parse", False),
                                     ("typo_name", "unknown predicate", True)):
        msgs = []
        for _ in range(2):
            err_n = raises(SE.SkillEnvError, ctx_n.predicate, bad)
            msgs.append(str(err_n) if err_n else "<nothing raised>")
        check(f"N3 {bad!r} refuses IDENTICALLY the second time — no refusal is memoised",
              msgs[0] == msgs[1] and fragment in msgs[0], msgs[0][:70])
        check(f"...and the parse cache holds it only if the TREE was valid ({cacheable})",
              (bad in SP._PARSED_PREDICATES) is cacheable)
    SE._RESOLVED_CUSTOM.pop("no.such.module:nope", None)
    for _ in range(2):
        err_n = raises(SE.SkillEnvError, SE._resolve_custom, "no.such.module:nope")
    check("N4 ...and neither does an unresolvable tier-3 target",
          bool(err_n) and "cannot import" in str(err_n)
          and "no.such.module:nope" not in SE._RESOLVED_CUSTOM)
    check("N5 ...while a resolvable one IS cached, and to the same callable",
          SE._resolve_custom(CUSTOM_TARGET) is SE._resolve_custom(CUSTOM_TARGET)
          and CUSTOM_TARGET in SE._RESOLVED_CUSTOM)

    # ── [O] per-term contributions ──────────────────────────────────────────────────────────────
    # The input `bridle/skill/diagnose.py` shipped without: `build_reward_fn` returned the scalar and
    # nothing produced a `term_stats` mapping (whole-branch review, I7). `build_contribution_fn`
    # hands back the accumulator's per-row delta.
    #
    # O2 IS THE ANTI-DRIFT PIN AND IS THE REASON THIS BLOCK EXISTS AT ALL. `scripts/reward_
    # equivalence.py` section [2] already computes a per-term breakdown, by building `len(ops)+1`
    # PREFIX folds and subtracting adjacent ones. Two mechanisms for one quantity is the shape this
    # codebase keeps getting bitten by, so the definition is pinned: the in-fold deltas must equal
    # the prefix differences, row by row, or one of the two moved.
    print("\n[O] per-term contributions — the input the third feedback tier never had")
    import dataclasses
    plan_o = descend_shaped_plan()
    env_o, action_o, info_o = FakeEnv(), action_of(torch), {}
    with _warnings_off():
        total = SE.build_reward_fn(plan_o, slots=SE.StateSlots())(env_o, None, action_o, info_o)
        got, rows = SE.build_contribution_fn(
            plan_o, slots=SE.StateSlots())(env_o, None, action_o, info_o)
        prefixes = [SE.build_reward_fn(dataclasses.replace(plan_o, ops=plan_o.ops[:k]),
                                       slots=SE.StateSlots())(env_o, None, action_o, info_o)
                    for k in range(len(plan_o.ops) + 1)]
    names = SE.plan_row_names(plan_o)

    check("O1 the default `build_reward_fn` still returns a bare tensor — the hot path is unchanged",
          torch.is_tensor(total) and tuple(total.shape) == (N,), type(total).__name__)
    check("O2 the contributions fold returns the SAME reward as the plain one, bitwise",
          torch.equal(got, total), f"{got.tolist()} vs {total.tolist()}")
    check("O3 one contribution per row, keyed by the row address, in fold order",
          tuple(rows) == names and len(names) == len(plan_o.ops), str(tuple(rows))[:80])
    worst_pref = max(float((rows[names[k]] - (prefixes[k + 1] - prefixes[k])).abs().max())
                     for k in range(len(plan_o.ops)))
    check("O4 every row equals `scripts/reward_equivalence.py` section [2]'s PREFIX-FOLD "
          "contribution — one definition of 'what this row paid', not two",
          worst_pref <= 2e-6, f"worst row differs by {worst_pref:.3e}")
    telescoped = sum(rows[n] for n in names)
    check("O5 the contributions telescope back to the reward (acc_k - acc_k-1 summed over k)",
          float((telescoped - total).abs().max()) <= 2e-6,
          f"max |sum - total| {float((telescoped - total).abs().max()):.3e}")

    # THE ORDERED FOLD'S ROW 5 IS `SuccessBonus{mode: replace}` with `condition: grasped`, and the
    # fake env grasps in envs 0, 1, 3. Its CONTRIBUTION is the jump from the accumulated shaping to
    # 12.0, i.e. `12.0 - acc_before` — NOT the literal 12.0. The plausible wrong implementation
    # (append the row's own VALUE rather than the accumulator's delta) reports 12.0 here and passes
    # O5 only because the two coincide for `add` rows; this is the check that separates them.
    replace_row = rows[names[5]]
    before = prefixes[5]
    grasping = (0, 1, 3)
    check("O6 a `mode: replace` row's contribution is the JUMP to the level, not the level",
          all(abs(float(replace_row[i]) - (12.0 - float(before[i]))) <= 2e-6 for i in grasping)
          and all(abs(float(replace_row[i]) - 12.0) > 0.1 for i in grasping),
          f"row 5 pays {[round(float(replace_row[i]), 4) for i in grasping]} on top of "
          f"{[round(float(before[i]), 4) for i in grasping]}")
    check("O7 ...and 0.0 in the env where its condition is false", abs(float(replace_row[2])) <= 2e-6,
          f"{float(replace_row[2])}")

    # THE ADDRESS IS THE PRODUCT AS MUCH AS THE TAG IS (the ablation: strip the typed content and
    # 97.6% becomes 11.5%). A row named `reward[3] HingePenalty` with a second hinge row in the
    # document does not tell the author which one to edit.
    check("O8 every address is unique and carries its index, term and subject",
          len(set(names)) == len(names)
          and names[3] == "reward[3] HingePenalty(height_above_seat_live)"
          and names[5] == "reward[5] SuccessBonus{replace}"
          and names[6] == "reward[6] ActionPenalty(action_norm)", str(names[3]))


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
