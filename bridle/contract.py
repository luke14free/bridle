"""The declared execution contract — the physical rules a rollout is executed under.

Everything here was implicit before 2026-08-11 and drifted between training and deploy. The measured
cost of that drift, twice:

  grab   training required a grasp to survive 16 steps, deploy exited after 6.
         pick-only 0.40 vs 0.83 on the same seeds (p=0.00012).
  place  training's descend hovers 1.5cm above resting (`_HOVER`), deploy releases there — onto a
         2.4cm cube, which the cube bounces off. pick-and-stack 0/20, 5 releases traced 2026-08-11.

Both were invisible to every benchmark, because benchmarks run INSIDE the training contract.

A Contract is FROZEN, VALIDATED, and FINGERPRINTED. The fingerprint is the point: a policy trained
under one contract is not executable under another, and `bridle.checkpoint` makes the runtime say so
instead of quietly producing a 0/20.
"""
import hashlib
import json
from dataclasses import asdict, dataclass, field

LATCH_RULES = ("target", "any", "none")
GRASP_SIGNALS = ("privileged", "proprio")
TOP_RULES = ("detected_half", "assumed_half", "platform_constant")


@dataclass(frozen=True)
class Actuation:
    """How actions are shaped for this embodiment. Parallel-jaw grippers only (v0.1)."""

    gripper_dim: int
    action_lo: float
    action_hi: float


@dataclass(frozen=True)
class GraspSignal:
    """How "am I holding something?" is answered.

    kind="privileged" reads the simulator's `is_grasping` — object-aware, but sim-only, and a
    violation of the zero-privilege rule (CLAUDE.md) if it reaches the deployed path.
    kind="proprio" reads only what a real SO-101 has: fingertip contact force and jaw aperture.

    ⚠ A PROPRIOCEPTIVE SIGNAL IS OBJECT-AGNOSTIC. Force and aperture say "something is between the
    jaws", never "the cube I was aiming at is". So `Grasp.latch_on="target"` is UNIMPLEMENTABLE
    under kind="proprio" and `validate()` rejects the combination. That is not a limitation to work
    around — it is the honest statement that a real robot cannot know which object it grabbed
    without perceiving it. Measured consolation (2026-08-11): on the deployed dino rung, switching
    latch_on any->target moved the gate 0.83 -> 0.85 at McNemar p=1.000, i.e. not measurably at all.
    """

    kind: str
    force_threshold_n: float
    #: Jaw-joint position at or below which the fingers count as CLOSED ON something, in the robot's
    #: own joint units (SO-101: fully open ~0, firmly closed on a 2.4cm cube ~-0.78; closing is
    #: negative). Both gates are needed and they catch opposite failures: force alone reads
    #: 150-240N from an OPEN jaw pressing the TABLE (coord_deploy's warning — it told the policy it
    #: was holding before it ever closed, ~0.5 success live), while jaw position alone cannot tell a
    #: cube from an empty fully-closed gripper. Fitted from recorded traces by bridle.calibrate,
    #: never guessed.
    jaw_closed_below: float


GRIPPER_RULES = ("free", "zero_always", "zero_after_latch")
TERMINATION_RULES = ("sustained_grasp", "linger_after_latch", "on_goal", "on_force", "sustained_settled")


@dataclass(frozen=True)
class Grasp:
    """WHAT counts as a grasp — the rule and the sensor. Not when the loop stops (that is Execution)."""

    latch_on: str
    signal: GraspSignal


@dataclass(frozen=True)
class Execution:
    """HOW the rollout loop runs: how long, what the gripper does, and when it stops.

    Every deployed rollout loop in this project turned out to be one of these five terminations in a
    fixed order, plus one of three gripper rules. Writing them down as DATA is what lets a single
    Runner execute all of them — before this, each of the five loops re-implemented its own subset
    and the 2026-08-11 target-latch fix reached exactly one of the five.

    terminate: an ORDERED tuple, evaluated per step in sequence — order is load-bearing. The legacy
        loop checks linger BEFORE the goal test and latches AFTER both, so on the step a grasp first
        latches the linger counter does NOT advance. Reproducing that ordering exactly is what makes
        adopting Runner a no-op instead of an off-by-one.

        "sustained_grasp"    end once the grasp has SURVIVED `hold_steps` consecutive steps.
                             grab_env requires 16 ("a fingertip/loose grip drifts open and fails");
                             deploy used 6 and shipped grips that collapsed 17.5N -> 0.5N while
                             dragging the cube 20-40cm. Worth 0.40 vs 0.83 on the same seeds.
        "linger_after_latch" end `linger_steps` after the FIRST latch, whether or not it survived.
                             Deliberately weaker than sustained_grasp and NOT a substitute: it is
                             the "grip it, don't lift it" path, which verifies the grasp separately.
        "on_goal"            end once the TCP is within `goal_tolerance` of the commanded point.
                             reach episodes END at the ~3cm handoff in training; at deploy the full
                             budget kept acting PAST arrival, grinding the jaws into the table at
                             150-240N and handing grab a state it never trained from.
        "on_force"           end once finger force exceeds `force_threshold`.
        "sustained_settled"  end once "centred at release height" has held `hold_steps` steps.

    gripper:
        "free"              the policy owns the gripper dim.
        "zero_always"       gripper dim forced to 0 every step (carry prims: never re-open mid-carry).
        "zero_after_latch"  free until the grasp latches, then held at 0.
    """

    budget: int
    gripper: str = "free"
    terminate: tuple = ()
    #: Rules evaluated at the TOP of a step, BEFORE the policy is queried — as opposed to `terminate`,
    #: which is evaluated after the step. This is not gratuitous generality: the deployed rungs
    #: genuinely disagree here. `run_prim` increments its linger counter AFTER the step; the DINO
    #: rung (`run_prim_dino_grab`, the deployed default) increments it BEFORE, at the top of the next
    #: step. With the same `linger_steps` the DINO rung therefore executes one FEWER step after the
    #: latch. Two rungs, one named behaviour, an off-by-one between them — found 2026-08-12 while
    #: converting them, and exactly the drift this library exists to surface. Both timings are
    #: modelled so each rung converts as a PROVABLE no-op; unifying them is a separate, measured
    #: decision, and the fingerprints differ meanwhile, which correctly says they are not the same
    #: contract.
    terminate_pre_step: tuple = ()
    hold_steps: int | None = None
    linger_steps: int | None = None
    goal_tolerance: float | None = None
    force_threshold: float | None = None


@dataclass(frozen=True)
class Release:
    """The place leg: where, and how, the gripper lets go of a held object.

    height_above_resting: metres above the height at which the held object would REST on the
        destination. `descend_env`'s reward is an attractor at exactly this offset (`_HOVER`), NOT
        at zero — descending to zero pressed the object into the surface and broke the grasp in
        16/16 losses (slip-fix 2026-06-04). On an 8cm platform, letting go 1.5cm up is harmless.
        On a 2.4cm cube it is the whole bug.

    centering_tolerance: metres of xy error permitted AT RELEASE. Physical limit for cube-on-cube is
        the base's half-width (~0.012 for a 2.4cm cube): beyond that the held object's centre of
        mass is outside the support polygon and it topples. Measured 2026-08-11: two releases at
        xy 1.99cm and 2.06cm with essentially PERFECT height (+0.50cm, +0.34cm) both slid off.

    success_tolerance: metres of xy error at which TRAINING scores the descend a success.
    success_height_band: metres above resting within which TRAINING scores the descend "low enough"
        (`descend_env._LOW_BAND`). The height analogue of success_tolerance: `height_above_resting`
        is where the policy AIMS, this is how far off it may be and still be called done. It must be
        >= height_above_resting or the reward's own attractor sits outside the success region.

        ⚠ KNOWN DEFECT, PRESERVED DELIBERATELY. These two are the same physical quantity and today
        they disagree: training calls a descend successful at 0.045 while deploy refuses to release
        beyond 0.035, and physics needs ~0.012. Three numbers for one quantity. They are kept as
        separate fields ONLY so that routing both call sites through this Contract is a provable
        no-op (see the parity requirement in the 2026-08-12 design). Collapsing them to one is a
        deliberate, measured, retrain-bearing change and belongs to the stacking-fix spec, not here.
        `bridle/tests/test_contract.py::tolerance_drift` records the gap so it cannot be forgotten.

    destination_top_rule: how the destination's TOP surface is computed — a RULE, not a number.
        "detected_half"     top = detected centre z + detected half-size   (correct for any object)
        "assumed_half"      top = detected centre z + `assumed_half_m`     (what deploy does today)
        "platform_constant" top = `platform_top_z_m`                        (flat fixture)
        Encoding a rule as a constant is precisely what let a platform assumption reach a cube:
        `macro_place` adds a hardcoded +0.014 regardless of what perception measured, so a 0.012
        cube is aimed 2mm high and a 0.016 cube 2mm low.

    ramp_steps: steps over which the jaws open. An instantaneous full open can spring-launch the
        object — `drop_in_place` records a 16cm launch from exactly that and ramps to avoid it.
        0 means "open in one step", which is what the stack path does today.
    """

    height_above_resting: float
    centering_tolerance: float
    success_tolerance: float
    success_height_band: float
    destination_top_rule: str
    assumed_half_m: float
    platform_top_z_m: float
    ramp_steps: int


@dataclass(frozen=True)
class Contract:
    """How a rollout is executed. Runner is the only consumer; apps never implement a loop."""

    actuation: Actuation
    execution: Execution
    grasp: Grasp | None = None
    release: Release | None = None
    #: Free-form label for the primitive this contract governs ("grab", "stack"). Part of the
    #: fingerprint: two primitives with coincidentally identical numbers are still different
    #: contracts, and a ckpt should not silently cross between them.
    name: str = ""

    def validate(self) -> None:
        e = self.execution
        if e.budget <= 0:
            raise ValueError(f"budget must be > 0, got {e.budget}")
        if e.gripper not in GRIPPER_RULES:
            raise ValueError(f"gripper must be one of {GRIPPER_RULES}, got {e.gripper!r}")
        for t in tuple(e.terminate) + tuple(e.terminate_pre_step):
            if t not in TERMINATION_RULES:
                raise ValueError(f"terminate must be drawn from {TERMINATION_RULES}, got {t!r}")
        # A termination rule with no parameter is a loop that silently never ends. Each one names
        # the field it needs, so a missing value fails at construction rather than at step 40.
        needs = {"sustained_grasp": "hold_steps", "linger_after_latch": "linger_steps",
                 "on_goal": "goal_tolerance", "on_force": "force_threshold",
                 "sustained_settled": "hold_steps"}
        for t in tuple(e.terminate) + tuple(e.terminate_pre_step):
            if getattr(e, needs[t]) is None:
                raise ValueError(f"terminate rule {t!r} requires execution.{needs[t]}, which is None")
        if "sustained_grasp" in tuple(e.terminate) + tuple(e.terminate_pre_step) and self.grasp is None:
            raise ValueError("terminate 'sustained_grasp' needs a grasp phase, but grasp is None")
        if "zero_after_latch" == e.gripper and self.grasp is None:
            raise ValueError("gripper 'zero_after_latch' needs a grasp phase, but grasp is None")
        a = self.actuation
        if a.action_lo >= a.action_hi:
            raise ValueError(f"action_lo must be < action_hi, got {a.action_lo}/{a.action_hi}")
        if a.gripper_dim < 0:
            raise ValueError(f"gripper_dim must be >= 0, got {a.gripper_dim}")
        if self.grasp is not None:
            g = self.grasp
            if g.latch_on not in LATCH_RULES:
                raise ValueError(f"latch_on must be one of {LATCH_RULES}, got {g.latch_on!r}")
            if g.signal.kind not in GRASP_SIGNALS:
                raise ValueError(f"signal.kind must be one of {GRASP_SIGNALS}, got {g.signal.kind!r}")
            if g.signal.kind == "proprio" and g.latch_on == "target":
                # See GraspSignal's docstring: force+aperture cannot identify WHICH object.
                raise ValueError(
                    "latch_on='target' is unimplementable with a proprioceptive grasp signal — "
                    "force and aperture cannot tell which object is between the jaws. Use "
                    "latch_on='any' (measured equivalent on the deployed rung, McNemar p=1.000)."
                )
            if g.signal.force_threshold_n <= 0:
                raise ValueError(f"force_threshold_n must be > 0, got {g.signal.force_threshold_n}")
        if self.release is not None:
            r = self.release
            if r.destination_top_rule not in TOP_RULES:
                raise ValueError(f"destination_top_rule must be one of {TOP_RULES}, "
                                 f"got {r.destination_top_rule!r}")
            if r.height_above_resting < 0:
                raise ValueError("height_above_resting must be >= 0 (negative = press the object "
                                 f"into the surface), got {r.height_above_resting}")
            if r.centering_tolerance <= 0:
                raise ValueError(f"centering_tolerance must be > 0, got {r.centering_tolerance}")
            if r.ramp_steps < 0:
                raise ValueError(f"ramp_steps must be >= 0, got {r.ramp_steps}")
            if r.success_height_band < r.height_above_resting:
                raise ValueError(
                    f"success_height_band ({r.success_height_band}) < height_above_resting "
                    f"({r.height_above_resting}): the reward's own hover attractor would sit "
                    "outside the region training scores as success.")
            if r.success_tolerance < r.centering_tolerance:
                # The release gate must not be LOOSER than what training scores as success, or the
                # policy is optimised for a target it is then forbidden to act on. (Today they run
                # the other way — 0.045 success vs 0.035 gate — which is the recorded defect.)
                raise ValueError(
                    f"success_tolerance ({r.success_tolerance}) < centering_tolerance "
                    f"({r.centering_tolerance}): training would score a success the release gate "
                    "then refuses to act on."
                )

    def fingerprint(self) -> str:
        """A stable 12-hex-char digest of the whole contract.

        Stable across processes and machines: sha256 over canonical JSON, never `hash()` (which is
        salted per process by PYTHONHASHSEED and would make a stamped checkpoint unverifiable in the
        next run). Every field of a Contract is by construction part of the physical contract, so
        everything is hashed. Reward WEIGHTS, learning rates and network shape deliberately live
        OUTSIDE Contract and therefore outside the fingerprint: a policy trained with a different
        reward but the same physical contract is still executable under it. The question this digest
        answers is "can this policy be RUN under this contract?", not "was it trained how I remember?".
        """
        blob = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    def describe(self) -> str:
        """One-line human summary for logs and mismatch errors."""
        return f"{self.name or 'contract'}@{self.fingerprint()}"

    # ── measured factories ────────────────────────────────────────────────────────────────────
    # These are the SINGLE definition of the deployed numbers. Do not copy a value out of here into
    # a training env or a macro — read it. That copying is the entire bug class this library exists
    # to kill, and it has now cost two primitives.

    @classmethod
    def grab(cls) -> "Contract":
        """The MEASURED deployed grab contract (2026-08-11). Do not change without a paired gate run.

        WHICH VALUE CARRIES THE RESULT, measured separately:
          hold_steps 6 -> 16   pick-only 0.40 -> 0.83  (p=0.00012)  <- this one
          latch_on any -> target, coord rung   part of the same 0.83
          latch_on any -> target, dino rung    0.83 -> 0.85, McNemar p=1.000 (null)
        The HOLD is load-bearing; the target-specific latch is correct-but-not-measurable on top of
        it (a grip forced to survive 16 steps sheds a wrong-object contact anyway). latch_on stays
        "target" because it is what grab_env.evaluate uses and the contract must match training —
        not because it earned the number.

        signal.kind is "privileged" here to reproduce today's deployed behaviour exactly. Swapping
        it to "proprio" is a deliberate, separately-measured change (design §3.1, step 5), and it
        forces latch_on to "any" — see GraspSignal.
        """
        return cls(
            name="grab",
            actuation=Actuation(gripper_dim=5, action_lo=-1.0, action_hi=1.0),
            execution=Execution(budget=28, gripper="zero_after_latch",
                                terminate=("sustained_grasp",), hold_steps=16),
            grasp=Grasp(
                latch_on="target",
                # PROVISIONAL. Thresholds are carried but UNUSED while kind="privileged"; they are
                # placeholders until bridle.calibrate fits them against recorded traces (step 5).
                # Order of magnitude from the live traces: a solid hold reads 3.4-3.6N steadily
                # (2026-08-11 descend trace, 60 steps pinned) at jaw ~-0.78, a firm RL grab 17-80N,
                # and an OPEN jaw against the table reads 150-240N — which is exactly why force
                # alone is not enough and the jaw position has to gate it.
                signal=GraspSignal(kind="privileged", force_threshold_n=1.5, jaw_closed_below=-0.60),
            ),
        )

    @classmethod
    def for_prim(cls, name, budget, *, carry=False, grasp_latch=False, latch_on="target",
                 latch_linger=None, stop_at_goal=False, goal_tolerance=None,
                 stop_on_grasp=False, force_threshold=None, linger_pre_step=False) -> "Contract":
        """The contract for one deployed primitive rollout, built from the same knobs `run_prim`
        already took as arguments.

        This exists so the GENERIC rung — reach, lift, move_to_target, descend_to_target,
        place_into_bin — can run on Runner without inventing a hand-written contract per primitive.
        Each knob maps to exactly one Execution field, which is the point: `run_prim`'s arguments
        WERE an execution contract all along, passed positionally and interpreted by a loop nobody
        else could see.

        Rule ORDER here is the legacy loop's order and is load-bearing: linger, then goal, then the
        latch, then force.
        """
        terminate, pre = [], []
        if latch_linger is not None:
            (pre if linger_pre_step else terminate).append("linger_after_latch")
        if stop_at_goal:
            terminate.append("on_goal")
        if stop_on_grasp:
            terminate.append("on_force")
        gripper = "zero_always" if carry else ("zero_after_latch" if grasp_latch else "free")
        return cls(
            name=name,
            actuation=Actuation(gripper_dim=5, action_lo=-1.0, action_hi=1.0),
            execution=Execution(budget=int(budget), gripper=gripper, terminate=tuple(terminate),
                                terminate_pre_step=tuple(pre),
                                linger_steps=latch_linger, goal_tolerance=goal_tolerance,
                                force_threshold=force_threshold),
            grasp=(Grasp(latch_on=latch_on,
                         signal=GraspSignal(kind="privileged", force_threshold_n=1.5,
                                            jaw_closed_below=-0.60))
                   if grasp_latch else None),
        )

    @classmethod
    def stack(cls) -> "Contract":
        """The MEASURED deployed cube-on-cube place contract (2026-08-11), warts included.

        Every number here is what the system does TODAY, not what it should do. Reproducing the
        current behaviour exactly is the precondition for routing both call sites through this
        object without moving the gate (design §3.1). The stacking fix then changes these numbers,
        and ONLY these numbers.

        Provenance:
          height_above_resting 0.015  descend_env.py `_HOVER`
          centering_tolerance  0.035  playground_coord.py release alignment gate
          success_tolerance    0.045  descend_env.py `_CENTER_TOL`   <- disagrees with the above
          success_height_band  0.03   descend_env.py `_LOW_BAND`
          destination_top_rule "assumed_half" + 0.014  macro_place's hardcoded goal-z offset
          ramp_steps           0      the release opens fully in one step today
          terminate            ()     no sustained-centred requirement at all today — the descend
                                      always burns its 60-step budget and the macro checks
                                      alignment only afterwards
        """
        return cls(
            name="stack",
            actuation=Actuation(gripper_dim=5, action_lo=-1.0, action_hi=1.0),
            # gripper zero_always: a carry prim must never re-open mid-carry. terminate=() : today's
            # descend burns its full budget and the macro checks alignment only afterwards.
            execution=Execution(budget=60, gripper="zero_always", terminate=()),
            release=Release(
                height_above_resting=0.015,
                centering_tolerance=0.035,
                success_tolerance=0.045,
                success_height_band=0.03,
                destination_top_rule="assumed_half",
                assumed_half_m=0.014,
                platform_top_z_m=0.03,
                ramp_steps=0,
            ),
        )
