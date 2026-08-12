"""bridle.skill.vocab — the authorable vocabulary: terms, measures, predicates, chassis.

WHAT THIS IS: the prompt payload handed to a local 27–30B model authoring a skill.yaml. Pure data,
stdlib only (global constraint: `bridle` core stays dependency-free) — `vocab_document()` is a plain
string builder, nothing here imports torch/numpy.

WHERE THE NUMBERS COME FROM: `docs/superpowers/specs/2026-08-12-reward-vocabulary-audit.md` (10
independent auditors, 15 primitives, 99 reward rows, two adversarial critiques) and Amendment 1 in
`docs/superpowers/specs/2026-08-12-declarative-skill-spec-design.md`. Every chassis default below was
re-derived from the actual primitive source (`primitives/<name>/*_env.py`), not copied from the
audit's prose — the audit is the index, the env files are the ground truth.

WHY `why` IS NOT DECORATION: the audit's own critique found the vocabulary's real risk is not term
coverage (9 terms cover 99/99 rows with zero `needs_custom` flags) but bad WEIGHTS — numbers that
encode an inequality between rows (a shaping maximum that must stay below the arrival bonus; an
attractor that must not peak at the contact surface) which a bare `weight: float` field cannot express
or check. The literature's fix (L2R, 50%→90%) was making the model state its rationale before it
emits the number. Every chassis default here carries that rationale in `why`, because it is the only
surviving record of why a number is what it is — the source comments it was copied from, per commit
history, were written 1–5 times and never touched again; the JOURNAL contains no weight-sweep record.

THREE CORRECTIONS THE ADVERSARIAL CRITICS FORCED (design doc §1, all encoded below):
  1. Every `Measure` carries a `sign`. `height_above_seat_live` MUST be SIGNED: it feeds both the
     hover attractor (`1 - tanh(6*|sdz-hover|)`) and the crush penalty (`-3.0*clamp(-sdz, min=0)`).
     An unsigned reading — the natural default for an undifferentiated "measure library" — makes the
     crush penalty identically zero and silently deletes the term that exists because pressing to
     dz=0 broke 16/16 grasps (2026-06-04, descend_env.py).
  2. Every `Measure` carries a `frame`. descend_stack's reward grades the seat height against a
     FROZEN goal (`self._stack_goal`) while its `evaluate()` success gate grades the LIVE top
     (`self._live_top()`) — one quantity, two frames, so both must exist as distinct measures.
     BOTH therefore carry their frame IN THE KEY (`height_above_seat_live`,
     `height_above_seat_static_goal`) and the bare quantity is not a name anything may be written
     under — a spec naming it would mean whichever frame the reader assumed (design doc §1.2).
  3. `Ramp` (was `HeightRamp`) carries `normalize`, and `floor`/`cap` may be per-env values. lift's
     `8.0*clamp(z/0.06, 0, 1)` and compact_grasp's `10.0*clamp(z-half, 0, 0.04)` are NOT the same
     formula; applying lift's normalized default to compact_grasp trains a lift, not a seated grip
     (25x too large — audit critique 1 #1).
"""
import dataclasses
from enum import Enum

__all__ = [
    "Sign", "Frame", "Measure", "MEASURES", "Param", "Predicate", "PREDICATES",
    "Term", "TERMS", "Chassis", "CHASSIS", "vocab_document", "base_term",
]


# ── sign & frame ─────────────────────────────────────────────────────────────────────────────────

class Sign(Enum):
    """SIGNED: the measure can be negative (e.g. "below the seat" vs "above it") and a term that
    reads its direction (HingePenalty, a non-zero-setpoint DistancePull) needs that sign to exist.
    MAGNITUDE: the measure is a non-negative distance/speed/count — the common case."""
    SIGNED = "signed"
    MAGNITUDE = "magnitude"


class Frame(Enum):
    """LIVE: read fresh from the simulator every step. AT_RESET: compared against a value captured
    once at episode reset, to detect drift since then (spawn_xy). STATIC_GOAL: compared against a
    goal frozen once, mid-episode, that does NOT track the live scene (descend_stack's `_stack_goal`)
    — the frame descend_stack's reward uses while its success gate uses LIVE for the same quantity."""
    LIVE = "live"
    AT_RESET = "at_reset"
    STATIC_GOAL = "static_goal"


# ── measures ─────────────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass(frozen=True)
class Measure:
    name: str
    sign: Sign
    frame: Frame
    unit: str
    doc: str


def _m(name, sign, frame, unit, doc):
    return name, Measure(name, sign, frame, unit, doc)


MEASURES = dict([
    _m("tcp_to_object", Sign.MAGNITUDE, Frame.LIVE, "m",
       "distance, gripper TCP to the held/target object, read live. reach's entire dense signal is "
       "this measure at weight 1.0."),

    _m("object_to_goal_xy", Sign.MAGNITUDE, Frame.LIVE, "m",
       "xy-plane distance, held object to its goal (target_pos), read live. Feeds the carry "
       "chassis' primary DistancePull row (weight 1.5, identical across all four members)."),

    _m("object_to_goal_z", Sign.MAGNITUDE, Frame.LIVE, "m",
       "z-only distance to goal height, read live. carry splits xy from z with DIFFERENT "
       "weight/sharpness (1.5@k=4 xy vs 2.5@k=6 z) — one 3D-norm measure would change the task."),

    _m("object_to_goal_xy_plus_z", Sign.MAGNITUDE, Frame.LIVE, "m",
       "L1 composite ||dxy|| + |dz|, NOT a 3D Euclidean norm. move_over_bin's `_placement_distance` "
       "is exactly this, feeding its ProgressPotential (weight 10.0) — a norm substitute silently "
       "changes the gradient."),

    _m("height_above_resting", Sign.SIGNED, Frame.LIVE, "m",
       "object z minus its own natural resting height, read live. Feeds lift's Ramp "
       "(`8.0*clamp(z/0.06,0,1)`) and the `height_above_resting_in` success band."),

    _m("height_above_seat_live", Sign.SIGNED, Frame.LIVE, "m",
       "object z minus the DESTINATION seat's resting height (platform top + half, or a live stack "
       "top + half), read live. SIGNED: + = above seat, - = pressed into it. Feeds BOTH descend's "
       "hover attractor `2.5*(1-tanh(6*|sdz-hover|))` AND its crush penalty `-3.0*clamp(-sdz,min=0)` "
       "— UNSIGNED makes the crush penalty identically zero, deleting the term that exists because "
       "pressing to dz=0 broke 16/16 grasps (2026-06-04)."),

    _m("height_above_seat_static_goal", Sign.SIGNED, Frame.STATIC_GOAL, "m",
       "same quantity as height_above_seat_live but vs the FROZEN stack goal captured at episode "
       "init (self._stack_goal), not the live top. descend_stack's reward grades against this while "
       "evaluate() grades the live top — one quantity, two frames, why Measure.frame exists."),

    _m("object_z", Sign.MAGNITUDE, Frame.LIVE, "m",
       "raw world-frame z of the held object's center. lift's Ramp reads `cube_z` directly, not a "
       "resting-relative delta; floor=0.0 in code though the comment describes a resting-height "
       "subtraction, so a resting cube already collects 2.0 of Ramp's 8.0."),

    _m("gripper_qpos", Sign.SIGNED, Frame.LIVE, "rad",
       "gripper joint position: closed ~-0.73, opening drifts toward 0. Feeds the jaw-creep "
       "HingePenalty (threshold=-0.6, side=above) — DIFFERENT from GraspSignal.jaw_closed_below "
       "(~-0.78): same quantity, two live numbers."),

    _m("contact_force", Sign.MAGNITUDE, Frame.LIVE, "N",
       "net contact force over named robot links (finger pads). Feeds the finger-grinding "
       "HingePenalty (allowance 5.0N) and reach's opt-in handoff-quality term."),

    _m("object_xy_drift_from_reset", Sign.MAGNITUDE, Frame.AT_RESET, "m",
       "xy displacement from the episode-reset pose. Feeds spawn-bulldozing/container-drift "
       "HingePenalty rows. Capture per env_idx at init, masked under partial reset — a whole-tensor "
       "capture zeroes accumulated drift."),

    _m("scene_object_xy_drift", Sign.MAGNITUDE, Frame.AT_RESET, "m",
       "xy displacement of a NON-held scene/clutter object from ITS reset pose — the "
       "scene-disturbance sibling of object_xy_drift_from_reset."),

    _m("object_linear_velocity", Sign.MAGNITUDE, Frame.LIVE, "m/s",
       "held object's linear speed. Feeds VelocityPenalty's linear half (weight 0.3) and AtRest."),

    _m("object_angular_velocity", Sign.MAGNITUDE, Frame.LIVE, "rad/s",
       "held object's angular speed. Feeds VelocityPenalty's angular half (weight 0.05). CLAUDE.md "
       "gotcha (2): reads ~22 rad/s from ~98% contact-solver noise while the cube visibly rotates "
       "~0.45 rad/s — never gate success on it."),

    _m("action_norm", Sign.MAGNITUDE, Frame.LIVE, "-",
       "L2 norm of the current action. ActionPenalty's default `measure`, the ONLY one ever "
       "instantiated (weight=0.001, norm=l2, all 15 primitives) — despite nine files calling "
       "ActionPenalty a 'jerk penalty', no env stores a previous action."),

    _m("action_delta_norm", Sign.MAGNITUDE, Frame.LIVE, "-",
       "L2 norm of (action_t - action_t-1) — the jerk-LIKE variant nine files describe but none "
       "compute. Requires a new previous-action buffer. Ships as ActionPenalty's `measure` at "
       "chassis weight 0.0 — enabling it is a sweep, not a silent parity break."),

    _m("yaw_diff_mod_symmetry", Sign.MAGNITUDE, Frame.LIVE, "rad",
       "symmetry-reduced angular diff, held-object yaw vs target yaw (`_shape_aware_yaw`). NOT wired "
       "as a DistancePull target: the arm is 5-DoF, so a full-pose target is generally unreachable "
       "— a reward-hacking generator, not shaping."),

    _m("joint_pos_margin_to_limit", Sign.MAGNITUDE, Frame.LIVE, "rad",
       "distance from current joint position to its nearest hardware limit. Amendment-A addition "
       "for safety-margin HingePenalty rows; zero of the 99 audited rows use it today."),

    _m("joint_qpos", Sign.SIGNED, Frame.LIVE, "rad",
       "raw joint angle of a named joint/link. General-purpose; prefer gripper_qpos when the target "
       "IS the gripper."),
])


# ── predicates ───────────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass(frozen=True)
class Param:
    name: str
    type: str
    default: object = None
    doc: str = ""
    required: bool = False
    choices: tuple = ()  # explicit legal values for a str param, e.g. mode's ("add","replace","floor")
    # — schema-level, so a check can assert against it instead of grepping free-text doc.


@dataclasses.dataclass(frozen=True)
class Predicate:
    name: str
    params: tuple
    doc: str


def _p(name, params, doc):
    return name, Predicate(name, tuple(params), doc)


PREDICATES = dict([
    _p("grasped", [],
       "held object is currently grasped (contact+force threshold, agent.is_grasping). PRIVILEGED "
       "sim ground truth — must never appear in a deployed switching/termination rule."),

    _p("not_grasped", [],
       "PRIVILEGED. Equivalent to Not(grasped) but named directly: 4 primitives use a standalone "
       "drop-penalty row over it (DropPenalty/DroppedPenalty in the PredicateBonus merge)."),

    _p("above_z", [Param("z", "float", None, "z threshold, meters", True)],
       "object's z exceeds `z`. PRIVILEGED."),

    _p("below_height", [Param("z", "float", None, "z threshold, meters", True)],
       "object's z is under `z`. PRIVILEGED."),

    _p("within_radius", [
            Param("anchor", "str", None, "named point/measure the radius is centered on", True),
            Param("radius_expr", "str", None,
                  "radius; MAY be an expr over scene attrs, e.g. "
                  "'bin.inner_radius - 0.3*object.half_size'", True),
        ],
       "object lies within `radius_expr` of `anchor`. Required by 4 primitives whose tolerance is "
       "size-relative, not a fixed number."),

    _p("in_cylinder", [
            Param("radius", "float", None, "cylinder radius, meters", True),
            Param("floor", "float", 0.0, "z floor of the cylinder, meters"),
        ],
       "xy within `radius` of the anchor AND z above `floor`. Container-interior test."),

    _p("at_rest", [
            Param("linear", "float", None, "linear velocity ceiling, m/s"),
            Param("angular", "float", None, "angular velocity ceiling, rad/s"),
        ],
       "velocity under the given ceiling(s); either bound may be omitted. NEVER gate on angular "
       "alone for a grasped object (CLAUDE.md gotcha 2, ~98% contact-solver noise)."),

    _p("undisturbed", [
            Param("drift", "float", None, "xy drift tolerance from the reset pose, meters", True),
            Param("tilt", "float", None, "tilt tolerance", True),
        ],
       "moved less than `drift`, tilted less than `tilt`, since reset. Backs container-drift and "
       "stack-intact checks."),

    _p("height_above_resting_in", [Param("band", "float", None, "upper bound, meters", True)],
       "height_above_resting in [0, band]. descend uses this INSTEAD OF an at-rest gate for "
       "success — a held cube being positioned is never stationary, at-rest never latches."),

    _p("and_", [Param("terms", "list[predicate]", None, "predicates to conjoin", True)],
       "true iff every listed predicate is true. Required by 5 primitives."),

    _p("or_", [Param("terms", "list[predicate]", None, "predicates to disjoin", True)],
       "true iff any listed predicate is true."),

    _p("not_", [Param("term", "predicate", None, "predicate to negate", True)],
       "logical negation of one predicate."),

    _p("sustained", [
            Param("predicate", "predicate", None, "the predicate to track", True),
            Param("k", "int", 1, "number of steps required"),
            Param("consecutive", "bool", True,
                  "True: one failing step resets the streak (7 primitives). False: accumulates, "
                  "NEVER resets on slip (grab/sphere_grab) — NOT cosmetic, the cumulative version "
                  "false-passed flaky grips before 2026-06-25."),
        ],
       "`predicate` has held for `k` steps, per `consecutive`."),

    _p("latched", [Param("predicate", "predicate", None, "the predicate to latch", True)],
       "OR-accumulated: once true it stays true for the episode. move_to_target/move_over_bin's "
       "success — the bonus it feeds pays every remaining step."),

    _p("forall", [
            Param("predicate", "predicate", None, "predicate evaluated per member", True),
            Param("over", "str", None, "collection to quantify over, e.g. 'bricks_in_bin'", True),
        ],
       "true iff `predicate` holds for every member of `over`. What 'all bricks in the bin' needs."),

    _p("for_n", [
            Param("predicate", "predicate", None, "predicate evaluated per member", True),
            Param("over", "str", None, "the collection to quantify over", True),
            Param("n", "int", None, "minimum count of members satisfying predicate", True),
        ],
       "true iff at least `n` members of `over` satisfy `predicate` — forall's partial-credit sibling."),
])


# ── terms ────────────────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass(frozen=True)
class Term:
    name: str
    params: tuple
    needs_signed_measure: bool
    stateful: bool
    doc: str


def _t(name, params, doc, needs_signed_measure=False, stateful=False):
    return name, Term(name, tuple(params), needs_signed_measure, stateful, doc)


TERMS = dict([
    _t("ActionPenalty", [
            Param("weight", "float", 0.001, "penalty magnitude"),
            Param("norm", "str", "l2", "vector norm; l2 in all 15 audited instances"),
            Param("measure", "str", "action_norm",
                  "which measure to penalize; jerk-like variant is `action_delta_norm`"),
        ],
       "`reward -= weight * norm(measure)`. NOT a jerk penalty, despite NINE source files calling it "
       "one — no env has ever stored a previous action. `measure` defaults to `action_norm` "
       "(identical everywhere: weight=0.001, norm=l2); `action_delta_norm` is the jerk-LIKE variant "
       "those comments describe, and ships at chassis weight 0.0 so enabling it is a deliberate "
       "sweep, not a silent parity break."),

    _t("SuccessBonus", [
            Param("value", "float", 9.0, "terminal bonus paid on success"),
            Param("mode", "str", "add",
                  "add | replace — replace is `torch.where(success, value, reward)`, a positional "
                  "fold over the PRECEDING rows, not an independent field"),
            Param("scope", "str", "preceding", "which rows mode=replace overwrites"),
            Param("predicate_ref", "str", "per_step",
                  "per_step | latched — latched pays every step once success has EVER been true"),
        ],
       "constant paid on the success predicate. `mode=add` in 11/15 primitives (values 9.0 x7, "
       "12.0 x1, 50.0 x2, 12.0 x1); `mode=replace` in 4 — fires after every shaping row but BEFORE "
       "ActionPenalty, so the success step pays `value - action_penalty` and silently annihilates "
       "the drop/velocity/crush terms unless `scope` is honored positionally."),

    _t("PredicateBonus", [
            Param("weight", "float", 1.0, "signed constant; negative for a penalty row (e.g. drop)"),
            Param("predicate", "str", None, "name of a PREDICATES entry, may be a conjunction", True),
            Param("mode", "str", "add",
                  "add | replace | floor — amendment A: any gated row may need this, not only "
                  "SuccessBonus", choices=("add", "replace", "floor")),
            Param("scope", "str", "preceding", "rows `mode` operates over"),
        ],
       "`reward += weight * predicate`. The largest merge in the corpus (13/15 primitives, 23 "
       "instances) — HoldBonus, GraspBonus, HoldAloftBonus, InZoneBonus, InRegionBonus, DropPenalty "
       "are all this one row with a different predicate or a negated weight. `replace`/`floor` exist "
       "for the same reason as SuccessBonus.mode: a row may need to override, not just accumulate."),

    _t("DistancePull", [
            Param("weight", "float", 1.0, "shaping magnitude"),
            Param("measure", "str", None, "distance-like measure to shape", True),
            Param("kernel", "str", "one_minus_tanh", "one_minus_tanh | neg_linear | gaussian"),
            Param("k", "float", 4.0, "kernel sharpness (tanh/gaussian); observed 3.0/4.0/5.0/6.0"),
            Param("setpoint", "float", 0.0,
                  "the kernel's peak, in measure units — NON-ZERO in 4 instances, load-bearing"),
            Param("axes", "str", None, "restrict to a subset of axes, e.g. split xy from z"),
            Param("gate", "str", None, "predicate name; the whole row is multiplied by it"),
        ],
       "`weight * kernel(measure - setpoint) * gate`. Two params are load-bearing, not decoration: "
       "`setpoint != 0` — descend's hover attractor peaks at `_HOVER` ABOVE the seat and NEVER at "
       "the seat (setpoint=0 pulled the cube INTO the platform, broke 16/16 grasps, the 2026-06-04 "
       "slip-fix) — and `axes`: move_to_3d/descend split xy (weight 1.5, k=4) from z (weight 2.5, "
       "k=6); collapsing both into a 3D norm changes the task."),

    _t("HingePenalty", [
            Param("weight", "float", 1.0, "penalty magnitude"),
            Param("measure", "str", None, "the SIGNED measure being bounded", True),
            Param("threshold", "float", 0.0, "the bound"),
            Param("side", "str", "below",
                  "'above' penalizes measure > threshold, 'below' penalizes measure < threshold"),
            Param("gate", "str", None, "predicate name; multiplies the whole row"),
            Param("enabled_if", "str", None,
                  "spec-level condition gating whether this row exists at all"),
        ],
       "`-weight * clamp(signed_delta(side, measure, threshold), min=0) * gate`. One row covers six "
       "unrelated jobs across 9/15 primitives: crush, jaw creep, container drift, stack topple, "
       "spawn-xy bulldozing, finger-grinding force. REQUIRES a SIGNED measure — an unsigned "
       "height_above_seat_live makes `clamp(-sdz, min=0)` identically zero and deletes the term.",
       needs_signed_measure=True),

    _t("VelocityPenalty", [
            Param("body", "str", "held", "which body's velocity to penalize"),
            Param("linear_weight", "float", 0.3, "linear-speed penalty weight"),
            Param("angular_weight", "float", 0.05, "angular-speed penalty weight"),
        ],
       "`-linear_weight*norm(v) - angular_weight*norm(omega)`. Identical across the 4 carry chassis "
       "instances: 0.3/0.05. CLAUDE.md gotcha (2): a grasped cube's angular velocity is ~98% "
       "contact-solver noise (reads ~22 rad/s while the cube visibly rotates ~0.45 rad/s) — this "
       "term faithfully reproduces that mistake."),

    _t("Ramp", [
            Param("weight", "float", 8.0, "peak reward at cap"),
            Param("measure", "str", None, "the height-like measure being ramped", True),
            Param("floor", "float", 0.0,
                  "where the ramp starts; MAY be a per-env value (e.g. cube_half_sizes)"),
            Param("cap", "float", None,
                  "where the ramp saturates, measure units — an ABSOLUTE ceiling, not a span", True),
            Param("normalize", "bool", True,
                  "True: weight IS the maximum (divides by cap-floor). False: max is "
                  "weight*(cap-floor) — see doc below, the two real instances need different values"),
            Param("gate", "str", None, "predicate name; multiplies the whole row"),
        ],
       "`weight * clamp(measure-floor, 0, cap-floor) / (cap-floor if normalize else 1)`. Was "
       "`HeightRamp`, renamed in amendment A. `normalize` exists because ONE formula does not fit "
       "both real instances: lift/sphere_lift are normalized (max=8.0) but compact_grasp is not "
       "(max=0.4) — the normalized default on compact_grasp yields max 10.0, swamping GRASP_W=3.0 "
       "and training a lift instead of a seated grip, 25x too large."),

    _t("ProgressPotential", [
            Param("weight", "float", 5.0, "potential-difference weight"),
            Param("measure", "str", None, "distance-like measure the potential is built from", True),
            Param("gate", "str", None, "predicate name; multiplies the whole row"),
            Param("reseed_on_restore", "bool", True, "re-seed the buffer from the post-restore state"),
            Param("gamma", "float", 1.0, "discount on the potential difference"),
            Param("terminal_zero", "bool", False, "force potential to zero on the terminal step"),
        ],
       "`weight * (prev_measure - measure) * gate`, then `prev_measure <- measure`. The ONLY "
       "genuinely stateful term in the corpus: move_to_target (weight 5.0) and move_over_bin "
       "(weight 10.0, over `object_to_goal_xy_plus_z` — see that measure's L1-vs-norm note). "
       "Partial-reset rule the framework must own: seed ONLY the resetting rows' buffer entries — "
       "seeding all rows erases in-flight potential and injects a spurious one-step spike into "
       "every running env.",
       stateful=True),

    _t("RewardScale", [
            Param("divisor", "float", 12.0, "value dense reward is divided by before PPO sees it"),
            Param("unnormalized", "bool", False, "declare instead of dividing, if already at scale"),
        ],
       "a DOCUMENT-level field, not a summed reward row: `reward_ppo = dense / divisor` (or dense "
       "unchanged if `unnormalized`). 7/15 primitives (reach, sphere_reach, grab, sphere_grab, lift, "
       "sphere_lift, compact_grasp) inherit `compute_normalized_dense_reward` WITHOUT overriding it, "
       "so they train at dense/12.0 even though it's semantically wrong for them — lift's per-step "
       "max is ~18, reach's is ~9. A generated env that forgets this trains at 12x intended scale."),
])

assert len(TERMS) == 9, "the vocabulary is frozen at 9 terms — additions need a measured justification"


# ── chassis ──────────────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass(frozen=True)
class Chassis:
    name: str
    doc: str
    defaults: dict


def _row(why, **kw):
    kw["why"] = why
    return kw


CHASSIS = {
    "approach": Chassis(
        name="approach",
        doc="reach, sphere_reach (2 primitives). Pure approach: close the tcp-to-object gap, "
            "nothing else. reach == sphere_reach byte-for-byte per the audit.",
        defaults={
            "DistancePull": _row(
                weight=1.0, measure="tcp_to_object", kernel="neg_linear",
                why="reach's whole dense signal is `reward = -tcp_to_object` — neg_linear at "
                    "weight 1.0, no curvature yet."),
            "SuccessBonus": _row(
                value=9.0, mode="add",
                why="terminal +9.0, the 'avg reward / 9' dashboard convention shared across "
                    "primitives so runs stay comparable on one scale."),
            "ActionPenalty": _row(
                weight=0.001, measure="action_norm",
                why="same 0.001/l2 in all 15 audited primitives."),
            "RewardScale": _row(
                divisor=12.0, unnormalized=False,
                why="inherited 12.0 (see RewardScale term above): reach/sphere_reach never "
                    "override it, but this chassis' own max is ~9, not 12 — kept at 12.0 here only "
                    "to match the deployed lineage, not to copy onto a new chassis."),
        },
    ),
    "close_and_hold": Chassis(
        name="close_and_hold",
        doc="grab, sphere_grab (2 primitives). Approach chassis plus a hold bonus paid every step "
            "the grip survives a FROZEN gripper — the freeze lives in step(), outside this reward's "
            "reach (audit §2 #1: reward alone here would train 'touch the cube').",
        defaults={
            "DistancePull": _row(
                weight=1.0, measure="tcp_to_object", kernel="neg_linear",
                why="kept from approach — grab starts ~3cm from the cube and still needs the "
                    "gradient while closing."),
            "PredicateBonus": _row(
                weight=3.0, predicate="grasped",
                why="paid EVERY step held (not just on first contact) so the policy is rewarded for "
                    "SUSTAINING the grip through the frozen null-hold — 3x the raw distance term."),
            "SuccessBonus": _row(
                value=9.0, mode="add",
                why="terminal +9.0 once the grip survives HOLD_K frozen steps."),
            "ActionPenalty": _row(
                weight=0.001, measure="action_norm", why="same 0.001/l2 as every other primitive."),
            "RewardScale": _row(
                divisor=12.0, unnormalized=False,
                why="inherited 12.0 (see RewardScale term above): grab/sphere_grab never override "
                    "it; the sustained hold reward (~3) sits well under 12 — the divisor fits only "
                    "the success-step peak, not the signal dominating the episode."),
        },
    ),
    "hold_and_ramp": Chassis(
        name="hold_and_ramp",
        doc="lift, sphere_lift, compact_grasp (3 primitives; lift == sphere_lift byte-for-byte). "
            "Hold the grip while ramping a height-like measure — the one chassis where the real "
            "instances need DIFFERENT Ramp.normalize settings.",
        defaults={
            "PredicateBonus": _row(
                weight=1.0, predicate="grasped",
                why="keep the cube grasped through the ascent — releasing for a split second cuts "
                    "the lift-progress signal entirely."),
            "Ramp": _row(
                weight=8.0, measure="object_z", floor=0.0, cap=0.06, normalize=True, gate="grasped",
                why="lift/sphere_lift: `8.0*clamp(cube_z/0.06,0,1)`, normalized so weight IS the max "
                    "(8.0). compact_grasp needs normalize=False, floor=cube_half_sizes (per-env), "
                    "cap=floor+0.04 instead — `10.0*clamp(cube_z-half,0,0.04)`, max 0.4. This "
                    "default's normalize=True on compact_grasp would train a lift not a seated grip, "
                    "25x too large."),
            "SuccessBonus": _row(
                value=9.0, mode="add", why="terminal +9.0, dashboard convention."),
            "ActionPenalty": _row(weight=0.001, measure="action_norm", why="same 0.001/l2."),
            "RewardScale": _row(
                divisor=12.0, unnormalized=False,
                why="inherited 12.0 (see RewardScale term above): lift/sphere_lift never override "
                    "it, but this chassis' own max is ~18, not 12 — trains at ~2/3 scale. "
                    "compact_grasp's Ramp caps at 0.4 not 8.0, so recompute per instance."),
        },
    ),
    "carry": Chassis(
        name="carry",
        doc="move_to_3d, descend_to_target, descend_stack, place_into_bin (4 primitives, IDENTICAL "
            "weights per the audit). Held cube travels to a hover point over the goal without "
            "releasing; the richest chassis, 9 rows. Sourced from descend_to_target/descend_env.py.",
        defaults={
            "PredicateBonus": _row(
                weight=1.0, predicate="grasped",
                why="hold-on baseline — never drop the cube, release is a separate primitive."),
            "DistancePull_xy": _row(
                weight=1.5, measure="object_to_goal_xy", kernel="one_minus_tanh", k=4.0,
                gate="grasped",
                why="re-center over the target while held; move delivers ~6cm off. k=4 on xy is "
                    "DIFFERENT from k=6 on height below — one 3D-norm kernel changes the task."),
            "DistancePull_height": _row(
                weight=2.5, measure="height_above_seat_live", kernel="one_minus_tanh", k=6.0,
                setpoint=0.015, gate="grasped",
                why="descend to a HOVER just above the seat — attractor peaks at setpoint=_HOVER "
                    "(0.015), NEVER at the seat (setpoint=0). The old zero-setpoint version pulled "
                    "the cube INTO the platform: 16/16 grasp losses broke while LOW, fixed "
                    "2026-06-04. Peaking at contact is a recorded failure, not a free parameter."),
            "HingePenalty_crush": _row(
                weight=3.0, measure="height_above_seat_live", threshold=0.0, side="below",
                gate="grasped",
                why="pressing the cube BELOW resting height destabilizes the grasp — the other half "
                    "of the 2026-06-04 slip-fix above. Needs SIGNED height_above_seat_live: "
                    "unsigned, `clamp(-sdz,min=0)` is identically zero and this term vanishes."),
            "HingePenalty_grip": _row(
                weight=1.0, measure="gripper_qpos", threshold=-0.6, side="above", gate="grasped",
                why="keep the gripper CLOSED while held — grip_q drifted -0.73 to -0.44 over the "
                    "descent before this term existed."),
            "VelocityPenalty": _row(
                linear_weight=0.3, angular_weight=0.05,
                why="anti-fling on the way down. Identical 0.3/0.05 across all 4 members; angular "
                    "knowingly shapes on the ~98% contact-solver noise, CLAUDE.md gotcha (2)."),
            "PredicateBonus_drop": _row(
                weight=-0.5, predicate="not_grasped",
                why="strongly discourage letting go early — collapses the handoff to the next "
                    "primitive."),
            "SuccessBonus": _row(
                value=12.0, mode="replace", scope="preceding",
                why="held+low+centered jackpot REPLACES accumulated shaping "
                    "(`torch.where(success,12.0,reward)`), not adds — fires after every shaping row "
                    "above but before ActionPenalty below."),
            "ActionPenalty": _row(
                weight=0.001, measure="action_norm",
                why="same 0.001/l2; applied AFTER mode=replace, so it survives the reward's reset "
                    "to 12.0."),
        },
    ),
    "carry_with_potential": Chassis(
        name="carry_with_potential",
        doc="move_to_target, move_over_bin (2 primitives). Adds a potential-based progress term and "
            "a LATCHED success — once the cube has ever reached the zone, success stays true, "
            "handing the deceleration/stop phase to the next primitive.",
        defaults={
            "PredicateBonus_hold": _row(
                weight=0.3, predicate="grasped",
                why="small constant baseline, deliberately weak so it doesn't crowd out the "
                    "distance-shaped terms."),
            "PredicateBonus_hold_high": _row(
                weight=0.3, predicate="and_(grasped, above_z(z=0.06))",
                why="second half of the baseline, gated on carry altitude (MIN_CARRY_Z=0.06) so "
                    "descending onto the platform early never collects it."),
            "DistancePull": _row(
                weight=1.5, measure="object_to_goal_xy", kernel="one_minus_tanh", k=3.0,
                gate="and_(grasped, above_z(z=0.06))",
                why="gives an early policy a gradient toward the target instead of just holding the "
                    "cube up. Weight 1.5 is chosen to stay BELOW the 5.0 arrival bonus "
                    "(PredicateBonus_arrived) so there's still a strong commitment signal — crossing "
                    "into the zone beats hovering near it. The previous fully-sparse design (no "
                    "proximity term) cost 178M from-scratch steps at 0% success."),
            "ProgressPotential": _row(
                weight=5.0, measure="object_to_goal_xy", gate="grasped",
                why="potential-based per-step DECREASE in distance, telescoping to ~1.5 max per "
                    "episode. Assigned post-reward, seeded per env_idx at reset so step 0 always "
                    "scores zero progress — reordering this changes the reward."),
            "PredicateBonus_arrived": _row(
                weight=5.0,
                predicate="and_(grasped, above_z(z=0.06), within_radius(anchor=target_pos, "
                          "radius_expr=0.05))",
                why="sharp step at the tolerance boundary — d=5.1cm pays ~1.4, d=4.9cm pays ~5.4. "
                    "The jump IS the commitment signal, paid only for crossing in "
                    "(tolerance MOVE_TOLERANCE=0.05)."),
            "SuccessBonus": _row(
                value=50.0, mode="add", predicate_ref="latched",
                why="terminal +50 once the LATCHED success first fires (has_arrived, "
                    "OR-accumulated). Smaller than an earlier +200-with-strict-streak design because "
                    "this criterion is easier to satisfy: one frame at target is enough."),
            "ActionPenalty": _row(weight=0.001, measure="action_norm", why="same 0.001/l2, kept tiny."),
        },
    ),
    "release": Chassis(
        name="release",
        doc="release, release_into_bin (2 primitives). Open the gripper; the cube falls under "
            "gravity. Shortest chassis in the corpus.",
        defaults={
            "PredicateBonus": _row(
                weight=5.0, predicate="not_grasped",
                why="'the entire teaching signal' per source comment, paid the moment the gripper "
                    "opens enough that contact is lost. `released = ~is_grasping` in source — "
                    "exactly `not_grasped`."),
            "SuccessBonus": _row(
                value=9.0, mode="add",
                why="terminal +9.0 once release survives a 2-step not-grasping streak."),
            "ActionPenalty": _row(weight=0.001, measure="action_norm", why="same 0.001/l2."),
        },
    ),
}

assert len(CHASSIS) == 6, "6 chassis cover all 15 audited primitives — a 7th needs a new primitive"


# ── the document a 27–30B model reads ───────────────────────────────────────────────────────────

def _fmt_param(p: Param) -> str:
    req = ", required" if p.required else f", default={p.default!r}"
    ch = f", choices={list(p.choices)}" if p.choices else ""
    return f"  - `{p.name}` ({p.type}{req}{ch}): {p.doc}" if p.doc else f"  - `{p.name}` ({p.type}{req}{ch})"


def _fmt_row(term_name: str, row: dict) -> str:
    fields = ", ".join(f"{k}={v!r}" for k, v in row.items() if k != "why")
    return f"- **{term_name}** {{{fields}}}\n  why: {row['why']}"


def base_term(term_name: str) -> str:
    """A chassis row key may carry a disambiguating suffix (e.g. "DistancePull_xy") when a chassis
    instantiates the same term twice with different measures — the suffix is a row label, not a
    second term type. Strip it back to the real TERMS key. Shared by the renderer and by
    test_vocab.py's declared-params check, so the two can't drift apart on what "the term" means.
    """
    head = term_name.split("_")[0]
    return head if head in TERMS else term_name


def vocab_document() -> str:
    """Render the entire authorable surface as compact markdown: the prompt payload handed to a
    local 27-30B model authoring a skill.yaml. Kept dense on purpose (design doc §8) — the audit
    budgeted 3,400-4,600 tokens for this whole payload, alongside a task description and one
    worked example, so every sentence here earns its place.
    """
    lines = []
    add = lines.append

    add("# Bridle skill vocabulary — the authorable reward surface")
    add("")
    add(f"{len(TERMS)} reward terms, {len(MEASURES)} measures, {len(PREDICATES)} predicates, "
        f"{len(CHASSIS)} chassis. Weights are FREE — you set them — but every chassis default below "
        "carries a `why`: read it before changing the number. The measured failure mode for "
        "LLM-authored rewards is bad WEIGHTS, not bad term choice.")
    add("")
    add("Every measure declares a SIGN (`signed`: can be negative, e.g. below vs above a surface; "
        "`magnitude`: always >= 0) and a FRAME (`live`: fresh every step; `at_reset`: vs a value "
        "captured once at reset; `static_goal`: vs a goal frozen once). Both load-bearing: an "
        "unsigned height_above_seat_live silently zeroes a crush penalty that exists because a "
        "signed version broke 16/16 grasps; a frame mismatch grades reward and success on two "
        "truths. A two-frame quantity carries its frame IN THE KEY; there is no bare spelling.")
    add("")

    add("## Terms")
    add("")
    for name, term in TERMS.items():
        flags = []
        if term.stateful:
            flags.append("STATEFUL")
        if term.needs_signed_measure:
            flags.append("needs a SIGNED measure")
        flag_str = f"  [{', '.join(flags)}]" if flags else ""
        add(f"### {name}{flag_str}")
        for p in term.params:
            add(_fmt_param(p))
        add(f"\n{term.doc}")
        add("")

    add("## Measures")
    add("")
    add("`name  [sign, frame, unit]  —  doc`")
    add("")
    for name, m in MEASURES.items():
        add(f"- `{name}`  [{m.sign.value}, {m.frame.value}, {m.unit}] — {m.doc}")
    add("")

    add("## Predicates")
    add("")
    for name, pred in PREDICATES.items():
        params = ", ".join(p.name for p in pred.params)
        sig = f"{name}({params})" if params else name
        add(f"- `{sig}` — {pred.doc}")
    add("")
    add("Composing predicates: `predicate`/`gate` accept a bare name above, or a nested call over "
        "EXISTING names — never invent a compound name. Example: `and_(grasped, above_z(z=0.06), "
        "within_radius(anchor=target_pos, radius_expr=0.05))`.")
    add("")

    add("## Chassis (weight presets — start here)")
    add("")
    for name, chassis in CHASSIS.items():
        add(f"### {name}")
        add(chassis.doc)
        for term_name, row in chassis.defaults.items():
            add(_fmt_row(base_term(term_name), row))
        add("")

    add("## Constraints `compile()` checks before any GPU run")
    add("")
    add("- `max_shaping_below: success_bonus` — per-step shaping maxima must stay below the success "
        "value (move_to_target: 1.5 < 5.0 < 50.0) — the fully-sparse alternative measured 178M "
        "steps at 0% success.")
    add("- `attractor_setpoint_not_at: contact` — a DistancePull setpoint over a SIGNED measure must "
        "not peak at the contact surface (descend's hover, 0.015, never 0) — peaking at contact is "
        "the 16/16 grasp-loss failure mode.")

    return "\n".join(lines)
