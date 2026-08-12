"""bridle.skill.compile — the `SkillSpec --compile()--> RewardPlan` arrow.

WHAT THIS IS: the second tier of feedback (schema -> COMPILE -> preflight, design doc §8). `spec.py`
checked that every row is well-formed in isolation; this module lowers the rows to `Op`s, checks the
things that only make sense ACROSS rows (a shaping maximum against the success value), and stamps the
result with a digest that survives the process it was computed in. Stdlib only — no torch, no numpy —
so a reward is fully checkable on a laptop before a GPU-second is spent.

THREE PROPERTIES, AND THE MEASUREMENT BEHIND EACH:

  1. THE FOLD IS ORDERED, NOT A SUM. `acc = op(acc)` in document order. descend's row 8 is
     `SuccessBonus{mode: replace, scope: preceding}` and row 9 is `ActionPenalty`, so the deployed
     env pays `12.0 - 0.001*||a||` on the success step (descend_env.py:210-212) — not 12.0, and not
     the sum of everything. A compiler that sums rows trains a different reward from the one measured
     at 0.85 while producing identical rows, identical weights and identical log lines.

  2. THE FLOODING CHECK REFUSES PER-STEP AND ONLY WARNS ON THE HORIZON (user decision,
     phase2-decisions §1). See `_check_flooding` — the reasoning lives at the check, because in this
     codebase the comment carries the measurement.

  3. THE FINGERPRINT IS sha256 OVER CANONICAL JSON, NEVER `hash()`. `hash()` is salted per process by
     PYTHONHASHSEED, so a checkpoint stamped with it is unverifiable on the next run — which is the
     one thing the stamp exists to do. Same rule as `Contract.fingerprint()`/`Rig.fingerprint()`.

WHAT IT REFUSES BEYOND `spec.py`, AND WHY THE LIST IS NOT "EXTRA": a parameter the fold cannot honour
is REFUSED, never quietly dropped. `kernel: one_minus_tan`, `scope: all`, `axes: xy`,
`predicate_ref: whenever` all pass the schema (those fields carry no `choices` in the vocabulary) and
would otherwise train a reward the author did not write, log clean, and contribute nothing — the exact
shape of the crush penalty that silently vanished over an unsigned measure and cost 16/16 grasps.

THE HORIZON IS AN ARGUMENT, NOT A DOCUMENT FIELD. `SkillSpec` has no `execution:` block, so
`compile_spec(spec, *, horizon=None, terminate_on_success=None)` takes it from the caller. With
`horizon=None` the integrated check reports that it could NOT be computed; it never substitutes a
default and never omits the line. "Cannot verify" rendering as "verified" is a shipped bug in this
repo's own history: `bridle lineage` printed `0 violation(s)` and exited 0 on a machine with no
`systemctl`, i.e. a clean bill of health for checks it never ran.
"""
import dataclasses
import difflib
import hashlib
import json
import math
import re
import warnings
from types import MappingProxyType

from bridle.skill.expr import Expr
from bridle.skill.spec import ROW_TERMS, SkillSpec
from bridle.skill.vocab import MEASURES, TERMS

__all__ = [
    "CompileError", "FloodingError", "Op", "RewardPlan", "UNBOUNDED",
    "compile_spec", "evaluate_plan",
]


# ── errors ──────────────────────────────────────────────────────────────────────────────────────

class CompileError(Exception):
    """One refusal from the compiler, addressed to a model that cannot read this file.

    Same shape as `SpecError` on purpose (path, legal set, nearest-match suggestion): the author is a
    local 27-30B LLM and the measured cost of feedback it cannot act on is 58.3% +/- 47.3% one-shot
    against 97.6% with actionable refinement. A refusal that only says "no" costs a round trip.
    """

    def __init__(self, path, problem, legal=None, suggestion=None):
        self.path = path
        self.suggestion = suggestion
        message = f"{path}: {problem}"
        if suggestion is not None:
            message += f" — did you mean {suggestion!r}?"
        if legal:
            message += f" — legal values: {', '.join(sorted(str(v) for v in legal))}"
        super().__init__(message)


class FloodingError(CompileError):
    """Per-step shaping can out-earn completing the task. See `_check_flooding`."""


def _suggest(value, candidates):
    if not isinstance(value, str):
        return None
    close = difflib.get_close_matches(value, sorted(str(c) for c in candidates), n=1, cutoff=0.6)
    return close[0] if close else None


def _num(x):
    """Numbers in refusals are printed via `repr(float)` — always with a decimal point, so `12` and
    `12.0` cannot read as two different quantities in two different messages."""
    return repr(float(x))


# ── batch-safe primitives ───────────────────────────────────────────────────────────────────────
# Written branch-free — `c * a + (1 - c) * b`, the same idiom `expr.py` uses — because the identical
# fold has to mean the same thing for a CPU float in a unit test and a batched CUDA tensor of 4096
# environments in Task 5's adapter. A Python `if c: a else: b` calls `bool()` on the whole batch and
# takes ONE branch for all 4096 envs at once, which is not a slower version of the right answer; it
# is a different reward. Same reason `tanh`/`exp` dispatch to the value's own method first: a torch
# tensor has `.tanh()`, a float does not, and one expression must mean one thing for both.

def _where(c, a, b):
    return c * a + (1 - c) * b


def _max(a, b):
    return _where(a > b, a, b)


def _min(a, b):
    return _where(a < b, a, b)


def _clamp(x, lo, hi):
    return _min(_max(x, lo), hi)


def _relu(x):
    return _max(x, 0.0 * x)


def _dispatch(method_name, math_fn):
    def call(x):
        method = getattr(x, method_name, None)
        return method() if callable(method) else math_fn(x)
    return call


_tanh = _dispatch("tanh", math.tanh)
_exp = _dispatch("exp", math.exp)


# ── binding `params.X` ──────────────────────────────────────────────────────────────────────────
# `spec.py` accepts `setpoint: params.hover` and guarantees only that `hover` is DECLARED. Binding is
# this module's job, and it happens BEFORE every check below: descend's hover attractor is written
# `setpoint: params.hover`, so a compiler that checked the setpoint before binding would be checking
# the string `"params.hover"` against the contact surface and would never see a `hover` of 0.0.
# Binding here is also what puts the number itself into the fingerprint — `hover: 0.015 -> 0.016` is
# a different reward and must be a different digest.

_PARAM_REF_RE = re.compile(r"\bparams\.([A-Za-z_][A-Za-z0-9_]*)")


def _param_value(path, name, spec_params):
    entry = spec_params.get(name)
    if entry is None:      # unreachable: parse_spec refuses an undeclared reference
        raise CompileError(path, f"`params.{name}` is not declared in this document",
                           legal=list(spec_params), suggestion=_suggest(name, spec_params))
    return entry["value"]


def _bind_number(path, value, spec_params):
    """A numeric field: either already a number, or exactly `params.X` naming one."""
    if isinstance(value, str):
        if not value.startswith("params."):
            raise CompileError(path, f"{value!r} is not a number and not a `params.X` reference to "
                                     f"one", legal=[f"params.{n}" for n in spec_params])
        name = value[len("params."):]
        bound = _param_value(path, name, spec_params)
        if type(bound) not in (int, float) or isinstance(bound, bool):
            raise CompileError(
                path,
                f"`{value}` has to resolve to a number for this field, but `params.{name}` declares "
                f"value={bound!r} ({type(bound).__name__}). Give the param a numeric `value`, or "
                f"write the number here")
        return float(bound)
    return float(value)


def _bind_text(path, text, spec_params):
    """A predicate/gate string: every `params.X` inside it is substituted, so the compiled plan is
    self-contained and the number lands in the fingerprint like any other."""
    def sub(match):
        bound = _param_value(path, match.group(1), spec_params)
        if isinstance(bound, bool) or type(bound) not in (int, float, str):
            raise CompileError(path, f"`{match.group(0)}` resolves to {bound!r} "
                                     f"({type(bound).__name__}), which cannot be substituted into "
                                     f"an expression")
        return bound if isinstance(bound, str) else repr(float(bound))
    return _PARAM_REF_RE.sub(sub, text)


# ── values the fold can honour ──────────────────────────────────────────────────────────────────
# A closed set per (term, parameter), checked here rather than in the schema because these are the
# values THIS COMPILER implements, not the values the vocabulary describes. The vocabulary documents
# `kernel: one_minus_tanh | neg_linear | gaussian` in prose and carries no `choices`, so a typo
# reaches this module intact; accepting it and silently folding the default is how a row comes to
# train, log, and mean something nobody wrote.

_HONOURED = {
    ("DistancePull", "kernel"): ("one_minus_tanh", "neg_linear", "gaussian"),
    ("HingePenalty", "side"): ("above", "below"),
    ("ActionPenalty", "norm"): ("l2",),
    ("VelocityPenalty", "body"): ("held",),
    ("SuccessBonus", "mode"): ("add", "replace", "floor"),
    ("SuccessBonus", "scope"): ("preceding",),
    ("SuccessBonus", "predicate_ref"): ("per_step", "latched"),
    ("PredicateBonus", "mode"): ("add", "replace", "floor"),
    ("PredicateBonus", "scope"): ("preceding",),
}

#: Parameters the vocabulary declares but this fold does not implement, with the only value that is
#: therefore safe (their "off" default) and what to write instead. Refusing is the point: `axes: xy`
#: silently ignored is a 3D pull where the author asked for a planar one, and move_to_3d/descend split
#: xy (1.5@k=4) from z (2.5@k=6) precisely because collapsing them changes the task.
_UNIMPLEMENTED = {
    ("DistancePull", "axes"): (
        None, "axis restriction is not implemented; the measures are already axis-specific — use "
              "`object_to_goal_xy` or `object_to_goal_z`"),
    ("HingePenalty", "enabled_if"): (
        None, "`enabled_if` decides whether a row EXISTS and nothing implements it yet; delete the "
              "row, or use `gate:` for a per-step condition"),
    ("ProgressPotential", "gamma"): (
        1.0, "a discount other than 1.0 is not implemented, and the two readings of it "
             "(`prev - gamma*m` vs `gamma*(prev - m)`) are different rewards — guessing would "
             "silently pick one"),
    ("ProgressPotential", "terminal_zero"): (
        False, "forcing the potential to zero on the terminal step is not implemented"),
}


def _check_honoured(path, term, name, value):
    legal = _HONOURED.get((term, name))
    if legal is not None and value not in legal:
        raise CompileError(f"{path}.{name}", f"{value!r} is not a {name} this compiler implements "
                                             f"for {term}", legal=legal,
                           suggestion=_suggest(value, legal))
    unimplemented = _UNIMPLEMENTED.get((term, name))
    if unimplemented is not None and value != unimplemented[0]:
        raise CompileError(f"{path}.{name}",
                           f"{name}={value!r}: {unimplemented[1]}", legal=[repr(unimplemented[0])])


# ── the ordered fold's unit ─────────────────────────────────────────────────────────────────────

@dataclasses.dataclass(frozen=True)
class Op:
    """One lowered reward row.

    `kind`      how it combines with the accumulator: `add` (`acc + v`), `replace`
                (`where(cond, level, acc)`) or `floor` (`where(cond, max(acc, level), acc)`).
    `scope`     which rows the combination reaches — `preceding` (every row emitted so far) or None
                for a plain add. Positional by construction: the accumulator IS the preceding rows.
    `fn_key`    which evaluator: a `TERMS` name, or `expr`/`custom` for the tier-2/tier-3 escapes.
    `params`    every parameter resolved and BOUND (no `params.X` strings survive here).
    `stateful`  the row carries a per-env buffer across steps (ProgressPotential, and only it).
    """

    kind: str
    scope: str | None
    fn_key: str
    params: MappingProxyType
    stateful: bool


# ── per-step maxima ─────────────────────────────────────────────────────────────────────────────

class _Unbounded:
    """The per-step maximum this document cannot state. NOT zero, and the difference is the whole
    point: a row treated as zero is a row silently excluded from the flooding sum."""

    def __repr__(self):
        return "UNBOUNDED"


UNBOUNDED = _Unbounded()


def _max_predicate_bonus(p):
    """`weight * predicate`, predicate in {0, 1} — so the maximum is the weight, or 0 for the
    negative-weight rows the corpus uses as drop penalties (descend's -0.5 on `not_grasped`)."""
    return max(p["weight"], 0.0)


def _max_distance_pull(p):
    """`weight * kernel(measure - setpoint) * gate`, gate in {0, 1}.

    `one_minus_tanh` peaks at 1.0 AT the setpoint (`1 - tanh(0)`) and `gaussian` likewise, so a
    positive weight IS the maximum — this is where descend's 1.5 (xy) and 2.5 (hover) come from.
    `neg_linear` is `-|delta|`, i.e. non-positive, so its maximum is 0 — but only while the weight is
    positive: a negative weight over neg_linear grows without bound as the object moves away.
    """
    weight, kernel = p["weight"], p["kernel"]
    if kernel == "neg_linear":
        return 0.0 if weight >= 0 else UNBOUNDED
    return max(weight, 0.0)


def _max_hinge_penalty(p):
    """`-weight * clamp(signed_delta, min=0) * gate`: non-positive for a non-negative weight, so it
    contributes 0 to the positive shaping a policy can farm. A NEGATIVE weight turns the hinge into an
    unbounded ramp away from the threshold, which no document has ever wanted."""
    return 0.0 if p["weight"] >= 0 else UNBOUNDED


def _max_velocity_penalty(p):
    """`-linear_weight*|v| - angular_weight*|omega|`: non-positive, maximum 0 at rest."""
    return 0.0 if p["linear_weight"] >= 0 and p["angular_weight"] >= 0 else UNBOUNDED


def _max_action_penalty(p):
    """`-weight * ||a||`: non-positive, maximum 0 at a zero action."""
    return 0.0 if p["weight"] >= 0 else UNBOUNDED


def _max_ramp(p):
    """`weight * clamp(m - floor, 0, cap - floor) / (cap - floor if normalize else 1)`.

    ONE FORMULA DOES NOT FIT BOTH REAL INSTANCES, which is why `normalize` exists and why this bound
    reads it: lift is `8.0*clamp(z/0.06, 0, 1)`, normalized, maximum 8.0; compact_grasp is
    `10.0*clamp(z - half, 0, 0.04)`, UN-normalized, maximum 0.4. Applying the normalized rule to
    compact_grasp reports 10.0 for a row that pays 0.4 — 25x too large, the audit's critique-1 #1.
    """
    span = p["cap"] - p["floor"]
    peak = p["weight"] if p["normalize"] else p["weight"] * span
    return max(peak, 0.0)


def _max_success_bonus(p):
    """The completion side of the comparison, never the shaping side — `_check_flooding` reads this
    as the value to stay under, and never sums it into the shaping total."""
    return max(p["value"], 0.0)


def _max_unbounded(p):
    """ProgressPotential's `weight * (prev - measure)` is bounded only by how far the measure can
    travel in one step, which is physics the document does not state; a tier-2 `expr:` is arbitrary
    arithmetic over measures whose ranges are equally unstated; a tier-3 `custom:` is opaque by
    construction. All three must make the check SAY it could not conclude."""
    return UNBOUNDED


#: One explicit table, keyed by term. Every entry above carries why its bound is correct, because the
#: bound is the number the flooding refusal is computed from and a wrong one refuses a working reward
#: (or passes a flooding one) with total confidence.
_PER_STEP_MAXIMUM = {
    "PredicateBonus": _max_predicate_bonus,
    "DistancePull": _max_distance_pull,
    "HingePenalty": _max_hinge_penalty,
    "VelocityPenalty": _max_velocity_penalty,
    "ActionPenalty": _max_action_penalty,
    "Ramp": _max_ramp,
    "ProgressPotential": _max_unbounded,
    "SuccessBonus": _max_success_bonus,
    "expr": _max_unbounded,
    "custom": _max_unbounded,
}
assert set(_PER_STEP_MAXIMUM) == set(ROW_TERMS) | {"expr", "custom"}, (
    "every row term needs a per-step bound: a term missing from this table would fall through to a "
    "default, and any default here is either a silent zero or a silent refusal")


# ── lowering ────────────────────────────────────────────────────────────────────────────────────

#: 0 on any of these measures IS the surface the object rests on, so a peaked attractor at setpoint=0
#: pulls a GRASPED object into it. That is not a hypothetical: descend's pre-2026-06-04 hover was
#: `2.5*(1-tanh(6*dz))`, peaking at dz=0, and broke 16/16 grasps while low; the fix was to peak at
#: _HOVER=0.015 instead. `gripper_qpos` is deliberately NOT here — its 0 is "fully open", not a
#: contact surface — and `object_to_goal_z` is a free 3D goal (move_to_3d peaks at 0 on purpose).
_CONTACT_SURFACE_MEASURES = frozenset({
    "height_above_seat_live", "height_above_seat_static_goal", "height_above_resting",
})
_PEAKED_KERNELS = frozenset({"one_minus_tanh", "gaussian"})


def _check_row_semantics(path, term, values):
    if term == "DistancePull":
        if (values["kernel"] in _PEAKED_KERNELS
                and values["measure"] in _CONTACT_SURFACE_MEASURES
                and values["setpoint"] == 0.0):
            raise CompileError(
                f"{path}.setpoint",
                f"this DistancePull peaks AT the contact surface: setpoint=0.0 over "
                f"{values['measure']!r}, where 0 means the object is resting on the seat. Peaking at "
                f"contact is a recorded failure, not a free parameter — descend's "
                f"`2.5*(1-tanh(6*dz))` pulled the held cube INTO the platform and broke 16/16 grasps "
                f"(2026-06-04); the fix was a setpoint of 0.015, a hover ABOVE the seat. Set a "
                f"positive setpoint, or use HingePenalty if you meant to bound the height")
    if term == "Ramp" and values["cap"] <= values["floor"]:
        raise CompileError(
            f"{path}.cap",
            f"cap={_num(values['cap'])} is not above floor={_num(values['floor'])}, so the ramp has "
            f"no span to climb (and a normalized one divides by {_num(values['cap'] - values['floor'])}). "
            f"`cap` is an ABSOLUTE ceiling in measure units, not a span above the floor")


def _lower_term_row(index, row, spec):
    path = f"reward[{index}]"
    term = row.term
    declared = {p.name: p for p in TERMS[term].params}
    values = {}
    for name, value in row.params.items():
        param = declared[name]
        if value is not None:
            if param.type in ("float", "int") and isinstance(value, str):
                value = _bind_number(f"{path}.{name}", value, spec.params)
            elif param.type == "float":
                value = float(value)     # `weight: 1` and `weight: 1.0` are one reward, one digest
            elif isinstance(value, str):
                value = _bind_text(f"{path}.{name}", value, spec.params)
        _check_honoured(path, term, name, value)
        values[name] = value

    _check_row_semantics(path, term, values)

    kind, scope = "add", None
    if "mode" in values:
        # `mode`/`scope` become the Op's kind/scope rather than staying parameters: one
        # representation, so a fingerprint cannot record a scope the fold is not using. An `add` row
        # keeps no scope at all — it reaches nothing but the accumulator.
        kind = values.pop("mode")
        declared_scope = values.pop("scope", None)
        scope = declared_scope if kind != "add" else None
    if term == "SuccessBonus":
        # The row's condition is the DOCUMENT's success criterion — the same truth the env publishes
        # as `info["success"]`. Carried on the op so the plan is self-contained and so the criterion
        # hashes into the fingerprint: `where(success, 12.0, acc)` with a different `success` is a
        # different reward function, however identical the rows look.
        values["condition"] = _bind_text("success", spec.success, spec.params)
    stateful = TERMS[term].stateful
    if stateful:
        # One buffer per stateful ROW, not per measure: two ProgressPotential rows over the same
        # measure are two potentials, and sharing a slot would make each read the other's previous
        # value. The row index is in the name because that is what makes it unique.
        values["slot"] = f"{path}.prev_{values['measure']}"
    return Op(kind=kind, scope=scope, fn_key=term, params=MappingProxyType(values),
              stateful=stateful)


def _expr_bindings(index, expr, spec_params):
    """The declared params a tier-2 expression reads, BOUND to their numbers.

    `spec.py` lets an expression name a declared param directly (`... - hover`), shadowing a measure
    of the same name. Those numbers have to travel WITH the expression: otherwise the evaluator has
    no value for the name, and — worse — two documents whose only difference is `hover: 0.015` vs
    `0.016` would compile to the same source text and therefore the same fingerprint, which is a
    digest claiming two different rewards are one.
    """
    bindings = {}
    for name in sorted(expr.names):
        if name not in spec_params:
            continue
        value = spec_params[name]["value"]
        if isinstance(value, bool) or type(value) not in (int, float):
            raise CompileError(
                f"reward[{index}].expr",
                f"the expression reads the declared param {name!r}, whose value is {value!r} "
                f"({type(value).__name__}) — arithmetic needs a number")
        bindings[name] = float(value)
    return bindings


def _lower_row(index, row, spec):
    if row.term is not None:
        return _lower_term_row(index, row, spec)
    if row.expr is not None:
        return Op(kind="add", scope=None, fn_key="expr",
                  params=MappingProxyType({
                      "expr": row.expr,
                      "bindings": MappingProxyType(_expr_bindings(index, row.expr, spec.params))}),
                  stateful=False)
    return Op(kind="add", scope=None, fn_key="custom",
              params=MappingProxyType({"target": row.custom}), stateful=False)


#: Measures a term reads without naming them in a parameter. VelocityPenalty's `body` says WHOSE
#: velocity, not which quantity — the adapter still has to compute both, so the plan has to ask.
_IMPLICIT_MEASURES = {
    "VelocityPenalty": ("object_linear_velocity", "object_angular_velocity"),
}


def _measures_of(op):
    named = op.params.get("measure")
    out = set(_IMPLICIT_MEASURES.get(op.fn_key, ()))
    if isinstance(named, str):
        out.add(named)
    if op.fn_key == "expr":
        # A declared param shadows a measure of the same name (spec.py's rule), and it is already
        # bound into the op — asking the adapter to measure it would read the wrong quantity.
        out |= {n for n in op.params["expr"].names
                if n in MEASURES and n not in op.params["bindings"]}
    return out


# ── the flooding check ──────────────────────────────────────────────────────────────────────────

def _row_maximum(op):
    return _PER_STEP_MAXIMUM[op.fn_key](op.params)


def _check_flooding(ops, *, horizon, terminate_on_success):
    """Refuse when per-step shaping can out-earn completing the task. Warn — never refuse — on the
    horizon-integrated number. Returns the warnings; raises `FloodingError`.

    WHY THE REFUSAL IS PER-STEP AND THE HORIZON IS ONLY A WARNING (user decision, 2026-08-12; the
    original amendment proposed refusing on the integrated ratio):

      Both recorded incidents in this corpus are PER-STEP comparisons. `move_to_target_env.py:205`
      states it outright — "Weight 1.5 keeps the maximum (1.5 at target) below the arrived step at
      tolerance (5.0), so there's still a strong commitment signal" — and lines 199-203 record the
      cost of the sparse alternative that avoids the question entirely: 178M from-scratch steps at 0%
      success. descend's hover attractor is the same shape of reasoning: it peaks at _HOVER rather
      than at contact, and peaking at contact broke 16/16 grasps (2026-06-04).

      The integrated number, by contrast, condemns a working reward. Deployed `descend_to_target`
      earns 1.0 + 1.5 + 2.5 = 5.0/step of shaping over `max_episode_steps=64` — 320 against a success
      value of 12.0 — and is measured at 0.85 success. A gate that refuses the exact lineage the
      phase must prove equivalence against is measurably wrong, not strict. So the ratio is computed,
      printed with BOTH numbers, and left to a human.

    WHAT COUNTS AS SHAPING: rows that ACCUMULATE (`kind == "add"`), excluding the SuccessBonus rows —
    those are the completion side of the comparison. Rows that replace or floor do not accumulate.

    WHAT AN UNBOUNDED ROW DOES: it makes this check INCOMPLETE and say so. Treating it as zero would
    render "not checked" as "checked and clean", which is how `bridle lineage` came to print
    `0 violation(s)` and exit 0 on a machine with no `systemctl`. The bounded part is still reported
    as a lower bound, and a bounded part that already floods still refuses — those rows out-earn
    completion on their own.
    """
    notes = []
    shaping = [(i, op) for i, op in enumerate(ops) if op.kind == "add" and op.fn_key != "SuccessBonus"]
    bonuses = [(i, op) for i, op in enumerate(ops) if op.fn_key == "SuccessBonus"]

    bounded, unbounded = [], []
    for i, op in shaping:
        maximum = _row_maximum(op)
        (unbounded if maximum is UNBOUNDED else bounded).append((i, op, maximum))
    total = sum(m for _, _, m in bounded)

    if unbounded:
        notes.append(
            f"flooding check INCOMPLETE: {len(unbounded)} reward row(s) state no per-step maximum "
            f"this document can bound — "
            f"{'; '.join(f'reward[{i}] {op.fn_key} UNBOUNDED' for i, op, _ in unbounded)}. The "
            f"bounded rows sum to {_num(total)}/step, which is a LOWER bound on the shaping, not a "
            f"verdict: the check could not conclude and this is not a pass.")

    if not bonuses:
        notes.append(
            "flooding check INCOMPLETE: this reward has no SuccessBonus row, so there is no "
            f"completion value for the {_num(total)}/step of shaping to be compared against. That "
            "is not a pass — it is a reward whose ceiling this compiler cannot locate.")
        return notes

    # One document, one completion value. With more than one SuccessBonus row the largest is the
    # ceiling shaping has to stay under; a smaller sibling cannot make a flood safe.
    value = max(_row_maximum(op) for _, op in bonuses)
    if total >= value:
        offenders = "; ".join(f"reward[{i}] {op.fn_key} max {_num(m)}"
                              for i, op, m in bounded if m != 0.0)
        raise FloodingError(
            "reward",
            f"per-step shaping can out-earn completing the task: the additive rows sum to a maximum "
            f"of {_num(total)} per step, which is >= the success value {_num(value)}. Offending "
            f"rows: {offenders}. Every recorded incident here is a per-step comparison — "
            f"move_to_target_env.py:205 chose weight 1.5 so the proximity maximum stays below the "
            f"5.0 arrival bonus, and its lines 199-203 record what the sparse alternative cost: 178M "
            f"from-scratch steps at 0% success. Lower a shaping weight, or raise the success value "
            f"above {_num(total)}")

    notes.extend(_integrated_note(total, value, ops=bonuses, horizon=horizon,
                                  terminate_on_success=terminate_on_success))
    return notes


def _integrated_note(total, value, *, ops, horizon, terminate_on_success):
    """The horizon-integrated ratio: a WARNING with both numbers, or an explicit "could NOT be
    computed" when no horizon was supplied. Never a refusal, and never silence."""
    if horizon is None:
        return [
            "the horizon-integrated shaping check could NOT be computed: no `horizon=` was passed to "
            "compile_spec, and this compiler does not substitute a default. NOT VERIFIED is not the "
            "same as verified — `bridle lineage` once printed `0 violation(s)` and exited 0 on a "
            "machine with no `systemctl`, reporting a clean bill of health for checks it had not "
            f"run. Pass horizon=<max_episode_steps> to compare {_num(total)}/step against the "
            f"success value {_num(value)} over a real episode."]

    latched = any(op.params.get("predicate_ref") == "latched" for _, op in ops)
    if terminate_on_success is True:
        steps, why = 1, "the episode ends on success, so the bonus is paid once"
    elif terminate_on_success is False:
        steps, why = horizon, "the episode continues after success, so the bonus is paid every "\
                              "remaining step"
    elif latched:
        steps, why = horizon, "the bonus is `predicate_ref: latched`, so it pays every step once "\
                              "success has ever been true"
    else:
        steps, why = 1, "termination was not stated (`terminate_on_success=None`), so the bonus is "\
                        "conservatively counted once"
    earned, paid = total * horizon, value * steps
    # Emitted whether or not it exceeds: the decision was that this ratio is COMPUTED AND PRINTED
    # with both numbers, so the reader sees the comparison rather than inferring it from silence.
    head = ("WARNING (not a refusal): shaping out-earns completion over the episode — "
            if earned >= paid else "note: ")
    return [
        f"{head}integrated over the episode, shaping pays {_num(earned)} "
        f"({_num(total)}/step x {horizon} steps) against {_num(paid)} for completing it "
        f"({_num(value)} x {steps} step(s) — {why}). Printed, never refused: the deployed "
        f"descend_to_target lineage integrates to ~27x its success value and is measured at 0.85 "
        f"success, so a gate on this number would refuse a working reward. The refusal is the "
        f"per-step comparison, not this one."]


# ── the plan ────────────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass(frozen=True)
class RewardPlan:
    """A compiled reward: an ordered list of `Op`s plus everything the adapter needs to run them.

    `scale` is `reward_scale.divisor` CARRIED, not applied: `compute_normalized_dense_reward` returns
    `compute_dense_reward(...) / 12.0` (grasp_cube.py:838), so the divisor belongs to the normalized
    path and the fold this plan describes is the UNSCALED one. Any parity comparison against
    `compute_dense_reward` compares the unscaled fold.

    `warnings` is a field, not just an stderr line, because a warning that cannot be read back cannot
    be asserted on, printed by `bridle plan`, or attached to a run. `compile_spec` also emits each one
    through the `warnings` module, so a caller who does nothing still sees them.
    """

    ops: tuple
    scale: float
    measures_needed: frozenset
    state_slots: tuple
    warnings: tuple

    def fingerprint(self) -> str:
        """Stable 12-hex digest of the reward FUNCTION. sha256 over canonical JSON, never `hash()`,
        which is salted per process and would make a stamped checkpoint unverifiable on the next run.

        IN: every op in order, with its kind, scope and bound parameters (including the success
        criterion the replace row reads), plus the scale. Order is part of the payload because the
        fold is ordered — two rows exchanged is a different program even when it computes the same
        number today, and the digest says so rather than quietly claiming the runs are comparable.

        OUT: `why` prose (it is the record of the rationale, not the reward), and the `horizon` /
        `terminate_on_success` arguments (they parameterise the CHECKS, not the arithmetic — the same
        reward compiled for two episode lengths is the same reward). Also out: `env_id`, `scene` and
        `contract` — "is this the same task?" is `Contract.fingerprint()`'s question, already
        answered elsewhere; this one answers "is this the same reward function?".
        """
        payload = {
            "ops": [{"kind": op.kind, "scope": op.scope, "fn": op.fn_key,
                     "params": _jsonable(op.params)} for op in self.ops],
            "scale": self.scale,
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    def describe(self) -> str:
        return f"plan@{self.fingerprint()} ({len(self.ops)} ops, scale {self.scale})"


def _jsonable(value):
    """Canonical JSON form of a parameter. Anything this cannot render raises rather than being
    dropped: a value the digest cannot see is a digest that lies about what it covers."""
    if isinstance(value, Expr):
        return {"expr": value.source}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (MappingProxyType, dict)):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    raise CompileError("reward", f"a reward parameter of type {type(value).__name__} cannot be "
                                 f"fingerprinted, so it cannot be compiled: {value!r}")


def compile_spec(spec: SkillSpec, *, horizon=None, terminate_on_success=None) -> RewardPlan:
    """Lower a validated `SkillSpec` to a `RewardPlan`, or raise `CompileError`.

    `horizon` is the env's `max_episode_steps` and `terminate_on_success` whether success ends the
    episode. Both are OPTIONAL and neither is invented when absent: they parameterise the
    horizon-integrated warning, and with `horizon=None` that warning says it could not be computed.
    They are arguments rather than document fields because `SkillSpec` has no `execution:` block —
    the episode budget belongs to the env registration (`@register_env(..., max_episode_steps=64)`),
    not to the reward.
    """
    if horizon is not None and (isinstance(horizon, bool) or not isinstance(horizon, int)
                                or horizon <= 0):
        raise CompileError("horizon", f"horizon is a positive whole number of steps "
                                      f"(the env's max_episode_steps) or None, got {horizon!r}")
    if terminate_on_success not in (True, False, None):
        raise CompileError("terminate_on_success",
                           f"terminate_on_success is True, False or None (unknown), got "
                           f"{terminate_on_success!r}", legal=["True", "False", "None"])

    ops = tuple(_lower_row(i, row, spec) for i, row in enumerate(spec.reward))

    divisor = spec.reward_scale.get("divisor")
    unnormalized = bool(spec.reward_scale.get("unnormalized"))
    scale = 1.0 if unnormalized else _bind_number("reward_scale.divisor", divisor, spec.params)
    if scale == 0.0:
        raise CompileError("reward_scale.divisor", "the divisor is 0.0 — `reward_ppo = dense / "
                                                   "divisor` cannot divide by it. Write "
                                                   "`unnormalized: true` if the reward is already "
                                                   "at scale")

    notes = _check_flooding(ops, horizon=horizon, terminate_on_success=terminate_on_success)
    for note in notes:
        # Emitted as well as stored: the decision was that the integrated ratio is PRINTED, and a
        # warning only reachable through an attribute is one nobody reads.
        warnings.warn(note, stacklevel=2)

    measures = set()
    for op in ops:
        measures |= _measures_of(op)
    return RewardPlan(
        ops=ops, scale=scale, measures_needed=frozenset(measures),
        state_slots=tuple(op.params["slot"] for op in ops if op.stateful),
        warnings=tuple(notes))


# ── the stdlib evaluator ────────────────────────────────────────────────────────────────────────
# The same fold as Task 5's adapter, over plain Python floats: no torch, no env, so a reward is
# testable on a CPU in milliseconds. Every helper is written batch-safe (see the primitives at the
# top), so the term math here is not a second implementation with its own rounding — the adapter can
# call these with tensors and get the same arithmetic.

_SUCCESS_KEY = {"per_step": "success", "latched": "success_latched"}


def _read(values, key, what):
    if key not in values:
        raise CompileError("evaluate_plan", f"no value supplied for {what} {key!r} — the plan needs "
                                            f"{sorted(values)} plus this one",
                           suggestion=_suggest(key, values))
    return values[key]


def _gate(values, params):
    gate = params.get("gate")
    return 1.0 if gate is None else _read(values, gate, "gate predicate")


def _kernel(kind, delta, k):
    if kind == "one_minus_tanh":
        return 1.0 - _tanh(k * abs(delta))
    if kind == "neg_linear":
        return -abs(delta)
    return _exp(-k * delta * delta)      # gaussian; _check_honoured admits nothing else


def _v_predicate_bonus(p, values):
    return p["weight"] * _read(values, p["predicate"], "predicate")


def _v_distance_pull(p, values):
    measure = _read(values, p["measure"], "measure")
    return p["weight"] * _kernel(p["kernel"], measure - p["setpoint"], p["k"]) * _gate(values, p)


def _v_hinge_penalty(p, values):
    measure = _read(values, p["measure"], "measure")
    # side=above penalizes measure ABOVE the threshold, side=below penalizes it below; one signed
    # delta, clamped at zero, so the row is silent on the permitted side.
    delta = measure - p["threshold"] if p["side"] == "above" else p["threshold"] - measure
    return -p["weight"] * _relu(delta) * _gate(values, p)


def _v_velocity_penalty(p, values):
    return (-p["linear_weight"] * _read(values, "object_linear_velocity", "measure")
            - p["angular_weight"] * _read(values, "object_angular_velocity", "measure"))


def _v_action_penalty(p, values):
    return -p["weight"] * _read(values, p["measure"], "measure")


def _v_ramp(p, values):
    measure = _read(values, p["measure"], "measure")
    span = p["cap"] - p["floor"]
    climbed = _clamp(measure - p["floor"], 0.0, span)
    return p["weight"] * (climbed / span if p["normalize"] else climbed) * _gate(values, p)


def _v_progress_potential(p, values):
    previous = _read(values, p["slot"], "state slot")
    measure = _read(values, p["measure"], "measure")
    # The buffer UPDATE (`prev <- measure`, seeded per env_idx under partial reset) belongs to the
    # adapter that owns the buffers; this evaluator is a pure function of the values it is handed.
    return p["weight"] * (previous - measure) * _gate(values, p)


def _v_success_bonus(p, values):
    return p["value"] * _success(p, values)


def _success(p, values):
    return _read(values, _SUCCESS_KEY[p["predicate_ref"]], "success signal")


def _v_expr(p, values):
    # The document's own params win over a measure of the same name: the author wrote the param, so
    # the author meant the param (the shadowing rule `spec.py` states when it checks expr names).
    return p["expr"].evaluate(dict(values, **p["bindings"]) if p["bindings"] else values)


def _v_custom(p, values):
    raise CompileError("evaluate_plan",
                       f"custom row {p['target']!r} is opaque to the stdlib evaluator — tier 3 is an "
                       f"imported `module:function` and only the adapter can call it. Fold it there, "
                       f"or express the row with `expr:` to make it checkable here")


_VALUE = {
    "PredicateBonus": _v_predicate_bonus,
    "DistancePull": _v_distance_pull,
    "HingePenalty": _v_hinge_penalty,
    "VelocityPenalty": _v_velocity_penalty,
    "ActionPenalty": _v_action_penalty,
    "Ramp": _v_ramp,
    "ProgressPotential": _v_progress_potential,
    "SuccessBonus": _v_success_bonus,
    "expr": _v_expr,
    "custom": _v_custom,
}
assert set(_VALUE) == set(_PER_STEP_MAXIMUM), "every term needs both a value and a bound"

#: `replace`/`floor` need the row split into (condition, level) rather than one number: `where(cond,
#: level, acc)`. Only the two terms carrying a `mode` parameter can be lowered to those kinds, so only
#: those two need the split.
_CONDITION_LEVEL = {
    "SuccessBonus": lambda p, values: (_success(p, values), p["value"]),
    "PredicateBonus": lambda p, values: (_read(values, p["predicate"], "predicate"), p["weight"]),
}


def evaluate_plan(plan: RewardPlan, values: dict):
    """Fold the plan over `values` and return the UNSCALED dense reward.

    `values` maps every name the plan reads to a number: measure names, the predicate/gate STRINGS as
    written in the document (`grasped`, or a whole `and_(...)` call — this evaluator does not
    implement predicate semantics, it asks for their truth), `success`/`success_latched`, and one
    entry per `plan.state_slots`. Unscaled on purpose: `plan.scale` is the normalized path's divisor
    (phase2-decisions §4), and numerical parity is against `compute_dense_reward`.
    """
    acc = 0.0
    for op in plan.ops:
        # A Python `if` on op.kind is safe and a Python `if` on a VALUE is not: the kind is fixed at
        # compile time, identical for all 4096 environments, while `success` differs per environment
        # and must therefore go through the branch-free `_where` below.
        if op.kind == "add":
            acc = acc + _VALUE[op.fn_key](op.params, values)
            continue
        condition, level = _CONDITION_LEVEL[op.fn_key](op.params, values)
        if op.kind == "replace":
            acc = _where(condition, level, acc)
        else:
            acc = _where(condition, _max(acc, level), acc)
    return acc
