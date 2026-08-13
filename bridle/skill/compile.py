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

WHAT IT REFUSES BEYOND `spec.py`, AND WHERE THE TIER BOUNDARY IS. Two tiers refuse a bad parameter
value, and they own different questions:

  SCHEMA TIER (`spec.py` reading `Param.choices` from `vocab.py`) owns "is this a legal value for
  this parameter at all". `kernel: one_minus_tan`, `mode: multiply`, `predicate_ref: whenever`,
  `side`, `norm` — every closed set the vocabulary can enumerate — are refused there, as
  `SpecError`, before this module runs. Those five carried no `choices` until 2026-08-12 and were
  caught here instead; the `_HONOURED` entries that used to do it have been DELETED rather than kept,
  because a refusal branch that can never fire advertises a check that never runs.

  COMPILE TIER (`_HONOURED` / `_UNIMPLEMENTED` below) owns "this value is legal in the vocabulary but
  THIS FOLD does not implement it" — `scope: all`, `body: tcp`, `axes: xy`, `gamma: 0.99`. The
  vocabulary leaves those sets open on purpose (a scope the fold learns tomorrow is not an illegal
  scope today), so nothing upstream can catch them, and accepting one silently folds a reward the
  author did not write: it trains, it logs clean, and it contributes nothing — the exact shape of the
  crush penalty that vanished over an unsigned measure and cost 16/16 grasps.

  The boundary moved once and will move again, so `test_skillcompile.py` asserts each refusal AT the
  tier that raises it, with that tier's exception type, rather than accepting either.

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
from types import MappingProxyType
from warnings import warn      # `warn`, not `warnings`: `RewardPlan.warnings` is a FIELD of this
                               # module's central type, and one `warnings` in one file reading as
                               # both the stdlib module and a plan attribute is a reader-trap.

from bridle.skill.expr import Expr
from bridle.skill.spec import ROW_TERMS, SkillSpec
from bridle.skill.vocab import MEASURES, TERMS, Sign

__all__ = [
    "CompileError", "FloodingError", "Note", "Op", "RewardPlan", "UNBOUNDED",
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

def _numeric(c):
    """A condition as a NUMBER, whatever it arrived as, so `1 - c` below is arithmetic rather than a
    type error.

    THIS LINE IS THE DIFFERENCE BETWEEN THE DOCSTRING AND THE TRUTH. Every helper here builds its
    condition from a comparison (`a > b`), and a torch comparison yields a BOOL tensor:

        >>> 1 - torch.tensor([True, False])
        RuntimeError: Subtraction, the `-` operator, with a bool tensor is not supported.

    so `_relu(torch.tensor([-1., 2.]))` raised, and `HingePenalty` and `Ramp` could not fold a raw
    tensor AT ALL (measured 2026-08-13), while this module's docstring promised one fold for a CPU
    float and a batched CUDA tensor alike. Normalising HERE and not at the call sites is what makes
    that promise true for every caller, including the ones that hand over a bare tensor rather than
    a wrapper whose comparisons already return floats.

    `c * 1` is the one operation that works for all four cases and changes no value in any of them:
    exact for a float (IEEE multiply by 1), identity for an int, `True/False -> 1/0` for a Python
    bool, and bool -> int64 for a tensor. Same idiom `expr._eval_compare` already uses to keep a
    bare comparison numeric (`result = 1; result = result * ok`).
    """
    return c * 1


def _where(c, a, b):
    c = _numeric(c)
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
    if value is None:
        # UNREACHABLE TODAY and defensive on purpose. `spec.py` refuses a null on a required numeric
        # field, and every optional one this is called with carries a numeric default
        # (`RewardScale.divisor` defaults to 12.0, so `reward_scale: {}` still binds 12.0). But the
        # only promise this module makes its callers is that they catch `CompileError`, and
        # `float(None)` is a bare `TypeError` that walks straight past it — one changed default
        # upstream and the contract breaks in the caller, not here.
        raise CompileError(path, "this field needs a number and none was supplied — write the "
                                 "number, or a `params.X` reference to one",
                           legal=[f"params.{n}" for n in spec_params])
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
# THE COMPILE TIER of the two-tier refusal (module docstring). ONLY parameters whose legal set the
# VOCABULARY LEAVES OPEN belong in this table: `spec.py` already refuses anything outside a
# `Param.choices`, so an entry that merely restates one is a branch that can never fire. `kernel`,
# `mode`, `predicate_ref`, `side` and `norm` gained `choices` on 2026-08-12 and were deleted from here
# for exactly that reason — their refusals now read `SpecError`, one tier earlier, same message shape.
# What is left is the residue: values that ARE legal in the vocabulary and that this fold does not
# implement.

#: How each legal `scope` reaches back into the fold. `preceding` means "the accumulator", which is
#: what `evaluate_plan` combines a replace/floor row against; there is no second entry because there
#: is no second behaviour in the fold. ONE table, read both by `_HONOURED` below (what compiles) and
#: by `evaluate_plan` at the bottom of this file (what runs), so a scope cannot be declared legal
#: without stating what it actually reaches. The alternative is the failure this whole tier exists to
#: stop: `scope: all` accepted and quietly folded as `preceding`.
_SCOPE_REACH = {
    "preceding": lambda acc: acc,
}

_HONOURED = {
    ("VelocityPenalty", "body"): ("held",),
    ("SuccessBonus", "scope"): tuple(_SCOPE_REACH),
    ("PredicateBonus", "scope"): tuple(_SCOPE_REACH),
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

    def __hash__(self):
        """Written out because the generated one does not work. `eq=True, frozen=True` makes the
        dataclass synthesise a `__hash__` over the field tuple, which invites `hash(op)`,
        `op in {...}` and `set(plan.ops)` — and every one of them raised `TypeError: unhashable
        type: 'mappingproxy'`, because `params` is a `MappingProxyType`. Hashed over a canonical
        form of exactly the fields `__eq__` compares, so equal Ops hash equal.
        """
        return hash((self.kind, self.scope, self.fn_key, self.stateful, _hashable(self.params)))


def _hashable(value):
    """A hashable stand-in for a parameter value, mirroring what `Op.__eq__` compares: a mapping
    becomes its sorted key/value pairs, a sequence becomes a tuple, and anything else is already
    hashable — an `Expr` by identity, which is also how `__eq__` sees it."""
    if isinstance(value, (MappingProxyType, dict)):
        return tuple(sorted(((str(k), _hashable(v)) for k, v in value.items()),
                            key=lambda item: item[0]))
    if isinstance(value, (list, tuple)):
        return tuple(_hashable(v) for v in value)
    return value


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

def _choices_of(term, name):
    """The vocabulary's legal set for one parameter. READ, never restated: both sets below are the
    compiler's half of a rule `vocab_document()` advertises to the authoring model, and a hand-copied
    set stops matching that promise the day the vocabulary grows — silently, and in the direction
    that ACCEPTS what the document says is refused."""
    param = next(p for p in TERMS[term].params if p.name == name)
    return frozenset(param.choices or ())


# THE 16/16-GRASP RULE HAS TWO AXES — the measure and the kernel — and it has now been narrower than
# advertised on each of them in turn. Both sets below are therefore DERIVED from the vocabulary with
# a written exclusion dict, so the default for anything the vocabulary gains is to be CHECKED, and
# taking something out costs one line with its reason.

#: AXIS 1, THE MEASURE. 0 on any of these IS the surface the object rests on, so a pull peaked at
#: setpoint=0 drags a GRASPED object into it. Not a hypothetical: descend's pre-2026-06-04 hover was
#: `2.5*(1-tanh(6*dz))`, peaking at dz=0, and broke 16/16 grasps while low; the fix was to peak at
#: _HOVER=0.015 instead. (The hand-list this replaced named three of the five signed measures and
#: justified the gap by citing `object_to_goal_z`, which is MAGNITUDE and was never a candidate,
#: while `joint_qpos` — a real and correct exclusion — went unmentioned.)
_ZERO_IS_NOT_A_CONTACT_SURFACE = {
    "gripper_qpos": "0 is a fully OPEN jaw, not a surface: closed is ~-0.73 and the jaw-creep hinge "
                    "sits at -0.6, so a pull peaked at 0 asks the gripper to open — a legal request",
    "joint_qpos": "0 is a joint's zero angle, a pose the arm holds in free space; nothing is "
                  "resting on anything",
}


def _derive_contact_surfaces(measures):
    """Every SIGNED measure the vocabulary has, minus the written exclusions. A FUNCTION so the
    derivation itself is testable: asserting the two sets partition the signed measures is satisfied
    by ANY partition, including the hand-list this replaced, and cannot see whether a measure added
    tomorrow lands in the check."""
    return frozenset(name for name, measure in measures.items()
                     if measure.sign is Sign.SIGNED and name not in _ZERO_IS_NOT_A_CONTACT_SURFACE)


_CONTACT_SURFACE_MEASURES = _derive_contact_surfaces(MEASURES)

# A hand-listed set here is checked by NOTHING ELSE, and the check that catches it is not this one.
# The round-3 review replaced this global with a literal frozenset, left the helper defined, and all
# nine derivation-related predicates in `test_skillcompile.py` stayed green — every one of them
# applies the HELPER to a synthetic vocabulary and compares the result to itself, so none of them
# can see what the compiler actually uses. This assert is the cheap half of the fix and it is
# honestly weaker than it looks: a hand-list that is still value-EQUAL today (which is what a
# copy-paste produces) passes it. It fires only once the two have already diverged — after a rename,
# or after a vocabulary addition someone hand-merged wrong — which is worth one line but is not the
# property.
#
# THE PROPERTY IS THAT A MEASURE ADDED TOMORROW LANDS IN THE CHECK, and only a run over a DIFFERENT
# vocabulary can see it: `test_skillcompile.py` grows `vocab.MEASURES` (and DistancePull's `kernel`
# choices) in a SUBPROCESS before importing this module and asserts both sets below grew with it.
# That is the check that goes red for a value-equal hand-list; keep the two together.
assert _CONTACT_SURFACE_MEASURES == _derive_contact_surfaces(MEASURES), (
    "the contact-surface set the compiler uses has diverged from `_derive_contact_surfaces("
    "MEASURES)` — a hand-listed set here stops growing the day the vocabulary does")

assert set(_ZERO_IS_NOT_A_CONTACT_SURFACE) <= set(MEASURES), (
    "an exclusion naming a measure the vocabulary does not have is a line that stopped meaning "
    "anything at a rename and now silently excludes nothing")
assert all(MEASURES[n].sign is Sign.SIGNED for n in _ZERO_IS_NOT_A_CONTACT_SURFACE), (
    "an exclusion only means something for a SIGNED measure — an unsigned one can never enter this "
    "set — so a MAGNITUDE name here is a stale line pretending to carry a decision")
assert {"height_above_seat_live", "height_above_seat_static_goal",
        "height_above_resting"} <= _CONTACT_SURFACE_MEASURES, (
    "the three measures whose 0 is a seat or resting surface must stay checked: this is the "
    "16/16-grasp rule, and `height_above_seat` -> `height_above_seat_live` already renamed once "
    "(phase2-decisions §2), so a rename must fail loudly here rather than delete the check")

#: AXIS 2, THE KERNEL. Kernels whose maximum is NOT at the setpoint — EMPTY, and that is the finding:
#: all three the vocabulary offers peak there. `one_minus_tanh` is `1 - tanh(k|d|)` and `gaussian` is
#: `exp(-k d^2)`, both maximal at 1.0 when d=0; `neg_linear` is `-|d|`, maximal at 0.0 when d=0 — a
#: shallower peak, not the absence of one, and `vocab.py`'s `setpoint` param calls itself "the
#: kernel's peak" for all three with a default of 0.0. Naming only the first two left
#: `DistancePull{measure: height_above_seat_live, kernel: neg_linear}` compiling clean ON DEFAULT
#: PARAMETERS while pulling a grasped object down onto the seat — the same 16/16-grasp failure the
#: refusal above exists to stop, reached through the kernel it did not cover. The row's WEIGHT sign
#: is deliberately not part of the rule (a negative weight inverts any of the three into a repeller);
#: that has always been true of the two kernels already here, and one uniform rule is the one the
#: document can state.
_KERNEL_PEAK_IS_NOT_AT_SETPOINT = {}


def _derive_peaked_kernels(kernels):
    """Same shape, same reason, as `_derive_contact_surfaces` — see it."""
    return frozenset(k for k in kernels if k not in _KERNEL_PEAK_IS_NOT_AT_SETPOINT)


_PEAKED_KERNELS = _derive_peaked_kernels(_choices_of("DistancePull", "kernel"))

#: The kernel axis of the same divergence guard — read the comment above the measure axis' assert
#: for what it does and does not catch. Weaker still here, because `_KERNEL_PEAK_IS_NOT_AT_SETPOINT`
#: is empty, so the derivation IS the vocabulary's `choices` and a hand-list of the three is equal
#: to it in every respect except that it stops growing. The subprocess probe named above is what
#: covers that on this axis too.
assert _PEAKED_KERNELS == _derive_peaked_kernels(_choices_of("DistancePull", "kernel")), (
    "the peaked-kernel set the compiler uses has diverged from `_derive_peaked_kernels` over the "
    "vocabulary's own `choices` — a hand-list here silently stops covering a kernel it gains")

assert set(_KERNEL_PEAK_IS_NOT_AT_SETPOINT) <= _choices_of("DistancePull", "kernel"), (
    "an exclusion naming a kernel the vocabulary does not offer excludes nothing and hides that it "
    "excludes nothing")
assert "one_minus_tanh" in _PEAKED_KERNELS, (
    "the kernel of the recorded incident must stay checked: descend's `2.5*(1-tanh(6*dz))` peaked "
    "at dz=0 and broke 16/16 grasps (2026-06-04)")


def _check_row_semantics(path, term, values):
    if term == "DistancePull":
        if (values["kernel"] in _PEAKED_KERNELS
                and values["measure"] in _CONTACT_SURFACE_MEASURES
                and values["setpoint"] == 0.0):
            # THE RULE IGNORES THE WEIGHT'S SIGN; THE MESSAGE MUST NOT. Refusing a negative-weight
            # row here is deliberate (see `_KERNEL_PEAK_IS_NOT_AT_SETPOINT`: one uniform rule is the
            # one the document can state), but telling that author the row "is maximised at the
            # setpoint" is false — `_max_distance_pull` in this same file says a negative weight
            # inverts the kernel — and the error messages ARE the API for a model that cannot read
            # this source. So the sentence describes the row that was actually written.
            extremum = ("maximised at the setpoint"
                        if values["weight"] >= 0.0 else
                        f"MINIMISED at the setpoint, because weight={_num(values['weight'])} "
                        f"inverts the kernel into a repeller centred on that same surface")
            raise CompileError(
                f"{path}.setpoint",
                f"this DistancePull is extremal AT the contact surface: setpoint=0.0 over "
                f"{values['measure']!r} with kernel={values['kernel']!r}, which is {extremum}, "
                f"and 0 there means the object is resting on the seat. Concentrating a row on "
                f"contact is a recorded failure, not a free parameter — descend's "
                f"`2.5*(1-tanh(6*dz))` pulled the held cube INTO the platform and broke 16/16 grasps "
                f"(2026-06-04); the fix was a setpoint of 0.015, a hover ABOVE the seat. The rule "
                f"does not read the weight's sign, because either sign makes contact the one height "
                f"this row is about. Set a positive setpoint, or use HingePenalty if you meant to "
                f"bound the height")
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


class Note(str):
    """One compile-time note: its text, and the LEVEL saying whether a caller can act on it.

    A `str` SUBCLASS rather than a `(level, text)` pair because `plan.warnings` is the authoritative
    channel and already has consumers that treat an entry as text — the plan report runs
    `textwrap.fill(note, ...)` over each one (`report.format_warnings`) and the tests join them.
    A pair would have broken both for a field this module does not own the printing of. Everything a
    string does, a Note does — including `copy`, `deepcopy` and `pickle`, which needed the
    `__getnewargs__` below and raised `TypeError` for as long as this class existed without it, so
    the claim was false when it was first written (round-3 review).

    `.level` is the part that used to be dropped at the `RewardPlan`
    boundary, where "flooding check INCOMPLETE ... this is not a pass" and a clean integrated ratio
    arrived indistinguishable and the distinction survived only in whether `warn()` had already
    fired — i.e. in a channel the stdlib dedupes per (message, location) and `bridle skill` silences.

    `Note.ACT` also goes out through the `warnings` module: the caller has something to DO (pass a
    horizon, bound a row, lower a weight). `Note.FYI` is recorded only. The split exists because
    emitting on EVERY successful compile made `python -W error` unable to compile any document at
    all, which turns a channel meant to carry signal into one a careful caller must switch off
    wholesale.
    """

    FYI = "note"
    ACT = "act"

    def __new__(cls, level, text):
        note = super().__new__(cls, text)
        note.level = level
        return note

    def __getnewargs__(self):
        """`copy`, `deepcopy` and `pickle` all rebuild a `str` subclass as
        `cls.__new__(cls, *obj.__getnewargs__())`, and `str`'s own returns the 1-TUPLE `(text,)` —
        which calls `Note.__new__(cls, text)` and raises
        `TypeError: Note.__new__() missing 1 required positional argument: 'text'`. All three were
        broken; nothing in `bridle/` copies or pickles a note today, so it was latent rather than
        live, but a `RewardPlan` crossing a process boundary (a training launcher, a cached compile)
        is the obvious way it stops being latent. Returning `(level, text)` is the whole fix, and it
        is what makes the docstring's "everything a string does, a Note does" true."""
        return (self.level, str(self))

    def __repr__(self):
        return f"Note({self.level!r}, {str.__repr__(self)})"


_FYI, _ACT = Note.FYI, Note.ACT


def _check_flooding(ops, *, horizon, terminate_on_success):
    """Refuse when per-step shaping can out-earn completing the task. Warn — never refuse — on the
    horizon-integrated number. Returns a list of `Note`s; raises `FloodingError`.

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

    WHY THE REFUSAL IS GATED ON THE SUCCESS ROW'S `mode` (C1, whole-branch review 2026-08-13; the
    check previously compared shaping against the success value in every mode):

      The un-gated comparison refused this compiler's OWN chassis default. `hold_and_ramp` ships
      `PredicateBonus{weight=1.0, predicate=grasped}` + `Ramp{weight=8.0, normalize=True}` against
      `SuccessBonus{value=9.0, mode=add}` — 1.0 + 8.0 = 9.0, landing exactly on `total >= value` —
      and that is deployed `lift` verbatim (`primitives/lift/lift_env.py:84-92`):
          reward = 1.0 * is_grasped
          reward = reward + 8.0 * is_grasped * torch.clamp(cube_z / LIFT_TARGET_Z, 0.0, 1.0)
          reward = reward + 9.0 * info["success"].float()
      The chassis covers lift, sphere_lift and compact_grasp: three DEPLOYED primitives measured at
      0.9-1.0 success. The section header a 27-30B author reads above it says "start here".

      The comparison was wrong, not merely tight. In `mode: add` the success step pays shaping PLUS
      the bonus, so completing always earns strictly more than not completing from the same state —
      there is no choice to get wrong, and "shaping total vs success value" compares two quantities
      the agent never has to trade against each other. The flooding pathology — preferring to farm
      shaping rather than finish — needs the bonus to REPLACE (or floor) the shaping, which is what
      forces the trade. So:
        * `mode: replace` / `mode: floor` -> keep `total >= value`, and say in the refusal that the
          row overwrites rather than adds. `>=` not `>`: at equality completion pays exactly what
          maximal farming pays, the completing action carries zero advantage, and under `floor` the
          row is vacuous outright (`max(shaping, value) == shaping`).
        * `mode: add` -> the right quantity is the bonus's SIGN: refuse only `value <= 0`, where
          completion adds nothing (or is penalised). The shaping-vs-value figure is still reported,
          as a note, because going silent on the configuration this check got wrong would read as
          "checked and clean".
      The horizon-integrated number remains a warning in every mode; for `add` it is the number
      actually worth reading, since terminating on success is what forgoes future shaping.

      NOT DONE, deliberately: nudging the chassis default off 9.0/8.0/1.0 to clear the boundary.
      Those values ARE the deployed reward; changing them to satisfy a check is how the record of
      what was trained stops matching what was trained.

    WHAT COUNTS AS SHAPING: rows that ACCUMULATE (`kind == "add"`), excluding the SuccessBonus rows —
    those are the completion side of the comparison. Rows that replace or floor do not accumulate,
    and are therefore OUTSIDE this comparison entirely — a gap the notes state out loud whenever such
    a row is present, because a `PredicateBonus{mode: floor, weight: 100.0, scope: preceding}` over an
    easy predicate measures 99.998/step against descend's success value of 12.0 and does not trip
    anything here.

    WHAT AN UNBOUNDED ROW DOES: it makes this check INCOMPLETE and say so. Treating it as zero would
    render "not checked" as "checked and clean", which is how `bridle lineage` came to print
    `0 violation(s)` and exit 0 on a machine with no `systemctl`. The bounded part is still reported
    as a lower bound, and a bounded part that already floods still refuses — those rows out-earn
    completion on their own.
    """
    notes = []
    shaping = [(i, op) for i, op in enumerate(ops) if op.kind == "add" and op.fn_key != "SuccessBonus"]
    bonuses = [(i, op) for i, op in enumerate(ops) if op.fn_key == "SuccessBonus"]
    overriding = [(i, op) for i, op in enumerate(ops)
                  if op.kind != "add" and op.fn_key != "SuccessBonus"]

    bounded, unbounded = [], []
    for i, op in shaping:
        maximum = _row_maximum(op)
        (unbounded if maximum is UNBOUNDED else bounded).append((i, op, maximum))
    total = sum(m for _, _, m in bounded)

    if unbounded:
        notes.append(Note(_ACT,
            f"flooding check INCOMPLETE: {len(unbounded)} reward row(s) state no per-step maximum "
            f"this document can bound — "
            f"{'; '.join(f'reward[{i}] {op.fn_key} UNBOUNDED' for i, op, _ in unbounded)}. The "
            f"bounded rows sum to {_num(total)}/step, which is a LOWER bound on the shaping, not a "
            f"verdict: the check could not conclude and this is not a pass."))

    if overriding:
        # Stated in the note, not only in a code comment: a reader of the output has no way to know
        # which rows the sum above left out, and the omission is large enough to invert the verdict.
        notes.append(Note(_ACT,
            f"KNOWN GAP, this check does not cover "
            f"{'; '.join(f'reward[{i}] {op.fn_key} mode={op.kind}' for i, op in overriding)}: a "
            f"`replace` or `floor` row does not ACCUMULATE, so it is outside the per-step sum by "
            f"construction and no weight on it can trip the refusal. Measured: a "
            f"`PredicateBonus{{mode: floor, weight: 100.0, scope: preceding}}` over `grasped` folds "
            f"to 99.998/step against descend's success value of 12.0 and compiles clean. Compare "
            f"those rows' levels against the success value by hand."))

    if not bonuses:
        notes.append(Note(_ACT,
            "flooding check INCOMPLETE: this reward has no SuccessBonus row, so there is no "
            f"completion value for the {_num(total)}/step of shaping to be compared against. That "
            "is not a pass — it is a reward whose ceiling this compiler cannot locate."))
        return notes

    # One document, one completion value. With more than one SuccessBonus row the largest is the
    # ceiling shaping has to stay under; a smaller sibling cannot make a flood safe. This is the
    # figure the horizon-integrated note below compares against, whatever the modes are.
    value = max(_row_maximum(op) for _, op in bonuses)
    offenders = "; ".join(f"reward[{i}] {op.fn_key} max {_num(m)}"
                          for i, op, m in bounded if m != 0.0)

    # WHICH COMPARISON IS RIGHT DEPENDS ON THE SUCCESS ROW'S `mode` — see this function's docstring,
    # "WHY THE REFUSAL IS GATED ON THE SUCCESS ROW'S `mode`". `op.kind` IS that mode.
    overriding_bonuses = [(i, op, _row_maximum(op)) for i, op in bonuses if op.kind != "add"]

    # `add`: completion pays shaping PLUS the bonus, so it out-earns not-completing at the same
    # state iff the bonus is strictly positive. That, and not "shaping vs value", is the quantity
    # this mode has to satisfy — and it is authorable, `value` carries no positivity bound in
    # `vocab.TERMS["SuccessBonus"]`, so this branch is reachable rather than decorative.
    #
    # THE AUTHORED VALUE, NOT `_row_maximum`: that helper clamps with `max(value, 0.0)` because it
    # feeds the ceiling a shaping total must stay under, so it reports 0.0 for `value: -1.0` and a
    # refusal quoting it would tell an author a number their document does not contain.
    dead = [(i, op, float(op.params["value"])) for i, op in bonuses
            if op.kind == "add" and float(op.params["value"]) <= 0.0]
    if dead:
        raise FloodingError(
            "reward",
            f"completing the task pays nothing: "
            f"{'; '.join(f'reward[{i}] SuccessBonus value {_num(m)} mode=add' for i, _, m in dead)}. "
            f"In `mode: add` the bonus is ADDED to the {_num(total)}/step of shaping already "
            f"accumulated, so completion out-earns not-completing only while the value is strictly "
            f"positive; at {_num(dead[0][2])} the success row is a no-op (or worse, a penalty for "
            f"finishing) and the whole reward is its shaping. Give the SuccessBonus a positive "
            f"value, or state `mode: replace` if the bonus is meant to override the shaping")

    if overriding_bonuses:
        # `replace`/`floor`: the bonus REPLACES (or floors) the shaping, so the agent is made to
        # CHOOSE between farming and finishing, and shaping that reaches the bonus wins that choice.
        # `>=` and not `>`: at exact equality completing pays exactly what maximal farming pays, so
        # the completing action carries zero advantage and PPO has no gradient toward it; under
        # `floor` the row is literally vacuous there (`max(shaping, value) == shaping`). Relaxing to
        # `>` would also be relaxing a check so a document passes, which this phase forbids.
        ceiling = max(m for _, _, m in overriding_bonuses)
        modes = "; ".join(f"reward[{i}] SuccessBonus {_num(m)} mode={op.kind}"
                          for i, op, m in overriding_bonuses)
        if total >= ceiling:
            raise FloodingError(
                "reward",
                f"per-step shaping can out-earn completing the task: the additive rows sum to a "
                f"maximum of {_num(total)} per step, which is >= the success value "
                f"{_num(ceiling)}. Offending rows: {offenders}. The success row {modes} does not "
                f"ADD to that shaping, it overwrites it, so the agent has to choose between the two "
                f"and this weighting makes farming the better choice. Every recorded incident here "
                f"is a per-step comparison — move_to_target_env.py:205 chose weight 1.5 so the "
                f"proximity maximum stays below the 5.0 arrival bonus, and its lines 199-203 record "
                f"what the sparse alternative cost: 178M from-scratch steps at 0% success. Lower a "
                f"shaping weight, or raise the success value above {_num(total)}")
    elif total >= value:
        # The case that used to refuse. It is deployed `lift`, so it is a note, and the note states
        # the comparison it DID make rather than going quiet — silence here would read as "checked
        # and clean" on the exact configuration this check was wrong about.
        notes.append(Note(_FYI,
            f"the per-step shaping maximum {_num(total)} reaches the success value {_num(value)}, "
            f"and this is NOT refused because the success row is `mode: add`: at success the agent "
            f"collects the shaping AND the bonus ({_num(total)} + {_num(value)}), so completing "
            f"always pays strictly more than not completing and there is no choice to get wrong. "
            f"The refusal is reserved for `mode: replace`/`floor`, where the bonus overwrites the "
            f"shaping. Contributing rows: {offenders}. Deployed `lift` is exactly this shape — "
            f"1.0 (grasped) + 8.0 (normalized ramp) = 9.0/step against a 9.0 `mode: add` success "
            f"value (lift_env.py:84-92), measured at 0.9-1.0 success. The number worth reading here "
            f"is the horizon-integrated one below."))

    notes.extend(_integrated_note(total, value, ops=bonuses, horizon=horizon,
                                  terminate_on_success=terminate_on_success,
                                  partial=bool(unbounded)))
    return notes


def _integrated_note(total, value, *, ops, horizon, terminate_on_success, partial):
    """The horizon-integrated ratio: a WARNING with both numbers, or an explicit "could NOT be
    computed" when no horizon was supplied. Never a refusal, and never silence.

    `partial` says unbounded rows were partitioned out upstream, so `total` is a lower bound and this
    line has to say so. Without it the pair of notes contradicts itself: `_check_flooding` prints
    "INCOMPLETE ... this is not a pass" and then this line prints a clean-looking figure computed from
    the bounded subset, and the second one is the one that reads like a verdict.
    """
    at_least = "at least " if partial else ""
    caveat = (" That shaping figure is a LOWER bound and not a verdict: the unbounded rows named "
              "above are not in it, so the real total is higher by an amount this document does not "
              "state." if partial else "")
    if horizon is None:
        return [Note(_ACT,
            "the horizon-integrated shaping check could NOT be computed: no `horizon=` was passed to "
            "compile_spec, and this compiler does not substitute a default. NOT VERIFIED is not the "
            "same as verified — `bridle lineage` once printed `0 violation(s)` and exited 0 on a "
            "machine with no `systemctl`, reporting a clean bill of health for checks it had not "
            f"run. Pass horizon=<max_episode_steps> to compare {at_least}{_num(total)}/step against "
            f"the success value {_num(value)} over a real episode.")]

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
    # Computed whether or not it exceeds — the decision was that this ratio is COMPUTED AND PRINTED
    # with both numbers, so the reader sees the comparison rather than inferring it from silence. It
    # reaches the `warnings` module only when it exceeds or when it is a lower bound; a clean ratio is
    # kept in `plan.warnings`, which is the channel `bridle plan` prints and the one nothing dedupes.
    exceeds = earned >= paid
    head = ("WARNING (not a refusal): shaping out-earns completion over the episode — "
            if exceeds else "note: ")
    return [Note(_ACT if (exceeds or partial) else _FYI,
        f"{head}integrated over the episode, shaping pays {at_least}{_num(earned)} "
        f"({_num(total)}/step x {horizon} steps) against {_num(paid)} for completing it "
        f"({_num(value)} x {steps} step(s) — {why}).{caveat} Printed, never refused: the deployed "
        f"descend_to_target lineage integrates to ~27x its success value and is measured at 0.85 "
        f"success, so a gate on this number would refuse a working reward. The refusal is the "
        f"per-step comparison, not this one.")]


# ── the plan ────────────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass(frozen=True)
class RewardPlan:
    """A compiled reward: an ordered list of `Op`s plus everything the adapter needs to run them.

    `scale` is `reward_scale.divisor` CARRIED, not applied: `compute_normalized_dense_reward` returns
    `compute_dense_reward(...) / 12.0` (grasp_cube.py:838), so the divisor belongs to the normalized
    path and the fold this plan describes is the UNSCALED one. Any parity comparison against
    `compute_dense_reward` compares the unscaled fold.

    `warnings` is the AUTHORITATIVE channel and always carries every note, including the clean ones:
    a warning that cannot be read back cannot be asserted on, printed by `bridle plan`, or attached to
    a run. `compile_spec` re-emits through the stdlib `warnings` module only the notes a caller can
    ACT on — because emitting on every successful compile meant `python -W error` could not compile
    any document at all, and because that module's default filter is once per (message, location), so
    a second identical compile in one process prints nothing. This field is the record that neither
    limitation touches. (Field named `warnings` while the module is imported as `warn` — see the
    import.)

    Its entries are `Note`s: strings, so every existing consumer keeps working unchanged, that also
    carry `.level` (`Note.ACT` / `Note.FYI`). Without it a reader of the authoritative channel could
    not tell "flooding check INCOMPLETE ... this is not a pass" from a clean integrated ratio, and
    the distinction survived only in whether `warn()` had already fired — which is precisely the
    channel this field exists because one cannot rely on.
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
        # Stored unconditionally, emitted selectively. A clean integrated ratio is a fact about the
        # document, not a request: raising it made every successful compile a `UserWarning` and made
        # `-W error` refuse legal specs, so a careful caller's only option was to silence the channel
        # — and then the notes that DO need action go with it. The level travels ON the note into
        # `plan.warnings`, so this decision is legible to a consumer of that field and not only to
        # whoever was listening to the `warnings` module at the time.
        if note.level == _ACT:
            warn(note, stacklevel=2)

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
        # The list is what WAS supplied, and says so. It used to be introduced as "the plan needs
        # ... plus this one", which named the caller's own dict as the requirement — backwards, in a
        # module whose premise is that the error message is the API.
        raise CompileError("evaluate_plan",
                           f"no value supplied for {what} {key!r}; the plan needs it in `values`, "
                           f"and what WAS supplied is {sorted(values)}",
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
assert set(_CONDITION_LEVEL) == {t for t in ROW_TERMS
                                 if any(p.name == "mode" for p in TERMS[t].params)}, (
    "a term that declares `mode` can be lowered to a replace/floor Op, and the fold looks its "
    "(condition, level) split up here — a term gaining a `mode` in the vocabulary with no entry "
    "would KeyError mid-fold, inside the adapter on a GPU, instead of at import on a laptop. Same "
    "guard as `_VALUE` and `_PER_STEP_MAXIMUM` carry")


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
        # What the row's mode operates OVER comes from `_SCOPE_REACH`, the same table `_HONOURED`
        # builds its legal scope set from — so a scope this fold cannot implement is refused at
        # compile time instead of arriving here and being folded as `preceding` anyway.
        reached = _SCOPE_REACH[op.scope](acc)
        if op.kind == "replace":
            acc = _where(condition, level, reached)
        else:
            acc = _where(condition, _max(reached, level), acc)
    return acc
