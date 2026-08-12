"""Unit test for bridle.skill.compile — the ordered fold, the constraint checks, the fingerprint.

WHY THIS EXISTS, property by property, with the measurement each one was paid for:

  ORDERED FOLD   descend's row 8 is `SuccessBonus{mode: replace, scope: preceding}` and row 9 is
                 `ActionPenalty`, so on the success step the deployed env pays `12.0 - 0.001*||a||`
                 (descend_env.py:210-212) — not 12.0, and not the sum of everything. A compiler that
                 sums rows instead of folding them produces a DIFFERENT reward from the one measured
                 at 0.85, and it does so silently: same rows, same weights, same log lines.

  FLOODING       refused PER-STEP, warned on the horizon (phase2-decisions §1). Both recorded
                 incidents are per-step comparisons: move_to_target_env.py:205 chose weight 1.5 so
                 the proximity maximum stays under the 5.0 arrival bonus, and its lines 199-203
                 record what the fully-sparse alternative cost — 178M from-scratch steps at 0%
                 success. Refusing on the INTEGRATED number would refuse deployed descend
                 (5.0/step x 64 steps = 320 against a success value of 12.0, measured at 0.85).

  UNBOUNDED      a row whose per-step maximum the document cannot state (ProgressPotential, a tier-2
                 `expr:`, a tier-3 `custom:`) must make the check SAY it could not conclude. Treating
                 it as zero renders "not checked" as "checked and clean" — the failure mode
                 `bridle lineage` shipped when it printed `0 violation(s)` and exited 0 on a machine
                 with no `systemctl`. The same rule is why `horizon=None` says NOT COMPUTED rather
                 than substituting a default.

  TWO TIERS      a bad parameter value is refused either by the SCHEMA (`spec.py` reading a
                 `Param.choices` from `vocab.py`, raising `SpecError`) or by the COMPILER
                 (`_HONOURED`/`_UNIMPLEMENTED`, raising `CompileError`), and which one is not
                 cosmetic — it is the difference between "no document may say this" and "this fold
                 does not implement it yet". Every refusal below is asserted AT the tier that raises
                 it, with that tier's exception type. When `choices` arrived for kernel, mode,
                 predicate_ref, side and norm (2026-08-12) three checks here went red; the fix was to
                 move those assertions to the schema tier and delete the compile-tier entries that
                 had become unreachable, NOT to widen the accepted exception type — a check that
                 accepts either can no longer tell you which tier refuses anything, and a compile-tier
                 branch that can never fire advertises a check that never runs.

  BATCH-SAFE     the fold must evaluate over 4096 environments at once, so `_where`/`_max`/`_relu`/
                 `_clamp` are written branch-free and `tanh`/`exp` dispatch to the value's own method.
                 A suite of scalar floats cannot see any of that: a Python `if c: a else: b` returns
                 the right answer for every scalar and takes ONE branch for a whole batch. `Vec`
                 below is a 3-element duck-typed stand-in for the tensor Task 5's adapter folds, and
                 it raises on `bool()` so a branch on a batched value is a failure, not a coin flip.

  FINGERPRINT    sha256 over canonical JSON, never `hash()`. `hash()` is salted per process, so a
                 checkpoint stamped with it is unverifiable on the next run — which is exactly what
                 the stamp exists to prevent. An in-process comparison cannot tell the two apart, so
                 the check below runs the computation in SUBPROCESSES with different PYTHONHASHSEEDs,
                 and first proves the seeds really did differ (`hash()` of the same string differs)
                 so a green result cannot come from a botched probe.

Run: python -m pytest bridle/tests/test_skillcompile.py
     PYTHONPATH=. python bridle/tests/test_skillcompile.py

The descend fixture is imported from test_skillspec rather than re-derived: it is the deployed
document, every weight is a trained number, and two copies would drift.
"""
import copy
import json
import math
import operator
import os
import pickle
import re
import subprocess
import sys
import warnings
from pathlib import Path

from bridle.skill.compile import (
    CompileError, FloodingError, Note, Op, RewardPlan, compile_spec, evaluate_plan,
)
from bridle.skill.compile import (
    _bind_number, _clamp, _CONTACT_SURFACE_MEASURES, _derive_contact_surfaces,
    _derive_peaked_kernels, _HONOURED, _KERNEL_PEAK_IS_NOT_AT_SETPOINT, _max, _min, _PEAKED_KERNELS,
    _relu, _SCOPE_REACH, _where, _ZERO_IS_NOT_A_CONTACT_SURFACE,
)
from bridle.skill.spec import SpecError, parse_spec
from bridle.skill.vocab import MEASURES, TERMS, Frame, Measure, Sign, vocab_document
from bridle.tests.test_skillspec import descend_doc, doc_with, row_edited

FAILS = []

REPO = Path(__file__).resolve().parents[2]

#: SO100DescendToTarget-v1's registered `max_episode_steps` (descend_env.py:96). Not a guess and not
#: a default the compiler may substitute — the caller states it.
HORIZON = 64

#: Deliberately not 1.0: `12.0 - 0.001*||a||` has to be distinguishable from BOTH 12.0 and
#: 12.0 - 0.001, or the test passes for an implementation that hardcodes either.
ACTION_NORM = 2.0


def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        FAILS.append(name)


def close(a, b, tol=1e-12):
    return isinstance(a, (int, float)) and abs(a - b) < tol


class Vec:
    """A 3-element batch, duck-typed: every operator the fold uses, applied element-wise.

    Stands in for the torch tensor Task 5's adapter folds over up to 4096 environments at once, on
    stdlib only. It exists because vectorisation is this phase's binding constraint and a suite of
    scalar floats cannot test it — `_where`'s branch-free `c*a + (1-c)*b` and a plain `if c: a else:
    b` agree on every scalar and disagree on every batch, where the `if` takes ONE branch for all
    4096 environments. `__bool__` therefore RAISES: a fold that branches on a batched value fails
    here instead of quietly returning one environment's answer for all of them.
    """

    def __init__(self, xs):
        self.xs = tuple(float(x) for x in xs)

    def _zip(self, other, f, flip=False):
        o = other.xs if isinstance(other, Vec) else (float(other),) * len(self.xs)
        # `type(self)`, not `Vec`: a BoolVec that stopped being a BoolVec after one arithmetic op
        # would exercise the bool-condition path once and then quietly fall back to plain floats,
        # which is how `_clamp`'s SECOND comparison would stop being tested.
        return type(self)(f(b, a) if flip else f(a, b) for a, b in zip(self.xs, o))

    def __add__(self, o): return self._zip(o, operator.add)
    def __radd__(self, o): return self._zip(o, operator.add, flip=True)
    def __sub__(self, o): return self._zip(o, operator.sub)
    def __rsub__(self, o): return self._zip(o, operator.sub, flip=True)
    def __mul__(self, o): return self._zip(o, operator.mul)
    def __rmul__(self, o): return self._zip(o, operator.mul, flip=True)
    def __truediv__(self, o): return self._zip(o, operator.truediv)
    def __gt__(self, o): return self._zip(o, lambda a, b: float(a > b))
    def __lt__(self, o): return self._zip(o, lambda a, b: float(a < b))
    def __abs__(self): return type(self)(abs(x) for x in self.xs)
    def __neg__(self): return type(self)(-x for x in self.xs)
    def tanh(self): return type(self)(math.tanh(x) for x in self.xs)
    def exp(self): return type(self)(math.exp(x) for x in self.xs)
    def __repr__(self): return f"{type(self).__name__}{self.xs}"

    def __bool__(self):
        raise AssertionError("the fold called bool() on a batch — a Python `if` on a batched "
                             "condition takes one branch for all 4096 environments, which is a "
                             "different reward, not a slower one")


class BoolMask:
    """What a comparison on a real tensor returns: a BOOLEAN batch, which REFUSES subtraction.

    `Vec` above cannot see the defect this exists for, because `Vec.__gt__` hands back floats — it
    is a batch whose comparisons are already numeric, which is the one shape of batch the
    branch-free helpers happened to work for. torch is not that shape:

        >>> 1 - torch.tensor([True, False])
        RuntimeError: Subtraction, the `-` operator, with a bool tensor is not supported.

    so `compile._relu(torch.tensor([-1., 2.]))` raised, `HingePenalty` and `Ramp` could not fold a
    raw tensor at all, and both modules' docstrings said otherwise. `__sub__`/`__rsub__` therefore
    raise here the way torch does; `* 1` is the conversion the fold has to perform before treating a
    condition as a number, and it is the only route out of this type.
    """

    def __init__(self, bs):
        self.bs = tuple(bool(b) for b in bs)

    def _as_numbers(self):
        # A BoolVec, not a plain Vec: the numeric form of a mask feeds straight back into `_where`'s
        # arithmetic, and a plain Vec there would make every comparison DOWNSTREAM of the first one
        # numeric again — `_clamp`'s second helper would then never see a boolean condition.
        return BoolVec(1.0 if b else 0.0 for b in self.bs)

    def __mul__(self, o): return self._as_numbers() * o
    def __rmul__(self, o): return o * self._as_numbers()
    def __sub__(self, o): raise AssertionError(self._REFUSAL)
    def __rsub__(self, o): raise AssertionError(self._REFUSAL)
    def __repr__(self): return f"BoolMask{self.bs}"

    _REFUSAL = ("arithmetic on a BOOLEAN batch — torch raises `Subtraction, the `-` operator, with "
                "a bool tensor is not supported` here, so a helper that computes `1 - c` on a raw "
                "comparison cannot fold a tensor at all")

    def __bool__(self):
        raise AssertionError("bool() on a batched mask — see Vec.__bool__")


class BoolVec(Vec):
    """A `Vec` whose COMPARISONS yield a `BoolMask`, i.e. a torch-shaped batch rather than a
    conveniently pre-numeric one. Everything else is inherited."""

    def __gt__(self, o): return BoolMask(a > b for a, b in zip(self.xs, self._other(o)))
    def __lt__(self, o): return BoolMask(a < b for a, b in zip(self.xs, self._other(o)))

    def _other(self, o):
        return o.xs if isinstance(o, Vec) else (float(o),) * len(self.xs)


def close_all(got, expected, tol=1e-12):
    """Element-wise `close` over a Vec against a list of scalars."""
    return (isinstance(got, Vec) and len(got.xs) == len(expected)
            and all(close(a, b, tol) for a, b in zip(got.xs, expected)))


# ── fixtures ────────────────────────────────────────────────────────────────────────────────────

def plan_of(doc, **kw):
    """Compile a document. Warnings are silenced HERE, not globally: `compile_spec` emits them
    through the `warnings` module on purpose (a warning nobody sees is not a warning), and the one
    check that cares asserts on that emission explicitly."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return compile_spec(parse_spec(doc), **kw)


def values(**over):
    """One step of the descend env, as plain Python floats: grasped, held 3mm below the hover
    setpoint, drifting 2cm off centre, gripper firmly closed, nothing crushed."""
    v = {
        "grasped": 1.0,
        "not_grasped": 0.0,
        "object_to_goal_xy": 0.02,
        "height_above_seat_live": 0.012,
        "gripper_qpos": -0.70,
        "object_linear_velocity": 0.05,
        "object_angular_velocity": 0.40,
        "action_norm": ACTION_NORM,
        "success": 0.0,
        "success_latched": 0.0,
    }
    v.update(over)
    return v


def descend_reward_by_hand(v, success):
    """descend_env.py:191-212, transcribed by hand.

    The expected value comes from the DEPLOYED formula, not from the object under test — a test that
    asks the compiler to agree with itself would pass for any fold, including a plain sum.
    """
    g = v["grasped"]
    r = 1.0 * g
    r = r + 1.5 * (1.0 - math.tanh(4.0 * v["object_to_goal_xy"])) * g
    r = r + 2.5 * (1.0 - math.tanh(6.0 * abs(v["height_above_seat_live"] - 0.015))) * g
    r = r - 3.0 * max(-v["height_above_seat_live"], 0.0) * g
    r = r - 1.0 * max(v["gripper_qpos"] - (-0.6), 0.0) * g
    r = r - 0.3 * v["object_linear_velocity"] - 0.05 * v["object_angular_velocity"]
    r = r - 0.5 * v["not_grasped"]
    if success:                      # `torch.where(info["success"], 12.0, reward)` — row 8
        r = 12.0
    r = r - 0.001 * v["action_norm"]  # row 9, AFTER the replace
    return r


def swapped(i, j):
    """The fixture with two reward rows exchanged."""
    d = descend_doc()
    d["reward"][i], d["reward"][j] = d["reward"][j], d["reward"][i]
    return d


def potential_doc():
    """move_to_target's shape: the only genuinely stateful term in the corpus, and a LATCHED success
    bonus. Weights are the carry_with_potential chassis defaults (move_to_target_env.py:199-215)."""
    return {
        "name": "move_to_target", "kind": "carry_with_potential", "contract": "stack",
        "env_id": "SO100MoveToTarget-v1",
        "scene": {"held": {"type": "cube", "half": 0.014}},
        "reward": [
            {"term": "PredicateBonus", "weight": 0.3, "predicate": "grasped",
             "why": "small constant baseline, deliberately weak."},
            {"term": "DistancePull", "weight": 1.5, "measure": "object_to_goal_xy",
             "kernel": "one_minus_tanh", "k": 3.0, "gate": "grasped",
             "why": "weight 1.5 keeps the maximum below the 5.0 arrival bonus; the sparse "
                    "alternative cost 178M steps at 0%."},
            {"term": "ProgressPotential", "weight": 5.0, "measure": "object_to_goal_xy",
             "gate": "grasped",
             "why": "per-step DECREASE in distance; telescopes to ~1.5 max per episode."},
            {"term": "SuccessBonus", "value": 50.0, "mode": "add", "predicate_ref": "latched",
             "why": "terminal +50 once the latched arrival first fires."},
            {"term": "ActionPenalty", "weight": 0.001, "why": "same 0.001/l2 as every primitive."},
        ],
        "success": "latched(within_radius(anchor=target_pos, radius_expr=0.05))",
    }


def ungated_doc():
    """reach's shape on the `approach` chassis — the one place `_gate`'s absent-gate path is
    reachable. Every carry-family chassis hands a DistancePull `gate: grasped` when the row omits
    one, so deleting a `gate:` key from the descend fixture does not produce an ungated row;
    `approach`'s DistancePull default carries no gate because at reach time there is nothing held to
    gate on."""
    return {
        "name": "reach", "kind": "approach", "contract": "stack", "env_id": "SO100Reach-v1",
        "scene": {"held": {"type": "cube", "half": 0.014}},
        "reward": [
            {"term": "DistancePull", "weight": 1.5, "measure": "tcp_to_object",
             "kernel": "one_minus_tanh", "k": 4.0,
             "why": "reach's entire dense signal, ungated: nothing is held yet."},
            {"term": "SuccessBonus", "value": 9.0, "mode": "add", "why": "terminal +9.0."},
            {"term": "ActionPenalty", "weight": 0.001, "why": "same 0.001/l2."},
        ],
        "success": "above_z(z=0.06)",
    }


def ramp_doc(weight, cap, normalize):
    """compact_grasp vs lift, the 25x bug as a document: lift's Ramp is normalized (max = weight),
    compact_grasp's is not (max = weight*(cap-floor) = 10.0*0.04 = 0.4)."""
    return {
        "name": "grasp", "kind": "hold_and_ramp", "contract": "stack", "env_id": "SO100Lift-v1",
        "scene": {"held": {"type": "cube", "half": 0.014}},
        "reward": [
            {"term": "PredicateBonus", "weight": 1.0, "predicate": "grasped",
             "why": "keep the cube grasped through the ascent."},
            {"term": "Ramp", "weight": weight, "measure": "object_z", "floor": 0.0, "cap": cap,
             "normalize": normalize, "gate": "grasped",
             "why": "lift ramps normalized to 8.0; compact_grasp's un-normalized seat bias maxes "
                    "at 0.4 and must not be rescaled to 10.0."},
            {"term": "SuccessBonus", "value": 9.0, "mode": "add", "why": "terminal +9.0."},
            {"term": "ActionPenalty", "weight": 0.001, "why": "same 0.001/l2."},
        ],
        "success": "above_z(z=0.06)",
    }


def error_from(fn, *a, **kw):
    """Whatever was raised (or None). Deliberately catches everything: a raw KeyError/TypeError out
    of the compiler is a bug — the caller's contract is that they only ever catch CompileError."""
    try:
        fn(*a, **kw)
    except BaseException as exc:      # noqa: BLE001 — see docstring
        return exc
    return None


def result_of(fn, *a):
    """The value, or whatever it raised — one broken helper is one failed check, not an aborted run
    (same reason as `folded`). Module scope because two blocks need it: the boolean-batch helpers
    below, and the `Note` copy/pickle round trip, whose pre-fix behaviour is a `TypeError`."""
    try:
        return fn(*a)
    except BaseException as exc:      # noqa: BLE001 — see docstring
        return exc


def refuses(label, doc, *fragments, kind=CompileError, **kw):
    """A COMPILE-TIER refusal: the value is legal in the vocabulary and this fold does not implement
    it. `kind` stays pinned to a single exception type on purpose — widening it to "SpecError or
    CompileError" is how a suite stops being able to say which tier refuses a value, which is the
    whole content of these checks now that the boundary has moved once."""
    exc = error_from(plan_of, doc, **kw)
    check(f"{label}: refused with {kind.__name__}", isinstance(exc, kind))
    msg = str(exc) if exc is not None else "<nothing raised>"
    for frag in fragments:
        check(f"{label}: message says {frag!r}", frag in msg)
    return exc


def refuses_at_schema(label, doc, *fragments):
    """A SCHEMA-TIER refusal: the value is outside a `Param.choices`, so `parse_spec` refuses it and
    the compiler never sees it. Asserted here, in the COMPILER's suite, because "which tier owns
    this" is a fact about the compiler too — and because the compile-tier entries that used to catch
    these were deleted, and a deleted check with nothing behind it is worse than no check."""
    exc = error_from(parse_spec, doc)
    check(f"{label}: refused with SpecError, before the compiler runs", isinstance(exc, SpecError))
    msg = str(exc) if exc is not None else "<nothing raised>"
    for frag in fragments:
        check(f"{label}: message says {frag!r}", frag in msg)
    return exc


def compiles(label, doc, **kw):
    """Assert a legal document compiles, and hand back the plan (None if it did not).

    The mirror of `refuses`, and the same reason test_skillspec has `accepts`: a plausible regression
    here REFUSES a legal document, and an uncaught CompileError at that point would abort the run and
    hide every check after it — which is exactly what a mutation test must not do. Compiles ONCE:
    doing it twice to report and then return doubles every warning and every side effect the compiler
    has, in a helper whose job is to observe one.
    """
    try:
        plan = plan_of(doc, **kw)
    except BaseException as exc:      # noqa: BLE001 — see `error_from`
        check(f"{label}: compiles (raised {type(exc).__name__}: {exc})", False)
        return None
    check(f"{label}: compiles", True)
    return plan


def folded(plan, vals):
    """`evaluate_plan`, or whatever it raised — `close()` reports a non-number as a failed check
    instead of letting one broken fold take the rest of the run down with it."""
    try:
        return evaluate_plan(plan, vals)
    except BaseException as exc:      # noqa: BLE001 — see docstring
        return exc


def slots(plan):
    return getattr(plan, "state_slots", ())


def param_of(term, name):
    return [p for p in TERMS[term].params if p.name == name][0]


def warning_text(plan):
    return "\n".join(getattr(plan, "warnings", ()) or ())


# ── the cross-process fingerprint probe ─────────────────────────────────────────────────────────

_PROBE = (
    "from bridle.skill.spec import parse_spec;"
    "from bridle.skill.compile import compile_spec;"
    "from bridle.tests.test_skillspec import descend_doc;"
    "import json;"
    "print(json.dumps([compile_spec(parse_spec(descend_doc())).fingerprint(),"
    " hash('descend_to_target')]))"
)


def probe(seed):
    """Compile the descend spec in a FRESH interpreter under `PYTHONHASHSEED=<seed>`, returning
    `(fingerprint, hash('descend_to_target'))`. Two seeds, two processes: the only setup in which a
    `hash()`-based digest and a sha256 one look different."""
    env = dict(os.environ, PYTHONHASHSEED=str(seed), PYTHONPATH=str(REPO))
    out = subprocess.run([sys.executable, "-W", "ignore", "-c", _PROBE],
                         capture_output=True, text=True, env=env, cwd=str(REPO))
    if out.returncode != 0:
        return None, out.stderr.strip().splitlines()[-1:] or ["<no stderr>"]
    return json.loads(out.stdout), None


# ── the GROWN-VOCABULARY probe: is the compiler's 16/16-grasp rule DERIVED, or hand-listed? ──────
# THE CHECK THE ROUND-3 REVIEW FOUND MISSING, and the reason it needs a subprocess. `compile.py`
# derives `_CONTACT_SURFACE_MEASURES` and `_PEAKED_KERNELS` from the vocabulary AT IMPORT. The
# reviewer replaced both with literal frozensets, left both helpers defined, and all nine
# derivation-related predicates in this file stayed green: each of them applies the HELPER to a
# vocabulary the fixture controls and compares that against itself, so none of them looks at what
# the compiler uses. A same-process `_CONTACT_SURFACE_MEASURES == _derive_contact_surfaces(MEASURES)`
# does not close it either — a copy-pasted literal is value-EQUAL today, and that mutation passes
# such an assert (measured 2026-08-13). The property is "a measure/kernel the vocabulary gains
# tomorrow is IN the check by default", which only a run against a DIFFERENT vocabulary can see; and
# the vocabulary has to be grown BEFORE `compile` is imported, i.e. in another interpreter.
#: The synthetic entries every derivation check grows the vocabulary with. Names the vocabulary can
#: never legitimately gain, on purpose: an earlier draft used `seat_clearance`/`seat_span`/`cauchy`,
#: and adding a real fourth kernel named `cauchy` (a Cauchy kernel is a normal thing to want) turned
#: three checks red for the wrong reason — the same defect as a hardcoded `len(kernels) == 3`.
NEW_SIGNED = "signed_measure_added_tomorrow"
NEW_MAGNITUDE = "magnitude_measure_added_tomorrow"
NEW_KERNEL = "kernel_added_tomorrow"

_GROWN_PROBE = (
    "import json;"
    "from bridle.skill import vocab;"
    "from bridle.skill.vocab import Frame, Measure, Sign;"
    f"vocab.MEASURES.__setitem__({NEW_SIGNED!r},"
    f" Measure({NEW_SIGNED!r}, Sign.SIGNED, Frame.LIVE, 'm', 'a signed measure added tomorrow'));"
    f"vocab.MEASURES.__setitem__({NEW_MAGNITUDE!r},"
    f" Measure({NEW_MAGNITUDE!r}, Sign.MAGNITUDE, Frame.LIVE, 'm', 'an unsigned one'));"
    "k = next(p for p in vocab.TERMS['DistancePull'].params if p.name == 'kernel');"
    # `Param` is a frozen dataclass, so growing its `choices` needs the base setter. Growing the
    # vocabulary's OWN object is the point: patching a copy would be the same self-comparison the
    # in-file checks already make.
    f"object.__setattr__(k, 'choices', tuple(k.choices) + ({NEW_KERNEL!r},));"
    "import bridle.skill.compile as c;"
    "print(json.dumps([sorted(c._CONTACT_SURFACE_MEASURES), sorted(c._PEAKED_KERNELS)]))"
)


def grown_probe():
    """`(contact_surface_measures, peaked_kernels)` as `compile.py` derives them in a fresh
    interpreter whose vocabulary gained one SIGNED measure, one MAGNITUDE measure and one kernel."""
    env = dict(os.environ, PYTHONPATH=str(REPO))
    out = subprocess.run([sys.executable, "-W", "ignore", "-c", _GROWN_PROBE],
                         capture_output=True, text=True, env=env, cwd=str(REPO))
    if out.returncode != 0:
        return None, out.stderr.strip().splitlines()[-1:] or ["<no stderr>"]
    return json.loads(out.stdout), None


# ── checks ──────────────────────────────────────────────────────────────────────────────────────

def run_checks():
    # ── the ordered fold: acc = op(acc), in order ───────────────────────────────────────────────
    plan = plan_of(descend_doc(), horizon=HORIZON)
    check("the deployed descend spec compiles", isinstance(plan, RewardPlan))
    check("every row lowered to an Op, in document order",
          isinstance(plan.ops[0], Op)
          and [op.fn_key for op in plan.ops]
          == ["PredicateBonus", "DistancePull", "DistancePull", "HingePenalty", "HingePenalty",
              "VelocityPenalty", "PredicateBonus", "SuccessBonus", "ActionPenalty"])
    check("the replace row is lowered as a replace over the preceding rows",
          (plan.ops[7].kind, plan.ops[7].scope) == ("replace", "preceding"))
    check("the rows around it are plain adds",
          [plan.ops[i].kind for i in (6, 8)] == ["add", "add"])

    v_off = values(success=0.0)
    off = folded(plan, v_off)
    check("with success false, the fold is the plain sum — and equals descend_env.py's",
          close(off, descend_reward_by_hand(v_off, success=False)))

    v_on = values(success=1.0)
    on = folded(plan, v_on)
    check("replace overwrites only the preceding rows",
          close(on, 12.0 - 0.001 * ACTION_NORM))
    check("rows after the replace still apply (the success step is not a bare 12.0)",
          not close(on, 12.0))
    check("the success step is NOT the sum of everything",
          not close(on, off + 12.0))
    check("the success step matches descend_env.py's own fold",
          close(on, descend_reward_by_hand(v_on, success=True)))

    # floor: a lower bound over the same scope, never an overwrite.
    floor_doc = doc_with(reward=descend_doc()["reward"][:1] + [
        {"term": "PredicateBonus", "weight": 4.0, "predicate": "grasped", "mode": "floor",
         "scope": "preceding", "why": "raise the accumulated reward to at least 4.0 while held."},
        descend_doc()["reward"][8],
    ])
    fplan = plan_of(floor_doc, horizon=HORIZON)
    check("floor raises the accumulator to its level where the predicate holds",
          close(folded(fplan, values(grasped=1.0)), 4.0 - 0.001 * ACTION_NORM))
    check("floor leaves the accumulator alone where the predicate is false",
          close(folded(fplan, values(grasped=0.0, not_grasped=1.0)),
                0.0 - 0.001 * ACTION_NORM))
    high_doc = doc_with(reward=[
        {"term": "PredicateBonus", "weight": 9.0, "predicate": "grasped",
         "why": "an accumulator already above the floor."},
        {"term": "PredicateBonus", "weight": 4.0, "predicate": "grasped", "mode": "floor",
         "scope": "preceding", "why": "a floor must never LOWER what is already there."},
        descend_doc()["reward"][8],
    ])
    check("floor never lowers an accumulator already above its level",
          close(folded(plan_of(high_doc, horizon=HORIZON), values()),
                9.0 - 0.001 * ACTION_NORM))

    # `_gate` returns 1.0 for an ABSENT gate. Every other fixture writes a gate, so nothing exercised
    # the default and `return 0.0 if gate is None` used to pass the whole suite. Deleting descend's
    # `gate:` is NOT enough — the carry chassis supplies `gate: grasped` to any DistancePull that
    # omits one — so this needs the `approach` chassis, whose DistancePull default has no gate.
    ungated = compiles("reach's ungated DistancePull", ungated_doc(), horizon=HORIZON)
    check("...and the row really did compile with no gate at all, chassis included",
          ungated is not None and ungated.ops[0].params["gate"] is None)
    check("an absent gate multiplies the row by 1.0, not by 0.0",
          close(folded(ungated, values(tcp_to_object=0.03)),
                1.5 * (1.0 - math.tanh(4.0 * 0.03)) - 0.001 * ACTION_NORM))

    # ── reward_scale is carried, NOT folded in (phase2-decisions §4) ────────────────────────────
    check("reward_scale is carried as `scale`, not divided into the fold", close(plan.scale, 12.0))
    check("the fold itself is UNSCALED — parity is against compute_dense_reward, not the "
          "normalized path", close(on, 12.0 - 0.001 * ACTION_NORM))
    check("`unnormalized: true` means scale 1.0",
          close(plan_of(doc_with(reward_scale={"unnormalized": True}), horizon=HORIZON).scale, 1.0))

    # ── what the plan tells the adapter it needs ────────────────────────────────────────────────
    check("measures_needed covers every measure the rows read, including VelocityPenalty's two",
          set(plan.measures_needed) == {"object_to_goal_xy", "height_above_seat_live",
                                        "gripper_qpos", "object_linear_velocity",
                                        "object_angular_velocity", "action_norm"})
    check("descend needs no state slots", plan.state_slots == ())
    pplan = compiles("the stateful move_to_target spec", potential_doc(), horizon=HORIZON)
    check("a stateful row asks for exactly one slot", len(slots(pplan)) == 1)
    slot = slots(pplan)[0] if slots(pplan) else "<no slot>"
    # `"reward[2]"`, not `"2"`: the property is that the slot names the ROW (two potentials over one
    # measure must not share a buffer, or each reads the other's previous value), and `"2"` also
    # passes for a slot that merely happens to contain the digit.
    check("the slot names the row and the measure it buffers",
          "object_to_goal_xy" in slot and "reward[2]" in slot)

    def pvals(**over):
        """One step of move_to_target, with the potential's buffer holding last step's 5cm."""
        return dict(values(**over), **{slot: 0.05})

    pbase = (0.3 + 1.5 * (1.0 - math.tanh(3.0 * 0.02)) + 5.0 * (0.05 - 0.02)
             - 0.001 * ACTION_NORM)
    check("ProgressPotential folds as weight*(prev - measure)*gate",
          close(folded(pplan, pvals()), pbase))

    # `predicate_ref: latched` reads `success_latched`; `per_step` reads `success`. BOTH DIRECTIONS,
    # because the fixture sets both keys to 0.0 and never overrode the latch — so mutating
    # `_SUCCESS_KEY["latched"]` to `"success"` used to pass every check in this file. The distinction
    # is the reason the parameter exists: a latched bonus wired to the per-step signal stops paying
    # the moment success flickers off, and move_to_target's +50 is paid on a latch for exactly that.
    check("a latched SuccessBonus pays off `success_latched`",
          close(folded(pplan, pvals(success_latched=1.0)), pbase + 50.0))
    check("...and a latched bonus does NOT pay off the per-step `success`",
          close(folded(pplan, pvals(success=1.0)), pbase))
    check("the converse: descend's per_step bonus pays on `success`...",
          close(folded(plan, values(success=1.0)), 12.0 - 0.001 * ACTION_NORM))
    check("...and does NOT pay on a latch it never asked for",
          close(folded(plan, values(success_latched=1.0)), off))

    # ── FLOODING: refused per-step ──────────────────────────────────────────────────────────────
    check("the deployed descend spec passes the flooding check",
          error_from(plan_of, descend_doc(), horizon=HORIZON) is None)

    flooding = row_edited(1, weight=10.0)      # 1.0 + 10.0 + 2.5 = 13.5 >= the 12.0 success value
    exc = refuses("per-step shaping over the success value", flooding,
                  "13.5", "12.0", "reward[1]", "move_to_target_env.py:205", "178M", "5.0",
                  kind=FloodingError, horizon=HORIZON)
    msg = str(exc) if exc is not None else ""
    check("the refusal names every contributing row, not just the biggest",
          "reward[0]" in msg and "reward[2]" in msg)
    check("the refusal does NOT count the non-positive rows (13.5, not 13.5 minus the penalties)",
          "reward[3]" not in msg and "reward[5]" not in msg)
    check("FloodingError is a CompileError", issubclass(FloodingError, CompileError))

    check("the refusal fires at exactly equal, not only strictly above",
          isinstance(error_from(plan_of, row_edited(1, weight=8.5), horizon=HORIZON),
                     FloodingError))     # 1.0 + 8.5 + 2.5 == 12.0
    check("one notch under the success value is accepted",
          error_from(plan_of, row_edited(1, weight=8.4), horizon=HORIZON) is None)

    # THE KNOWN GAP, stated in the output and not only in a code comment. `replace`/`floor` rows do
    # not accumulate, so they are outside the per-step sum by construction and no weight on one can
    # trip the refusal — a reader of a clean compile has no way to know which rows were left out.
    gap_rows = descend_doc()["reward"]
    gap_rows.insert(7, {"term": "PredicateBonus", "weight": 100.0, "predicate": "grasped",
                        "mode": "floor", "scope": "preceding",
                        "why": "an easy predicate floored far above the success value."})
    gap = plan_of(doc_with(reward=gap_rows), horizon=HORIZON)
    check("a floor row at weight 100 out-earns the 12.0 success value and still compiles",
          close(folded(gap, values()), 100.0 - 0.001 * ACTION_NORM)
          and close(folded(gap, values(success=1.0)), 12.0 - 0.001 * ACTION_NORM))
    check("...so the notes say so, naming the row and the mode the sum could not cover",
          all(f in warning_text(gap) for f in ("KNOWN GAP", "reward[7]", "mode=floor", "99.998")))
    check("...and a document with no such row makes no such claim",
          "KNOWN GAP" not in warning_text(plan))

    # ── the per-step maximum table: Ramp.normalize is 25x, not cosmetic ─────────────────────────
    check("a normalized Ramp's maximum IS its weight (10.0 + 1.0 >= the 9.0 success value)",
          isinstance(error_from(plan_of, ramp_doc(10.0, 0.04, True), horizon=HORIZON),
                     FloodingError))
    check("an un-normalized Ramp's maximum is weight*(cap-floor) = 0.4, and passes",
          error_from(plan_of, ramp_doc(10.0, 0.04, False), horizon=HORIZON) is None)
    refuses("a Ramp whose cap is not above its floor", ramp_doc(10.0, 0.0, True),
            "cap", "floor", horizon=HORIZON)

    # ── FLOODING: the horizon is a WARNING, never a refusal ─────────────────────────────────────
    text = warning_text(plan)
    check("the horizon-integrated ratio is computed and warned about, not refused",
          "320.0" in text and "12.0" in text)
    # `"64" in text` — the previous form — stays green off the `320.0` and `768.0` already in the
    # line, so it would pass for a warning that printed the wrong horizon entirely. Pin the phrase.
    check("the integrated warning states the horizon it used, in a form a wrong number cannot fake",
          "5.0/step x 64 steps" in text)
    check("the integrated warning says it is a warning, not a refusal",
          "WARNING (not a refusal)" in text)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        compile_spec(parse_spec(descend_doc()), horizon=HORIZON)
    check("the warning is emitted through the warnings module, not only stored",
          any("320.0" in str(w.message) for w in caught))

    # WHICH notes reach the `warnings` module: only the ones a caller can act on. Emitting on EVERY
    # successful compile made `python -W error` unable to compile any document at all, so a careful
    # caller's only defence was to silence the channel — taking the actionable notes with it.
    with warnings.catch_warnings(record=True) as quiet:
        warnings.simplefilter("always")
        clean = compile_spec(parse_spec(descend_doc()), horizon=HORIZON,
                             terminate_on_success=False)
    check("a compile with nothing to act on emits no warning at all", len(quiet) == 0)
    check("...but the ratio is still RECORDED, both numbers, on the plan",
          "320.0" in warning_text(clean) and "against 768.0" in warning_text(clean))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        werror = error_from(compile_spec, parse_spec(descend_doc()), horizon=HORIZON,
                            terminate_on_success=False)
        # THE LABEL IS NARROW ON PURPOSE. It used to read as a general "-W error can compile a legal
        # document" guarantee while pinning only `terminate_on_success=False`; the same document
        # compiled the default way, and any `horizon=None` compile, still raise — and SHOULD, since
        # each carries a note the caller can act on. What the level split fixed was emitting on
        # EVERY compile including the clean ones, not emitting at all. Both directions pinned so
        # neither claim can drift into the other.
        default_werror = error_from(compile_spec, parse_spec(descend_doc()), horizon=HORIZON)
        nohorizon_werror = error_from(compile_spec, parse_spec(descend_doc()))
    check("...so a caller under `-W error` can compile a document with nothing to ACT on",
          werror is None)
    check("...while one that HAS something to act on still stops that caller, by design",
          isinstance(default_werror, UserWarning) and isinstance(nohorizon_werror, UserWarning)
          and "WARNING (not a refusal)" in str(default_werror)
          and "could NOT be computed" in str(nohorizon_werror))
    with warnings.catch_warnings(record=True) as loud:
        warnings.simplefilter("always")
        compile_spec(parse_spec(potential_doc()), horizon=HORIZON)
    check("a note there IS something to do about still reaches the warnings module",
          any("INCOMPLETE" in str(w.message) for w in loud))
    # `plan.warnings` is the channel nothing dedupes: the stdlib filter is once per (message,
    # location), so a second identical compile in one process prints nothing.
    check("a repeated identical compile still carries the ratio on its plan",
          all("320.0" in warning_text(plan_of(descend_doc(), horizon=HORIZON,
                                              terminate_on_success=False)) for _ in range(2)))

    # THE LEVEL IS PART OF THE NOTE. `compile_spec` computes `(level, text)` and used to store only
    # the text, so a consumer of the AUTHORITATIVE channel could not tell "flooding check INCOMPLETE
    # ... this is not a pass" from an FYI ratio; the distinction survived only in whether `warn()`
    # had already fired — i.e. only in the channel `plan.warnings` exists because one cannot rely on
    # it (`bridle skill` silences it, and the stdlib dedupes it per message+location).
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always")
        lplan = compile_spec(parse_spec(potential_doc()), horizon=HORIZON)
    # `getattr(n, "level", None)`, not `n.level`: a note that regressed to a plain `str` raises
    # `AttributeError` here, and this file's own `folded`/`result_of` convention is that ONE broken
    # thing is ONE FAILED CHECK, not an aborted run with a traceback and every check after it
    # unreported. Round-3 review; `None` is in neither level so the check still goes red.
    check("every note on the plan carries a level",
          bool(lplan.warnings)
          and all(getattr(n, "level", None) in (Note.ACT, Note.FYI) for n in lplan.warnings))
    check("...and is still a plain string, so every existing consumer is untouched",
          all(isinstance(n, str) for n in lplan.warnings)
          and clean.warnings[0] == str(clean.warnings[0])
          and "320.0" in "\n".join(clean.warnings))
    # Same `getattr` guard as above, and for the same reason: these three read `.level` too, so a
    # plain-`str` regression used to take the whole run down at the first of them.
    def level_of(n):
        return getattr(n, "level", None)

    check("the notes emitted through the `warnings` module are EXACTLY the ACT ones",
          [str(w.message) for w in emitted]
          == [str(n) for n in lplan.warnings if level_of(n) == Note.ACT])
    check("an INCOMPLETE note is ACT; a clean integrated ratio is FYI",
          any(level_of(n) == Note.ACT and "INCOMPLETE" in n for n in lplan.warnings)
          and [level_of(n) for n in clean.warnings] == [Note.FYI])

    # A `Note` IS A STRING IN EVERY WAY THE DOCSTRING CLAIMS — including the three that were broken.
    # `str.__getnewargs__` returns the 1-tuple `(text,)`, which `copy`, `deepcopy` and `pickle` all
    # splat into `Note.__new__(cls, text)`: `TypeError: Note.__new__() missing 1 required positional
    # argument: 'text'`, for all three. Latent (nothing in `bridle/` copies or pickles a note) until
    # a `RewardPlan` crosses a process boundary, which is what a training launcher or a cached
    # compile does. `.level` must survive the round trip, or the copy is the pre-Note bare string
    # again — the exact loss the class exists to prevent.
    note = Note(Note.ACT, "flooding check INCOMPLETE: this is not a pass")
    # `result_of`, not a bare call: the pre-fix behaviour RAISES, and one broken round trip is one
    # failed check, not an aborted run (the `folded`/`result_of` convention this file states).
    unpickle = lambda n: pickle.loads(pickle.dumps(n))      # noqa: E731
    round_trips = [result_of(copy.copy, note), result_of(copy.deepcopy, note),
                   result_of(unpickle, note)]
    check("a Note survives copy, deepcopy and pickle, carrying its level",
          all(isinstance(r, Note) and str(r) == str(note) and getattr(r, "level", None) == Note.ACT
              for r in round_trips))
    check("...and the FYI level round-trips too, so the level is carried and not defaulted",
          getattr(result_of(unpickle, Note(Note.FYI, "x")), "level", None) == Note.FYI)

    none_text = warning_text(plan_of(descend_doc()))
    check("with no horizon the integrated check says it could NOT be computed",
          "could NOT be computed" in none_text)
    check("...and names the incident that rule came from",
          "0 violation(s)" in none_text)
    check("...and does not report a pass it did not perform",
          "320" not in none_text)
    check("...but the spec still compiles",
          isinstance(plan_of(descend_doc()), RewardPlan))

    term_text = warning_text(plan_of(descend_doc(), horizon=HORIZON, terminate_on_success=True))
    stay_text = warning_text(plan_of(descend_doc(), horizon=HORIZON, terminate_on_success=False))
    check("terminate_on_success=True pays the bonus once: 320.0 against 12.0",
          "320.0" in term_text and "against 12.0" in term_text and "WARNING" in term_text)
    # The label used to read "terminate_on_success changes the horizon used", which nothing here
    # does: the horizon is 64 either way. What changes is how many steps the BONUS is paid over.
    check("terminate_on_success=False pays the bonus every remaining step: 12.0 x 64 = 768.0",
          "against 768.0" in stay_text and "320.0" in stay_text)
    check("...and with the bonus repeating there is nothing left to warn about",
          "WARNING" not in stay_text)

    # ── UNBOUNDED rows: the check must say it could not conclude ────────────────────────────────
    ptext = warning_text(pplan)
    check("ProgressPotential is reported UNBOUNDED", "UNBOUNDED" in ptext and "reward[2]" in ptext)
    check("...and the check says it could not conclude, rather than passing",
          "not a pass" in ptext or "could not conclude" in ptext)
    check("...and still quotes the bounded part it CAN state", "1.8" in ptext)
    # The two notes have to agree with each other: `_check_flooding` says INCOMPLETE and "not a
    # pass", then the integrated line computes its figure from the bounded subset alone. Without the
    # qualifier the second line reads like a verdict and is the one a reader believes.
    ptail = ptext.split("integrated over the episode")[-1]
    check("...and the integrated line carries the qualifier the INCOMPLETE line earned",
          "at least" in ptail and "LOWER bound" in ptail)
    check("...while a fully bounded document's integrated line claims no such caveat",
          "at least" not in text.split("integrated over the episode")[-1])

    # A tier-2 row may name a DECLARED PARAM (`hover`) instead of a literal — spec.py says the param
    # shadows a measure of the same name. Its number has to travel with the expression, or the
    # evaluator has no value for it and two documents differing only in `hover` share a fingerprint.
    param_expr = {"expr": "2.0 * (1 - tanh(6 * abs(height_above_seat_live - hover)))",
                  "why": "descend's hover attractor written out, reading the declared param."}
    # Row 2's setpoint is written as a literal here so `hover` appears in exactly ONE place — the
    # expression — and the fingerprint check below can only be answered by the binding.
    literal_rows = row_edited(2, setpoint=0.015)["reward"]
    pdoc = doc_with(reward=literal_rows + [param_expr])
    check("an expr row's declared params are bound into the plan",
          close(folded(plan_of(pdoc, horizon=HORIZON), values()),
                off + 2.0 * (1.0 - math.tanh(6 * abs(0.012 - 0.015)))))
    moved = descend_doc()["params"]
    moved["hover"]["value"] = 0.016
    check("...so changing that param changes the fingerprint too",
          plan_of(doc_with(reward=literal_rows + [param_expr], params=moved)).fingerprint()
          != plan_of(pdoc).fingerprint())

    expr_row = {"expr": "2.0 * (1 - tanh(6 * abs(height_above_seat_live - 0.015)))",
                "why": "the tier-2 escape hatch: an arbitrary expression has no stateable maximum."}
    etext = warning_text(plan_of(doc_with(reward=descend_doc()["reward"] + [expr_row]),
                                 horizon=HORIZON))
    check("a tier-2 expr row is UNBOUNDED", "UNBOUNDED" in etext and "reward[9]" in etext)

    custom_row = {"custom": "primitives.descend_to_target.descend_env:crush_term",
                  "why": "tier 3 is opaque by construction."}
    cdoc = doc_with(reward=descend_doc()["reward"] + [custom_row])
    ctext = warning_text(plan_of(cdoc, horizon=HORIZON))
    check("a tier-3 custom row is UNBOUNDED", "UNBOUNDED" in ctext and "reward[9]" in ctext)

    check("an unbounded row does not rescue a spec whose BOUNDED rows already flood",
          isinstance(error_from(plan_of, doc_with(
              reward=row_edited(1, weight=10.0)["reward"] + [expr_row]), horizon=HORIZON),
              FloodingError))

    no_bonus = doc_with(reward=[r for r in descend_doc()["reward"] if r.get("term") != "SuccessBonus"])
    check("with no SuccessBonus row the check says there is nothing to compare against",
          "no SuccessBonus" in warning_text(plan_of(no_bonus, horizon=HORIZON)))

    # ── FINGERPRINT: sha256 over canonical JSON, never hash() ───────────────────────────────────
    fp = plan.fingerprint()
    check("fingerprint is 12 lowercase hex chars", bool(re.fullmatch(r"[0-9a-f]{12}", fp)))
    check("fingerprint is stable in-process", plan_of(descend_doc()).fingerprint() == fp)

    # ── THE TWO ABSOLUTE PINS. Everything else about the digest and the fold in this file is
    # RELATIVE — reorder != original, batch == scalar, weight-changed != unchanged — and not one of
    # those can see a UNIFORM shift. A change that moved every computed value by the same factor, or
    # that changed what goes into the canonical JSON for every plan alike, would leave the whole
    # suite green while meaning that a reward trained under this compiler no longer reproduces:
    # the checkpoint's stamp would not match a recompile of its own document, and the numerical
    # parity sweep against `descend_env.py`'s `compute_dense_reward` would be comparing against a
    # different function than the one it was written for. So two numbers are pinned outright.
    #
    # IF EITHER OF THESE CHANGES, that is the finding — do not re-pin to make the suite green. Find
    # what moved, and state whether the rewards already trained under this compiler still hold.
    #
    #   28f4a705d261  the deployed descend document's plan fingerprint (`skills/descend.yaml` as
    #                 `test_skillspec.descend_doc()` builds it). sha256 over the plan's canonical
    #                 JSON, truncated to 12 hex. It is the same digest a stamped checkpoint carries.
    #   4.79826020570353  the UNSCALED fold of that plan over `values()` — one non-success step of
    #                 descend: grasped, 3mm below the 0.015 hover setpoint, 2cm off centre, jaw at
    #                 -0.70, nothing crushed, ||a|| = 2.0. Unscaled because parity is against
    #                 `compute_dense_reward`, not the /12.0 normalized path (phase2-decisions §4).
    #                 Nine rows fold into it, so the digits are load-bearing: a term dropped, a
    #                 kernel changed, or a weight rebound moves them.
    check("the descend plan's ABSOLUTE fingerprint is unchanged (a reward trained under this "
          f"compiler is no longer reproducible if this moves) — got {fp}", fp == "28f4a705d261")
    check("the descend plan's ABSOLUTE unscaled fold over one non-success step is unchanged — got "
          f"{off!r}", close(off, 4.79826020570353))
    # ...and the two pins are not each other's alias: the fold is pinned to 1e-12, so a change too
    # small to move the digest's input still moves the number, and a `why`-only edit moves neither.
    check("...and a weight edit moves BOTH, so neither pin is a constant this file could satisfy "
          "by accident",
          plan_of(row_edited(1, weight=1.6), horizon=HORIZON).fingerprint() != "28f4a705d261"
          and not close(folded(plan_of(row_edited(1, weight=1.6), horizon=HORIZON), v_off),
                        4.79826020570353))

    a, err_a = probe(0)
    b, err_b = probe(12345)
    check(f"the fingerprint subprocesses ran (stderr: {err_a or err_b})", a is not None and b is not None)
    if a and b:
        check("the two subprocesses really did use different hash seeds (hash() differs)",
              a[1] != b[1])
        check("fingerprint is stable across processes with different PYTHONHASHSEED",
              a[0] == b[0] == fp)

    check("changing a weight changes the fingerprint",
          plan_of(row_edited(1, weight=1.6)).fingerprint() != fp)
    hover = descend_doc()["params"]
    hover["hover"]["value"] = 0.016
    check("changing a params.X value the rows bind changes the fingerprint",
          plan_of(doc_with(params=hover)).fingerprint() != fp)
    check("changing the success criterion changes the fingerprint — it is IN the fold",
          plan_of(doc_with(success="all[grasped]")).fingerprint() != fp)
    check("changing only a `why` does NOT change the fingerprint: prose is not the reward",
          plan_of(row_edited(0, why="rewritten prose, same number.")).fingerprint() == fp)
    check("the horizon is not part of the fingerprint — it is not part of the reward function",
          plan_of(descend_doc(), horizon=HORIZON, terminate_on_success=True).fingerprint() == fp)

    # Rows 4 and 5 are the two HingePenalty rows: both plain adds, so exchanging them cannot change
    # the arithmetic — but the fold is ORDERED, so the digest must still record that they moved.
    reordered = plan_of(swapped(3, 4), horizon=HORIZON)
    check("reordering two independent rows does NOT change behaviour",
          close(folded(reordered, v_off), off))
    check("...but DOES change the fingerprint (order is semantic under an ordered fold)",
          reordered.fingerprint() != fp)

    # ── an Op is HASHABLE, because `eq=True, frozen=True` says it is ────────────────────────────
    # `params` is a `MappingProxyType`, which is unhashable, so the dataclass-generated `__hash__`
    # raised `TypeError: unhashable type: 'mappingproxy'` for every `hash(op)`, `op in {...}` and
    # `set(plan.ops)` — on a type whose own declaration invites all three. Nothing did it yet, which
    # is the only reason it was never seen.
    twin = plan_of(descend_doc(), horizon=HORIZON)
    check("an Op hashes instead of raising, and equal Ops hash equal",
          plan.ops[8] == twin.ops[8] and hash(plan.ops[8]) == hash(twin.ops[8]))
    check("...so set membership works and does not collapse distinct ops",
          len({plan.ops[8], twin.ops[8]}) == 1 and len(set(plan.ops)) == len(plan.ops)
          and plan.ops[0] in set(twin.ops))
    # `pdoc`'s last row is the tier-2 expression reading the declared `hover`, so its params hold
    # BOTH an `Expr` and a non-empty nested `bindings` mapping — the two shapes a naive
    # `hash(tuple(params.items()))` would still choke on.
    check("...including an op whose params hold a nested mapping and an Expr",
          isinstance(hash(plan_of(pdoc, horizon=HORIZON).ops[9]), int)
          and dict(plan_of(pdoc, horizon=HORIZON).ops[9].params["bindings"]) == {"hover": 0.015})
    check("...and an op differing only in a weight is a separate member, never an alias",
          len({plan_of(row_edited(1, weight=1.6), horizon=HORIZON).ops[1], plan.ops[1]}) == 2)

    # ── refusals, each asserted AT THE TIER THAT OWNS IT ────────────────────────────────────────
    # Silently ignoring an authored parameter is the "trains, logs, contributes nothing" failure this
    # whole phase exists to stop, so every one of these is a refusal, not a default. WHICH tier
    # refuses is the part that moved: `kernel`, `mode` and `predicate_ref` gained `Param.choices` on
    # 2026-08-12 and are now schema-tier, which made three compile-tier checks here fail. They are
    # re-asserted as SpecError rather than accepted as "SpecError or CompileError", and the dead
    # `_HONOURED` entries behind them were deleted — a refusal branch that can never fire is a check
    # the module advertises and does not perform.
    #
    # SCHEMA TIER — "no document may say this": the vocabulary enumerates the legal set.
    refuses_at_schema("an unknown kernel", row_edited(1, kernel="one_minus_tan"),
                      "one_minus_tan", "one_minus_tanh", "reward[1]")
    refuses_at_schema("a mode outside the vocabulary's three", row_edited(7, mode="multiply"),
                      "multiply", "replace")
    refuses_at_schema("a predicate_ref that is neither per_step nor latched",
                      row_edited(7, predicate_ref="whenever"), "predicate_ref", "latched")
    refuses_at_schema("a side outside above/below", row_edited(3, side="beside"),
                      "beside", "below")
    # The other two deleted `_HONOURED` entries. Named in the check below as "no longer a dead
    # branch" but, until now, with no replacement asserted anywhere — so deleting the schema
    # `choices` that replaced them would have left both values SILENTLY ACCEPTED and the suite green.
    refuses_at_schema("a norm outside the one the fold implements", row_edited(8, norm="l1"),
                      "reward[8].norm", "l1", "l2")
    refuses_at_schema("a PredicateBonus mode outside the vocabulary's three",
                      row_edited(0, mode="multiply"), "reward[0].mode", "multiply", "floor")
    check("...and the compiler no longer keeps a dead branch for any of them",
          not any(key in _HONOURED for key in
                  (("DistancePull", "kernel"), ("SuccessBonus", "mode"), ("HingePenalty", "side"),
                   ("SuccessBonus", "predicate_ref"), ("PredicateBonus", "mode"),
                   ("ActionPenalty", "norm"))))

    # COMPILE TIER — "legal in the vocabulary, not implemented by this fold". `scope` and `axes`
    # carry no `choices`, so nothing upstream can catch them and these are the compiler's own.
    refuses("a scope the fold does not implement", row_edited(7, scope="all"),
            "scope", "preceding", horizon=HORIZON)
    refuses("an `axes` restriction nothing implements", row_edited(1, axes="xy"),
            "axes", "object_to_goal_xy", horizon=HORIZON)
    # The one RETAINED `_HONOURED` entry. Nothing asserted it: not the refusal, and not the fact
    # that makes compile-tier ownership correct in the first place (`body` carries no `choices`, so
    # `spec.py` cannot catch it). `body` gaining `choices` would make this branch dead exactly the
    # way kernel/mode/side/norm's did, and the suite would not have noticed.
    refuses("a VelocityPenalty body the fold does not implement", row_edited(5, body="tcp"),
            "reward[5].body", "tcp", "held", horizon=HORIZON)
    # ALL FOUR `_HONOURED` KEYS, not three. `("PredicateBonus", "scope")` is the fourth
    # (`compile.py`'s `_HONOURED`); its table VALUE is pinned further down (the `_SCOPE_REACH`
    # block), but its openness was not, so that entry could go dead exactly the way
    # kernel/mode/side/norm's did — `choices` arriving on it would move the refusal a tier without
    # anything here noticing.
    check("...and all four really are open in the vocabulary, which is why the compiler owns them",
          not param_of("SuccessBonus", "scope").choices
          and not param_of("PredicateBonus", "scope").choices
          and not param_of("DistancePull", "axes").choices
          and not param_of("VelocityPenalty", "body").choices)
    check("...and the openness check covers every key the table has, so a fifth cannot go unpinned",
          set(_HONOURED) == {("VelocityPenalty", "body"), ("SuccessBonus", "scope"),
                             ("PredicateBonus", "scope")})

    # THE `_UNIMPLEMENTED` TABLE, ALL FOUR ENTRIES. Only `DistancePull.axes` above was exercised;
    # the other three were a table nothing read. `gamma` is the one whose own comment says the two
    # readings of it (`prev - gamma*m` vs `gamma*(prev - m)`) are DIFFERENT REWARDS, so an entry
    # that silently stopped refusing would pick one of them on the author's behalf.
    def potential_row_edited(**changes):
        """move_to_target's fixture with its ProgressPotential row edited — the descend document has
        no stateful row, so `gamma`/`terminal_zero` are not reachable through `row_edited`."""
        d = potential_doc()
        d["reward"][2].update(changes)
        return d

    refuses("a HingePenalty `enabled_if` nothing implements", row_edited(3, enabled_if="grasped"),
            "reward[3].enabled_if", "decides whether a row EXISTS", "gate:", horizon=HORIZON)
    refuses("a ProgressPotential discount other than 1.0", potential_row_edited(gamma=0.99),
            "reward[2].gamma", "prev - gamma*m", "gamma*(prev - m)", horizon=HORIZON)
    refuses("a ProgressPotential `terminal_zero`", potential_row_edited(terminal_zero=True),
            "reward[2].terminal_zero", "terminal step", horizon=HORIZON)
    check("...and each entry's own OFF value still compiles, so these are refusals and not a ban",
          error_from(plan_of, potential_row_edited(gamma=1.0, terminal_zero=False),
                     horizon=HORIZON) is None
          and error_from(plan_of, row_edited(3, enabled_if=None), horizon=HORIZON) is None)

    # The legal scope set is not a literal: it is `tuple(_SCOPE_REACH)`, the same table the fold
    # dispatches through. Teaching the table a second scope must therefore CHANGE THE FOLD — the
    # thing that cannot happen is a new legal scope quietly continuing to mean `preceding`. The
    # accumulator here is 9.0 and the floor's level is 4.0, so `preceding` (reach = 9.0, floor is a
    # no-op) and `nothing` (reach = 0.0, floor lands) give different numbers.
    scoped_rows = [
        {"term": "PredicateBonus", "weight": 9.0, "predicate": "grasped",
         "why": "an accumulator already above the floor's level."},
        {"term": "PredicateBonus", "weight": 4.0, "predicate": "grasped", "mode": "floor",
         "scope": "nothing", "why": "a floor whose scope reaches nothing, not the preceding rows."},
        descend_doc()["reward"][8]]
    check("the legal scope set IS the fold's dispatch table, not a literal beside it",
          _HONOURED[("SuccessBonus", "scope")] == tuple(_SCOPE_REACH)
          and _HONOURED[("PredicateBonus", "scope")] == tuple(_SCOPE_REACH))
    _SCOPE_REACH["nothing"] = lambda acc: 0.0 * acc
    # `_HONOURED` snapshots the keys at import, so the legal set is patched alongside — the property
    # under test is that the FOLD dispatches through this table, which is what makes a scope added to
    # it mean something other than `preceding`.
    patched = dict(_HONOURED)
    _HONOURED[("PredicateBonus", "scope")] = tuple(_SCOPE_REACH)
    try:
        scoped = folded(plan_of(doc_with(reward=scoped_rows), horizon=HORIZON), values())
        check("a scope in `_SCOPE_REACH` compiles and the fold uses ITS reach, not `preceding`",
              close(scoped, 4.0 - 0.001 * ACTION_NORM))
        check("...and that is a DIFFERENT number from what `preceding` would have folded",
              not close(scoped, 9.0 - 0.001 * ACTION_NORM))
    finally:
        del _SCOPE_REACH["nothing"]
        _HONOURED.update(patched)
    check("...and with the table restored the same scope is refused again",
          isinstance(error_from(plan_of, row_edited(7, scope="nothing"), horizon=HORIZON),
                     CompileError))

    # ── params.X is BOUND at compile time ───────────────────────────────────────────────────────
    check("a params.X reference is bound to its number, not left as a string",
          close(plan.ops[2].params["setpoint"], 0.015))
    bad = descend_doc()["params"]
    bad["hover"]["value"] = "high"
    refuses("a params.X whose value is not a number", doc_with(params=bad),
            "params.hover", "high", horizon=HORIZON)

    # ── the attractor that must not peak at contact (16/16 grasps, 2026-06-04) ──────────────────
    refuses("a DistancePull peaking at the seat surface", row_edited(2, setpoint=0.0),
            "16/16", "setpoint", horizon=HORIZON)
    zero = descend_doc()["params"]
    zero["hover"]["value"] = 0.0
    refuses("...including when it arrives through params.X", doc_with(params=zero),
            "16/16", horizon=HORIZON)
    refuses("...and over height_above_resting, which the hardcoded set only happened to include",
            row_edited(2, measure="height_above_resting", setpoint=0.0), "16/16", horizon=HORIZON)

    # THE REFUSAL IS RIGHT FOR A NEGATIVE WEIGHT AND THE PROSE WAS NOT. The rule ignores the weight's
    # sign on purpose (one uniform rule is the one `vocab_document()` can state), but the message
    # told that author their row "is maximised at the setpoint" — false, as `_max_distance_pull`
    # says in the same file: a negative weight inverts the kernel into a repeller. The error messages
    # ARE the API for a model that cannot read the source, so a refusal that misdescribes the row is
    # a refusal the author cannot act on.
    refuses("a NEGATIVE-weight pull centred on the seat surface is still refused",
            row_edited(2, weight=-2.5, setpoint=0.0), "16/16", "setpoint", horizon=HORIZON)
    neg = str(error_from(plan_of, row_edited(2, weight=-2.5, setpoint=0.0), horizon=HORIZON))
    pos = str(error_from(plan_of, row_edited(2, weight=2.5, setpoint=0.0), horizon=HORIZON))
    check("...and the message says MINIMISED, names the weight, and calls it a repeller",
          "MINIMISED at the setpoint" in neg and "-2.5" in neg and "repeller" in neg)
    check("...and does NOT tell that author the row is maximised there",
          "maximised at the setpoint" not in neg)
    check("...while a positive weight still reads `maximised at the setpoint`",
          "maximised at the setpoint" in pos and "MINIMISED" not in pos)

    # THE SET IS DERIVED FROM THE VOCABULARY, and the vocabulary's prose has to describe the same
    # rule: `vocab_document()` is where this refusal is ADVERTISED to the authoring model. Code,
    # comment and document used to state three different rules — the code checked 3 of the 5 SIGNED
    # measures, the comment justified the gap by citing `object_to_goal_z` (MAGNITUDE, never a
    # candidate), and the document promised the rule over every SIGNED measure.
    signed = {n for n, m in MEASURES.items() if m.sign is Sign.SIGNED}
    check("every SIGNED measure is either checked or excluded with a stated reason",
          signed == set(_CONTACT_SURFACE_MEASURES) | set(_ZERO_IS_NOT_A_CONTACT_SURFACE)
          and all(_ZERO_IS_NOT_A_CONTACT_SURFACE.values()))
    # THE PARTITION ABOVE IS SATISFIED BY ANY PARTITION, including the hand-list that was replaced —
    # it cannot see the headline property, which is that a SIGNED measure added tomorrow is IN the
    # check by DEFAULT. That needs the derivation run over a vocabulary the fixture controls.
    grown = dict(MEASURES, **{NEW_SIGNED: Measure(
        NEW_SIGNED, Sign.SIGNED, Frame.LIVE, "m", "a signed measure the vocabulary gains")})
    derived = _derive_contact_surfaces(grown)
    check("a SIGNED measure the vocabulary gains tomorrow lands in the check with no edit here",
          NEW_SIGNED in derived and derived - {NEW_SIGNED} == _CONTACT_SURFACE_MEASURES)
    check("...while a MAGNITUDE one does not — the derivation reads `sign`, not the name",
          NEW_MAGNITUDE not in _derive_contact_surfaces(dict(MEASURES, **{NEW_MAGNITUDE: Measure(
              NEW_MAGNITUDE, Sign.MAGNITUDE, Frame.LIVE, "m", "an unsigned measure")})))
    check("...and an EXCLUDED name is still excluded after the vocabulary grows",
          set(_ZERO_IS_NOT_A_CONTACT_SURFACE) & derived == set())
    check("no MAGNITUDE measure is in the set — move_to_3d peaks at object_to_goal_z == 0 on purpose",
          not any(MEASURES[n].sign is Sign.MAGNITUDE for n in _CONTACT_SURFACE_MEASURES)
          and error_from(plan_of, row_edited(1, setpoint=0.0), horizon=HORIZON) is None)
    check("a peaked pull at gripper_qpos == 0 is 'open the jaw', not a contact failure",
          error_from(plan_of, row_edited(2, measure="gripper_qpos", setpoint=0.0),
                     horizon=HORIZON) is None)
    bullet = [ln for ln in vocab_document().splitlines() if "attractor_setpoint_not_at" in ln][0]
    check("the document advertises the rule over exactly the measures the compiler applies it to",
          {n for n in signed if n in bullet} == set(_ZERO_IS_NOT_A_CONTACT_SURFACE))

    # ── ...AND THE SET THE COMPILER USES IS THAT DERIVATION, not a hand-list that happens to match.
    # Everything above this line is satisfied by a literal frozenset (see `_GROWN_PROBE`, which
    # documents the mutation and why the obvious same-process assert does not catch it). These are
    # the checks that go red for it, on BOTH axes at once.
    grown_sets, gerr = grown_probe()
    check(f"the grown-vocabulary subprocess ran (stderr: {gerr})", grown_sets is not None)
    gmeasures, gkernels = grown_sets if grown_sets else ([], [])
    check("a SIGNED measure the vocabulary gains is in the set THE COMPILER USES, not merely in "
          "the helper's output",
          NEW_SIGNED in gmeasures
          and set(gmeasures) - {NEW_SIGNED} == set(_CONTACT_SURFACE_MEASURES))
    check("...and a MAGNITUDE one is not, so the probe's vocabulary really did grow both ways",
          bool(gmeasures) and NEW_MAGNITUDE not in gmeasures)
    check("a kernel the vocabulary gains is in the set THE COMPILER USES",
          NEW_KERNEL in gkernels and set(gkernels) - {NEW_KERNEL} == set(_PEAKED_KERNELS))

    # ── THE SAME RULE ON THE KERNEL AXIS ────────────────────────────────────────────────────────
    # Round 1 closed the measure axis and left this one open: `_PEAKED_KERNELS` named two of the
    # three kernels and omitted `neg_linear`, which is `-|measure - setpoint|` — maximised AT the
    # setpoint, at 0.0 rather than 1.0, which is a shallower peak and not the absence of one. So
    # `DistancePull{measure: height_above_seat_live, kernel: neg_linear}` on DEFAULT parameters
    # (`setpoint` defaults to 0.0) compiled clean while pulling a grasped object onto the seat: the
    # 16/16-grasp failure of 2026-06-04, reached through the kernel the refusal did not cover.
    kernels = param_of("DistancePull", "kernel").choices
    # NOT `len(kernels) == 3`: a legitimately added fourth kernel would turn that red for the wrong
    # reason, and the doc-bullet check at the end of this block already forces the document to be
    # updated when the set changes. What must hold is that the three kernels the bullet DESCRIBES
    # are still offered (so the prose is not describing a vocabulary that moved on) and that every
    # kernel offered is checked.
    check("every kernel the vocabulary offers is checked, and the three the document describes are "
          "still among them",
          set(kernels) == set(_PEAKED_KERNELS)
          and {"one_minus_tanh", "gaussian", "neg_linear"} <= set(kernels))
    for kernel in kernels:
        # ONE refusal per kernel, so dropping any single one from `_PEAKED_KERNELS` turns this red —
        # which the previous "is it in the set" style of check could not do.
        refuses(f"a {kernel} DistancePull peaking at the seat surface",
                row_edited(2, kernel=kernel, setpoint=0.0),
                "16/16", "setpoint", kernel, horizon=HORIZON)
    check("...and each of them still compiles at descend's real hover setpoint, 0.015",
          all(error_from(plan_of, row_edited(2, kernel=k, setpoint=0.015), horizon=HORIZON) is None
              for k in kernels))
    check("...and over a measure whose zero is not a surface, at any kernel",
          all(error_from(plan_of, row_edited(2, kernel=k, measure="gripper_qpos", setpoint=0.0),
                         horizon=HORIZON) is None for k in kernels))
    # THE "WITH A STATED REASON" HALF WAS VACUOUS ON THIS AXIS. `_KERNEL_PEAK_IS_NOT_AT_SETPOINT` is
    # empty — that emptiness IS the finding on the kernel axis — so `all({}.values())` and
    # `set({}) <= set(kernels)` are trivially true and exercised nothing at all. Lifting the two
    # halves into a predicate and testing the PREDICATE on a populated dict is what makes the label
    # true today, rather than true only on the day someone excludes a kernel.
    def excluded_with_a_stated_reason(excluded, legal):
        """Every exclusion names something the vocabulary actually offers, and gives a reason."""
        return set(excluded) <= set(legal) and all(excluded.values())

    check("every legal kernel is either checked or excluded with a stated reason",
          set(kernels) == set(_PEAKED_KERNELS) | set(_KERNEL_PEAK_IS_NOT_AT_SETPOINT)
          and excluded_with_a_stated_reason(_KERNEL_PEAK_IS_NOT_AT_SETPOINT, kernels))
    check("...and that rule is not vacuous: it refuses a blank reason and an exclusion naming a "
          "kernel the vocabulary does not offer",
          excluded_with_a_stated_reason({"neg_linear": "a stated reason"}, kernels)
          and not excluded_with_a_stated_reason({"neg_linear": ""}, kernels)
          and not excluded_with_a_stated_reason({NEW_KERNEL: "a stated reason"}, kernels))
    check("...and the same rule holds on the MEASURE axis, where the dict is non-empty",
          excluded_with_a_stated_reason(_ZERO_IS_NOT_A_CONTACT_SURFACE, MEASURES))
    kgrown = _derive_peaked_kernels(set(kernels) | {NEW_KERNEL})
    check("a kernel the vocabulary gains tomorrow lands in the check with no edit here",
          NEW_KERNEL in kgrown and kgrown - {NEW_KERNEL} == set(_PEAKED_KERNELS))
    check("the document advertises the rule over exactly the kernels the compiler applies it to",
          {k for k in kernels if k in bullet} == set(_PEAKED_KERNELS))

    # ── the fold over a BATCH, not a scalar ─────────────────────────────────────────────────────
    # Vectorisation is this phase's binding constraint: one fold runs over up to 4096 environments at
    # once, which is why `_where`/`_max`/`_relu`/`_clamp` are branch-free and `tanh`/`exp` dispatch to
    # the value's own method. Scalar floats cannot test any of that — `c*a + (1-c)*b` and
    # `a if c else b` agree on every scalar and disagree on every batch. Three environments, one
    # fold, per-environment success and per-environment geometry.
    def batched(p, over):
        """(the fold over a 3-element batch, the three scalar folds it has to equal)."""
        batch = {k: Vec([v] * 3) for k, v in values().items()}
        batch.update({k: Vec(col) for k, col in over.items()})
        scalar = [values(**{k: col[i] for k, col in over.items()}) for i in range(3)]
        return folded(p, batch), [folded(p, s) for s in scalar]

    for label, bplan, over in (
            ("the descend fold (_where on a per-env success, _tanh, _relu on the crush hinge)",
             plan, {"success": [0.0, 1.0, 0.0],
                    "height_above_seat_live": [0.012, 0.012, -0.004]}),
            ("a gaussian kernel (_exp dispatching to the value's own method)",
             plan_of(row_edited(1, kernel="gaussian"), horizon=HORIZON),
             {"object_to_goal_xy": [0.02, 0.06, 0.20]}),
            ("a `floor` row (_max)", fplan,
             {"grasped": [1.0, 0.0, 1.0], "not_grasped": [0.0, 1.0, 0.0]}),
            ("a normalized `Ramp` (_clamp = _min + _max, and a divide)",
             plan_of(ramp_doc(1.0, 0.04, True), horizon=HORIZON),
             {"object_z": [-0.01, 0.02, 0.09]}),
    ):
        got, want = batched(bplan, over)
        check(f"{label} folds element-wise over a 3-element batch", close_all(got, want))
    check("...and a Python `if` on a batch is a failure here, not a silent one-branch",
          isinstance(error_from(bool, Vec([0.0, 1.0, 0.0])), AssertionError))

    # ── ...over a batch whose COMPARISONS ARE BOOLEAN, which is the shape torch actually has ─────
    # `Vec` cannot see this: its `__gt__` returns floats, so it is a batch that happens to be
    # pre-numeric — the one kind the helpers worked for. A torch comparison yields a BOOL tensor and
    # `1 - bool_tensor` is a RuntimeError, so `compile._relu(torch.tensor([-1., 2.]))` raised and
    # `HingePenalty`/`Ramp` could not fold a raw tensor AT ALL, while the module docstring promised
    # one fold for a CPU float and a batched CUDA tensor alike. Task 5's adapter worked around it
    # from outside with a wrapper whose comparisons return floats; that fixed one caller and left
    # the stdlib path broken for everyone who believed the docstring.
    check("a boolean batch refuses `1 - c` the way a torch bool tensor does",
          isinstance(error_from(operator.sub, 1, BoolMask([True, False, True])), AssertionError)
          and isinstance(error_from(operator.sub, BoolMask([True, False, True]), 1), AssertionError))
    check("...and BoolVec's comparisons really do produce one, or none of this bites",
          isinstance(BoolVec([1.0, 2.0, 3.0]) > 2.0, BoolMask)
          and isinstance(BoolVec([1.0, 2.0, 3.0]) < 2.0, BoolMask))

    # `result_of` (module scope, beside `error_from`) is what keeps a raising helper a reported
    # check rather than an aborted run.
    for label, fn, args, want in (
            ("_relu", _relu, (BoolVec([-1.0, 0.0, 2.0]),), [0.0, 0.0, 2.0]),
            ("_max", _max, (BoolVec([-1.0, 0.0, 2.0]), 0.5), [0.5, 0.5, 2.0]),
            ("_min", _min, (BoolVec([-1.0, 0.0, 2.0]), 0.5), [-1.0, 0.0, 0.5]),
            ("_clamp", _clamp, (BoolVec([-1.0, 0.5, 2.0]), 0.0, 1.0), [0.0, 0.5, 1.0]),
            ("_where", _where, (BoolMask([True, False, True]), BoolVec([1.0, 2.0, 3.0]),
                                BoolVec([10.0, 20.0, 30.0])), [1.0, 20.0, 3.0]),
    ):
        check(f"compile.{label} folds a boolean-conditioned batch element-wise",
              close_all(result_of(fn, *args), want))

    # ...and the whole fold, not only the helpers: HingePenalty (`_relu`) and Ramp (`_clamp`) are
    # the two rows that could not run at all, so both go through a boolean-conditioned batch here.
    bool_batch = {k: BoolVec([v] * 3) for k, v in values().items()}
    bool_batch["height_above_seat_live"] = BoolVec([0.012, -0.004, 0.03])
    bool_scalar = [values(height_above_seat_live=h) for h in (0.012, -0.004, 0.03)]
    check("the descend fold (HingePenalty/_relu) runs over a boolean-conditioned batch",
          close_all(folded(plan, bool_batch), [folded(plan, s) for s in bool_scalar]))
    rplan = plan_of(ramp_doc(1.0, 0.04, True), horizon=HORIZON)
    rbatch = {k: BoolVec([v] * 3) for k, v in values().items()}
    rbatch["object_z"] = BoolVec([-0.01, 0.02, 0.09])
    check("a normalized Ramp (_clamp) runs over one too",
          close_all(folded(rplan, rbatch),
                    [folded(rplan, values(object_z=z)) for z in (-0.01, 0.02, 0.09)]))

    # ── ...and a boolean batch arriving as a VALUE, not only as a comparison's output ────────────
    # `bool_batch` above makes `success` a BoolVec — a NUMBER — so `SuccessBonus{mode: replace}`'s
    # `_where(success, level, reached)` only ever saw a condition that some comparison had already
    # produced. The real adapter reads `info["success"]`, which is a torch BOOL tensor, and hands it
    # straight to `_where`: the condition arrives as a bool and nothing in the fold has compared
    # anything yet. That path lived only in the deleted torch script, so nothing in the suite
    # covered it — `_numeric`'s `c * 1` is exactly what has to fire here, and a `_where` written as
    # `c * a + (1 - c) * b` WITHOUT it raises on this input the way torch does.
    sbatch = {k: BoolVec([v] * 3) for k, v in values().items()}
    sbatch["success"] = BoolMask([False, True, False])
    check("the replace row folds a boolean `success` handed in as a VALUE, not as a comparison",
          close_all(folded(plan, sbatch),
                    [folded(plan, values(success=s)) for s in (0.0, 1.0, 0.0)]))
    check("...and the success environment really is the 12.0 branch, so the mask was READ",
          close_all(folded(plan, sbatch),
                    [descend_reward_by_hand(values(), success=False),
                     12.0 - 0.001 * ACTION_NORM,
                     descend_reward_by_hand(values(), success=False)]))
    # A `floor` row takes the same condition through `_where(c, _max(reached, level), acc)`, so the
    # PredicateBonus predicate is the second place a raw bool can arrive as a value.
    pbatch = {k: BoolVec([v] * 3) for k, v in values().items()}
    pbatch["grasped"] = BoolMask([True, False, True])
    pbatch["not_grasped"] = BoolMask([False, True, False])
    check("...and so does a `floor` row whose PREDICATE arrives as a boolean batch",
          close_all(folded(fplan, pbatch),
                    [folded(fplan, values(grasped=g, not_grasped=1.0 - g))
                     for g in (1.0, 0.0, 1.0)]))

    # ── the evaluator's own contract ────────────────────────────────────────────────────────────
    missing = values()
    del missing["gripper_qpos"]
    exc = error_from(evaluate_plan, plan, missing)
    check("a missing value is a CompileError naming it, never a bare KeyError",
          isinstance(exc, CompileError) and "gripper_qpos" in str(exc))
    check("...and the message labels the list as what WAS supplied, not as what the plan needs",
          "what WAS supplied" in str(exc))
    exc = error_from(evaluate_plan, plan_of(cdoc, horizon=HORIZON), values())
    check("the stdlib evaluator refuses a custom row instead of guessing it",
          isinstance(exc, CompileError) and "custom" in str(exc))

    # `_bind_number(None)` is unreachable through any document today — `RewardScale.divisor` defaults
    # to 12.0, so even an empty block binds a number — but the module's only promise to a caller is
    # that they catch `CompileError`, and `float(None)` is a bare `TypeError` that walks past it.
    check("`reward_scale: {}` still binds the inherited divisor rather than a None",
          close(plan_of(doc_with(reward_scale={}), horizon=HORIZON).scale, 12.0))
    check("a numeric field handed None is a CompileError, not a raw TypeError",
          isinstance(error_from(_bind_number, "reward_scale.divisor", None, {}), CompileError))


def test_bridle():
    """pytest entry point — the same checks, reported as one assertion. The standalone `main()`
    below stays the primary interface: the project venv has no pytest."""
    FAILS.clear()
    run_checks()
    assert not FAILS, f"{len(FAILS)} check(s) failed: {FAILS}"


def main():
    run_checks()
    print(f"\n{len(FAILS)} failure(s)")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
