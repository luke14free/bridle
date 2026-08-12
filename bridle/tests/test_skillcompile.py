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
import json
import math
import os
import re
import subprocess
import sys
import warnings
from pathlib import Path

from bridle.skill.compile import (
    CompileError, FloodingError, Op, RewardPlan, compile_spec, evaluate_plan,
)
from bridle.skill.spec import parse_spec
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


def refuses(label, doc, *fragments, kind=CompileError, **kw):
    exc = error_from(plan_of, doc, **kw)
    check(f"{label}: refused with {kind.__name__}", isinstance(exc, kind))
    msg = str(exc) if exc is not None else "<nothing raised>"
    for frag in fragments:
        check(f"{label}: message says {frag!r}", frag in msg)
    return exc


def compiles(label, doc, **kw):
    """Assert a legal document compiles, and hand back the plan (None if it did not).

    The mirror of `refuses`, and the same reason test_skillspec has `accepts`: a plausible regression
    here REFUSES a legal document, and an uncaught CompileError at that point would abort the run and
    hide every check after it — which is exactly what a mutation test must not do.
    """
    exc = error_from(plan_of, doc, **kw)
    check(f"{label}: compiles", exc is None)
    return plan_of(doc, **kw) if exc is None else None


def folded(plan, vals):
    """`evaluate_plan`, or whatever it raised — `close()` reports a non-number as a failed check
    instead of letting one broken fold take the rest of the run down with it."""
    try:
        return evaluate_plan(plan, vals)
    except BaseException as exc:      # noqa: BLE001 — see docstring
        return exc


def slots(plan):
    return getattr(plan, "state_slots", ())


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
    check("the slot names the row and the measure it buffers",
          bool(slots(pplan)) and "object_to_goal_xy" in slots(pplan)[0]
          and "2" in slots(pplan)[0])
    check("ProgressPotential folds as weight*(prev - measure)*gate",
          close(folded(pplan, dict(values(), **{slots(pplan)[0]: 0.05} if slots(pplan) else {})),
                0.3 + 1.5 * (1.0 - math.tanh(3.0 * 0.02)) + 5.0 * (0.05 - 0.02)
                - 0.001 * ACTION_NORM))

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
    check("the integrated warning states the horizon it used", "64" in text)
    check("the integrated warning says it is a warning, not a refusal",
          "WARNING (not a refusal)" in text)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        compile_spec(parse_spec(descend_doc()), horizon=HORIZON)
    check("the warning is emitted through the warnings module, not only stored",
          any("320.0" in str(w.message) for w in caught))

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
    check("terminate_on_success changes the horizon used — a repeating bonus integrates too",
          "against 768.0" in stay_text and "320.0" in stay_text)
    check("...and with the bonus repeating there is nothing left to warn about",
          "WARNING" not in stay_text)

    # ── UNBOUNDED rows: the check must say it could not conclude ────────────────────────────────
    ptext = warning_text(pplan)
    check("ProgressPotential is reported UNBOUNDED", "UNBOUNDED" in ptext and "reward[2]" in ptext)
    check("...and the check says it could not conclude, rather than passing",
          "not a pass" in ptext or "could not conclude" in ptext)
    check("...and still quotes the bounded part it CAN state", "1.8" in ptext)

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

    # ── refusals for a value the fold cannot honour ─────────────────────────────────────────────
    # Silently ignoring an authored parameter is the "trains, logs, contributes nothing" failure
    # this whole phase exists to stop, so every one of these is a refusal, not a default.
    refuses("an unknown kernel", row_edited(1, kernel="one_minus_tan"),
            "one_minus_tan", "one_minus_tanh", "reward[1]", horizon=HORIZON)
    refuses("a scope the fold does not implement", row_edited(7, scope="all"),
            "scope", "preceding", horizon=HORIZON)
    refuses("a mode the fold does not implement", row_edited(7, mode="multiply"),
            "multiply", "replace", horizon=HORIZON)
    refuses("an `axes` restriction nothing implements", row_edited(1, axes="xy"),
            "axes", "object_to_goal_xy", horizon=HORIZON)
    refuses("a predicate_ref that is neither per_step nor latched",
            row_edited(7, predicate_ref="whenever"), "predicate_ref", "latched", horizon=HORIZON)

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

    # ── the evaluator's own contract ────────────────────────────────────────────────────────────
    missing = values()
    del missing["gripper_qpos"]
    exc = error_from(evaluate_plan, plan, missing)
    check("a missing value is a CompileError naming it, never a bare KeyError",
          isinstance(exc, CompileError) and "gripper_qpos" in str(exc))
    exc = error_from(evaluate_plan, plan_of(cdoc, horizon=HORIZON), values())
    check("the stdlib evaluator refuses a custom row instead of guessing it",
          isinstance(exc, CompileError) and "custom" in str(exc))


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
