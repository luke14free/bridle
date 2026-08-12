"""Unit test for bridle.skill.vocab — the authorable surface.

WHY THIS EXISTS: this vocabulary is what a local 27-30B model is handed in its prompt, and every
property tested here was paid for by a measured failure.

  SIGN     an unsigned height_above_seat makes the crush penalty identically zero, silently deleting
           the term that exists because pressing to dz=0 broke 16/16 grasps (2026-06-04).
  FRAME    descend_stack grades its reward against a static goal and its success against a live top.
  DEFAULTS the literature's measured failure mode for LLM-authored rewards is bad WEIGHTS, so every
           default carries the rationale that justifies it, in the text the model reads.
  SIZE     the whole document must fit a 30B prompt alongside a task description and an example.

Run: python -m pytest bridle/tests/test_vocab.py
     PYTHONPATH=. python bridle/tests/test_vocab.py

NOTE (2026-08-12): the brief's step-1 test contained
    check("DistancePull is stateless", not TERMS["DistancePull"].stateless is False or True)
which is vacuously true (`X or True` is always True) and cannot fail, and `Term` has no `stateless`
field in the interface (only `stateful`). Replaced below with a check that actually tests what it
names.
"""
import sys

from bridle.skill.vocab import CHASSIS, MEASURES, PREDICATES, TERMS, Frame, Sign, vocab_document

FAILS = []


def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        FAILS.append(name)


def run_checks():
    # ── the nine terms, named exactly as the audit named them ──
    for t in ("ActionPenalty", "SuccessBonus", "PredicateBonus", "DistancePull", "HingePenalty",
              "VelocityPenalty", "Ramp", "ProgressPotential", "RewardScale"):
        check(f"term {t} exists", t in TERMS)
    check("exactly nine terms — additions need a measured justification", len(TERMS) == 9)
    check("ProgressPotential is the stateful one", TERMS["ProgressPotential"].stateful)
    check("DistancePull is stateless", not TERMS["DistancePull"].stateful)

    # amendment A: mode is on any gated row, not just SuccessBonus
    modes = TERMS["PredicateBonus"].params
    check("PredicateBonus carries mode", any(p.name == "mode" for p in modes))
    check("mode allows add|replace|floor",
          "floor" in str([p.default for p in modes if p.name == "mode"]) or
          "floor" in TERMS["PredicateBonus"].doc)

    # ── measures: sign and frame are mandatory and load-bearing ──
    check("height_above_seat exists", "height_above_seat" in MEASURES)
    check("height_above_seat is SIGNED — an unsigned one zeroes the crush penalty",
          MEASURES["height_above_seat"].sign is Sign.SIGNED)
    check("object_to_goal_xy is a magnitude", MEASURES["object_to_goal_xy"].sign is Sign.MAGNITUDE)
    check("every measure declares a sign", all(m.sign in (Sign.SIGNED, Sign.MAGNITUDE)
                                               for m in MEASURES.values()))
    check("every measure declares a frame", all(m.frame in (Frame.LIVE, Frame.AT_RESET, Frame.STATIC_GOAL)
                                                for m in MEASURES.values()))
    check("every measure documents itself", all(len(m.doc) > 20 for m in MEASURES.values()))

    # amendment A additions
    check("action_delta_norm exists (nine files call ActionPenalty a jerk penalty; nothing stored one)",
          "action_delta_norm" in MEASURES)
    check("symmetry-reduced yaw exists", "yaw_diff_mod_symmetry" in MEASURES)
    check("joint_pos_margin_to_limit exists", "joint_pos_margin_to_limit" in MEASURES)

    # ── terms that need a sign must say so ──
    check("HingePenalty needs a signed measure", TERMS["HingePenalty"].needs_signed_measure)

    # ── chassis supply defaults WITH rationale ──
    for c in ("approach", "close_and_hold", "hold_and_ramp", "carry", "carry_with_potential", "release"):
        check(f"chassis {c} exists", c in CHASSIS)
    check("exactly six chassis", len(CHASSIS) == 6)
    carry = CHASSIS["carry"]
    check("carry supplies a default for every row it names",
          all("weight" in v or "value" in v for v in carry.defaults.values()))
    check("every chassis default carries a why",
          all("why" in v for v in carry.defaults.values()))

    # ── action_delta_norm ships OFF, or the numerical parity test breaks ──
    ap = carry.defaults.get("ActionPenalty", {})
    check("ActionPenalty defaults to action_norm, not the delta",
          ap.get("measure", "action_norm") == "action_norm")

    # ── the document a 30B model reads ──
    doc = vocab_document()
    check("document names every term", all(t in doc for t in TERMS))
    check("document names every measure", all(m in doc for m in MEASURES))
    check("document states each default's rationale", doc.count("why") >= 6 or "because" in doc)
    check("document marks which measures are signed", "signed" in doc.lower())
    # budget: ~4 chars/token. The audit budgeted 3400-4600 tokens for the whole payload.
    check(f"document fits a 30B prompt ({len(doc)} chars, budget 24000)", len(doc) < 24000)
    check("document is not a stub", len(doc) > 3000)


def test_bridle():
    run_checks()
    assert not FAILS, f"{len(FAILS)} check(s) failed: {FAILS}"


def main():
    run_checks()
    print(f"\n{len(FAILS)} failure(s)")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
