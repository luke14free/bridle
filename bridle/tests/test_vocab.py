"""Unit test for bridle.skill.vocab — the authorable surface.

WHY THIS EXISTS: this vocabulary is what a local 27-30B model is handed in its prompt, and every
property tested here was paid for by a measured failure.

  SIGN     an unsigned height_above_seat_live makes the crush penalty identically zero, silently
           deleting the term that exists because pressing to dz=0 broke 16/16 grasps (2026-06-04).
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
import re
import sys

from bridle.skill.spec import _quantity
from bridle.skill.vocab import (
    CHASSIS, MEASURES, PREDICATES, TERMS, Frame, Sign, base_term, vocab_document,
)

FAILS = []


def _predicate_names_in(expr: str) -> set:
    """Extract the predicate names an expression string references: either a bare name
    ('grasped') or a nested call ('and_(grasped, above_z(z=0.06))'). A name in CALL position
    (immediately followed by '(') is a predicate reference; anything else inside the call
    (kwargs like 'anchor=target_pos') is scene data, not a predicate, so it's ignored. A string
    with no call at all is itself a bare predicate name.
    """
    calls = set(re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(", expr))
    return calls if calls else {expr.strip()}


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
    # (2026-08-12 review) the previous check here —
    #   "floor" in str([p.default for p in modes if p.name == "mode"]) or "floor" in <term doc>
    # — passes as soon as the word "floor" appears ANYWHERE in the term's free-text doc, regardless
    # of whether `mode` itself accepts it; `Param.default` for mode is just the string "add", never
    # a choice list, so the check never actually inspected the schema. Fixed by giving `mode` an
    # explicit `choices` tuple (Param.choices) and asserting against THAT — schema, not prose.
    mode_param = next(p for p in modes if p.name == "mode")
    check("mode declares an explicit choices list (schema, not prose)", bool(mode_param.choices))
    check("mode choices are exactly add|replace|floor",
          set(mode_param.choices) == {"add", "replace", "floor"})

    # ── every closed set is `choices`, never prose (2026-08-12 re-review, minor 3) ────────────────
    # PredicateBonus.mode was fixed above and the same hole was left open in five more places:
    # `predicate_ref` stated `per_step | latched` in its doc and declared `choices=()`, so
    # `predicate_ref: "per_stpe"` parsed clean and reached the fold; `kernel`, `side` and `norm` the
    # same. SuccessBonus.mode was worse than prose-only — its prose said `add | replace` while the
    # fold has always honoured `floor` too. The expected sets are compile.py's `_HONOURED` table,
    # i.e. what the fold actually implements, so the schema tier cannot declare legal a value the
    # fold will refuse (or the reverse).
    for term_name, pname, expected in (
            ("SuccessBonus", "predicate_ref", {"per_step", "latched"}),
            ("SuccessBonus", "mode", {"add", "replace", "floor"}),
            ("HingePenalty", "side", {"above", "below"}),
            ("DistancePull", "kernel", {"one_minus_tanh", "neg_linear", "gaussian"}),
            ("ActionPenalty", "norm", {"l2"}),
    ):
        p = next(x for x in TERMS[term_name].params if x.name == pname)
        check(f"{term_name}.{pname} declares its legal set as `choices`, not prose",
              set(p.choices) == expected)
    # ...and the general rule, so the NEXT parameter added with a prose set cannot repeat the gap:
    # the house convention for spelling a legal set in a doc is `a | b`, and a doc that does it must
    # back it with `choices`. (This is the recurrence guard; the table above is the content.)
    prose_only = sorted(f"{t}.{p.name}" for t, term in TERMS.items() for p in term.params
                        if p.type == "str" and re.search(r"\w \| \w", p.doc) and not p.choices)
    check("no str param spells a legal set in prose without declaring `choices`" +
          (f" (prose-only: {prose_only})" if prose_only else ""),
          not prose_only)
    check("the prose-set guard bites on the shape it names (some doc does spell `a | b`)",
          any(re.search(r"\w \| \w", p.doc) for term in TERMS.values() for p in term.params))

    # ── measures: sign and frame are mandatory and load-bearing ──
    check("height_above_seat_live exists", "height_above_seat_live" in MEASURES)
    check("height_above_seat_live is SIGNED — an unsigned one zeroes the crush penalty",
          "height_above_seat_live" in MEASURES and
          MEASURES["height_above_seat_live"].sign is Sign.SIGNED)
    # (2026-08-12 review, finding 4) the LIVE reading used to be keyed on the BARE quantity while its
    # static-goal twin carried a frame suffix, so the same string meant "the live seat" in MEASURES
    # and "illegal, say which frame" in the schema — and the carry chassis wrote the illegal one.
    # Both frames now carry their frame in the key; the bare quantity names nothing.
    check("the bare quantity is not a measure — both frames are frame-qualified",
          "height_above_seat" not in MEASURES)
    check("both frames of the seat height exist and are SIGNED",
          {"height_above_seat_live", "height_above_seat_static_goal"} <= set(MEASURES) and
          all(MEASURES[n].sign is Sign.SIGNED
              for n in ("height_above_seat_live", "height_above_seat_static_goal")))

    # ── the frame-collision rule, as a GENERAL invariant (2026-08-12 re-review, Important 1) ──────
    # Design doc §1.2: a quantity readable in more than one frame carries its frame IN THE KEY, and
    # the bare quantity names nothing. Nothing enforced that beyond the seat height. spec.py builds
    # its ambiguity table `_FRAME_VARIANTS` under the filter `len(keys) > 1 and q not in MEASURES`,
    # so a family that grew a frame-qualified sibling beside a BARE key — add
    # `height_above_resting_static_goal` next to the existing `height_above_resting` — drops OUT of
    # the table, and the bare name goes on resolving silently to LIVE. Same defect, different family.
    # Neither guard written for the seat-height rename can see it: `set(LEGAL_MEASURE_NAMES) ==
    # set(MEASURES)` is a tautology (LEGAL_MEASURE_NAMES is literally `frozenset(MEASURES)`), and the
    # sibling check above only asserts the one string "height_above_seat" is absent.
    #
    # So assert the property itself, over every key: a measure's quantity stem is either the measure
    # itself or is not a measure at all. `_quantity` is imported from spec.py rather than reimplemented
    # so the test cannot drift from the stem function the ambiguity table is actually built with.
    # PROVEN TO BITE: adding a colliding `height_above_resting_static_goal` to MEASURES fails this
    # check with `collisions: ['height_above_resting_static_goal']` while every other check stays green.
    collisions = sorted(k for k in MEASURES if _quantity(k) != k and _quantity(k) in MEASURES)
    check("no frame-qualified measure shadows a bare measure of the same quantity" +
          (f" (collisions: {collisions})" if collisions else ""),
          not collisions)
    check("the collision check bites on the shape it names (a stem-stripping frame suffix exists)",
          any(_quantity(k) != k for k in MEASURES))
    check("every chassis default names a measure that exists (the rename can't half-land)",
          all(row["measure"] in MEASURES
              for chassis in CHASSIS.values() for row in chassis.defaults.values()
              if "measure" in row))
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
    # NOTE (2026-08-12 review): this used to require "weight" or "value" literally in every row.
    # VelocityPenalty has neither (its real params are linear_weight/angular_weight) — the old
    # check only passed because carry's VelocityPenalty carried a stray, undeclared `weight` key
    # (Finding 3), which is exactly the class of bug the declared-params check below now catches
    # directly. Relaxed to "supplies more than just a why" so fixing that bug doesn't reintroduce it.
    check("carry supplies a default for every row it names",
          all(len(v) > 1 for v in carry.defaults.values()))
    check("every chassis default carries a why",
          all("why" in v for v in carry.defaults.values()))

    # ── every chassis-default key is a declared parameter of its term (2026-08-12 review, Finding 3)
    # carry's VelocityPenalty used to carry weight=0.3 alongside linear_weight/angular_weight — an
    # undeclared field (Term("VelocityPenalty").params has no `weight`) shown to the model as if it
    # were real. This makes that whole CLASS of drift impossible, not merely absent right now.
    for cname, chassis in CHASSIS.items():
        for term_name, row in chassis.defaults.items():
            term = TERMS[base_term(term_name)]
            declared = {p.name for p in term.params}
            extra = {k for k in row if k != "why"} - declared
            check(f"{cname}.{term_name} has no undeclared params" +
                  (f" (extra: {sorted(extra)})" if extra else ""),
                  not extra)

    # ── every predicate/gate name anywhere in CHASSIS resolves to a real PREDICATES entry
    # (2026-08-12 review, Finding 1). carry_with_potential used to reference fabricated compound
    # names ("grasped_and_high", "at_target_xy_grasped_high") that PREDICATES never defined, and
    # `release` referenced "released" the same way (released = ~is_grasping in source, i.e. exactly
    # `not_grasped`). A composite value like `and_(grasped, above_z(z=0.06))` is fine — every name in
    # CALL position must resolve; bare names must resolve directly.
    bad_predicate_refs = []
    for cname, chassis in CHASSIS.items():
        for term_name, row in chassis.defaults.items():
            for field in ("predicate", "gate"):  # NOT predicate_ref: that's an add|latched mode
                val = row.get(field)
                if val is None:
                    continue
                for name in _predicate_names_in(str(val)):
                    if name not in PREDICATES:
                        bad_predicate_refs.append(f"{cname}.{term_name}.{field}={val!r} -> {name!r}")
    check("every predicate/gate name in CHASSIS exists in PREDICATES", not bad_predicate_refs)
    for ref in bad_predicate_refs:
        print(f"    unresolved: {ref}")

    # ── action_delta_norm ships OFF, or the numerical parity test breaks ──
    ap = carry.defaults.get("ActionPenalty", {})
    check("ActionPenalty defaults to action_norm, not the delta",
          ap.get("measure", "action_norm") == "action_norm")

    # ── descend's success criterion, which was NOT EXPRESSIBLE before 2026-08-13 ─────────────────
    # `height_above_resting_in(band)` is `0 <= h <= band`. `descend_env.py`'s `low` is `h < band`
    # with NO LOWER BOUND, on purpose: a cube pressed below its resting height is still low, and it
    # is the crush penalty, not the success gate, that handles pressing. Task 6 measured the gap —
    # 37 of 4456 sampled states on this component alone, 64/64 of the states below the seat, and 37
    # of 64 at FULL criterion level once `centered` is forced true. A NEW predicate rather than an
    # optional `floor` param on the old one: the schema now treats an authored `null` as "not
    # supplied" (a deliberate Task 3 fix), so `floor: null` would fall back to the default and
    # reintroduce exactly this bug.
    check("below_resting_height exists — descend's `low` gate is now expressible",
          "below_resting_height" in PREDICATES)
    below = PREDICATES.get("below_resting_height")
    band = PREDICATES.get("height_above_resting_in")
    check("...taking the same single `band` parameter as the bounded predicate it sits beside",
          below is not None and band is not None
          and [p.name for p in below.params] == [p.name for p in band.params] == ["band"]
          and all(p.required for p in below.params))
    # The docs are the ONLY thing that makes the choice between the two visible to the author — the
    # whole point of a second predicate rather than a parameter. Both directions, so neither doc can
    # be rewritten into a version that no longer names its sibling.
    check("...and each of the two names the other, so an author can tell which one they want",
          below is not None and band is not None
          and "height_above_resting_in" in below.doc
          and "below_resting_height" in band.doc)
    check("...and the new one carries the measurement that forced it, not just an assertion",
          below is not None and "4456" in below.doc and "37" in below.doc and "64" in below.doc
          and "UNBOUNDED BELOW" in below.doc)
    check("...while the bounded one still says it IS bounded below",
          band is not None and "[0, band]" in band.doc)

    # ── the document a 30B model reads ──
    doc = vocab_document()
    check("document names every term", all(t in doc for t in TERMS))
    check("document names every measure", all(m in doc for m in MEASURES))
    # Nothing asserted this, and a predicate the vocabulary defines but the PAYLOAD omits is a
    # predicate the author is never told about — which is the whole defect 2b closes for the
    # `success:` grammar, one level up.
    check("document names every predicate", all(p in doc for p in PREDICATES))
    # THE `success:` GRAMMAR WAS DOCUMENTED NOWHERE THE AUTHOR CAN SEE IT. `spec.py` leaves that
    # grammar to the evaluator on purpose; the evaluator's `_desugar_brackets` documents the bracket
    # sugar in a Python docstring; the design doc §4 example and the acceptance fixture both WRITE
    # the bracket form. So the model was expected to produce grammar it had never been shown.
    check("the payload documents the `success:` bracket sugar it expects the author to write",
          "all[a, b]" in doc and "any[a, b]" in doc and "and_(a, b)" in doc and "or_(a, b)" in doc)
    check("document states each default's rationale", doc.count("why") >= 6 or "because" in doc)
    check("document marks which measures are signed", "signed" in doc.lower())
    # ── SIZE: a derived token budget, not a round char count (2026-08-12 re-review, Important 2) ──
    # This used to read `len(doc) < 24000` against a measured 23,995 — FOUR characters of slack. The
    # next word added to any measure doc, term doc or chassis `why` would fail a test in a file
    # unrelated to the change, and the cheapest way to green it is deleting rationale prose: exactly
    # the evidence the "comments carry their measurement" rule exists to protect. The 24,000 was also
    # a raw char count standing in for a token budget, with no recorded derivation. Both fixed here.
    #
    # WHERE THE CEILING COMES FROM. This document is a 27-30B model's prompt payload and has to fit
    # alongside a task description and one worked example. The audit's §5 SIZE table costs the
    # vocabulary at 3,400-4,600 tokens and the FULL authoring prompt — vocabulary plus the 55-field
    # scene/tolerance schema inline — at a worst case of ~8,000, and concludes: "That fits a 30B
    # model's working context with large margin (32k minimum for anything current). Context length is
    # not the constraint." 8,000 tokens is therefore the ceiling: the largest payload the audit
    # costed and still called comfortable, applied to its largest single component.
    #
    # WHY THE MARGIN IS WHAT IT IS. The document measures ~7,430 estimated tokens today (29,724
    # chars) — above the audit's 3,400-4,600 vocabulary line because that table prices no chassis at
    # all (the 6 presets with their `why` rationales are the audit's own §6 recommendation, costed
    # nowhere in its §5), plus amendment 1's added measures and terms, plus the 17th predicate and
    # the `success:` grammar line added 2026-08-13 (+1,179 chars over the 24,696 measured before
    # them), plus the "## The document you are writing" section and the mode-aware rewrite of the
    # flooding bullet added the same day for C2 (+3,849 chars, +962 est. tokens over 25,875).
    #
    # THAT LEAVES ~570 TOKENS, AND THAT IS TIGHT — SAID OUT LOUD RATHER THAN QUIETLY ACCEPTED. The
    # ceiling is the worst case for the WHOLE authoring prompt, which the design assumed would be
    # this document PLUS a task description PLUS one worked example; ~570 tokens does not hold the
    # last two. `bridle skill vocab` therefore NAMES the worked example
    # (`primitives/descend_to_target/skill.yaml`) instead of inlining it, and the next addition here
    # has to come with a re-measured ceiling. What it must NOT come with is deleted rationale prose,
    # which is the evidence the "comments carry their measurement" rule exists to protect. A
    # document that DOUBLED would estimate ~14,900 tokens and still fail, which is the point of
    # there being a ceiling at all.
    #
    # ~4 chars/token is the working conversion this repo already uses. It is an estimate, and the
    # ceiling is sized so that being 30% wrong about it does not change the verdict.
    est_tokens = len(doc) / 4
    check(f"the prompt payload fits its token budget ({len(doc)} chars ~= {est_tokens:.0f} est. "
          f"tokens, ceiling 8000 = the audit's §5 worst case for the whole authoring prompt)",
          est_tokens < 8000)
    check(f"...and the ceiling still bites: a doubled document ({2 * est_tokens:.0f} est. tokens) "
          f"would fail it", 2 * est_tokens >= 8000)
    check("document is not a stub", len(doc) > 3000)

    # ── the payload carries the DOCUMENT GRAMMAR, not only the vocabulary (C2) ───────────────────
    # Measured 2026-08-13 on the 25,875-char payload: ZERO occurrences of `params.`, `reward_scale`,
    # `severity`, `scene:`, `kind:`, `preflight`, `contract:`, `expr:`, `custom:`, `init:`,
    # `- term:`, `name:`, `reward:` or `mode: floor`. Not one top-level key, the three tiers never
    # mentioned, `floor` a legal `choices` value on two terms with its semantics nowhere. The author
    # is a local 27-30B model that cannot introspect the API, `bridle skill vocab` prints this string
    # and nothing else, and its own help calls it "the payload you put in the authoring model's
    # prompt" — so a grammar that is not in here is a grammar the author was never given.
    #
    # Each group below is one thing an author cannot write the document without. Grouped rather than
    # listed flat so a failure says WHICH rule went missing, not "a substring is absent".
    grammar = {
        "every top-level key, required and optional": [
            "name:", "kind:", "contract:", "env_id", "scene:", "reward:", "success:",
            "params:", "init:", "reward_scale", "preflight",
        ],
        "the two schema rules beyond the term vocabulary (phase2-decisions §3)": [
            "EVERY scene object declares a `type`", "severity",
        ],
        "`why:` is mandatory and not inherited": [
            "`why:` IS MANDATORY ON EVERY ROW", "never inherited",
        ],
        "the three row tiers, each with an example": [
            "THREE TIERS OF ROW", "- {term:", "- {expr:", "- {custom:",
        ],
        "the tier-1 `params.hover` vs tier-2 bare-`hover` asymmetry": [
            "`params.` IS SPELLED TWO WAYS", "params.hover", "bare `hover`",
        ],
        "all three `mode` values and what each does to the fold": [
            "`add` (default), `acc + x`", "`replace`, `where(condition, value, acc-over-scope)`",
            "`floor`, `max(acc, value)`", "scope: preceding",
        ],
        "the fold is ordered, so row order is part of the program": [
            "ROW ORDER IS THE PROGRAM", "acc = row(acc)",
        ],
    }
    for rule, fragments in grammar.items():
        absent = [f for f in fragments if f not in doc]
        check(f"the payload states {rule}"
              + (f" — missing: {absent}" if absent else ""), not absent)

    # The measured shape of the C2 defect, restated as one check: the whole reason it survived
    # review is that every individual token above reads like an implementation detail, while their
    # JOINT absence is "the document grammar is not in the document".
    never_again = ["params.", "reward_scale", "severity", "scene:", "kind:", "preflight",
                   "contract:", "expr:", "custom:", "init:", "- term:", "name:", "reward:"]
    zero_again = [t for t in never_again if doc.count(t) == 0]
    check("...and none of the 13 tokens whose count was zero on 2026-08-13 is zero again"
          + (f" — zero again: {zero_again}" if zero_again else ""), not zero_again)


def test_bridle():
    run_checks()
    assert not FAILS, f"{len(FAILS)} check(s) failed: {FAILS}"


def main():
    run_checks()
    print(f"\n{len(FAILS)} failure(s)")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
