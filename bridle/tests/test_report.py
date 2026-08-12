"""Unit test for bridle.skill.report — the text `bridle skill compile` prints.

WHY THIS EXISTS AS ITS OWN TEST. The renderer used to be three private helpers inside `cli.py`, where
the only way to exercise it was to run the CLI and read its stdout (2026-08-13 review, finding 8). It
is now a module that takes a `SkillSpec`, a `RewardPlan` and two plain values, so the properties
below are asserted against the returned string directly — no argparse namespace, no subprocess.

WHAT IS WORTH ASSERTING, AND WHY IT IS THE PROVENANCE. The reader of this text is a local 27-30B
model, and the measured failure mode of LLM-authored rewards is bad WEIGHTS, not bad term choice. A
weight the chassis supplied trains exactly as hard as one the author typed, so the load-bearing
property is that NOTHING IS PRINTED UNATTRIBUTED: every parameter line says `authored`,
`chassis '<name>' default`, `term default` or `compiler-supplied`, and an inherited row prints the
rationale that came with it — once for the row, not once per field. A report that showed the numbers
without their origin would let an author "fix" a weight they never wrote and never see it.

Run: PYTHONPATH=. python bridle/tests/test_report.py     (the project venv has no pytest)
     python -m pytest bridle/tests/test_report.py
"""
import inspect
import re
import sys
import warnings

from bridle.skill.compile import compile_spec
from bridle.skill.report import format_plan, format_warnings, wrap
from bridle.skill.spec import parse_spec

FAILS = []


def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        FAILS.append(name)


def reach_doc():
    """The smallest document that exercises every provenance branch at once.

    Chosen so no branch is vacuous: row 0 authors NOTHING but `term` and `why`, so its weight,
    measure and kernel are all `chassis 'approach' default`; row 1 authors `mode`/`scope` while
    inheriting `value`, so one row mixes authored with inherited AND is the `replace/preceding` row
    the column alignment needs; and `SuccessBonus` makes the compiler attach a `condition`, which is
    the `compiler-supplied` case. Every number here is the deployed `approach` chassis default
    (vocab.CHASSIS['approach']), so a change to those defaults changes this fixture rather than
    silently passing against a stale copy.
    """
    return {
        "name": "reach", "kind": "approach", "contract": "reach", "env_id": "SO100Reach-v1",
        "scene": {"target": {"type": "cube", "half": 0.02}},
        "params": {"tol": {"value": 0.05, "severity": "run", "doc": "arrival tolerance"}},
        "reward": [
            {"term": "DistancePull",
             "why": "inherit the whole approach signal: reward = -tcp_to_object, neg_linear at "
                    "weight 1.0. Authoring nothing here is the point of the fixture."},
            {"term": "SuccessBonus", "mode": "replace", "scope": "preceding",
             "why": "the terminal jackpot REPLACES accumulated shaping; the 9.0 value itself is "
                    "inherited from the chassis and must be labelled as such."},
            {"term": "ActionPenalty",
             "why": "the same 0.001/l2 as all 15 audited primitives, inherited, applied last."},
        ],
        "success": "within_radius(tcp_to_object, params.tol)",
    }


def rendered(**kw):
    doc = reach_doc()
    spec = parse_spec(doc)
    with warnings.catch_warnings():
        # Exactly what `cmd_skill` does: compile's notes reach the reader through
        # `RewardPlan.warnings`, printed in full by `format_warnings`, not through the `warnings`
        # module — which prints once per process and buries the text in interpreter furniture.
        warnings.simplefilter("ignore")
        plan = compile_spec(spec, horizon=kw.pop("compile_horizon", 64),
                            terminate_on_success=kw.pop("terminate", None))
    kw.setdefault("horizon", 64)
    kw.setdefault("terminate_on_success", "unknown")
    return spec, plan, format_plan(doc, spec, plan, **kw)


#: `  [7] replace/preceding  SuccessBonus` — index, the fold operation, and the term it applies.
_ROW = re.compile(r"^  \[(\d+)\] (\S+)( +)(\S+)$", re.M)
#: `        weight         = 1.0                    chassis 'approach' default`
_PARAM = re.compile(r"^ {8}(\w+) {2,}= (.+)$", re.M)


def provenance_of(param_line):
    """The attribution at the end of one parameter line, or None if it carries none.

    Deliberately NOT a substring search for the word "default": `chassis 'approach' default` and
    `term default` are different claims — one means a number came from the deployed lineage and the
    other means nobody chose it for this skill — and a check that could not tell them apart would
    pass on a report that swapped them.
    """
    for tag in ("authored", "term default", "compiler-supplied"):
        if tag in param_line:
            return tag
    m = re.search(r"chassis '(\w+)' default", param_line)
    return f"chassis {m.group(1)}" if m else None


def run_checks():
    spec, plan, text = rendered()

    # ── the renderer is decoupled from the CLI, which is why it has this test at all ─────────────
    params = tuple(inspect.signature(format_plan).parameters)
    check("format_plan takes a doc, a spec, a plan and two values — no argparse namespace",
          params == ("doc", "spec", "plan", "horizon", "terminate_on_success"))

    # ── every row of the fold is printed, once, in document order ────────────────────────────────
    rows = _ROW.findall(text)
    check("every op in the plan gets exactly one row line", len(rows) == len(plan.ops) == 3)
    check("row lines are numbered in DOCUMENT order — the fold is a program, not a set",
          [int(i) for i, _, _, _ in rows] == list(range(len(plan.ops))))
    check("each row line names the term it renders",
          [label for _, _, _, label in rows]
          == ["DistancePull", "SuccessBonus", "ActionPenalty"])

    # ── finding 5: the term names line up, including across a scoped row ─────────────────────────
    # Guarded against vacuity first: padding only the scope was invisible on a fold of all-`add`
    # rows, and it was a fold that MIXED them that put the labels four columns apart.
    ops = {op for _, op, _, _ in rows}
    check("the fixture actually mixes op shapes, so the alignment check below is not vacuous",
          ops == {"add", "replace/preceding"})
    columns = {len("  [x] ") + len(op) + len(pad) for _, op, pad, _ in rows}
    check("every term name starts in the same column, `add` and `replace/preceding` alike",
          len(columns) == 1)

    # ── nothing is printed unattributed ──────────────────────────────────────────────────────────
    param_lines = [ln for ln in text.splitlines() if _PARAM.match(ln)]
    check("the fixture renders parameter lines at all", len(param_lines) >= 8)
    check("every parameter line says where its value came from",
          all(provenance_of(ln) is not None for ln in param_lines))
    sources = {provenance_of(ln) for ln in param_lines}
    check("all four provenances are exercised, so none of them is dead rendering code",
          sources == {"authored", "term default", "compiler-supplied", "chassis approach"})

    # ── an inherited value is labelled, and its rationale is printed ONCE for the row ────────────
    # A chassis default is a whole row (`DistancePull` is weight 1.0 AND measure tcp_to_object AND
    # kernel neg_linear, and the rationale is why those go together), so repeating it under each
    # field would bury the rest of the plan in copies of one paragraph.
    check("a value the author never typed is labelled as the chassis' own",
          any("chassis 'approach' default" in ln and ln.strip().startswith("weight")
              for ln in param_lines))
    inherited_why = "reach's whole dense signal"
    check("the chassis' rationale for an inherited row reaches the report",
          inherited_why in text)
    check("...and appears once for the row, not once per inherited parameter",
          text.count(inherited_why) == 1)

    # ── the compiler's own additions are shown, because they hash into the fingerprint ───────────
    check("the success criterion the compiler attaches to SuccessBonus is printed, and attributed",
          any(ln.strip().startswith("condition") and provenance_of(ln) == "compiler-supplied"
              for ln in param_lines))
    check("the fingerprint the plan will be stamped with is in the report",
          f"plan@{plan.fingerprint()}" in text)

    # ── phase2-decisions §4: the divisor belongs to the NORMALIZED path ──────────────────────────
    # compute_normalized_dense_reward returns compute_dense_reward/12.0, so a parity comparison must
    # be against the UNSCALED fold. The report says so; if that sentence goes, the next person to run
    # a parity check compares the wrong two numbers.
    check("the report states that reward_scale is carried, not folded into the rows",
          "CARRIED, not folded" in text and "UNSCALED" in text)

    # ── horizon: supplied, and honestly absent ───────────────────────────────────────────────────
    _, _, no_h = rendered(horizon=None, compile_horizon=None)
    check("a supplied horizon is printed with the flag it came from",
          "horizon: 64 (from --horizon)" in text)
    check("an absent horizon says NOT SUPPLIED and names the flag that would supply it",
          "NOT SUPPLIED" in no_h and "--horizon" in no_h)
    check("...and never invents a number in its place", "horizon: 64" not in no_h)

    # ── terminate_on_success is echoed verbatim, including `unknown` ─────────────────────────────
    # `unknown` is a real answer (it selects compile's conservative branch), so the report must be
    # able to print it as one rather than rendering it as a missing value.
    for answer in ("yes", "no", "unknown"):
        _, _, t = rendered(terminate_on_success=answer)
        check(f"terminate_on_success: {answer} is echoed as given",
              f"terminate_on_success: {answer}" in t)

    # ── format_warnings: a warning computed and never shown is a warning not computed ────────────
    fake = type("P", (), {"warnings": ["first note, about the horizon-integrated ratio",
                                       "second note, about an INCOMPLETE flooding check"]})()
    block = format_warnings(fake)
    check("every warning string survives into the block", all(n in block for n in fake.warnings))
    check("the block counts them, so a reader can tell one was dropped", "warnings (2)" in block)
    check("no warnings renders as an explicit `none`, not as an empty string",
          format_warnings(type("P", (), {"warnings": []})()).strip() == "warnings: none")

    # ── wrap: the indent is the contract, and a long token is never severed ──────────────────────
    long_token = "primitives/descend_to_target/runs/descend-rr-seed20/ckpt_best.pt"
    wrapped = wrap(f"the checkpoint is {long_token} and it must stay readable", "    ")
    check("wrap indents every line it produces",
          all(ln.startswith("    ") for ln in wrapped.splitlines()))
    check("wrap never breaks a path in half — a severed path cannot be copied",
          long_token in wrapped)


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
