"""Unit test for bridle.skill.diagnose — the third feedback tier, after the GPU has spoken.

WHY THE TAGS ARE WHAT IS TESTED HERE. Every LLM-authored-reward system measures one-shot authoring
as the losing configuration: 58.3% +/- 47.3% one-shot against 97.6% with a few refinement rounds
(2605.28918, 10 seeds) — and the ablation says the TYPED CONTENT of the feedback carries the gain,
not the loop: stripping the diagnostic tags collapses it to 11.5%. So the assertions below are about
the tag, the row it names, and whether the message says what to do — not about the plumbing.

THE OTHER HALF, AND IT IS THE HALF THAT IS EASY TO GET WRONG: a diagnostic that fires on a reward
which is WORKING is worse than none, because it teaches the author to ignore the block. phase2-
decisions §1 records the measurement — deployed `descend_to_target` earns 5.0/step of shaping against
a success value of 12.0, integrates to ~27x over its 64-step horizon, and is measured at 0.85 success,
so the horizon-integrated gate that would have condemned it was demoted to a warning. The same rule
binds here: `flooding_stats_at_high_success` below feeds the WORST composition in this file to
`diagnose` at 0.86 success and demands silence.

Run: PYTHONPATH=. python bridle/tests/test_diagnose.py     (the project venv has no pytest)
     python -m pytest bridle/tests/test_diagnose.py
"""
import sys

from bridle.skill.diagnose import TAGS, WHOLE_REWARD, Diagnostic, diagnose, format_diagnostics

FAILS = []


def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        FAILS.append(name)


def stats(**kw):
    """Exactly the keys a training loop actually logged, and nothing else.

    Deliberately does NOT fill in a missing `min`/`max` from the mean: "constant across the rollout"
    and "variation was never logged" are different facts, and a helper that quietly conflated them
    would make the `incomplete` checks below untestable.
    """
    return dict(kw)


def tags(diagnostics):
    return sorted(d.tag for d in diagnostics)


def rows_tagged(diagnostics, tag):
    return sorted(d.row for d in diagnostics if d.tag == tag)


# ── fixtures ────────────────────────────────────────────────────────────────────────────────────

def healthy_stats():
    """Per-step term contributions from a reward that WORKS: deployed `descend_to_target`, whose
    rows are 1.0 (hold) + 1.5 (xy pull) + 2.5 (hover pull) of shaping, two penalties, and a 12.0
    SuccessBonus paid on the terminal step of a 64-step episode. Measured at 0.85 success.

    The means are per-step and plausible rather than logged — no such per-term log exists yet, which
    is what Amendment B2 (this module) is for. What IS load-bearing and taken from the deployed env:
    every row VARIES over the rollout, no shaping row is close to its own maximum, and the run
    succeeds. Those three facts are what must produce silence.
    """
    return {
        "reward[0] PredicateBonus grasped": stats(min=0.0, mean=0.87, max=1.0),
        "reward[1] DistancePull xy": stats(min=0.02, mean=0.71, max=1.5),
        "reward[2] DistancePull hover": stats(min=0.05, mean=1.16, max=2.5),
        "reward[3] HingePenalty crush": stats(min=-0.42, mean=-0.03, max=0.0),
        "reward[4] HingePenalty grip": stats(min=-0.31, mean=-0.02, max=0.0),
        "reward[5] ActionPenalty": stats(min=-0.004, mean=-0.0012, max=0.0),
        "reward[6] SuccessBonus": stats(min=0.0, mean=0.19, max=12.0),
    }


def flooding_stats():
    """One row pays 84 of the 85 the policy earns per step — Amendment B2's own worked example
    ("term 5 is 84% of return -> flooding"). Both rows are far from their own maxima (85 collected
    of 210 available), so this is a COMPOSITION problem and not a saturated one."""
    return {"row1": stats(min=0.0, mean=84.0, max=200.0),
            "row2": stats(min=0.0, mean=1.0, max=10.0)}


def hacking_stats():
    """No row dominates (largest share 35%), but the policy has collected 94% of every point this
    reward has to offer and still finishes 1% of episodes. Shaping is maxed out and the task is not
    done: the reward's optimum is not the task's optimum."""
    return {"pull_xy": stats(min=0.2, mean=0.98, max=1.0),
            "pull_z": stats(min=0.1, mean=0.95, max=1.0),
            "hold": stats(min=0.0, mean=0.90, max=1.0)}


def run_checks():
    # ── the four decisive cases from the brief ──────────────────────────────────────────────────
    constant = diagnose({"row2": stats(min=1.0, mean=1.0, max=1.0)}, 0.3, 60)
    check("a constant term is tagged unoptimizable (tag `constant`)",
          any(d.tag == "constant" for d in constant))

    flooding = diagnose(flooding_stats(), 0.02, 60)
    check("a term dominating what the policy earns is tagged flooding",
          any(d.tag == "flooding" for d in flooding))

    hacking = diagnose(hacking_stats(), 0.01, 64)
    check("shaping collected to saturation with near-zero success is tagged hacking",
          any(d.tag == "hacking" for d in hacking))

    check("a healthy run produces no diagnostics at all",
          diagnose(healthy_stats(), 0.86, 55) == [])

    # ── the tags are the product: each one names a row and says what to do ──────────────────────
    # Not "is non-empty" — the ablation number this module exists for (11.5% with the tags stripped)
    # is about the CONTENT. A tag with no address and no imperative is the stripped condition.
    every = (constant + flooding + hacking
             + diagnose({"a": stats(min=0.0, mean=0.0, max=0.0)}, 0.1, 40)
             + diagnose({"a": stats(min=-3.0, mean=-1.0, max=0.0)}, 0.1, 40)
             + diagnose({"a": stats(mean=1.0)}, 0.02, 40)
             + diagnose({}, 0.02, 40))
    # The addresses every call above could legitimately produce: a key of the mapping it was handed,
    # or the whole-fold address. Checked as MEMBERSHIP and not merely "a non-empty string", which is
    # what the first version of this check did while its label claimed otherwise — an address the
    # author cannot look up in their own document is the same failure as no address.
    addresses = ({"row1", "row2", "pull_xy", "pull_z", "hold", "a"} | {WHOLE_REWARD})
    check("every diagnostic carries a tag from the declared set",
          every and all(d.tag in TAGS for d in every))
    check("every diagnostic addresses a row the author can look up, or the whole fold",
          every and all(d.row in addresses for d in every))

    # SPLIT BY WHAT THE ADDRESS IS, because one check could not bite on both halves (2026-08-13
    # review, finding 1). `d.row in d.message` is a real test for a row named `pull_xy`, and no test
    # at all for the whole-fold rows, whose address is the bare English word "reward": both
    # whole-fold messages were rewritten to address nothing while leaving "reward" in the prose as an
    # ordinary noun ("check your reward document") and the single combined check reported 0 failures.
    row_addressed = [d for d in every if d.row != WHOLE_REWARD]
    whole_fold = [d for d in every if d.row == WHOLE_REWARD]
    check("every row-addressed diagnostic repeats that row's own name in its message",
          row_addressed and all(d.row in d.message for d in row_addressed))
    # The phrase is spelled out here rather than imported from diagnose.py: importing it would make
    # this check follow a rename instead of catching one, and the point is that the whole-fold
    # messages say WHICH THING they are about in words the author reads, not that two constants match.
    check("every whole-fold diagnostic says `the reward fold as a whole` in prose — the bare word "
          "`reward` is satisfiable by accident, so it is not what is demanded here",
          whole_fold and all("the reward fold as a whole" in d.message for d in whole_fold))

    check("every diagnostic says what to DO about it, not just that it happened",
          every and all(prescribes(d.message) for d in every))
    check("every diagnostic quotes at least one number rather than being pure prose",
          every and all(any(ch.isdigit() for ch in d.message) for d in every))

    # ── a working reward is never diagnosed (phase2-decisions §1, the 0.85 measurement) ─────────
    # The composition-level tags are conditioned on the run FAILING. This is the check that stops
    # this module repeating the mistake the horizon-integrated gate made: a rule that condemns the
    # deployed descend lineage (~27x integrated, 0.85 success) is measurably wrong, not strict.
    check("the WORST composition here is silent when the run is actually succeeding",
          diagnose(flooding_stats(), 0.86, 60) == [])
    check("...and so is the saturated one",
          diagnose(hacking_stats(), 0.86, 64) == [])

    # ── flooding and hacking are different pathologies, not two names for one ───────────────────
    # If either collapsed into the other, the typed feedback would be a single undifferentiated
    # "your reward is bad" — which is the 11.5% condition wearing a tag.
    check("flooding fires WITHOUT hacking when the dominant row is far from its own maximum",
          tags(flooding) == ["flooding"])
    check("hacking fires WITHOUT flooding when the rows are balanced but collected to saturation",
          tags(hacking) == ["hacking"])
    check("flooding names the dominant row, not the small one", rows_tagged(flooding, "flooding") == ["row1"])
    check("hacking names the largest earner", rows_tagged(hacking, "hacking") == ["pull_xy"])

    # ── structural findings hold whatever the success rate ──────────────────────────────────────
    # A constant row is unoptimizable at 0.95 success exactly as at 0.05: it adds the same number to
    # every state's return, so it moves no advantage. Suppressing it on a good run would hide a free
    # deletion behind a number that has nothing to do with it.
    check("a constant row is still reported on a SUCCEEDING run",
          any(d.tag == "constant"
              for d in diagnose({"r": stats(min=2.0, mean=2.0, max=2.0),
                                 "v": stats(min=0.0, mean=0.4, max=1.0)}, 0.95, 50)))
    check("an all-zero row is tagged `dead`, not `constant` — the fix is a different one",
          tags(diagnose({"a": stats(min=0.0, mean=0.0, max=0.0)}, 0.9, 40)) == ["dead"])
    check("a constant row is NOT also accused of flooding — a constant cannot be farmed",
          "flooding" not in tags(constant))

    # ── no varying positive shaping at all ──────────────────────────────────────────────────────
    check("a failing run whose every row is a penalty is tagged sparse, addressed to the whole fold",
          tags(diagnose({"pen": stats(min=-3.0, mean=-1.0, max=0.0)}, 0.1, 40)) == ["sparse"]
          and rows_tagged(diagnose({"pen": stats(min=-3.0, mean=-1.0, max=0.0)}, 0.1, 40),
                          "sparse") == [WHOLE_REWARD])
    check("a SUCCEEDING run of penalties is not called sparse — it evidently found the goal",
          diagnose({"pen": stats(min=-3.0, mean=-1.0, max=0.0)}, 0.9, 40) == [])
    check("the only positive row being constant also counts as no shaping to climb",
          "sparse" in tags(constant))

    # ── not-checked must never render as checked ────────────────────────────────────────────────
    partial = diagnose({"row1": stats(mean=1.0)}, 0.02, 40)
    check("a row that logged no min/max is tagged incomplete", "incomplete" in tags(partial))
    check("...and is NOT claimed constant on the strength of a mean alone",
          "constant" not in tags(partial) and "dead" not in tags(partial))
    check("...and is NOT claimed sparse either — its variation is unknown, not absent",
          "sparse" not in tags(partial))

    # ── an unlogged SPREAD withholds the spread verdicts and ONLY those (finding 2) ───────────────
    # The withholding above stops at constant/dead/sparse, and stopping there left the other half of
    # the boundary unpinned: this row IS accused of flooding. That is the intended semantics — the
    # split is by which number a verdict reads, and flooding reads the mean, which was logged (see
    # the comment on `farmable` in diagnose.py). Pinned as exact equality, in both directions, so
    # moving the boundary either way fails here instead of changing the feedback silently.
    check("...and IS still tagged flooding — the spread was never logged, but the mean was read",
          "flooding" in tags(partial))
    check("...and exactly those two, nothing further inferred from the mean alone",
          tags(partial) == ["flooding", "incomplete"])
    check("a run that logged no per-term stats at all is incomplete, not healthy",
          tags(diagnose({}, 0.02, 40)) == ["incomplete"])
    check("...even when the run succeeded — an empty result would read as a clean bill of health",
          tags(diagnose({}, 0.93, 40)) == ["incomplete"])
    # The old label here read "...rather than folded into a verdict", which its body never checked
    # and which was not even true — flooding fires on this row, on purpose. Same boundary as
    # `partial` above, reached by the other unreadable-spread route (a logging bug, not a missing
    # key), and pinned the same exact way.
    check("min above max makes the SPREAD unreadable — incomplete, plus the mean-based verdict, "
          "and no spread verdict",
          tags(diagnose({"r": stats(min=5.0, mean=1.0, max=0.0)}, 0.02, 40))
          == ["flooding", "incomplete"])
    # `== ["incomplete"]` and not `in`: with nothing readable, "there is no shaping to climb" is a
    # claim the data does not support, so `sparse` must NOT fire alongside it.
    check("a run whose every mean is unreadable is ONLY incomplete — no verdict is inferred",
          tags(diagnose({"r": stats(min=0.0, mean=None, max=1.0)}, 0.02, 40)) == ["incomplete"])
    # Found by self-review, and it CRASHED (`TypeError: unsupported format string passed to
    # NoneType.__format__`): min == max makes the row non-varying, which reached the `constant`
    # branch, which printed the mean it did not have. A diagnostics function that raises at the end
    # of a multi-hour training run destroys exactly the feedback it exists to produce, so it does
    # not get to have an unhandled path.
    check("a non-varying row whose mean was never logged is incomplete, not a verdict built on None",
          tags(diagnose({"r": stats(min=5.0, max=5.0)}, 0.02, 40)) == ["incomplete"])

    # ── the caller's own mistakes are refused, not silently absorbed ─────────────────────────────
    # 86 instead of 0.86 is the live hazard: every composition rule is conditioned on the run
    # failing, so a percentage passed as a fraction would suppress ALL of them and report perfect
    # health for a catastrophic run.
    check("a success rate outside [0, 1] is refused, not read as a very successful run",
          refuses(lambda: diagnose(flooding_stats(), 86.0, 60)))
    check("a negative success rate is refused", refuses(lambda: diagnose({}, -0.1, 60)))
    check("a non-positive episode length is refused", refuses(lambda: diagnose({}, 0.5, 0)))
    check("a term_stats that is not a mapping is refused", refuses(lambda: diagnose([], 0.5, 60)))
    check("success_rate=0.0 and 1.0 are legal, not off-by-one refusals",
          not refuses(lambda: diagnose({}, 0.0, 60)) and not refuses(lambda: diagnose({}, 1.0, 60)))

    # ── success_rate=None: MEASURED ROWS, UNMEASURED OUTCOME ─────────────────────────────────────
    # The caller is real: `bridle.adapters.skill_telemetry` emits per window of control steps, and a
    # window in which no episode ended has every row's min/mean/max and no outcome at all. The two
    # wrong answers are (a) refuse, so the free structural findings are lost, and (b) substitute 0.0,
    # which licenses every composition rule on no evidence. Neither is taken.
    none_flood = diagnose(flooding_stats(), None, 60)
    check("an unmeasured success rate does NOT license the composition tags",
          "flooding" not in tags(none_flood) and "hacking" not in tags(none_flood)
          and "sparse" not in tags(none_flood))
    check("...it is reported as incomplete against the whole fold, so 'not checked' is not 'clean'",
          [(d.tag, d.row) for d in none_flood] == [("incomplete", WHOLE_REWARD)])
    check("...and the message names the missing input and what supplying it would buy",
          "success rate" in none_flood[0].message and "flooding" in none_flood[0].message
          and "0.0 would license" in none_flood[0].message)
    # The structural half is a statement about the ROW and holds at any success rate — including at
    # none. Losing it was the whole cost of refusing.
    none_const = diagnose({"a": stats(min=2.0, mean=2.0, max=2.0),
                           "b": stats(min=0.0, mean=0.0, max=0.0)}, None)
    check("the structural tags DO still run with no success rate — they are free and row-local",
          tags(none_const) == ["constant", "dead", "incomplete"]
          and rows_tagged(none_const, "constant") == ["a"]
          and rows_tagged(none_const, "dead") == ["b"])
    check("an EMPTY term_stats with no rate reports the stronger 'nothing was logged', once",
          [(d.tag, d.row) for d in diagnose({}, None)] == [("incomplete", WHOLE_REWARD)]
          and "no per-term contributions were logged" in diagnose({}, None)[0].message)
    check("None is the ONLY non-fraction accepted — a string is still a caller bug",
          refuses(lambda: diagnose(flooding_stats(), "unknown", 60)))

    # ── ep_len and horizon are both CONTEXT, so they are optional the same way (finding 6) ───────
    # ep_len was mandatory while doing strictly less than the optional horizon: it reaches exactly
    # one message fragment and no rule reads it, so a caller with no measured mean episode length had
    # to invent one — the substitution the "a rule this module cannot compute must say so" principle
    # exists to prevent. Resolved by making ep_len optional, not by making horizon mandatory.
    check("the call is legal with ep_len left out altogether",
          not refuses(lambda: diagnose(hacking_stats(), 0.01)))
    no_len = diagnose(hacking_stats(), 0.01)
    check("omitting ep_len leaves this run's verdict identical — no rule reads it",
          tags(no_len) == ["hacking"] and tags(no_len) == tags(hacking))
    check("...the per-step numbers survive and only the per-episode total is absent",
          "per step" in no_len[0].message and "-step episode" not in no_len[0].message)
    check("...and supplying ep_len is what adds that per-episode total back",
          "-step episode" in diagnose(hacking_stats(), 0.01, 64)[0].message)
    check("omitting BOTH leaves no horizon clause either",
          "horizon of" not in no_len[0].message)
    check("...and supplying horizon alone states it without fabricating an episode length",
          "horizon of 64" in diagnose(hacking_stats(), 0.01, horizon=64)[0].message
          and "-step episode" not in diagnose(hacking_stats(), 0.01, horizon=64)[0].message)
    check("ep_len=None and horizon=None are legal — 'not available' is an answer",
          not refuses(lambda: diagnose(hacking_stats(), 0.01, None, horizon=None)))
    # Absent and nonsense are different facts, and only the first is honest. A supplied value still
    # has to be a positive number, for BOTH arguments — a `0` horizon used to be silently swallowed
    # by an `if horizon:` truthiness test, which is a default by another name.
    for label, bad in (("ep_len", 0), ("ep_len", -3), ("ep_len", "60"), ("ep_len", True)):
        check(f"a supplied {label}={bad!r} is refused, not read as absent",
              refuses(lambda b=bad: diagnose(hacking_stats(), 0.01, b)))
    for bad in (0, -3, "64", True):
        check(f"a supplied horizon={bad!r} is refused, not read as absent",
              refuses(lambda b=bad: diagnose(hacking_stats(), 0.01, 64, horizon=b)))

    # ── the output is stable enough to diff between refinement rounds ───────────────────────────
    mixed = {"row1": stats(min=0.0, mean=84.0, max=200.0),
             "row2": stats(min=1.0, mean=1.0, max=1.0),
             "row3": stats(mean=0.5)}
    first, second = diagnose(mixed, 0.02, 60), diagnose(mixed, 0.02, 60)
    check("two calls on the same input return the same list, in the same order", first == second)
    order = [TAGS.index(d.tag) for d in first]
    check("diagnostics come back in the declared tag order", order == sorted(order))
    check("a Diagnostic is frozen — the block handed to the author cannot be edited after the fact",
          frozen(first[0]))

    # ── the rendered block, which is the thing the author actually reads ────────────────────────
    text = format_diagnostics(first)
    check("the rendered block carries every tag through to the text",
          all(f"[{d.tag}]" in text for d in first))
    check("the rendered block carries every row address through to the text",
          all(d.row in text for d in first))
    check("an empty result renders as an explicit statement, not an empty string",
          format_diagnostics([]).strip() != "" and "no diagnostic" in format_diagnostics([]).lower())


#: Sentence-initial imperatives. The first version of `prescribes` searched the WHOLE message,
#: lower-cased, for any of these — and passed a message whose only remaining "advice" was the noun
#: phrase "or a gate that is not true on every step", because `"gate "` appears in it. A check that
#: a mutation cannot break is not a check: this one now demands an actual instruction, at the start
#: of a sentence, which is what "says what to do" means.
_IMPERATIVES = ("Delete", "Lower", "Check", "Move", "Add ", "Log ", "Raise", "Tighten", "Supply",
                "Write", "Set ", "Gate ")


def prescribes(message):
    return any(s.strip().startswith(_IMPERATIVES)
               for s in message.replace("\n", " ").split(". "))


def refuses(call):
    try:
        call()
        return False
    except (ValueError, TypeError):
        return True


def frozen(diagnostic):
    try:
        diagnostic.tag = "mutated"
        return False
    except Exception:
        return isinstance(diagnostic, Diagnostic)


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
