"""bridle.skill.diagnose — the third feedback tier: what the numbers say after a training run.

WHAT THIS IS. `spec.py` checks a row in isolation, `compile.py` checks the rows against each other,
and neither can see what the policy actually did with them. This module reads the per-term
contribution statistics a finished run logged — Eureka's `policy_feedback`: per-component min/mean/max
plus the success rate — and turns them into TYPED diagnostics addressed to the author, which here is a
local 27-30B model that cannot read this file.

WHY THE TAG IS THE PRODUCT AND THE LOOP IS NOT. Every LLM-authored-reward system measures one-shot
authoring as the losing configuration: 58.3% +/- 47.3% one-shot against 97.6% with a few refinement
rounds (2605.28918, 10 seeds) — and one-shot's variance is the real finding, it is bimodal, a coin
flip. The ablation then says the TYPED CONTENT carries the gain, not the iteration: stripping the
diagnostic tags collapses 97.6% to 11.5%. We cannot buy the loop anyway (Eureka searches ~400 full RL
runs per environment on 8xA100, affordable because IsaacGym trains a candidate in minutes; ours take
hours), so the tags are the whole of what we can buy. A `Diagnostic` that does not name the row and
say what to do about it has failed at its job, exactly as an error message that does not name the path
has.

THE RULE THAT SHAPES EVERY THRESHOLD BELOW: **a reward that is succeeding is never diagnosed.** This
module must not repeat the mistake the horizon-integrated flooding gate made. Deployed
`descend_to_target` earns 1.0 + 1.5 + 2.5 = 5.0/step of shaping over a 64-step horizon against a
success value of 12.0 — ~27x integrated — and is measured at 0.85 success (phase2-decisions §1). A
rule that condemns that lineage is measurably wrong, not strict; and a block of warnings that fires on
working rewards is a block the author learns to skip, which costs the 97.6% outright. So every
COMPOSITION diagnostic is conditioned on the run failing. Only the STRUCTURAL ones — a row that never
varied, a row that was never readable — hold at any success rate, because they are statements about
the row itself and their fix is free.

NOT CHECKED IS NOT CLEAN. A row that logged no min/max cannot be judged constant or varying, and
saying nothing about it would render "not checked" as "checked and clean" — the exact shape of
`bridle lineage` printing `0 violation(s)` and exiting 0 on a machine with no `systemctl`. Those rows
get an `incomplete` tag, and so does a run that logged no per-term statistics at all.

Stdlib only, like the rest of `bridle` core: no torch, no numpy. The statistics arrive as plain
numbers from whoever ran the training.
"""
import dataclasses
import textwrap
from collections.abc import Mapping

__all__ = ["Diagnostic", "TAGS", "WHOLE_REWARD", "diagnose", "format_diagnostics"]


#: The address a whole-fold finding carries in `Diagnostic.row`. Same string `compile.py` uses as the
#: `path` of its own whole-fold refusal (`FloodingError("reward", ...)`), so one document-level
#: address means one thing across all three feedback tiers.
WHOLE_REWARD = "reward"

#: How a whole-fold message names its own subject IN PROSE, and the reason it needs a phrase rather
#: than the bare address. `WHOLE_REWARD` is the ordinary English word "reward", which turns up in
#: ordinary advice ("check your reward document") — so a message merely CONTAINING it has not
#: necessarily said what it is about. Both whole-fold messages below open with this phrase, and
#: `test_diagnose.py` asserts the phrase, not the bare word: on 2026-08-13 a reviewer rewrote both
#: messages to address nothing at all, left "reward" in the prose incidentally, and the check that
#: existed to catch exactly that reported 0 failures.
_WHOLE_FOLD_PHRASE = "the reward fold as a whole"

#: Declared once, in the order diagnostics are returned: the composition failures first (they explain
#: the run), then the row-level structural ones, then what could not be checked. Exported so a caller
#: can assert against the set rather than string-matching prose, and so a typo'd tag cannot ship.
TAGS = ("flooding", "hacking", "dead", "constant", "sparse", "incomplete")

#: Below this success rate the run is failing and the composition tags are allowed to speak. A DESIGN
#: CHOICE, not a measurement — named so it can be moved, and every message prints the measured rate so
#: the reader can overrule it. It sits far below anything this project has ever shipped (descend 0.85,
#: pick_place macro 0.89, the round-robin chain 0.979), which is the point: the cost of a false alarm
#: on a working reward is the author ignoring the block.
_FAILING = 0.5

#: A single row paying at least this share of everything the policy earns is paying more than all the
#: other rows combined. Amendment B2's worked example is 84% ("term 5 is 84% of return -> flooding").
_DOMINANT_SHARE = 0.5

#: Collected/available shaping at which the policy has, for practical purposes, finished climbing.
#: Also a design choice; the ratio is printed in the message.
_CAPTURED = 0.9

#: Width below which a row's min..max spread is float-logger noise rather than variation. A term that
#: moves by less than a nanounit over a whole rollout is constant for every purpose PPO has.
_FLAT = 1e-9


@dataclasses.dataclass(frozen=True)
class Diagnostic:
    """One typed finding about a finished run.

    `tag`     one of `TAGS`. The load-bearing field: the 11.5% ablation is what happens when this is
              stripped and only the prose survives.
    `row`     WHICH row — a key of the `term_stats` mapping, or `WHOLE_REWARD` for a finding about the
              fold as a whole. The same kind of address as `SpecError.path` / `CompileError.path`.
    `message` what was measured and WHAT TO DO. Both halves are required; "term 3 looks wrong" is the
              stripped condition wearing a tag.

    Frozen: the block handed to an author is a record of what a run measured, and a caller that can
    edit a tag after the fact can make the record disagree with the run.
    """

    tag: str
    row: str
    message: str

    def __str__(self):
        return f"[{self.tag}] {self.row}: {self.message}"


# ── reading what the training loop logged ───────────────────────────────────────────────────────

@dataclasses.dataclass(frozen=True)
class _Term:
    """One row's logged statistics, with `None` meaning NOT REPORTED — never zero, never inferred.

    `problem` is the human-readable reason this row could not be fully read, or None. It exists so an
    unreadable row produces a diagnostic instead of quietly dropping out of the sums, which is how a
    check comes to pass by not running.
    """

    name: str
    mean: float | None
    lo: float | None
    hi: float | None
    problem: str | None

    @property
    def varies(self):
        """True / False / None-for-unknown. Three-valued on purpose: `False` licenses the `constant`
        tag and excludes the row from the farmable total, `None` licenses neither."""
        if self.lo is None or self.hi is None or self.hi < self.lo:
            return None
        return (self.hi - self.lo) > _FLAT

    @property
    def earns(self):
        return self.mean is not None and self.mean > 0.0


def _number(value):
    """A real number, or None. `bool` is excluded deliberately: `True` is an `int` in Python and a
    mean of `True` is a logging bug, not a contribution of 1.0."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _field(raw, key):
    """min/mean/max out of either a mapping (`{"mean": 1.0}`) or an object exposing the attributes —
    a training loop that logs a small stats dataclass should not have to convert it first."""
    if isinstance(raw, Mapping):
        return raw.get(key)
    return getattr(raw, key, None)


def _read_term(name, raw):
    mean, lo, hi = (_number(_field(raw, k)) for k in ("mean", "min", "max"))
    missing = [k for k, v in (("mean", mean), ("min", lo), ("max", hi)) if v is None]
    problem = None
    if missing:
        problem = ("no readable " + "/".join(f"`{m}`" for m in missing)
                   + " was logged for this row")
    elif hi < lo:
        problem = f"the logged min ({lo!r}) is above the logged max ({hi!r})"
    return _Term(name=name, mean=mean, lo=lo, hi=hi, problem=problem)


def _optional_steps(label, given):
    """A positive number of steps, or None for "not available". Refuses everything else.

    `ep_len` and `horizon` both come through here so they cannot drift apart again: they are context
    for one message fragment each, no rule reads either, and the honest answer when a caller has not
    measured one is to say so rather than to substitute a plausible default.
    """
    if given is None:
        return None
    value = _number(given)
    if value is None or value <= 0:
        raise ValueError(f"{label} is a positive number of steps, or None to say it is not "
                         f"available; got {given!r}. There is no default and none is invented: an "
                         f"absent one costs a fragment of one message, a fabricated one puts a "
                         f"number nobody measured into feedback the author will act on.")
    return value


def _pct(x):
    return f"{100.0 * x:.1f}%"


def _n(x):
    """Four significant figures, enough to recognise a weight in the document it came from."""
    return f"{x:.4g}"


# ── the checks ──────────────────────────────────────────────────────────────────────────────────

def diagnose(term_stats, success_rate, ep_len=None, *, horizon=None):
    """Read a finished run's per-term statistics and return the typed diagnostics, in `TAGS` order.

    `term_stats`   {row name -> {"min": .., "mean": .., "max": ..}} — the per-step contribution of
                   each reward row over the rollout. Names are free text and are echoed verbatim into
                   `Diagnostic.row`, so a caller that labels them `reward[2] DistancePull` gets
                   feedback addressed in the same coordinates the document uses. Missing keys are
                   reported, never inferred.
    `success_rate` fraction in [0, 1], or `None` for NOT MEASURED. Refused outside that: every
                   composition rule is conditioned on the run failing, so a percentage passed by
                   mistake (86.0 for 0.86) would suppress all of them and report perfect health for
                   a catastrophic run. `None` is a different fact from a low rate and is treated as
                   one — the STRUCTURAL rules still run (they are statements about a row, true at any
                   success rate) and an `incomplete` says out loud that the composition tier did not.
                   It exists because the caller that has term statistics and no success rate is real:
                   `bridle.adapters.skill_telemetry` emits per WINDOW of control steps, and a window
                   in which no episode ended has measured every row and no outcome. Reporting the
                   free findings and naming the missing input beats reporting nothing, and beats
                   substituting a 0.0 that would license the whole composition block on no evidence.
    `ep_len`       optional mean episode length in steps, used to state per-episode totals.
    `horizon`      optional `max_episode_steps`.

    `ep_len` and `horizon` are both CONTEXT and are treated identically: each is printed when
    supplied and simply absent when not, and NO RULE is conditioned on either. `ep_len` used to be
    mandatory while doing strictly less than the optional `horizon` — it reaches exactly one message
    fragment — so a caller with no reliable mean episode length had to fabricate one, which is
    precisely the silent substitution the "a rule this module cannot compute must say so" principle
    exists to prevent (2026-08-13 review, finding 6). SUPPLIED, either must be a positive number: a
    zero, a negative or a string is a caller bug and is refused, because absent and nonsense are
    different facts and only the first one is honest.

    Returns `[]` only when every row was readable, every row varied, and nothing about the
    composition is worth saying at this success rate. An empty list from an empty `term_stats` would
    mean "clean", so that case returns an `incomplete` instead — and so, for the same reason, does a
    `success_rate` of None: a report with no verdicts in it must not be readable as a clean bill of
    health when the input that buys the verdicts was never supplied.
    """
    if not isinstance(term_stats, Mapping):
        raise TypeError(f"term_stats is a mapping of row name -> {{min, mean, max}}, got "
                        f"{type(term_stats).__name__}")
    rate = None if success_rate is None else _number(success_rate)
    if success_rate is not None and (rate is None or not 0.0 <= rate <= 1.0):
        raise ValueError(f"success_rate is a fraction in [0, 1], or None for not-measured; got "
                         f"{success_rate!r} — a percentage here would silently suppress every "
                         f"composition check, because they are conditioned on the run failing")
    # Both context arguments, validated the same way — see the docstring.
    length, span = _optional_steps("ep_len", ep_len), _optional_steps("horizon", horizon)

    terms = [_read_term(name, raw) for name, raw in term_stats.items()]
    order = {t.name: i for i, t in enumerate(terms)}
    order[WHOLE_REWARD] = len(terms)

    found = []
    found += _structural(terms)
    found += _unreadable(terms)
    if rate is None:
        # Only when there ARE rows: with none, `_unreadable` has already said the stronger version of
        # this ("no per-term contributions were logged"), and two whole-fold incompletes on one
        # report is noise, not rigour.
        if terms:
            found.append(Diagnostic(
                "incomplete", WHOLE_REWARD,
                f"{_WHOLE_FOLD_PHRASE} was measured but its OUTCOME was not: no success rate came "
                f"with these {len(terms)} row(s), so the composition checks (flooding, hacking, "
                f"sparse) could not run — every one of them is conditioned on the run failing and "
                f"there is nothing here that says whether it did. The structural checks above DID "
                f"run and hold at any success rate. Supply the fraction of episodes that succeeded "
                f"over the same window these statistics cover; a substituted 0.0 would license the "
                f"whole composition block on no evidence at all."))
    elif rate < _FAILING:
        found += _composition(terms, rate, length, span)
    return sorted(found, key=lambda d: (TAGS.index(d.tag), order.get(d.row, len(order))))


def _structural(terms):
    """`dead` and `constant`: statements about the ROW, true at any success rate.

    A constant row adds the same number to every state's return, so it shifts no advantage and moves
    no gradient — it is unoptimizable whether the run succeeds or not, and deleting it is free. It is
    reported on a succeeding run for exactly that reason.
    """
    out = []
    for t in terms:
        # `t.mean is None` and `varies is False` can coexist — min == max with no mean logged — and
        # both branches below quote the mean. A structural verdict printed from a number that was
        # never reported would be inventing it; the row is already reported as `incomplete`.
        if t.varies is not False or t.mean is None:
            continue
        if abs(t.hi) <= _FLAT and abs(t.lo) <= _FLAT and abs(t.mean) <= _FLAT:
            out.append(Diagnostic(
                "dead", t.name,
                f"{t.name} was identically 0.0 for the whole rollout (min = mean = max = 0), so it "
                f"trains, logs, and contributes nothing. Either its gate/predicate never became "
                f"true, or its measure never left the region where the term clamps to zero. This is "
                f"the shape of the crush penalty that went silently zero over an unsigned measure — "
                f"the term whose absence broke 16/16 grasps (2026-06-04). Check the gate, check the "
                f"sign of the measure, check the threshold; or delete the row."))
            continue
        out.append(Diagnostic(
            "constant", t.name,
            f"{t.name} is constant at {_n(t.mean)} across the whole rollout (min = max = "
            f"{_n(t.lo)}), so it is unoptimizable: it adds the same number to every state's return, "
            f"which shifts every advantage equally and changes no gradient. Delete the row, or make "
            f"it depend on something that actually varies — a measure that moves during the episode, "
            f"or a gate that is not true on every step."))
    return out


def _unreadable(terms):
    """`incomplete`: the checks that could NOT run, said out loud."""
    out = [Diagnostic(
        "incomplete", t.name,
        f"{t.name}: {t.problem}, so the constant/dead checks could not run for it and it is only "
        f"partly counted in the composition checks. Not checked is not the same as clean. Log min, "
        f"mean and max for every reward row — 3 numbers per row per rollout — and run this again.")
        for t in terms if t.problem]
    if not terms:
        out.append(Diagnostic(
            "incomplete", WHOLE_REWARD,
            f"{_WHOLE_FOLD_PHRASE} went unmeasured: no per-term contributions were logged for this "
            f"run, so none of the {len(TAGS)} patterns the reward diagnostics check for could be "
            f"ruled out. This empty result is NOT a clean bill of health — it is the absence of the "
            f"measurement. Log the per-step min/mean/max of every reward row alongside the success "
            f"rate."))
    return out


def _composition(terms, rate, length, horizon):
    """`flooding`, `hacking`, `sparse`: statements about the rows AGAINST EACH OTHER, and only
    reachable when the run is failing (see the module docstring's 0.85 measurement)."""
    out = []
    # Farmable = the policy has some choice about it. A constant or dead row is excluded from every
    # rule here: it cannot be farmed instead of finishing the task, because it pays the same whatever
    # the policy does. Calling it "flooding" would be wrong, not merely noisy.
    #
    # A row whose variation was NEVER LOGGED (`varies is None` — no min/max, or min above max) is
    # admitted, deliberately, and the rule is WHICH NUMBER A VERDICT READS. `constant`/`dead` are
    # claims about the SPREAD, so an unlogged spread withholds them. `flooding`/`hacking` are claims
    # about a MEAN against the other means, and those means WERE read — withholding a verdict the
    # data does support would render "not measured" as "nothing to say", which is the exact failure
    # the `incomplete` tag exists to prevent. The unknown is never silent either way: the same row is
    # already carrying an `incomplete` telling the author to log min/max, so a reader sees the
    # accusation and its caveat together. `sparse` obeys the same rule from the other side — it fires
    # only when NO row is farmable, so an unknown-variation earner suppresses it, because its
    # variation is unknown rather than absent.
    #
    # Pinned in test_diagnose.py (`{"a": {"mean": 1.0}}` and the min>max row at 0.02 success are each
    # exactly ["flooding", "incomplete"], nothing more and nothing less) so that flipping this
    # boundary fails a test instead of changing the semantics silently.
    farmable = [t for t in terms if t.earns and t.varies is not False]
    earned = sum(t.mean for t in terms if t.earns)

    if farmable and earned > 0.0:
        for t in sorted(farmable, key=lambda t: -t.mean):
            share = t.mean / earned
            if share < _DOMINANT_SHARE:
                continue
            out.append(Diagnostic(
                "flooding", t.name,
                f"{t.name} pays {_n(t.mean)} per step on average — {_pct(share)} of everything this "
                f"policy earns ({_n(earned)}/step across {len(terms)} rows) — while only "
                f"{_pct(rate)} of episodes succeed. The policy is being paid mostly for this one "
                f"row rather than for finishing the task. Lower its weight, or gate it on the "
                f"predicate it is meant to reward, so that completing the task is the largest thing "
                f"on offer. (compile-time flooding bounds the DECLARED per-step maxima; this one "
                f"measures what was actually collected.)"))

    # Saturation, not magnitude. "High return with low success" cannot be judged against an absolute
    # scale — every skill has its own — but it CAN be judged against the reward's own ceiling: if the
    # policy has already collected nearly every point the shaping has to give and the task is still
    # not done, the reward's optimum is not the task's optimum. That is a different claim from "this
    # run is young": an undertrained policy sits far below the ceiling.
    climbable = [t for t in terms if t.hi is not None and t.hi > 0.0 and t.varies is not False]
    available = sum(t.hi for t in climbable)
    if climbable and available > 0.0 and all(t.mean is not None for t in climbable):
        collected = sum(max(t.mean, 0.0) for t in climbable)
        captured = collected / available
        if captured >= _CAPTURED:
            top = max(climbable, key=lambda t: t.mean)
            # Two independent context clauses, each printed only if its number was supplied. Neither
            # is defaulted; an absent one narrows the sentence rather than inventing a figure.
            context = ""
            if length is not None:
                context += f", i.e. {_n(collected * length)} per {_n(length)}-step episode"
            if horizon is not None:
                context += f" against a horizon of {_n(horizon)}"
            out.append(Diagnostic(
                "hacking", top.name,
                f"the policy has already collected {_pct(captured)} of the shaping this reward has "
                f"to offer ({_n(collected)} of {_n(available)} per step{context}) while only "
                f"{_pct(rate)} of episodes succeed; {top.name} is the largest earner at "
                f"{_n(top.mean)}/step. Shaping is maxed out and the task is still not being done, so "
                f"the reward's optimum is not the task's optimum — this is a mis-specified "
                f"objective, not an undertrained policy, which would sit far below this ceiling. "
                f"Move value out of the shaping rows into the success bonus, or tighten the "
                f"gate/predicate so the shaping cannot be held without making progress."))

    # No varying positive shaping at all. Note the guard on `readable`: with nothing readable we know
    # nothing, and "there is no signal" would be a claim the data does not support.
    readable = [t for t in terms if t.mean is not None]
    if readable and not farmable:
        out.append(Diagnostic(
            "sparse", WHOLE_REWARD,
            f"{_WHOLE_FOLD_PHRASE} offers no varying positive shaping — not one of its "
            f"{len(terms)} row(s) ({', '.join(t.name for t in terms)}) both varies and pays "
            f"positively, and only {_pct(rate)} of episodes succeed. There "
            f"is nothing for the policy to climb — it has to find the success bonus by chance. The "
            f"fully sparse alternative was measured at 178M from-scratch steps at 0% success "
            f"(move_to_target_env.py:199-203). Add one shaping row over a measure that improves as "
            f"the task is done, or check whether the gates on the rows you have ever become true."))
    return out


# ── rendering ───────────────────────────────────────────────────────────────────────────────────

def format_diagnostics(diagnostics, width=96):
    """The block handed back to the author between refinement rounds.

    The tag is printed in brackets ahead of the address, unwrapped, because the tag and the address
    are the two fields the ablation says carry the gain and neither may be lost to a line break.
    """
    if not diagnostics:
        return ("no diagnostics — every reward row varied, and nothing about the composition is "
                "worth saying at this success rate. That is the absence of the patterns this checks "
                "for, not a guarantee that the reward is a good one.")
    head = f"{len(diagnostics)} diagnostic(s) from this run, in severity order:"
    body = []
    for d in diagnostics:
        body.append(f"\n  [{d.tag}] {d.row}")
        body.append(textwrap.fill(d.message, width=width, initial_indent="      ",
                                  subsequent_indent="      ", break_long_words=False,
                                  break_on_hyphens=False))
    return head + "\n" + "\n".join(body)
