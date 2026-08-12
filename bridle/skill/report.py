"""bridle.skill.report — rendering a compiled skill for the author to read.

WHAT THIS IS. `spec.py` parses the document, `compile.py` folds it into a `RewardPlan`, and neither
prints anything. This module is the printer: given a `SkillSpec` and its `RewardPlan`, it produces
the text of `bridle skill compile` — every row in fold order, every parameter, and WHERE EACH VALUE
CAME FROM. `cli.py` calls it and does nothing else with the result.

WHY IT IS A MODULE AND NOT THREE HELPERS IN `cli.py`. It lived there, and `cli.py` grew 398 -> 666
lines with ~100 of them being this rendering; no other subcommand carries anything like it, and
`cmd_skill` reads as parse-and-dispatch again with this gone (2026-08-13 review, finding 8). The
other half is testability: these functions take a `SkillSpec` and a `RewardPlan`, not an argparse
namespace, so `test_report.py` can assert on the rendered text without building a CLI invocation.
The one CLI-shaped string that survives is `(from --horizon)`, kept verbatim because naming the flag
is the point of the line.

WHY PROVENANCE IS PRINTED AT ALL. The reader is a local 27-30B model, and the measured failure mode
of LLM-authored rewards is bad WEIGHTS, not bad term choice. A weight the chassis supplied trains
exactly as hard as one the author typed, so nothing the author did not write may be invisible to
them: every parameter line says `authored`, `chassis '<name>' default`, `term default` or
`compiler-supplied`, and an inherited row prints the rationale that came with it.

Stdlib only, like the rest of `bridle` core.
"""
import textwrap

__all__ = ["format_plan", "format_warnings", "wrap"]

#: Sentinel for "this op does not carry that parameter", distinct from a parameter carried with the
#: value `None` — `axes`, `gate` and `enabled_if` are all legitimately None, and rendering those two
#: cases the same way would tell the author a value was dropped when it was set to "off".
_MISSING = object()


def wrap(text, indent):
    return textwrap.fill(text, width=98, initial_indent=indent, subsequent_indent=indent,
                         break_long_words=False, break_on_hyphens=False)


def format_warnings(plan):
    """`RewardPlan.warnings` printed in full, always.

    compile_spec also emits each note through the `warnings` module; the CLI silences that channel
    and prints the field instead, because `UserWarning: ... (compile.py:704)` buries the text in
    interpreter furniture and prints once per process. These notes carry the horizon-integrated ratio
    and the "flooding check INCOMPLETE ... this is not a pass" line — a warning computed and never
    shown is the same as not computing it.
    """
    if not plan.warnings:
        return "\nwarnings: none"
    out = [f"\nwarnings ({len(plan.warnings)}) — computed and printed, never dropped:"]
    for i, note in enumerate(plan.warnings, 1):
        out.append(textwrap.fill(note, width=98, initial_indent=f"  {i}. ",
                                 subsequent_indent="     ", break_long_words=False,
                                 break_on_hyphens=False))
    return "\n".join(out)


def _provenance(name, authored, inherited, chassis):
    """Where this parameter's value came from: the author, the chassis, or the term's own default.

    Recomputed with `spec.py`'s OWN `_chassis_defaults_for` rather than a second reimplementation of
    the inheritance rule — a chassis may instantiate one term twice under suffixed keys (carry has
    `DistancePull_xy` at k=4.0 and `DistancePull_height` at k=6.0) and a provenance report that
    disagreed with the parser about which row was inherited would be worse than none.
    """
    if name in authored:
        return "authored", None
    if inherited.get(name) is not None:
        return f"chassis {chassis.name!r} default", inherited.get("why")
    return "term default", None


def _row_lines(index, raw, row, op, chassis, terms, chassis_defaults_for):
    out = []
    label = row.term or ("expr" if row.expr is not None else "custom")
    scope = f"/{op.scope}" if op.scope else ""
    # Pad the COMBINED kind+scope, not the scope alone. Padding only the scope put `add` (no scope,
    # 12 blank columns) and `replace/preceding` (10 columns of scope, then 12 more) four columns
    # apart, so the term names in a fold that mixes them did not line up — and the fold's shape is
    # what this listing exists to show. 18 = len("replace/preceding") + 1: `mode` is
    # choices=("add", "replace", "floor") (vocab.py) and `_SCOPE_REACH` (compile.py) holds exactly
    # one scope, so that is the widest pair either can produce today. A wider one would only push its
    # own label right — nothing is truncated — so this is a layout constant, not a limit.
    out.append(f"  [{index}] {op.kind + scope:<18} {label}")
    out.append(wrap(f"why: {row.why}", "        "))

    if row.term is None:
        # Tiers 2 and 3. The custom row is printed with its opacity stated: the fingerprint records
        # that part of this reward cannot be read from the document.
        if row.expr is not None:
            out.append(f"        expr     = {op.params['expr'].source!r}   authored")
            bindings = dict(op.params["bindings"])
            if bindings:
                out.append(f"        bindings = {bindings}   bound from `params:`")
        else:
            out.append(f"        custom   = {op.params['target']!r}   authored — TIER 3, opaque: "
                       f"only the adapter can call it, and nothing here can check what it returns")
        return out

    authored = {k: v for k, v in raw.items() if k not in ("term", "why") and v is not None}
    inherited, _unchosen = chassis_defaults_for(chassis, row.term, authored)
    said_why = False
    for param in terms[row.term].params:
        if param.name not in row.params:
            continue
        spec_value = row.params[param.name]
        op_value = op.params.get(param.name, _MISSING)
        value = spec_value if op_value is _MISSING else op_value
        source, why = _provenance(param.name, authored, inherited, chassis)
        note = ""
        if isinstance(spec_value, str) and spec_value.startswith("params."):
            note = f" (written `{spec_value}`)"     # the bound number AND what the author typed
        if op_value is _MISSING:
            note += "  [consumed by the fold: it became this op's kind/scope]"
        out.append(f"        {param.name:<14} = {value!r:<22} {source}{note}")
        if why and not said_why:
            # ONE `why` per inherited row, not one per parameter: a chassis default is a whole row
            # (`DistancePull_xy` is weight 1.5 AND measure object_to_goal_xy AND k=4.0, and the
            # rationale is why those numbers go together), so repeating it under each field would
            # bury the rest of the plan in four copies of one paragraph.
            out.append(wrap(f"why: {why}", " " * 25))
            said_why = True
        elif source == "term default" and param.doc:
            out.append(wrap(f"doc: {param.doc}", " " * 25))

    for extra in sorted(set(op.params) - set(row.params)):
        # Parameters the COMPILER added, named as such: the success criterion the replace row reads,
        # and the per-row state buffer a stateful term needs. Both hash into the fingerprint.
        out.append(f"        {extra:<14} = {op.params[extra]!r:<22} compiler-supplied")
    return out


def format_plan(doc, spec, plan, *, horizon, terminate_on_success):
    """The resolved plan, with every chassis-supplied default and its rationale shown.

    `doc` is the raw parsed mapping, needed because `spec.reward` no longer distinguishes a value the
    author typed from one the chassis supplied — that distinction is the whole point of the report,
    and only the raw rows still carry it.

    `horizon` and `terminate_on_success` are the CLI's own arguments, passed as values rather than
    read off an argparse namespace so that this function can be called (and tested) without one.
    """
    from bridle.skill.spec import _chassis_defaults_for
    from bridle.skill.vocab import CHASSIS, TERMS

    chassis = CHASSIS[spec.kind]
    raw_rows = doc.get("reward") or []
    scale = (f"{plan.scale} — CARRIED, not folded into the rows below: "
             f"compute_normalized_dense_reward returns compute_dense_reward/divisor, so the divisor "
             f"belongs to the normalized path and the fold printed here is the UNSCALED one")
    out = [
        f"skill:    {spec.name}",
        f"chassis:  {spec.kind}",
        f"contract: {spec.contract}",
        f"env_id:   {spec.env_id}",
        f"plan@{plan.fingerprint()} — {len(plan.ops)} ops, {len(plan.measures_needed)} measures, "
        f"{len(plan.state_slots)} state slot(s)",
        wrap(f"reward_scale: divisor {scale}", "  "),
        "",
        "reward fold — applied in DOCUMENT ORDER as acc = op(acc). This is a fold, not a sum: a",
        "`replace` row overwrites everything above it, so row order is part of the reward.",
        "",
    ]
    for i, (row, op) in enumerate(zip(spec.reward, plan.ops)):
        raw = raw_rows[i] if i < len(raw_rows) and isinstance(raw_rows[i], dict) else {}
        out += _row_lines(i, raw, row, op, chassis, TERMS, _chassis_defaults_for)
        out.append("")

    out += [
        wrap(f"success: {spec.success}", "  "),
        wrap(f"measures the env must supply ({len(plan.measures_needed)}): "
             f"{', '.join(sorted(plan.measures_needed))}", "  "),
        f"  state slots ({len(plan.state_slots)}): "
        f"{', '.join(plan.state_slots) if plan.state_slots else 'none'}",
        # `--horizon` is named literally: the reader is being told which flag to pass, so the CLI
        # spelling belongs in the string even though this module is not the CLI.
        f"  horizon: " + (f"{horizon} (from --horizon)" if horizon else
                          "NOT SUPPLIED — pass --horizon <max_episode_steps>; without it the "
                          "horizon-integrated check below reports that it could not be computed"),
        f"  terminate_on_success: {terminate_on_success}",
        format_warnings(plan),
    ]
    return "\n".join(out)
