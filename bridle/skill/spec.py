"""bridle.skill.spec — the skill document and its schema.

WHAT THIS IS: the `--schema-->` arrow of `skill.yaml --schema--> SkillSpec --compile()--> RewardPlan
--> ManiSkill env`. `parse_spec(doc)` takes the mapping a YAML loader produced and returns a frozen
`SkillSpec`, or raises `SpecError`. It never touches YAML itself (loading is the CLI's problem) and
never imports torch/numpy — `bridle` core is stdlib-only, so the whole schema is testable on CPU in
milliseconds.

THE ERROR MESSAGES ARE THE API. The intended author of a skill document is a local 27-30B LLM that
cannot introspect this module, so a refusal it cannot act on is a refusal that costs a round trip —
and the measured cost of authoring with no usable feedback is 58.3% +/- 47.3% success one-shot
against 97.6% with refinement rounds, with the ablation showing the TYPED CONTENT of the feedback
carrying the gain (strip the diagnostic tags and it collapses to 11.5%; amendment 1 §B). Hence every
`SpecError` carries three things, per design doc §8: the dotted PATH, the LEGAL SET, and a
nearest-match SUGGESTION (`difflib`).

WHAT THIS MODULE REFUSES, AND WHY EACH ONE WAS PAID FOR:

  `why` IS MANDATORY on every reward row (amendment 1 §B3). L2R went 50% -> 90% by making the model
  state the behaviour in prose before emitting the constrained call, and design doc §10 notes that
  YAML drops the source comments — so the `why` is the only surviving record of why a weight is what
  it is. A row without one is refused, not defaulted from the chassis: inheriting someone else's
  rationale is exactly the failure this rule exists to prevent.

  A TERM THAT NEEDS A SIGNED MEASURE REJECTS A MAGNITUDE ONE. `HingePenalty` computes
  `clamp(signed_delta, min=0)`; over an unsigned measure that is identically zero, so the row still
  trains, still logs, and silently contributes nothing. The crush penalty it deletes exists because
  pressing the cube to dz=0 broke 16/16 grasps (2026-06-04, descend_env.py).

  A QUANTITY THAT EXISTS IN MORE THAN ONE FRAME MUST BE NAMED WITH ITS FRAME. descend's reward
  grades the seat height against the LIVE seat while descend_stack's success gate grades a goal
  frozen at init — one quantity, two truths (vocab.py, correction 2). Both frames therefore carry
  their frame in the MEASURES key and the bare `height_above_seat` is refused: the author writes
  `height_above_seat_live` or `height_above_seat_static_goal`. The legal measure names ARE the
  MEASURES keys — no alias layer, one validation path for what an author wrote and what a chassis
  default supplied, and a `SkillSpec` re-emitted as YAML re-parses to the same spec.

  A `params.X` MUST RESOLVE. Per-skill params (§5) are how a skill declares the 69 corpus parameters
  `Contract` has no field for; a reference to an undeclared one is a silent zero at compile time.

WHAT THIS MODULE DOES NOT DO — it is the schema tier, not the compiler (design doc §8 has three
tiers of feedback: schema -> compile -> preflight):
  - it does not check weight RELATIONSHIPS (shaping maxima below the success value, an attractor
    that must not peak at contact). Those need the horizon and the termination rule: `compile()`.
  - it does not parse the `success:` criterion's grammar (`all[...]`, predicate calls) beyond its
    `params.X` references. That grammar belongs to whoever evaluates it.
  - it does not resolve `contract:` to a real `Contract`, or `env_id` to a registered env.
"""
import dataclasses
import difflib
import re
from types import MappingProxyType

from bridle.preflight import DYNAMIC, STATIC
from bridle.resolve import ADAPT, RETRAIN, RUN
from bridle.skill.expr import Expr, ExprError
from bridle.skill.expr import parse as parse_expr
from bridle.skill.vocab import CHASSIS, MEASURES, PREDICATES, TERMS, Frame, Sign, base_term

__all__ = [
    "SpecError", "Row", "SkillSpec", "parse_spec", "json_schema",
    "ROW_TERMS", "LEGAL_MEASURE_NAMES",
]


# ── the error ────────────────────────────────────────────────────────────────────────────────────

class SpecError(Exception):
    """One refusal, addressed to a model that cannot read this file.

    `path` is the dotted location in the document (`reward[2].measure`, `params.hover.severity`) so
    a caller — or an author — can point at the offending field without diffing two documents.
    `suggestion` is the nearest legal spelling, or None when nothing is close enough to be worth
    guessing; the message states the legal set either way, because "did you mean X?" with no set is
    unrecoverable when the guess is wrong.
    """

    def __init__(self, path, problem, legal=None, suggestion=None):
        self.path = path
        self.suggestion = suggestion
        message = f"{path}: {problem}"
        if suggestion is not None:
            message += f" — did you mean {suggestion!r}?"
        if legal:
            message += f" — legal values: {', '.join(sorted(legal))}"
        super().__init__(message)


def _suggest(value, candidates):
    if not isinstance(value, str):
        return None
    close = difflib.get_close_matches(value, sorted(candidates), n=1, cutoff=0.6)
    return close[0] if close else None


def _unknown(path, what, value, candidates):
    """The standard shape of an unknown-name refusal: what was written, the nearest match, the set."""
    return SpecError(path, f"unknown {what} {value!r}", legal=candidates,
                     suggestion=_suggest(value, candidates))


# ── measures: one name set, one validation path ─────────────────────────────────────────────────
# The legal measure names ARE the MEASURES keys — no aliases, no second spelling, so a `SkillSpec`
# re-emitted as YAML re-parses to the same spec and a chassis default and an authored value go
# through the SAME check. (2026-08-12 review, finding 4: they used not to. The live seat height was
# keyed on the bare quantity `height_above_seat` while its static-goal twin carried a frame suffix,
# so the schema's legal set and the vocabulary's key set disagreed on exactly that one name, and the
# carry chassis wrote the spelling the schema declared illegal — resolved silently by a second code
# path. The vocabulary was renamed: `height_above_seat_live`.)
LEGAL_MEASURE_NAMES = frozenset(MEASURES)

_FRAME_SUFFIXES = tuple(f"_{f.value}" for f in Frame)


def _quantity(name):
    """The measure name with its frame qualifier stripped: the physical quantity it reads."""
    for suffix in _FRAME_SUFFIXES:
        if name.endswith(suffix) and len(name) > len(suffix):
            return name[: -len(suffix)]
    return name


#: bare quantity -> the frame-qualified MEASURES keys that read it, for quantities readable in more
#: than one frame (exactly one today: the seat height, vocab.py correction 2). This RESOLVES NOTHING
#: — it exists only so the refusal for the bare name can say WHY it is not a name, rather than
#: "unknown measure". The bare quantity is the spelling every source comment and the design doc's §4
#: example use, so it is the likeliest thing an author writes, and "one name, two truths" is the
#: thing they have to be told.
#:
#: THE `q not in MEASURES` FILTER IS CONDITIONAL ON AN INVARIANT, AND THE INVARIANT IS TESTED
#: ELSEWHERE (2026-08-12 re-review, Important 1). It is here so this table can never shadow a
#: measure that IS a legal name; but it also means that if a family ever grew a frame-qualified
#: sibling beside a BARE key — `height_above_resting_static_goal` next to the existing
#: `height_above_resting` — that family would drop out of the table and the bare name would go on
#: resolving silently to LIVE, which is exactly the §1.2 rule the seat-height rename was for,
#: defeated in a different family. `test_vocab.py` asserts the general property that makes the
#: filter safe: a frame-qualified key's quantity stem is never itself a MEASURES key.
_FRAME_VARIANTS = {}
for _key, _measure in MEASURES.items():
    _FRAME_VARIANTS.setdefault(_quantity(_key), set()).add(_key)
_FRAME_VARIANTS = MappingProxyType(
    {q: frozenset(keys) for q, keys in _FRAME_VARIANTS.items()
     if len(keys) > 1 and q not in MEASURES})


def _frame_ambiguity_error(path, name):
    keys = sorted(_FRAME_VARIANTS[name])
    frames = sorted(MEASURES[k].frame.value for k in keys)
    return SpecError(
        path,
        f"measure {name!r} names a quantity that is readable in {len(keys)} frames "
        f"({', '.join(frames)}) and this row names neither — say which frame you mean: "
        f"{', '.join(keys)}. descend's reward grades this against the live seat while "
        f"descend_stack's success gate grades it against a goal frozen at init: one quantity, two "
        f"truths",
        legal=LEGAL_MEASURE_NAMES,
        suggestion=_suggest(name, keys))


def _resolve_measure(path, name):
    """A measure reference -> its MEASURES key. Raises on unknown or frame-ambiguous.

    The ONLY measure check in the module: an authored value and a chassis-supplied default both come
    through here, so nothing can reach `MEASURES[...]` unvalidated and raise a bare `KeyError` out of
    `parse_spec` — the author's contract is that they only ever catch `SpecError`.
    """
    if not isinstance(name, str):
        raise SpecError(path, f"a measure is named by a string, got {type(name).__name__}",
                        legal=LEGAL_MEASURE_NAMES)
    if name in MEASURES:
        return name
    if name in _FRAME_VARIANTS:
        raise _frame_ambiguity_error(path, name)
    raise _unknown(path, "measure", name, LEGAL_MEASURE_NAMES)


# ── predicates and per-skill param references ───────────────────────────────────────────────────

# A predicate field is either a bare name (`grasped`) or a nested call over existing names
# (`and_(grasped, above_z(z=0.06))`). Both the CALL names and the bare POSITIONAL names are
# predicate references and both are checked. What is NOT a predicate, and is blanked out before the
# names are read, is scene data: quoted strings, dotted attribute chains (`bin.inner_radius`,
# `params.hover` — those get their own check), keyword-argument NAMES, and the identifier a keyword
# argument is set to (`anchor=target_pos` — target_pos is a scene point, and vocab.py's own
# carry_with_potential chassis writes exactly that).
#
# (2026-08-12 review, finding 2: only the call names used to be checked, so the typo in
# `and_(grapsed, above_z(z=0.06))` — a BARE name, the position a compound predicate puts its
# operands in — was never looked up and the row parsed clean. vocab.py's chassis compose via `and_`,
# so this is the shape the author is shown and the shape they will copy.)
_QUOTED_RE = re.compile(r"'[^']*'|\"[^\"]*\"")
_DOTTED_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\s*\.\s*[A-Za-z_][A-Za-z0-9_]*)+")
_KWARG_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\s*=(?!=)(?:\s*[A-Za-z_][A-Za-z0-9_]*)?")
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_PARAM_REF_RE = re.compile(r"\bparams\.([A-Za-z_][A-Za-z0-9_]*)")


def _blank(pattern, text):
    """Replace every match with the same number of spaces — keeps offsets, kills word boundaries."""
    return pattern.sub(lambda m: " " * len(m.group(0)), text)


def _predicate_names(text):
    """Every identifier in a predicate expression that has to name a PREDICATES entry."""
    for pattern in (_QUOTED_RE, _DOTTED_RE, _KWARG_RE):
        text = _blank(pattern, text)
    return set(_IDENT_RE.findall(text))


def _check_predicate(path, value, declared_params):
    if not isinstance(value, str) or not value.strip():
        raise SpecError(path, "a predicate is named by a non-empty string (a bare name, or a call "
                              "over existing names like `and_(grasped, above_z(z=0.06))`)",
                        legal=PREDICATES)
    _check_param_refs(path, value, declared_params)
    names = _predicate_names(value)
    if not names:
        raise SpecError(path, f"{value!r} names no predicate: a predicate field is a bare name from "
                              f"the list below, or a call over them like "
                              f"`and_(grasped, above_z(z=0.06))`", legal=PREDICATES)
    for name in sorted(names):
        if name not in PREDICATES:
            raise _unknown(path, "predicate", name, PREDICATES)


def _check_param_refs(path, text, declared_params):
    """Every `params.X` in an authored string must resolve to a param the document declared."""
    for name in _PARAM_REF_RE.findall(text):
        if name not in declared_params:
            raise SpecError(path, f"`params.{name}` is not declared in this document's `params:` "
                                  f"block, so there is nothing for compile() to bind it to",
                            legal=declared_params, suggestion=_suggest(name, declared_params))


def _is_param_ref(param, value):
    """Does this value stand in for a NUMBER that compile() binds from `params:` later?

    Only numeric fields get the bypass. (2026-08-12 review, minor 2: the test used to be
    `value.startswith("params.")` alone, so it skipped `choices` as well as typing and
    `mode: params.hover` was accepted on PredicateBonus although `mode` is a closed set of three
    strings. A string field's legal values are knowable NOW; deferring them to compile() buys
    nothing and loses the refusal.)
    """
    return (param.type in ("float", "int") and isinstance(value, str)
            and value.startswith("params."))


def _legal_values_for(param):
    """The enumerable legal set for a parameter, or None when its type is an open one (a float).
    Used by refusals that have to state a legal set without having a wrong value to diff against.
    """
    if param.choices:
        return param.choices
    if param.name == "measure":
        return LEGAL_MEASURE_NAMES
    if param.name in ("predicate", "gate"):
        return PREDICATES
    return None


# ── value typing ────────────────────────────────────────────────────────────────────────────────
# `bool` is a subclass of `int` in Python, so these use `type(v) is` rather than isinstance: `True`
# is not a weight, and a term that silently accepted it would train something nobody wrote.
_TYPE_OK = {
    "float": lambda v: type(v) in (int, float),
    "int": lambda v: type(v) is int,
    "bool": lambda v: type(v) is bool,
    "str": lambda v: isinstance(v, str),
}


def _check_type(path, param, value):
    ok = _TYPE_OK.get(param.type)
    if ok is not None and not ok(value):
        raise SpecError(path, f"{param.name!r} must be a {param.type}, got "
                              f"{type(value).__name__} ({value!r})")
    if param.choices and value not in param.choices:
        raise SpecError(path, f"{value!r} is not a legal value for {param.name!r}",
                        legal=param.choices, suggestion=_suggest(value, param.choices))


# ── rows ────────────────────────────────────────────────────────────────────────────────────────

#: The terms a reward ROW may name. `RewardScale` is excluded: it is a document-level field
#: (`reward_scale:`), not a summed row — an env that treats it as one trains at 12x the intended
#: scale (vocab.py, RewardScale.doc).
ROW_TERMS = tuple(name for name in TERMS if name != "RewardScale")

_TIERS = ("term", "expr", "custom")
_CUSTOM_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*:[A-Za-z_][A-Za-z0-9_]*$")


@dataclasses.dataclass(frozen=True)
class Row:
    """One reward row, at exactly one of the three tiers (design doc §3).

    tier 1  `term` + `params`: a vocabulary term with EVERY parameter resolved — authored value,
            else chassis default, else the term's own default. `params["measure"]` is always a
            MEASURES key, whatever spelling the author used.
    tier 2  `expr`: a parsed `bridle.skill.expr.Expr`, its free names already checked against the
            measures, predicates and declared params.
    tier 3  `custom`: an opaque `module:function`. `bridle plan` prints it and the fingerprint
            records that part of this reward cannot be read.

    `why` is never empty and never inherited — see the module docstring.
    """

    term: str | None
    params: MappingProxyType
    expr: Expr | None
    custom: str | None
    why: str


_DISCRIMINATORS = ("measure", "predicate")


def _chassis_defaults_for(chassis, term_name, authored):
    """`(the chassis row this authored row inherits from, the candidates it could not choose from)`.

    A chassis may instantiate one term several times under suffixed keys (carry has `DistancePull_xy`
    at k=4.0 and `DistancePull_height` at k=6.0 — collapsing them into one 3D kernel changes the
    task). When it does, the authored `measure`/`predicate` is what picks the right one; with no
    discriminator nothing is inherited and the term's own defaults apply, because guessing between
    1.5@k=4 and 2.5@k=6 is exactly the silent substitution this whole module exists to prevent.

    The unchosen candidates come back with the answer so a downstream refusal can NAME them — an
    author told only "the chassis supplies nothing" cannot tell that it in fact supplies two things
    and they have to pick (2026-08-12 review, finding 3).

    PRIVATE BY NAME, BUT NOT PRIVATE IN FACT: `bridle/skill/report.py`'s `format_plan` (what
    `bridle skill compile` prints) imports this function to label each parameter it renders
    `authored` / `chassis '<name>' default` / `term default`. That coupling is
    deliberate and must survive a refactor — a provenance report that re-implemented the suffixed-key
    rule would diverge exactly where it matters, on the chassis that instantiates one term twice
    (`DistancePull_xy` at k=4.0 against `DistancePull_height` at k=6.0), and would then tell the
    author a weight was inherited from a row the parser did not use. Change the signature and you
    change the CLI: `grep -rn _chassis_defaults_for` before touching it.

    `chassis` is never None: `parse_spec` refuses an unknown `kind:` before assigning
    `CHASSIS[kind]`, so every caller has a real chassis. The `chassis is None` guard that used to
    open this function, and the `chassis.name if chassis else 'none'` fallback in
    `_required_param_error`, were unreachable (2026-08-12 re-review, minor 6) — and an unreachable
    fallback is worse than none, because it reads as a supported mode and invites a caller to try it.
    """
    candidates = [row for key, row in chassis.defaults.items() if base_term(key) == term_name]
    if len(candidates) == 1:
        return candidates[0], []
    if not candidates:
        return {}, []
    for field in _DISCRIMINATORS:
        # A key present with a null value is not a discriminator — see the null rule in
        # `_parse_term_row`; `measure:` with nothing after it must behave exactly like no key.
        if authored.get(field) is None:
            continue
        hits = [row for row in candidates if row.get(field) == authored[field]]
        if len(hits) == 1:
            return hits[0], []
    return {}, candidates


def _describe_candidates(candidates):
    """`measure='object_to_goal_xy' (weight=1.5, k=4.0)`, one per unchosen chassis row — enough for
    an author to see WHICH row they meant and what picking it would inherit."""
    out = []
    for row in candidates:
        field = next((f for f in _DISCRIMINATORS if row.get(f) is not None), None)
        head = f"{field}={row[field]!r}" if field else "(no measure/predicate to name it by)"
        rest = ", ".join(f"{k}={v!r}" for k, v in row.items()
                         if k != "why" and k != field and not isinstance(v, str))
        out.append(f"{head} ({rest})" if rest else head)
    return "; ".join(out)


def _required_param_error(path, param, term_name, chassis, candidates):
    """The refusal for a required parameter nothing supplied.

    (2026-08-12 review, finding 3: this used to read "neither the row nor the 'carry' chassis
    supplies one" — which for `DistancePull` under `carry` is false twice over, since that chassis
    supplies TWO DistancePull rows and the reason nothing was inherited is that the row named no
    `measure` to choose between them. It carried no legal set and no suggestion either: the only
    refusal in the module with neither, in the module whose premise is that a refusal a 27-30B
    author cannot act on costs a round trip.)
    """
    where = f"{path}.{param.name}"
    head = f"{param.name!r} is required by {term_name} ({param.doc or param.type})"
    legal = _legal_values_for(param)
    if candidates:
        # Unconditional legal set on this branch: the parameter's own if it has an enumerable one,
        # else the discriminators that would pick a candidate row — either way something to write.
        discriminators = sorted({str(row[f]) for row in candidates
                                 for f in _DISCRIMINATORS if row.get(f) is not None})
        return SpecError(
            where,
            f"{head}. The '{chassis.name}' chassis supplies {len(candidates)} {term_name} rows and "
            f"this row names no `{_DISCRIMINATORS[0]}`/`{_DISCRIMINATORS[1]}` to choose between "
            f"them, so it inherited nothing — name the row you mean, or supply "
            f"{param.name!r} yourself. The candidates are: {_describe_candidates(candidates)}",
            legal=legal or discriminators)
    return SpecError(
        where,
        f"{head} and neither the row nor the '{chassis.name}' chassis supplies one",
        legal=legal)


def _parse_term_row(path, raw, chassis, declared_params):
    term_name = raw["term"]
    if term_name == "RewardScale":
        raise SpecError(f"{path}.term",
                        "RewardScale is a document-level field, not a reward row: write "
                        "`reward_scale: {divisor: 12.0}` at the top level. It divides the summed "
                        "dense reward before PPO sees it; summed as a row it would be 12x wrong",
                        legal=ROW_TERMS)
    if not isinstance(term_name, str) or term_name not in TERMS:
        raise _unknown(f"{path}.term", "reward term", term_name, ROW_TERMS)

    term = TERMS[term_name]
    declared = {p.name: p for p in term.params}
    authored = {k: v for k, v in raw.items() if k not in ("term", "why")}
    for key in authored:
        if key not in declared:
            raise SpecError(f"{path}.{key}", f"unknown parameter {key!r} for {term_name}",
                            legal=declared, suggestion=_suggest(key, declared))

    inherited, unchosen = _chassis_defaults_for(chassis, term_name, authored)
    values = {}
    for param in term.params:
        # THE NULL RULE (2026-08-12 review, finding 1). A key PRESENT with a null value is not a
        # supplied value: `measure:` with nothing after it is the commonest YAML slip there is, and
        # treating its None as authored short-circuited required-ness, typing, `choices` AND the
        # signed-measure gate below (which is written `is not None` precisely to let a term's own
        # optional default through). A `HingePenalty` with `measure: null` parsed clean and handed
        # compile() a crush penalty with no measure. So an authored null falls through exactly like
        # an absent key — chassis default, then term default, then the required-parameter refusal —
        # and the only None that reaches `values` is one that came from `param.default`, which is
        # how the vocabulary spells "optional, off" (`axes`, `gate`, `enabled_if`).
        if authored.get(param.name) is not None:
            value = authored[param.name]
        elif inherited.get(param.name) is not None:
            value = inherited[param.name]
        elif param.required:
            raise _required_param_error(path, param, term_name, chassis, unchosen)
        else:
            value = param.default
            if value is None:
                values[param.name] = None
                continue

        field_path = f"{path}.{param.name}"
        if param.name == "measure":
            # One path for authored and chassis-supplied alike — see `_resolve_measure`.
            value = _resolve_measure(field_path, value)
        elif param.name in ("predicate", "gate"):
            _check_predicate(field_path, value, declared_params)
        elif isinstance(value, str):
            _check_param_refs(field_path, value, declared_params)
        if not _is_param_ref(param, value):
            _check_type(field_path, param, value)
        values[param.name] = value

    if term.needs_signed_measure and values.get("measure") is not None:
        measure = MEASURES[values["measure"]]
        if measure.sign is not Sign.SIGNED:
            raise SpecError(
                f"{path}.measure",
                f"{term_name} needs a signed measure but {values['measure']!r} has "
                f"sign={measure.sign.value}: over an unsigned reading its "
                f"`clamp(signed_delta, min=0)` is identically zero, so the row trains, logs, and "
                f"contributes nothing. The crush penalty this deletes exists because pressing to "
                f"dz=0 broke 16/16 grasps (2026-06-04)",
                legal=[n for n in LEGAL_MEASURE_NAMES if MEASURES[n].sign is Sign.SIGNED])

    return Row(term=term_name, params=_freeze(values), expr=None, custom=None, why=raw["why"])


def _expr_error(path, source, exc, declared_params):
    """`expr.py`'s refusal, made actionable in the one case this module knows more than it does.

    THE PARAM PREFIX IS SPELLED TWO WAYS ACROSS THE TIERS (2026-08-12 re-review, Important 3). A
    tier-1 row writes `setpoint: params.hover`, and so does the `success:` criterion; a tier-2 row
    wants the BARE name (`expr: "2.5 * hover"`), because an expression is PARSED and `params.hover`
    parses as attribute access — a construct the grammar refuses outright, which is what turns
    `x.__class__` into a parse-time error rather than a possibility. Re-raising expr.py's generic
    whitelist text for that case ("'attribute access (e.g. x.attr)' is not allowed ...") names no
    path fix, no legal spelling and no suggestion, and lands on an author who learned the `params.`
    prefix one tier up: precisely the round trip this module exists to prevent. `params.` in the
    source says unambiguously what was meant, so say so and print the corrected line.
    """
    if _PARAM_REF_RE.search(source):
        fixed = _PARAM_REF_RE.sub(r"\1", source)
        return SpecError(
            path,
            f"inside an `expr:` a declared param is named by its BARE name, with no `params.` "
            f"prefix — write {fixed!r}, not {source!r}. (The prefix is right in a term row's "
            f"fields and in `success:`; it is wrong here because an expression is parsed, and "
            f"`params.x` parses as attribute access, which the expression grammar refuses "
            f"outright.) The underlying refusal was: {exc}",
            legal=declared_params or None,
            suggestion=fixed)
    return SpecError(path, str(exc))


def _parse_expr_row(path, raw, declared_params):
    for key in raw:
        if key not in ("expr", "why"):
            raise SpecError(f"{path}.{key}",
                            f"an `expr:` row takes only `expr` and `why`, not {key!r} — an "
                            f"expression gates and scales itself (`... * grasped`)")
    try:
        parsed = parse_expr(raw["expr"])
    except ExprError as exc:
        raise _expr_error(f"{path}.expr", raw["expr"], exc, declared_params) from exc
    for name in sorted(parsed.names):
        # A declared param shadows a measure of the same name: the author wrote the param, so the
        # author meant the param. Nothing in the corpus collides today.
        if name in declared_params or name in PREDICATES or name in MEASURES:
            continue
        if name in _FRAME_VARIANTS:
            raise _frame_ambiguity_error(f"{path}.expr", name)
        legal = set(LEGAL_MEASURE_NAMES) | set(PREDICATES) | set(declared_params)
        raise SpecError(f"{path}.expr", f"unknown name {name!r} in the expression: it is not a "
                                        f"measure, a predicate, or a param this document declares",
                        legal=legal, suggestion=_suggest(name, legal))
    return Row(term=None, params=MappingProxyType({}), expr=parsed, custom=None, why=raw["why"])


def _parse_custom_row(path, raw):
    for key in raw:
        if key not in ("custom", "why"):
            raise SpecError(f"{path}.{key}",
                            f"a `custom:` row takes only `custom` and `why`, not {key!r}")
    target = raw["custom"]
    if not isinstance(target, str) or not _CUSTOM_RE.match(target):
        raise SpecError(f"{path}.custom",
                        f"a custom row names an importable `module:function`, got {target!r} — "
                        f"e.g. `primitives.descend_to_target.descend_env:crush_term`")
    return Row(term=None, params=MappingProxyType({}), expr=None, custom=target, why=raw["why"])


def _parse_row(index, raw, chassis, declared_params):
    path = f"reward[{index}]"
    if not isinstance(raw, dict):
        raise SpecError(path, f"a reward row is a mapping with one of `term:`, `expr:` or "
                              f"`custom:`, got {type(raw).__name__}")
    tiers = [t for t in _TIERS if t in raw]
    if len(tiers) != 1:
        found = f"{tiers}" if tiers else "none of them"
        raise SpecError(path, f"a reward row declares exactly one of `term:` (a vocabulary term), "
                              f"`expr:` (an arithmetic expression) or `custom:` "
                              f"(module:function); this row declares {found}")
    why = raw.get("why")
    if not isinstance(why, str) or not why.strip():
        raise SpecError(f"{path}.why",
                        "every reward row needs a non-empty `why`: the rationale for the number "
                        "you chose, in prose. It is mandatory because stating the rationale before "
                        "the number is what took an LLM-authored reward from 50% to 90%, and "
                        "because YAML drops comments — this field is the only surviving record of "
                        "why the weight is what it is")
    if tiers == ["term"]:
        return _parse_term_row(path, raw, chassis, declared_params)
    if tiers == ["expr"]:
        return _parse_expr_row(path, raw, declared_params)
    return _parse_custom_row(path, raw)


# ── document sections ───────────────────────────────────────────────────────────────────────────

def _freeze(value):
    """A document fragment, deep-frozen: mappings become `MappingProxyType`, sequences tuples.

    `dataclass(frozen=True)` only stops the FIELD being rebound, so before this
    `spec.scene["goal"]["type"] = "MUTATED"` and `spec.params["hover"]["value"] = 99.0` both
    succeeded (2026-08-12 review, minor 1). Those numbers hash into the contract fingerprint that
    says which policy was trained under which spec, so a spec editable after construction is a
    fingerprint that stops describing the run. Freezing also copies, which is what keeps `parse_spec`
    from handing back views onto the caller's dict.
    """
    if isinstance(value, dict):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    return value


_REQUIRED_FIELDS = ("name", "kind", "contract", "env_id", "scene", "reward", "success")
_OPTIONAL_FIELDS = ("init", "params", "reward_scale", "preflight")
_FIELDS = _REQUIRED_FIELDS + _OPTIONAL_FIELDS

_SEVERITIES = (RUN, ADAPT, RETRAIN)
_PARAM_FIELDS = ("value", "severity", "doc")
_PREFLIGHT_TIERS = (STATIC, DYNAMIC)
_BOUND_FIELDS = ("min", "max", "expect", "needs")


def _parse_params(raw):
    """The `params:` block: per-skill physics `Contract` has no field for (§5). 69 such parameters
    exist across the corpus — bin rims, seat depths, stack tolerances — and requiring a core-type
    change per skill would fail LLM authorship on skill #1. Each carries a `severity` because they
    hash into the fingerprint and go through `resolve()` exactly like a core field.
    """
    if not isinstance(raw, dict):
        raise SpecError("params", f"`params:` is a mapping of name -> "
                                  f"{{value, severity, doc}}, got {type(raw).__name__}")
    out = {}
    for name, body in raw.items():
        path = f"params.{name}"
        if not isinstance(body, dict):
            raise SpecError(path, f"a param is a mapping of {{value, severity, doc}}, got "
                                  f"{type(body).__name__}")
        for key in body:
            if key not in _PARAM_FIELDS:
                raise _unknown(f"{path}.{key}", "param field", key, _PARAM_FIELDS)
        if "value" not in body:
            raise SpecError(f"{path}.value", "a param must declare a `value`")
        severity = body.get("severity")
        if severity is None:
            raise SpecError(f"{path}.severity",
                            "a param must declare a `severity`: whether changing this number can "
                            "reuse a checkpoint (run), needs adaptation (adapt), or invalidates it "
                            "(retrain) — see bridle.resolve", legal=_SEVERITIES)
        if severity not in _SEVERITIES:
            raise _unknown(f"{path}.severity", "severity", severity, _SEVERITIES)
        if "doc" in body and not isinstance(body["doc"], str):
            raise SpecError(f"{path}.doc", f"`doc` is prose, got {type(body['doc']).__name__}")
        out[name] = _freeze(body)
    return out


def _parse_scene(raw):
    if not isinstance(raw, dict) or not raw:
        raise SpecError("scene", f"`scene:` is a non-empty mapping of role -> object, got "
                                 f"{type(raw).__name__}")
    for role, body in raw.items():
        path = f"scene.{role}"
        if not isinstance(body, dict):
            raise SpecError(path, f"a scene object is a mapping, got {type(body).__name__}")
        kind = body.get("type")
        if not isinstance(kind, str) or not kind.strip():
            raise SpecError(f"{path}.type", "every scene object declares a `type` (cube, platform, "
                                            "bin, ...) — it is what the generator builds a body from")
    return _freeze(raw)


def _parse_reward_scale(raw, chassis, declared_params):
    """`reward_scale:` — `reward_ppo = dense / divisor`, stated explicitly (§1.4). 7 of 15 primitives
    inherit `compute_normalized_dense_reward` without overriding it and so train at dense/12.0 even
    where that is semantically wrong (lift's per-step max is ~18, reach's ~9); a generated env that
    forgets the field entirely trains at 12x the intended scale.

    THE ASYMMETRY, RESOLVED TOWARD ALLOWING (2026-08-12 review, minor 3): `divisor: params.scale` is
    accepted, exactly as `setpoint: params.hover` is in a reward row. It used to be refused as a
    type error while the row form was allowed, which is a rule an author has to learn twice. The
    divisor is a number a skill may well want declared once in `params:` with a severity — changing
    it rescales every gradient, so `retrain` is the honest tag, and `params:` is the only place this
    document can say that. Same numeric-only restriction as rows: `unnormalized` is a bool and takes
    no reference.

    TWO MORE ASYMMETRIES CLOSED (2026-08-12 re-review, minors 4 and 5). (a) `reward_scale: null` —
    the whole BLOCK nulled, not a field inside it — used to be silently read as `{}` and fall
    through, while `init: null`, `params: null` and `preflight: null` were all refused; one null
    rule now covers the document. A null field INSIDE the block still falls through, exactly as a
    null key inside a reward row does. (b) the type and `params.X` checks used to run on AUTHORED
    values only, so a chassis-supplied or term-default divisor reached `RewardPlan` unchecked —
    harmless while the inherited value is the literal 12.0, but it breaks the "one validation path
    for authored and chassis-supplied alike" rule this module states for measures and enforces in
    `_parse_term_row`, and it is the path a future per-chassis divisor would arrive on.
    """
    term = TERMS["RewardScale"]
    declared = {p.name: p for p in term.params}
    if raw is None:
        raise SpecError("reward_scale", "`reward_scale:` is a mapping of {divisor, unnormalized} "
                                        "and this one is empty — omit the key entirely to inherit "
                                        "the chassis' divisor, or write `{unnormalized: true}` to "
                                        "declare the reward is already at the scale PPO should see")
    authored = raw
    if not isinstance(authored, dict):
        raise SpecError("reward_scale", f"`reward_scale:` is a mapping of "
                                        f"{{divisor, unnormalized}}, got {type(authored).__name__}")
    for key in authored:
        if key not in declared:
            raise _unknown(f"reward_scale.{key}", "reward_scale field", key, declared)
    inherited = chassis.defaults.get("RewardScale", {})
    values = {}
    for param in term.params:
        # The null rule of `_parse_term_row` applies here too: `divisor:` with nothing after it
        # falls through to the chassis' 12.0 rather than becoming a None nobody can divide by.
        if authored.get(param.name) is not None:
            value = authored[param.name]
        elif inherited.get(param.name) is not None:
            value = inherited[param.name]
        else:
            value = param.default
        field_path = f"reward_scale.{param.name}"
        if value is not None:   # a None here is a term default spelling "optional, off"
            if isinstance(value, str):
                _check_param_refs(field_path, value, declared_params)
            if not _is_param_ref(param, value):
                _check_type(field_path, param, value)
        values[param.name] = value
    return values


def _parse_preflight(raw):
    """`preflight:` — the asserts `bridle.preflight` runs before the GPU is spent. Shape only; what
    a path means is that module's business.
    """
    if not isinstance(raw, dict):
        raise SpecError("preflight", f"`preflight:` is a mapping of tier -> asserts, got "
                                     f"{type(raw).__name__}", legal=_PREFLIGHT_TIERS)
    for tier, body in raw.items():
        if tier not in _PREFLIGHT_TIERS:
            raise _unknown(f"preflight.{tier}", "preflight tier", tier, _PREFLIGHT_TIERS)
        if not isinstance(body, dict):
            raise SpecError(f"preflight.{tier}", f"a preflight tier is a mapping of target -> "
                                                 f"bounds, got {type(body).__name__}")
        for target, bounds in body.items():
            path = f"preflight.{tier}.{target}"
            if not isinstance(bounds, dict):
                raise SpecError(path, f"a preflight assert is a mapping of "
                                      f"{{min, max, expect, needs}}, got {type(bounds).__name__}")
            for key in bounds:
                if key not in _BOUND_FIELDS:
                    raise _unknown(f"{path}.{key}", "preflight bound", key, _BOUND_FIELDS)
            if not any(key in bounds for key in ("min", "max", "expect")):
                raise SpecError(path, "a preflight assert with no bound (min/max/expect) is not a "
                                      "claim, it is a check that can never fail")
    return _freeze(raw)


# ── the document ────────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass(frozen=True)
class SkillSpec:
    """A validated skill document. Every field is resolved against the vocabulary but nothing is
    compiled: `reward` rows carry their effective parameters, `success` carries its text, and
    turning either into a callable is `compile()`'s job.
    """

    name: str
    kind: str
    contract: str
    env_id: str
    scene: MappingProxyType
    init: MappingProxyType
    params: MappingProxyType
    reward_scale: MappingProxyType
    reward: tuple
    success: str
    preflight: MappingProxyType


def _require_text(field, value):
    if not isinstance(value, str) or not value.strip():
        raise SpecError(field, f"`{field}:` is a non-empty string, got {value!r}")
    return value


def parse_spec(doc: dict) -> SkillSpec:
    """Validate a skill document against the vocabulary and return a frozen `SkillSpec`.

    Takes the mapping a YAML loader produced, not YAML text — loading is the CLI's problem, and
    keeping it out of here is what keeps `bridle` core dependency-free. The document is never
    mutated: defaults are resolved into the returned spec, not written back into the caller's dict.
    """
    if not isinstance(doc, dict):
        raise SpecError("<document>", f"a skill document is a mapping of fields, got "
                                      f"{type(doc).__name__}", legal=_FIELDS)
    for key in doc:
        if key not in _FIELDS:
            raise _unknown(key, "field", key, _FIELDS)
    for field in _REQUIRED_FIELDS:
        if field not in doc:
            raise SpecError(field, f"required field `{field}:` is missing", legal=_FIELDS)

    name = _require_text("name", doc["name"])
    contract = _require_text("contract", doc["contract"])
    env_id = _require_text("env_id", doc["env_id"])

    kind = doc["kind"]
    if not isinstance(kind, str) or kind not in CHASSIS:
        raise _unknown("kind", "chassis", kind, CHASSIS)
    chassis = CHASSIS[kind]

    params = _parse_params(doc["params"]) if "params" in doc else {}
    scene = _parse_scene(doc["scene"])

    init = doc.get("init", {})
    if not isinstance(init, dict):
        raise SpecError("init", f"`init:` is a mapping, got {type(init).__name__}")

    # `doc.get(..., {})` and NOT `doc.get(...)`: an ABSENT `reward_scale:` inherits the chassis
    # divisor, a PRESENT-but-null one is refused like `init: null` — `_parse_reward_scale` can only
    # tell those apart if the absent case never reaches it as None.
    reward_scale = _parse_reward_scale(doc.get("reward_scale", {}), chassis, params)

    rows = doc["reward"]
    if not isinstance(rows, list):
        raise SpecError("reward", f"`reward:` is a list of rows, got {type(rows).__name__}")
    if not rows:
        raise SpecError("reward", "`reward:` is empty — a skill with no reward row has nothing to "
                                  "optimise")
    reward = tuple(_parse_row(i, row, chassis, params) for i, row in enumerate(rows))

    success = _require_text("success", doc["success"])
    _check_param_refs("success", success, params)

    preflight = _parse_preflight(doc["preflight"]) if "preflight" in doc else {}

    return SkillSpec(
        name=name, kind=kind, contract=contract, env_id=env_id,
        scene=_freeze(scene), init=_freeze(init),
        params=_freeze(params), reward_scale=_freeze(reward_scale),
        reward=reward, success=success, preflight=_freeze(preflight))


# ── the schema a model is handed ────────────────────────────────────────────────────────────────

_JSON_TYPES = {"float": "number", "int": "integer", "bool": "boolean", "str": "string"}
_PARAM_REF_PATTERN = r"^params\.[A-Za-z_][A-Za-z0-9_]*$"

_WHY_SCHEMA = {
    "type": "string", "minLength": 1,
    "description": "MANDATORY. Why this row, and why these numbers. State it before choosing them.",
}


def _param_json(param):
    """The JSON Schema for one term parameter, WITHOUT its description (descriptions differ per
    term for a shared name like `weight`, and merging must compare the constraint, not the prose).
    """
    if param.choices:
        return {"enum": list(param.choices)}
    if param.name == "measure":
        return {"enum": sorted(LEGAL_MEASURE_NAMES)}
    json_type = _JSON_TYPES.get(param.type)
    if json_type in ("number", "integer"):
        # ...or a `params.X` reference standing in for the number, resolved at compile time (§5).
        return {"anyOf": [{"type": json_type},
                          {"type": "string", "pattern": _PARAM_REF_PATTERN}]}
    return {"type": json_type} if json_type else {}


def _merge_param_schemas(variants):
    """One property per parameter NAME across the eight row terms. Identical constraints merge
    cleanly; a name two terms constrain differently widens to the shared JSON type rather than
    picking a winner — a schema that rejected a legal document would be worse than one that admits
    an extra.

    NO NAME DIVERGES TODAY (2026-08-12 re-review, minor 3). `mode` used to be the example — a closed
    choice set on PredicateBonus and open on SuccessBonus — and giving SuccessBonus its real
    `choices` made the two identical, so all five shared names (`gate`, `measure`, `mode`, `scope`,
    `weight`) now merge cleanly and the widening branch is defensive only. It stays because the
    divergence it handles is a schema-emission decision, not a bug: the alternative, picking one
    term's constraint for a shared property, is how a flat row schema comes to refuse a legal
    document. (The flat `term_row` bag itself — 27 keys under `additionalProperties: false`, where a
    `oneOf` over per-term variants is the real fix — is deferred to the whole-branch review.)
    """
    first = variants[0]
    if all(v == first for v in variants):
        return dict(first)
    kinds = {v.get("type") or ("string" if "enum" in v else None) for v in variants}
    if len(kinds) == 1 and None not in kinds:
        return {"type": kinds.pop()}
    return {}


def json_schema() -> dict:
    """A JSON Schema (draft 2020-12) for the skill document — the machine-readable half of what a
    27-30B author is handed, alongside `vocab.vocab_document()`'s prose. It carries the SHAPE (which
    fields, which enums, `why` required everywhere); the rationale for the numbers lives in the
    vocabulary document, and the checks that need cross-row reasoning live in `parse_spec` and
    `compile()`. A schema alone cannot express "this measure must be signed for this term".
    """
    collected, descriptions = {}, {}
    for term_name in ROW_TERMS:
        for param in TERMS[term_name].params:
            collected.setdefault(param.name, []).append(_param_json(param))
            descriptions.setdefault(param.name, param.doc)
    row_params = {}
    for param_name, variants in collected.items():
        schema = _merge_param_schemas(variants)
        if descriptions.get(param_name):
            schema["description"] = descriptions[param_name]
        row_params[param_name] = schema

    scale = TERMS["RewardScale"]
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "bridle skill document",
        "description": "scene + reward + success for one robot skill; compiles to a RewardPlan.",
        "type": "object",
        "additionalProperties": False,
        "required": list(_REQUIRED_FIELDS),
        "properties": {
            "name": {"type": "string", "minLength": 1},
            "kind": {"enum": sorted(CHASSIS),
                     "description": "chassis: supplies gotcha defaults AND reward-weight defaults"},
            "contract": {"type": "string", "minLength": 1},
            "env_id": {"type": "string", "minLength": 1},
            "scene": {
                "type": "object", "minProperties": 1,
                "additionalProperties": {"type": "object", "required": ["type"],
                                         "properties": {"type": {"type": "string"}}},
            },
            "init": {"type": "object"},
            "params": {"type": "object", "additionalProperties": {"$ref": "#/$defs/param"}},
            "reward_scale": {
                "type": "object", "additionalProperties": False,
                "properties": {p.name: dict(_param_json(p), description=p.doc)
                               for p in scale.params},
            },
            "reward": {
                "type": "array", "minItems": 1,
                "items": {"oneOf": [{"$ref": "#/$defs/term_row"},
                                    {"$ref": "#/$defs/expr_row"},
                                    {"$ref": "#/$defs/custom_row"}]},
            },
            "success": {"type": "string", "minLength": 1},
            "preflight": {
                "type": "object", "additionalProperties": False,
                "properties": {tier: {"type": "object",
                                      "additionalProperties": {"$ref": "#/$defs/assert"}}
                               for tier in _PREFLIGHT_TIERS},
            },
        },
        "$defs": {
            "term_row": {
                "type": "object", "additionalProperties": False,
                "required": ["term", "why"],
                "properties": {"term": {"enum": list(ROW_TERMS)}, "why": _WHY_SCHEMA, **row_params},
            },
            "expr_row": {
                "type": "object", "additionalProperties": False,
                "required": ["expr", "why"],
                "properties": {
                    "expr": {"type": "string", "minLength": 1,
                             "description": "arithmetic over measures, predicates and this "
                                            "document's declared params, each named BARE: write "
                                            "`2.5 * hover`, NOT `2.5 * params.hover` — the "
                                            "`params.` prefix belongs to a term row's fields and "
                                            "to `success:`, and inside an expression it parses as "
                                            "attribute access, which is refused. Operators "
                                            "+ - * / ** and comparisons; calls abs tanh exp log "
                                            "sqrt clamp min max where"},
                    "why": _WHY_SCHEMA,
                },
            },
            "custom_row": {
                "type": "object", "additionalProperties": False,
                "required": ["custom", "why"],
                "properties": {
                    "custom": {"type": "string", "pattern": _CUSTOM_RE.pattern,
                               "description": "module:function — opaque, and fingerprinted as such"},
                    "why": _WHY_SCHEMA,
                },
            },
            "param": {
                "type": "object", "additionalProperties": False,
                "required": ["value", "severity"],
                "properties": {
                    "value": {},
                    "severity": {"enum": list(_SEVERITIES),
                                 "description": "what changing this number costs: reuse the ckpt "
                                                "(run), adapt it, or retrain"},
                    "doc": {"type": "string"},
                },
            },
            "assert": {
                "type": "object", "additionalProperties": False,
                "anyOf": [{"required": ["min"]}, {"required": ["max"]}, {"required": ["expect"]}],
                "properties": {"min": {"type": "number"}, "max": {"type": "number"},
                               "expect": {}, "needs": {"type": "string"}},
            },
        },
    }
