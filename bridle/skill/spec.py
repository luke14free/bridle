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
  grades `height_above_seat` against the LIVE seat while descend_stack's success gate grades a goal
  frozen at init — one name, two truths (vocab.py, correction 2). So `height_above_seat` on its own
  is refused and the author writes `height_above_seat_live` or `height_above_seat_static_goal`.

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
import copy
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
    "ROW_TERMS", "MEASURE_ALIASES", "LEGAL_MEASURE_NAMES", "AMBIGUOUS_QUANTITIES",
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


# ── measures: frames, aliases, and the one ambiguous quantity ───────────────────────────────────
# A measure name in MEASURES already encodes its frame in the KEY (`height_above_seat` is the live
# one, `height_above_seat_static_goal` the frozen one) — vocab.py made them distinct entries on
# purpose. What the key does NOT do is force the author to have thought about which they meant, and
# for the one quantity that exists twice that choice is the difference between grading the reward on
# the live seat and grading it on a goal captured at init. So the bare quantity name is refused and
# both frames are addressable by an explicit `<quantity>_<frame>` spelling.

_FRAME_SUFFIXES = tuple(f"_{f.value}" for f in Frame)


def _quantity(name):
    """The measure name with its frame qualifier stripped: the physical quantity it reads."""
    for suffix in _FRAME_SUFFIXES:
        if name.endswith(suffix) and len(name) > len(suffix):
            return name[: -len(suffix)]
    return name


_FAMILY = {}
for _key, _measure in MEASURES.items():
    _FAMILY.setdefault(_quantity(_key), {})[_measure.frame.value] = _key

#: quantity -> {frame: MEASURES key}, for every quantity readable in more than one frame. Exactly
#: one today (`height_above_seat`), which is the one vocab.py's correction 2 was written about.
AMBIGUOUS_QUANTITIES = MappingProxyType(
    {q: MappingProxyType(dict(frames)) for q, frames in _FAMILY.items() if len(frames) > 1})

#: every legal way to write a measure -> the MEASURES key it means. For an unambiguous quantity that
#: is just the key; for an ambiguous one it is the two frame-qualified spellings and NOT the bare
#: name, so `Row.params["measure"]` is always a real MEASURES key and an author always said which
#: frame. Public because compile() binds `Expr.names`, which carry the authored spelling.
MEASURE_ALIASES = {}
for _quant, _frames in _FAMILY.items():
    if len(_frames) > 1:
        for _frame_name, _key in _frames.items():
            MEASURE_ALIASES[f"{_quant}_{_frame_name}"] = _key
    else:
        MEASURE_ALIASES[next(iter(_frames.values()))] = next(iter(_frames.values()))
MEASURE_ALIASES = MappingProxyType(MEASURE_ALIASES)

LEGAL_MEASURE_NAMES = frozenset(MEASURE_ALIASES)


def _ambiguous_frame_error(path, name):
    frames = AMBIGUOUS_QUANTITIES[name]
    options = ", ".join(f"{name}_{frame!s}" for frame in sorted(frames))
    return SpecError(
        path,
        f"measure {name!r} names a quantity that is readable in {len(frames)} frames "
        f"({', '.join(sorted(frames))}) and this row names neither — say which frame you mean: "
        f"{options}. descend's reward grades this against the live seat while descend_stack's "
        f"success gate grades it against a goal frozen at init: one name, two truths",
        suggestion=f"{name}_{Frame.LIVE.value}" if Frame.LIVE.value in frames else None)


def _resolve_measure(path, name):
    """An author-written measure reference -> its MEASURES key. Raises on unknown or frame-ambiguous."""
    if not isinstance(name, str):
        raise SpecError(path, f"a measure is named by a string, got {type(name).__name__}",
                        legal=LEGAL_MEASURE_NAMES)
    if name in MEASURE_ALIASES:
        return MEASURE_ALIASES[name]
    if name in AMBIGUOUS_QUANTITIES:
        raise _ambiguous_frame_error(path, name)
    raise _unknown(path, "measure", name, LEGAL_MEASURE_NAMES)


def _canonical_default_measure(name):
    """The same resolution for a measure the VOCABULARY wrote (a chassis default), which names the
    bare quantity: carry's rows were read off descend_env.py, which grades the live seat. Resolving
    it as live keeps the vocabulary's own defaults inheritable — the ambiguity refusal is aimed at
    what the AUTHOR wrote, which is the choice nobody has made yet.
    """
    if name in MEASURE_ALIASES:
        return MEASURE_ALIASES[name]
    return _FAMILY.get(name, {}).get(Frame.LIVE.value, name)


# ── predicates and per-skill param references ───────────────────────────────────────────────────

# A predicate field is either a bare name (`grasped`) or a nested call over existing names
# (`and_(grasped, above_z(z=0.06))`). Names in CALL position are predicate references; everything
# else inside the parens is scene data (`anchor=target_pos`) and is not checked here.
_CALL_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(")
_PARAM_REF_RE = re.compile(r"\bparams\.([A-Za-z_][A-Za-z0-9_]*)")


def _predicate_names(text):
    calls = set(_CALL_RE.findall(text))
    return calls if calls else {text.strip()}


def _check_predicate(path, value, declared_params):
    if not isinstance(value, str) or not value.strip():
        raise SpecError(path, "a predicate is named by a non-empty string (a bare name, or a call "
                              "over existing names like `and_(grasped, above_z(z=0.06))`)",
                        legal=PREDICATES)
    _check_param_refs(path, value, declared_params)
    for name in sorted(_predicate_names(value)):
        if name not in PREDICATES:
            raise _unknown(path, "predicate", name, PREDICATES)


def _check_param_refs(path, text, declared_params):
    """Every `params.X` in an authored string must resolve to a param the document declared."""
    for name in _PARAM_REF_RE.findall(text):
        if name not in declared_params:
            raise SpecError(path, f"`params.{name}` is not declared in this document's `params:` "
                                  f"block, so there is nothing for compile() to bind it to",
                            legal=declared_params, suggestion=_suggest(name, declared_params))


def _is_param_ref(value):
    return isinstance(value, str) and value.startswith("params.")


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


def _chassis_defaults_for(chassis, term_name, authored):
    """The chassis row this authored row inherits from, or {}.

    A chassis may instantiate one term several times under suffixed keys (carry has `DistancePull_xy`
    at k=4.0 and `DistancePull_height` at k=6.0 — collapsing them into one 3D kernel changes the
    task). When it does, the authored `measure`/`predicate` is what picks the right one; with no
    discriminator nothing is inherited and the term's own defaults apply, because guessing between
    1.5@k=4 and 2.5@k=6 is exactly the silent substitution this whole module exists to prevent.
    """
    if chassis is None:
        return {}
    candidates = [row for key, row in chassis.defaults.items() if base_term(key) == term_name]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        return {}
    for field in ("measure", "predicate"):
        if field not in authored:
            continue
        want = authored[field]
        if field == "measure" and isinstance(want, str):
            want = MEASURE_ALIASES.get(want, want)
        hits = []
        for row in candidates:
            have = row.get(field)
            if have is None:
                continue
            if field == "measure":
                have = _canonical_default_measure(have)
            if have == want:
                hits.append(row)
        if len(hits) == 1:
            return hits[0]
    return {}


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

    inherited = _chassis_defaults_for(chassis, term_name, authored)
    values = {}
    for param in term.params:
        if param.name in authored:
            value, from_author = authored[param.name], True
        elif param.name in inherited:
            value, from_author = inherited[param.name], False
        elif param.required:
            raise SpecError(
                f"{path}.{param.name}",
                f"{param.name!r} is required by {term_name} ({param.doc or param.type}) and "
                f"neither the row nor the '{chassis.name if chassis else 'none'}' chassis "
                f"supplies one")
        else:
            value, from_author = param.default, False

        if value is None:
            values[param.name] = None
            continue
        field_path = f"{path}.{param.name}"
        if param.name == "measure":
            value = (_resolve_measure(field_path, value) if from_author
                     else _canonical_default_measure(value))
        elif param.name in ("predicate", "gate"):
            _check_predicate(field_path, value, declared_params)
        elif isinstance(value, str):
            _check_param_refs(field_path, value, declared_params)
        if not _is_param_ref(value):
            # A `params.X` stands in for a number compile() binds later, so its type is the param's
            # to answer for, not this field's.
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
                legal=[n for n in LEGAL_MEASURE_NAMES
                       if MEASURES[MEASURE_ALIASES[n]].sign is Sign.SIGNED])

    return Row(term=term_name, params=MappingProxyType(values), expr=None, custom=None,
               why=raw["why"])


def _parse_expr_row(path, raw, declared_params):
    for key in raw:
        if key not in ("expr", "why"):
            raise SpecError(f"{path}.{key}",
                            f"an `expr:` row takes only `expr` and `why`, not {key!r} — an "
                            f"expression gates and scales itself (`... * grasped`)")
    try:
        parsed = parse_expr(raw["expr"])
    except ExprError as exc:
        raise SpecError(f"{path}.expr", str(exc)) from exc
    for name in sorted(parsed.names):
        # A declared param shadows a measure of the same name: the author wrote the param, so the
        # author meant the param. Nothing in the corpus collides today.
        if name in declared_params or name in PREDICATES or name in MEASURE_ALIASES:
            continue
        if name in AMBIGUOUS_QUANTITIES:
            raise _ambiguous_frame_error(f"{path}.expr", name)
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
        out[name] = copy.deepcopy(body)
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
    return copy.deepcopy(raw)


def _parse_reward_scale(raw, chassis):
    """`reward_scale:` — `reward_ppo = dense / divisor`, stated explicitly (§1.4). 7 of 15 primitives
    inherit `compute_normalized_dense_reward` without overriding it and so train at dense/12.0 even
    where that is semantically wrong (lift's per-step max is ~18, reach's ~9); a generated env that
    forgets the field entirely trains at 12x the intended scale.
    """
    term = TERMS["RewardScale"]
    declared = {p.name: p for p in term.params}
    authored = raw if raw is not None else {}
    if not isinstance(authored, dict):
        raise SpecError("reward_scale", f"`reward_scale:` is a mapping of "
                                        f"{{divisor, unnormalized}}, got {type(authored).__name__}")
    for key in authored:
        if key not in declared:
            raise _unknown(f"reward_scale.{key}", "reward_scale field", key, declared)
    inherited = chassis.defaults.get("RewardScale", {}) if chassis else {}
    values = {}
    for param in term.params:
        if param.name in authored:
            value = authored[param.name]
            _check_type(f"reward_scale.{param.name}", param, value)
        elif param.name in inherited:
            value = inherited[param.name]
        else:
            value = param.default
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
    return copy.deepcopy(raw)


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

    reward_scale = _parse_reward_scale(doc.get("reward_scale"), chassis)

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
        scene=MappingProxyType(scene), init=MappingProxyType(copy.deepcopy(init)),
        params=MappingProxyType(params), reward_scale=MappingProxyType(reward_scale),
        reward=reward, success=success, preflight=MappingProxyType(preflight))


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
    cleanly; a name two terms constrain differently (`mode` is a closed choice set on
    PredicateBonus and open on SuccessBonus) widens to the shared JSON type rather than picking a
    winner — a schema that rejected a legal document would be worse than one that admits an extra.
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
                             "description": "arithmetic over measures, predicates and params: "
                                            "+ - * / **, comparisons, and abs tanh exp log sqrt "
                                            "clamp min max where"},
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
