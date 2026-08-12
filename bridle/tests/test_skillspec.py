"""Unit test for bridle.skill.spec — the skill document and its schema.

WHY THIS EXISTS: the intended AUTHOR of a skill document is a local 27-30B LLM that cannot read this
Python. For that author **the error messages ARE the API** (design doc §8): a refusal that does not
name the path, state the legal set and suggest the nearest match is a refusal the model cannot
self-correct from, and the measured cost of no-feedback authoring is 58.3% +/- 47.3% one-shot vs
97.6% with refinement (amendment 1 §B). So almost every check below asserts on the TEXT of a
refusal, not merely that one happened.

The other properties paid for by a measured failure:

  WHY        mandatory on every row (amendment 1 §B3). L2R took 50% -> 90% by making the model state
             the rationale before emitting the number, and §10's "YAML drops the comments" means the
             `why` is the only surviving record of why a weight is what it is.
  SIGN       HingePenalty over an unsigned measure makes `clamp(-sdz, min=0)` identically zero and
             silently deletes the term that exists because pressing to dz=0 broke 16/16 grasps
             (2026-06-04). The row still trains — it just trains without the term.
  FRAME      descend's reward grades the seat height against the live seat while descend_stack's
             success grades a frozen goal. One quantity, two frames, so two names.
  DEFAULTS   a chassis supplies the deployed weights; the diff against them is what gets reviewed.

Run: python -m pytest bridle/tests/test_skillspec.py
     PYTHONPATH=. python bridle/tests/test_skillspec.py

THREE DEVIATIONS FROM THE BRIEF'S SKETCH, all forced by the vocabulary as it actually shipped:

  1. The sketch's unknown-term case is `"Hovar" -> "HoverAt"`. `HoverAt` is not one of the nine terms
     (vocab.py asserts `len(TERMS) == 9`), so no implementation can suggest it. Split in two below:
     a realistic typo (`DistancePul` -> `DistancePull`) proves the nearest-match hint, and `Hovar`
     — which difflib matches to nothing — proves the fallback still states the legal set.
  2. The sketch's unknown-measure case is `"hieght_above" -> "height_above"`; the full measure name
     is used (`hieght_above_seat`), which contains both fragments verbatim.
  3. §4's document declares one param (`hover`) but its success line references `params.low_band`
     and `params.center_tol`. That document is internally inconsistent; the fixture declares all
     three, because a `params.X` that resolves to nothing is exactly what parse_spec must refuse.

  ...and one deviation the ambiguous-frame rule forces: §4 writes `measure: height_above_seat`,
  which this schema refuses as frame-ambiguous (two frames exist for that quantity, so the row must
  name one). The fixture writes `height_above_seat_live`, and the bare form is the frame test.

  (2026-08-12 review, finding 4) That rule used to be enforced by an ALIAS layer in spec.py, because
  `height_above_seat` was itself the MEASURES key for the live reading — so the legal set and the
  key set disagreed on exactly that name, a chassis default wrote the illegal spelling and a second
  code path resolved it silently, and `Row.params["measure"]` stored a spelling the parser refuses.
  The vocabulary was renamed instead (`height_above_seat_live`); the alias layer is gone, the legal
  set IS `MEASURES`, and `test_round_trips` below is the invariant that keeps it that way.
"""
import json
import sys

from bridle.skill.spec import (
    LEGAL_MEASURE_NAMES, Row, SkillSpec, SpecError, json_schema, parse_spec,
)
from bridle.skill.vocab import CHASSIS, MEASURES, TERMS, Sign

FAILS = []

DROP = object()  # sentinel: remove this key rather than set it


def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        FAILS.append(name)


def descend_doc():
    """A fresh copy of the §4 `descend_to_target` document — the acceptance skill of this phase —
    as the dict a YAML loader would hand `parse_spec`. Every weight is the deployed value, so a
    change here is a change against a trained lineage, not against a guess. See the module
    docstring for the four places this differs from §4 as printed.
    """
    return {
        "name": "descend_to_target",
        "kind": "carry",
        "contract": "stack",
        "env_id": "SO100DescendToTarget-v1",
        "scene": {
            "goal": {"type": "platform", "half": 0.04, "top_z": 0.03},
            "held": {"type": "cube", "half": [0.014, 0.016]},
            "clutter": {"type": "cube", "count": [0, 4], "ring": [0.055, 0.125]},
        },
        "init": {"snapshot": "descend_init"},
        "params": {
            "hover": {"value": 0.015, "severity": "retrain",
                      "doc": "reward attractor height above resting"},
            "low_band": {"value": 0.01, "severity": "retrain",
                         "doc": "band height_above_resting_in accepts as 'seated'"},
            "center_tol": {"value": 0.012, "severity": "retrain",
                           "doc": "lateral tolerance; 12mm is the measured stacking basin"},
        },
        "reward_scale": {"divisor": 12.0},
        "reward": [
            {"term": "PredicateBonus", "weight": 1.0, "predicate": "grasped",
             "why": "hold-on baseline — never drop the cube, release is a separate primitive."},
            {"term": "DistancePull", "weight": 1.5, "measure": "object_to_goal_xy",
             "kernel": "one_minus_tanh", "k": 4.0, "gate": "grasped",
             "why": "re-center over the target while held; move delivers ~6cm off. k=4 on xy is "
                    "deliberately different from k=6 on height."},
            {"term": "DistancePull", "weight": 2.5, "measure": "height_above_seat_live",
             "kernel": "one_minus_tanh", "k": 6.0, "setpoint": "params.hover", "gate": "grasped",
             "why": "the attractor peaks at the hover height and NEVER at the seat; setpoint=0 "
                    "pulled the cube into the platform and broke 16/16 grasps (2026-06-04)."},
            {"term": "HingePenalty", "weight": 3.0, "measure": "height_above_seat_live",
             "threshold": 0.0, "side": "below", "gate": "grasped",
             "why": "the other half of the 2026-06-04 slip fix: pressing the cube below the seat "
                    "destabilizes the grasp."},
            {"term": "HingePenalty", "weight": 1.0, "measure": "gripper_qpos", "threshold": -0.6,
             "side": "above", "gate": "grasped",
             "why": "keep the gripper closed while held — grip_q drifted -0.73 to -0.44 over the "
                    "descent before this term existed."},
            {"term": "VelocityPenalty", "body": "held", "linear_weight": 0.3,
             "angular_weight": 0.05,
             "why": "anti-fling on the way down; 0.3/0.05 identical across all four carry members."},
            {"term": "PredicateBonus", "weight": -0.5, "predicate": "not_grasped",
             "why": "discourage letting go early — it collapses the handoff to the next primitive."},
            {"term": "SuccessBonus", "value": 12.0, "mode": "replace", "scope": "preceding",
             "why": "the held+low+centered jackpot REPLACES accumulated shaping rather than adding "
                    "to it, and fires before the action penalty below."},
            {"term": "ActionPenalty", "weight": 0.001, "norm": "l2",
             "why": "same 0.001/l2 as all 15 primitives; applied after the replace so it survives."},
        ],
        "success": "all[grasped, height_above_resting_in(params.low_band), "
                   "centered_on_goal(params.center_tol)]",
        "preflight": {
            "static": {"descend_env._CENTER_TOL": {"max": 0.030}},
            "dynamic": {"descend_low_once": {"min": 0.5, "needs": "warm_start"},
                        "success_once": {"max": 0.99}},
        },
    }


def doc_with(**overrides):
    """The fixture with top-level fields replaced (or DROPped)."""
    d = descend_doc()
    for k, v in overrides.items():
        if v is DROP:
            d.pop(k, None)
        else:
            d[k] = v
    return d


def row_edited(index, **changes):
    """The fixture with one reward row's keys set or DROPped."""
    d = descend_doc()
    row = d["reward"][index]
    for k, v in changes.items():
        if v is DROP:
            row.pop(k, None)
        else:
            row[k] = v
    return d


def error_from(doc):
    """Parse and return whatever was raised (or None). Deliberately catches everything: a raw
    KeyError/TypeError escaping parse_spec is a bug — the author only ever catches SpecError."""
    try:
        parse_spec(doc)
    except BaseException as exc:      # noqa: BLE001 — see docstring
        return exc
    return None


def accepts(label, doc):
    """The mirror of `refuses`: assert the document parses, and hand back the spec (None if not).

    Used where a plausible REGRESSION would refuse a legal document — an uncaught SpecError there
    would abort the run and hide every check after it, which is exactly what a mutation test needs
    not to happen.
    """
    exc = error_from(doc)
    check(f"{label}: accepted", exc is None)
    return parse_spec(doc) if exc is None else None


def refuses(label, doc, *fragments, path=None, suggestion=None):
    """Assert the document is refused with a message a model can act on: SpecError, and every
    fragment present in `str(exc)` verbatim."""
    exc = error_from(doc)
    check(f"{label}: refused with SpecError", isinstance(exc, SpecError))
    msg = str(exc) if exc is not None else "<nothing raised>"
    for frag in fragments:
        check(f"{label}: message says {frag!r}", frag in msg)
    if path is not None:
        check(f"{label}: path is {path!r}", getattr(exc, "path", None) == path)
    if suggestion is not None:
        check(f"{label}: suggests {suggestion!r}", getattr(exc, "suggestion", None) == suggestion)
    return exc


def run_checks():
    # ── the acceptance document parses, unchanged in shape and order ────────────────────────────
    spec = parse_spec(descend_doc())
    check("a well-formed spec parses", isinstance(spec, SkillSpec))
    check("rows keep their order",
          [r.term for r in spec.reward][:2] == ["PredicateBonus", "DistancePull"])
    check("all nine rows survive", len(spec.reward) == 9)
    check("reward is an immutable tuple", isinstance(spec.reward, tuple))
    check("rows are Rows", all(isinstance(r, Row) for r in spec.reward))
    check("the document's own fields come through", (spec.name, spec.kind, spec.contract,
          spec.env_id) == ("descend_to_target", "carry", "stack", "SO100DescendToTarget-v1"))
    check("scene survives", spec.scene["goal"]["type"] == "platform")
    check("init survives", spec.init["snapshot"] == "descend_init")
    check("success survives verbatim", spec.success.startswith("all[grasped,"))
    check("preflight survives", spec.preflight["static"]["descend_env._CENTER_TOL"]["max"] == 0.030)
    check("params survive with their severity", spec.params["hover"]["severity"] == "retrain")
    check("a params.X row value is left as a reference for compile() to bind",
          spec.reward[2].params["setpoint"] == "params.hover")

    check("a non-mapping document is refused, not crashed",
          isinstance(error_from(None), SpecError) and isinstance(error_from([]), SpecError))
    try:
        spec.name = "other"
        mutable = True
    except Exception:
        mutable = False
    check("SkillSpec is frozen", not mutable)
    try:
        spec.reward[0].params["weight"] = 99.0
        row_mutable = True
    except Exception:
        row_mutable = False
    check("a Row's params cannot be mutated in place", not row_mutable)

    # (2026-08-12 review, minor 1) `frozen=True` only stops the FIELD being rebound. Every value
    # below hashes into the contract fingerprint that records which policy was trained under which
    # spec, so a nested container that can be edited after construction is a fingerprint that stops
    # describing the run.
    def mutates(fn):
        try:
            fn()
            return True
        except Exception:
            return False

    for label, mutate in (
        ("scene", lambda: spec.scene["goal"].__setitem__("type", "MUTATED")),
        ("params", lambda: spec.params["hover"].__setitem__("value", 99.0)),
        ("preflight", lambda: spec.preflight["static"]["descend_env._CENTER_TOL"]
                                  .__setitem__("max", 99.0)),
        ("init", lambda: spec.init.__setitem__("snapshot", "MUTATED")),
        ("reward_scale", lambda: spec.reward_scale.__setitem__("divisor", 99.0)),
        ("a scene list", lambda: spec.scene["held"]["half"].append(0.02)),
    ):
        check(f"{label} cannot be mutated after construction", not mutates(mutate))
    check("the fixture really does carry a nested list for that last check to bite on",
          isinstance(descend_doc()["scene"]["held"]["half"], list))

    # parsing must not write defaults back into the caller's dict
    doc = descend_doc()
    parse_spec(doc)
    check("parse_spec does not mutate the document it was handed",
          doc == descend_doc())

    # ── chassis defaults: the deployed weights, and what an author's diff against them means ────
    # DistancePull's TERM default weight is 1.0; the carry chassis' xy row is 1.5 — so this check
    # can only pass if the CHASSIS supplied it.
    check("DistancePull's term default is 1.0, the carry chassis' xy row is 1.5 (the test bites)",
          [p.default for p in TERMS["DistancePull"].params if p.name == "weight"] == [1.0]
          and CHASSIS["carry"].defaults["DistancePull_xy"]["weight"] == 1.5)
    filled = parse_spec(row_edited(1, weight=DROP))
    check("chassis defaults fill an omitted weight", filled.reward[1].params["weight"] == 1.5)
    over = parse_spec(row_edited(1, weight=0.25))
    check("an explicit weight overrides the chassis default", over.reward[1].params["weight"] == 0.25)
    # the carry chassis instantiates DistancePull twice; the row is matched by its measure, so the
    # height row inherits k=6.0 and not the xy row's k=4.0.
    sharp = parse_spec(row_edited(2, k=DROP))
    check("the chassis row is matched by measure, not just by term name",
          sharp.reward[2].params["k"] == 6.0)
    # neither authored nor in the carry chassis -> the term's own default
    check("term defaults fill what the chassis does not mention",
          spec.reward[7].params["predicate_ref"] == "per_step")
    check("reward_scale comes through", spec.reward_scale["divisor"] == 12.0)
    check("an omitted reward_scale falls back to the inherited 12.0",
          parse_spec(doc_with(reward_scale=DROP)).reward_scale["divisor"] == 12.0)

    # ── EVERY refusal must be self-correctable by a model that cannot introspect the API ────────
    refuses("unknown term (typo)", row_edited(0, term="DistancePul"),
            "DistancePul", "DistancePull", path="reward[0].term", suggestion="DistancePull")
    refuses("unknown term (no near match) still states the legal set",
            row_edited(0, term="Hovar"), "Hovar", "PredicateBonus", "DistancePull",
            path="reward[0].term")
    refuses("unknown measure", row_edited(2, measure="hieght_above_seat"),
            "hieght_above_seat", "height_above", path="reward[2].measure")
    refuses("HingePenalty over a magnitude measure", row_edited(3, measure="tcp_to_object"),
            "sign", "signed", "HingePenalty", "tcp_to_object", path="reward[3].measure")
    refuses("a quantity with two frames must be named with one",
            row_edited(2, measure="height_above_seat"),
            "frame", "live", "height_above_seat_static_goal", path="reward[2].measure")
    refuses("a row with no why", row_edited(4, why=DROP), "why", "rationale", path="reward[4].why")
    refuses("a row whose why is blank", row_edited(4, why="   "), "why", "rationale")
    refuses("a row whose why is not prose", row_edited(4, why=1.0), "why", "rationale")

    # ── finding 4: one name set, so a spec survives being re-emitted and re-parsed ──────────────
    # The invariant that made the finding visible: the schema's legal set and the vocabulary's key
    # set are the SAME set. When they were not, `height_above_seat_live` was legal-but-not-a-key and
    # `height_above_seat` was a-key-but-not-legal, and the difference was papered over by an alias
    # table on one path and a silent live-frame default on the other.
    check("the legal measure names ARE the MEASURES keys (no alias layer)",
          set(LEGAL_MEASURE_NAMES) == set(MEASURES))
    check("a stored measure is the name it was written under",
          spec.reward[2].params["measure"] == "height_above_seat_live")
    check("the frozen-goal measure is nameable directly",
          parse_spec(row_edited(2, measure="height_above_seat_static_goal")
                     ).reward[2].params["measure"] == "height_above_seat_static_goal")
    # re-emit every resolved measure and re-parse it: a spec that cannot round-trip cannot be
    # written back to YAML, diffed, or fingerprinted from its own text.
    reparsed = parse_spec(row_edited(2, measure=spec.reward[2].params["measure"]))
    check("a resolved measure re-parses to itself (round trip)",
          reparsed.reward[2].params["measure"] == spec.reward[2].params["measure"])
    check("every chassis-supplied measure is a legal authored spelling too",
          all(row["measure"] in LEGAL_MEASURE_NAMES
              for c in CHASSIS.values() for row in c.defaults.values() if "measure" in row))
    # a chassis-supplied measure now goes through the SAME validation as an authored one — there is
    # no second resolution path that could hand `MEASURES[...]` a key it never checked
    inherited_measure = parse_spec({
        "name": "reach", "kind": "approach", "contract": "reach", "env_id": "SO100Reach-v1",
        "scene": {"held": {"type": "cube", "half": 0.014}},
        "reward": [{"term": "DistancePull", "why": "the approach chassis supplies the measure."}],
        "success": "grasped",
    })
    check("a chassis-supplied measure lands as a real MEASURES key",
          inherited_measure.reward[0].params["measure"] in MEASURES)

    # ── the rest of the vocabulary is checked too, not just terms and measures ──────────────────
    refuses("unknown chassis", doc_with(kind="cary"), "cary", "carry", path="kind",
            suggestion="carry")
    refuses("unknown predicate", row_edited(0, predicate="grapsed"), "grapsed", "grasped",
            path="reward[0].predicate", suggestion="grasped")
    refuses("unknown predicate in CALL position inside a gate",
            row_edited(1, gate="and_(grasped, abov_z(z=0.06))"),
            "abov_z", "above_z", path="reward[1].gate")
    # ── finding 2: the BARE operands of a compound predicate are names too ──────────────────────
    # The check above was labelled "unknown predicate inside a gate" but the typo it plants is in
    # CALL position, which the first implementation already looked up. The operand position — where
    # `and_(grasped, ...)` actually puts a predicate, and the shape vocab.py's own chassis write —
    # was never checked at all, so this parsed clean and compile() would bind nothing.
    refuses("a typo'd BARE operand inside a compound gate",
            row_edited(1, gate="and_(grapsed, above_z(z=0.06))"),
            "grapsed", "grasped", path="reward[1].gate", suggestion="grasped")
    refuses("a typo'd BARE operand inside a compound predicate",
            row_edited(0, predicate="or_(grasped, not_grapsed)"),
            "not_grapsed", "not_grasped", path="reward[0].predicate", suggestion="not_grasped")
    refuses("a predicate string that names nothing at all",
            row_edited(0, predicate="!!!"), "names no predicate", path="reward[0].predicate")
    check("a compound gate over real predicates is accepted",
          parse_spec(row_edited(1, gate="and_(grasped, above_z(z=0.06))")
                     ).reward[1].params["gate"].startswith("and_("))
    # ...and the price of checking bare names must not be false refusals: a keyword argument's VALUE
    # is scene data (`anchor=target_pos` is a point, not a predicate), and every predicate string
    # the shipped vocabulary writes has to keep parsing. carry_with_potential writes the hardest one.
    chassis_predicates = sorted({row[field] for c in CHASSIS.values() for row in c.defaults.values()
                                 for field in ("predicate", "gate") if field in row})
    check("the vocabulary writes more than one compound predicate (the check bites)",
          sum("(" in p for p in chassis_predicates) >= 2)
    for text in chassis_predicates:
        check(f"a chassis' own predicate is accepted verbatim: {text[:48]}",
              error_from(row_edited(0, predicate=text)) is None)
    refuses("unknown row parameter", row_edited(0, wieght=1.0), "wieght", "weight",
            path="reward[0].wieght", suggestion="weight")
    refuses("a parameter of the wrong type", row_edited(0, weight="heavy"),
            "weight", "float", path="reward[0].weight")
    refuses("a value outside a declared choice set", row_edited(0, mode="clobber"),
            "clobber", "replace", path="reward[0].mode")

    # ── minor 3: a legal set stated only in prose cannot be checked ──────────────────────────────
    # (2026-08-12 re-review) `SuccessBonus.predicate_ref` declared `choices=()` with `per_step |
    # latched` in its doc text alone, so `predicate_ref: "per_stpe"` parsed clean and reached the
    # fold — in the module whose premise is that every refusal names the legal set. `kernel`, `side`
    # and `norm` had the same gap, and SuccessBonus' `mode` was worse than prose-only (its prose said
    # `add | replace` while the fold has always honoured `floor`). The sets are compile.py's
    # `_HONOURED` table, so the two tiers cannot disagree about what is legal.
    refuses("a typo'd predicate_ref", row_edited(7, predicate_ref="per_stpe"),
            "per_stpe", "per_step", "latched", path="reward[7].predicate_ref",
            suggestion="per_step")
    refuses("a typo'd HingePenalty side", row_edited(3, side="abve"), "abve", "above", "below",
            path="reward[3].side", suggestion="above")
    refuses("a typo'd DistancePull kernel", row_edited(1, kernel="one_minus_tan"),
            "one_minus_tan", "one_minus_tanh", "neg_linear", path="reward[1].kernel",
            suggestion="one_minus_tanh")
    refuses("a norm the fold does not implement", row_edited(8, norm="l3"), "l3", "l2",
            path="reward[8].norm")
    refuses("a SuccessBonus mode outside the set the fold honours", row_edited(7, mode="multiply"),
            "multiply", "replace", path="reward[7].mode")
    check("...and the values the vocabulary now declares legal are still the deployed ones",
          parse_spec(descend_doc()).reward[7].params["predicate_ref"] == "per_step" and
          parse_spec(descend_doc()).reward[3].params["side"] == "below")
    refuses("a required parameter with no default anywhere",
            doc_with(reward=[{"term": "Ramp", "measure": "object_z", "why": "ramp the lift."}]),
            "cap", "required", path="reward[0].cap")

    # ── finding 3: the required-parameter refusal must be true and actionable ───────────────────
    # The carry chassis supplies TWO DistancePull rows; nothing was inherited because the row names
    # no measure to choose between them. The refusal used to say "neither the row nor the 'carry'
    # chassis supplies one" — the opposite of the truth — with no legal set and no suggestion, the
    # only refusal in the module with neither. What the author needs is the candidates.
    ambiguous_row = refuses(
        "a required parameter the chassis supplies TWICE names both candidates",
        doc_with(reward=[{"term": "DistancePull", "why": "pull it in."}]),
        "required", "carry", "2 DistancePull rows", "object_to_goal_xy", "height_above_seat_live",
        path="reward[0].measure")
    check("...and does NOT claim the chassis supplies nothing",
          "chassis supplies one" not in str(ambiguous_row))
    check("...and states a legal set, unconditionally",
          "legal values:" in str(ambiguous_row))
    check("...and shows what picking a candidate would inherit (the weights that differ)",
          "1.5" in str(ambiguous_row) and "2.5" in str(ambiguous_row))
    # naming one of them resolves it — the refusal's own instruction has to work
    check("naming the candidate's measure is the fix the message asks for",
          parse_spec(doc_with(reward=[{"term": "DistancePull", "measure": "object_to_goal_xy",
                                       "why": "pull it in."}])).reward[0].params["weight"] == 1.5)
    # the same shape over a predicate-discriminated pair (carry's grasped / not_grasped bonuses)
    refuses("...and the same for a predicate-discriminated pair",
            doc_with(reward=[{"term": "PredicateBonus", "why": "bonus."}]),
            "required", "2 PredicateBonus rows", "grasped", "not_grasped",
            path="reward[0].predicate")

    # ── finding 1: a key present with a null value is NOT a supplied value ──────────────────────
    # `measure:` with nothing after it is the commonest LLM YAML slip there is. Treating that None
    # as authored short-circuited required-ness, typing, `choices` AND the signed-measure gate (whose
    # guard reads `is not None`), so `HingePenalty` with `measure: null` parsed and stored None.
    refuses("a null measure on HingePenalty is refused, not stored",
            row_edited(3, measure=None), "measure", "required", path="reward[3].measure")
    check("a null key and an absent key are refused identically",
          str(error_from(row_edited(3, measure=None))) == str(error_from(row_edited(3,
                                                                                   measure=DROP))))
    # (2026-08-12 re-review, minor 2) This read as a global invariant — "no accepted row leaves a
    # needs_signed_measure term without a measure" — while iterating TWO hand-picked documents,
    # NEITHER of which contains a null measure, so it could not fail under the defect it appears to
    # guard. Generalised over every way a measure can reach the parser; the null and absent arms are
    # the ones the old form never reached, and they are the ones `_parse_term_row`'s null rule and
    # the signed gate's `is not None` guard are for.
    signed_terms = sorted(t for t, term in TERMS.items() if term.needs_signed_measure)
    check("the vocabulary declares a needs_signed_measure term (the loop has something to bite on)",
          signed_terms == ["HingePenalty"])
    leaks = []
    for term_name in signed_terms:
        for label, variant in (("authored signed", "height_above_seat_live"),
                               ("authored magnitude", "tcp_to_object"),
                               ("authored null", None),
                               ("absent", DROP),
                               ("not a string", 0.0),
                               ("unknown name", "hieght_above_seat_live")):
            d = row_edited(3, term=term_name, measure=variant)
            exc = error_from(d)
            if exc is None:
                got = parse_spec(d).reward[3].params.get("measure")
                if got not in MEASURES or MEASURES[got].sign is not Sign.SIGNED:
                    leaks.append(f"{term_name}/{label} accepted -> measure={got!r}")
            elif not isinstance(exc, SpecError):
                leaks.append(f"{term_name}/{label} raised {type(exc).__name__}: {exc}")
    check("every way a measure can reach a needs_signed_measure term either refuses with a "
          "SpecError or lands a real SIGNED measure" + (f" (leaks: {leaks})" if leaks else ""),
          not leaks)
    refuses("a null predicate on PredicateBonus is refused",
            row_edited(0, predicate=None), "required", path="reward[0].predicate")
    refuses("a null cap does not satisfy Ramp's required cap",
            doc_with(reward=[{"term": "Ramp", "measure": "object_z", "cap": None,
                              "why": "ramp the lift."}]),
            "cap", "required", path="reward[0].cap")
    # ...and the other direction: a null falls THROUGH, exactly like an absent key
    nulled = parse_spec(row_edited(0, weight=None))
    check("a null weight falls through to the chassis default rather than storing None",
          nulled.reward[0].params["weight"] == 1.0)
    check("a null optional still ends up at the term's own default, not None",
          parse_spec(row_edited(0, mode=None)).reward[0].params["mode"] == "add")
    # (2026-08-12 re-review, minor 1) This pair used to ship under the label "a term default that IS
    # None still comes through as None (axes/gate are optional)" while asserting `scope ==
    # "preceding"` (a non-None TERM default) and `gate == "grasped"` (a CHASSIS-inherited value, on
    # PredicateBonus, which has no `gate` parameter at all). Nothing in the file asserted the None
    # branch — `_parse_term_row`'s `values[name] = None; continue` arm, the one the signed-measure
    # gate's `is not None` guard is written around. Relabelled to what it checks, plus a real one.
    check("an omitted optional falls back to the term default, and to the chassis value where the "
          "chassis has one",
          spec.reward[0].params.get("scope") == "preceding" and
          parse_spec(row_edited(1, gate=DROP)).reward[1].params["gate"] == "grasped")
    approach_pull = parse_spec({
        "name": "reach", "kind": "approach", "contract": "reach", "env_id": "SO100Reach-v1",
        "scene": {"held": {"type": "cube", "half": 0.014}},
        "reward": [{"term": "DistancePull", "why": "the approach chassis supplies the measure."}],
        "success": "grasped",
    }).reward[0]
    optional_nones = sorted(p.name for p in TERMS["DistancePull"].params
                            if p.default is None and not p.required)
    check("DistancePull declares None-defaulted optionals, and the approach chassis supplies "
          "neither (the check bites)",
          optional_nones == ["axes", "gate"] and
          not any(n in CHASSIS["approach"].defaults["DistancePull"] for n in optional_nones))
    check("a term default that IS None comes through as a present None, not a missing key",
          all(n in approach_pull.params and approach_pull.params[n] is None
              for n in optional_nones))
    check("an authored null on a single-candidate chassis inherits the chassis value",
          parse_spec({
              "name": "reach", "kind": "approach", "contract": "reach", "env_id": "SO100Reach-v1",
              "scene": {"held": {"type": "cube", "half": 0.014}},
              "reward": [{"term": "DistancePull", "measure": None, "why": "a bare `measure:`."}],
              "success": "grasped",
          }).reward[0].params["measure"] == "tcp_to_object")
    refuses("unknown top-level field", doc_with(reward_scal={"divisor": 12.0}),
            "reward_scal", "reward_scale", path="reward_scal", suggestion="reward_scale")
    refuses("a missing required field", doc_with(success=DROP), "success", path="success")
    refuses("an empty reward list", doc_with(reward=[]), "reward", path="reward")
    refuses("a reward that is not a list", doc_with(reward={"term": "ActionPenalty"}), "reward")
    refuses("a scene entry with no type", doc_with(scene={"goal": {"half": 0.04}}),
            "type", path="scene.goal.type")

    # RewardScale is a document-level field, not a summed row — a generated env that treats it as a
    # row trains at 12x the intended scale (vocab.py, RewardScale.doc).
    refuses("RewardScale as a reward row",
            row_edited(0, term="RewardScale", divisor=12.0), "RewardScale", "reward_scale",
            path="reward[0].term")

    # ── per-skill params (§5): declared, severity-tagged, and actually referenced ───────────────
    refuses("a params.X that resolves to nothing", row_edited(2, setpoint="params.hovr"),
            "hovr", "hover", path="reward[2].setpoint", suggestion="hover")
    # (2026-08-12 review, minor 2) the `params.X` bypass is for NUMBERS whose type only compile()
    # can answer for. It used to be a bare `startswith("params.")`, so it also skipped `choices` and
    # `mode: params.hover` was accepted on PredicateBonus although `mode` is a closed set of three.
    refuses("a params.X does not buy its way past a closed choice set",
            row_edited(0, mode="params.hover"), "params.hover", "replace", path="reward[0].mode")
    check("...while a params.X in a NUMERIC field is still deferred to compile()",
          parse_spec(row_edited(2, setpoint="params.hover")
                     ).reward[2].params["setpoint"] == "params.hover")
    # (2026-08-12 review, minor 3) resolved toward ALLOWING: `reward_scale` takes a `params.X` in
    # its numeric field exactly as a reward row does, rather than being the one place the rule flips.
    scaled = accepts("reward_scale takes a params.X reference, like a row value",
                     doc_with(reward_scale={"divisor": "params.hover"}))
    check("...and stores it verbatim for compile() to bind",
          scaled is not None and scaled.reward_scale["divisor"] == "params.hover")
    refuses("...and it is checked against `params:` like a row value",
            doc_with(reward_scale={"divisor": "params.hovr"}), "hovr", "hover",
            path="reward_scale.divisor", suggestion="hover")
    refuses("...but a bool field takes no reference (numeric-only, as in rows)",
            doc_with(reward_scale={"unnormalized": "params.hover"}), "unnormalized", "bool",
            path="reward_scale.unnormalized")
    nulled_scale = accepts("a null reward_scale field is not a divisor of None",
                           doc_with(reward_scale={"divisor": None}))
    check("...it falls through to the chassis' 12.0",
          nulled_scale is not None and nulled_scale.reward_scale["divisor"] == 12.0)

    # ── minor 4: one null rule for the whole document ────────────────────────────────────────────
    # (2026-08-12 re-review) A null FIELD inside a block falls through, as above and as in a reward
    # row. A null BLOCK is refused — but `reward_scale: null` used to be read as `{}` and fall
    # through silently, while `init: null`, `params: null` and `preflight: null` were all refused.
    # An author cannot learn a rule that holds for three keys out of four.
    refuses("a null `reward_scale:` block is refused, not silently read as {}",
            doc_with(reward_scale=None), "reward_scale", "omit the key", path="reward_scale")
    for field in ("init", "params", "preflight"):
        check(f"...and `{field}: null` is still refused too (the rule is the document's, not one "
              f"key's)", isinstance(error_from(doc_with(**{field: None})), SpecError))
    check("...while an ABSENT reward_scale still inherits, rather than being refused as null",
          parse_spec(doc_with(reward_scale=DROP)).reward_scale["divisor"] == 12.0)

    # ── minor 5: one validation path for authored and chassis-supplied alike ─────────────────────
    # (2026-08-12 re-review) `_parse_reward_scale` ran `_check_type`/`_check_param_refs` on AUTHORED
    # values only, so an inherited or defaulted divisor reached the plan unchecked. Low risk today —
    # every chassis' value is the literal 12.0 — but `_parse_term_row` validates inherited values and
    # this module states that rule for measures in its own comments, so the two paths disagreed on
    # principle. The only way to feed an invalid INHERITED value is to plant one; the test plants one
    # and puts it back.
    approach_scale = CHASSIS["approach"].defaults["RewardScale"]
    saved_scale = dict(approach_scale)
    scale_doc = {
        "name": "reach", "kind": "approach", "contract": "reach", "env_id": "SO100Reach-v1",
        "scene": {"held": {"type": "cube", "half": 0.014}},
        "reward": [{"term": "DistancePull", "why": "the approach chassis supplies the measure."}],
        "success": "grasped",
    }
    check("the planted-value test bites only because the chassis really does supply a divisor",
          approach_scale.get("divisor") == 12.0 and
          parse_spec(dict(scale_doc)).reward_scale["divisor"] == 12.0)
    try:
        approach_scale["divisor"] = "twelve"
        refuses("a CHASSIS-supplied divisor is type-checked like an authored one",
                dict(scale_doc), "divisor", "float", path="reward_scale.divisor")
        approach_scale["divisor"] = "params.nonesuch"
        refuses("...and its params.X references are resolved like an authored one",
                dict(scale_doc), "nonesuch", path="reward_scale.divisor")
    finally:
        approach_scale.clear()
        approach_scale.update(saved_scale)
    check("the chassis default was restored after the planted-value test",
          CHASSIS["approach"].defaults["RewardScale"] == saved_scale and
          parse_spec(dict(scale_doc)).reward_scale["divisor"] == 12.0)
    refuses("a params.X in the success criterion that resolves to nothing",
            doc_with(success="all[grasped, height_above_resting_in(params.lowband)]"),
            "lowband", "low_band", path="success")
    check("the success criterion's declared params.X references are accepted",
          parse_spec(descend_doc()).success.endswith("params.center_tol)]"))
    refuses("a param with no value", doc_with(params={"hover": {"severity": "retrain"}}),
            "value", path="params.hover.value")
    refuses("a param with an unknown severity",
            doc_with(params={"hover": {"value": 0.015, "severity": "retran"}}),
            "retran", "retrain", path="params.hover.severity", suggestion="retrain")
    refuses("a param with no severity", doc_with(params={"hover": {"value": 0.015}}),
            "severity", "retrain", path="params.hover.severity")

    # ── preflight asserts, checked against what bridle.preflight can actually run ───────────────
    refuses("an unknown preflight tier", doc_with(preflight={"statix": {}}), "statix", "static",
            path="preflight.statix", suggestion="static")
    refuses("a preflight assert with no bound",
            doc_with(preflight={"dynamic": {"success_once": {"needs": "warm_start"}}}),
            "bound", path="preflight.dynamic.success_once")
    refuses("an unknown preflight bound",
            doc_with(preflight={"dynamic": {"success_once": {"maks": 0.99}}}), "maks", "max",
            path="preflight.dynamic.success_once.maks")

    # ── tier 2: an expr row's free names must resolve to measures, predicates or params ─────────
    expr_row = {"expr": "2.5 * (1 - tanh(6 * abs(height_above_seat_live - hover))) * grasped",
                "why": "the vocabulary cannot express the hover attractor and the crush penalty as "
                       "one row; this is descend's own formula, written out."}
    e = parse_spec(doc_with(reward=[expr_row]))
    check("an expr row parses", e.reward[0].expr is not None)
    check("an expr row keeps its source verbatim", e.reward[0].expr.source == expr_row["expr"])
    check("an expr row has no term", e.reward[0].term is None)
    check("an expr row's names are exposed for compile() to bind",
          e.reward[0].expr.names == frozenset({"height_above_seat_live", "hover", "grasped"}))
    refuses("expr naming an undeclared measure is refused",
            doc_with(reward=[dict(expr_row, expr="2.5 * hieght_above_seat_live")]),
            "hieght_above_seat_live", path="reward[0].expr")
    refuses("expr naming an undeclared param is refused",
            doc_with(reward=[dict(expr_row, expr="2.5 * (height_above_seat_live - hovr)")]),
            "hovr", path="reward[0].expr")
    check("expr naming a declared param is fine",
          parse_spec(doc_with(reward=[dict(expr_row, expr="center_tol * object_to_goal_xy")]
                              )).reward[0].expr is not None)
    refuses("an expr the micro-language refuses is a SpecError, not an ExprError",
            doc_with(reward=[dict(expr_row, expr="(1).__class__")]), path="reward[0].expr")
    refuses("an expr row still needs a why",
            doc_with(reward=[{"expr": "1.0"}]), "why", "rationale", path="reward[0].why")
    refuses("a bare quantity is ambiguous inside an expr too",
            doc_with(reward=[dict(expr_row, expr="2.5 * height_above_seat")]),
            "frame", "live", path="reward[0].expr")

    # ── Important 3: `params.X` is spelled two ways across the tiers, so the refusal must say so ──
    # (2026-08-12 re-review) Tier 1 teaches `setpoint: params.hover`, and `success:` uses the same
    # prefix; tier 2 wants the BARE name, because an expression is PARSED and `params.hover` parses
    # as attribute access — the construct the grammar refuses outright to keep `x.__class__` out.
    # `expr: "2.5 * params.hover"` used to re-raise expr.py's generic whitelist text verbatim
    # ("'attribute access (e.g. x.attr)' is not allowed in a reward expression; allowed constructs
    # are arithmetic ... and calls to one of the whitelisted functions"), which names no path fix, no
    # legal spelling and no suggestion — landing on an author who learned the prefix one tier up.
    check("the bare spelling is the accepted one (the finding's premise)",
          parse_spec(doc_with(reward=[dict(expr_row, expr="2.5 * hover")])).reward[0].expr is not None)
    prefixed = refuses("a `params.` prefix inside an expr is told which spelling to use",
                       doc_with(reward=[dict(expr_row, expr="2.5 * params.hover")]),
                       "BARE", "params.hover", "'2.5 * hover'", path="reward[0].expr",
                       suggestion="2.5 * hover")
    check("...and states the params this document declares, as the legal set",
          all(p in str(prefixed) for p in ("center_tol", "hover", "low_band")))
    check("...and the corrected spelling it prints is one that actually parses",
          parse_spec(doc_with(reward=[dict(expr_row, expr="2.5 * hover")])
                     ).reward[0].expr.source == "2.5 * hover")
    check("...while a non-params attribute chain still gets the generic whitelist refusal",
          "attribute access" in str(error_from(
              doc_with(reward=[dict(expr_row, expr="2.5 * bin.inner_radius")]))))
    check("...and a params.-free expr error is untouched by the new branch",
          "attribute access" in str(error_from(
              doc_with(reward=[dict(expr_row, expr="(1).__class__")]))))

    # ── tier 3: allowed, but marked ─────────────────────────────────────────────────────────────
    custom = {"custom": "primitives.descend_to_target.descend_env:crush_term",
              "why": "the seat-crush term reads a per-env buffer the vocabulary has no name for."}
    c = parse_spec(doc_with(reward=descend_doc()["reward"] + [custom]))
    check("custom row parses", len(c.reward) == 10)
    check("custom row is flagged opaque", c.reward[-1].custom is not None)
    check("custom row keeps module:function verbatim", c.reward[-1].custom == custom["custom"])
    check("custom row has no term and no expr",
          c.reward[-1].term is None and c.reward[-1].expr is None)
    check("tier-1 rows are NOT flagged opaque", all(r.custom is None for r in spec.reward))
    refuses("a custom that is not module:function",
            doc_with(reward=[dict(custom, custom="crush_term")]), "crush_term", "module",
            path="reward[0].custom")
    refuses("a row that is all three tiers at once",
            doc_with(reward=[{"term": "ActionPenalty", "expr": "1.0", "why": "both."}]),
            "term", "expr", path="reward[0]")
    refuses("a row that is no tier at all", doc_with(reward=[{"weight": 1.0, "why": "nothing."}]),
            "term", "expr", "custom", path="reward[0]")

    # ── json_schema(): the machine-readable half of the prompt payload ──────────────────────────
    js = json_schema()
    check("json_schema returns a mapping", isinstance(js, dict))
    check("json_schema declares draft 2020-12",
          js.get("$schema") == "https://json-schema.org/draft/2020-12/schema")
    check("json_schema requires the load-bearing fields",
          {"name", "kind", "contract", "env_id", "scene", "reward", "success"}
          <= set(js["required"]))
    check("json_schema refuses unknown top-level fields", js.get("additionalProperties") is False)
    check("json_schema's kind enum is the chassis list",
          js["properties"]["kind"]["enum"] == sorted(CHASSIS))
    check("json_schema has one variant per tier",
          len(js["properties"]["reward"]["items"]["oneOf"]) == 3)
    row_variants = [js["$defs"][n] for n in ("term_row", "expr_row", "custom_row")]
    check("json_schema makes why mandatory in every tier",
          all("why" in v["required"] for v in row_variants))
    term_enum = js["$defs"]["term_row"]["properties"]["term"]["enum"]
    check("json_schema's term enum omits RewardScale (a document field, not a row)",
          "RewardScale" not in term_enum and len(term_enum) == len(TERMS) - 1)
    measure_enum = js["$defs"]["term_row"]["properties"]["measure"]["enum"]
    check("json_schema's measure enum offers the frame-qualified name",
          "height_above_seat_live" in measure_enum)
    check("json_schema's measure enum withholds the ambiguous bare quantity",
          "height_above_seat" not in measure_enum)
    check("json_schema's measure enum still covers the unambiguous measures",
          {"tcp_to_object", "object_to_goal_xy"} <= set(measure_enum) and
          len(measure_enum) == len(MEASURES))
    # the closed sets reach the machine-readable half too, not only parse_spec (minor 3)
    for pname, expected in (("kernel", {"one_minus_tanh", "neg_linear", "gaussian"}),
                            ("side", {"above", "below"}),
                            ("predicate_ref", {"per_step", "latched"}),
                            ("norm", {"l2"}),
                            ("mode", {"add", "replace", "floor"})):
        check(f"json_schema publishes {pname}'s legal set as an enum",
              set(js["$defs"]["term_row"]["properties"][pname].get("enum", ())) == expected)
    # (Important 3) the schema description is the other half of the two-spellings fix: an author
    # reading only the schema must not have to discover the rule from a refusal.
    expr_desc = js["$defs"]["expr_row"]["properties"]["expr"]["description"]
    check("the expr_row description disambiguates the two spellings of a param reference",
          "BARE" in expr_desc and "params.hover" in expr_desc and "success:" in expr_desc)
    # it is handed to a model and written to disk, so it has to survive a round trip
    check("json_schema is JSON-serializable", isinstance(json.dumps(js), str))

    # ── the optional half of the document really is optional ────────────────────────────────────
    minimal = {
        "name": "reach", "kind": "approach", "contract": "reach", "env_id": "SO100Reach-v1",
        "scene": {"held": {"type": "cube", "half": 0.014}},
        "reward": [{"term": "DistancePull",
                    "why": "reach's entire dense signal is the tcp-to-object gap."}],
        "success": "within_radius(anchor=object, radius_expr=0.02)",
    }
    m = parse_spec(minimal)
    check("init/params/preflight are optional",
          (dict(m.init), dict(m.params), dict(m.preflight)) == ({}, {}, {}))
    check("a single-candidate chassis supplies even a REQUIRED parameter",
          (m.reward[0].params["measure"], m.reward[0].params["kernel"])
          == ("tcp_to_object", "neg_linear"))


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
