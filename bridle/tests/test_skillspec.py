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
  FRAME      descend's reward grades `height_above_seat` against the live seat while descend_stack's
             success grades a frozen goal. One quantity, two frames.
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
"""
import json
import sys

from bridle.skill.spec import Row, SkillSpec, SpecError, json_schema, parse_spec
from bridle.skill.vocab import CHASSIS, MEASURES, TERMS

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

    # the frame-qualified form resolves back to the MEASURES key compile() will look up
    check("a frame-qualified measure resolves to its canonical key",
          spec.reward[2].params["measure"] == "height_above_seat")
    check("the frozen-goal measure is nameable directly",
          parse_spec(row_edited(2, measure="height_above_seat_static_goal")
                     ).reward[2].params["measure"] == "height_above_seat_static_goal")

    # ── the rest of the vocabulary is checked too, not just terms and measures ──────────────────
    refuses("unknown chassis", doc_with(kind="cary"), "cary", "carry", path="kind",
            suggestion="carry")
    refuses("unknown predicate", row_edited(0, predicate="grapsed"), "grapsed", "grasped",
            path="reward[0].predicate", suggestion="grasped")
    refuses("unknown predicate inside a gate", row_edited(1, gate="and_(grasped, abov_z(z=0.06))"),
            "abov_z", "above_z", path="reward[1].gate")
    check("a compound gate over real predicates is accepted",
          parse_spec(row_edited(1, gate="and_(grasped, above_z(z=0.06))")
                     ).reward[1].params["gate"].startswith("and_("))
    refuses("unknown row parameter", row_edited(0, wieght=1.0), "wieght", "weight",
            path="reward[0].wieght", suggestion="weight")
    refuses("a parameter of the wrong type", row_edited(0, weight="heavy"),
            "weight", "float", path="reward[0].weight")
    refuses("a value outside a declared choice set", row_edited(0, mode="clobber"),
            "clobber", "replace", path="reward[0].mode")
    refuses("a required parameter with no default anywhere",
            doc_with(reward=[{"term": "Ramp", "measure": "object_z", "why": "ramp the lift."}]),
            "cap", "required", path="reward[0].cap")
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
