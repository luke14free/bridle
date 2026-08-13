"""Unit test for bridle.skill.author — the typed Python front-end of the skill document.

WHAT IT HAS TO PROVE. `author.py` exists so a skill can be written as Python instead of YAML. The
whole architectural claim is that this is **one pipeline and two spellings**: the builders produce
the same `dict` a YAML loader would, hand it to the SAME `parse_spec -> compile_spec`, and get the
SAME plan fingerprint. If that is not true, the Python front-end is a second vocabulary with a
second set of refusals, which is exactly the thing the schema tier was built to prevent.

So the checks below bite in both directions:

  SAME DIGEST   the Python-authored `descend_to_target` compiles to `plan@95babe2a3cc5`, the digest
                the deployed `primitives/descend_to_target/skill.yaml` compiles to — including the
                exact `success:` TEXT, which hashes (`_lower_term_row` carries it as the
                SuccessBonus op's `condition`). It is asserted with terse `why` prose to make the
                complementary point: `why` is deliberately OUT of the digest.
  ROUND TRIP    `Skill.doc -> to_yaml() -> yaml.safe_load -> parse_spec` gives that same digest, so
                the two spellings are interconvertible and not merely similar.
  COVERAGE      every `vocab.TERMS` entry and every `vocab.PREDICATES` entry has a builder, and
                there are no builders for anything else — generated from the vocabulary, asserted at
                import time in `author.py` and re-asserted here, so a vocabulary addition cannot
                leave the front-end behind.
  REFUSALS      a builder mistake raises `SpecError` with the dotted path, the legal set and the
                nearest match — the same quality of message the YAML path gives, because it IS the
                YAML path. A bare `TypeError`/`AttributeError` out of the front-end would be a
                refusal a 27-30B author cannot act on.
  WHY           mandatory. The Python surface must not be the loophole around the one field that is
                the only surviving record of why a weight is what it is.

Run: PYTHONPATH=. python bridle/tests/test_author.py     (the project venv has no pytest)
"""
import sys
import warnings

from bridle.skill.author import (
    LauncherError, Skill, Training, predicates as P, terms as T,
)
from bridle.skill.compile import compile_spec
from bridle.skill.spec import ROW_TERMS, SpecError, parse_spec
from bridle.skill.vocab import PREDICATES, TERMS

FAILS = []

#: The digest the deployed `primitives/descend_to_target/skill.yaml` compiles to (2026-08-13). A
#: real checkpoint is stamped with it — `logs/descend_to_target-skill-seed20-plan95babe2a3cc5.log` —
#: so this is not a golden value that may be re-baselined when convenient.
DESCEND_PLAN = "95babe2a3cc5"


def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        FAILS.append(name)


def error_from(fn):
    try:
        fn()
    except SpecError as e:
        return e
    except Exception as e:                                        # noqa: BLE001 — the point of the test
        return e
    return None


def refuses(label, fn, *fragments, path=None, suggestion=None):
    """Assert `fn()` is refused with a message an author can act on: `SpecError`, every fragment
    present verbatim, and the dotted path where it applies."""
    exc = error_from(fn)
    check(f"{label}: refused with SpecError (not {type(exc).__name__ if exc else 'nothing'})",
          isinstance(exc, SpecError))
    msg = str(exc) if exc is not None else "<nothing raised>"
    for frag in fragments:
        check(f"{label}: message says {frag!r}", frag in msg)
    if path is not None:
        check(f"{label}: path is {path!r}", getattr(exc, "path", None) == path)
    if suggestion is not None:
        check(f"{label}: suggests {suggestion!r}", getattr(exc, "suggestion", None) == suggestion)
    return exc


class FakeEnv:
    """Stands in for a ManiSkill env class. `bridle.skill` is stdlib-only and must never import a
    simulator, so it treats a class as an OPAQUE reference — which means any class will do here, and
    that is the property being tested."""


def descend_skill(**overrides):
    """`descend_to_target`, authored in Python. The weights, measures, gates, fold order and success
    criterion are the deployed document's; the `why` prose is deliberately terse (see the module
    docstring)."""
    kw = dict(
        name="descend_to_target",
        env="SO100DescendToTarget-v1",
        kind="carry",
        contract="stack",
        init={"snapshot": "descend_init"},
        params={"hover": 0.015, "low_band": 0.03, "center_tol": 0.045},
        reward_scale={"divisor": 12.0},
        reward=[
            T.PredicateBonus(1.0, P.grasped, why="hold on."),
            T.DistancePull(1.5, "object_to_goal_xy", k=4.0, gate=P.grasped, why="re-centre."),
            T.DistancePull(2.5, "height_above_seat_live", k=6.0, setpoint="params.hover",
                           gate=P.grasped, why="hover, not contact."),
            T.HingePenalty(3.0, "height_above_seat_live", threshold=0.0, side="below",
                           gate=P.grasped, why="crush."),
            T.HingePenalty(1.0, "gripper_qpos", threshold=-0.6, side="above", gate=P.grasped,
                           why="jaw creep."),
            T.VelocityPenalty("held", linear_weight=0.3, angular_weight=0.05, why="anti-fling."),
            T.PredicateBonus(-0.5, P.not_grasped, why="do not let go."),
            T.SuccessBonus(12.0, mode="replace", scope="preceding", predicate_ref="per_step",
                           why="jackpot."),
            T.ActionPenalty(0.001, norm="l2", why="jerk."),
        ],
        success=P.all(P.grasped,
                      P.below_resting_height("params.low_band"),
                      P.within_radius("target_pos", "params.center_tol")),
    )
    kw.update(overrides)
    return Skill(**kw)


def run_checks():
    # `compile_spec` re-emits its actionable notes through the stdlib `warnings` module, and
    # descend's integrated-shaping note is one of them. It is a FACT about the deployed reward, not
    # a defect in this test — `plan.warnings` is the authoritative channel and `test_skillcompile`
    # asserts on it there.
    warnings.simplefilter("ignore")

    # ── coverage: the builders are GENERATED from the vocabulary, in both directions ─────────────
    check("every vocabulary term has a builder",
          all(hasattr(T, name) for name in TERMS))
    check("the term namespace holds nothing the vocabulary does not",
          set(T.names()) == set(TERMS))
    check("every vocabulary predicate has a builder",
          all(hasattr(P, name) for name in PREDICATES))
    check("the predicate namespace is the vocabulary plus the two bracket sugars",
          set(P.names()) == set(PREDICATES) | {"all", "any"})
    check("a zero-parameter predicate is an attribute, not a call",
          str(P.grasped) == "grasped" and str(P.not_grasped) == "not_grasped")

    # ── the descend document, authored in Python ────────────────────────────────────────────────
    skill = descend_skill()
    spec = skill.spec()
    plan = compile_spec(spec, horizon=64, terminate_on_success=True)
    check(f"the Python-authored descend compiles to plan@{DESCEND_PLAN}",
          plan.fingerprint() == DESCEND_PLAN)
    check("rows keep their authored order",
          [r.term for r in spec.reward]
          == ["PredicateBonus", "DistancePull", "DistancePull", "HingePenalty", "HingePenalty",
              "VelocityPenalty", "PredicateBonus", "SuccessBonus", "ActionPenalty"])
    check("positional arguments bind to the term's parameters in vocabulary order",
          (spec.reward[1].params["weight"], spec.reward[1].params["measure"]) == (1.5,
                                                                                 "object_to_goal_xy"))
    check("a `params.X` reference survives to compile and binds there",
          spec.reward[2].params["setpoint"] == "params.hover"
          and plan.ops[2].params["setpoint"] == 0.015)
    check("a bare number param is expanded to {value, severity} with the conservative severity",
          dict(spec.params["hover"]) == {"value": 0.015, "severity": "retrain"})
    check("`why` prose is out of the digest (terse prose, same fingerprint)",
          spec.reward[0].why == "hold on." and plan.fingerprint() == DESCEND_PLAN)

    # ── the success criterion is rendered, and its TEXT is what hashes ───────────────────────────
    check("a multi-operand `all[...]` renders one operand per line, aligned under the bracket",
          spec.success == ("all[grasped,\n"
                           "    below_resting_height(band=params.low_band),\n"
                           "    within_radius(anchor=target_pos, radius_expr=params.center_tol)]"))
    moved = descend_skill(success=P.all(P.grasped,
                                        P.height_above_resting_in("params.low_band"),
                                        P.within_radius("target_pos", "params.center_tol")))
    check("a different success criterion moves the digest (the check bites)",
          compile_spec(moved.spec(), horizon=64).fingerprint() != DESCEND_PLAN)
    check("a nested predicate renders inline, positionally for the boolean combinators",
          str(P.not_(P.grasped)) == "not_(term=grasped)"
          and str(P.and_(P.grasped, P.above_z(0.06))) == "and_(grasped,\n     above_z(z=0.06))")

    # ── round trip: Python -> YAML -> the same pipeline -> the same digest ───────────────────────
    text = skill.to_yaml()
    check("to_yaml() emits the document's own keys", "env_id: " in text and "success:" in text)
    try:
        import yaml
    except ImportError:
        print("  SKIP  the YAML leg was NOT verified — PyYAML is not importable in this "
              "interpreter, so 'round-trips' is unproven, not proven")
        yaml = None
    if yaml is not None:
        reloaded = yaml.safe_load(text)
        check("to_yaml() re-parses to the identical document", reloaded == skill.doc)
        rt = compile_spec(parse_spec(reloaded), horizon=64, terminate_on_success=True)
        check(f"the YAML spelling of the same skill compiles to plan@{DESCEND_PLAN}",
              rt.fingerprint() == DESCEND_PLAN)

    # ── `scene:` is optional, and still parses when present ─────────────────────────────────────
    check("a document with no `scene:` parses", dict(skill.spec().scene) == {})
    scened = descend_skill(scene={"goal": {"type": "platform", "half": 0.04, "top_z": 0.03}})
    check("a `scene:` block still parses when supplied",
          scened.spec().scene["goal"]["type"] == "platform")
    check("`scene:` is descriptive — supplying one does not move the reward digest",
          compile_spec(scened.spec(), horizon=64, terminate_on_success=True).fingerprint()
          == DESCEND_PLAN)
    refuses("a scene object with no type",
            lambda: descend_skill(scene={"goal": {"half": 0.04}}).spec(),
            "declares a `type`", path="scene.goal.type")

    # ── env by CLASS: an opaque reference the core never resolves ────────────────────────────────
    by_class = descend_skill(env=FakeEnv)
    check("an env class serialises as module:qualname",
          by_class.doc["env_id"] == f"{FakeEnv.__module__}:FakeEnv")
    check("the class itself is kept for the adapter that resolves it", by_class.env is FakeEnv)
    check("`env` is out of the reward digest, so a class and an id give the same plan",
          compile_spec(by_class.spec(), horizon=64,
                       terminate_on_success=True).fingerprint() == DESCEND_PLAN)
    refuses("an env that is neither an id nor a class",
            lambda: descend_skill(env=42), "registered env id", "env class", path="env")
    refuses("an empty env id", lambda: descend_skill(env="  "), "registered env id", path="env")

    # ── `why` is mandatory, and the Python surface is not the loophole ───────────────────────────
    refuses("a reward row with no why", lambda: T.ActionPenalty(0.001),
            "every reward row needs a non-empty `why`", path="ActionPenalty.why")
    refuses("a reward row with a blank why", lambda: T.ActionPenalty(0.001, why="   "),
            "every reward row needs a non-empty `why`", path="ActionPenalty.why")

    # ── builder mistakes get the schema tier's own refusals ─────────────────────────────────────
    refuses("an unknown term name", lambda: T.DistancePul,
            "unknown reward term", "DistancePull", path="terms.DistancePul",
            suggestion="DistancePull")
    refuses("an unknown predicate name", lambda: P.graspd,
            "unknown predicate", "grasped", path="predicates.graspd", suggestion="grasped")
    refuses("an unknown term parameter",
            lambda: descend_skill(reward=[T.ActionPenalty(0.001, nrom="l2", why="x.")]).spec(),
            "unknown parameter 'nrom'", path="reward[0].nrom", suggestion="norm")
    refuses("an unknown measure",
            lambda: descend_skill(
                reward=[T.DistancePull(1.5, "hieght_above_seat", why="x.")]).spec(),
            "unknown measure", path="reward[0].measure", suggestion="height_above_seat_live")
    refuses("a frame-ambiguous measure",
            lambda: descend_skill(
                reward=[T.DistancePull(1.5, "height_above_seat", why="x.")]).spec(),
            "readable in 2 frames", path="reward[0].measure")
    refuses("a value outside a closed choice set",
            lambda: descend_skill(
                reward=[T.SuccessBonus(12.0, mode="replce", why="x.")]).spec(),
            "not a legal value for 'mode'", "add, floor, replace",
            path="reward[0].mode", suggestion="replace")
    refuses("more positional arguments than the term has parameters",
            lambda: T.ActionPenalty(0.001, "l2", "action_norm", "extra", why="x."),
            "takes 3 positional", "ActionPenalty", path="ActionPenalty")
    refuses("the same parameter given positionally and by keyword",
            lambda: T.ActionPenalty(0.001, weight=0.002, why="x."),
            "given twice", "weight", path="ActionPenalty.weight")
    refuses("RewardScale used as a reward row",
            lambda: descend_skill(reward=[T.RewardScale(divisor=12.0)]).spec(),
            "document-level field, not a reward row", path="reward[0].term")
    check("RewardScale IS the builder for the document-level field",
          descend_skill(reward_scale=T.RewardScale(divisor=12.0)).doc["reward_scale"]
          == {"divisor": 12.0})
    refuses("an unknown predicate inside success",
            lambda: descend_skill(success="all[grasped, centered_on_goal(0.045)]").spec(),
            "unknown predicate 'centered_on_goal'", path="success")
    refuses("a params.X reference to an undeclared param",
            lambda: descend_skill(
                reward=[T.DistancePull(1.5, "object_to_goal_xy", setpoint="params.hovr",
                                       why="x.")]).spec(),
            "is not declared in this document's `params:` block", suggestion="hover")

    # ── Training: the defaults ARE `primitives/descend_to_target/teacher_train.sh` ───────────────
    t = Training()
    check("Training defaults are the deployed lineage's hyperparameters",
          (t.seed, t.num_envs, t.num_steps, t.update_epochs, t.num_minibatches, t.gamma,
           t.ent_coef, t.total_steps, t.partial_reset, t.early_stop_success, t.early_stop_patience)
          == (20, 4096, 16, 8, 32, 0.9, 0.001, 150_000_000, True, 0.90, 25))
    check("Training is carried on the Skill and is NOT part of the document",
          descend_skill(training=Training(seed=7)).training.seed == 7
          and "training" not in skill.doc)

    # ── the horizon is stated or resolved, and NEVER invented ───────────────────────────────────
    check("a stated max_episode_steps is what horizon() returns",
          descend_skill(max_episode_steps=64).horizon() == 64)
    check("an env nothing can resolve reports NO horizon rather than a default",
          descend_skill(env=FakeEnv).horizon() is None)
    check("the defining file is captured, so runs land next to the skill",
          skill.source == __file__)

    # ── the launcher seam ────────────────────────────────────────────────────────────────────────
    seen = {}

    def fake_launcher(*, skill, spec, plan, mode, options):
        seen.update(skill=skill, spec=spec, plan=plan, mode=mode, options=options)
        return 0

    rc = skill.train(launcher=fake_launcher, horizon=64)
    check("train() calls the launcher with the compiled plan and mode='train'",
          rc == 0 and seen["mode"] == "train" and seen["plan"].fingerprint() == DESCEND_PLAN)
    seen.clear()
    check("verify() calls the same launcher with mode='verify'",
          skill.verify(launcher=fake_launcher, horizon=64) == 0 and seen["mode"] == "verify")

    called = []

    def never(**kw):
        called.append(kw)
        return 0

    bad = descend_skill(reward=[T.DistancePull(2.5, "height_above_seat_live", k=6.0, setpoint=0.0,
                                               gate=P.grasped, why="peaks at contact.")])
    exc = error_from(lambda: bad.train(launcher=never, horizon=64))
    check("a compile refusal happens BEFORE the launcher is reached",
          exc is not None and not called and "contact surface" in str(exc))

    exc = error_from(lambda: skill.train(launcher=None, horizon=64))
    check("with no launcher registered, train() says so instead of guessing",
          isinstance(exc, LauncherError) and "launcher" in str(exc).lower())


def test_bridle():
    """pytest entry point — the same checks, reported as one assertion."""
    FAILS.clear()
    run_checks()
    assert not FAILS, f"{len(FAILS)} check(s) failed: {FAILS}"


def main():
    run_checks()
    print(f"\n{len(FAILS)} failure(s)")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
