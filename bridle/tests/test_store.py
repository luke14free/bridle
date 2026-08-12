"""Unit test for App / Recipe / Store.plan(). No sim, no GPU.

WHY THIS EXISTS: `plan(app, rig)` is the user-facing verb of the product — "here is what it will take
to run this skill on your robot". The cases below are the four answers it must be able to give, and
the one it must never give: silently running weights trained for a different problem.

Run: python -m pytest bridle/tests/test_store.py
"""
import dataclasses
import os
import sys
import tempfile

from bridle.app import App, Artifact, EnvSpec, EvalSpec, Recipe, Stage
from bridle.contract import Contract
from bridle.resolve import ADAPT, RETRAIN, RUN
from bridle.rig import Rig
from bridle.store import BLOCKED, Store, app_from_dict, app_to_dict

FAILS = []


def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        FAILS.append(name)


def _app(contract, requires=None):
    return App(
        name="descend_to_target",
        title="Descend to target",
        description="Lower a held object onto a destination without releasing it.",
        when_to_use="After the object is grasped and carried above the destination.",
        args={"destination": "xyz of the surface to descend onto"},
        requires=requires or {},
        recipe=Recipe(
            env=EnvSpec(id="SO100DescendToTarget-v1", max_episode_steps=400),
            reward="primitives.descend_to_target.reward",
            stages=(Stage("teacher", {"algo": "ppo", "steps": 20_000_000}),
                    Stage("round_robin", {"peers": ["grab", "move_to_target"]}),
                    Stage("distill", {"from": "teacher"}),
                    Stage("student", {"obs": ["rgb", "proprio"]}))),
        artifacts=(Artifact(path="runs/descend-coordv3g/ckpt.pt", contract=contract,
                            eval=EvalSpec(protocol="descend_eval", n=64,
                                          reported={"rig": "so101-default", "success": 0.69})),))


def run_checks():
    rig = Rig.so101(cameras=("base",))
    trained = dataclasses.replace(Contract.stack(), rig=rig)
    app = _app(trained)
    app.validate()
    store = Store(tempfile.mkdtemp())

    # ── RUN: the contract matches ─────────────────────────────────────────────────────────────
    p = store.plan(app, rig)
    check("matching contract -> RUN", p.action == RUN)
    check("RUN names the checkpoint to load", p.checkpoint == "runs/descend-coordv3g/ckpt.pt")

    # ── RETRAIN: THE STACK CASE, as a plan ────────────────────────────────────────────────────
    # The skill was trained to release 1.5cm above resting. We now want 2mm. Under the old world
    # this ran anyway and scored 0/20 for two days.
    cube_top = dataclasses.replace(
        trained, release=dataclasses.replace(trained.release, height_above_resting=0.002))
    p = store.plan(app, rig, target_contract=cube_top)
    check("a different release height -> RETRAIN", p.action == RETRAIN)
    check("RETRAIN lists the full recipe pipeline",
          p.stages == ("teacher", "round_robin", "distill", "student"))
    check("...and the diff names the field",
          any(f == "release.height_above_resting" for f, *_ in p.resolution.reasons))

    # ── ADAPT: the rig moved, the task didn't ─────────────────────────────────────────────────
    moved = dataclasses.replace(rig, cameras=(dataclasses.replace(rig.cameras[0], pos=(-0.4, -0.2, 0.5)),))
    p = store.plan(app, moved)
    check("a moved camera -> ADAPT", p.action == ADAPT)
    check("ADAPT re-runs only the perception stages", p.stages == ("distill", "student"))
    check("ADAPT still offers the checkpoint to start from", p.checkpoint is not None)

    # ── BLOCKED: the rig physically cannot ────────────────────────────────────────────────────
    # Distinct from RETRAIN on purpose: "you need GPU hours" and "your robot has no camera" are
    # different sentences, and collapsing them sends someone to train vision on a blind rig.
    vision_app = _app(trained, requires={"sensors": ["rgb"], "cameras": ["wrist"]})
    blind = dataclasses.replace(rig, cameras=(), sensors=("proprio",))
    p = store.plan(vision_app, blind)
    check("a rig missing the required sensor -> BLOCKED", p.action == BLOCKED)
    check("BLOCKED says what is missing", any("rgb" in b for b in p.blockers))
    check("BLOCKED is not RETRAIN", p.action != RETRAIN)

    # ── an app with no artifacts must be BUILT, never guessed at ──────────────────────────────
    fresh = dataclasses.replace(app, artifacts=())
    p = store.plan(fresh, rig)
    check("an app with no weights -> RETRAIN", p.action == RETRAIN)

    # ── validation: the two ways an app is meaningless ────────────────────────────────────────
    try:
        App(name="x", title="", description="", when_to_use="").validate()
        check("an app with neither recipe nor artifacts is rejected", False)
    except ValueError:
        check("an app with neither recipe nor artifacts is rejected", True)
    try:
        dataclasses.replace(app, artifacts=(Artifact(path="x.pt", contract=None),)).validate()
        check("an UNSTAMPED artifact is rejected", False)
    except ValueError:
        check("an UNSTAMPED artifact is rejected", True)

    # ── round-trip through disk ───────────────────────────────────────────────────────────────
    path = store.save(app)
    check("save writes a file", os.path.exists(path))
    back = store.get("descend_to_target")
    check("round-trip preserves the app name", back.name == app.name)
    check("round-trip preserves the CONTRACT FINGERPRINT",
          back.artifacts[0].contract.fingerprint() == trained.fingerprint())
    check("round-trip preserves the recipe fingerprint",
          back.recipe.fingerprint() == app.recipe.fingerprint())
    check("a round-tripped app still plans RUN on the same rig",
          store.plan(back, rig).action == RUN)

    # ── unknown keys fail loudly ──────────────────────────────────────────────────────────────
    # A store that silently drops a field is a store that silently ships the wrong skill.
    d = app_to_dict(app); d["totally_unknown"] = 1
    try:
        app_from_dict(d); check("an unknown key in a stored app is rejected", False)
    except ValueError:
        check("an unknown key in a stored app is rejected", True)


def test_bridle():
    FAILS.clear()
    run_checks()
    assert not FAILS, f"{len(FAILS)} check(s) failed: {FAILS}"


def main():
    run_checks()
    print(f"\n{len(FAILS)} failure(s)")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
