"""Unit test for the Foundry. No sim, no GPU — stages are faked.

WHY THIS EXISTS: the Foundry is where a mismatched skill becomes a matching one. Its two obligations
are both lessons from real failures:

  1. the target contract reaches training as DATA, never as a number someone re-typed
     (a training literal and a deploy literal disagreeing = the 0/20 stack failure)
  2. an unrunnable build fails at PLAN time, not at hour three of a GPU job

Run: python -m pytest bridle/tests/test_foundry.py
"""
import dataclasses
import sys
import tempfile

from bridle.app import App, Artifact, EnvSpec, Recipe, Stage
from bridle.contract import Contract
from bridle.foundry import Foundry, ShellStageRunner, StageError, StageResult, contract_env
from bridle.resolve import RETRAIN, RUN
from bridle.rig import Rig
from bridle.store import Store

FAILS = []


def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        FAILS.append(name)


def _app(contract):
    return App(name="descend_to_target", title="Descend", description="lower a held object",
               when_to_use="after grasp",
               recipe=Recipe(env=EnvSpec(id="SO100DescendToTarget-v1"),
                             stages=(Stage("teacher", {"script": "teacher_train.sh"}),
                                     Stage("distill", {"script": "distill.sh"}),
                                     Stage("student", {"script": "train.sh"}))),
               artifacts=(Artifact(path="runs/old/ckpt.pt", contract=contract),))


def run_checks():
    rig = Rig.so101(cameras=("base",))
    trained = dataclasses.replace(Contract.stack(), rig=rig)
    app = _app(trained)
    store = Store(tempfile.mkdtemp())

    # ── the contract reaches training as DATA ─────────────────────────────────────────────────
    cube_top = dataclasses.replace(
        trained, release=dataclasses.replace(trained.release, height_above_resting=0.002))
    env = contract_env(cube_top)
    check("contract_env carries the release height", env["PRIM_DESCEND_HOVER"] == repr(0.002))
    check("contract_env carries the fingerprint",
          env["BRIDLE_CONTRACT_FINGERPRINT"] == cube_top.fingerprint())
    check("the two contracts have DIFFERENT fingerprints in the env",
          contract_env(trained)["BRIDLE_CONTRACT_FINGERPRINT"] != env["BRIDLE_CONTRACT_FINGERPRINT"])

    # ── an unrunnable build fails at PLAN time ────────────────────────────────────────────────
    plan = store.plan(app, rig, target_contract=cube_top)
    check("the cube-top contract plans RETRAIN", plan.action == RETRAIN)
    empty = Foundry()
    try:
        empty.build(app, rig, plan, target_contract=cube_top)
        check("a Foundry with no runners refuses BEFORE launching", False)
    except StageError as e:
        check("a Foundry with no runners refuses BEFORE launching", "no runner registered" in str(e))

    # ── a dry run shows exactly what WOULD happen, for free ───────────────────────────────────
    f = Foundry({k: ShellStageRunner(k, dry_run=True) for k in ("teacher", "distill", "student")})
    job = f.build(app, rig, plan, target_contract=cube_top)
    check("dry-run job succeeds", job.ok)
    check("dry run executes every recipe stage", [r.kind for r in job.results] ==
          ["teacher", "distill", "student"])
    check("the dry run shows the contract in the command environment",
          all("PRIM_DESCEND_HOVER=0.002" in r.detail for r in job.results))
    check("the job records the contract it built against",
          job.contract.fingerprint() == cube_top.fingerprint())

    # ── RUN needs no build at all ─────────────────────────────────────────────────────────────
    job = f.build(app, rig, store.plan(app, rig))
    check("a matching contract builds nothing", job.action == RUN and len(job.results) == 1)

    # ── a failing stage stops the pipeline; later stages consume earlier outputs ───────────────
    def boom(stage, ctx):
        return StageResult(stage.kind, False, "GPU caught fire")
    f2 = Foundry({"teacher": boom,
                  "distill": ShellStageRunner("distill", dry_run=True),
                  "student": ShellStageRunner("student", dry_run=True)})
    job = f2.build(app, rig, plan, target_contract=cube_top)
    check("a failed stage stops the run", not job.ok and len(job.results) == 1)

    # ── a raising stage must not take the process down ────────────────────────────────────────
    def raiser(stage, ctx):
        raise RuntimeError("disk full")
    f3 = Foundry({"teacher": raiser})
    job = f3.build(app, rig, dataclasses.replace(plan, stages=("teacher",)), target_contract=cube_top)
    check("a raising stage is captured as a failure", not job.ok and "disk full" in job.results[0].detail)

    # ── BLOCKED is not a build ────────────────────────────────────────────────────────────────
    blind = dataclasses.replace(rig, cameras=(), sensors=("proprio",))
    vis = dataclasses.replace(app, requires={"sensors": ["rgb"]})
    try:
        f.build(vis, blind, store.plan(vis, blind))
        check("BLOCKED refuses to build", False)
    except StageError as e:
        check("BLOCKED refuses to build", "not a training problem" in str(e))

    # ── what comes out is stamped ─────────────────────────────────────────────────────────────
    sd = Foundry.stamp_result({"w": [1]}, cube_top)
    from bridle.checkpoint import ContractMismatch, verify
    verify(sd, cube_top)
    check("the produced checkpoint verifies under its build contract", True)
    try:
        verify(sd, trained)
        check("...and REFUSES under the old one", False)
    except ContractMismatch:
        check("...and REFUSES under the old one", True)


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
