"""Unit test for Rig + resolve(). No sim, no GPU.

WHY THIS EXISTS: this is the product. "Download a skill and run it on your robot" is a coin flip
unless something can say, mechanically, whether the skill's assumptions hold on YOUR setup — and if
they don't, whether the policy is recoverable or has to be regenerated.

The cases below are not hypothetical. The stack case is the 2026-08-11 failure verbatim: a policy
trained to release above a flat platform, executed above a 2.4cm cube, which ran happily and scored
0/20 for two days. Under `resolve` that is a RETRAIN verdict naming the field, before a single step
is taken.

Run: python -m pytest bridle/tests/test_resolve.py
"""
import dataclasses
import sys

from bridle.contract import Contract
from bridle.resolve import ADAPT, RETRAIN, RUN, resolve_contracts, severity_of
from bridle.rig import Camera, Gripper, Rig

FAILS = []


def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        FAILS.append(name)


def run_checks():
    # ── Rig basics ────────────────────────────────────────────────────────────────────────────
    rig = Rig.so101(cameras=("base", "wrist"))
    rig.validate()
    check("reference rig validates", True)
    check("rig fingerprint is stable", rig.fingerprint() == Rig.so101(cameras=("base", "wrist")).fingerprint())
    check("a different camera set is a different rig",
          Rig.so101(cameras=("base",)).fingerprint() != rig.fingerprint())

    for name, build in [
        ("unknown gripper kind rejected", lambda: dataclasses.replace(
            rig, gripper=dataclasses.replace(rig.gripper, kind="suction"))),
        ("unknown control mode rejected", lambda: dataclasses.replace(rig, control_mode="telepathy")),
        ("zero dof rejected", lambda: dataclasses.replace(rig, dof=0)),
        ("duplicate camera names rejected", lambda: dataclasses.replace(
            rig, cameras=(Camera("base", 128, 128), Camera("base", 128, 128)))),
        ("rgb sensor with no cameras rejected", lambda: dataclasses.replace(
            rig, cameras=(), sensors=("proprio", "rgb"))),
    ]:
        try:
            build().validate(); check(name, False)
        except ValueError:
            check(name, True)

    # ── identical contracts -> RUN, without walking fields ────────────────────────────────────
    g = Contract.grab()
    r = resolve_contracts(g, Contract.grab())
    check("identical contracts resolve to RUN", r.verdict == RUN and not r.reasons)

    # ── THE 2026-08-11 STACK FAILURE, as a resolve() verdict ──────────────────────────────────
    # A policy trained to release 1.5cm above resting, executed against a contract that wants 2mm.
    # This ran for two days and scored 0/20 because nothing compared the two.
    s = Contract.stack()
    fixed = dataclasses.replace(s, release=dataclasses.replace(s.release, height_above_resting=0.002))
    r = resolve_contracts(s, fixed)
    check("moving the release height forces RETRAIN", r.verdict == RETRAIN)
    check("...and names the field that forced it",
          r.reasons and r.reasons[0][0] == "release.height_above_resting")
    check("...with both values, so the diff is actionable",
          r.reasons[0][2] == 0.015 and r.reasons[0][3] == 0.002)

    # ── the hold-step change: recoverable, not fatal ───────────────────────────────────────────
    # 6 vs 16 was worth 0.40 vs 0.83. The policy is not WRONG, the requirement moved -> ADAPT.
    r = resolve_contracts(g, dataclasses.replace(
        g, execution=dataclasses.replace(g.execution, hold_steps=24)))
    check("changing hold_steps resolves to ADAPT", r.verdict == ADAPT)

    # ── deploy-side gates must NOT invalidate a policy ─────────────────────────────────────────
    # The policy never observed these; forcing a retrain for them would make the honest path so
    # expensive that nobody would take it.
    r = resolve_contracts(s, dataclasses.replace(
        s, release=dataclasses.replace(s.release, centering_tolerance=0.012, ramp_steps=8)))
    check("tightening the deploy-side release gate is RUN", r.verdict == RUN)
    r = resolve_contracts(g, dataclasses.replace(
        g, execution=dataclasses.replace(g.execution, budget=60)))
    check("a longer step budget is RUN", r.verdict == RUN)

    # ── rig changes ───────────────────────────────────────────────────────────────────────────
    base_only = dataclasses.replace(g, rig=Rig.so101(cameras=("base",)))
    r = resolve_contracts(g, base_only)
    check("losing the wrist camera resolves to ADAPT (re-distil the student)", r.verdict == ADAPT)

    other_arm = dataclasses.replace(g, rig=dataclasses.replace(g.rig, embodiment="panda", dof=7))
    r = resolve_contracts(g, other_arm)
    check("a different arm forces RETRAIN", r.verdict == RETRAIN)

    moved_cam = dataclasses.replace(g, rig=dataclasses.replace(
        g.rig, cameras=(dataclasses.replace(g.rig.cameras[0], pos=(-0.4, -0.15, 0.5)),) + g.rig.cameras[1:]))
    r = resolve_contracts(g, moved_cam)
    check("moving a camera resolves to ADAPT, not RUN", r.verdict == ADAPT)

    relabelled = dataclasses.replace(g, rig=dataclasses.replace(g.rig, name="my-arm"))
    r = resolve_contracts(g, relabelled)
    check("renaming the rig is RUN (documentation, not physics)", r.verdict == RUN)

    # ── unknown fields are RETRAIN, not RUN ───────────────────────────────────────────────────
    # An unrecognised difference is not evidence of safety. This is the property that keeps the
    # table honest as the Contract grows: a new field is conservative until someone classifies it.
    check("an unknown field defaults to RETRAIN", severity_of("release.some_new_field") == RETRAIN)
    check("longest-prefix wins over a general rule",
          severity_of("grasp.signal.force_threshold_n") == RUN and severity_of("grasp.latch_on") == ADAPT)

    # ── the explanation is worth reading ──────────────────────────────────────────────────────
    r = resolve_contracts(s, fixed)
    txt = r.explain()
    check("explain() leads with the verdict", txt.startswith("RETRAIN"))
    check("explain() shows the offending field", "release.height_above_resting" in txt)


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
