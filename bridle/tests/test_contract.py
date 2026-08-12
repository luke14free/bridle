"""Unit test for bridle.Contract. No sim, no GPU — pure dataclass validation.

WHY THIS EXISTS: on 2026-08-11 training latched on the TARGET cube and required the grip to
survive HOLD_K=16 frozen steps, while deploy latched on ANY cube and exited after 6. Nothing
detected it; it cost 96.5% of pick failures and the gate read 0.40 instead of 0.83. On 2026-08-11
the SAME bug class reappeared on the place leg: training hovers 1.5cm above resting, deploy releases
there, onto a 2.4cm cube — pick-and-stack 0/20.

The Contract is the declaration that makes those disagreements expressible; `fingerprint()` +
`bridle.checkpoint` are what make executing a policy under the wrong one a startup error instead of
a silent 0/20.

Run: python -m pytest bridle/tests/test_contract.py
(Also collectable by pytest — every check is a plain assert-equivalent with a name.)
"""
import dataclasses
import sys

from bridle.contract import Actuation, Contract, Execution, Grasp, GraspSignal, Release

FAILS = []


def check(name, cond):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}")
        FAILS.append(name)


def rejects(name, build):
    try:
        build().validate()
        check(name, False)
    except ValueError:
        check(name, True)


def _grasp_contract(**over):
    base = dict(latch_on="target", signal=GraspSignal(kind="privileged", force_threshold_n=1.5, jaw_closed_below=-0.6))
    base.update(over)
    return Contract(name="t",
                    actuation=Actuation(gripper_dim=5, action_lo=-1.0, action_hi=1.0),
                    execution=Execution(budget=28, gripper="zero_after_latch",
                                        terminate=("sustained_grasp",), hold_steps=16),
                    grasp=Grasp(**base))


def run_checks():
    # ── the measured grab contract ────────────────────────────────────────────────────────────
    c = Contract.grab()
    check("grab latches on the TARGET, not any cube", c.grasp.latch_on == "target")
    check("grab holds for 16 steps (grab_env HOLD_K)", c.execution.hold_steps == 16)
    check("grab freezes the gripper on latch", c.execution.gripper == "zero_after_latch")
    check("grab budget is 28", c.execution.budget == 28)
    check("grab declares no release phase", c.release is None)
    c.validate()
    check("valid grab contract validates clean", True)

    # ── the measured stack contract: TODAY's values, warts included ───────────────────────────
    s = Contract.stack()
    s.validate()
    check("stack release height is descend_env's _HOVER", s.release.height_above_resting == 0.015)
    check("stack release gate is the deployed 0.035", s.release.centering_tolerance == 0.035)
    check("stack top rule is the RULE deploy actually uses",
          s.release.destination_top_rule == "assumed_half" and s.release.assumed_half_m == 0.014)
    check("stack burns its budget (no sustained-settled requirement today)",
          s.execution.terminate == ())
    check("stack declares no grasp phase", s.grasp is None)

    # ── the recorded defect (design §5 Release): three numbers for one physical quantity ───────
    # This is NOT a passing design. It is preserved verbatim so that routing both call sites
    # through the Contract is a provable no-op, and recorded here so it cannot be quietly forgotten.
    # The stacking-fix spec collapses these to one value and retrains; when it does, this check is
    # what will fail and demand updating.
    gap_cm = (s.release.success_tolerance - s.release.centering_tolerance) * 100
    check(f"KNOWN DEFECT recorded: training scores success at {s.release.success_tolerance*100:.1f}cm "
          f"but deploy releases only within {s.release.centering_tolerance*100:.1f}cm "
          f"(gap {gap_cm:.1f}cm; physics needs ~1.2cm)",
          s.release.success_tolerance == 0.045 and s.release.centering_tolerance == 0.035)

    # ── fingerprint ───────────────────────────────────────────────────────────────────────────
    check("fingerprint is stable across calls", c.fingerprint() == Contract.grab().fingerprint())
    check("fingerprint is 12 hex chars", len(c.fingerprint()) == 12 and
          all(ch in "0123456789abcdef" for ch in c.fingerprint()))
    check("different contracts fingerprint differently", c.fingerprint() != s.fingerprint())
    moved = dataclasses.replace(
        s, release=dataclasses.replace(s.release, height_above_resting=0.002))
    check("moving the release height changes the fingerprint",
          moved.fingerprint() != s.fingerprint())
    renamed = dataclasses.replace(c, name="grab2")
    check("the name is part of the identity (two prims, same numbers, different contract)",
          renamed.fingerprint() != c.fingerprint())

    # ── validation ────────────────────────────────────────────────────────────────────────────
    rejects("bad latch_on rejected", lambda: _grasp_contract(latch_on="sometimes"))
    rejects("a termination rule with no parameter is rejected", lambda: dataclasses.replace(
        c, execution=dataclasses.replace(c.execution, hold_steps=None)))
    rejects("bad grasp signal kind rejected", lambda: _grasp_contract(
        signal=GraspSignal(kind="telepathy", force_threshold_n=1.5, jaw_closed_below=-0.6)))
    # The honest consequence of a proprioceptive signal: force+aperture cannot say WHICH object.
    rejects("latch_on='target' with a proprio signal rejected", lambda: _grasp_contract(
        signal=GraspSignal(kind="proprio", force_threshold_n=1.5, jaw_closed_below=-0.6)))
    rejects("bad destination_top_rule rejected", lambda: dataclasses.replace(
        s, release=dataclasses.replace(s.release, destination_top_rule="vibes")))
    rejects("negative release height rejected", lambda: dataclasses.replace(
        s, release=dataclasses.replace(s.release, height_above_resting=-0.01)))
    rejects("success tolerance TIGHTER than the release gate rejected", lambda: dataclasses.replace(
        s, release=dataclasses.replace(s.release, success_tolerance=0.01)))
    rejects("zero budget rejected", lambda: dataclasses.replace(
        c, execution=dataclasses.replace(c.execution, budget=0)))

    try:
        c.execution.budget = 1
        check("Contract sub-records are frozen", False)
    except Exception:
        check("Contract sub-records are frozen", True)


def test_bridle():
    """pytest entry point — the same checks, reported as one assertion.

    The standalone `main()` below stays the primary interface: the project venv has no pytest, and a
    test you cannot run without installing something is a test that stops being run.
    """
    FAILS.clear()
    run_checks()
    assert not FAILS, f"{len(FAILS)} check(s) failed: {FAILS}"


def main():
    run_checks()
    print(f"\n{len(FAILS)} failure(s)")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
