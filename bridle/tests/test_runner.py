"""Unit test for bridle.Runner with FAKE callables — no sim, no GPU, no torch.

WHY: Runner is the only place a step is taken. If training and deploy both go through it, the
2026-08-11 mismatch (latch on any-cube + exit after 6 vs latch on target + survive 16) cannot be
expressed. These tests pin the two behaviours that were wrong.

Run: python -m pytest bridle/tests/test_runner.py
"""
import sys

from bridle.contract import Contract
from bridle.runner import Runner
from bridle.trace import Trace

FAILS = []


def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        FAILS.append(name)


def make_world(grasp_at, release_at=None):
    """Fake world: grasp_fn goes True at `grasp_at`, and back False at `release_at` (a slipping
    grip — exactly the failure the 16-step hold is meant to reject)."""
    state = {"k": 0, "actions": []}

    def policy_fn():
        return [0.1, 0.0, 0.0, 0.0, 0.0, -0.5]

    def step_fn(action):
        state["actions"].append(list(action))
        state["k"] += 1

    def grasp_fn():
        if release_at is not None and state["k"] >= release_at:
            return False
        return state["k"] >= grasp_at

    def gripper_zero_fn(action):
        a = list(action)
        a[5] = 0.0
        return a

    return state, policy_fn, step_fn, grasp_fn, gripper_zero_fn


def run_checks():
    c = Contract.grab()   # latch_on=target, hold_steps=16, budget=28

    # 1. A grip that HOLDS: latches at 4, survives to the end -> success.
    st, p, s, g, z = make_world(grasp_at=4)
    r = Runner(c, Trace("grab")).run_grasp(p, s, g, z)
    check("latches once the grasp signal fires", r.latched is True)
    check("holds for at least hold_steps", r.held_steps >= c.execution.hold_steps)
    check("a surviving grip succeeds", r.succeeded is True)

    # 2. A grip that SLIPS at 8: latched but never survives 16 -> NOT success.
    #    This is the exact episode deploy was shipping as a win.
    st2, p2, s2, g2, z2 = make_world(grasp_at=4, release_at=8)
    r2 = Runner(c, Trace("grab")).run_grasp(p2, s2, g2, z2)
    check("a slipping grip still latches", r2.latched is True)
    check("a slipping grip does NOT reach hold_steps", r2.held_steps < c.execution.hold_steps)
    check("a slipping grip is NOT success", r2.succeeded is False)

    # 3. The gripper is frozen after latch (contract says freeze_gripper_on_latch).
    post = st["actions"][6:]
    check("gripper dim is zeroed after latch", all(a[5] == 0.0 for a in post))
    check("arm dims still under policy control", all(a[0] == 0.1 for a in post))

    # 4. Budget is respected when the grasp never fires.
    st3, p3, s3, g3, z3 = make_world(grasp_at=10**6)
    r3 = Runner(c, Trace("grab")).run_grasp(p3, s3, g3, z3)
    check("never exceeds the contract budget", r3.steps == c.execution.budget)
    check("no grasp -> no success", r3.succeeded is False)

    # 5. The trace records every step.
    t = Trace("grab")
    st4, p4, s4, g4, z4 = make_world(grasp_at=4)
    Runner(c, t).run_grasp(p4, s4, g4, z4)
    check("trace has one row per step", t.summary()["n_steps"] == len(st4["actions"]))
    check("trace records the latch step", t.summary()["latched_at"] == 4)

    # ── the RELEASE phase (added 2026-08-12 with the generalised Runner) ──────────────────────
    # Same loop, different question: "am I centred at release height?" instead of "am I holding it?".
    # Without this phase the place leg would have needed a second rollout loop — and counting them
    # is how we found there were already five.
    sc = Contract.stack()   # release.hold_steps is None -> burn the budget, which is today's path

    st5, p5, s5, settled5, _ = make_world(grasp_at=4)
    r5 = Runner(sc, Trace("release")).run_release(p5, s5, settled5)
    check("release with hold_steps=None burns the whole budget", r5.steps == sc.execution.budget)
    check("release with hold_steps=None never reports early success", r5.succeeded is False)
    check("release never latches (no latch in this phase)", r5.latched is False)

    # With a sustained-settled requirement, it DOES terminate early — the behaviour the stacking
    # fix will switch on. Settled from step 6, needs 5 consecutive -> ends at step 10.
    import dataclasses
    sc5 = dataclasses.replace(sc, execution=dataclasses.replace(
        sc.execution, terminate=("sustained_settled",), hold_steps=5))
    st6, p6, s6, settled6, _ = make_world(grasp_at=6)
    r6 = Runner(sc5, Trace("release")).run_release(p6, s6, settled6)
    check("release terminates on a SUSTAINED settled state", r6.succeeded is True)
    check("release ends exactly hold_steps after settling", r6.steps == 10)

    # A state that settles then unsettles must NOT count — the run resets, exactly as a slipping
    # grip does. This is the place-leg analogue of the 6-vs-16 bug.
    st7, p7, s7, settled7, _ = make_world(grasp_at=6, release_at=8)
    r7 = Runner(sc5, Trace("release")).run_release(p7, s7, settled7)
    check("a settled state that breaks does NOT succeed", r7.succeeded is False)

    # No predicate at all: pure record-and-run, which is what today's descend does.
    st8, p8, s8, _, _ = make_world(grasp_at=4)
    r8 = Runner(sc, Trace("release")).run_release(p8, s8, None)
    check("release with no predicate runs the budget and records", r8.steps == sc.execution.budget)

    # A contract without the phase must refuse, not silently do nothing.
    try:
        Runner(Contract.grab(), Trace("x")).run_release(p8, s8, None)
        check("run_release on a grasp-only contract raises", False)
    except ValueError:
        check("run_release on a grasp-only contract raises", True)


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
