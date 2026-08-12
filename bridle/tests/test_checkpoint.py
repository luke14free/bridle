"""Unit test for bridle.checkpoint. No sim, no GPU.

WHY THIS EXISTS: on 2026-08-11 a policy trained against flat platform tops was deployed against
2.4cm cube tops. Nothing objected. pick-and-stack read 0/20 for two days while the search went
looking for a bug in the policy — there wasn't one; it was executing a contract it had never been
trained under. These tests are the assertion that such a run now fails at startup instead.

Run: python -m pytest bridle/tests/test_checkpoint.py
"""
import dataclasses
import sys

from bridle.checkpoint import ContractMismatch, diff, stamp, stamped_fingerprint, verify
from bridle.contract import Contract

FAILS = []


def check(name, cond):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}")
        FAILS.append(name)


def run_checks():
    stack = Contract.stack()
    ckpt = stamp({"weights": [1, 2, 3]}, stack)

    check("stamping preserves the rest of the checkpoint", ckpt["weights"] == [1, 2, 3])
    check("the stamp carries the fingerprint", stamped_fingerprint(ckpt) == stack.fingerprint())

    verify(ckpt, stack)
    check("matching contract verifies clean", True)

    # THE 2026-08-11 SCENARIO: same primitive, one number moved. This is the run that produced 0/20.
    fixed = dataclasses.replace(
        stack, release=dataclasses.replace(stack.release, height_above_resting=0.002))
    try:
        verify(ckpt, fixed)
        check("a moved release height is REFUSED", False)
    except ContractMismatch as e:
        check("a moved release height is REFUSED", True)
        check("the error names the field that moved",
              "height_above_resting" in str(e))
        check("the error shows both values", "0.015" in str(e) and "0.002" in str(e))

    d = diff(ckpt, fixed)
    check("diff reports exactly the changed field",
          list(d) == ["release.height_above_resting"] and d["release.height_above_resting"] == (0.015, 0.002))
    check("diff of an identical contract is empty", diff(ckpt, stack) == {})

    # A different primitive entirely must not be executable either.
    try:
        verify(ckpt, Contract.grab())
        check("a different primitive's contract is REFUSED", False)
    except ContractMismatch:
        check("a different primitive's contract is REFUSED", True)

    # ── migration behaviour: every checkpoint that exists today is unstamped ──────────────────
    legacy = {"weights": [1, 2, 3]}
    check("an unstamped checkpoint has no fingerprint", stamped_fingerprint(legacy) is None)
    verify(legacy, stack)                      # warns, does not raise
    check("unstamped + on_missing='warn' proceeds (migration default)", True)
    try:
        verify(legacy, stack, on_missing="error")
        check("unstamped + on_missing='error' raises", False)
    except ContractMismatch:
        check("unstamped + on_missing='error' raises", True)


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
