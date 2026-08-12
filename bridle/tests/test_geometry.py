"""Unit test for bridle.geometry. No sim, no GPU — pure arithmetic.

WHY THIS EXISTS: the 2026-08-11 stack failure was geometry, and it was untestable because it lived
inline in a 200-line macro. The numbers below are the MEASURED live values from that night's traces
(the project journal), so a regression here is a regression against reality, not against a guess.

Run: python -m pytest bridle/tests/test_geometry.py
"""
import dataclasses
import sys

from bridle.contract import Contract
from bridle.geometry import (
    destination_top_z, is_supported, release_center_z, resting_center_z,
)

FAILS = []


def check(name, cond):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}")
        FAILS.append(name)


def close(a, b, tol=1e-9):
    return abs(a - b) < tol


def run_checks():
    r = Contract.stack().release

    # ── the live seed-200 release, reproduced exactly from the trace ──────────────────────────
    # GT: destination cube centre 0.012, half 0.012 -> top 0.024; held cube half 0.012.
    # Traced release: held centre z = 0.050, i.e. +1.44cm above its resting height of 0.036.
    top = destination_top_z(r, detected_z=0.012)
    check("today's rule reproduces the deployed goal top (assumed half 0.014)", close(top, 0.026))
    check("...which is 2mm ABOVE the true top of a 0.012-half cube (0.024)", close(top - 0.024, 0.002))

    true_top = destination_top_z(dataclasses.replace(r, destination_top_rule="detected_half"),
                                 detected_z=0.012, detected_half=0.012)
    check("the detected_half rule gets the true top", close(true_top, 0.024))
    check("resting centre of the held cube is 0.036", close(resting_center_z(true_top, 0.012), 0.036))
    check("release centre under today's contract is 0.051 (~the traced 0.050)",
          close(release_center_z(r, true_top, 0.012), 0.051))

    # The fix, expressed as data: drop height_above_resting and the release lands on the cube.
    fixed = dataclasses.replace(r, height_above_resting=0.002)
    check("dropping height_above_resting to 2mm puts the release 2mm above resting",
          close(release_center_z(fixed, true_top, 0.012) - 0.036, 0.002))

    # ── the large-cube case perception cannot currently see ───────────────────────────────────
    big_top = destination_top_z(dataclasses.replace(r, destination_top_rule="detected_half"),
                                detected_z=0.016, detected_half=0.016)
    check("a 0.016-half cube's top is 0.032", close(big_top, 0.032))
    check("today's assumed_half rule aims 2mm BELOW it",
          close(destination_top_z(r, detected_z=0.016), 0.030))

    # ── platform rule ─────────────────────────────────────────────────────────────────────────
    plat = destination_top_z(dataclasses.replace(r, destination_top_rule="platform_constant"),
                             detected_z=0.29)   # detected z deliberately absurd; the rule ignores it
    check("platform_constant ignores the detection and returns the fixture height", close(plat, 0.03))

    # ── refusing to guess ─────────────────────────────────────────────────────────────────────
    try:
        destination_top_z(dataclasses.replace(r, destination_top_rule="detected_half"),
                          detected_z=0.012)
        check("detected_half without a size REFUSES rather than assuming", False)
    except ValueError:
        check("detected_half without a size REFUSES rather than assuming", True)

    # ── support: the second, independent cause of the 0/20 ────────────────────────────────────
    check("a release 2.06cm off a 2.4cm cube is NOT supported (it slid off, measured)",
          is_supported(r, xy_error=0.0206, base_half=0.012) is False)
    check("a release 0.73cm off IS supported (seed 200 — it failed on HEIGHT, not centring)",
          is_supported(r, xy_error=0.0073, base_half=0.012) is True)
    check("the deployed gate admits nearly 3x what a 2.4cm cube can support",
          r.centering_tolerance > 0.012 * 2)


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
