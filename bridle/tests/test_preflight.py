"""Unit test for bridle.preflight. No sim, no GPU — pure data.

WHY THIS EXISTS: on 2026-08-12 a training run went 15M steps with is_grasped_at_end at 0.055
(healthy: 0.859) because PRIM_CARRY_GRIP_HOLD defaults to OFF. `success = is_grasped & low &
centered` could never fire. Nothing objected. These asserts are what objects.

Run: python -m pytest bridle/tests/test_preflight.py
     PYTHONPATH=. python bridle/tests/test_preflight.py
"""
import sys

from bridle.preflight import (
    DYNAMIC, STATIC, Assert, Loosened, evaluate, format_effective, format_failures, merge,
)

FAILS = []


def check(name, cond):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}")
        FAILS.append(name)


def raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc:
        return True
    except Exception:
        return False
    return False


def run_checks():
    # ── bounds ──
    check("min holds at the bound", Assert("m", DYNAMIC, min=0.5).holds(0.5))
    check("min fails below", not Assert("m", DYNAMIC, min=0.5).holds(0.055))
    check("max holds at the bound", Assert("m", DYNAMIC, max=0.99).holds(0.99))
    check("max fails above", not Assert("m", DYNAMIC, max=0.99).holds(1.0))
    check("expect matches exactly", Assert("p", STATIC, expect=True).holds(True))
    check("expect rejects a truthy non-match", not Assert("p", STATIC, expect=True).holds(1))
    check("expect=False holds on False", Assert("p", STATIC, expect=False).holds(False))
    check("expect=False fails on True", not Assert("p", STATIC, expect=False).holds(True))
    check("expect=0 holds on 0", Assert("p", STATIC, expect=0).holds(0))
    check("a None observation never holds", not Assert("m", DYNAMIC, min=0.5).holds(None))

    # ── kind supplies the carry gotchas; the author does not have to know them ──
    eff = merge("carry", ())
    paths = {a.path for a in eff}
    check("kind=carry supplies the grip freeze assert",
          "primitives.coord_mixin.CARRY_GRIP_HOLD" in paths)
    check("kind=carry supplies is_grasped_at_end", "is_grasped_at_end" in paths)
    check("kind asserts are attributed", all(a.source == "kind=carry" for a in eff))

    # ── authored asserts are additive ──
    authored = (Assert("descend_low_once", DYNAMIC, min=0.5),)
    eff = merge("carry", authored)
    check("authored assert survives the merge",
          any(a.path == "descend_low_once" and a.source == "authored" for a in eff))
    check("merge keeps the kind asserts too", len(eff) == len(merge("carry", ())) + 1)

    # ── an authored assert may TIGHTEN but never LOOSEN a kind assert ──
    tighter = (Assert("is_grasped_at_end", DYNAMIC, min=0.8),)
    check("tightening a kind assert is allowed",
          any(a.min == 0.8 for a in merge("carry", tighter) if a.path == "is_grasped_at_end"))
    looser = (Assert("is_grasped_at_end", DYNAMIC, min=0.0),)
    check("loosening a kind assert is REFUSED", raises(Loosened, merge, "carry", looser))
    check("an unknown kind is allowed with no extra asserts", merge(None, authored) == authored)

    # ── evaluation ──
    ok = {"is_grasped_at_end": 0.859, "descend_low_once": 1.0,
          "primitives.coord_mixin.CARRY_GRIP_HOLD": True,
          "primitives.coord_mixin.COORD_OBS": True}
    check("a healthy run passes", evaluate(merge("carry", authored), ok) == [])

    # THE REGRESSION: the exact numbers from the thrown-away run
    bad = dict(ok, is_grasped_at_end=0.055)
    bad["primitives.coord_mixin.CARRY_GRIP_HOLD"] = False
    fails = evaluate(merge("carry", authored), bad)
    bad_paths = {f.assertion.path for f in fails}
    check("the grip-freeze bug fails the STATIC tier", "primitives.coord_mixin.CARRY_GRIP_HOLD" in bad_paths)
    check("the grip-freeze bug fails the DYNAMIC tier", "is_grasped_at_end" in bad_paths)

    # a MISSING metric is a failure, not a pass — this catches a typo'd assert name
    missing = evaluate((Assert("descend_centred_once", DYNAMIC, min=0.5),), ok)
    check("a missing metric fails rather than silently passing", len(missing) == 1)
    check("a missing metric reports observed=None", missing[0].observed is None)

    # ── from_scratch skips only the asserts that need a warm start ──
    ws = (Assert("success_once", DYNAMIC, min=0.3, needs="warm_start"),)
    check("needs=warm_start is evaluated normally by default",
          len(evaluate(ws, {"success_once": 0.0})) == 1)
    check("needs=warm_start is skipped under --from-scratch",
          evaluate(ws, {"success_once": 0.0}, from_scratch=True) == [])

    # ── output is for a machine reader: path, bound, observed ──
    txt = format_failures(fails)
    check("failure text names the path", "is_grasped_at_end" in txt)
    check("failure text states the bound", ">= 0.5" in txt)
    check("failure text states the observation", "0.055" in txt)
    eff_txt = format_effective(merge("carry", authored))
    check("effective list shows provenance", "kind=carry" in eff_txt and "authored" in eff_txt)


def test_bridle():
    run_checks()
    assert not FAILS, f"{len(FAILS)} check(s) failed: {FAILS}"


def main():
    run_checks()
    print(f"\n{len(FAILS)} failure(s)")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
