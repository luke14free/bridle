"""Unit test for bridle.preflight. No sim, no GPU — pure data.

WHY THIS EXISTS: on 2026-08-12 a training run went 15M steps with is_grasped_at_end at 0.055
(healthy: 0.859) because PRIM_CARRY_GRIP_HOLD defaults to OFF. `success = is_grasped & low &
centered` could never fire. Nothing objected. These asserts are what objects.

Run: python -m pytest bridle/tests/test_preflight.py
     PYTHONPATH=. python bridle/tests/test_preflight.py
"""
import sys

import bridle.preflight as preflight
from bridle.preflight import (
    DYNAMIC, NOT_MEASURED, STATIC, Assert, DuplicateFloor, Loosened, evaluate, format_effective,
    format_failures, merge,
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

    # ── a boundless assert cannot be constructed — it is a check that can never fail ──
    check("a boundless assert is REFUSED at construction", raises(ValueError, Assert, "p", STATIC))
    check("expect alone is still constructible", Assert("p", STATIC, expect=True) is not None)
    check("min alone is still constructible", Assert("p", DYNAMIC, min=0.5) is not None)
    check("max alone is still constructible", Assert("p", DYNAMIC, max=0.99) is not None)
    check("min=0.0 is still constructible (zero is a real bound)",
          Assert("p", DYNAMIC, min=0.0) is not None)

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

    # ── two floors on the same path in KIND_ASSERTS is a malformed kind table, not an input to
    # reconcile — merge must refuse it rather than silently checking one path against two bounds ──
    real_kind_asserts = preflight.KIND_ASSERTS
    preflight.KIND_ASSERTS = dict(real_kind_asserts, dup=(
        Assert("dup.path", DYNAMIC, min=0.5, source="kind=dup"),
        Assert("dup.path", DYNAMIC, min=0.5, source="kind=dup"),
    ))
    try:
        check("two kind floors on the same path is REFUSED, not merged",
              raises(DuplicateFloor, merge, "dup", (Assert("dup.path", DYNAMIC, min=0.8),)))
    finally:
        preflight.KIND_ASSERTS = real_kind_asserts
    check("single-floor-per-path merge still works after the restore",
          any(a.min == 0.8 for a in merge("carry", tighter) if a.path == "is_grasped_at_end"))

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

    # ── F4: format_effective marks a needs=warm_start assert, so a reader can tell which lines
    # --from-scratch will silently skip ──
    needs_txt = format_effective((Assert("success_once", DYNAMIC, min=0.3, needs="warm_start"),))
    check("format_effective marks a needs=warm_start assert",
          "needs=warm_start" in needs_txt and "--from-scratch" in needs_txt)
    no_needs_txt = format_effective((Assert("is_grasped_at_end", DYNAMIC, min=0.5),))
    check("format_effective adds no needs marker for a plain assert", "needs=" not in no_needs_txt)

    # ── F1(a): NOT_MEASURED is a distinct sentinel — it must still FAIL every bound shape (an
    # unmeasured assert is not a pass), but format_failures must render it as "not measured", never
    # as "observed missing" (a measured-absent path, i.e. a typo'd name) nor as a violated bound ──
    check("NOT_MEASURED fails a min-floor", not Assert("m", DYNAMIC, min=0.5).holds(NOT_MEASURED))
    check("NOT_MEASURED fails a max-ceiling", not Assert("m", DYNAMIC, max=0.99).holds(NOT_MEASURED))
    check("NOT_MEASURED fails an expect", not Assert("p", STATIC, expect=True).holds(NOT_MEASURED))
    check("NOT_MEASURED is not None (a distinct sentinel)", NOT_MEASURED is not None)

    nm_fails = evaluate((Assert("is_grasped_at_end", DYNAMIC, min=0.5),),
                        {"is_grasped_at_end": NOT_MEASURED})
    check("evaluate fails an unmeasured assert", len(nm_fails) == 1)
    nm_txt = format_failures(nm_fails)
    check("format_failures renders NOT_MEASURED distinctly",
          "not measured" in nm_txt and "static tier failed first" in nm_txt)
    check("format_failures never renders NOT_MEASURED as 'observed missing'",
          "observed missing" not in nm_txt)
    # a genuinely-measured-but-absent path (the typo case) must keep reading as "missing", not get
    # swept into the same text by a loose substring match
    missing_txt = format_failures(missing)
    check("a truly missing (measured-absent) path still reads as 'missing'",
          "observed missing" in missing_txt)

    # ── F1(b)/F3: collect's short-circuit, exercised with a fake `evaluate` and a fake `measure` —
    # no simulator, no GPU. `bridle.adapters.preflight` only imports torch/gym lazily inside
    # `dynamic_metrics`'s own body, so importing `collect` here is safe on CPU. ──
    from bridle.adapters.preflight import collect

    # `bridle.preflight.STATIC` ("static") is a real, always-resolvable attribute — reused as a
    # stand-in STATIC path so this test needs no throwaway module of its own.
    static_ok = (Assert("bridle.preflight.STATIC", STATIC, expect="static"),)
    static_bad = (Assert("bridle.preflight.STATIC", STATIC, expect="not-static"),)
    dyn = (Assert("is_grasped_at_end", DYNAMIC, min=0.5),)

    calls = []

    def fake_measure(env_id, module, ckpt, envs, steps):
        calls.append((env_id, module, ckpt, envs, steps))
        return {"is_grasped_at_end": 0.859}

    vals = collect(static_bad + dyn, "env", "mod", measure=fake_measure)
    check("default short-circuit (stop_on_static_failure=True) skips the fake measurement "
          "when static fails", calls == [])
    check("the short-circuit marks the dynamic path NOT_MEASURED, not absent",
          vals.get("is_grasped_at_end") is NOT_MEASURED)

    calls.clear()
    vals = collect(static_ok + dyn, "env", "mod", measure=fake_measure)
    check("a passing static tier still runs the (fake) dynamic measurement", len(calls) == 1)
    check("the dynamic value comes from the (fake) measurement",
          vals.get("is_grasped_at_end") == 0.859)

    calls.clear()
    vals = collect(static_bad + dyn, "env", "mod", measure=fake_measure,
                   stop_on_static_failure=False)
    check("stop_on_static_failure=False measures regardless of the failing static tier "
          "(scripts/preflight_regression.sh opts into this)", len(calls) == 1)
    check("...and returns the real (fake) value instead of NOT_MEASURED",
          vals.get("is_grasped_at_end") == 0.859)

    calls.clear()
    vals = collect(static_bad, "env", "mod", measure=fake_measure)
    check("zero DYNAMIC asserts -> measure is never called, short-circuit or not", calls == [])

    calls.clear()
    seen_static = []

    def fake_evaluate(asserts, values, from_scratch=False):
        seen_static.append(tuple(asserts))
        return True   # a fake "static failed" verdict, independent of the real evaluate() logic

    vals = collect(static_ok + dyn, "env", "mod", measure=fake_measure, evaluate=fake_evaluate)
    check("evaluate is injectable: a fake evaluate() drives the short-circuit even though the "
          "asserts would pass the real one", calls == [] and vals.get("is_grasped_at_end") is NOT_MEASURED)
    check("the injected evaluate receives exactly the static-tier asserts",
          seen_static == [static_ok])


def test_bridle():
    run_checks()
    assert not FAILS, f"{len(FAILS)} check(s) failed: {FAILS}"


def main():
    run_checks()
    print(f"\n{len(FAILS)} failure(s)")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
