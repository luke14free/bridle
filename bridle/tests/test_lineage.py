"""Unit test for bridle.lineage. No sim, no GPU, no yaml — pure data.

WHY THIS EXISTS: on 2026-08-12 two training launches were thrown away. One trained a lineage
nothing deploys; the other omitted PRIM_CARRY_GRIP_HOLD (which defaults to OFF) and ran at
is_grasped_at_end 0.055 instead of 0.859, where success = is_grasped & low & centered can never
fire. Both were recoverable only from a prose sentence in a YAML field. The functions under test
are what make the training environment DATA instead of prose.

Run: python -m pytest bridle/tests/test_lineage.py
     PYTHONPATH=. python bridle/tests/test_lineage.py
"""
import sys

from bridle.lineage import (
    Change, EmptyDiff, UnknownOverride, apply_overrides, capture_env,
    compare_records, format_diff, require_change, require_known, resolve_env,
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
    # ── capture: only the training namespaces, nothing else ──
    env = {"PRIM_CARRY_GRIP_HOLD": "1", "COORD_EXP": "x", "HOME": "/home/luca",
           "PATH": "/usr/bin", "WANDB_PROJECT": "bridle", "LD_LIBRARY_PATH": "/x"}
    got = capture_env(env)
    check("capture keeps PRIM_", got.get("PRIM_CARRY_GRIP_HOLD") == "1")
    check("capture keeps COORD_", got.get("COORD_EXP") == "x")
    check("capture keeps WANDB_", got.get("WANDB_PROJECT") == "bridle")
    check("capture drops HOME", "HOME" not in got)
    check("capture drops PATH", "PATH" not in got)
    check("capture drops LD_LIBRARY_PATH", "LD_LIBRARY_PATH" not in got)

    # ── the real relaunch: one variable changed, everything else inherited ──
    base = {"PRIM_COORD_OBS": "1", "PRIM_CARRY_GRIP_HOLD": "1",
            "PRIM_CARRY_GRIP_CLOSE": "0.0", "PRIM_DESCEND_CENTER_TOL": "0.025"}
    new, changes = apply_overrides(base, {"PRIM_DESCEND_CENTER_TOL": "0.012"})
    check("override applied", new["PRIM_DESCEND_CENTER_TOL"] == "0.012")
    check("the grip freeze is INHERITED, not re-typed", new["PRIM_CARRY_GRIP_HOLD"] == "1")
    check("exactly one change", len(changes) == 1)
    check("change records before and after",
          changes[0] == Change("PRIM_DESCEND_CENTER_TOL", "0.025", "0.012"))

    # a NEW key is a change with before=None
    _, added = apply_overrides(base, {"PRIM_DESCEND_LOW_BAND": "0.025"})
    check("adding a key is a change with before=None",
          added == [Change("PRIM_DESCEND_LOW_BAND", None, "0.025")])

    # setting a key to the value it already has is NOT a change
    _, same = apply_overrides(base, {"PRIM_COORD_OBS": "1"})
    check("re-setting an identical value is not a change", same == [])

    # ── refusals ──
    check("empty diff is refused", raises(EmptyDiff, require_change, []))
    check("non-empty diff is allowed", require_change(changes) is None)
    readable = {"PRIM_DESCEND_CENTER_TOL", "PRIM_DESCEND_LOW_BAND"}
    check("override of a variable the env never reads is refused",
          raises(UnknownOverride, require_known, {"PRIM_DSTACK_CENTER_TOL": "0.012"}, readable))
    check("override of a readable variable is allowed",
          require_known({"PRIM_DESCEND_CENTER_TOL": "0.012"}, readable) is None)

    # ── the diff is the first thing on screen, so it must be readable ──
    txt = format_diff(changes)
    check("diff names the variable", "PRIM_DESCEND_CENTER_TOL" in txt)
    check("diff shows old -> new", "0.025" in txt and "0.012" in txt)

    # ── resolve the environment the way systemd does, then compare RECORDS ──
    # Repeated assignment is NOT a defect: a drop-in overriding the base unit is the mechanism
    # working, and deploy-widegrab.conf's in-file reverts are deliberate and documented. The real
    # defect is two records of one fact disagreeing.
    base_unit = [
        "Environment=COORD_CKPT_grab=/x/grab-coordv2-seed20/ckpt_GOOD_0p78.pt",
        "Environment=PRIM_COORD_OBS=1",
    ]
    dropin = [
        "# a drop-in overriding the base unit — INTENDED",
        "Environment=COORD_CKPT_grab=/x/grab-coord-wide-disp4-seed20/final_ckpt.pt",
        "Environment=GRAB_COORD_REFRESH_R=0",
        "# ...reverted the same day, with the measurement above",
        "Environment=GRAB_COORD_REFRESH_R=0.004",
    ]
    eff = resolve_env([("base.service", base_unit), ("10-dropin.conf", dropin)])
    check("later source wins", eff["COORD_CKPT_grab"] == "/x/grab-coord-wide-disp4-seed20/final_ckpt.pt")
    check("later assignment within a source wins", eff["GRAB_COORD_REFRESH_R"] == "0.004")
    check("a single assignment survives", eff["PRIM_COORD_OBS"] == "1")
    check("comments are ignored", "#" not in "".join(eff))

    # THE LIVE DEFECT: _pgenv.sh claims to mirror the unit but carries the SHADOWED value.
    claimed = {"COORD_CKPT_grab": "/x/grab-coordv2-seed20/ckpt_GOOD_0p78.pt",
               "PRIM_COORD_OBS": "1"}
    bad = compare_records(eff, claimed, "scripts/_pgenv.sh", prefix="COORD_CKPT_")
    check("the mirror mismatch is caught", len(bad) == 1)
    check("mismatch names the variable", bad[0].key == "COORD_CKPT_grab")
    check("mismatch reports both sides",
          bad[0].effective.endswith("final_ckpt.pt") and bad[0].claimed.endswith("0p78.pt"))
    check("mismatch names the record", bad[0].record == "scripts/_pgenv.sh")
    check("an agreeing record is silent",
          compare_records(eff, {"COORD_CKPT_grab": eff["COORD_CKPT_grab"]},
                          "x", prefix="COORD_CKPT_") == [])
    # a record that simply does not mention a variable is not claiming anything about it
    check("an absent key is not a mismatch",
          compare_records(eff, {}, "x", prefix="COORD_CKPT_") == [])
    # the prefix scopes what a record is taken to be claiming
    check("prefix scopes the comparison",
          compare_records(eff, {"PRIM_COORD_OBS": "0"}, "x", prefix="COORD_CKPT_") == [])

    # ── the relaunch plan: the whole of today's mistake #2, prevented ──
    from bridle.relaunch import build_plan
    manifest = {"name": "place_coord_v3",
                "training": {"launcher": "bash train.sh", "source": "captured",
                             "env": {"PRIM_CARRY_GRIP_HOLD": "1", "PRIM_CARRY_GRIP_CLOSE": "0.0",
                                     "PRIM_COORD_OBS": "1", "PRIM_DESCEND_CENTER_TOL": "0.025"}}}
    doc = {"kind": "carry", "dynamic": {"descend_low_once": {"min": 0.5}}}
    readable = {"PRIM_CARRY_GRIP_HOLD", "PRIM_COORD_OBS", "PRIM_DESCEND_CENTER_TOL"}
    plan = build_plan(manifest, doc, {"PRIM_DESCEND_CENTER_TOL": "0.012"}, "tol12", readable)
    check("relaunch inherits the grip freeze", plan.env["PRIM_CARRY_GRIP_HOLD"] == "1")
    check("relaunch applies the one override", plan.env["PRIM_DESCEND_CENTER_TOL"] == "0.012")
    check("relaunch records exactly one change", len(plan.changes) == 1)
    check("relaunch names the run", plan.env["COORD_EXP"] == "tol12" or plan.exp == "tol12")
    check("relaunch merged the kind asserts",
          any(a.path == "is_grasped_at_end" for a in plan.asserts))
    check("a no-op relaunch is refused",
          raises(EmptyDiff, build_plan, manifest, doc, {}, "x", readable))
    check("an unreadable override is refused",
          raises(UnknownOverride, build_plan, manifest, doc,
                 {"PRIM_DSTACK_CENTER_TOL": "0.012"}, "x", readable))


def test_bridle():
    run_checks()
    assert not FAILS, f"{len(FAILS)} check(s) failed: {FAILS}"


def main():
    run_checks()
    print(f"\n{len(FAILS)} failure(s)")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
