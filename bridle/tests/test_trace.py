"""Unit test for bridle.Trace. No sim, no GPU.

WHY: GRAB_TRACE found the 2026-08-11 root cause in ~20 minutes after a 3-hour retrain found
nothing. Tracing is not a debug afterthought; it is the instrument. Always on, structured.

Run: python -m pytest bridle/tests/test_trace.py
"""
import json
import os
import sys
import tempfile

from bridle.trace import Trace

FAILS = []


def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        FAILS.append(name)


def run_checks():
    t = Trace(primitive="grab")
    t.record(0, grip_cmd=-0.26, force=0.0, latched=False)
    t.record(1, grip_cmd=-0.42, force=17.5, latched=False)
    t.record(2, grip_cmd=0.0, force=11.5, latched=True)

    check("records every step", len(t.rows) == 3)
    check("keeps field values", t.rows[1]["force"] == 17.5)
    check("stamps the step index", t.rows[2]["step"] == 2)
    check("summary counts steps", t.summary()["n_steps"] == 3)
    check("summary finds the latch step", t.summary()["latched_at"] == 2)

    t2 = Trace(primitive="reach")
    check("no latch -> latched_at is None", t2.summary()["latched_at"] is None)

    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.jsonl")
        t.to_jsonl(p)
        lines = [json.loads(x) for x in open(p) if x.strip()]
        check("jsonl has a header plus one row per step", len(lines) == 4)
        check("header names the primitive", lines[0].get("primitive") == "grab")
        check("rows survive the round trip", lines[3]["force"] == 11.5)


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
