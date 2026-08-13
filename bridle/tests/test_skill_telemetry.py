"""Unit test for bridle.adapters.skill_telemetry — the producer the third feedback tier never had.

WHAT IS UNDER TEST AND WHY IT IS HERE RATHER THAN ON THE GPU. `bridle/skill/diagnose.py` shipped
with 58 checks and ZERO CALLERS (whole-branch review, finding I7): nothing in either repo produced a
`term_stats` mapping. This file covers the two pieces that close that — the WINDOW that turns a
stream of per-row contributions into min/mean/max plus a success rate, and the EMISSION that calls
`diagnose` and puts the result in the log, in a JSONL record and on wandb. Both are pure bookkeeping
over CPU tensors, so a simulator would add nothing but minutes.

EVERY CHECK BELOW BITES IN BOTH DIRECTIONS, and the two that matter most are the pair [E2]/[E3]:
identical per-row statistics produce a `flooding` diagnostic at a failing success rate and SILENCE at
a working one. That is the property `diagnose`'s module docstring is built around (deployed descend
earns ~27x its success value integrated and is measured at 0.85; a block that fires on it is a block
the author learns to skip, which costs the measured 97.6% outright), and it is the property a wiring
layer is most likely to break by passing the wrong success number.

Run: PYTHONPATH=. python bridle/tests/test_skill_telemetry.py    (the project venv has no pytest)
     python -m pytest bridle/tests/test_skill_telemetry.py
"""
import json
import os
import sys
import tempfile

FAILS = []


def check(name, cond, note=""):
    tail = f"  — {note}" if note else ""
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{tail}")
    if not cond:
        FAILS.append(name)


def raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc as e:
        return e
    except Exception:
        return False
    return False


def close(a, b, tol=1e-6):
    return a is not None and abs(a - b) <= tol


def flat(text):
    """Whitespace-collapsed, for asserting on a message that `format_diagnostics` may have wrapped
    at 96 columns. A raw substring search over wrapped prose tests the wrap width, not the prose."""
    return " ".join(text.split())


# ── fixtures ────────────────────────────────────────────────────────────────────────────────────

ROWS = ("reward[0] DistancePull(object_to_goal_xy)", "reward[1] PredicateBonus(grasped)")


def flooding_stream(torch):
    """Four control steps of a reward where ONE row pays almost everything.

    row[0]: [4.9, 5.0, 5.1, 5.0] every step  -> min 4.9, mean 5.0,   max 5.1
    row[1]: zero except a single 2.0 spike   -> min 0.0, mean 0.125, max 2.0

    earned = 5.125/step, row[0]'s share 97.6% (>= `_DOMINANT_SHARE` 0.5) so `flooding` must fire on
    it; captured = 5.125 / 7.1 = 72.2% (< `_CAPTURED` 0.9) so `hacking` must NOT, which is what makes
    the expected tag list exactly one entry rather than "at least one".
    """
    a = torch.tensor([4.9, 5.0, 5.1, 5.0])
    spike = torch.tensor([2.0, 0.0, 0.0, 0.0])
    zero = torch.zeros(4)
    return [{ROWS[0]: a.clone(), ROWS[1]: (spike if i == 3 else zero).clone()} for i in range(4)]


def ticking(torch, step):
    """`elapsed_steps` for a 2-step horizon over 4 envs: 1, 2, 1, 2, ... so every env ends an
    episode on every odd step and a 4-step window closes exactly 4 episodes."""
    return torch.full((4,), float(1 + (step % 2)))


class Capture:
    def __init__(self):
        self.lines = []

    def __call__(self, text):
        self.lines.append(text)

    @property
    def text(self):
        return "\n".join(self.lines)


class FakeWandb:
    """Stands in for the real module. `run=None` is the "nobody called init" case the brief requires
    to degrade to stdout; `boom=True` is the dead-socket case."""

    class _Run:
        step = 4096

    def __init__(self, live=True, boom=False):
        self.run = self._Run() if live else None
        self.boom = boom
        self.logged = []

    def log(self, payload, step=None):
        if self.boom:
            raise RuntimeError("wandb: network unreachable")
        self.logged.append((payload, step))


class _Swap:
    """Install a module under a name in `sys.modules` for the duration of a block."""

    def __init__(self, name, module):
        self.name, self.module = name, module

    def __enter__(self):
        self.had = self.name in sys.modules
        self.was = sys.modules.get(self.name)
        if self.module is None:
            sys.modules[self.name] = None          # import raises ImportError on a None entry
        else:
            sys.modules[self.name] = self.module
        return self.module

    def __exit__(self, *exc):
        if self.had:
            sys.modules[self.name] = self.was
        else:
            sys.modules.pop(self.name, None)
        return False


# ── the checks ──────────────────────────────────────────────────────────────────────────────────

def run_checks():
    try:
        import torch
    except ImportError as exc:                                      # pragma: no cover
        # NOT a silent skip: torch is in the project venv, so its absence means the run was pointed
        # at the wrong interpreter, and "cannot verify" must not render as "verified".
        check("torch is importable, so this file can run at all", False,
              f"{exc} — run with /home/luca/robotics/maniskill/.venv/bin/python")
        return
    from bridle.adapters.skill_telemetry import EMIT_EVERY_STEPS, RewardTelemetry, TermWindow

    # ── [W] the window ──────────────────────────────────────────────────────────────────────────
    print("\n[W] TermWindow — min/mean/max per row, and the episode bookkeeping")

    w = TermWindow(ROWS)
    for step, contrib in enumerate(flooding_stream(torch)):
        w.observe(contrib, success=torch.zeros(4), elapsed=ticking(torch, step))
    stats, rate, ep_len, meta = w.snapshot()
    a, b = stats[ROWS[0]], stats[ROWS[1]]
    # The mean is PER STEP PER ENVIRONMENT. row[1] paid 2.0 once, on one env, over 4 steps x 4 envs
    # = 16 env-steps, so 0.125. Dividing by the STEP count instead (the plausible wrong divisor)
    # gives 0.5 and this check fails; so does summing without dividing (2.0).
    check("W1 per-row min/mean/max are the per-step-per-env contribution",
          close(a["min"], 4.9) and close(a["mean"], 5.0) and close(a["max"], 5.1)
          and close(b["min"], 0.0) and close(b["mean"], 0.125) and close(b["max"], 2.0),
          f"{a} {b}")
    check("W1b the window reports how much it saw", meta["steps"] == 4 and meta["env_steps"] == 16,
          str(meta))

    # A 2-step horizon over 4 envs: steps 0..3 close one full episode per env on step 2 (elapsed
    # goes 2 -> 1), so 4 episodes ended, each of measured length 2, none of them a success.
    check("W2 episode bookkeeping: 4 episodes of length 2, none successful",
          meta["episodes"] == 4 and close(rate, 0.0) and close(ep_len, 2.0),
          f"episodes={meta['episodes']} rate={rate} ep_len={ep_len}")

    # PARTIAL RESET: rows end their episodes on DIFFERENT steps, which is every step of a real run
    # (`--partial-reset`). Hand-computed: envs 1,3 end on step 3 (length 2, no success), env 0 ends
    # on step 4 (length 3), env 2 ends on step 5 (length 4, and it latched a success on step 2).
    # So 4 episodes, 1 success, total length 2+2+3+4 = 11.
    w2 = TermWindow(ROWS)
    elapsed = [[1, 1, 1, 1], [2, 2, 2, 2], [3, 1, 3, 1], [1, 2, 4, 2], [2, 3, 1, 3]]
    successes = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    steady = {ROWS[0]: torch.ones(4), ROWS[1]: torch.zeros(4)}
    for e, s in zip(elapsed, successes):
        w2.observe({k: v.clone() for k, v in steady.items()},
                   success=torch.tensor(s, dtype=torch.float32),
                   elapsed=torch.tensor(e, dtype=torch.float32))
    _, rate2, len2, meta2 = w2.snapshot()
    check("W3 under partial reset the four envs' episodes are counted where they actually ended",
          meta2["episodes"] == 4 and close(rate2, 0.25) and close(len2, 11.0 / 4.0),
          f"episodes={meta2['episodes']} rate={rate2} ep_len={len2}")

    # An episode that never ends inside the window is NOT a failure and NOT a success — it is
    # unmeasured, and `diagnose` refuses a None success rate rather than being handed a 0.0 that
    # would license every composition rule on no evidence.
    w3 = TermWindow(ROWS)
    for step in range(4):
        w3.observe({k: v.clone() for k, v in steady.items()}, success=torch.zeros(4),
                   elapsed=torch.full((4,), float(step + 1)))
    _, rate3, len3, meta3 = w3.snapshot()
    check("W4 a window in which no episode ended reports success_rate=None, NOT 0.0",
          rate3 is None and len3 is None and meta3["episodes"] == 0, f"{rate3} {len3} {meta3}")

    # A SECOND READ INSIDE ONE CONTROL STEP MUST BE A NO-OP for the episode bookkeeping. This is the
    # `StateSlots.fresh_rows` hazard and it bit for real during the GPU integration check on
    # 2026-08-13: a harness that called `compute_dense_reward` itself, on top of the call `env.step`
    # already makes, saw every episode split at the duplicate and reported `ep_len` 2.0 on a 64-step
    # horizon. Below, each step of a 4-step 2-step-horizon stream is observed TWICE; the answer must
    # be the same 4 episodes of length 2 that W2 measures from single reads, not 8 of length 1.
    w_dup = TermWindow(ROWS)
    for step in range(4):
        for _ in range(2):
            w_dup.observe({k: v.clone() for k, v in steady.items()}, success=torch.zeros(4),
                          elapsed=ticking(torch, step))
    _, _, len_dup, meta_dup = w_dup.snapshot()
    check("W4a a duplicated read of the same control step books no episode and adds no length",
          meta_dup["episodes"] == 4 and close(len_dup, 2.0),
          f"episodes={meta_dup['episodes']} ep_len={len_dup}")

    w4 = TermWindow(ROWS)
    w4.observe({k: v.clone() for k, v in steady.items()})
    _, rate4, _, _ = w4.snapshot()
    check("W5 with no `elapsed`/`success` at all the rate is None, not fabricated", rate4 is None,
          str(rate4))

    # The keys ARE the addresses `diagnose` reports against. A window that accepted a different set
    # would attribute one row's numbers to another row's name.
    err = raises(ValueError, w4.observe, {"reward[9] Other": torch.zeros(4)})
    check("W6 a contribution dict with different row addresses is refused, naming both sets",
          bool(err) and ROWS[0] in str(err) and "reward[9] Other" in str(err), str(err)[:90])

    # `reset()` starts a new window and keeps the IN-FLIGHT episode state: clearing `_ep_len` would
    # silently shorten every episode that straddles an emission.
    w5 = TermWindow(ROWS)
    for step in range(3):
        w5.observe({ROWS[0]: torch.full((4,), 9.0), ROWS[1]: torch.zeros(4)},
                   success=torch.zeros(4), elapsed=torch.full((4,), float(step + 1)))
    w5.reset()
    w5.observe({ROWS[0]: torch.full((4,), 1.0), ROWS[1]: torch.zeros(4)},
               success=torch.zeros(4), elapsed=torch.full((4,), 1.0))
    w5.observe({ROWS[0]: torch.full((4,), 1.0), ROWS[1]: torch.zeros(4)},
               success=torch.zeros(4), elapsed=torch.full((4,), 2.0))
    s5, _, len5, meta5 = w5.snapshot()
    # The episode that was 3 steps old when the window closed must still be reported as 3 steps
    # long, not as 1. With one sentinel for both the window counters and the in-flight buffers,
    # `reset()` re-seeded the buffers and it measured 1 — every episode straddling an emission
    # re-timed from the emission.
    check("W7 reset() drops the old window's numbers (9.0 is gone) but keeps the episode in flight "
          "(the closed episode is 3 steps long, not 1)",
          close(s5[ROWS[0]]["max"], 1.0) and meta5["episodes"] == 4 and close(len5, 3.0),
          f"max={s5[ROWS[0]]['max']} episodes={meta5['episodes']} ep_len={len5}")

    # ── [E] the emission ────────────────────────────────────────────────────────────────────────
    print("\n[E] RewardTelemetry — cadence, the typed block, the record, and the wandb guard")

    def emit_run(*, success_value, every=4, path=None, wandb_module="absent", rows_stream=None):
        cap = Capture()
        tel = RewardTelemetry(ROWS, every=every, path=path, log=cap, label="test", horizon=2,
                              plan_fingerprint="95babe2a3cc5")
        stream = rows_stream or flooding_stream(torch)
        fake = None
        if wandb_module == "absent":
            ctx = _Swap("wandb", None)
        else:
            fake = wandb_module
            ctx = _Swap("wandb", fake)
        with ctx:
            for step, contrib in enumerate(stream):
                tel.step(contrib, success=torch.full((4,), float(success_value)),
                         elapsed=ticking(torch, step))
        return tel, cap, fake

    tel, cap, _ = emit_run(success_value=0.0)
    check("E1 one emission after exactly `every` steps, and not before",
          tel.emissions == 1 and cap.text.count("reward diagnostics —") == 1,
          f"emissions={tel.emissions}")

    # THE TYPED CONTENT IS THE PRODUCT (the ablation: stripping the tags collapses 97.6% to 11.5%),
    # so the assertion is on the tag, the address, and the presence of an instruction.
    check("E2 a failing run with one dominant row emits exactly [flooding], addressed to reward[0]",
          "[flooding] " + ROWS[0] in cap.text and "[hacking]" not in cap.text
          and "[constant]" not in cap.text and "[dead]" not in cap.text,
          cap.text[cap.text.find("[flooding]"):][:60] if "[flooding]" in cap.text else "no tag")
    # Searched against the whitespace-flattened text: `format_diagnostics` wraps at 96 columns, so
    # an instruction can legitimately straddle a line break and a raw substring search would be
    # testing the wrap width rather than the content.
    check("E2b the block says what to do about it, not just that something is wrong",
          "Lower its weight, or gate it on the predicate" in flat(cap.text),
          "message carries the action")
    check("E2c the block names the window it describes, so a reader knows what it is a statement of",
          "window of 4 control steps" in cap.text and "4 episodes ended" in cap.text
          and "plan@95babe2a3cc5" in cap.text)

    # THE OTHER DIRECTION, on THE SAME statistics. This is the check that fails if the wiring hands
    # `diagnose` the wrong success number, which is the single most likely way to break the tier.
    tel_ok, cap_ok, _ = emit_run(success_value=1.0)
    check("E3 the SAME per-row statistics at a WORKING success rate emit no diagnostic at all",
          "no diagnostics" in cap_ok.text and "[flooding]" not in cap_ok.text,
          "silence on a succeeding run")
    check("E3b ...and the per-row numbers are still printed, so silence is not blindness",
          "4.9000" in cap_ok.text and "5.1000" in cap_ok.text)

    # A window in which no episode ended must SAY the verdicts did not run. "Not checked is not
    # clean" is `diagnose`'s own rule and it has to survive the wiring.
    cap_nm = Capture()
    tel_nm = RewardTelemetry(ROWS, every=4, log=cap_nm, label="test")
    for step, contrib in enumerate(flooding_stream(torch)):
        tel_nm.step(contrib, success=torch.zeros(4), elapsed=torch.full((4,), float(step + 1)))
    check("E4 a window with no completed episode reports NOT MEASURED, runs no composition rule, "
          "and says so with a tag rather than by falling silent",
          "NOT MEASURED" in cap_nm.text and "[flooding]" not in cap_nm.text
          and "[incomplete] reward" in cap_nm.text and "4.9000" in cap_nm.text,
          "verdicts named as absent, numbers present")

    # THE WINDOW IS PER-EMISSION. A row that was constant while one window ran and varies in the
    # next must not still be reported constant — that is the whole reason the window resets.
    cap_two = Capture()
    tel_two = RewardTelemetry(ROWS, every=4, log=cap_two, label="test")
    for step in range(4):
        tel_two.step({ROWS[0]: torch.full((4,), 3.0), ROWS[1]: torch.full((4,), 1.0)},
                     success=torch.zeros(4), elapsed=ticking(torch, step))
    first = cap_two.text
    for step in range(4, 8):
        tel_two.step({ROWS[0]: torch.tensor([1.0, 2.0, 3.0, 4.0]),
                      ROWS[1]: torch.tensor([0.5, 1.0, 1.5, 2.0])},
                     success=torch.zeros(4), elapsed=ticking(torch, step))
    second = cap_two.text[len(first):]
    check("E5 `constant` is reported in the window where the row was flat and NOT in the next one, "
          "where it varied",
          "[constant]" in first and "[constant]" not in second,
          f"window1 constant={'[constant]' in first} window2 constant={'[constant]' in second}")

    # ── the record `bridle skill diagnose` reads ────────────────────────────────────────────────
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "nested", "reward_terms.jsonl")
        tel_j, _, _ = emit_run(success_value=0.0, path=path)
        for step, contrib in enumerate(flooding_stream(torch)):
            tel_j.step(contrib, success=torch.zeros(4), elapsed=ticking(torch, step))
        with open(path) as fh:
            records = [json.loads(line) for line in fh if line.strip()]
        ok_shape = (len(records) == 2
                    and set(records[-1]["term_stats"]) == set(ROWS)
                    and records[-1]["success_rate"] == 0.0
                    and records[-1]["diagnostics"][0]["tag"] == "flooding")
        check("E6 one JSONL record per emission, carrying the stats AND the diagnostics",
              ok_shape, f"{len(records)} record(s)")
        # THE RECORD IS THE CLI'S INPUT CONTRACT: it must feed `diagnose` unchanged, or the verb
        # reads a file it cannot use.
        from bridle.skill.diagnose import diagnose as _diagnose
        replayed = _diagnose(records[-1]["term_stats"], records[-1]["success_rate"],
                             records[-1]["ep_len"], horizon=records[-1]["horizon"])
        check("E6b the record round-trips through `diagnose` and reproduces the same tag and row",
              [(d.tag, d.row) for d in replayed]
              == [(d["tag"], d["row"]) for d in records[-1]["diagnostics"]],
              f"{[(d.tag, d.row) for d in replayed]}")

    # ── wandb: three failure modes, none of them fatal ──────────────────────────────────────────
    live = FakeWandb(live=True)
    tel_w, cap_w, fake = emit_run(success_value=0.0, wandb_module=live)
    payload, step = (fake.logged[0] if fake.logged else ({}, None))
    check("E7 with a live wandb run the per-row scalars are logged, keyed by the row's index",
          len(fake.logged) == 1
          and "reward_terms/reward0_DistancePull_object_to_goal_xy/mean" in payload
          and close(payload["reward_terms/reward0_DistancePull_object_to_goal_xy/mean"], 5.0, 1e-5)
          and payload["reward_terms/diag/flooding"] == 1,
          f"{len(payload)} keys")
    check("E7b it is filed at `wandb.run.step` — ppo_state.py logs with an explicit step and wandb "
          "drops a payload that is behind", step == 4096, str(step))

    dead = FakeWandb(live=False)
    tel_d, cap_d, _ = emit_run(success_value=0.0, wandb_module=dead)
    check("E8 `wandb.run is None` degrades to the log and does not raise",
          tel_d.emissions == 1 and "wandb.run is None" in cap_d.text
          and "[flooding]" in cap_d.text, "logged to stdout instead")

    boom = FakeWandb(live=True, boom=True)
    tel_b, cap_b, _ = emit_run(success_value=0.0, wandb_module=boom)
    check("E9 a raising `wandb.log` cannot kill the fold — the emission completes and says why",
          tel_b.emissions == 1 and "wandb.log failed" in cap_b.text and "[flooding]" in cap_b.text,
          "tracking never kills a training run")

    check("E10 with wandb not importable at all the emission still happens",
          tel.emissions == 1 and "wandb not importable" in cap.text and "[flooding]" in cap.text)

    # A FAILURE INSIDE THE EMISSION IS SWALLOWED AND THE WINDOW STILL RESETS. Without the `finally`
    # the next window would be double-length and every "per step" number in it would be a statement
    # about a period nobody named.
    cap_x = Capture()
    tel_x = RewardTelemetry(ROWS, every=2, log=cap_x, label="test")
    tel_x.window.snapshot = lambda: (_ for _ in ()).throw(RuntimeError("snapshot exploded"))
    for step in range(2):
        tel_x.step({ROWS[0]: torch.ones(4), ROWS[1]: torch.zeros(4)},
                   success=torch.zeros(4), elapsed=ticking(torch, step))
    check("E11 an exception inside the emission is reported and swallowed, and the window is reset "
          "anyway", "SKIPPED this window" in cap_x.text and "snapshot exploded" in cap_x.text
          and tel_x.window.steps == 0, cap_x.text[:80])

    # ── [C] the CLI verb that reads the record ──────────────────────────────────────────────────
    # `bridle skill diagnose` was added ONLY because this file now writes something for it to read;
    # a verb with no input is worse than no verb. So its input contract is tested where the input is
    # produced, on a file this module actually wrote.
    print("\n[C] bridle skill diagnose — the verb, against a record this module wrote")
    import contextlib
    import io

    from bridle.cli import main as cli_main

    def cli(*argv):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = cli_main(["skill", "diagnose", *argv])
        return rc, buf.getvalue()

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "reward_terms.jsonl")
        tel_c = RewardTelemetry(ROWS, every=4, path=path, log=lambda *_: None, label="cli",
                                horizon=2, plan_fingerprint="95babe2a3cc5")
        with _Swap("wandb", None):
            for round_ in range(2):
                for step, contrib in enumerate(flooding_stream(torch)):
                    tel_c.step(contrib, success=torch.zeros(4),
                               elapsed=ticking(torch, step + 4 * round_))
        rc, out = cli(path)
        check("C1 the verb reads the last record and re-derives the same typed finding",
              rc == 0 and "[flooding] " + ROWS[0] in out and "plan@95babe2a3cc5" in out
              and "record -1 of 2" in out, f"rc={rc}")
        rc_l, out_l = cli(path, "--list")
        check("C2 --list prints one line per emission with its tags",
              rc_l == 0 and "2 emission(s)" in out_l and out_l.count("flooding") == 2,
              f"rc={rc_l}")
        rc_s, out_s = cli(path, "--stored")
        check("C3 --stored prints what the RUN emitted, tag and address intact",
              rc_s == 0 and "AS EMITTED BY THE RUN ITSELF (1)" in out_s
              and "[flooding] " + ROWS[0] in out_s, f"rc={rc_s}")
        rc_i, out_i = cli(path, "--index", "9")
        check("C4 an out-of-range index FAILS and says how many records there are",
              rc_i == 1 and "outside the 2 record(s)" in out_i, f"rc={rc_i}")
        rc_m, out_m = cli(os.path.join(tmp, "absent.jsonl"))
        check("C5 a missing file FAILS and names what writes one, rather than printing nothing",
              rc_m == 1 and "train_from_skill.py" in out_m and "--no-diagnostics" in out_m,
              f"rc={rc_m}")
        empty = os.path.join(tmp, "empty.jsonl")
        open(empty, "w").close()
        rc_e, out_e = cli(empty)
        # An empty file is NOT "0 diagnostics, all clear" — that is the `bridle lineage` failure
        # shape (`0 violation(s)`, exit 0, on a machine that could not run the check at all).
        check("C6 an empty file is reported as 'no records yet' and exits non-zero, not as clean",
              rc_e == 1 and "no records yet" in out_e, f"rc={rc_e}")

    # ── the cadence constant ────────────────────────────────────────────────────────────────────
    check("E12 the default cadence is one block per eval — 25 updates x --ppo.num-steps=16, the "
          "`eval_freq` default the descend lineage runs under", EMIT_EVERY_STEPS == 25 * 16,
          str(EMIT_EVERY_STEPS))


def test_bridle():
    run_checks()
    assert not FAILS, FAILS


def main():
    run_checks()
    print(f"\n{len(FAILS)} failure(s)" + (f": {FAILS}" if FAILS else ""))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
