"""bridle.adapters.skill_telemetry — the third feedback tier, wired to a running PPO job.

WHAT THIS IS. `bridle/skill/diagnose.py` turns per-term contribution statistics into typed
diagnostics. It shipped with 58 tests and ZERO CALLERS (2026-08-13 whole-branch review, finding I7):
nothing produced `term_stats`, because `build_reward_fn` returned only the scalar. This module is the
producer and the emitter — the window that accumulates a rollout's per-row min/mean/max, the episode
bookkeeping that makes a success rate out of what a reward function can see, and the emission that
puts the diagnostics in the run log and the scalars on the wandb dashboard.

WHY IT IS WORTH THE HOT-PATH COST, stated as the measurement that motivated it. On 2026-08-13 the
skill-document reproduction of `descend_to_target` sat at `eval_success_at_end` 0.0-0.0625 at ~12.5M
steps while the lineage it reproduces (`descend-teacher-seed20`) was at 0.375-0.8125 around 13.1M.
The reward is proven equal to the deployed one on 0/4456 recorded states above 1e-5
(`scripts/reward_equivalence.py`), so the interesting question is no longer "is it the same
function" but "what is the policy doing with it" — is one row flooding the return, is a row constant
and therefore unoptimizable, is the policy collecting the shaping ceiling without finishing. Those
are exactly the six patterns `diagnose` names, and none of them were answerable from the log.

WHAT THE WINDOW IS, AND WHY IT RESETS. Every emission describes the LAST N control steps and nothing
before them. A running statistic over all of history would answer "was this row ever flat" when the
question a reader has at 3am is "is this row flat NOW" — and a min/max that never resets is
monotone, so it converges on a verdict that can no longer change. `diagnose`'s own framing is "a
finished run's statistics"; a window is the same object over a recent slice.

TRACKING MUST NEVER BE ABLE TO KILL A TRAINING RUN. Every wandb path, every file write and the
diagnose call itself are wrapped: a failure downgrades to a printed note and the fold keeps folding.
The one thing that is NOT swallowed is a caller bug in `observe` (a wrong-shaped contribution), which
would mean the numbers are wrong rather than absent.

torch lives here, like everywhere else under `bridle/adapters/`; `bridle/skill/**` stays stdlib-only.
"""
import json
import os
import time

from bridle.skill.diagnose import diagnose, format_diagnostics

__all__ = ["TermWindow", "RewardTelemetry", "EMIT_EVERY_STEPS"]


#: CONTROL STEPS BETWEEN EMISSIONS, and the number is chosen rather than picked.
#:
#: `train_ppo_state.py` runs `--ppo.num-steps=16` control steps per update and evaluates every
#: `eval_freq` updates, whose default is 25 and which `primitives/descend_to_target`'s lineage does
#: not override (`lerobot_sim2real/rl/ppo_state.py:117,756`). 25 * 16 = 400, so a diagnostic block
#: lands in the log next to the `eval_success_at_end` line it is there to explain, rather than
#: somewhere between two of them. `--ppo.early-stop-patience=25` counts those same evals, so the
#: cadence is also "one block per patience window".
#:
#: The other end of the trade, measured on the run this was built for: 4096 envs at 7850 SPS
#: (`logs/descend_to_target-skill-seed20-plan95babe2a3cc5.log`, epoch 291) is 8.35 s per update, so
#: 400 steps is ~3.5 minutes and a 150M-step run emits ~91 blocks. Frequent enough to see a trend
#: within an hour of watching; rare enough that the log is not mostly diagnostics.
EMIT_EVERY_STEPS = 400


class TermWindow:
    """Running per-row min/mean/max plus the episode outcomes, over ONE window, on the GPU.

    NOTHING HERE SYNCHRONISES UNTIL `snapshot()`. Reducing a `(rows, num_envs)` batch to three
    `(rows,)` tensors is a handful of kernels; calling `.item()` on any of them is a host-device
    round trip, which in a reward hot path is the cost `skill_env._WARNED_RESTING_FRAME` exists to
    document. So the accumulators stay device tensors for the whole window and are read exactly once,
    at emission, `EMIT_EVERY_STEPS` steps apart.

    MEASURED on 4096 CUDA envs and descend's 9 rows (2026-08-13, on the box running the live job):
    `observe` costs 24 us for the per-row reduction and 61 us more for the episode bookkeeping
    below — 85 us against a 353 us reward fold and a 524 ms control step, i.e. 0.016% of a step.
    The bookkeeping is the larger half because it is ~16 tiny kernels, launch-bound rather than
    work-bound; it is left unfused because the number that matters is the one against the step.

    THE EPISODE BOOKKEEPING, AND THE ONE THING A REWARD FUNCTION CANNOT SEE. `compute_dense_reward`
    is handed `info` and the env; it is NOT told `terminated`/`truncated`. So an episode boundary is
    detected from the env's own per-env step counter: `elapsed_steps` is `(num_envs,)` and a row that
    restarted since the previous call has an `elapsed` no greater than the one it had before. That is
    correct under `--partial-reset`, where the counters differ across the batch and rows finish at
    different steps — the same reason `StateSlots.fresh_rows` compares elementwise against that
    counter rather than counting calls.

    A window with NO completed episode reports `success_rate=None`. It is not 0.0, and it is not the
    last window's value: `diagnose` conditions every composition rule on the run failing, so an
    invented 0.0 would license the whole composition block on no evidence at all.
    """

    def __init__(self, row_names):
        self.rows = tuple(row_names)
        #: EPISODE-IN-FLIGHT state, allocated once and NEVER cleared by `reset()` — see there. It
        #: needs its own sentinel for exactly that reason: while the window counters below were the
        #: only sentinel, `reset()` set them to None and the next `observe` re-seeded these too, so
        #: every episode straddling an emission was re-timed from the emission (measured: a 4-step
        #: episode reported as 1 step).
        self._ep_len = self._ep_success = self._prev = None
        self.reset()

    def reset(self):
        """Start a new window. The EPISODE-IN-FLIGHT state (`_ep_len`, `_ep_success`, `_prev`) is
        deliberately NOT cleared — those belong to episodes that are still running and span the
        boundary. Clearing them would silently shorten every episode that straddles an emission."""
        self.steps = 0
        self._min = self._max = self._sum = None
        self._n = 0
        self._episodes = self._successes = self._length = None

    # ── the hot path ────────────────────────────────────────────────────────
    def observe(self, contributions, success=None, elapsed=None):
        """One control step. `contributions` is `{row address -> (N,) Tensor}` from
        `build_contribution_fn`; `success` is `info["success"]` and `elapsed` is
        `env.elapsed_steps`, either of which may be absent."""
        import torch
        if tuple(contributions) != self.rows:
            raise ValueError(
                f"this window was built for rows {self.rows} and was handed {tuple(contributions)}. "
                f"The row addresses are the keys `diagnose` reports against, so a window that "
                f"silently accepted a different set would attribute one row's numbers to another")
        batch = torch.stack([contributions[name] for name in self.rows])   # (rows, num_envs)
        lo, hi, total = batch.amin(dim=1), batch.amax(dim=1), batch.sum(dim=1)
        if self._min is None:
            self._min, self._max, self._sum = lo, hi, total
        else:
            self._min = torch.minimum(self._min, lo)
            self._max = torch.maximum(self._max, hi)
            self._sum = self._sum + total
        self._n += int(batch.shape[1])
        self.steps += 1
        self._observe_episodes(torch, success, elapsed)

    def _observe_episodes(self, torch, success, elapsed):
        if not torch.is_tensor(elapsed) or not torch.is_tensor(success):
            return
        elapsed = elapsed.reshape(-1).float()
        done = success.reshape(-1).float()
        if self._ep_len is None:
            # `_prev = elapsed - 1` and not `elapsed`: at the first call after a reset `elapsed` is
            # already 1, and seeding `_prev` with it would read `1 <= 1` as a boundary and book a
            # zero-length episode that never happened.
            self._prev = elapsed - 1.0
            self._ep_len = torch.zeros_like(elapsed)
            self._ep_success = torch.zeros_like(elapsed)
        if self._episodes is None:
            zero = torch.zeros((), dtype=torch.float32, device=elapsed.device)
            self._episodes, self._successes, self._length = zero, zero.clone(), zero.clone()
        # `fresh` GUARDS AGAINST A SECOND READ INSIDE ONE CONTROL STEP, and that is not a theoretical
        # hazard — it is the same one `StateSlots.fresh_rows` exists for, and it BIT during this
        # module's own GPU integration check (2026-08-13): a harness that called
        # `compute_dense_reward` once itself on top of the one `env.step` performs saw every episode
        # split in two and reported `ep_len` 2.0 on a 64-step horizon. Left unguarded that is a
        # silently wrong success rate, which is the one input every composition rule reads.
        #
        # THE BOUNDARY IS THEREFORE STRICT (`<`), NOT `<=`, AND THE BLIND SPOT IS STATED: two
        # consecutive episodes of exactly ONE control step each are indistinguishable, from the step
        # counter alone, from one step read twice, and this reads them as the latter — so they merge
        # into one 2-step episode and one latched outcome. The trade is deliberate. A 1-step episode
        # needs success (or termination) on the first action after a reset, which the horizons in
        # this corpus (descend: 64) make vanishingly rare; a duplicated read is a mistake a future
        # caller can make in one line, and it would corrupt every window rather than one episode.
        fresh = elapsed != self._prev
        boundary = elapsed < self._prev
        finished = boundary & (self._ep_len > 0)
        mask = finished.float()
        self._episodes = self._episodes + mask.sum()
        self._successes = self._successes + (self._ep_success * mask).sum()
        self._length = self._length + (self._ep_len * mask).sum()
        self._ep_len = (torch.where(boundary, torch.zeros_like(self._ep_len), self._ep_len)
                        + fresh.to(self._ep_len.dtype))
        self._ep_success = torch.where(boundary, torch.zeros_like(self._ep_success),
                                       self._ep_success)
        # Latched on EVERY read including a repeat: `maximum` is idempotent, and a success published
        # on a step this window saw twice is still a success this window saw.
        self._ep_success = torch.maximum(self._ep_success, done)
        self._prev = elapsed

    # ── the one synchronisation ─────────────────────────────────────────────
    def snapshot(self):
        """`(term_stats, success_rate, ep_len, meta)` as plain Python — the shape `diagnose` reads.

        `success_rate` and `ep_len` are `None` when the window completed no episode. `diagnose`
        REFUSES a `success_rate` of None, so the caller must decide what to do about that rather than
        being handed a fabricated number (`_optional_steps`' rule, applied one level up).
        """
        if self._min is None or not self._n:
            return {}, None, None, {"steps": self.steps, "episodes": 0, "env_steps": self._n}
        lo, hi = self._min.tolist(), self._max.tolist()
        # PER STEP PER ENVIRONMENT, which is the unit every threshold in `diagnose` is written
        # against ("pays N per step on average", "N/step across R rows") and the unit
        # `reward_equivalence.py` section [2] prints. `_n` counts env-steps, so one division does it;
        # dividing by the step count instead would report a whole batch's payout as one row's.
        mean = [v / self._n for v in self._sum.tolist()]
        stats = {name: {"min": lo[i], "mean": mean[i], "max": hi[i]}
                 for i, name in enumerate(self.rows)}
        episodes = int(self._episodes.item()) if self._episodes is not None else 0
        rate = length = None
        if episodes:
            rate = float(self._successes.item()) / episodes
            length = float(self._length.item()) / episodes
        return stats, rate, length, {"steps": self.steps, "episodes": episodes,
                                     "env_steps": self._n}


class RewardTelemetry:
    """Accumulate, and every `every` control steps emit the diagnostics and the scalars.

    `log`      where the prose goes. Defaults to `print`, which under `train_from_skill.py --train`
               is `dup2`'d to the run's log file, so the block lands in the log a reader tails.
    `path`     JSONL, one record per emission. This is what makes `bridle skill diagnose` a verb with
               something to read rather than an empty one, and it is the record that outlives the
               process — the log scrolls, wandb needs a network.
    `prefix`   wandb key namespace. `train_ppo_state.py` builds TWO envs from one id (training and
               eval, `ppo_state.py:469-470`); the caller gives each its own prefix so their series
               cannot land on top of each other.
    """

    def __init__(self, row_names, *, every=EMIT_EVERY_STEPS, path=None, log=print,
                 prefix="reward_terms", label="", horizon=None, plan_fingerprint=None):
        self.window = TermWindow(row_names)
        self.every = int(every)
        self.path = path
        self.log = log
        self.prefix = prefix
        self.label = label
        self.horizon = horizon
        self.plan_fingerprint = plan_fingerprint
        self.emissions = 0
        self.total_steps = 0

    def step(self, contributions, success=None, elapsed=None):
        self.window.observe(contributions, success, elapsed)
        self.total_steps += 1
        if self.window.steps >= self.every:
            self.emit()

    # ── emission ────────────────────────────────────────────────────────────
    def emit(self):
        """Read the window, say what it means, reset it. Returns the record it wrote (for tests).

        The whole body is guarded: a diagnostics block is worth a training run's log line, never a
        training run. The window is reset in a `finally`, so a failure cannot turn the next emission
        into a double-length one and quietly change what "recent" means.
        """
        try:
            return self._emit()
        except Exception as exc:                                   # noqa: BLE001 — see the docstring
            self._say(f"  reward diagnostics SKIPPED this window ({type(exc).__name__}: {exc}) — "
                      f"training is unaffected")
            return None
        finally:
            self.emissions += 1
            self.window.reset()

    def _emit(self):
        stats, rate, ep_len, meta = self.window.snapshot()
        head = (f"reward diagnostics — {self.label or 'skill env'} — window of {meta['steps']} "
                f"control steps ({meta['env_steps']} env-steps, {meta['episodes']} episodes ended), "
                f"total {self.total_steps} steps"
                + (f", plan@{self.plan_fingerprint}" if self.plan_fingerprint else ""))
        lines = ["", head, f"    {'row':<52}{'min':>12}{'mean':>12}{'max':>12}"]
        for name, s in stats.items():
            lines.append(f"    {name:<52}{s['min']:>12.4f}{s['mean']:>12.4f}{s['max']:>12.4f}")
        if rate is None:
            # NOT a substituted 0.0 — see `TermWindow.snapshot`. `diagnose` takes the None and runs
            # the structural half (`dead`/`constant` hold at any success rate), then says out loud
            # that the composition half could not run. Which is the whole point of passing it: the
            # free findings are still worth having, and "not checked" must not read as "clean".
            lines.append("    success rate: NOT MEASURED — no episode ended inside this window")
        else:
            lines.append(f"    success rate: {rate:.4f} over {meta['episodes']} episodes, "
                         f"mean episode length {ep_len:.1f}"
                         + (f" of a {self.horizon}-step horizon" if self.horizon else ""))
        found = diagnose(stats, rate, ep_len, horizon=self.horizon)
        lines.append("")
        lines.append("    " + format_diagnostics(found).replace("\n", "\n    "))
        self._say("\n".join(lines))

        record = {
            "t": time.time(),
            "label": self.label,
            "plan": self.plan_fingerprint,
            "horizon": self.horizon,
            "window": meta,
            "total_steps": self.total_steps,
            "emission": self.emissions,
            "success_rate": rate,
            "ep_len": ep_len,
            "term_stats": stats,
            "diagnostics": [{"tag": d.tag, "row": d.row, "message": d.message} for d in found],
        }
        self._write(record)
        self._to_wandb(stats, rate, ep_len, found)
        return record

    def _say(self, text):
        try:
            self.log(text)
        except Exception:                                          # noqa: BLE001 — never fatal
            pass

    def _write(self, record):
        if not self.path:
            return
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
            with open(self.path, "a") as fh:
                fh.write(json.dumps(record) + "\n")
        except OSError as exc:
            self._say(f"  (could not append to {self.path}: {exc})")

    def _to_wandb(self, stats, rate, ep_len, found):
        """The scalars, if a run is live. GUARDED IN THREE PLACES, deliberately.

        `wandb` may be absent, `wandb.run` may be None (nobody called `init`, or `--ppo.track` is
        off), and `log` may raise on a dead socket. Each is a reason to print and carry on, and the
        prose block above has already gone to the log either way — this is the dashboard copy, not
        the record.

        THE STEP IS `wandb.run.step`, NOT ONE OF OURS. `ppo_state.py` logs everything with an
        explicit `step=global_step` (`ppo_state.py:433`), and wandb drops a payload whose step is
        behind the current one. This env instance does not know `global_step`; pinning to the run's
        current step files the row alongside the most recent PPO metrics, which is where a reader
        comparing `eval/success_at_end` against a flooding tag wants it.
        """
        try:
            import wandb
        except Exception as exc:                                   # noqa: BLE001
            self._say(f"    (wandb not importable: {type(exc).__name__} — scalars are in the log "
                      f"above only)")
            return False
        if getattr(wandb, "run", None) is None:
            self._say("    (wandb.run is None — scalars are in the log above only)")
            return False
        payload = {}
        for name, s in stats.items():
            key = f"{self.prefix}/{_wandb_key(name)}"
            payload[f"{key}/min"] = s["min"]
            payload[f"{key}/mean"] = s["mean"]
            payload[f"{key}/max"] = s["max"]
        if rate is not None:
            payload[f"{self.prefix}/window_success_rate"] = rate
            payload[f"{self.prefix}/window_ep_len"] = ep_len
        for tag in ("flooding", "hacking", "dead", "constant", "sparse", "incomplete"):
            payload[f"{self.prefix}/diag/{tag}"] = sum(1 for d in found if d.tag == tag)
        try:
            wandb.log(payload, step=getattr(wandb.run, "step", None))
        except Exception as exc:                                   # noqa: BLE001
            self._say(f"    (wandb.log failed: {type(exc).__name__}: {exc} — scalars are in the log "
                      f"above only)")
            return False
        return True


def _wandb_key(row_name):
    """`reward[3] HingePenalty(height_above_seat_live)` -> `reward3_HingePenalty_height_above_seat_live`.

    wandb reads `/` as panel-group nesting, so the address's own punctuation has to go or every row
    becomes its own section. The INDEX SURVIVES, because it is what makes the address unique and
    what a reader matches against the log block and against `skill.yaml`.
    """
    out = []
    for ch in row_name:
        if ch.isalnum() or ch == "_":
            out.append(ch)
        elif ch == "[":
            # The one character that DISAPPEARS rather than becoming a separator, so the index stays
            # welded to its word: `reward0_DistancePull_...`, not `reward_0__DistancePull_...`.
            continue
        elif ch in "]() ":
            out.append("_")
    key = "".join(out).strip("_")
    while "__" in key:
        key = key.replace("__", "_")
    return key
