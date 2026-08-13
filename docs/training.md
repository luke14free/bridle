# Training from a document

[← README](../README.md)

```
skill.yaml ─▶ SkillSpec ─▶ RewardPlan ─▶ env ─▶ preflight ─▶ PPO ─▶ stamped checkpoint
             schema       compile        adapter   ~30 s
             (no GPU)     (no GPU)
```

Everything before `PPO` runs on a laptop. The expensive step is the last one you take.

## The compiled plan

```python
from bridle.skill.spec    import parse_spec
from bridle.skill.compile import compile_spec

plan = compile_spec(parse_spec(doc), horizon=64)

plan.ops              # the fold, in order
plan.measures_needed  # only these get computed at runtime
plan.warnings         # notes the compiler wants a human to read
plan.fingerprint()    # 'plan@95babe2a3cc5'
```

The fingerprint is sha256 over canonical JSON — never Python's `hash()`, which is salted per process
and would make a stamped checkpoint unverifiable on the next run. It covers the ops and the scale. It
deliberately does **not** cover `why:` prose, the horizon, or the `scene:` block.

It moves when any weight, term, row order or success criterion moves. That is the point: *was this
trained under a different reward?* becomes answerable.

## Becoming an environment

`bridle.adapters.skill_env` turns the plan into functions a simulator env can call. This is the only
part that imports torch.

```python
from bridle.adapters.skill_env import build_reward_fn, build_success_fn, build_reset_fn

class MySkillEnv(TheRegisteredEnv):          # subclass the env `env_id` names
    compute_dense_reward = build_reward_fn(plan)
```

**Override the reward only.** Leaving the scene, the episode initialisation, `evaluate()` and the
termination plumbing untouched means the reward is the single changed variable, and a training run is
comparable to the lineage it reproduces rather than to nothing.

Everything is batched. There is no Python loop over environments and no Python `if` on a batched
condition — one would take a single branch for all 4096 environments at once. `where` is written
branch-free as `c*a + (1-c)*b` for exactly that reason.

`StateSlots` owns per-environment buffers for stateful terms and writes **only the rows in the reset
mask**. Handing a simulator's setter a full `(num_envs, …)` tensor raises the instant one environment
resets on its own — which is every step under partial reset — and seeding a whole tensor injects a
spurious progress spike into every environment that did not restart.

## Verify before you spend the GPU

A compiled reward that is *not* the function it claims to be produces a checkpoint nobody can
interpret. So prove it first, against the implementation you are replacing:

- **On recorded states** — restore a snapshot, compute both rewards from one env state with no step
  in between, and compare. On the reference port: **0 of 4456 states above 1e-5.**
- **On a live rollout** — step the original env and the derived env with identical actions from the
  same seed. Stronger, because it walks states a policy actually reaches rather than recorded initial
  ones: **0 of 12,288 env-steps above 1e-5.**

Two traps worth knowing, both of which quietly make such a test vacuous:

- **Check your coverage, not just your diff.** If every gated term is off — an ungrasped object, a
  reset frame before contacts resolve — the reward collapses to a constant and two constants match
  trivially. Report the gate fractions next to the diff.
- **The environment may not be deterministic.** With domain randomisation on, a deployed env compared
  *against itself* diverged to 1.44 in reward — the identical figure a naive comparison blamed on the
  port. Run the comparison deterministically, and keep a control that re-measures that divergence so
  you know the determinism was necessary rather than convenient.

**If it does not match, do not relax the tolerance.** Find the term that differs; the per-term
breakdown says which. A parity test relaxed until it passes proves nothing.

## Preflight

Before a long run, assert the structural things that make it *capable* of succeeding — a config flag
that is actually set in this process, a warm-start metric above a floor. A doomed run refused in 30
seconds beats one discovered 15M steps later.

The rule that makes it worth having: **"cannot verify" must never render as "verified."** A gate that
prints `0 violations` when the thing it checks was unreachable is worse than no gate.

## Diagnostics

Training emits per-term contributions — min, mean, max per row — and turns them into **typed**
diagnostics: a term that is constant across the rollout is *unoptimizable*; a term that is 84% of
return is *flooding*; high return with low success is *hacking*. Each names its row and says what to
do about it.

The typing is the load-bearing part, not the loop. One-shot reward authoring measures 58.3% ± 47.3%
against 97.6% with a few refinement rounds — and stripping the diagnostic tags collapses the benefit
to 11.5%.

## Stage planning with Foundry

`Foundry` executes a recipe's stages against your rig. bridle does not own the taxonomy of training:
a recipe *names* stages, and you register a runner per kind. The target contract reaches each stage
as environment, so training reads the numbers instead of repeating them.

```python
from bridle import Foundry, ShellStageRunner

foundry = Foundry({k: ShellStageRunner(k, cwd=".", dry_run=True)
                   for k in ("teacher", "round_robin", "distill", "student")})
print(foundry.build(app, rig, plan).explain())
```

A stage with no registered runner is an error before anything launches, not three GPU-hours in.
`dry_run=True` prints the exact command and environment without executing.

## Stamp the result

`stamp(state_dict, contract)` at save; `verify(state_dict, contract)` at load raises
`ContractMismatch` with a field diff. Put the plan fingerprint in the run name too — some trackers
overwrite the run name from their own arguments and silently drop the tags you set through the
environment, so a fingerprint that lives only in a tag can vanish.
