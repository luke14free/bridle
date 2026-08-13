# Status, and every known limit

[← README](../README.md)

v0.1. Apache 2.0.

## Built, tested, no GPU required

`Rig` and `Contract` with validation and fingerprints · `resolve` and the four verdicts ·
`checkpoint` stamp / verify / diff · `Runner`, `Rollout` and `Trace` · `App` / `Recipe` / `Artifact`
/ `Store` on disk with `plan()` · `Foundry` stage planning, dry runs and result stamping · the
provider interface, `Orchestrator` and the steerable `AgentSession` · the TUI and the viewer · the
placement geometry, the proprioceptive grasp signal and the routine that fits its thresholds from
recorded traces · `bridle.skill` — the document, its schema, its compiler and the plan fingerprint ·
`bridle.adapters.skill_env` — a compiled plan running as a live environment's reward · a ManiSkill
adapter binding a live session to `Runner`.

21 test modules, no simulator, no GPU, a few seconds.

## Known limits

- **One simulator backend.** ManiSkill. `EnvSpec` is ManiSkill-shaped today, and the first port to
  another backend will pay for it.
- **You supply the execution.** bridle knows which skills are valid, not how to drive your simulator:
  the adapter takes a duck-typed session, and the CLI needs an `--executor`.
- **No skills ship with the library.** The store starts empty; you author or import apps.
- **`scene:` is parsed, not synthesised.** A skill document's `env_id` must name an environment that
  already exists. Authoring a genuinely new scene still needs a Python env file.
- **A skill document has not yet trained a policy to completion.** The compiled reward is proven
  numerically equivalent to a hand-written one, and it trains — but the claim "a reward authored this
  way reaches the same success as the reward it reproduces" is not yet demonstrated end to end. Treat
  it as unproven until it is.
- **`forall` / `for_n` are declared but not evaluable.** They are what "all bricks in the bin" will
  need; today a document using them is refused rather than silently wrong.
- **Parallel-jaw grippers only.** Suction and multi-finger are not modelled.
- **`Runner` is a scalar loop.** Training environments are usually vectorised, so "held N steps" can
  be a weaker claim in training than at deploy.
- **`resolve` compares contracts, not behaviour.** A `run` verdict says nothing contradicts; it does
  not promise the skill is good. That is what the eval is for.
- **`ShellStageRunner` needs a launcher callable** for real runs; without one it refuses rather than
  block on a multi-hour job.
- **Unstamped checkpoints warn rather than fail** (`verify(..., on_missing="warn")`), for migration.
  Flip to `"error"` once everything you load is stamped.
- **The prompt payload is near its ceiling.** `bridle skill vocab` is ~7,600 estimated tokens against
  a budget of 8,000. The next vocabulary addition has to buy its space — and not by deleting the
  rationale prose, which is the evidence the whole design exists to carry forward.

## Planned, not built

- **Composition.** An LLM chains skills at run time today, but nothing composes them into a new
  stored capability.
- **A real-hardware backend**, to carry a skill from sim to a physical arm. The proprioception-only
  grasp signal exists for that goal.
- **Scene synthesis**, so a genuinely new task needs no Python at all.

## The invariants

`AGENTS.md` is the working guide: never copy a number out of a contract, never add a second rollout
loop, stamp everything, classify every new field's severity. Each exists because breaking it cost
someone a day.
