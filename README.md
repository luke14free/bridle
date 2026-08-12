# bridle

**A harness for robot skills — bring your own LLM, your own simulator, and your own robot.**

Think of it as a coding agent for robotics. You describe your rig; bridle gives your LLM a library of
robot skills that actually work *on that rig*, retraining them when they don't, and lets you compose
new capabilities out of the ones you have.

> **Status: early.** The execution substrate below is built, tested, and running a real robot. The
> rest of this page is the design it is being built toward. Sections are marked ✅ **built** or
> 🚧 **designed** — nothing here is aspirationally described as finished.

---

## The problem

Robot skills do not transfer. A grasp policy trained on someone else's arm, camera and control loop
is not a grasp policy on yours — and worse, it usually *runs anyway* and silently underperforms.

That is not a theoretical worry. Two measurements from the codebase bridle was extracted from:

- A policy trained to hold a grasp for **16** steps, deployed in a loop that let go after **6**:
  success **0.40 vs 0.83** on identical seeds (p=0.00012).
- A policy trained to release above a flat **platform**, deployed above a 2.4 cm **cube**: it released
  1.4 cm too high and the cube bounced off. Stacking scored **0/20 for two days** while the search
  looked for a bug in the policy. There wasn't one — it was executing a contract it had never been
  trained under.

Neither was visible to any benchmark, because **benchmarks run inside the training contract**.

So the first thing a robotics harness needs is not an agent loop. It is a way to state what a skill
assumes, and to notice — mechanically — when those assumptions do not hold on your machine.

## The idea

Every skill carries the **contract** it was trained under: the rig it assumed, the loop that executed
it, the physical tolerances it was graded on. Your setup declares its own contract. Then:

| | |
|---|---|
| fingerprints **match** | run the pretrained weights |
| fingerprints **differ** | the diff says *which* assumptions broke → fine-tune, re-distil, or retrain |
| no weights for your rig | regenerate the skill from its **recipe** |

That last row is why an app in bridle is a **recipe first, weights second** — a reproducible training
procedure (environment, teacher, reward, pipeline, evaluation), with pretrained checkpoints as a fast
path guarded by the fingerprint. Given how contract-sensitive these policies measurably are, the
honest promise is *reproducible skills*, not *portable weights*.

## What's built today ✅

The execution substrate: the contract, the loop, and the arithmetic. Stdlib-only — no torch, no
simulator, testable anywhere.

```python
from bridle import Contract, Runner, Trace
from bridle.runner import Rollout

contract = Contract.grab()               # the single definition of the numbers

# TRAINING and DEPLOY both read the same object — neither keeps its own literal.
hold = contract.execution.hold_steps     # not a copy of 16

result = Runner(contract, Trace("grab")).run(Rollout(
    policy=policy_fn, step=step_fn, grasped=grasp_fn, gripper_zero=zero_fn))
```

**`Contract`** — frozen, validated, fingerprinted:

- `Actuation` — action shaping (`gripper_dim`, clamp)
- `Execution` — **how the loop runs**: budget, a gripper rule, an ORDERED tuple of termination rules
- `Grasp` — what a grasp *is*: the latch rule and the sensor
- `Release` — placement geometry: release height, centring tolerance, destination-top rule, ramp

Three gripper rules and five termination rules turned out to describe **every** rollout in a real
robotics codebase:

| gripper | | terminate | |
|---|---|---|---|
| `free` | policy owns the gripper | `sustained_grasp` | grasp survived N steps |
| `zero_always` | carry: never re-open | `linger_after_latch` | N steps after the first latch |
| `zero_after_latch` | grab | `on_goal` | TCP within tolerance of the commanded point |
| | | `on_force` | finger force over threshold |
| | | `sustained_settled` | centred at release height for N steps |

Rule ORDER is part of the contract, and `terminate_pre_step` vs `terminate` says whether a rule is
evaluated before the policy or after the step. Not gratuitous: two rungs of the same codebase
implemented "linger N steps after latch" with an off-by-one between them, and modelling both timings
is what let each convert as a *provable* no-op.

**`Runner`** — the only place a step is taken. `Runner.run(Rollout(...))` executes the contract; the
app supplies callbacks (`policy`, `step`, `grasped`, `at_goal`, `force`, `gripper_zero`, `on_latch`,
`before_step`, `after_step`, `observe`). Adding a phase must never mean adding a loop.

**Checkpoints carry their contract** — the mechanism the rest of the design is built on:

```python
from bridle.checkpoint import stamp, verify

stamp(state_dict, contract)     # at training time
verify(state_dict, contract)    # at load time -> ContractMismatch, with a field-level diff
```

```
ContractMismatch: checkpoint was trained under stack@d05265e578c1 but is being run under stack@a91f...
  Differing fields:
    release.height_above_resting: checkpoint=0.015  runtime=0.002
```

A silent 0/20 found by trace three days later becomes a startup error naming the field.

## Zero privilege ✅

The only grasp signal bridle ships is **proprioceptive** — fingertip contact force and jaw position,
both of which a real arm has. A simulator's `is_grasping` is object-aware, convenient, and *privileged
state*; standardising it would bake a sim-only oracle into every contract written against this
library.

Two gates are required, because each fails alone and in opposite directions: an **open** jaw pressing
the table reads 150–240 N, and a **closed empty** gripper looks exactly like a closed loaded one.

The honest consequence: proprioception cannot say *which* object is held, so `latch_on="target"` is
unimplementable and `validate()` rejects it. Thresholds are **fitted** from recorded traces
(`bridle.calibrate`), never guessed.

## The rest of it ✅🚧

| | component | what it does | |
|---|---|---|---|
| **M1** | `Rig` + `resolve()` | your setup as data; per-field **run / adapt / retrain** with reasons | ✅ |
| **M2** | App store as **recipes** | manifest + recipe + stamped artifacts; `plan(app, rig)` also answers **blocked** | ✅ |
| **M3** | `Foundry` | executes a recipe on your rig, injecting the contract; stamps what comes out | ✅ |
| **M4** | Orchestrator | BYO LLM; tools built from the store, filtered to what your rig can run | ✅ |
| **M5** | Real bridge | sim → real → sim, calibration and deployment to hardware | 🚧 |

Also planned: composing novel capabilities by mixing existing skills, and LLM-authored IK for
motions no skill covers.

**One backend, by choice.** ManiSkill only. A second engine would validate abstractions we do not
yet need validated; the cost is that `EnvSpec` is ManiSkill-shaped today and the first port will pay
for it.

### Risks worth naming up front

- **Transferability is the whole bet.** If a hold-step difference is worth 0.43 success, most skills
  will hit the retrain path and "download an app" means "download a training job".
- **Backend-agnostic environment specs are a research project.** Contact models and solvers differ;
  a recipe that converges in one simulator may not in another. To be measured, not assumed.
- **Retraining costs hours-to-days of GPU per skill.** That has to be visible in the UX or the
  product feels broken.

## The TUI

```bash
pip install -e . && bridle tui --model local:qwen3-32b
```

```
 bridle · so101-default · local:qwen3-32b ···························· acting
 ⋯ I'll pick the red cube first.        │ skills (16 ready / 27)
 → pick(obj='red cube')                 │ ● reach              ready
 ✓ picked the red cube (14.2 N)         │ ● descend_to_target  ready
 → place(dest='green cube')             │ ▲ grab_rgb_wrist     needs re-distil
                                        │ ✕ pick_place         needs rebuild
                                        │ · sphere_grab        rig can't run it
                                        ├─────────────────────────────────────
                                        │ jobs
                                        │ descend_to_target  training  ep 413
 ─────────────────────────────────────────────────────────────────────────────
 › _        http://127.0.0.1:8799 · enter send · esc interrupt · ^N model · ^C quit
```

**Steering is the point.** A coding agent that goes wrong writes a bad file. A robot agent that goes
wrong *moves a real arm*, and noticing three tool calls later means a scattered scene. So typing
while the agent runs queues guidance it sees before choosing its next skill, and `esc` stops it
**after the current skill returns** — never mid-skill, because a half-executed grasp leaves the
gripper in a state no policy was trained from.

The skills pane is the contract spine made visible: every skill, and whether it *runs* on your rig,
needs *re-distilling*, needs a *rebuild*, or *can't run here at all*. The agent's tool list is
filtered by the same call, so the model is never offered a skill your robot cannot do.

Bring any model — `local:` / `vllm:` / `ollama:` / `openai:` / `openrouter:` / `anthropic:`, with
`^N` to switch mid-session. `--executor module:function` wires it to your simulator; without one it
runs in dry mode and says so, because silently doing nothing while looking like it worked is the
failure this project exists to prevent.

```bash
bridle skills                 # what runs on this rig, and what doesn't
bridle plan descend_to_target # why a skill needs adapting or rebuilding
```

## The window

A terminal agent cannot show you a robot. You can read that a skill returned `ok` and still not
notice it dragged the cube 20 cm across the table on the way — the failure that cost days here, and
was only ever caught by watching.

```python
from bridle import Rig, Store
from bridle.ui import Viewer

viewer = Viewer(Store("~/.bridle/apps"), Rig.so101()).start()   # http://127.0.0.1:8799
viewer.push_frame(jpeg_bytes)
```

The window shows your rig, the live simulator, running jobs, and every skill annotated with whether
it **runs / needs re-distil / needs a rebuild / can't run on this rig** — from the same `plan()` call
that filters the agent's tool list, so the two cannot disagree.

The TUI and the window are complements, not alternatives: the terminal is where you *talk to* the
agent, the browser is where you *watch* the robot. Video does not belong in a terminal.

bridle has no third-party dependencies, including for its UI — `curses` and `http.server` ship with
Python. Any external harness can still drive it by reading `/api/state`; see
**[docs/pi-extension.md](docs/pi-extension.md)**.

## For agents and LLMs

If you are an LLM writing code against bridle — or a coding agent working in this repo — read
**[AGENTS.md](AGENTS.md)**. It carries the invariants that are easy to get wrong (never copy a number
out of a Contract; never add a second rollout loop; unknown fields mean retrain), the reasoning
behind the severity table, and runnable patterns for each object.

## Install

```bash
pip install -e .                 # core has no dependencies
pip install -e '.[maniskill]'    # adds the torch-backed adapter
```

## Testing

```bash
python -m pytest bridle/tests
```

Each test also runs standalone and exits non-zero on failure, so it works without pytest installed.

## What it has already caught

Real defects found by adopting it, in one codebase, in two days:

- Training required a grasp to survive 16 steps; deploy exited after 6. **0.40 vs 0.83** (p=0.00012).
- A descend policy hovered 1.5 cm above resting exactly as trained; deploy released it there onto a
  2.4 cm cube. **Stacking 0/20** for two days.
- Two grab rungs implementing the same "linger" rule with an off-by-one between them.
- One release tolerance existing as *three* different numbers (0.045 training, 0.035 deploy, ~0.012
  physics).
- Training and deploy disagreeing on what "survived N steps" means — accumulating vs consecutive.

Every one was invisible to benchmarks, because benchmarks run inside the training contract.

## A note on the tests you don't see here

bridle was extracted from a working robot codebase, and the tests that prove it *there* —
step-for-step parity against the loops it replaced, a guard that fails if a rollout loop reappears
outside `Runner`, and a pin on a train/deploy semantic divergence — depend on that codebase's
simulator and live service. They are that project's **adoption** suite, not the library's, so they
stay with it.

What ships here is what runs anywhere: the contract, the loop, and the arithmetic.

---

Licensed under Apache-2.0.
