# bridle

**A harness for robot skills. Bring your own LLM, your own simulator, and your own robot.**

You describe your rig — arm, gripper, control mode, cameras, sensors. bridle gives your LLM a library
of robot skills that are known to work *on that rig*. When a skill does not fit your setup, bridle
says exactly which assumption broke, and can rebuild the skill from its recipe.

It is for people running manipulation policies who are tired of a skill that "runs" on their setup
and quietly underperforms because it was trained for a different one.

---

## The core idea: contracts

A trained policy assumes things. An embodiment and an action space. A rollout loop that stops at a
particular moment. Physical tolerances it was graded against. Those assumptions are normally
implicit, spread across a training script and a deploy path, with nothing comparing them.

In bridle they are one object: a **`Contract`**. Every skill carries the contract it was trained
under; your setup declares its own. Comparing the two mechanically answers *does this skill work on
my robot?*

A contract has a stable 12-character **fingerprint** — sha256 over its canonical form, so it survives
across processes and machines. Checkpoints are stamped with it, and loading one under a different
contract is a startup error with a field-level diff rather than a silent bad run.

Equality alone would be too blunt: a longer step budget cannot hurt a policy, a different action
range certainly can. So differences are resolved **per field** into one of four answers:

| verdict | meaning |
|---|---|
| `run` | every difference is inert for the policy — load the weights |
| `adapt` | the policy is recoverable — re-run the perception stages |
| `retrain` | it was trained for a different problem — regenerate from the recipe |
| `blocked` | your rig cannot run this skill at all (no camera, wrong gripper) |

The severity of each field lives in `bridle/resolve.py`: `execution.budget` is `run`, `rig.cameras`
is `adapt` (a vision student memorises the view; the teacher is unaffected), `rig.control_mode` is
`retrain`. A field with no entry resolves to `retrain` — an unrecognised difference is not evidence
of safety.

This is why a skill here is a **recipe first, weights second**. A recipe is a reproducible training
procedure — environment, reward, stages, evaluation. Pretrained weights are a fast path guarded by
the fingerprint. When the fingerprint does not match, there is still something to run.

## Install

```bash
pip install -e .                 # core: stdlib only, Python >= 3.11
pip install -e '.[maniskill]'    # + torch, for the ManiSkill adapter
pip install -e '.[dev]'          # + pytest
```

Core — `Contract`, `Rig`, `Runner`, `Trace`, `Store`, `Foundry`, the LLM loop, the TUI, the viewer —
has **no dependencies at all** and imports on a machine with no GPU and no simulator. PyYAML is
optional: the store falls back to JSON without it.

## Declare your setup

A `Rig` is deliberately not a URDF. It carries only what can invalidate a policy: what the robot
commands, what it grips with, what it observes.

```python
from bridle import Camera, Gripper, Rig

rig = Rig(
    name="bench-arm",
    embodiment="so101",
    dof=5,                                  # arm DOF, excluding the gripper
    control_mode="pd_joint_delta_pos",
    control_hz=20.0,
    gripper=Gripper(kind="parallel_jaw", dim=5, stroke_m=0.035),
    cameras=(Camera(name="base", width=128, height=128,
                    pos=(-0.5, -0.15, 0.5), target=(0.25, 0.05, 0.08), fov_deg=30.0),),
    sensors=("proprio", "rgb", "force"),
)
rig.validate()
rig.fingerprint()          # '69fd2ade8be6'
```

`Rig.so101(cameras=("base", "wrist"))` is a ready-made one. A contract adds the task: how actions are
shaped, how the rollout loop runs, what counts as a grasp, and where the gripper lets go.

```python
from bridle import Actuation, Contract, Execution, Grasp, GraspSignal

contract = Contract(
    name="grab",
    rig=rig,
    actuation=Actuation(gripper_dim=5, action_lo=-1.0, action_hi=1.0),
    execution=Execution(budget=28, gripper="zero_after_latch",
                        terminate=("sustained_grasp",), hold_steps=16),
    grasp=Grasp(latch_on="any",
                signal=GraspSignal(kind="proprio", force_threshold_n=1.5,
                                   jaw_closed_below=-0.60)),
)
contract.validate()        # raises ValueError on an incoherent contract
```

`validate()` rejects contracts that cannot mean anything: a termination rule with no parameter, a
negative release height, a success criterion the deploy gate would then refuse to act on. It also
rejects `latch_on="target"` with a proprioceptive signal — force and jaw aperture say *something* is
between the jaws, never *which* object. `Contract.grab()` and `Contract.stack()` are worked examples.

## Run a rollout

There is exactly one rollout loop. `Execution` describes it as data — a step budget, a gripper rule,
and an ordered tuple of termination rules — so every skill runs through the same `Runner`.

```python
from bridle import Runner, Trace
from bridle.runner import Rollout

trace  = Trace("grab")
result = Runner(contract, trace).run(Rollout(
    policy=policy_fn,          # ()       -> action
    step=step_fn,              # (action) -> advance the world one control step
    grasped=grasp_fn,          # ()       -> bool
    gripper_zero=zero_fn,      # (action) -> action with the gripper dim zeroed
))

result.succeeded, result.steps, result.reason   # reason names the rule that ended the rollout
trace.to_jsonl("grab.jsonl")                    # per-step record
```

`Runner` takes plain callables, so it imports no simulator and no tensor library. Binding those
callables to a live simulator is the adapter's job (`bridle.adapters.maniskill`).

## Ask whether a skill fits

A skill on disk is an `App`: a manifest the LLM reads, a `Recipe` that regenerates it, and stamped
`Artifact`s. `Store.plan(app, rig)` is the product's verb.

```python
from bridle import Store

store = Store("~/.bridle/apps")
plan  = store.plan(store.get("descend_to_target"), rig)

plan.action        # 'run' | 'adapt' | 'retrain' | 'blocked'
print(plan.explain())
# RETRAIN  descend_to_target
#     stages: teacher, round_robin, distill, student
#     RETRAIN — trained for a different problem.
#         [retrain] release.height_above_resting: trained=0.015 target=0.002
```

## Rebuild what does not fit

`Foundry` executes a recipe's stages against your rig. bridle does not own the taxonomy of training:
a recipe *names* stages, and you register a runner per kind. The target contract reaches each stage
as environment, so training reads the numbers instead of repeating them.

```python
from bridle import Foundry, ShellStageRunner

foundry = Foundry({k: ShellStageRunner(k, cwd=".", dry_run=True)
                   for k in ("teacher", "round_robin", "distill", "student")})
job = foundry.build(app, rig, plan)
print(job.explain())
```

A stage with no registered runner is an error before anything launches, not three GPU-hours in.
`dry_run=True` prints the exact command and environment without executing. Whatever a build produces
gets stamped — `stamp(state_dict, contract)` at save time, `verify(state_dict, contract)` at load
time, which raises `ContractMismatch` with a field diff.

## Give an LLM only the skills that work

An LLM offered a tool will call it. So the tool list contains only skills that resolve to `run` on
your rig; everything else comes back separately, with its verdict and the reason.

```python
from bridle import Orchestrator, build_tools, from_spec

provider = from_spec("local:qwen3-32b")     # or "anthropic:...", "openai:...", "ollama:..."
tools, unavailable = build_tools(store, rig)
# unavailable -> [('sphere_grab', 'blocked',
#                  "the rig does not meet this skill's hard requirements: "
#                  "camera 'wrist' (rig has ['base'])")]

def executor(app_name, arguments):
    ...                                      # you drive your simulator here
    return True, "picked the red cube"

session = Orchestrator(provider, store, rig, executor).run("stack the red cube on the green one")
```

`Provider` is one method: `complete(messages, tools) -> {"text": ..., "tool_calls": [...]}`. An
OpenAI-compatible HTTP client, an Anthropic Messages client and a scripted fake ship with it, all
over `urllib`, no SDK.

## The CLI

```bash
bridle skills                              # what runs on this rig, and what doesn't
bridle skill check primitives/descend_to_target/skill.yaml   # is this skill.yaml well-formed?
bridle plan descend_to_target              # why a skill needs adapting or rebuilding
bridle tui --model local:qwen3-32b         # agent TUI + simulator window
```

**`skill` and `skills` are two different commands, and the `s` is the whole difference.**
`bridle skills` (plural) LISTS the apps already trained and in the store, and says whether each one
runs on this rig. `bridle skill` (singular) is the authoring side: a `skill.yaml` — scene, reward,
success — that `bridle` compiles, checks and trains. Its three verbs are

```bash
bridle skill vocab                         # the document grammar + every term, measure and chassis
bridle skill check  <skill.yaml>           # schema, then compile. Exit 1 on the first refusal
bridle skill compile <skill.yaml>          # the resolved plan: every row in fold order, every
                                           #   chassis default, and the plan fingerprint
```

Neither `check` nor `compile` starts a simulator, so a reward is fully checkable before a GPU-second
is spent. `bridle skill vocab` prints the payload you put in the authoring model's prompt — the
intended author of a skill document is a local 27–30B LLM, which is why the refusals name the dotted
path, state the legal set and suggest the nearest match.

`bridle tui` opens two things, because they answer different questions. The terminal is where you
talk to the agent: typing while it runs queues guidance it sees before choosing its next skill, and
ESC stops it cleanly *after* the current skill returns, never mid-grasp. The browser window
(`bridle.ui.Viewer`, default `http://127.0.0.1:8799`) is where you watch — the skill list with live
verdicts, running jobs, and simulator frames you push with `viewer.push_frame(jpeg_bytes)`. Without
`--executor module:function` the TUI runs in **dry mode**: skill calls are reported, nothing moves.

## Status

Built and covered by tests, stdlib-only, no GPU required: `Rig` and `Contract` with validation and
fingerprints; `resolve`; `checkpoint` stamp / verify / diff; `Runner`, `Rollout` and `Trace`;
`App` / `Recipe` / `Artifact` / `Store` on disk with `plan()`; `Foundry` stage planning, dry runs and
result stamping; the provider interface, `Orchestrator` and the steerable `AgentSession`; the TUI and
the viewer; the placement geometry, the proprioceptive grasp signal and the routine that fits its
thresholds from recorded traces; `bridle.skill` — the `skill.yaml` document, its schema, its
compiler and the plan fingerprint; and a ManiSkill adapter binding a live session to `Runner`.

Known limits:

- **One simulator backend.** ManiSkill. `EnvSpec` is ManiSkill-shaped today, and the first port to
  another backend will pay for it.
- **You supply the execution.** bridle knows which skills are valid, not how to drive your
  simulator: the adapter takes a duck-typed session, and the CLI needs an `--executor`.
- **No skills ship with the library.** The store starts empty; you author or import apps. The CLI
  subcommands construct an SO-101 rig, with only the camera list configurable — other rigs work
  through the Python API.
- **`ShellStageRunner` needs a launcher callable** for real runs; without one it refuses rather than
  block on a multi-hour job.
- **Unstamped checkpoints warn rather than fail** (`verify(..., on_missing="warn")`), for migration.
  Flip to `"error"` once everything you load is stamped.
- **`Runner` is a scalar loop.** Training environments are usually vectorised, so "held N steps" can
  be a weaker claim in training than at deploy.
- **`resolve` compares contracts, not behaviour.** A `run` verdict says nothing contradicts; it does
  not promise the skill is good. That is what the eval is for.
- **Parallel-jaw grippers only** in v0.1; suction and multi-finger are not modelled.

Planned, not built: composing new capabilities out of existing skills (an LLM chains them at run
time today, but nothing composes and stores the result), and a real-hardware backend to carry a skill
from sim to a physical arm. The proprioception-only grasp signal exists for that second goal.

## Tests

Twelve test modules, no simulator, no GPU, a few seconds. Both entry points work:

```bash
python -m pytest bridle/tests                       # everything
PYTHONPATH=. python3 bridle/tests/test_resolve.py   # any module standalone, non-zero on failure
```

Around 220 named checks cover contract validation and fingerprinting, every severity verdict, the
rollout loop's rule ordering, checkpoint mismatch diffs, store round-trips, foundry planning, the
agent's interrupt guarantee, and both interfaces' rendering.

`AGENTS.md` is the working guide for the library's invariants — never copy a number out of a
contract, never add a second rollout loop, stamp everything, classify every new field's severity.

## Licence

Apache 2.0. See [LICENSE](LICENSE).
