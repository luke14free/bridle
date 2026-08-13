# bridle for agents

You are an LLM writing code against `bridle`, or working inside this repository. This file is the
contract for doing that well. It is not a tutorial — it is the set of things that are easy to get
wrong, why they matter, and what to do instead.

`README.md` explains what bridle is *for*. This file explains how to *use* it without breaking the
guarantee it exists to provide.

---

## 1. The one-sentence model

**A skill carries the contract it was trained under. Running it under a different contract is an
error, not a gamble.** Everything else in this library is machinery for stating that contract,
comparing two of them, and rebuilding a skill when they do not match.

If a change you are about to make would let a policy run under a contract it was not trained for,
the change is wrong, however convenient.

## 2. Core objects, and what each one is *for*

| object | answers |
|---|---|
| `Rig` | what robot is this — embodiment, dof, control mode + rate, gripper, cameras, sensors |
| `Contract` | what a skill assumes — `Actuation`, `Execution`, `Grasp`, `Release`, plus the `Rig` |
| `Runner` | how a rollout is executed — **the only place a step is taken** |
| `Trace` | what happened, per step |
| `App` | a skill: manifest + `Recipe` + stamped `Artifact`s |
| `Store` | apps on disk; `plan(app, rig)` → run / adapt / retrain / blocked |
| `Foundry` | executes a recipe to build a skill for a rig |
| `Orchestrator` | an LLM driving skills, with a BYO provider |
| `bridle.skill` | what a skill *is*, before it is trained: `skill.yaml` — scene, reward, success — through `parse_spec` → `compile_spec` → `RewardPlan`, with a fingerprint. Stdlib only, no simulator. `bridle skill check\|compile\|vocab` |

Everything above is importable from the top level: `from bridle import Contract, Rig, Store`.

## 3. Invariants — break these and the library stops meaning anything

**3.1 Never copy a number out of a Contract into a training script or a deploy path.** Read it.

```python
hold = contract.execution.hold_steps        # yes
hold = 16                                   # NO — this is the bug the library exists to prevent
```
Training once required a grasp to survive 16 steps while deploy exited after 6. Same intent, two
literals, nothing comparing them: **0.40 vs 0.83** success on identical seeds (p=0.00012).

**3.2 Never introduce a second rollout loop.** If you need different stopping behaviour, express it
in `Execution.terminate`, not in a new `for` loop. Writing this rule down is what caused someone to
count the loops in the reference codebase and find *five*, only one of which had received an
important fix.

**3.3 Every checkpoint gets stamped.**
```python
stamp(state_dict, contract)     # at save time
verify(state_dict, contract)    # at load time
```
An unstamped checkpoint is worse than no checkpoint: it will run anyway.

**3.4 Privileged state never reaches the deployed path.** A simulator's `is_grasping` is fine for
*calibrating* a proprioceptive signal and for *scoring* a failure. It must never be what the robot
acts on. `Contract.validate()` enforces the honest consequence: proprioception cannot identify
*which* object is held, so `latch_on="target"` with `signal.kind="proprio"` is rejected.

**3.5 Unknown means retrain.** In `resolve.SEVERITY`, a field with no entry resolves to `RETRAIN`.
When you add a `Contract` field, classify it deliberately. An unrecognised difference is not
evidence of safety.

## 4. The severity table is empirical, not stylistic

`bridle/resolve.py` maps each contract field to `RUN` / `ADAPT` / `RETRAIN`. These are claims about
physics and learning, and the important ones were paid for in failed runs:

- `release.height_above_resting` → **RETRAIN**. The descend reward is an *attractor* at this height;
  moving it changes what the policy optimises. Getting this wrong is the 0/20 stacking failure.
- `execution.hold_steps` → **ADAPT**. The policy is not wrong, the requirement moved.
- `release.centering_tolerance` → **RUN**. A deploy-side gate the policy never observed. Its
  training-side twin `success_tolerance` is RETRAIN.

Do not "tidy" this table. Changing an entry is a claim that a policy will or will not survive a
change — support it with a measurement.

## 5. Patterns you will actually need

**Run a rollout**
```python
from bridle import Contract, Runner, Trace
from bridle.runner import Rollout

contract = Contract.grab()
result = Runner(contract, Trace("grab")).run(Rollout(
    policy=policy_fn,            # () -> action
    step=step_fn,                # (action) -> advance the world
    grasped=grasp_fn,            # () -> bool
    gripper_zero=zero_fn,        # (action) -> action with the gripper dim zeroed
))
result.succeeded, result.steps, result.reason
```

**Ask whether a skill fits a rig**
```python
from bridle import Rig, Store
plan = Store("~/.bridle/apps").plan(app, Rig.so101(cameras=("base",)))
print(plan.explain())            # names the fields that forced the verdict
```

**Build a skill that does not fit**
```python
from bridle import Foundry, ShellStageRunner
job = Foundry({"teacher": ShellStageRunner("teacher", cwd=".", dry_run=True)}) \
        .build(app, rig, plan, target_contract=target)
```
`dry_run=True` prints the exact command and environment without executing. Use it first — a stage
here can be a multi-hour GPU job.

**Give an LLM only the skills that work**
```python
from bridle import Orchestrator, build_tools
tools, unavailable = build_tools(store, rig)      # unavailable = (name, verdict, why)
```

## 6. Adding a Contract field — the checklist

1. Add it to the right sub-record (`Actuation` / `Execution` / `Grasp` / `Release`), not the top level.
2. Add a `SEVERITY` entry in `resolve.py`. Without one it defaults to `RETRAIN`, which is safe but
   probably not what you mean.
3. If training must see it, add it to `foundry.contract_env` — and check whether a *paired* field
   must travel with it. Shipping `success_tolerance` without `centering_tolerance` produces a
   contract `validate()` rejects; that is the guard working, not a bug to route around.
4. Add a test asserting the severity you chose.

## 7. Working in this repository

```bash
python -m pytest bridle/tests          # everything; stdlib-only, no GPU, seconds
PYTHONPATH=. python bridle/tests/test_resolve.py    # any test standalone, exits non-zero on failure
```

Both entry points must keep working. The project venv of the reference consumer has no pytest, and a
test you cannot run without installing something is a test that stops being run.

**Style.** Comments explain *why*, with the measurement behind the decision where one exists. This
codebase's docstrings carry numbers (`0.40 vs 0.83`, `150–240 N`, `0/20`) on purpose: they are the
evidence for choices that otherwise look arbitrary and get "simplified" away later.

**Core stays dependency-free.** No torch, no numpy, no requests in `bridle/*.py`. Backends go in
`bridle/adapters/` behind an optional extra. If core needs a dependency, the design is wrong —
"testable without a simulator" is what makes the contract testable at all.

**Never import an application from the library.** `bridle.adapters.maniskill` takes callables and a
duck-typed session; it imports nothing from any consumer. A library importing its own consumer
cannot be installed on its own.

## 8. Known limits — do not paper over these

- **Training shares the contract's numbers, not its loop.** Training envs are batched and vectorised;
  `Runner` is a scalar loop. One live consequence: a training env may *accumulate* held steps while
  `Runner` requires *consecutive* ones, so "survived N steps" can mean two different things. Same
  number, weaker claim in training.
- **One backend.** ManiSkill. `EnvSpec` is ManiSkill-shaped and the first port will pay for it.
- **Unstamped checkpoints warn rather than fail** (`verify(..., on_missing="warn")`), because every
  checkpoint predating this library is unstamped. Flip to `"error"` once a fleet is stamped.
- **`resolve` compares contracts, not behaviour.** A `RUN` verdict says nothing contradicts; it does
  not promise the skill is good. That is what the eval is for.
