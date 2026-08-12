# bridle

**One rollout loop, one declared contract — so training and deployment cannot disagree about the
rules.**

For grasping robot arms. `v0.1` scope: single-arm grasp-and-place on a parallel-jaw gripper.

---

## The problem it solves

Train a policy in a loop that requires a grasp to survive 16 steps. Deploy it in a loop that quits
after 6. Both loops are correct on their own; nothing anywhere compares them; every benchmark passes,
because benchmarks run **inside the training loop**. On real seeds that gap was worth
**0.40 vs 0.83** success (p=0.00012).

The same class of bug then reappeared on the place leg. The physical rule "where does the gripper let
go?" was spelled out in six literals across three files — and two of them already disagreed:

| quantity | in the training env | in the deploy macro | what physics needs |
|---|---|---|---|
| release height above resting | `_HOVER = 0.015` | *(implicit)* | ~0.002 |
| centering tolerance | `_CENTER_TOL = 0.045` | release gate `0.035` | **~0.012** |
| destination top | `platform_top + half` | hardcoded `+0.014` | measured half |

The policy hovered 1.5 cm above resting, exactly as trained. Deploy released it there — onto a 2.4 cm
cube. The cube bounced off. Stacking scored **0/20** for two days while the search looked for a bug
in the policy. There wasn't one: it was executing a contract it had never been trained under.

bridle makes that arrangement impossible to express.

## How

```python
from bridle import Contract, Runner, Trace
from bridle.runner import Rollout

contract = Contract.grab()               # the single definition of the numbers

# TRAINING and DEPLOY both read the same object — neither keeps its own literal.
hold = contract.execution.hold_steps     # not a copy of 16

result = Runner(contract, Trace("grab")).run(Rollout(
    policy=policy_fn, step=step_fn, grasped=grasp_fn, gripper_zero=zero_fn))
```

**`Contract`** is frozen, validated and *fingerprinted*:

- `Actuation` — action shaping (`gripper_dim`, clamp)
- `Execution` — **how the loop runs**: budget, a gripper rule, and an ORDERED tuple of termination
  rules
- `Grasp` — what a grasp *is*: the latch rule and the sensor
- `Release` — placement geometry: release height, centering tolerance, destination-top rule, ramp

`Execution` is the load-bearing piece. Three gripper rules and five termination rules turned out to
describe **every** rollout in a real robotics codebase:

| gripper | | terminate | |
|---|---|---|---|
| `free` | policy owns the gripper | `sustained_grasp` | grasp survived N steps |
| `zero_always` | carry: never re-open | `linger_after_latch` | N steps after the first latch |
| `zero_after_latch` | grab | `on_goal` | TCP within tolerance of the commanded point |
| | | `on_force` | finger force over threshold |
| | | `sustained_settled` | centred at release height for N steps |

Rule ORDER is part of the contract, and `terminate_pre_step` vs `terminate` says whether a rule is
evaluated before the policy or after the step. That is not gratuitous: two rungs of the same
codebase implemented "linger N steps after latch" with an off-by-one between them, and modelling
both timings is what let each convert as a *provable* no-op instead of a silent behaviour change.

**`Runner`** is the only place a step is taken. `Runner.run(Rollout(...))` executes the contract;
the app supplies callbacks (`policy`, `step`, `grasped`, `at_goal`, `force`, `gripper_zero`,
`on_latch`, `before_step`, `after_step`, `observe`). Adding a phase must never mean adding a loop.

**Checkpoints carry their contract.** This is the payload:

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

A silent 0/20 discovered by trace three days later becomes a startup error naming the field.

## Zero privilege

The only grasp signal bridle ships is **proprioceptive** — fingertip contact force and jaw position,
both of which a real arm has. The simulator's `is_grasping` is object-aware and convenient and is
*privileged state*; standardising it would bake a sim-only oracle into every contract written against
this library.

Two gates are required, because each fails alone and in opposite directions: an **open** jaw pressing
the table reads 150–240 N, and a **closed empty** gripper looks exactly like a closed loaded one.

The honest consequence: proprioception cannot say *which* object is held, so `latch_on="target"` is
unimplementable and `validate()` rejects it. Thresholds are **fitted** from recorded traces
(`bridle.calibrate`), never guessed — privileged state used at calibration time to replace itself,
which is the only use of it the rule allows.

## Install

```bash
pip install -e .            # from the repo root; core has no dependencies
pip install -e '.[maniskill]'   # adds the torch-backed adapter
```

## Testing

Core is stdlib-only and needs no GPU:

```bash
python -m pytest bridle/tests -k "not parity"
```

Each test also runs standalone and exits non-zero on failure, so it works without pytest installed:

```bash
PYTHONPATH=. python bridle/tests/test_contract.py
```

The parity tests need the simulator. They are the ones that matter most: they prove `Runner`
reproduces the legacy deploy loop *step for step* on fixed seeds, which is what makes adopting the
library a provable no-op rather than a leap.

## Status

`v0.1`, alpha, API expected to move. Honest limits today:

- Grasp-and-place on a parallel-jaw gripper only. No chains, no other embodiments, no real hardware.
- One backend adapter (ManiSkill).
- Unstamped checkpoints warn rather than fail, because every checkpoint predating this release is
  unstamped. That default flips once the fleet is stamped.
- **Training runs the contract's numbers, not yet the contract's loop.** Training envs are batched
  and vectorised; `Runner` is a scalar loop. The constants are shared; the per-step *semantics* are
  still implemented twice. One live consequence is pinned in `tests/test_train_deploy_hold.py`:
  training ACCUMULATES held steps while `Runner` requires CONSECUTIVE ones, so a grip that slips and
  re-grabs passes in training and fails at deploy. Same number, weaker claim. Fixing it changes what
  training scores as success and needs a retrain.

## What it actually caught

Not hypotheticals — defects found by adopting it, in one codebase, in two days:

- Training required a grasp to survive 16 steps; deploy exited after 6. **0.40 vs 0.83** (p=0.00012).
- The descend policy hovered 1.5cm above resting exactly as trained; deploy released it there onto a
  2.4cm cube. **Stacking 0/20** for two days.
- Two grab rungs implementing the same "linger" rule with an off-by-one between them.
- The release tolerance existed as *three* different numbers (0.045 training, 0.035 deploy, ~0.012
  physics) for one physical quantity.
- Training and deploy disagreeing on what "survived N steps" means (accumulating vs consecutive).

Every one was invisible to benchmarks, because benchmarks run inside the training contract.

Licensed under Apache-2.0.

## A note on the tests you don't see here

bridle was extracted from a working robot codebase, and the tests that prove it *there* — step-for-step
parity against the loops it replaced, a guard that fails if a rollout loop reappears outside `Runner`,
and a pin on a train/deploy semantic divergence — depend on that codebase's simulator and live
service. They are that project's **adoption** suite, not the library's, so they stay with it.

What ships here is what runs anywhere: the contract, the loop, and the arithmetic, with no
dependencies and no GPU.
