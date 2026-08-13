# Rig, Contract, and the four verdicts

[← README](../README.md)

A trained policy assumes things: an embodiment and an action space, a rollout loop that stops at a
particular moment, physical tolerances it was graded against. Those assumptions are normally
implicit, spread across a training script and a deploy path, with nothing comparing them. In bridle
they are one object.

## Rig — what can invalidate a policy

Deliberately not a URDF. It carries only what the robot commands, grips with, and observes.

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

`Rig.so101(cameras=("base", "wrist"))` is a ready-made one.

## Contract — the rig plus the task

How actions are shaped, how the rollout loop runs, what counts as a grasp, where the gripper lets go.

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

A contract has a stable 12-character **fingerprint** — sha256 over its canonical form, so it survives
across processes and machines. Checkpoints are stamped with it, and loading one under a different
contract is a startup error with a field-level diff rather than a silent bad run.

## The four verdicts

Equality would be too blunt: a longer step budget cannot hurt a policy, a different action range
certainly can. So differences resolve **per field**.

| verdict | meaning |
|---|---|
| `run` | every difference is inert for the policy — load the weights |
| `adapt` | the policy is recoverable — re-run the perception stages |
| `retrain` | it was trained for a different problem — regenerate from the recipe |
| `blocked` | your rig cannot run this skill at all (no camera, wrong gripper) |

Severities live in `bridle/resolve.py`: `execution.budget` is `run`, `rig.cameras` is `adapt` (a
vision student memorises the view; the teacher is unaffected), `rig.control_mode` is `retrain`. **A
field with no entry resolves to `retrain`** — an unrecognised difference is not evidence of safety.

This is why a skill here is a **recipe first, weights second**. A recipe is a reproducible training
procedure. Pretrained weights are a fast path guarded by the fingerprint; when it does not match,
there is still something to run.

## Asking whether a skill fits

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

## One rollout loop

There is exactly one. `Execution` describes it as data — a step budget, a gripper rule, an ordered
tuple of termination rules — so every skill runs through the same `Runner`.

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

`Runner` takes plain callables, so it imports no simulator and no tensor library. Binding them to a
live simulator is the adapter's job (`bridle.adapters.maniskill`).

**Never add a second rollout loop.** A project that grew five of them measured a fix reaching one and
silently missing four; that is the origin of this rule.

## Checkpoint stamping

`stamp(state_dict, contract)` at save, `verify(state_dict, contract)` at load — which raises
`ContractMismatch` with a field diff. Unstamped checkpoints warn rather than fail
(`verify(..., on_missing="warn")`) for migration; flip to `"error"` once everything you load is
stamped.
