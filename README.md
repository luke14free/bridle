# bridle

**A harness for robot skills. Bring your own LLM, your own simulator, your own robot.**

A trained policy is full of assumptions — an embodiment, an action range, a rollout loop that stops
at a particular moment, tolerances it was graded against. Normally those live scattered across a
training script and a deploy path, with nothing comparing them. So a skill "runs" on your setup and
quietly underperforms, and you find out in the eval, or on the bench, or never.

bridle's one idea: **make the assumptions data, and check them before you spend a GPU.**

```
your rig ─┐
          ├─▶ does this skill fit?  ──▶ run │ adapt │ retrain │ blocked
a skill  ─┘                                              │
                                                         ▼
skill.yaml ──▶ compile ──▶ check ──▶ preflight ──▶ train ──▶ stamped checkpoint
             (no GPU)    (no GPU)    (30 s)
```

Everything left of `train` runs on a laptop with no simulator and no GPU. That is the point: the
expensive step is the last one you take, not the first.

---

## 1 · Declare your setup

A `Rig` is deliberately not a URDF. It carries only what can invalidate a policy — what the robot
commands, grips with, and observes.

```python
from bridle import Rig, Gripper, Camera

rig = Rig(name="bench-arm", embodiment="so101", dof=5,
          control_mode="pd_joint_delta_pos", control_hz=20.0,
          gripper=Gripper(kind="parallel_jaw", dim=5, stroke_m=0.035),
          cameras=(Camera(name="base", width=128, height=128),),
          sensors=("proprio", "rgb"))

rig.fingerprint()          # '69fd2ade8be6' — sha256, stable across machines
```

A **`Contract`** adds the task on top of the rig, and every skill carries the one it was trained
under. Comparing two contracts answers *does this skill work on my robot?* mechanically — per field,
because a longer step budget cannot hurt a policy and a different action range certainly can.

→ **[docs/rig-and-contracts.md](docs/rig-and-contracts.md)** — contracts, the four verdicts, field
severities, checkpoint stamping, and the single rollout loop.

## 2 · Declare a skill

A skill is a YAML document: a scene, a reward, and a criterion for success. No Python.

```yaml
name: descend_to_target
kind: carry                       # a chassis — supplies weight defaults and their rationale
env_id: SO100DescendToTarget-v1

reward:
  - PredicateBonus  {weight: 1.0, predicate: grasped,
                     why: "hold on — never drop the cube; release is a separate skill"}
  - DistancePull    {weight: 2.5, measure: height_above_seat_live, k: 6.0,
                     setpoint: params.hover, gate: grasped,
                     why: "peaks at hover, NOT at contact — peaking at contact broke 16/16 grasps"}
  - SuccessBonus    {value: 12.0, mode: replace, scope: preceding, why: "..."}

success: all[grasped, below_resting_height(band=params.low_band), ...]
```

Nine reward terms cover every row of a 15-primitive corpus. A shape they cannot express falls back to
a safe arithmetic expression; a genuine one-off falls back to Python and is fingerprinted as opaque.
`why:` is **mandatory on every row** — the rationale is the thing YAML usually destroys.

→ **[docs/skill-yaml.md](docs/skill-yaml.md)** — the document grammar, all nine terms, measures with
sign and frame, predicates, and the six chassis.

## 3 · Check it before the GPU

```console
$ bridle skill check my_skill.yaml
reward[2].measure: unknown measure 'height_above_seet' — did you mean 'height_above_seat_live'?
```

The intended author of a skill document is a **local 27–30B model**, so the error messages *are* the
API: every refusal names the dotted path, states the legal set, and suggests the nearest match. The
compiler also refuses combinations with a recorded failure mode — shaping that out-earns completion,
an attractor that peaks at the contact surface — and prints the resolved plan with every default it
supplied on your behalf, so nothing you did not write is invisible.

```console
$ bridle skill vocab            # the whole authorable surface — paste into your model's prompt
$ bridle skill compile f.yaml   # every row in fold order + the plan fingerprint
```

Neither starts a simulator.

## 4 · Train it

The compiled plan becomes the reward function of a real environment. Nothing else about the
environment changes, so a run is comparable to the lineage it reproduces.

```python
from bridle.skill.spec    import parse_spec
from bridle.skill.compile import compile_spec
from bridle.adapters.skill_env import build_reward_fn

plan = compile_spec(parse_spec(doc), horizon=64)
plan.fingerprint()                        # 'plan@95babe2a3cc5' — moves if any weight moves

class MySkillEnv(TheRegisteredEnv):       # scene, init and termination untouched
    compute_dense_reward = build_reward_fn(plan)
```

Then PPO, exactly as before. The plan fingerprint goes into the run name, so *which reward document
trained this checkpoint?* is answerable from your dashboard instead of from memory.

→ **[docs/training.md](docs/training.md)** — the derived env, numerical-equivalence verification,
per-term diagnostics, `Foundry` stage planning, and checkpoint stamping.

## 5 · Hand it to an LLM

An LLM offered a tool will call it. So the tool list contains **only** the skills that resolve to
`run` on your rig; everything else comes back separately with its verdict and the reason.

```python
tools, unavailable = build_tools(store, rig)
# unavailable -> [('sphere_grab', 'blocked', "camera 'wrist' (rig has ['base'])")]

Orchestrator(from_spec("local:qwen3-32b"), store, rig, executor).run("stack the red cube")
```

`bridle tui` gives you a terminal to talk to the agent — typing while it runs queues guidance it sees
before its next skill, ESC stops it cleanly *after* the current skill returns, never mid-grasp — plus
a browser view with live verdicts, running jobs and simulator frames.

→ **[docs/agents-and-tui.md](docs/agents-and-tui.md)** — providers, the orchestrator, the steerable
session, the TUI and the viewer.

---

## Install

```bash
pip install -e .                 # core: stdlib only, Python >= 3.11
pip install -e '.[maniskill]'    # + torch, for the ManiSkill adapter
```

The core — contracts, rig, runner, trace, store, foundry, the LLM loop, the TUI, the whole `skill`
compiler — has **no dependencies at all** and imports on a machine with no GPU and no simulator.

## Tests

```bash
PYTHONPATH=. python3 bridle/tests/test_resolve.py   # any module standalone, non-zero on failure
python -m pytest bridle/tests                       # everything
```

21 test modules, no simulator, no GPU, a few seconds.

## Status

v0.1, Apache 2.0. Honest about what it is not: one simulator backend (ManiSkill), you supply the
execution, no skills ship with the library, parallel-jaw grippers only, and a `run` verdict says
nothing contradicts — not that the skill is good. That is what the eval is for.

→ **[docs/status.md](docs/status.md)** — what is built, every known limit, and what is planned.

`AGENTS.md` is the working guide to the library's invariants: never copy a number out of a contract,
never add a second rollout loop, stamp everything, classify every new field's severity.
