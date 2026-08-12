"""The single rollout loop. There must be exactly one of these.

On 2026-08-11 there were TWO — grab_env.step() for training and playground_coord.run_prim() for
deploy — and they disagreed about when a grasp counts. That disagreement was worth 0.43 of task
success and no benchmark could see it, because benchmarks run inside the training contract. Writing
down "never introduce a second rollout loop" then made someone COUNT them, and there were five.

REBUILT 2026-08-12 to be able to run all of them. The first cut only knew how to latch-and-hold a
grasp, which meant every other rung — carry prims, the reach handoff, the grab ladder, the compact
grasp — had to keep its own loop, and a library that only the easy case can adopt is a library that
changes nothing. Runner now executes an `Execution`: a budget, a gripper rule, and an ORDERED tuple
of termination rules, which between them describe every rollout this project runs.

Runner takes plain callables so it imports no sim and no torch: the adapter supplies them.
"""
from dataclasses import dataclass, field
from typing import Callable, Optional

from bridle.contract import Contract
from bridle.trace import Trace


@dataclass
class RunResult:
    latched: bool
    held_steps: int
    steps: int
    succeeded: bool
    #: Which termination rule ended the rollout, or None if the budget ran out. A rollout that ended
    #: for a reason nobody can name is how "descend stops descending" went undiagnosed for two days.
    reason: Optional[str] = None


@dataclass
class Rollout:
    """The app's side of one rollout: everything Runner needs that is not a contract decision.

    Bundled rather than passed as ten keyword arguments — the loops being converted need different
    subsets, and a signature that grows per caller is how the old five loops drifted apart.

    policy()            -> action
    step(action)        -> advance the world one control step
    grasped()           -> bool, for latching and "sustained_grasp"
    at_goal()           -> bool, for "on_goal" (the app owns the distance test; it needs the goal)
    force()             -> float, for "on_force"
    settled()           -> bool, for "sustained_settled"
    gripper_zero(action)-> action with the gripper dim zeroed
    on_latch()          -> side effect at the first latch (e.g. freezing the real gripper target)
    before_step(k, action, latched) -> app-side per-step hook (abort checks, tracing) BEFORE the step
    after_step(k)       -> app-side per-step hook (frame capture) AFTER the step
    observe()           -> dict of extra readings merged into the trace row
    """

    policy: Callable
    step: Callable
    grasped: Optional[Callable] = None
    at_goal: Optional[Callable] = None
    force: Optional[Callable] = None
    settled: Optional[Callable] = None
    gripper_zero: Optional[Callable] = None
    on_latch: Optional[Callable] = None
    before_step: Optional[Callable] = None
    after_step: Optional[Callable] = None
    observe: Optional[Callable] = None


class Runner:
    def __init__(self, contract: Contract, trace: Trace | None = None):
        contract.validate()
        self.contract = contract
        self.trace = trace

    def run(self, rollout: Rollout) -> RunResult:
        """Execute one rollout under the contract. The ONLY place a step is taken.

        RULE ORDER IS LOAD-BEARING and reproduces the legacy loop exactly: per step, evaluate the
        termination rules in `execution.terminate` order, and latch AFTER them. That ordering means
        on the step where a grasp first latches, the linger counter does not advance — an off-by-one
        that would otherwise show up as a rollout ending one step early, which is precisely the size
        of the bug this library exists to prevent.
        """
        c, e = self.contract, self.contract.execution
        r = rollout
        latched, held, k, reason = False, 0, 0, None

        for k in range(e.budget):
            # ── rules evaluated at the TOP of the step, before the policy is even queried ──────
            # The DINO rung increments its linger counter here rather than after the step, so with
            # the same linger_steps it runs one FEWER step post-latch than run_prim does. Modelled
            # rather than normalised: see Execution.terminate_pre_step.
            for rule in e.terminate_pre_step:
                if rule == "linger_after_latch" and latched:
                    held += 1
                    if held >= e.linger_steps:
                        return RunResult(True, held, k, True, rule)
                elif rule == "on_goal" and r.at_goal is not None and r.at_goal():
                    return RunResult(latched, held, k, True, rule)

            action = r.policy()
            if e.gripper == "zero_always" or (e.gripper == "zero_after_latch" and latched):
                action = r.gripper_zero(action)

            if r.before_step is not None:
                r.before_step(k, action, latched)
            r.step(action)
            if r.after_step is not None:
                r.after_step(k)

            grasped = bool(r.grasped()) if r.grasped is not None else False
            settled = bool(r.settled()) if r.settled is not None else False

            # ── rules evaluated BEFORE this step's latch ──────────────────────────────────────
            # `linger_after_latch` counts steps that have elapsed SINCE a previous step latched, so
            # it must see the pre-latch value: on the very step a grasp first latches, the counter
            # does not advance. `on_goal` likewise fires on arrival regardless of grasp state. This
            # is the legacy loop's order and reproducing it is worth an off-by-one — which is
            # exactly the size of bug this library exists to prevent.
            for rule in e.terminate:
                if rule == "linger_after_latch" and latched:
                    held += 1
                    if held >= e.linger_steps:
                        return RunResult(True, held, k + 1, True, rule)
                elif rule == "on_goal" and r.at_goal is not None and r.at_goal():
                    return RunResult(latched, held, k + 1, True, rule)

            if c.grasp is not None and c.grasp.latch_on != "none" and not latched and grasped:
                latched = True
                if r.on_latch is not None:
                    r.on_latch()

            # ── rules evaluated AFTER the latch ───────────────────────────────────────────────
            # `held` is the run of consecutive satisfied steps for whichever sustained rule is in
            # force. Only a SUSTAINED state counts: grab_env's "a fingertip/loose grip drifts open
            # and fails" is the same statement as "a cube resting for one frame is not placed".
            if "sustained_grasp" in e.terminate:
                held = (held + 1 if grasped else 0) if latched else 0
            elif "sustained_settled" in e.terminate:
                held = held + 1 if settled else 0

            if self.trace is not None:
                extra = r.observe() if r.observe is not None else {}
                self.trace.record(k + 1, latched=latched, grasped=grasped, held=held, **extra)

            for rule in e.terminate:
                if rule == "sustained_grasp" and latched and held >= e.hold_steps:
                    return RunResult(True, held, k + 1, True, rule)
                if rule == "sustained_settled" and held >= e.hold_steps:
                    return RunResult(latched, held, k + 1, True, rule)
                if rule == "on_force" and r.force is not None and float(r.force()) > e.force_threshold:
                    return RunResult(latched, held, k + 1, True, rule)

        return RunResult(latched, held, k + 1, False, reason)

    # ── named phases: the same loop, read at the call site ────────────────────────────────────
    def run_grasp(self, policy_fn, step_fn, grasp_fn, gripper_zero_fn, observe_fn=None) -> RunResult:
        """Close on the object and hold it, per the contract's grasp phase."""
        if self.contract.grasp is None:
            raise ValueError(f"{self.contract.describe()} declares no grasp phase")
        return self.run(Rollout(policy=policy_fn, step=step_fn, grasped=grasp_fn,
                                gripper_zero=gripper_zero_fn, observe=observe_fn))

    def run_release(self, policy_fn, step_fn, settled_fn=None, observe_fn=None) -> RunResult:
        """Carry and lower onto the destination, per the contract's release phase."""
        if self.contract.release is None:
            raise ValueError(f"{self.contract.describe()} declares no release phase")
        return self.run(Rollout(policy=policy_fn, step=step_fn, settled=settled_fn,
                                gripper_zero=lambda a: a, observe=observe_fn))
