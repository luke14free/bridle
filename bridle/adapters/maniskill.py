"""Bind bridle's plain callables to a live ManiSkill-backed chain session.

Keeps every sim/torch import out of bridle core, so Contract/Runner/Trace stay testable with no GPU.
`replay_grab_*` exist for the parity test only — they are not the deploy entry point.

DEPENDENCY DIRECTION (inverted 2026-08-12). This module used to `import composer.llm.playground_coord`
and call its private `_grasping_target` / `_grasping_any` — the library importing its own consumer,
which makes bridle uninstallable on its own and means a change in the app can break the library.
Now the CALLER supplies the grasp predicates and bridle picks between them by `contract.latch_on`:
bridle decides the POLICY (which rule is in force), the app supplies the MECHANISM (how to evaluate
it). The session object is still duck-typed — see `SESSION_PROTOCOL` — but it is only ever *passed
in*, never imported.

Note this also deleted the lazy-import dance the old module needed: `playground_coord.GRAB_REFRESH_R`
is read from the environment at ITS import time, so a module-level `import ... as PC` here froze that
value to whatever the environment held when this adapter first loaded. With no import at all, the
hazard is gone rather than worked around.
"""
import torch

#: What `make_callables` requires of the `session` object it is handed. Duck-typed on purpose —
#: bridle must not import the app that defines it. Any object with these members works.
SESSION_PROTOCOL = (
    "step(action)",              # advance the sim one control step
    "_clamp_ctrl_target()",      # re-clamp the controller target after the step
    "latch_grip()",              # freeze the gripper target at its current jaw qpos
    "act_lo", "act_hi",          # per-dim action bounds (tensors)
    "finger_contact_force()",    # scalar N — proprioceptive, exists on real hardware
    "base.agent.robot.get_qpos()", "ai",   # jaw joint position (last dim) for this env slot
)


def read_proprio(session) -> dict:
    """The two readings a real SO-101 actually has: fingertip load and jaw position.

    Deliberately NOT `is_grasping`. Returned as a plain dict so it can be (a) merged into a Trace
    row for calibration and (b) fed to `bridle.signals.grasped_from_proprio` at run time — the same
    numbers serving both, which is what stops the calibrated signal and the deployed signal from
    drifting apart (the entire bug class this library exists for).
    """
    return {"force_n": float(session.finger_contact_force()),
            "jaw_pos": float(session.base.agent.robot.get_qpos()[session.ai][-1])}


def make_proprio_grasp_fn(session, contract):
    """A grasp predicate reading ONLY proprioception, per the contract's fitted thresholds."""
    from bridle.signals import grasped_from_proprio

    signal = contract.grasp.signal

    def fn():
        p = read_proprio(session)
        return grasped_from_proprio(p["force_n"], p["jaw_pos"], signal)

    return fn


def make_callables(session, agent, contract, obs_fn, grasp_predicates, stochastic=False):
    """Return (policy_fn, step_fn, grasp_fn, gripper_zero_fn) for one rollout.

    `grasp_predicates` maps a latch rule name to a zero-arg callable returning bool, e.g.
    `{"target": lambda: _grasping_target(s, cube)}`. Only the rule named by `contract.latch_on`
    is ever called; supplying the others is optional. A missing rule is a KeyError at BUILD time,
    not a silent no-grasp at step 40 of a rollout.

    `obs_fn` is the CALLER's observation closure — built ONCE by the caller (e.g. macro_pick's
    `obs = refreshing_grab_obs(s, things, target)`, threaded through run_prim's own `obs_fn`
    parameter) and used here as-is, never rebuilt. An earlier draft of this adapter ignored the
    parameter and rebuilt its own from the perception cache (`composer.llm.playground.S["things"]`)
    instead — that broke two ways at once: (1) `reperceive`/`detect_query` re-cache that global with
    brand-new dict objects on every call, and `CoordObsBuilder.build_grab` excludes the target by
    OBJECT IDENTITY (`c is not target_cube`), so a freshly rebuilt `things` list makes the target's
    OWN duplicate look like clutter and shifts every slot — the documented "skew #10" discontinuity
    that once collapsed grab to 0/84 (see the comment above `run_prim`'s call site in
    playground_coord.py); (2) even ignoring identity, rebuilding the closure on every call discards
    `refreshing_grab_obs`'s internal live-refresh state every step instead of advancing it, silently
    reverting to the frozen-coordinate behaviour that closure's own docstring measured at 0.100
    success (2026-08-10). Using the caller's closure, built once, sidesteps both.

    `stochastic` mirrors `run_prim`'s own flag: a deterministic policy in an unchanged state
    replays the identical failure, so retries (`tries > 0`) must sample.
    """
    s = session
    latched_once = False   # local edge-detector, NOT Runner's `latched` — see grasp_fn below

    if contract.grasp is None or contract.grasp.latch_on == "none":
        predicate = None
    else:
        try:
            predicate = grasp_predicates[contract.grasp.latch_on]
        except (KeyError, TypeError):
            raise KeyError(
                f"contract.grasp.latch_on={contract.grasp.latch_on!r} but grasp_predicates supplies "
                f"{sorted(grasp_predicates) if grasp_predicates else grasp_predicates!r}. "
                "The caller must provide a predicate for the rule the contract declares."
            ) from None

    def policy_fn():
        return torch.clamp(agent.get_action(obs_fn(), deterministic=not stochastic), s.act_lo, s.act_hi)

    def step_fn(action):
        s.step(action)
        s._clamp_ctrl_target()

    def grasp_fn():
        nonlocal latched_once
        grasped = bool(predicate()) if predicate is not None else False
        if grasped and not latched_once:
            # Freeze the gripper TARGET at its current jaw qpos the instant the grasp first
            # latches — same transition, same call, as legacy's `latched = True; s.latch_grip()`
            # in run_prim's non-bridle loop. Without it the accumulated pd_joint_target_delta_pos
            # command from the raw policy's still-closing action persists into the carry prims,
            # risking the ~110N crush-wedge `drop_in_place` documents. Runner is sim-free by
            # design and has no notion of "latch the real gripper", so this has to happen here —
            # gated on the grasped-but-not-yet-latched edge so it fires exactly once per rollout.
            s.latch_grip()
            latched_once = True
        return grasped

    def gripper_zero_fn(action):
        a = action.clone()
        a[..., contract.actuation.gripper_dim] = 0.0
        return a

    return policy_fn, step_fn, grasp_fn, gripper_zero_fn
