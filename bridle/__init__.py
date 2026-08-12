"""bridle — one rollout loop and one declared contract, so training and deployment cannot disagree.

For grasping robot arms. v0.1 scope: single-arm grasp-and-place on a parallel-jaw gripper.

    from bridle import Contract, Runner, Trace
    contract = Contract.grab()
    Runner(contract, Trace("grab")).run_grasp(policy_fn, step_fn, grasp_fn, gripper_zero_fn)

Core (contract, runner, trace, signals, geometry, calibrate, checkpoint) is stdlib-only: no torch, no
simulator, testable on any machine. Backends live in `bridle.adapters` and are optional extras.

Design: the design notes
Origin:  the design notes
"""
from bridle.checkpoint import ContractMismatch, stamp, verify
from bridle.contract import Actuation, Contract, Grasp, GraspSignal, Release
from bridle.runner import Runner, RunResult
from bridle.trace import Trace

__version__ = "0.1.0"

__all__ = [
    "Actuation", "Contract", "ContractMismatch", "Grasp", "GraspSignal", "Release",
    "RunResult", "Runner", "Trace", "stamp", "verify", "__version__",
]
