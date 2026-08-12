"""bridle — a harness for robot skills. Bring your own LLM, simulator and robot.

Skills carry the contract they were trained under, so running one on a rig it does not fit is a
startup error with a field-level diff, not a silent failure discovered days later.

    from bridle import Contract, Rig, Runner, Store, Trace
    from bridle.runner import Rollout

    rig  = Rig.so101(cameras=("base",))
    plan = Store("~/.bridle/apps").plan(app, rig)     # run / adapt / retrain / blocked

Core (contract, rig, resolve, runner, trace, signals, geometry, calibrate, checkpoint, app, store,
foundry, llm, orchestrator) is stdlib-only: no torch, no simulator, testable on any machine.
Backends live in `bridle.adapters` and are optional extras.

For agents and LLMs writing code against this library: read AGENTS.md — it carries the invariants
that are easy to get wrong and the reasons behind them.
"""
from bridle.agent import AgentSession, Event
from bridle.app import App, Artifact, EnvSpec, EvalSpec, Recipe, Stage
from bridle.checkpoint import ContractMismatch, stamp, verify
from bridle.contract import Actuation, Contract, Execution, Grasp, GraspSignal, Release
from bridle.foundry import Foundry, Job, ShellStageRunner, StageError, StageResult
from bridle.llm import (AnthropicProvider, OpenAICompatProvider, Provider,
                        ScriptedProvider, from_spec)
from bridle.orchestrator import Orchestrator, build_tools
from bridle.resolve import ADAPT, RETRAIN, RUN, Resolution, resolve, resolve_contracts
from bridle.rig import Camera, Gripper, Rig
from bridle.runner import Rollout, Runner, RunResult
from bridle.store import BLOCKED, Plan, Store
from bridle.trace import Trace
from bridle.ui import Viewer

__version__ = "0.1.0"

__all__ = [
    # contract & rig — what a skill assumes
    "Actuation", "Contract", "Execution", "Grasp", "GraspSignal", "Release",
    "Camera", "Gripper", "Rig",
    # does this skill fit this rig?
    "Resolution", "resolve", "resolve_contracts", "RUN", "ADAPT", "RETRAIN", "BLOCKED",
    # skills on disk
    "App", "Artifact", "EnvSpec", "EvalSpec", "Recipe", "Stage", "Store", "Plan",
    # building them
    "Foundry", "Job", "ShellStageRunner", "StageError", "StageResult",
    # running them
    "Rollout", "Runner", "RunResult", "Trace",
    # checkpoints that know what they are
    "ContractMismatch", "stamp", "verify",
    # bring your own LLM
    "Orchestrator", "Provider", "OpenAICompatProvider", "AnthropicProvider",
    "ScriptedProvider", "from_spec", "build_tools",
    # the agentic TUI
    "AgentSession", "Event",
    # the simulator window
    "Viewer",
    "__version__",
]
