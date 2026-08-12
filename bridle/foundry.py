"""Execute a recipe on a rig, and stamp what comes out. No sim, no torch.

`Store.plan()` says WHAT must happen; the Foundry makes it happen:

    plan = store.plan(app, rig)          # RUN / ADAPT / RETRAIN / BLOCKED
    job  = Foundry(runners).build(app, rig, plan)

WHY THIS IS THE KEYSTONE. Everything before it can only *describe* a mismatch. This is where a skill
whose contract does not match your rig becomes a skill that does — the first point at which the
product's promise ("skills that work on YOUR robot") is kept rather than checked.

TWO RULES, both learned the hard way in the codebase this came from:

1. **The target contract is passed to training, never re-typed into it.** The 0/20 stack failure was
   a training env whose release height was a literal and a deploy path whose release height was a
   different literal. A stage receives `contract_env` derived from the contract itself; nothing
   downstream is trusted to remember a number.

2. **Whatever comes out is stamped with the contract it was built under.** An unstamped checkpoint is
   a checkpoint that will one day be run under the wrong contract, confidently.

bridle does NOT own the taxonomy of how policies get trained. A recipe NAMES stages; a `StageRunner`
executes one; the implementations belong to whoever authored the skill. An unknown stage kind is an
error at PLAN time, not at hour three of a GPU job.
"""
import os
from dataclasses import dataclass, field


class StageError(RuntimeError):
    pass


@dataclass
class StageResult:
    kind: str
    ok: bool
    detail: str = ""
    outputs: dict = field(default_factory=dict)


@dataclass
class Job:
    """A planned build: the stages to run, and the contract everything is built against."""

    app: str
    action: str
    contract: object
    stages: tuple = ()
    results: list = field(default_factory=list)
    checkpoint: str = None

    @property
    def ok(self) -> bool:
        return bool(self.results) and all(r.ok for r in self.results)

    def explain(self) -> str:
        head = f"{self.action.upper()} {self.app} under {self.contract.describe()}"
        return head + "\n" + "\n".join(
            f"    [{'ok ' if r.ok else 'FAIL'}] {r.kind}: {r.detail}" for r in self.results)


def contract_env(contract) -> dict:
    """Environment variables that make a training process build THIS contract.

    The bridge between a declared contract and a training script that reads env vars. Kept explicit
    and small: every entry is a number training would otherwise have hard-coded, which is exactly
    the class of value that drifted and cost 0.43 of task success and a 0/20.

    Names are the app's own convention, supplied by the recipe; bridle only guarantees that the
    VALUES come from the contract rather than from someone's memory.
    """
    fp = contract.fingerprint()
    out = {"BRIDLE_CONTRACT_FINGERPRINT": fp,
           "BRIDLE_CONTRACT_NAME": contract.name or "",
           # EXPERIMENT TRACKING. Every build is a training run somebody will want to compare
           # against another one months later, and a run you cannot find is a run you will repeat.
           # The run is NAMED BY ITS CONTRACT FINGERPRINT, so the question "what was this policy
           # actually trained for?" is answerable from the dashboard alone rather than from whoever
           # remembers launching it.
           "BRIDLE_WANDB": os.environ.get("BRIDLE_WANDB", "1"),
           "WANDB_PROJECT": os.environ.get("WANDB_PROJECT", "bridle"),
           "WANDB_RUN_GROUP": contract.name or "unnamed",
           "WANDB_NAME": f"{contract.name or 'run'}@{fp}",
           "WANDB_TAGS": f"bridle,{contract.name or 'unnamed'},{fp}"}
    r = getattr(contract, "release", None)
    if r is not None:
        out.update({
            "PRIM_DESCEND_HOVER": repr(r.height_above_resting),
            "PRIM_DESCEND_LOW_BAND": repr(r.success_height_band),
            "PRIM_DESCEND_CENTER_TOL": repr(r.success_tolerance),
            # The deploy-side gate travels WITH the training tolerance. They are the same physical
            # quantity, and shipping one without the other produces an incoherent contract —
            # Contract.validate() rejects it, which is how this omission was caught before three
            # GPU-hours rather than after.
            "PRIM_DESCEND_RELEASE_TOL": repr(r.centering_tolerance),
        })
    g = getattr(contract, "grasp", None)
    if g is not None:
        out["PRIM_GRAB_HOLD_K"] = str(contract.execution.hold_steps)
    return out


class StageRunner:
    """Executes one stage kind. Subclass, or pass any callable(stage, ctx) -> StageResult."""

    kind = ""

    def __call__(self, stage, ctx) -> StageResult:
        raise NotImplementedError


class ShellStageRunner(StageRunner):
    """Runs a stage as a shell script, with the contract injected as environment.

    `dry_run` returns the exact command and environment WITHOUT executing — because a stage here can
    be a multi-hour GPU job, and "show me what you would do" has to be free.
    """

    def __init__(self, kind, cwd=".", dry_run=False, launcher=None):
        self.kind, self.cwd, self.dry_run = kind, cwd, dry_run
        self.launcher = launcher            # callable(cmd, env, cwd) -> str, for systemd/nohup/etc.

    def __call__(self, stage, ctx) -> StageResult:
        script = stage.params.get("script")
        if not script:
            return StageResult(self.kind, False, "stage declares no `script` to run")
        env = {**os.environ, **ctx["contract_env"]}
        cmd = stage.params.get("cmd") or f"bash {script}"
        if self.dry_run:
            shown = " ".join(f"{k}={v}" for k, v in sorted(ctx["contract_env"].items()))
            return StageResult(self.kind, True, f"DRY RUN: {shown} {cmd}",
                               {"cmd": cmd, "env": ctx["contract_env"]})
        if self.launcher is None:
            return StageResult(self.kind, False,
                               "no launcher configured; refusing to block on a multi-hour job")
        handle = self.launcher(cmd, env, self.cwd)
        return StageResult(self.kind, True, f"launched: {handle}", {"handle": handle, "cmd": cmd})


class Foundry:
    """Builds apps against a rig by executing their recipe stages."""

    def __init__(self, runners=None):
        self.runners = dict(runners or {})

    def register(self, kind, runner):
        self.runners[kind] = runner

    def plan_stages(self, app, plan):
        """The Stage objects this plan needs, in order. Raises if any kind has no runner.

        Failing here — before anything is launched — is deliberate: discovering an unrunnable stage
        after three GPU-hours is a failure mode that pays for itself to avoid.
        """
        by_kind = {s.kind: s for s in (app.recipe.stages if app.recipe else ())}
        missing_from_recipe = [k for k in plan.stages if k not in by_kind]
        if missing_from_recipe:
            raise StageError(f"plan wants stages {missing_from_recipe} that the recipe does not define")
        no_runner = [k for k in plan.stages if k not in self.runners]
        if no_runner:
            raise StageError(
                f"no runner registered for stage kind(s) {no_runner}. Register one, or the build "
                f"would fail partway through. Known kinds: {sorted(self.runners)}")
        return tuple(by_kind[k] for k in plan.stages)

    def build(self, app, rig, plan, target_contract=None) -> Job:
        """Execute `plan` for `app` on `rig`. Returns a Job; does not raise on stage failure."""
        from bridle.store import BLOCKED
        from bridle.resolve import RUN

        if plan.action == BLOCKED:
            raise StageError(f"{app.name} is BLOCKED on this rig: {list(plan.blockers)}. "
                             "This is not a training problem — the rig cannot run the skill.")
        contract = target_contract if target_contract is not None else app.contract_for(rig)
        job = Job(app=app.name, action=plan.action, contract=contract,
                  checkpoint=plan.checkpoint)
        if plan.action == RUN:
            job.results.append(StageResult("resolve", True, "contract matches; nothing to build"))
            return job

        stages = self.plan_stages(app, plan)
        ctx = {"app": app, "rig": rig, "contract": contract,
               "contract_env": contract_env(contract), "plan": plan}
        for st in stages:
            runner = self.runners[st.kind]
            try:
                res = runner(st, ctx)
            except Exception as e:                       # a stage must not take the process with it
                res = StageResult(st.kind, False, f"{type(e).__name__}: {e}")
            job.results.append(res)
            if not res.ok:
                break                                    # later stages consume earlier outputs
        return job

    @staticmethod
    def stamp_result(state_dict, contract):
        """Stamp a produced checkpoint with the contract it was built under.

        Thin wrapper over bridle.checkpoint.stamp, here so the Foundry's contract with its callers
        is complete: what goes in is a recipe, what comes out is a checkpoint that knows what it is.
        """
        from bridle.checkpoint import stamp
        return stamp(state_dict, contract)
