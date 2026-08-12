"""The local app store: apps on disk as YAML, and the plan that turns one into something runnable.

    store = Store("~/.bridle/apps")
    for app in store.apps():
        print(app.name, store.plan(app, my_rig).action)

`plan(app, rig)` is the user-facing verb of the whole product:

    RUN      -> here is the checkpoint to load
    ADAPT    -> here are the stages to re-run, and why
    RETRAIN  -> here is the full recipe to execute
    BLOCKED  -> your rig cannot run this skill at all (no camera, wrong gripper)

BLOCKED is separate from RETRAIN on purpose. "You need three GPU-hours" and "your robot physically
cannot do this" are different sentences, and collapsing them would send someone to train a vision
policy on a rig with no camera.

YAML rather than JSON because humans author these, and a skill nobody can read is a skill nobody will
trust. PyYAML is an optional dependency: the store degrades to JSON if it is missing, so bridle core
stays dependency-free.
"""
import json
import os
from dataclasses import asdict, dataclass, field, is_dataclass

from bridle.app import App, Artifact, EnvSpec, EvalSpec, Recipe, Stage
from bridle.contract import Actuation, Contract, Execution, Grasp, GraspSignal, Release
from bridle.resolve import ADAPT, RETRAIN, RUN, resolve_contracts
from bridle.rig import Camera, Gripper, Rig

BLOCKED = "blocked"

#: Which stages a given severity requires re-running. The claim: a rig-level change (a moved camera)
#: invalidates the perception half but not the teacher, so re-distilling is enough — whereas a task
#: change (a different release height) invalidates what the teacher itself was optimising.
ADAPT_STAGES = ("distill", "student")


@dataclass
class Plan:
    """What to do with this app on this rig, and the evidence for it."""

    action: str
    app: str
    reason: str = ""
    checkpoint: str = None          # RUN: the artifact to load
    stages: tuple = ()              # ADAPT/RETRAIN: what to execute
    resolution: object = None       # the underlying Resolution, for the field-level diff
    blockers: tuple = ()

    def explain(self) -> str:
        head = f"{self.action.upper():8s} {self.app}"
        if self.action == BLOCKED:
            return head + "\n" + "\n".join(f"    missing: {b}" for b in self.blockers)
        if self.action == RUN:
            return head + f"\n    checkpoint: {self.checkpoint}"
        body = f"\n    stages: {', '.join(self.stages)}"
        if self.resolution is not None:
            body += "\n    " + self.resolution.explain().replace("\n", "\n    ")
        return head + body


# ── (de)serialisation ─────────────────────────────────────────────────────────────────────────
# Explicit rather than reflective. A generic dataclass walker would silently accept an unknown key
# and drop it, and a store that quietly ignores a field is a store that quietly ships the wrong
# skill. Every constructor below fails loudly on an unexpected key.

def _mk(cls, d, **extra):
    if d is None:
        return None
    known = {f for f in cls.__dataclass_fields__}
    unknown = set(d) - known - set(extra)
    if unknown:
        raise ValueError(f"{cls.__name__}: unknown key(s) {sorted(unknown)}; known are {sorted(known)}")
    return cls(**{**{k: v for k, v in d.items() if k in known}, **extra})


def rig_from_dict(d):
    return _mk(Rig, d,
               gripper=_mk(Gripper, d.get("gripper")),
               cameras=tuple(_mk(Camera, c) for c in d.get("cameras", [])),
               sensors=tuple(d.get("sensors", ("proprio",))))


def contract_from_dict(d):
    if d is None:
        return None
    grasp = d.get("grasp")
    release = d.get("release")
    return _mk(Contract, d,
               actuation=_mk(Actuation, d["actuation"]),
               execution=_mk(Execution, d["execution"],
                             terminate=tuple(d["execution"].get("terminate", ())),
                             terminate_pre_step=tuple(d["execution"].get("terminate_pre_step", ()))),
               grasp=(_mk(Grasp, grasp, signal=_mk(GraspSignal, grasp["signal"])) if grasp else None),
               release=(_mk(Release, release) if release else None),
               rig=(rig_from_dict(d["rig"]) if d.get("rig") else None))


def app_from_dict(d):
    recipe = d.get("recipe")
    arts = d.get("artifacts", [])
    app = _mk(App, d,
              recipe=(_mk(Recipe, recipe,
                          env=_mk(EnvSpec, recipe["env"]),
                          stages=tuple(_mk(Stage, s) for s in recipe.get("stages", [])))
                      if recipe else None),
              artifacts=tuple(_mk(Artifact, a,
                                  contract=contract_from_dict(a.get("contract")),
                                  eval=_mk(EvalSpec, a.get("eval")))
                              for a in arts))
    app.validate()
    return app


def app_to_dict(app):
    def clean(o):
        if is_dataclass(o):
            return clean(asdict(o))
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items() if v is not None and v != {} and v != ()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        return o
    return clean(app)


# ── the store ─────────────────────────────────────────────────────────────────────────────────
class Store:
    def __init__(self, root):
        self.root = os.path.expanduser(root)

    def _load_text(self, text):
        try:
            import yaml
            return yaml.safe_load(text)
        except ImportError:
            return json.loads(text)

    def _dump_text(self, obj):
        try:
            import yaml
            return yaml.safe_dump(obj, sort_keys=False, width=100)
        except ImportError:
            return json.dumps(obj, indent=2)

    def paths(self):
        if not os.path.isdir(self.root):
            return []
        return sorted(os.path.join(self.root, f) for f in os.listdir(self.root)
                      if f.endswith((".yaml", ".yml", ".json")))

    def apps(self):
        out = []
        for p in self.paths():
            with open(p) as f:
                out.append(app_from_dict(self._load_text(f.read())))
        return out

    def get(self, name):
        for a in self.apps():
            if a.name == name:
                return a
        raise KeyError(f"no app named {name!r} in {self.root}")

    def save(self, app):
        app.validate()
        os.makedirs(self.root, exist_ok=True)
        p = os.path.join(self.root, f"{app.name}.yaml")
        with open(p, "w") as f:
            f.write(self._dump_text(app_to_dict(app)))
        return p

    # ── the product's verb ────────────────────────────────────────────────────────────────────
    def plan(self, app, rig, target_contract=None) -> Plan:
        blockers = app.missing_requirements(rig)
        if blockers:
            return Plan(BLOCKED, app.name, blockers=tuple(blockers),
                        reason="the rig does not meet this skill's hard requirements")

        recipe_stages = tuple(s.kind for s in (app.recipe.stages if app.recipe else ()))

        if not app.artifacts:
            return Plan(RETRAIN, app.name, stages=recipe_stages,
                        reason="no pretrained artifact exists for any rig")

        target = target_contract if target_contract is not None else app.contract_for(rig)
        best, best_art = None, None
        for art in app.artifacts:
            r = resolve_contracts(art.contract, target)
            if best is None or {RUN: 0, ADAPT: 1, RETRAIN: 2}[r.verdict] < {RUN: 0, ADAPT: 1, RETRAIN: 2}[best.verdict]:
                best, best_art = r, art
            if r.verdict == RUN:
                break

        if best.verdict == RUN:
            return Plan(RUN, app.name, checkpoint=best_art.path, resolution=best,
                        reason="the trained contract matches this rig")
        if best.verdict == ADAPT:
            # Only stages the recipe ACTUALLY defines. The previous fallback returned ADAPT_STAGES
            # regardless, so a recipe defining neither of them produced a plan naming stages that do
            # not exist — which Foundry then rejects, turning a recoverable skill into an error the
            # caller cannot act on. If there is nothing to re-run, the honest answer is a full
            # rebuild, not an adaptation that cannot be executed.
            stages = tuple(s for s in recipe_stages if s in ADAPT_STAGES)
            if not stages:
                return Plan(RETRAIN, app.name, stages=recipe_stages, resolution=best,
                            reason="recoverable in principle, but the recipe defines none of the "
                                   f"adaptation stages {ADAPT_STAGES} — rebuilding instead")
            return Plan(ADAPT, app.name, stages=stages, checkpoint=best_art.path, resolution=best,
                        reason="the policy is recoverable by re-running the perception stages")
        return Plan(RETRAIN, app.name, stages=recipe_stages, resolution=best,
                    reason="trained for a different problem")
