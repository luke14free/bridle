"""An App: what a robot skill IS. Frozen dataclasses + YAML — no sim, no torch.

    App = manifest (what the LLM needs to choose it)
        + recipe   (how to REGENERATE it on your rig)
        + artifacts (pretrained weights, each stamped with the contract it was trained under)

RECIPE FIRST, WEIGHTS SECOND — and that ordering is the whole architecture.

The tempting design is "an app is a checkpoint; adapt it by fine-tuning". It is far less to build and
much faster when it works. The evidence says it mostly will not: in the codebase bridle came from, a
hold-step difference alone was worth 0.40 vs 0.83 success, and a release-height difference was worth
0/20. If policies are that contract-sensitive, cross-rig weight transfer is the exception, not the
rule — so a store that ships only weights is a store that mostly ships broken skills.

Shipping a reproducible training PROCEDURE makes the honest promise (reproducible skills, not
portable weights) and degrades gracefully: weights when the fingerprint matches, recipe when it does
not. `bridle.resolve` is what chooses between them, per field, with reasons.

WHAT A RECIPE IS NOT: a training framework. bridle does not invent learning algorithms; a recipe
NAMES the stages and their parameters, and the Foundry executes them against a rig. The stage
implementations belong to whoever authored the skill.
"""
import hashlib
import json
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class Stage:
    """One step of a training pipeline.

    `kind` is free-form on purpose — bridle does not own the taxonomy of how policies get trained.
    The Foundry resolves a kind to a runner; an unknown kind is an error at plan time, not at hour
    three of a GPU job.
    """

    kind: str                      # "teacher" | "round_robin" | "distill" | "student" | custom
    params: dict = field(default_factory=dict)


@dataclass(frozen=True)
class EnvSpec:
    """The scene a skill is trained in, declaratively.

    Deliberately narrow: an id plus kwargs, not a general scene language. A backend-agnostic scene
    description is a research project, and inventing one before a second backend exists would be
    designing for an imagined requirement. The cost is that this is ManiSkill-shaped today, and the
    first port will pay for it — accepted knowingly (see the architecture spec's risk section).
    """

    id: str
    kwargs: dict = field(default_factory=dict)
    max_episode_steps: int = 400   # chain/long rollouts MUST set this; the default 40 silently
                                   # auto-resets mid-rollout and fabricates failures


@dataclass(frozen=True)
class EvalSpec:
    """How the skill is graded, and what it scored when its author ran it.

    `reported` is evidence, not a promise: it records rig + number + n, so a user can see what the
    claim rests on. A skill claiming 0.85 with no n and no rig is a skill claiming nothing.
    """

    protocol: str
    n: int = 0
    reported: dict = field(default_factory=dict)   # {"rig": "...", "success": 0.85, "n": 48}


@dataclass(frozen=True)
class Recipe:
    """Everything needed to regenerate the policy on a rig."""

    env: EnvSpec
    stages: tuple = ()
    reward: str = ""               # module path / identifier of the reward implementation
    notes: str = ""

    def fingerprint(self) -> str:
        blob = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:12]


@dataclass(frozen=True)
class Artifact:
    """A pretrained checkpoint, and the contract it was trained under.

    The contract is the point. Without it a checkpoint is an unlabelled jar: runnable, and no way to
    know whether running it means anything.
    """

    path: str
    contract: object                # bridle.contract.Contract
    eval: EvalSpec = None


@dataclass(frozen=True)
class App:
    """A named capability an LLM can choose and a Foundry can build."""

    name: str
    title: str
    description: str                # what it does — the LLM reads this to choose
    when_to_use: str                # when it applies — and when it does not
    args: dict = field(default_factory=dict)     # arg name -> description, for tool synthesis
    recipe: Recipe = None
    artifacts: tuple = ()
    version: str = "0.1.0"
    requires: dict = field(default_factory=dict)  # rig requirements, e.g. {"sensors": ["rgb"]}

    def validate(self) -> None:
        if not self.name:
            raise ValueError("app.name is required")
        if self.recipe is None and not self.artifacts:
            raise ValueError(
                f"app {self.name!r} has neither a recipe nor artifacts — it cannot be run and "
                "cannot be built. One of the two is the minimum meaningful app.")
        for a in self.artifacts:
            if a.contract is None:
                raise ValueError(
                    f"app {self.name!r} has an unstamped artifact ({a.path}). A checkpoint with no "
                    "contract cannot be checked against a rig, which makes it worse than no "
                    "checkpoint: it will run anyway.")

    def contract_for(self, rig):
        """The contract this app would run under on `rig`.

        Takes the first artifact's contract as the task shape and re-points it at the target rig:
        the TASK is the app's, the RIG is the user's, and `resolve` compares the two. An app with no
        artifacts has no task shape to borrow, so it must be built from the recipe.
        """
        import dataclasses
        if not self.artifacts:
            raise ValueError(f"app {self.name!r} has no artifact to derive a contract from; "
                             "build it from the recipe first")
        return dataclasses.replace(self.artifacts[0].contract, rig=rig)

    def missing_requirements(self, rig) -> list:
        """Hard rig requirements this app declares that `rig` does not meet.

        Distinct from `resolve`: this is "can this rig run this skill AT ALL" (no camera, no force
        sensor), not "will these weights transfer". A missing sensor is not retrainable-around.
        """
        out = []
        for s in self.requires.get("sensors", []):
            if s not in rig.sensors:
                out.append(f"sensor {s!r} (rig has {list(rig.sensors)})")
        for c in self.requires.get("cameras", []):
            if c not in [cam.name for cam in rig.cameras]:
                out.append(f"camera {c!r} (rig has {[cam.name for cam in rig.cameras]})")
        gk = self.requires.get("gripper_kind")
        if gk and rig.gripper.kind != gk:
            out.append(f"gripper kind {gk!r} (rig has {rig.gripper.kind!r})")
        return out
