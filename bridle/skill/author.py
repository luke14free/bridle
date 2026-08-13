"""bridle.skill.author — the skill document, spelled as typed Python.

    from bridle.skill import Skill, Training, terms as T, predicates as P

    descend = Skill(
        name="descend_to_target",
        env=SO100DescendToTargetEnv,               # a class, or a registered id string
        kind="carry", contract="stack",
        params={"hover": 0.015, "low_band": 0.03, "center_tol": 0.045},
        reward=[
            T.PredicateBonus(1.0, P.grasped, why="hold on — release is a separate skill"),
            ...
        ],
        success=P.all(P.grasped, P.below_resting_height("params.low_band")),
        training=Training(),
    )
    descend.train()

WHY THIS EXISTS, AND WHAT IT IS NOT A SECOND OF. The reward has to be **data** rather than code —
that is what makes the plan fingerprint, the pre-GPU refusals and weight sweeps possible, and it is
not negotiable. But "data" never meant "YAML": `bridle.skill.spec.parse_spec` takes a plain `dict`
and this package imports `yaml` nowhere. YAML was only ever the file format the CLI reads.

So this module is a set of typed builders that produce **exactly the document a YAML file would**
and hand it to the SAME `parse_spec -> compile_spec`. There is no second schema, no second
vocabulary and no second set of refusals: a builder mistake is refused by the schema tier, with the
schema tier's dotted path, legal set and nearest-match suggestion. One pipeline, two spellings —
`bridle/tests/test_author.py` asserts that the two spellings of `descend_to_target` compile to the
same digest, which is the whole architectural claim and would be worth nothing unstated.

THE BUILDERS ARE GENERATED FROM THE VOCABULARY, NOT HAND-LISTED. `terms` carries one builder per
`vocab.TERMS` entry and `predicates` one per `vocab.PREDICATES` entry, built by iterating those
tables at import time, with a coverage assert in both directions below. A hand-written list is a
list that goes stale the first time the vocabulary grows; `compile.py` sets the precedent with its
own import-time coverage asserts and this module keeps it.

`why=` IS MANDATORY HERE TOO. The schema tier refuses a reward row without one because the rationale
is the only surviving record of why a weight is what it is; a Python front-end that defaulted it
would be the loophole around the rule rather than a second spelling of it. The builder refuses it at
the call site, where the traceback still points at the row being written.

WHAT THIS MODULE DELIBERATELY DOES NOT DO — it never touches a simulator. `bridle.skill` is
stdlib-only and stays that way, so an `env=` class is an **opaque reference**: it is checked to be a
class, serialised as `module:qualname` for the document, and handed on. Resolving it — registry
lookup, subclassing, `max_episode_steps` — belongs to `bridle.adapters`, and running PPO on the
result belongs to the consumer (see `Skill.train`'s launcher seam).
"""
import argparse
import dataclasses
import difflib
import inspect
import json
import sys

from bridle.skill.compile import compile_spec
from bridle.skill.spec import ROW_TERMS, SpecError, parse_spec
from bridle.skill.vocab import PREDICATES, TERMS

__all__ = [
    "Skill", "Training", "Criterion", "TermRow", "ScaleRow", "LauncherError",
    "terms", "predicates", "register_launcher", "default_launcher",
]


# ── rendering values into the document's own spellings ──────────────────────────────────────────

def _render_value(value):
    """One argument of a predicate call, as the criterion text spells it.

    A STRING IS RENDERED BARE, exactly as the YAML does: `anchor=target_pos`, not
    `anchor='target_pos'`. Python quotes are how a Python author writes a NAME here — the criterion
    grammar has no other way to spell one — and quoting it through would hand the evaluator a string
    constant where it expects an identifier. `params.center_tol` reaches `_bind_text` the same way.
    """
    if isinstance(value, Criterion):
        return value.render()
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, str):
        return value
    return repr(value)


class Criterion:
    """A predicate reference, rendered to the text `success:` and `predicate:`/`gate:` are written in.

    THE RENDERED TEXT HASHES, so this renderer is deterministic and its output is pinned by a test.
    `_lower_term_row` carries the success criterion onto the `SuccessBonus` op as `condition`, and
    `RewardPlan.fingerprint` covers every op parameter — so `where(success, 12.0, acc)` under a
    different criterion is a different reward function and the digest says so. It also means the
    digest is sensitive to the criterion's WHITESPACE, which is a property of the fingerprint rather
    than of this module: a multi-operand combinator therefore renders one operand per line with the
    continuation aligned under the opening bracket, which is both the conventional way to wrap a
    call and, not by accident, exactly what a human wrote in
    `primitives/descend_to_target/skill.yaml` — the two spellings of that document have to produce
    the same bytes or they would produce different digests.
    """

    __slots__ = ("name", "args", "kwargs", "open", "close")

    def __init__(self, name, args=(), kwargs=(), *, open="(", close=")"):
        self.name = name
        self.args = tuple(args)
        self.kwargs = tuple(kwargs)          # ordered (name, value) pairs
        self.open = open
        self.close = close

    def render(self, col=0):
        if not self.args and not self.kwargs:
            return self.name
        head = f"{self.name}{self.open}"
        parts = [_render_positional(a, col + len(head)) for a in self.args]
        parts += [f"{k}={_render_value(v)}" for k, v in self.kwargs]
        if len(parts) > 1 and not self.kwargs and all(isinstance(a, Criterion) for a in self.args):
            # A boolean combinator over several operands: one per line, aligned under the bracket.
            joiner = ",\n" + " " * (col + len(head))
            return head + joiner.join(parts) + self.close
        return head + ", ".join(parts) + self.close

    def __str__(self):
        return self.render()

    def __repr__(self):
        return f"<Criterion {self.render()!r}>"

    def __eq__(self, other):
        return str(self) == str(other) if isinstance(other, (Criterion, str)) else NotImplemented

    def __hash__(self):
        return hash(str(self))


def _render_positional(value, col):
    return value.render(col) if isinstance(value, Criterion) else _render_value(value)


# ── the generated namespaces ────────────────────────────────────────────────────────────────────

def _suggest(value, candidates):
    close = difflib.get_close_matches(str(value), sorted(candidates), n=1, cutoff=0.6)
    return close[0] if close else None


class _Namespace:
    """Attribute access over a generated table, whose MISS is a `SpecError` and not an
    `AttributeError`. `T.DistancePul` is the same typo class as `term: DistancePul` in YAML and has
    to be answerable the same way: path, nearest match, legal set."""

    _what = "name"

    def __init__(self, table, label):
        self._table = dict(table)
        self._label = label

    def names(self):
        return tuple(self._table)

    def __dir__(self):
        return sorted(set(self._table) | set(super().__dir__()))

    def __getattr__(self, name):
        table = self.__dict__.get("_table") or {}
        if name in table:
            return table[name]
        if name.startswith("_"):
            raise AttributeError(name)
        raise SpecError(f"{self._label}.{name}", f"unknown {self._what} {name!r}",
                        legal=table, suggestion=_suggest(name, table))


class _TermNamespace(_Namespace):
    _what = "reward term"


class _PredicateNamespace(_Namespace):
    _what = "predicate"


def _predicate_builder(name, predicate):
    """One builder per `vocab.PREDICATES` entry, with the vocabulary's own parameter order.

    A zero-parameter predicate is bound as a `Criterion` VALUE rather than a function, so `P.grasped`
    is written the way the document spells it — a bare name, not a call.
    """
    params = predicate.params
    if not params:
        return Criterion(name)

    if name in ("and_", "or_"):
        # VARIADIC, because the corpus's spelling of them is. `terms: list[predicate]` is one
        # declared parameter, but `vocab.desugar_brackets` lowers `all[a, b, c]` to `and_(a, b, c)`
        # and `skill_predicates.Args.all_predicates` flattens either form — so binding these two
        # against a one-parameter signature would refuse the only spelling the evaluator ever sees.
        def build(*operands, **kwargs):
            terms_ = list(kwargs.pop("terms", ()) or ()) + list(operands)
            if kwargs:
                raise SpecError(f"predicates.{name}.{sorted(kwargs)[0]}",
                                f"{name} takes operand predicates and nothing else",
                                legal=("terms",))
            if not terms_:
                raise SpecError(f"predicates.{name}",
                                f"{name} needs at least one operand predicate")
            return Criterion(name, args=tuple(terms_))
    else:
        def build(*args, **kwargs):
            bound = _bind_args(f"predicates.{name}", name, [p.name for p in params], args, kwargs)
            return Criterion(name, kwargs=tuple(bound.items()))

    build.__name__ = name
    build.__doc__ = predicate.doc
    build.__signature__ = inspect.Signature(
        [inspect.Parameter(p.name, inspect.Parameter.POSITIONAL_OR_KEYWORD,
                           default=inspect.Parameter.empty if p.required else p.default)
         for p in params])
    return build


def _combinator(sugar, label):
    def build(*operands):
        if not operands:
            raise SpecError(f"predicates.{sugar}",
                            f"`{sugar}[...]` needs at least one operand predicate")
        return Criterion(sugar, args=operands, open="[", close="]")

    build.__name__ = sugar
    build.__doc__ = (f"`{sugar}[a, b, ...]` — the bracket sugar `success:` is written in, which "
                     f"`vocab.desugar_brackets` lowers to `{label}(a, b, ...)`.")
    return build


def _bind_args(path, label, order, args, kwargs):
    """Positional arguments onto the vocabulary's parameter order, refusing the two mistakes the
    schema tier cannot see because they never reach a document: too many positionals, and one
    parameter supplied twice."""
    if len(args) > len(order):
        raise SpecError(path, f"{label} takes {len(order)} positional argument(s) "
                              f"({', '.join(order)}), got {len(args)}", legal=order)
    bound = {}
    for name, value in zip(order, args):
        bound[name] = value
    for name, value in kwargs.items():
        if name in bound:
            raise SpecError(f"{path}.{name}",
                            f"{name!r} was given twice — once positionally and once by keyword")
        bound[name] = value
    return bound


@dataclasses.dataclass(frozen=True)
class TermRow:
    """One reward row, still as data: `term`, its authored parameters, and the mandatory `why`."""

    term: str
    params: dict
    why: str

    def as_document(self):
        row = {"term": self.term}
        row.update({k: (str(v) if isinstance(v, Criterion) else v)
                    for k, v in self.params.items()})
        row["why"] = self.why
        return row


@dataclasses.dataclass(frozen=True)
class ScaleRow:
    """`RewardScale` — a DOCUMENT-level field and not a summed row (`reward_ppo = dense / divisor`).
    It has a builder so the vocabulary's coverage assert is total; `Skill` refuses it inside
    `reward=[...]` and takes it as `reward_scale=`."""

    params: dict

    def as_document(self):
        return dict(self.params)


def _term_builder(name, term):
    order = [p.name for p in term.params]

    if name == "RewardScale":
        def build(*args, **kwargs):
            return ScaleRow(_bind_args(f"terms.{name}", name, order, args, kwargs))
    else:
        def build(*args, why=None, **kwargs):
            if not isinstance(why, str) or not why.strip():
                raise SpecError(
                    f"{name}.why",
                    "every reward row needs a non-empty `why`: the rationale for the number you "
                    "chose, in prose. It is mandatory because stating the rationale before the "
                    "number is what took an LLM-authored reward from 50% to 90%, and because the "
                    "document keeps no comments — this field is the only surviving record of why "
                    "the weight is what it is")
            return TermRow(name, _bind_args(f"{name}", name, order, args, kwargs), why)

    build.__name__ = name
    build.__doc__ = term.doc
    return build


#: The two generated namespaces. `terms as T` / `predicates as P` is the intended import.
terms = _TermNamespace({name: _term_builder(name, t) for name, t in TERMS.items()}, "terms")
predicates = _PredicateNamespace(
    dict([(name, _predicate_builder(name, p)) for name, p in PREDICATES.items()]
         + [("all", _combinator("all", "and_")), ("any", _combinator("any", "or_"))]),
    "predicates")

# ── coverage, asserted at import time in BOTH directions ────────────────────────────────────────
# The precedent is `compile.py`, which asserts its `_PER_STEP_MAXIMUM` table covers `ROW_TERMS`
# exactly. The point of generating the builders is that a vocabulary addition cannot leave the
# front-end behind; these say so out loud, so a future hand-edit that breaks the generation is a
# failed import and not a missing builder discovered by an author.
assert set(terms.names()) == set(TERMS), (
    f"the term builders are generated from vocab.TERMS and must cover it exactly: "
    f"{set(terms.names()) ^ set(TERMS)}")
assert set(predicates.names()) == set(PREDICATES) | {"all", "any"}, (
    f"the predicate builders are generated from vocab.PREDICATES plus the two bracket sugars: "
    f"{set(predicates.names()) ^ (set(PREDICATES) | {'all', 'any'})}")


# ── training hyperparameters ────────────────────────────────────────────────────────────────────

@dataclasses.dataclass(frozen=True)
class Training:
    """The PPO hyperparameters, defaulting to `primitives/descend_to_target/teacher_train.sh`.

    THE DEFAULTS ARE THE DELIVERABLE. The point of this surface is that a skill author writes almost
    nothing, and a run under a different batch size or a different gamma is comparable to nothing —
    so the defaults are the deployed lineage's, verbatim, rather than a library's idea of sensible.
    Every field here is read by the LAUNCHER (see `Skill.train`), which is what knows how to spell
    them for a particular trainer; nothing in this dataclass reaches the skill DOCUMENT, because the
    document describes the reward and the reward does not depend on the batch size.
    """

    seed: int = 20
    num_envs: int = 4096
    num_steps: int = 16
    update_epochs: int = 8
    num_minibatches: int = 32
    total_steps: int = 150_000_000
    gamma: float = 0.9
    ent_coef: float = 0.001
    partial_reset: bool = True
    num_eval_envs: int = 16
    num_eval_steps: int = 64
    eval_freq: int | None = None
    early_stop_success: float = 0.90
    early_stop_patience: int = 25
    capture_video: bool = False
    #: Consumer-side paths. `env_kwargs` is the trainer's `--env-kwargs-json-path`; `workdir` is
    #: where `runs/<exp>` lands (default: the directory of the file that defines the skill).
    env_kwargs: str | None = None
    workdir: str | None = None
    exp: str | None = None
    wandb: bool = True
    diagnostics: bool = True


# ── the launcher seam ───────────────────────────────────────────────────────────────────────────

class LauncherError(RuntimeError):
    """No launcher — `Skill.train()` was asked to train and nothing here knows how."""


_DEFAULT_LAUNCHER = []


def register_launcher(fn):
    """Register the process-wide default launcher. Returns `fn`, so it works as a decorator.

    WHY THE SEAM IS HERE AND NOT FILLED IN. `bridle` owns the PLAN — the vocabulary, the schema, the
    fold, the fingerprint and the contract — and it is stdlib-only on purpose: the whole schema tier
    is testable on a CPU in milliseconds, and nothing in it may import torch, ManiSkill, or a repo
    that does. Running PPO is the consumer's: it knows the trainer's flags, its checkpoint layout,
    its GPU etiquette and its run directory. lego-arm registers
    `scripts/train_from_skill.py:maniskill_ppo` on its side.

    That split is also what makes a second backend a NEW LAUNCHER rather than a change here: an
    Isaac consumer registers its own and every refusal, every default and every digest above this
    line stays exactly as it is.

    A launcher is called by keyword only:

        launcher(*, skill, spec, plan, mode, options) -> int

    `mode` is one of `"train"`, `"verify"`, `"register"` or `"plan"`; `options` carries whatever the
    front-end CLI parsed (`smoke`, `dry_run`, `no_wait`, ...). The return value is a process exit
    code.
    """
    _DEFAULT_LAUNCHER[:] = [fn]
    return fn


def default_launcher():
    return _DEFAULT_LAUNCHER[0] if _DEFAULT_LAUNCHER else None


# ── the skill ───────────────────────────────────────────────────────────────────────────────────

def _env_reference(env):
    """`env=` -> the string the document carries under `env_id:`.

    A CLASS IS AN OPAQUE REFERENCE HERE. This module may not import a simulator, so it checks only
    that the value IS a class and serialises it as `module:qualname` — which is what lets a
    Python-authored skill round-trip through YAML at all. Resolving either spelling to a live env is
    `bridle.adapters.env_ref`'s job, and it is the only place that knows what a ManiSkill uid is.
    """
    if isinstance(env, type):
        return f"{env.__module__}:{env.__qualname__}"
    if isinstance(env, str) and env.strip():
        return env
    raise SpecError("env", f"`env=` is a registered env id (a non-empty string) or an env class, "
                           f"got {type(env).__name__} ({env!r}). A class is serialised as "
                           f"`module:qualname` and resolved by the adapter, so the environment can "
                           f"be defined in the same file as the reward")


def _param_document(name, value):
    """A `params:` entry. A bare number is shorthand for `{value: N, severity: retrain}`.

    `retrain` IS THE CONSERVATIVE DEFAULT AND THAT IS WHY IT MAY BE ONE. `severity` answers "can a
    checkpoint trained under the old number be reused?" — `retrain` says no. Defaulting to the
    strictest answer can only cost a retrain that was not needed; defaulting to `run` would silently
    reuse a checkpoint that is no longer valid, which is what the field exists to prevent. Write the
    full `{value, severity, doc}` mapping to say anything else.
    """
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SpecError(f"params.{name}",
                        f"a param is a number (shorthand for `{{value: N, severity: retrain}}`) or "
                        f"the full mapping of {{value, severity, doc}}, got "
                        f"{type(value).__name__} ({value!r})")
    return {"value": value, "severity": "retrain"}


class Skill:
    """A skill document, authored in Python. `.doc` is the mapping; everything else goes through it.

    `env` is a registered env id or an env CLASS. `training` is the PPO configuration and is NOT
    part of the document — see `Training`. `scene` is optional and descriptive; see `_parse_scene`.
    """

    def __init__(self, *, name, env, kind, contract, reward, success,
                 params=None, scene=None, init=None, reward_scale=None, preflight=None,
                 training=None, max_episode_steps=None):
        self.name = name
        self.env = env
        self.env_ref = _env_reference(env)
        self.kind = kind
        self.contract = contract
        self.reward = list(reward)
        self.success = success
        self.params = dict(params or {})
        self.scene = scene
        self.init = init
        self.reward_scale = reward_scale
        self.preflight = preflight
        self.training = training or Training()
        #: The env's `max_episode_steps`, when the author states it. NEVER DEFAULTED: it
        #: parameterises compile's horizon-integrated check, and CLAUDE.md gotcha (1) is a full day
        #: lost to a horizon that silently differed. Left None, the check reports that it could not
        #: be computed, and the launcher — which can read the registration — supplies the real one
        #: and refuses a disagreement.
        self.max_episode_steps = max_episode_steps

    # ── the document ────────────────────────────────────────────────────────────────────────────

    @property
    def doc(self):
        """The mapping a YAML loader would have produced. Built fresh on every access."""
        rows = []
        for i, row in enumerate(self.reward):
            if isinstance(row, ScaleRow):
                raise SpecError(
                    f"reward[{i}].term",
                    "RewardScale is a document-level field, not a reward row: pass it as "
                    "`reward_scale=T.RewardScale(divisor=12.0)`. It divides the summed dense reward "
                    "before PPO sees it; summed as a row it would be 12x wrong", legal=ROW_TERMS)
            if isinstance(row, TermRow):
                rows.append(row.as_document())
            elif isinstance(row, dict):
                rows.append(dict(row))
            else:
                raise SpecError(f"reward[{i}]",
                                f"a reward row is a `terms.<Term>(...)` builder result or the "
                                f"mapping one produces, got {type(row).__name__}", legal=ROW_TERMS)
        doc = {
            "name": self.name,
            "kind": self.kind,
            "contract": self.contract,
            "env_id": self.env_ref,
        }
        if self.scene is not None:
            doc["scene"] = self.scene
        if self.init is not None:
            doc["init"] = dict(self.init)
        if self.params:
            doc["params"] = {k: _param_document(k, v) for k, v in self.params.items()}
        if self.reward_scale is not None:
            scale = self.reward_scale
            doc["reward_scale"] = (scale.as_document() if isinstance(scale, ScaleRow)
                                   else dict(scale))
        doc["reward"] = rows
        doc["success"] = str(self.success)
        if self.preflight is not None:
            doc["preflight"] = self.preflight
        return doc

    def spec(self):
        """`parse_spec(self.doc)` — the SAME schema tier a YAML document goes through."""
        return parse_spec(self.doc)

    def plan(self, *, horizon=None, terminate_on_success=True):
        """`compile_spec` over `self.spec()`. Every refusal — schema and compile — happens here."""
        h = horizon if horizon is not None else self.max_episode_steps
        return compile_spec(self.spec(), horizon=h, terminate_on_success=terminate_on_success)

    def to_yaml(self):
        """The same document, as YAML text — so the two spellings are interconvertible and not
        merely similar. Written with the stdlib (`bridle` core has no PyYAML dependency); the
        round-trip through a real loader is asserted in `bridle/tests/test_author.py`."""
        return _to_yaml(self.doc)

    # ── running it ──────────────────────────────────────────────────────────────────────────────

    def _dispatch(self, mode, launcher, horizon, options):
        plan = self.plan(horizon=horizon)
        fn = launcher if launcher is not None else default_launcher()
        if fn is None:
            raise LauncherError(
                f"{self.name}: compiled cleanly (plan@{plan.fingerprint()}) but no launcher is "
                f"registered, so there is nothing here that knows how to train it. `bridle` owns "
                f"the plan; running PPO belongs to the consumer — import the module that registers "
                f"one (lego-arm: `scripts.train_from_skill`), or pass `launcher=` explicitly. See "
                f"`bridle.skill.author.register_launcher` for the seam and why it is where it is.")
        return fn(skill=self, spec=self.spec(), plan=plan, mode=mode, options=dict(options or {}))

    def train(self, *, launcher=None, horizon=None, **options):
        """Compile, then hand the plan to the launcher. Refusals happen BEFORE anything is launched.

        The launcher is expected to verify the derived reward against the base env's own before it
        trains, and to refuse on a mismatch — `scripts/train_from_skill.py` in lego-arm does exactly
        that and is the reference implementation of the protocol.
        """
        return self._dispatch("train", launcher, horizon, options)

    def verify(self, *, launcher=None, horizon=None, **options):
        """Live-rollout equivalence against the base env's deployed reward. Run this first."""
        return self._dispatch("verify", launcher, horizon, options)

    def register(self, *, launcher=None, horizon=None, **options):
        """Build and register the derived env, print its id, train nothing."""
        return self._dispatch("register", launcher, horizon, options)

    # ── the file's own entry point ──────────────────────────────────────────────────────────────

    def main(self, argv=None, *, launcher=None):
        """`python my_skill.py [--check|--verify|--train|--register-only]`.

        `--check` is the default because it is the only mode that needs no GPU: it compiles the
        document and prints the plan fingerprint, which is the answer to "is this the reward I think
        it is?" and is the cheap half of the three feedback tiers.
        """
        p = argparse.ArgumentParser(description=f"{self.name} — a bridle skill, as one file")
        m = p.add_mutually_exclusive_group()
        m.add_argument("--check", action="store_true",
                       help="compile and print the plan fingerprint; no simulator (default)")
        m.add_argument("--verify", action="store_true",
                       help="live-rollout equivalence against the base env's deployed reward")
        m.add_argument("--register-only", action="store_true",
                       help="register the derived env and print its id, for an external trainer")
        m.add_argument("--train", action="store_true", help="run PPO on the derived env")
        p.add_argument("--yaml", action="store_true", help="print the document as YAML and stop")
        p.add_argument("--smoke", action="store_true", help="--train, but a few updates only")
        p.add_argument("--dry-run", action="store_true", help="--train, but print the argv and stop")
        p.add_argument("--no-wait", action="store_true",
                       help="do not wait for another training job to free the GPU")
        p.add_argument("--num-envs", type=int, default=None)
        p.add_argument("--total-steps", type=int, default=None)
        p.add_argument("--exp", default=None, help="experiment name (default carries the plan "
                                                   "fingerprint)")
        a = p.parse_args(argv if argv is not None else sys.argv[1:])

        if a.yaml:
            print(self.to_yaml())
            return 0
        plan = self.plan()
        print(f"skill    {self.name} ({self.kind} chassis, contract {self.contract})")
        print(f"env      {self.env_ref}")
        print(f"plan     plan@{plan.fingerprint()}, {len(plan.ops)} ops, scale {plan.scale}")
        for w in plan.warnings:
            print(f"  {w.level}: {' '.join(str(w).split())[:240]}")
        if not (a.verify or a.train or a.register_only):
            print("\nchecked, nothing launched. Tiers 1-2 of 3 (schema -> compile): the document is "
                  "well-formed and internally consistent, NOT that the reward trains.")
            return 0
        mode = "verify" if a.verify else ("register" if a.register_only else "train")
        options = {k: getattr(a, k) for k in
                   ("smoke", "dry_run", "no_wait", "num_envs", "total_steps", "exp")}
        return self._dispatch(mode, launcher, None, options)


# ── the stdlib YAML emitter ─────────────────────────────────────────────────────────────────────
# Small on purpose: it emits the skill document's own shapes (mappings, lists, strings, numbers,
# bools, null) and nothing else. `bridle` core has no PyYAML dependency and this is not the place to
# acquire one — the CLI is where YAML is READ, and a document this cannot render falls back to
# JSON, which every YAML 1.2 loader accepts.

def _scalar(value):
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    return json.dumps(value)


def _block_literal_ok(text):
    """Can this string be a `|-` block? It must have no trailing newline, no line with trailing
    whitespace (a loader is free to strip it) and a first line that does not start with a space
    (which would need an explicit indentation indicator). Anything else falls back to a quoted
    scalar, which escapes the newlines and round-trips just as exactly."""
    lines = text.split("\n")
    return (len(lines) > 1 and text == text.rstrip("\n") and lines[0][:1] != " "
            and not any(line != line.rstrip() for line in lines))


def _emit(key, value, indent, out):
    pad = " " * indent
    if isinstance(value, dict):
        if not value:
            out.append(f"{pad}{key}: {{}}")
            return
        out.append(f"{pad}{key}:")
        for k, v in value.items():
            _emit(k, v, indent + 2, out)
        return
    if isinstance(value, (list, tuple)):
        if not value:
            out.append(f"{pad}{key}: []")
            return
        if all(not isinstance(v, (dict, list, tuple)) for v in value):
            out.append(f"{pad}{key}: [" + ", ".join(_scalar(v) for v in value) + "]")
            return
        out.append(f"{pad}{key}:")
        for item in value:
            if isinstance(item, dict):
                sub = []
                for k, v in item.items():
                    _emit(k, v, indent + 4, out=sub)
                sub[0] = f"{pad}  - " + sub[0][indent + 4:]
                out.extend(sub)
            else:
                out.append(f"{pad}  - {_scalar(item)}")
        return
    if isinstance(value, str) and _block_literal_ok(value):
        out.append(f"{pad}{key}: |-")
        out.extend(f"{pad}  {line}" if line else "" for line in value.split("\n"))
        return
    out.append(f"{pad}{key}: {_scalar(value)}")


def _to_yaml(doc):
    out = []
    for key, value in doc.items():
        _emit(key, value, 0, out)
    return "\n".join(out) + "\n"
