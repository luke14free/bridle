"""Preflight execution against a ManiSkill env. Needs torch + ManiSkill; core does not.

This file MEASURES and does not DECIDE. Which asserts apply, whether a bound holds, and what a kind
implies all live in `bridle.preflight`, which is stdlib-only and unit-tested on CPU. Keeping the
seam here is what stops the gotcha logic from becoming GPU-only and therefore untested.
"""
import ast
import importlib
import os

from bridle.preflight import DYNAMIC, STATIC


def _resolve(path: str):
    """`a.b.c.NAME` -> the attribute, importing the longest importable prefix."""
    parts = path.split(".")
    for cut in range(len(parts) - 1, 0, -1):
        mod_name, attr_path = ".".join(parts[:cut]), parts[cut:]
        try:
            obj = importlib.import_module(mod_name)
        except ImportError:
            continue
        for a in attr_path:
            obj = getattr(obj, a)
        return obj
    raise ImportError(f"could not import any prefix of {path!r}")


class _Unresolved:
    """Sentinel for a path that could not be resolved at all (bad module or missing attribute —
    a SPEC error), as distinct from a path that resolved fine to a real value of `None` (an
    INVARIANT violation — e.g. a derived config attribute that is genuinely unset). Collapsing
    both to `None` makes a typo'd `path=` in an authored assert indistinguishable from the
    real bug preflight exists to catch.

    `bridle.preflight.Assert.holds` is core and intentionally untouched here: it fails on
    `value is None` before ever looking at `expect`/`min`/`max`. This sentinel is not `None`, so
    it does not hit that branch — instead it implements `==`/`<`/`>`/`<=`/`>=` to force every
    shape of `holds()` (expect-equality, min-floor, max-ceiling) to also come out False, so an
    unresolved path still fails every assert. `format_failures` then prints this sentinel's repr
    instead of `missing`/`None`, so the failure reads as a spec error, not a regressed invariant.
    """

    def __repr__(self):
        return "<unresolved path>"

    __str__ = __repr__

    def __eq__(self, other):
        return False

    def __hash__(self):
        return id(self)

    def __lt__(self, other):
        return True

    def __gt__(self, other):
        return True

    def __le__(self, other):
        return True

    def __ge__(self, other):
        return True


#: Returned by `static_values` for a path that failed to resolve. Never returned for a path that
#: resolved successfully to a real `None` — see `_Unresolved`'s docstring for why that matters.
UNRESOLVED = _Unresolved()


def static_values(paths) -> dict:
    """Read each dotted path AFTER import, so the value reflects what the env actually derived.

    A path that fails to import or has no such attribute maps to `UNRESOLVED`, not `None` — that
    distinction is the whole point, see `UNRESOLVED`'s docstring. A path that resolves and the
    attribute really is `None` keeps that `None` unchanged.
    """
    out = {}
    for p in paths:
        try:
            out[p] = _resolve(p)
        except (ImportError, AttributeError):
            out[p] = UNRESOLVED
    return out


def readable_env(module: str) -> set:
    """Every environment variable a module's SOURCE consults, by AST — never a hardcoded list.

    Used to refuse `--set` on a variable the target never reads (the PRIM_DESCEND_CENTER_TOL vs
    PRIM_DSTACK_CENTER_TOL class). Walks the module and everything it imports from `primitives.`,
    looking for os.environ.get("X") / os.environ["X"] / os.getenv("X")."""
    seen, found, queue = set(), set(), [module]
    while queue:
        name = queue.pop()
        if name in seen:
            continue
        seen.add(name)
        try:
            spec = importlib.util.find_spec(name)
        except (ImportError, ValueError):
            continue
        if spec is None or not spec.origin or not spec.origin.endswith(".py"):
            continue
        try:
            with open(spec.origin) as fh:
                tree = ast.parse(fh.read())
        except (SyntaxError, ValueError):
            # One module reachable via a `primitives.` import chain being unparseable (a WIP
            # file, a syntax error mid-edit) must not sink the whole scan — skip it and keep
            # walking the rest of the import graph.
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                continue
            if isinstance(node, ast.Call):
                f = node.func
                is_get = (isinstance(f, ast.Attribute) and f.attr in ("get", "getenv")
                          and node.args and isinstance(node.args[0], ast.Constant)
                          and isinstance(node.args[0].value, str))
                if is_get:
                    found.add(node.args[0].value)
            if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute) \
                    and node.value.attr == "environ" and isinstance(node.slice, ast.Constant):
                found.add(node.slice.value)
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                mod = getattr(node, "module", None) or ""
                names = [mod] if mod else [a.name for a in node.names]
                queue += [n for n in names if n.startswith("primitives")]
    return found


def dynamic_metrics(env_id: str, module: str, ckpt=None, envs: int = 64, steps: int = 64) -> dict:
    """One eval, the same shape the trainer's own eval uses, so the numbers are comparable to the
    training log rather than a new scale nobody can calibrate.

    Returns {f"{k}_once": rate, f"{k}_at_end": rate} for every boolean key in the info dict — the
    same names the trainer logs as eval_<k>_{once,at_end}_mean.
    """
    import gymnasium as gym
    import torch
    import mani_skill.envs  # noqa: F401  — registers the task envs
    from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
    importlib.import_module(module)          # registers THIS primitive's env id

    raw = gym.make(env_id, num_envs=envs, obs_mode="state", sim_backend="physx_cuda",
                    control_mode="pd_joint_target_delta_pos", max_episode_steps=400)
    # `gym.make(env_id, num_envs=...)` above returns a raw env, not a ManiSkillVectorEnv: ManiSkill's
    # `register_env` wires `entry_point=partial(registration.make, env_id=uid)` for plain `gym.make`
    # and only points `vector_entry_point` at `registration.make_vec` — and `gymnasium.make()` never
    # consults `vector_entry_point` (that's read solely by `gymnasium.make_vec()`, which nothing here
    # calls). So the wrap below is ASSERTING the "no mid-rollout reset" property this probe needs, not
    # repairing an observed bug in the line above it.
    #
    # It is worth asserting explicitly rather than relying on it by accident: `BaseEnv.step`
    # (`mani_skill/envs/sapien_env.py`) sets `terminated = info["success"].clone()` whenever
    # `evaluate()` returns a `"success"` key — `descend_env.py` does — so a future switch to
    # `gym.make_vec`, or a change to `ManiSkillVectorEnv`'s `auto_reset=True`/`ignore_terminations=False`
    # defaults, would silently reintroduce a mid-rollout reset that overwrites a just-succeeded
    # sub-env's `info` before this probe (which scans `info` directly) ever reads it. This probe's
    # rates must stay reset-free to be comparable to the trainer's own eval, which guards the same way
    # (`lerobot_sim2real/rl/ppo_state.py` builds `ManiSkillVectorEnv(..., ignore_terminations=not
    # args.partial_reset)` itself rather than trusting `gym.make*` defaults).
    env = ManiSkillVectorEnv(raw, envs, auto_reset=False, ignore_terminations=True)
    obs, _ = env.reset(seed=0)
    dev = env.unwrapped.device
    act_fn = None
    if ckpt and os.path.isfile(ckpt):
        from lerobot_sim2real.rl.ppo_state import Agent
        ag = Agent(env, {"state": obs}).to(dev)
        sd = torch.load(ckpt, map_location=dev)
        ag.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in sd.items()}, strict=False)
        ag.eval()
        act_fn = lambda o: ag.get_action({"state": o}, deterministic=True)   # noqa: E731
    else:
        space = env.action_space
        act_fn = lambda o: torch.as_tensor(space.sample(), device=dev).reshape(envs, -1)  # noqa: E731

    once, info = {}, {}
    with torch.no_grad():
        for _ in range(steps):
            obs, _, _, _, info = env.step(act_fn(obs))
            for k, v in info.items():
                if torch.is_tensor(v) and v.dtype == torch.bool:
                    once[k] = v.clone() if k not in once else (once[k] | v)
    out = {}
    for k, v in once.items():
        out[f"{k}_once"] = float(v.float().mean())
    for k, v in info.items():
        if torch.is_tensor(v) and v.dtype == torch.bool:
            out[f"{k}_at_end"] = float(v.float().mean())
    env.close()
    return out


def collect(asserts, env_id: str, module: str, ckpt=None, envs: int = 64, steps: int = 64) -> dict:
    """Observed values for every assert. Static first: if static fails there is no point paying for
    the simulator, and the caller stops on the first non-empty failure list."""
    values = static_values([a.path for a in asserts if a.tier == STATIC])
    if any(a.tier == DYNAMIC for a in asserts):
        values.update(dynamic_metrics(env_id, module, ckpt, envs, steps))
    return values
