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
    # MUST explicitly wrap with ManiSkillVectorEnv(..., auto_reset=False, ignore_terminations=True)
    # rather than trust any `gym.make*` call to do it safely — there is no kwarg-passthrough route.
    # `mani_skill/utils/registration.py`'s own vector constructor is
    #     def make_vec(env_id, **kwargs):
    #         env = gym.make(env_id, **kwargs)
    #         env = ManiSkillVectorEnv(env)
    #         return env
    # which forwards NOTHING to `ManiSkillVectorEnv(env)` — so it always takes the class defaults
    # (`mani_skill/vector/wrappers/gymnasium.py`, `ManiSkillVectorEnv.__init__`):
    #     auto_reset: bool = True,
    #     ignore_terminations: bool = False,
    # Under those defaults, `ManiSkillVectorEnv.step()` does:
    #     if dones.any() and self.auto_reset:
    #         final_info = torch_clone_dict(infos)
    #         obs, infos = self.reset(options=dict(env_idx=env_idx))
    #         infos["final_info"] = final_info
    # and `dones = terminations | truncations` where `terminations` comes straight from
    # `BaseEnv.step` (`mani_skill/envs/sapien_env.py`): `terminated = info["success"].clone()`
    # whenever `evaluate()` returns a `"success"` key — which `descend_env.py` does. So the instant
    # a sub-env succeeds, that index is reset within the SAME `.step()` call and its `info` is
    # overwritten with fresh post-reset (False) values; the real terminal values only survive under
    # `infos["final_info"]`, which a probe that scans `info` directly (as this one does) never reads.
    # `lerobot_sim2real/rl/ppo_state.py` hits this exact trap and avoids it the only way available —
    # building `ManiSkillVectorEnv` itself, not through any `gym.make*` kwarg:
    #     envs = gym.make(args.env_id, num_envs=args.num_envs, ...)
    #     envs = ManiSkillVectorEnv(envs, args.num_envs, ignore_terminations=not args.partial_reset, ...)
    # Measured cost of getting this wrong (bridle/preflight.py's own worked example, 2026-08-12): a
    # descend policy with real `is_grasped_at_end` 0.859 reads far lower once the successful steps'
    # info gets silently replaced by the post-reset frame. `auto_reset=False` here goes one step
    # further than ppo_state.py's `ignore_terminations` (which still auto-resets on truncation) —
    # this probe wants a fixed-length window with NO reset at all, ever, mid-rollout or otherwise.
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
