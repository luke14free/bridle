"""bridle.adapters.env_ref — resolving a skill document's `env_id:` to a real environment.

THE ONE PLACE EITHER SPELLING IS RESOLVED, and the reason it is an ADAPTER and not core. A skill may
name its environment two ways:

    env_id: SO100DescendToTarget-v1          # a uid in the simulator's registry
    env_id: my.module:MyEnvClass             # an importable class — what `Skill(env=MyEnv)` emits

`bridle.skill` is stdlib-only and may not import a simulator, so it treats a class as an opaque
reference and serialises it. Everything that needs a live env — the registry lookup, the class
import, `max_episode_steps` — happens here, where a backend dependency belongs. That split is what
makes a second backend a NEW MODULE beside this one rather than an edit to the schema: today this
knows about ManiSkill's `REGISTERED_ENVS` and nothing else does; an Isaac consumer adds its own
resolver and every refusal, default and digest above this line is untouched.

"CANNOT VERIFY" IS NEVER RENDERED AS "VERIFIED". `env_id` used to be an unresolved free string, so
`env_id: ThisEnvDoesNotExist-v9` passed `bridle skill check` with exit 0 and a stamped plan
fingerprint. `check_env_ref` below closes that — but it distinguishes THREE outcomes, not two:
`OK`, `UNKNOWN` (the registry was consulted and does not have it — a refusal), and `NOT_CHECKED`
(no simulator is importable in this interpreter, so nothing was consulted). A checker that reported
the third as a pass would be a check that passes by not running, which is a shape this project has
been bitten by repeatedly.
"""
import glob
import importlib
import os
import sys

__all__ = [
    "EnvRefError", "OK", "UNKNOWN", "NOT_CHECKED", "ResolvedEnv",
    "is_class_ref", "check_env_ref", "resolve_env_ref", "import_registering_module",
]

OK = "ok"
UNKNOWN = "unknown"
NOT_CHECKED = "not_checked"


class EnvRefError(Exception):
    """An `env_id:` that names nothing this interpreter can reach."""


class ResolvedEnv:
    """`(cls, uid, max_episode_steps)` plus how it was found.

    `uid` is None for a class that ManiSkill has never registered — legal, and the caller decides
    whether it can proceed. `max_episode_steps` is None in exactly the same case, and it is NEVER
    defaulted here: a wrong horizon is CLAUDE.md gotcha (1), a full day lost to an env that silently
    auto-reset mid-rollout, so a caller that needs one has to be told it is missing.
    """

    __slots__ = ("cls", "uid", "max_episode_steps", "how")

    def __init__(self, cls, uid, max_episode_steps, how):
        self.cls = cls
        self.uid = uid
        self.max_episode_steps = max_episode_steps
        self.how = how

    def __repr__(self):
        return (f"<ResolvedEnv {getattr(self.cls, '__name__', self.cls)} uid={self.uid!r} "
                f"max_episode_steps={self.max_episode_steps!r} via {self.how}>")


def is_class_ref(text):
    """`module.path:QualName` — the form `Skill(env=<class>)` serialises to. A ManiSkill uid
    (`SO100DescendToTarget-v1`) carries no colon, so the two forms cannot be confused."""
    return isinstance(text, str) and ":" in text


# ── the registry ────────────────────────────────────────────────────────────────────────────────

def _registry():
    """ManiSkill's `REGISTERED_ENVS`, or None when no simulator is importable here."""
    try:
        from mani_skill.utils.registration import REGISTERED_ENVS
    except Exception:                                             # noqa: BLE001 — absence is a state
        return None
    return REGISTERED_ENVS


def _dotted_module(path):
    """A file path -> the dotted module name it would be imported as, using `sys.path` as it stands.

    Deliberately reads `sys.path` rather than assuming a repo root: the caller's PYTHONPATH is what
    decides whether `primitives/descend_to_target/descend_env.py` is
    `primitives.descend_to_target.descend_env` or nothing at all, and guessing a root that is not on
    the path produces an import error that blames the wrong thing.
    """
    path = os.path.abspath(path)
    best = None
    for entry in sys.path:
        if not entry:
            continue
        root = os.path.abspath(entry)
        if path.startswith(root + os.sep):
            rel = os.path.relpath(path, root)
            if best is None or len(rel) < len(best):
                best = rel
    if best is None:
        return None
    if best.endswith(".py"):
        best = best[: -len(".py")]
    return best.replace(os.sep, ".")


def import_registering_module(env_id, search_dir, explicit=None):
    """Import whatever module registers `env_id`, and return its dotted name (or None).

    The search is the skill document's own directory: a primitive's env definition lives next to it
    (`primitives/<name>/<name>_env.py`, the PRIMITIVE_CONTRACT layout), so the document's location
    IS the pointer to the code. An import that raises is REPORTED, not hidden — a missing env whose
    module fails to import is a different problem from a missing env, and the two used to look the
    same from the outside.

    Returns `(module_name_or_None, tried, notes)`.
    """
    registry, tried, notes = _registry(), [], []
    if registry is None:
        return None, tried, notes
    if env_id in registry:
        return registry[env_id].cls.__module__, tried, notes
    if explicit:
        candidates = [explicit]
    elif search_dir:
        candidates = [m for m in (_dotted_module(f) for f in
                                  sorted(glob.glob(os.path.join(search_dir, "*_env.py")))) if m]
    else:
        candidates = []
    for mod in candidates:
        tried.append(mod)
        try:
            importlib.import_module(mod)
        except Exception as e:                                    # noqa: BLE001 — reported, not hidden
            notes.append(f"importing {mod} raised {type(e).__name__}: {e}")
            continue
        if env_id in registry:
            return mod, tried, notes
    return None, tried, notes


def _import_class(ref):
    module_name, _, qualname = ref.partition(":")
    try:
        module = importlib.import_module(module_name)
    except Exception as e:                                        # noqa: BLE001
        raise EnvRefError(f"env_id {ref!r} names a class in module {module_name!r}, which does not "
                          f"import: {type(e).__name__}: {e}") from e
    obj = module
    for part in qualname.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            raise EnvRefError(f"env_id {ref!r}: module {module_name!r} imported, but it has no "
                              f"{qualname!r}")
    if not isinstance(obj, type):
        raise EnvRefError(f"env_id {ref!r} resolves to a {type(obj).__name__}, not a class")
    return obj


def _registration_of(cls, registry):
    """The `(uid, max_episode_steps)` this class was registered under, or `(None, None)`.

    A class handed to `Skill(env=...)` is normally decorated with `@register_env(uid,
    max_episode_steps=N)`, and reading N off that registration is the only honest way to get the
    horizon: it is the number the trainer will actually enforce.
    """
    for uid, entry in registry.items():
        if getattr(entry, "cls", None) is cls:
            return uid, getattr(entry, "max_episode_steps", None)
    return None, None


# ── the two entry points ────────────────────────────────────────────────────────────────────────

def resolve_env_ref(env, *, search_dir=None, explicit_module=None):
    """`env_id:` (or a live class) -> a `ResolvedEnv`. Raises `EnvRefError` when it names nothing.

    `env` may be the class itself (from `Skill.env`), a `module:QualName` string, or a registered
    uid. All three end at the same place, which is the point of the function.
    """
    registry = _registry()
    if registry is None:
        raise EnvRefError("no simulator registry is importable in this interpreter "
                          "(`mani_skill.utils.registration`), so no env reference can be resolved")
    if isinstance(env, type):
        uid, horizon = _registration_of(env, registry)
        return ResolvedEnv(env, uid, horizon, "class")
    if is_class_ref(env):
        cls = _import_class(env)
        uid, horizon = _registration_of(cls, registry)
        return ResolvedEnv(cls, uid, horizon, "class-ref")
    module, tried, notes = import_registering_module(env, search_dir, explicit_module)
    if env not in registry:
        if tried:
            detail = f" (tried {tried})"
        elif search_dir and glob.glob(os.path.join(search_dir, "*_env.py")):
            # There ARE candidate files and none of them could be named: the directory's package
            # root is not on `sys.path`, which is a different problem from a missing env and used to
            # be reported as the same one.
            detail = (f" — {search_dir} holds *_env.py files but none of them is importable under "
                      f"the current sys.path, so no module name could be derived. Put the "
                      f"repository root on PYTHONPATH")
        else:
            detail = f" — no *_env.py sits next to it in {search_dir or '(no directory given)'}"
        detail += "".join(f"\n  note: {n}" for n in notes)
        raise EnvRefError(f"env_id {env!r} is registered by nothing importable from "
                          f"{search_dir or 'the current sys.path'}{detail}")
    entry = registry[env]
    return ResolvedEnv(entry.cls, env, entry.max_episode_steps, f"registry via {module}")


def check_env_ref(env, *, search_dir=None, explicit_module=None):
    """`(status, detail)` — `OK`, `UNKNOWN` (a refusal) or `NOT_CHECKED` (nothing was consulted)."""
    if _registry() is None:
        return NOT_CHECKED, ("no simulator registry is importable in this interpreter, so "
                             "`env_id` was NOT resolved — this is not a pass")
    try:
        resolved = resolve_env_ref(env, search_dir=search_dir, explicit_module=explicit_module)
    except EnvRefError as e:
        return UNKNOWN, str(e)
    where = (f"{resolved.cls.__module__}.{resolved.cls.__name__}"
             if resolved.cls is not None else "?")
    horizon = (f"max_episode_steps={resolved.max_episode_steps}"
               if resolved.max_episode_steps is not None
               else "max_episode_steps UNKNOWN (the class carries no registration)")
    return OK, f"{where} ({resolved.how}, {horizon})"
