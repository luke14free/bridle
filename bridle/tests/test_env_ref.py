"""Unit test for bridle.adapters.env_ref — resolving a document's `env_id:` to a real environment.

WHAT IT HAS TO PROVE. `env_id` was an unresolved free string: `env_id: ThisEnvDoesNotExist-v9`
reported `skill check OK, exit 0` and produced a stamped plan fingerprint. So the checks below are
about the three outcomes, and about the third one in particular:

  OK           the id (or the `module:QualName` class reference) names something that exists.
  UNKNOWN      the registry WAS consulted and does not have it — a refusal.
  NOT_CHECKED  no simulator is importable here, so nothing was consulted. This is not a pass, it is
               printed as "NOT CHECKED", and a test that let it read as OK would be re-introducing
               the "a check that passes by not running" shape this module exists to remove.

This test imports the simulator when one is available and says so plainly when it is not — it does
not silently degrade into asserting nothing.

Run: PYTHONPATH=. python bridle/tests/test_env_ref.py
"""
import sys

from bridle.adapters.env_ref import (
    NOT_CHECKED, OK, UNKNOWN, EnvRefError, check_env_ref, is_class_ref, resolve_env_ref,
)

FAILS = []


def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        FAILS.append(name)


class PlainClass:
    """A class no simulator has ever registered — the case where `max_episode_steps` is genuinely
    unknown and must be reported as unknown rather than defaulted."""


def run_checks():
    # ── the two spellings are distinguishable, and a uid is never mistaken for a class ──────────
    check("a module:QualName reference is recognised as a class",
          is_class_ref("primitives.descend_to_target.descend_env:SO100DescendToTargetEnv"))
    check("a ManiSkill uid is not — it carries no colon",
          not is_class_ref("SO100DescendToTarget-v1") and not is_class_ref("Reach-v1"))

    registry_available = True
    try:
        import mani_skill.utils.registration  # noqa: F401
    except Exception as e:                                        # noqa: BLE001
        registry_available = False
        print(f"  NOTE  no simulator in this interpreter ({type(e).__name__}) — the registry legs "
              f"below assert the NOT_CHECKED contract instead, which is the honest answer here")

    if not registry_available:
        status, detail = check_env_ref("SO100DescendToTarget-v1")
        check("with no registry, the answer is NOT_CHECKED and says so",
              status == NOT_CHECKED and "NOT resolved" in detail)
        check("...and it is not OK", status != OK)
        return

    # ── UNKNOWN: the refusal that used to be a pass ─────────────────────────────────────────────
    status, detail = check_env_ref("ThisEnvDoesNotExist-v9")
    check("an env id nothing registers is UNKNOWN, not OK",
          status == UNKNOWN and "ThisEnvDoesNotExist-v9" in detail)
    try:
        resolve_env_ref("ThisEnvDoesNotExist-v9")
        raised = None
    except EnvRefError as e:
        raised = e
    check("resolve_env_ref raises EnvRefError for it", isinstance(raised, EnvRefError))

    # ── OK by class, where the horizon is honestly unknown ──────────────────────────────────────
    status, detail = check_env_ref(f"{__name__}:PlainClass")
    check("a class reference that imports is OK", status == OK)
    check("...and an unregistered class reports max_episode_steps UNKNOWN rather than a default",
          "max_episode_steps UNKNOWN" in detail)
    resolved = resolve_env_ref(PlainClass)
    check("a live class resolves to itself with no uid and no horizon invented",
          (resolved.cls, resolved.uid, resolved.max_episode_steps) == (PlainClass, None, None))

    status, detail = check_env_ref(f"{__name__}:NoSuchClass")
    check("a class reference whose name is absent is UNKNOWN and names the module",
          status == UNKNOWN and __name__ in detail)
    status, detail = check_env_ref("no.such.module:Thing")
    check("a class reference whose module does not import is UNKNOWN and says why",
          status == UNKNOWN and "does not import" in detail)

    # ── OK by id, including the sibling-module search ───────────────────────────────────────────
    from mani_skill.utils.registration import REGISTERED_ENVS
    known = next(iter(REGISTERED_ENVS))
    status, detail = check_env_ref(known)
    check(f"an env the registry already holds ({known}) is OK", status == OK)
    check("...and its registered max_episode_steps is reported, not invented",
          "max_episode_steps=" in detail and "UNKNOWN" not in detail)
    resolved = resolve_env_ref(known)
    check("resolve_env_ref returns the registered class and horizon",
          resolved.uid == known
          and resolved.max_episode_steps == REGISTERED_ENVS[known].max_episode_steps)


def test_bridle():
    FAILS.clear()
    run_checks()
    assert not FAILS, f"{len(FAILS)} check(s) failed: {FAILS}"


def main():
    run_checks()
    print(f"\n{len(FAILS)} failure(s)")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
