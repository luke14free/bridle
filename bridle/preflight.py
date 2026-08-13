"""PREFLIGHT — refuse to spend GPU on a run that cannot succeed.

WHY THIS MODULE EXISTS. 2026-08-12: a descend retrain ran 15M steps with `is_grasped_at_end` at
0.055. Healthy is 0.859. The cube was on the floor before the descend finished, so
`success = is_grasped & low & centered` could never fire. The cause was `PRIM_CARRY_GRIP_HOLD`
defaulting to OFF. Nothing in the stack objected: the trainer logged metrics, the loss went down,
wandb drew curves. The run was structurally incapable of succeeding from step 0 and only a human
reading `is_grasped_at_end` in a log noticed.

TWO TIERS. The STATIC tier reads values the target module DERIVED after import; the DYNAMIC tier
rolls one short eval. Static is the important half, and asserting a DERIVED value rather than an
environment variable is the load-bearing choice: `PRIM_CARRY_GRIP_HOLD == "1"` only proves a string
was exported, while `primitives.coord_mixin.CARRY_GRIP_HOLD is True` proves the env READ it. That
difference is exactly the failure where a variable is set under a name the target never consults
(PRIM_DESCEND_CENTER_TOL at a primitive whose env reads PRIM_DSTACK_CENTER_TOL).

A STRUCTURAL CHECK IS NOT A COMPETENCE BAR. Every assert here must be satisfiable at step 0.
`is_grasped_at_end >= 0.5` qualifies because the descend env restores an already-grasped snapshot
and the carry grip freeze holds it: a random policy passes and only a broken CONFIGURATION fails. An
assert that depends on learned behaviour must set `needs="warm_start"` and is skipped for
from-scratch runs.

Stdlib only: the decisions live here so they can be tested in seconds with no simulator.
"""
from dataclasses import dataclass, replace

STATIC = "static"
DYNAMIC = "dynamic"


class Loosened(Exception):
    """An authored assert tried to weaken one the primitive's kind requires.

    This matters ahead of LLM-authored specs: whoever writes the spec also writes the asserts, and
    a model asked to make training pass will write `min: 0.0`. A kind assert is what the kind MEANS
    (a carry primitive that is not holding anything is not carrying), so it is a floor."""


class DuplicateFloor(Exception):
    """KIND_ASSERTS[kind] has two floors on the same path.

    A kind's floors are what the kind MEANS. A kind cannot mean two different things about one
    path, so this is a malformed kind table, not an input to reconcile — it is raised, not merged."""


@dataclass(frozen=True)
class Assert:
    """One checkable claim. `path` is a dotted module attribute (STATIC) or a metric name (DYNAMIC)."""

    path: str
    tier: str
    min: float | None = None
    max: float | None = None
    expect: object = None
    needs: str | None = None
    source: str = "authored"

    def __post_init__(self):
        if self.expect is None and self.min is None and self.max is None:
            raise ValueError(
                f"Assert(`{self.path}`) has no bound (no expect/min/max) — an assert with no "
                f"bound is not a claim, it is a check that can never fail.")

    def holds(self, value) -> bool:
        if value is None:                       # missing is a failure, never a pass
            return False
        if self.expect is not None:
            return value is self.expect or (type(value) is type(self.expect) and value == self.expect)
        if self.min is not None and value < self.min:
            return False
        if self.max is not None and value > self.max:
            return False
        return True

    def describe(self) -> str:
        if self.expect is not None:
            return f"{self.path} == {self.expect}"
        parts = []
        if self.min is not None:
            parts.append(f">= {self.min}")
        if self.max is not None:
            parts.append(f"<= {self.max}")
        return f"{self.path} {' and '.join(parts)}"


# ── the INITIATION DISTRIBUTION tier ────────────────────────────────────────────────────────────
#
# WHERE THE EPISODE STARTS IS A CHECKABLE PROPERTY OF A CONFIGURATION, and until 2026-08-13 nothing
# checked it. A descend run trained from an initiation set whose cube sat mean 29.6 cm from the
# target — with 0.000 of starts inside the ~6 cm its re-centring reward was tuned for, against a
# 4.5 cm success tolerance in a 64-step episode — and every assert in the file passed: they covered
# tolerances (STATIC) and learned behaviour (DYNAMIC competence bars) and nothing covered the
# STATES THE POLICY IS HANDED. Two runs died on it, ~a day of GPU.
#
# AN INITIATION-DISTRIBUTION ASSERT IS STRUCTURAL, NOT A COMPETENCE BAR — the distinction this
# module's docstring draws, and the strongest possible case of it: the measurement is taken at
# `reset()`, before a single action, so it does not depend on the policy AT ALL. There is no
# `needs="warm_start"` question to ask about one. A random policy, a warm-started one and no policy
# whatsoever produce the same number, and only a broken CONFIGURATION moves it.
#
# THE NAMING IS A GENERAL MEASURE, NOT A DESCEND SPECIAL CASE. Any per-env float the env publishes
# in its `info` dict can be summarised at reset:
#
#     init_<info_key>_mean          init_<info_key>_min / _max
#     init_<info_key>_frac_within_<x>     fraction of starts with <info_key> <= x
#
# so `init_cube_to_target_dist_frac_within_0.06` is descend's, and a bin-drop skill asserting
# `init_tcp_to_object_dist_max` or a reach skill asserting `init_obj_to_goal_dist_mean` costs no new
# code. Parsing lives HERE (stdlib, tested on CPU, no simulator) and measuring lives in
# `bridle.adapters.preflight.init_metrics` — the same MEASURES/DECIDES seam as the rest of the file.

INIT_PREFIX = "init_"

#: Suffix -> whether it takes a numeric argument. Order matters only for readability; the parse
#: below is exact, not a prefix guess.
INIT_STATS = {"mean": False, "min": False, "max": False, "frac_within": True}


def parse_init_stat(path):
    """`init_<info_key>_<stat>[_<arg>]` -> `(info_key, stat, arg)`, or None if `path` is not one.

    Returning None (rather than raising) is what lets `collect` sort a mixed list of DYNAMIC asserts
    into the two things that measure them, without either tier having to be told which is which.

    The argument is parsed as a float and kept as one: `..._frac_within_0.06` is 6 cm, and the name
    carries the unit of the underlying measure (metres here) because the info key does. A stat that
    takes no argument refuses one — `init_x_mean_0.5` is a typo, not a bound, and reading it as
    `mean` would silently drop the number the author wrote.
    """
    if not isinstance(path, str) or not path.startswith(INIT_PREFIX):
        return None
    rest = path[len(INIT_PREFIX):]
    for stat, takes_arg in INIT_STATS.items():
        token = "_" + stat
        if takes_arg:
            marker = token + "_"
            if marker in rest:
                key, _, tail = rest.rpartition(marker)
                try:
                    return (key, stat, float(tail)) if key else None
                except ValueError:
                    return None
        elif rest.endswith(token):
            key = rest[:-len(token)]
            return (key, stat, None) if key else None
    return None


#: What a KIND means, physically. These are floors, not defaults: the author never states them and
#: cannot weaken them. `carry` is the only kind Phase 0 needs; adding a kind is adding a row here
#: plus a test, deliberately.
KIND_ASSERTS = {
    # A carry primitive holds an object while it moves. If the grip freeze is off, the jaws drift
    # open over a 60-step horizon and the run is dead on arrival (measured 0.055 vs 0.859).
    "carry": (
        Assert("primitives.coord_mixin.CARRY_GRIP_HOLD", STATIC, expect=True, source="kind=carry"),
        Assert("primitives.coord_mixin.COORD_OBS", STATIC, expect=True, source="kind=carry"),
        Assert("is_grasped_at_end", DYNAMIC, min=0.5, source="kind=carry"),
    ),
}


def _tightens(authored: Assert, floor: Assert) -> bool:
    if floor.expect is not None:
        return authored.expect == floor.expect
    if floor.min is not None and (authored.min is None or authored.min < floor.min):
        return False
    if floor.max is not None and (authored.max is None or authored.max > floor.max):
        return False
    return True


def merge(kind, authored) -> tuple:
    """Kind floors plus authored asserts. An authored assert on the same path may only tighten."""
    floors = KIND_ASSERTS.get(kind or "", ())
    by_path = {}
    for i, a in enumerate(floors):
        if a.path in by_path:
            raise DuplicateFloor(
                f"kind={kind} has two floors on `{a.path}` — a kind's floors are what the kind "
                f"MEANS, and it cannot mean two different things about one path.")
        by_path[a.path] = i
    out = list(floors)
    for a in authored:
        idx = by_path.get(a.path)
        if idx is None:
            out.append(a)
            continue
        floor = out[idx]
        if not _tightens(a, floor):
            raise Loosened(
                f"authored assert `{a.describe()}` weakens what kind={kind} requires "
                f"(`{floor.describe()}`). A kind assert is a floor: tighten it or drop it.")
        out[idx] = replace(a, source=f"authored (tightens kind={kind})")
    return tuple(out)


class _NotMeasured:
    """Sentinel for a DYNAMIC assert `collect` chose not to measure because the STATIC tier already
    failed (the default, GPU-saving short-circuit — see `bridle.adapters.preflight.collect`).

    Distinct from `None` (a path that WAS measured and the result is genuinely absent — usually a
    typo'd assert name) and distinct from the adapter's own `_Unresolved` (a STATIC path that failed
    to import). An unmeasured assert must still FAIL — unmeasured is not a pass — so `holds()` must
    come out False for every bound shape (expect/min/max), the same trick `_Unresolved` uses: this
    is not `None` so it skips the `value is None` fast-fail, and instead forces `==` False and every
    ordering comparison True, which fails a min-floor, a max-ceiling, and an expect-equality alike.
    `format_failures` then renders this sentinel's repr instead of `observed missing` or a raw
    number, so the failure reads as "we never checked", not "we checked and it failed" — collapsing
    those two is exactly the ambiguity F1 exists to remove.
    """

    def __repr__(self):
        return "not measured (static tier failed first)"

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


#: Returned by `collect` for every DYNAMIC assert path it chose not to measure. Never returned for
#: a path that was measured and came back missing (that stays `None`) — see `_NotMeasured`'s
#: docstring for why the distinction matters.
NOT_MEASURED = _NotMeasured()


@dataclass(frozen=True)
class Failure:
    assertion: Assert
    observed: object


def evaluate(asserts, values: dict, from_scratch: bool = False) -> list:
    """Every assert that does not hold. A path absent from `values` FAILS (it is usually a typo)."""
    out = []
    for a in asserts:
        if from_scratch and a.needs == "warm_start":
            continue
        v = values.get(a.path)
        if not a.holds(v):
            out.append(Failure(a, v))
    return out


def format_effective(asserts) -> str:
    """The merged set with provenance — printed before every launch so nothing is invisible.

    An assert with `needs="warm_start"` is marked inline: it is silently skipped under
    `--from-scratch` (`evaluate`'s `from_scratch` branch), and a reader looking at this list ahead
    of a from-scratch launch has to be able to tell which lines will not actually be checked."""
    w = max((len(a.tier) for a in asserts), default=7)
    lines = []
    for a in asserts:
        needs = f"  [needs={a.needs}, skipped by --from-scratch]" if a.needs else ""
        lines.append(f"  {a.tier:<{w}}  {a.describe():<52}  ({a.source}){needs}")
    return "\n".join(lines)


#: One line of diagnosis per assert we have actually seen fail, with the measurement behind it.
HINTS = {
    "is_grasped_at_end":
        "the policy is not holding the object at the end of the rollout. For kind=carry this is "
        "almost always the grip freeze being off: check PRIM_CARRY_GRIP_HOLD=1 and "
        "PRIM_CARRY_GRIP_CLOSE=0.0 (measured 2026-08-12: 0.055 without, 0.859 with).",
    "primitives.coord_mixin.CARRY_GRIP_HOLD":
        "PRIM_CARRY_GRIP_HOLD is not set to 1 in this process. It defaults to OFF.",
    "primitives.coord_mixin.COORD_OBS":
        "PRIM_COORD_OBS is not set to 1, so the env is not building the coord observation the "
        "deployed policies read.",
}


def format_failures(failures) -> str:
    lines = ["PREFLIGHT FAILED"]
    for f in failures:
        if f.observed is NOT_MEASURED:
            obs = NOT_MEASURED          # str()s to "not measured (static tier failed first)"
        elif f.observed is None:
            obs = "missing"
        else:
            obs = f.observed
        lines.append(f"  [{f.assertion.tier}] {f.assertion.describe()}, observed {obs}")
        hint = HINTS.get(f.assertion.path)
        if hint:
            lines.append(f"      -> {hint}")
    lines.append("REFUSING TO LAUNCH.")
    return "\n".join(lines)
