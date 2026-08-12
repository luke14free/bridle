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
        return f"{self.path} {' and '.join(parts) if parts else '(no bound)'}"


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
    by_path = {a.path: a for a in floors}
    out = list(floors)
    for a in authored:
        floor = by_path.get(a.path)
        if floor is None:
            out.append(a)
            continue
        if not _tightens(a, floor):
            raise Loosened(
                f"authored assert `{a.describe()}` weakens what kind={kind} requires "
                f"(`{floor.describe()}`). A kind assert is a floor: tighten it or drop it.")
        out[out.index(floor)] = replace(a, source=f"authored (tightens kind={kind})")
    return tuple(out)


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
    """The merged set with provenance — printed before every launch so nothing is invisible."""
    w = max((len(a.tier) for a in asserts), default=7)
    return "\n".join(f"  {a.tier:<{w}}  {a.describe():<52}  ({a.source})" for a in asserts)


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
        obs = "missing" if f.observed is None else f.observed
        lines.append(f"  [{f.assertion.tier}] {f.assertion.describe()}, observed {obs}")
        hint = HINTS.get(f.assertion.path)
        if hint:
            lines.append(f"      -> {hint}")
    lines.append("REFUSING TO LAUNCH.")
    return "\n".join(lines)
