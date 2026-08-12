"""A LINEAGE is a deployed policy plus the exact environment that produced it.

WHY THIS MODULE EXISTS. On 2026-08-12 two training launches were thrown away in a row. The second
omitted `PRIM_CARRY_GRIP_HOLD=1` / `PRIM_CARRY_GRIP_CLOSE=0.0`, which DEFAULT TO OFF: the gripper was
never frozen during the carry, `is_grasped_at_end` ran 0.055 instead of 0.859, and
`success = is_grasped & low & centered` could never fire. The run trained happily and reported
metrics the whole time. The correct settings existed — as a PROSE SENTENCE in a YAML field
(`composer/store/apps/place_coord_v3.yaml`: "obs: coord 357, PRIM_CARRY_GRIP_HOLD freeze; trained w/
PRIM_DESCEND_CENTER_TOL=0.025..."). That sentence was right and was still missed, because prose
cannot be executed and cannot be checked.

The missing operation is "retrain this exact deployed lineage with ONE variable changed" — the most
common training action in the project, and the literal meaning of its own "one variable per run"
rule. Here it is as data: capture the environment, apply named overrides, and make the diff the
first thing anybody sees.

Stdlib only, deliberately: everything here must be testable in seconds with no simulator.
"""
import os
from dataclasses import dataclass

#: Environment prefixes that constitute a lineage. Drawn from the 47 variables measured across
#: `primitives/*/*.py` + `primitives/coord_mixin.py` on 2026-08-12; anything outside these
#: namespaces (PATH, HOME, LD_LIBRARY_PATH, ...) is machine state, not part of an experiment.
NAMESPACES = ("PRIM_", "COORD_", "GRAB_", "COMPACT_", "REACHGRAB_", "MOVE_", "BRIDLE_", "WANDB_")


class EmptyDiff(Exception):
    """A relaunch that changes nothing is either a mistake or a resume, and a resume is a
    different command. Refusing is what keeps "one variable per run" true by construction."""


class UnknownOverride(Exception):
    """An override naming a variable the target env never reads. This is the
    PRIM_DESCEND_CENTER_TOL-vs-PRIM_DSTACK_CENTER_TOL class: the export succeeds, the training
    runs, and the number you thought you changed was never read by anything."""


@dataclass(frozen=True)
class Change:
    """One field of the lineage diff. `before=None` means the variable was not previously set."""

    key: str
    before: str | None
    after: str | None


@dataclass(frozen=True)
class Mismatch:
    """Two records of one fact, disagreeing. `effective` is what the system actually resolves to;
    `claimed` is what `record` says it is. `effective=None` means the live env does not set this
    key AT ALL — the record's claim that it does is itself the disagreement (see `compare_records`)."""

    key: str
    effective: str | None
    claimed: str
    record: str


def capture_env(environ, namespaces=NAMESPACES) -> dict:
    """The training-relevant subset of `environ`, as data.

    Captured from the live process rather than typed by hand: a record that is written cannot drift
    from the run that wrote it."""
    return {k: str(v) for k, v in sorted(environ.items()) if k.startswith(namespaces)}


def apply_overrides(base: dict, overrides: dict):
    """Return (new_env, changes). Setting a key to the value it already holds is NOT a change."""
    new = dict(base)
    changes = []
    for k, v in sorted(overrides.items()):
        v = str(v)
        before = base.get(k)
        if before == v:
            continue
        new[k] = v
        changes.append(Change(k, before, v))
    return new, changes


def format_diff(changes) -> str:
    """The diff is the first thing printed by a relaunch, so it is the one thing guaranteed read."""
    if not changes:
        return "  (no change)"
    w = max(len(c.key) for c in changes)
    return "\n".join(
        f"  {c.key:<{w}}  {'(unset)' if c.before is None else c.before} -> {c.after}"
        for c in changes)


def require_change(changes) -> None:
    if not changes:
        raise EmptyDiff(
            "relaunch would change nothing. Pass --set K=V to change a variable, or use the "
            "resume path if you meant to continue this lineage.")


def require_known(overrides: dict, readable) -> None:
    """`readable` = the variables the target env actually consults, discovered by import (see
    bridle.adapters.preflight.readable_env), never a hardcoded list."""
    unknown = sorted(set(overrides) - set(readable))
    if unknown:
        near = sorted(readable)
        raise UnknownOverride(
            f"the target env never reads: {', '.join(unknown)}. It reads: {', '.join(near)}. "
            "Setting a variable nothing consults produces a run that silently ignores your change.")


def _assignments(lines):
    """(key, value) for every `Environment=K=V`, `export K=V` or bare `K=V` line, in order.

    ⚠ DOES NOT IMPLEMENT THE FULL `Environment=` GRAMMAR — two documented gaps, neither exercised
    by any unit in this repo today:
    (1) Multiple space-separated assignments on one `Environment=` line (systemd's
        `Environment=FOO=bar BAZ=qux` sets both FOO and BAZ) are NOT split apart: the whole
        remainder after the first `=` is taken as one value, so `Environment=FOO=bar BAZ=qux`
        yields `{'FOO': 'bar BAZ=qux'}` and BAZ is silently lost.
    (2) A bare `Environment=` line with nothing after it is systemd's documented reset — "clear
        all prior assignments" — but here it strips to `k == ""`, fails `k.isidentifier()`, and is
        just dropped: it is treated as a no-op, not a clear, so assignments from earlier sources in
        `resolve_env` are NOT cleared the way systemd would clear them.
    Do not assume either case is handled without checking here first."""
    out = []
    for raw in lines:
        s = raw.strip()
        if s.startswith("Environment="):
            s = s[len("Environment="):]
        elif s.startswith("export "):
            s = s[len("export "):]
        elif not s or s.startswith("#") or "=" not in s:
            continue
        k, _, v = s.partition("=")
        k, v = k.strip(), v.strip()
        if k and k.isidentifier():
            out.append((k, v))
    return out


def resolve_env(sources):
    """The environment a set of ordered sources actually resolves to. `sources` is
    [(name, lines), ...] in the order the system applies them — for systemd, the base unit followed
    by its drop-ins in lexical order. Within a source, the last assignment wins.

    ⚠ REPEATED ASSIGNMENT IS NOT A DEFECT, and an earlier draft of this module got that wrong.
    A drop-in overriding the base unit is the drop-in mechanism working as designed, and
    `playground-coord.service.d/deploy-widegrab.conf` deliberately assigns GRAB_COORD_REFRESH_R and
    DINO_GRAB_CORNER_COORD twice, because it is written as a chronological log in which a later line
    reverts an earlier decision with the measurement preserved inline. A checker that flagged either
    would be reporting correct code as broken, and a checker that fires on correct code gets ignored.
    Resolve first; compare records second."""
    out = {}
    for _name, lines in sources:
        for k, v in _assignments(lines):
            out[k] = v
    return out


def compare_records(effective: dict, claimed: dict, record: str, prefix: str = ""):
    """Where a record DISAGREES with the effective environment. Silence is the normal case.

    A record that does not mention a key claims nothing about it and is not a violation — this is
    a check for contradiction, not for completeness. `prefix` scopes what the record is taken to be
    describing (e.g. "COORD_CKPT_" for scripts/_pgenv.sh, which mirrors only the checkpoint pins).

    A key the record DOES claim, that the live env does not set AT ALL, is also a disagreement —
    `effective=None` in that Mismatch — not a silent pass: "this record's claim is unverifiable" is
    not the same fact as "this record's claim is true", and collapsing them would let a record
    describe a variable the live process never even touches without ever being told so.

    THE LIVE CASE (2026-08-12): scripts/_pgenv.sh says in its own header that it mirrors
    playground-coord.service, and carries COORD_CKPT_grab=grab-coordv2-seed20/ckpt_GOOD_0p78.pt
    while the service effectively loads grab-coord-wide-disp4-seed20/final_ckpt.pt. Every offline
    test run through _pgenv.sh was loading a different grab policy than the live service."""
    out = []
    for k, v in sorted(claimed.items()):
        if not k.startswith(prefix):
            continue
        if k not in effective:
            out.append(Mismatch(k, None, v, record))
        elif effective[k] != v:
            out.append(Mismatch(k, effective[k], v, record))
    return out


@dataclass(frozen=True)
class CkptPinViolation:
    """A `COORD_CKPT_<prim>` the live env pins that no app manifest agrees with.

    `kind="mismatch"`: some manifest names a DIFFERENT file under the SAME `runs/<lineage>/`
    directory as the one actually loaded — a stale ckpt selection within a lineage everyone agrees
    is the right one.
    `kind="unregistered"`: no manifest names this file, under this lineage or any other — nothing
    in the store documents what is actually running."""

    key: str
    effective: str
    kind: str
    note: str = ""


def check_ckpt_pins(effective: dict, manifest_ckpts, prefix: str = "COORD_CKPT_") -> list:
    """Where a checkpoint the live env pins is not agreed to by any app manifest.

    `manifest_ckpts` is `[(app_name, absolute_ckpt_path), ...]` — every app's resolved `ckpt:`
    field. Gathering that list means walking the store's files, which is I/O this stdlib-only
    module does not do; the caller (bridle.cli) collects it and hands it in, the same split as
    `resolve_env`/`compare_records` above.

    THE LIVE CASE (2026-08-12/13, `bridle lineage`'s own step-3 comment claimed this check existed
    and it did not): the resolved service loads
    `grab/runs/grab-coord-wide-disp4-seed20/final_ckpt.pt`. Production app `grab_coord_v2` still
    names `grab/runs/grab-coordv2-seed20/ckpt_GOOD_0p78.pt` — an entirely DIFFERENT lineage
    directory, not a same-lineage typo — so this is UNREGISTERED: the one manifest that claims to
    document "the" grab checkpoint disagrees with what the service actually loads, and no manifest
    at all documents the checkpoint that is actually running."""
    out = []
    for key, live in sorted(effective.items()):
        if not key.startswith(prefix):
            continue
        live = os.path.normpath(live)
        live_dir = os.path.basename(os.path.dirname(live))
        if any(os.path.normpath(p) == live for _, p in manifest_ckpts):
            continue  # some manifest names exactly this file — agreement
        same_lineage = [(n, p) for n, p in manifest_ckpts
                        if os.path.basename(os.path.dirname(os.path.normpath(p))) == live_dir]
        if same_lineage:
            for n, p in same_lineage:
                out.append(CkptPinViolation(
                    key, live, "mismatch",
                    f"app {n!r} names {os.path.basename(p)} under the same runs/{live_dir}/"))
        else:
            out.append(CkptPinViolation(key, live, "unregistered", "no app manifest names it"))
    return out
