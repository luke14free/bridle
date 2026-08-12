"""`bridle relaunch <app> --set K=V` — retrain a deployed lineage with one variable changed.

THE OPERATION THIS PROJECT WAS MISSING. It is the most common training action here and the literal
meaning of its own "one variable per run" rule, and until 2026-08-12 it was performed by reading
four files and a June journal entry. Doing it by hand cost two runs in one afternoon.

The diff is the argument list: whatever you do not `--set` is inherited from the deployed lineage,
so the grip freeze that was forgotten cannot be forgotten again.

bridle does NOT import lego-arm to do this. It reads the app manifest as data and executes
`training.launcher` as a string, exactly as Foundry/ShellStageRunner already do.
"""
import os
import shlex
import subprocess
from dataclasses import dataclass

from bridle.lineage import apply_overrides, format_diff, require_change, require_known
from bridle.preflight import DYNAMIC, STATIC, Assert, merge


@dataclass(frozen=True)
class RelaunchPlan:
    app: str
    exp: str
    env: dict
    changes: list
    asserts: tuple
    launcher: str


def _asserts_from_doc(doc) -> tuple:
    """`preflight.yaml` as already-parsed dicts (the CLI parses; this stays yaml-free).

    `Assert.__post_init__` refuses to construct a boundless assert (no expect/min/max) — it raises
    a bare `ValueError` that names only the path, because that is all a dataclass knows. A malformed
    `preflight.yaml` entry (e.g. `descend_low_once:` with nothing after the colon, or `{}`) hits that
    exact path. Letting the raw `ValueError` propagate here would read as a crash inside `bridle`'s
    own code with no indication the actual defect is upstream, in a hand-authored YAML file — so it
    is caught and re-raised naming the tier and the fact that it is a `preflight.yaml` authoring
    error, with the original message (which does have the path) chained on via `from e`.
    """
    out = []
    for tier in (STATIC, DYNAMIC):
        for path, spec in (doc.get(tier) or {}).items():
            spec = spec if isinstance(spec, dict) else {"expect": spec}
            try:
                out.append(Assert(path, tier, min=spec.get("min"), max=spec.get("max"),
                                  expect=spec.get("expect"), needs=spec.get("needs")))
            except ValueError as e:
                raise ValueError(
                    f"preflight.yaml: `{tier}: {path}` gives no bound (no min/max/expect) — "
                    f"an assert with no bound can never fail, so this entry is malformed, not "
                    f"just strict. Fix the YAML. ({e})") from e
    return tuple(out)


def build_plan(manifest: dict, preflight_doc: dict, overrides: dict, exp: str,
               readable) -> RelaunchPlan:
    t = manifest.get("training")
    if not t:
        raise ValueError(
            f"app {manifest.get('name')!r} has no `training:` block, so its lineage cannot be "
            "reproduced. Register it with capture_training(), or backfill it.")
    require_known(overrides, readable)
    env, changes = apply_overrides(t["env"], overrides)
    require_change(changes)
    # The lineage name must move with the lineage: a changed run must never write into the
    # unchanged run's directory, which is how a contaminated ckpt gets resumed from. Both
    # COORD_EXP and BRIDLE_EXP are set UNCONDITIONALLY — not only when already present in the
    # captured env — and routed through the same apply_overrides() that produced `changes` above,
    # so the rename is reported in the diff instead of happening silently behind it. The asymmetry
    # is deliberate: setting a name variable the launcher does not read is harmless, while failing
    # to set the one it DOES read is exactly the contamination this module exists to prevent — e.g.
    # primitives/descend_to_target/teacher_train.sh reads only `EXP=${BRIDLE_EXP:-descend-teacher-
    # seed20}`, so a lineage that relied on that shell default never exported BRIDLE_EXP, and an
    # `if key in t["env"]` guard on it would leave the relaunch writing into the original run dir.
    env, name_changes = apply_overrides(env, {"COORD_EXP": exp, "BRIDLE_EXP": exp})
    changes = changes + name_changes
    asserts = merge(preflight_doc.get("kind"), _asserts_from_doc(preflight_doc))
    return RelaunchPlan(manifest["name"], exp, env, changes, asserts, t["launcher"])


def run_dir_for(module: str, exp: str, cwd: str):
    """`primitives/<PRIM>/runs/<exp>` — the directory every launcher (train_coord_prim.sh,
    teacher_train.sh, ...) writes checkpoints into and resumes from by picking whichever ckpt
    sorts highest by number.

    Derived from the preflight MODULE path (`primitives.<PRIM>.<file>`), never from the app name:
    apps are not named after the primitive they train (`place_coord_v3` trains `descend_to_target`),
    and guessing the run directory from `--app` is the exact C1-class mistake this module exists to
    not repeat. Returns None if `module` is not shaped like `primitives.<name>...` — the caller
    should refuse rather than silently skip whatever check needed this directory.
    """
    parts = module.split(".")
    if len(parts) < 2 or parts[0] != "primitives" or not parts[1]:
        return None
    return os.path.join(cwd, "primitives", parts[1], "runs", exp)


def systemd_unit(name: str, cmd: str, env: dict, cwd: str) -> str:
    """Long jobs are systemd --user units, never bare nohup (CLAUDE.md): reboot-resumable, and
    `systemctl --user status` is the one place to look."""
    lines = "\n".join(f"Environment={k}={v}" for k, v in sorted(env.items()))
    # shlex.quote, not a hand-written "'{cmd}'": a launcher containing a single quote (a training
    # arg with an embedded string, say) would otherwise close the ExecStart quoting early and hand
    # systemd a broken command line.
    quoted_cmd = shlex.quote(cmd)
    return f"""[Unit]
Description=bridle relaunch — {name}
After=network.target

[Service]
Type=simple
WorkingDirectory={cwd}
{lines}
ExecStart=/bin/bash -lc {quoted_cmd}
Restart=on-failure
RestartSec=30

[Install]
WantedBy=default.target
"""


def install_and_start(unit_name: str, text: str) -> str:
    path = os.path.expanduser(f"~/.config/systemd/user/{unit_name}.service")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    subprocess.run(["systemctl", "--user", "enable", "--now", f"{unit_name}.service"], check=True)
    return unit_name
