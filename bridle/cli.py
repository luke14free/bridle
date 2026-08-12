"""`bridle` — the out-of-the-box entry point.

    bridle tui --model local:qwen3-32b        # agent TUI + simulator window
    bridle skills                             # what runs on this rig, and what doesn't
    bridle plan descend_to_target             # why a skill needs adapting or rebuilding

The TUI and the viewer come up together: the terminal is where you talk to the agent, the browser
window is where you watch the robot. Neither replaces the other.

EXECUTORS. bridle knows which skills are valid; it does not know how to drive YOUR simulator. Without
`--executor mod:func` the TUI runs in DRY mode, where a skill call is reported but nothing moves.
That is the honest default: silently doing nothing while looking like it worked is the failure this
whole project exists to prevent.
"""
import argparse
import importlib
import os
import sys

from bridle.agent import AgentSession
from bridle.llm import PRESETS, from_spec
from bridle.resolve import RUN
from bridle.rig import Rig
from bridle.store import Store


def _load(path):
    """`package.module:function` -> the callable."""
    mod, _, fn = path.partition(":")
    if not fn:
        raise SystemExit(f"--executor wants 'module:function', got {path!r}")
    return getattr(importlib.import_module(mod), fn)


def dry_executor(name, args):
    return True, f"[dry] would run {name}({args}) — no executor configured, nothing moved"


def cmd_skills(a):
    store, rig = Store(a.store), Rig.so101(cameras=tuple(a.cameras.split(",")))
    rows = []
    for app in store.apps():
        try:
            p = store.plan(app, rig)
            rows.append((p.action, app.name, p.reason if p.action != "blocked"
                         else ", ".join(p.blockers)))
        except Exception as e:
            rows.append(("error", app.name, f"{type(e).__name__}: {e}"))
    order = {RUN: 0, "adapt": 1, "retrain": 2, "blocked": 3, "error": 4}
    print(f"rig: {rig.describe()}\n")
    for action, name, why in sorted(rows, key=lambda r: (order.get(r[0], 9), r[1])):
        print(f"  {action:8s} {name:26s} {why[:70]}")
    print(f"\n{sum(1 for r in rows if r[0] == RUN)}/{len(rows)} skills run on this rig")
    return 0


def cmd_plan(a):
    store, rig = Store(a.store), Rig.so101(cameras=tuple(a.cameras.split(",")))
    plan = store.plan(store.get(a.app), rig)
    print(plan.explain())
    return 0


def cmd_tui(a):
    from bridle import tui as tui_mod
    from bridle.ui import Viewer

    store = Store(a.store)
    rig = Rig.so101(cameras=tuple(a.cameras.split(",")))
    executor = _load(a.executor) if a.executor else dry_executor
    provider = from_spec(a.model, base_url=a.base_url)
    session = AgentSession(provider, store, rig, executor)

    viewer = None
    if not a.no_viewer:
        viewer = Viewer(store, rig, port=a.viewer_port).start()
    if not a.executor:
        session.events.put(type("E", (), {"kind": "status", "text":
            "DRY MODE — no --executor, so skills are reported but nothing moves", "data": {},
            "t": 0})())
    tui_mod.run(session, viewer_url=(viewer.url if viewer else None),
                model_label=a.model, models=(a.models.split(",") if a.models else [a.model]))
    if viewer:
        viewer.stop()
    return 0


def cmd_relaunch(a):
    """Reproduce a deployed lineage with named overrides, gated by preflight."""
    # bridle declares `dependencies = []`; Store._load_text already does the lazy `import yaml`
    # with a json fallback, so use it rather than adding a third YAML-reading path to a codebase
    # whose thesis is that two records of one fact is the disease.
    from bridle.store import Store
    from bridle.adapters.preflight import collect, readable_env
    from bridle.lineage import NAMESPACES, EmptyDiff, UnknownOverride, format_diff
    from bridle.preflight import DYNAMIC, evaluate, format_effective, format_failures
    from bridle.relaunch import build_plan, install_and_start, systemd_unit

    store = Store(a.store)
    manifest = store._load_text(
        open(os.path.join(os.path.expanduser(a.store), f"{a.app}.yaml")).read())
    pf_path = a.preflight or os.path.join(a.cwd, "primitives", a.app, "preflight.yaml")
    if os.path.isfile(pf_path):
        doc = store._load_text(open(pf_path).read()) or {}
    else:
        doc = {}
        print(f"warning: no preflight.yaml at {pf_path} — running kind asserts only")
    overrides = dict(kv.split("=", 1) for kv in a.set or [])
    try:
        plan = build_plan(manifest, doc, overrides, a.exp,
                          readable_env(a.module or doc.get("module", "")))
    except (EmptyDiff, UnknownOverride, ValueError) as e:
        # These three already carry an actionable message (lineage.py / relaunch.py write them for
        # a human to read). The callers of this command include an automated agent, for which a raw
        # traceback is not a contract — so refuse cleanly instead of letting it propagate, the same
        # way a preflight failure below already returns 1 with a clean message.
        print(f"refused: {e}")
        return 1

    print(f"\nrelaunch {plan.app} as {plan.exp}\n\nchange:")
    print(format_diff(plan.changes))
    print("\neffective asserts:")
    print(format_effective(plan.asserts))

    # Delete every namespaced variable that survived from the calling shell but is NOT part of this
    # plan, before updating with plan.env. Preflight's STATIC tier imports the target module
    # in-process and reads os.environ directly, but systemd_unit() below only emits `Environment=`
    # lines for plan.env — so a stray namespaced var left in the process environment would let
    # preflight validate a configuration that is not the one the launched unit actually gets.
    # Preflight is worthless if it measures a different config than the one that runs.
    for k in [k for k in os.environ if k.startswith(NAMESPACES) and k not in plan.env]:
        del os.environ[k]
    os.environ.update(plan.env)
    values = collect(plan.asserts, doc.get("env_id", ""), a.module or doc.get("module", ""),
                     a.warm_start, doc.get("eval_envs", 64), doc.get("eval_steps", 64))
    failures = evaluate(plan.asserts, values, from_scratch=a.from_scratch)
    if failures:
        print("\n" + format_failures(failures))
        return 1
    print("\npreflight OK")
    if a.dry_run:
        print("(dry run — pass --launch to start)")
        return 0
    handle = install_and_start(f"bridle-{plan.exp}",
                               systemd_unit(plan.exp, plan.launcher, plan.env, a.cwd))
    print(f"launched: systemd --user {handle}")
    return 0


def cmd_lineage(a):
    """Assert every record that claims to describe the deployed environment agrees with it.

    THE RULE IS AGREEMENT BETWEEN RECORDS, NOT TIDINESS WITHIN ONE. Repeated assignment is not
    reported: a drop-in overriding the base unit is the systemd mechanism working as designed, and
    playground-coord.service.d/deploy-widegrab.conf assigns GRAB_COORD_REFRESH_R and
    DINO_GRAB_CORNER_COORD twice on purpose, as a chronological log where a later line reverts an
    earlier decision with the measurement preserved inline. Flagging either would report correct
    code as broken, and a checker that fires on correct code gets ignored.

    What IS a violation (live on 2026-08-12): scripts/_pgenv.sh says in its header that it mirrors
    playground-coord.service, and carries the SHADOWED COORD_CKPT_grab — so every offline test run
    through it loaded a different grab policy than the live service.
    """
    import glob
    import subprocess

    from bridle.lineage import compare_records, resolve_env
    from bridle.store import Store

    store = Store(a.store)
    bad = []

    # 1. Resolve the service environment the way systemd does: `systemctl cat` already emits the
    #    base unit followed by its drop-ins in application order, so a single ordered pass is the
    #    same resolution systemd performs.
    #
    #    An unperformed check is a violation, not a pass: if this fails for ANY reason (systemctl
    #    missing from PATH, the service stopped/renamed/uninstalled), `effective` stays {} and
    #    steps 2-3 below cannot compare against a live env. Printing "0 violation(s)" in that case
    #    would be indistinguishable from a genuine clean run, so every such path prints a distinct
    #    UNVERIFIED line and counts toward the nonzero exit — "cannot verify" must never render as
    #    "verified".
    try:
        unit = subprocess.run(["systemctl", "--user", "cat", "playground-coord.service"],
                              capture_output=True, text=True)
    except FileNotFoundError as e:
        print(f"UNVERIFIED systemctl not found on PATH ({e}); cannot resolve the live env")
        bad.append("systemctl")
        effective = {}
    else:
        if unit.returncode != 0:
            print("UNVERIFIED playground-coord.service: `systemctl --user cat` exited "
                  f"{unit.returncode} ({unit.stderr.strip() or 'no stderr'}); cannot resolve the "
                  "live env")
            bad.append("playground-coord.service")
            effective = {}
        else:
            effective = resolve_env([("playground-coord.service", unit.stdout.splitlines())])

    # 2. Every record that claims to mirror it must agree. This must run — and be reported as
    #    UNVERIFIED if it can't — independent of whether step 1 succeeded: a missing _pgenv.sh is
    #    its own unperformed check even when the live env resolved fine.
    pgenv = os.path.join(a.cwd, "scripts/_pgenv.sh")
    if not os.path.isfile(pgenv):
        print(f"UNVERIFIED scripts/_pgenv.sh not found at {pgenv}; cannot check it against the "
              "live env")
        bad.append("scripts/_pgenv.sh")
    elif not effective:
        print("UNVERIFIED scripts/_pgenv.sh: live env did not resolve (see above); comparison "
              "skipped")
        bad.append("scripts/_pgenv.sh")
    else:
        claimed = resolve_env([(pgenv, open(pgenv).read().splitlines())])
        for m in compare_records(effective, claimed, "scripts/_pgenv.sh", prefix="COORD_CKPT_"):
            print(f"MISMATCH   {m.record}: {m.key} claims {m.claimed}, live env resolves to "
                  f"{m.effective}")
            bad.append(m.key)

    # 3. Every app's ckpt exists, and where the live env pins that primitive, they agree. One
    #    unparsable manifest, or one with a malformed `ckpt` field, must not abort the scan of
    #    every other manifest — it is reported as its own violation and the scan continues.
    for path in sorted(glob.glob(os.path.join(os.path.expanduser(a.store), "*.yaml"))):
        try:
            m = store._load_text(open(path).read()) or {}
            ck = m.get("ckpt")
            if not ck:
                continue
            if not isinstance(ck, str):
                raise TypeError(f"ckpt must be a string, got {type(ck).__name__}: {ck!r}")
            full = ck if os.path.isabs(ck) else os.path.join(a.cwd, ck)
        except Exception as e:
            print(f"UNVERIFIED {path}: {type(e).__name__}: {e}; manifest could not be checked")
            bad.append(path)
            continue
        if not os.path.isfile(full):
            print(f"MISSING    {m.get('name')}: ckpt does not exist: {ck}")
            bad.append(m.get("name"))

    print(f"\n{len(bad)} violation(s)")
    return 1 if bad else 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="bridle", description=__doc__.split("\n")[0])
    ap.add_argument("--store", default=os.environ.get(
        "BRIDLE_STORE", "/home/luca/lego-arm/composer/store/apps"))
    ap.add_argument("--cameras", default="base", help="comma-separated camera names on your rig")
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("tui", help="agent TUI + simulator window")
    t.add_argument("--model", default="local:qwen3-32b",
                   help=f"<preset>:<model>; presets: {', '.join(sorted(PRESETS))}")
    t.add_argument("--models", default="", help="comma-separated specs to cycle with ^N")
    t.add_argument("--base-url", default=None)
    t.add_argument("--executor", default=None, help="module:function that actually drives a skill")
    t.add_argument("--viewer-port", type=int, default=8799)
    t.add_argument("--no-viewer", action="store_true")
    t.set_defaults(fn=cmd_tui)

    s = sub.add_parser("skills", help="what runs on this rig, and what doesn't")
    s.set_defaults(fn=cmd_skills)

    p = sub.add_parser("plan", help="why a skill needs adapting or rebuilding")
    p.add_argument("app")
    p.set_defaults(fn=cmd_plan)

    r = sub.add_parser("relaunch", help="retrain a deployed lineage with one variable changed")
    r.add_argument("app")
    r.add_argument("--set", action="append", metavar="K=V",
                   help="override one variable; repeatable. An empty diff is refused.")
    r.add_argument("--exp", required=True, help="name for the new lineage (COORD_EXP/BRIDLE_EXP)")
    r.add_argument("--module", default=None, help="python module that registers the env")
    r.add_argument("--preflight", default=None, help="path to preflight.yaml")
    r.add_argument("--warm-start", default=None, help="ckpt to seed and to preflight against")
    r.add_argument("--from-scratch", action="store_true", help="skip asserts needing a warm start")
    r.add_argument("--cwd", default="/home/luca/lego-arm")
    r.add_argument("--dry-run", action="store_true")
    r.set_defaults(fn=cmd_relaunch)

    lg = sub.add_parser("lineage", help="check deployed records against the live environment")
    lg.add_argument("--cwd", default="/home/luca/lego-arm")
    lg.set_defaults(fn=cmd_lineage)

    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
