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
    import shutil

    from bridle.store import Store
    from bridle.adapters.preflight import collect, readable_env
    from bridle.lineage import NAMESPACES, EmptyDiff, UnknownOverride, format_diff
    from bridle.preflight import DYNAMIC, evaluate, format_effective, format_failures
    from bridle.relaunch import build_plan, install_and_start, run_dir_for, systemd_unit

    store = Store(a.store)
    manifest_path = os.path.join(os.path.expanduser(a.store), f"{a.app}.yaml")
    try:
        with open(manifest_path) as f:
            manifest = store._load_text(f.read())
    except FileNotFoundError:
        print(f"refused: no app manifest at {manifest_path}")
        return 1
    if not manifest:
        print(f"refused: {manifest_path} parsed to nothing (empty or malformed)")
        return 1

    bad_sets = [kv for kv in (a.set or []) if "=" not in kv]
    if bad_sets:
        print(f"refused: --set wants K=V, got: {', '.join(bad_sets)!r}")
        return 1
    overrides = dict(kv.split("=", 1) for kv in a.set or [])

    # I6: relaunch never told the operator when the inherited env came from a hand-transcribed
    # record instead of a captured process. capture_training() tags a real capture "source:
    # captured"; a manifest built by hand before that existed (or backfilled after the fact, like
    # place_coord_v3) says "source: backfilled" and was never verified against a live process.
    if (manifest.get("training") or {}).get("source") == "backfilled":
        print(f"warning: {a.app}'s training record is BACKFILLED — transcribed by hand from prose/"
              "journal notes, not captured from a live process, and never independently verified. "
              "Treat the inherited env with extra scrutiny.")

    # C1(a): resolve the preflight path from the MANIFEST, not a guess off the app name. Apps are
    # not named after the primitive they train (`place_coord_v3` trains `descend_to_target`), so
    # `primitives/<app>/preflight.yaml` found nothing for any real app and silently ran with zero
    # asserts. Priority: an explicit --preflight override, then the manifest's own `preflight:` key,
    # then the app-name guess as a last resort (for an app that has genuinely never had one authored).
    manifest_pf = manifest.get("preflight")
    if a.preflight:
        pf_path, pf_source = a.preflight, "--preflight"
    elif manifest_pf:
        pf_path = manifest_pf if os.path.isabs(manifest_pf) else os.path.join(a.cwd, manifest_pf)
        pf_source = f"{a.app}.yaml `preflight:`"
    else:
        pf_path = os.path.join(a.cwd, "primitives", a.app, "preflight.yaml")
        pf_source = "guessed from the app name (manifest has no `preflight:` key)"
    if os.path.isfile(pf_path):
        doc = store._load_text(open(pf_path).read()) or {}
    else:
        doc = {}
        print(f"warning: no preflight.yaml at {pf_path} ({pf_source}) — running kind asserts only")

    module = a.module or doc.get("module", "")
    try:
        plan = build_plan(manifest, doc, overrides, a.exp, readable_env(module))
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

    # C1(b): zero asserts must NEVER render as "preflight OK" — to an automated caller that is
    # indistinguishable from six asserts having actually been checked and passed. Refuse unless the
    # operator explicitly says a vacuous preflight is acceptable.
    if not plan.asserts:
        msg = f"PREFLIGHT VACUOUS (0 asserts) — looked at {pf_path} ({pf_source})"
        if not a.allow_vacuous_preflight:
            print(f"\n{msg}")
            return 1
        print(f"\n{msg} — proceeding anyway (--allow-vacuous-preflight)")

    # I3: refuse to write into an existing lineage's run directory. Every launcher
    # (train_coord_prim.sh, teacher_train.sh, ...) resumes from whichever ckpt sorts highest by
    # number in runs/<exp>/ — starting a DIFFERENT relaunch into that same directory is checkpoint
    # contamination, not a fresh run, and is exactly the class of mistake this command exists to end.
    run_dir = run_dir_for(module, plan.exp, a.cwd)
    if run_dir is None:
        print(f"refused: could not derive the run directory from module {module!r} (expected "
              "'primitives.<name>...') — refusing rather than silently skipping the "
              "non-empty-run-directory check")
        return 1
    if os.path.isdir(run_dir) and os.listdir(run_dir) and not a.resume:
        print(f"refused: {run_dir} already exists and is non-empty. A fresh relaunch would resume "
              "from whatever ckpt sorts highest in there — pass --resume if that is intended, or "
              "choose a new --exp.")
        return 1

    # Delete every namespaced variable that survived from the calling shell but is NOT part of this
    # plan, before updating with plan.env. Preflight's STATIC tier imports the target module
    # in-process and reads os.environ directly, but systemd_unit() below only emits `Environment=`
    # lines for plan.env — so a stray namespaced var left in the process environment would let
    # preflight validate a configuration that is not the one the launched unit actually gets.
    # Preflight is worthless if it measures a different config than the one that runs.
    for k in [k for k in os.environ if k.startswith(NAMESPACES) and k not in plan.env]:
        del os.environ[k]
    os.environ.update(plan.env)
    values = collect(plan.asserts, doc.get("env_id", ""), module, a.warm_start,
                     doc.get("eval_envs", 64), doc.get("eval_steps", 64),
                     from_scratch=a.from_scratch)
    failures = evaluate(plan.asserts, values, from_scratch=a.from_scratch)
    if failures:
        print("\n" + format_failures(failures))
        return 1
    print("\npreflight OK")

    unit_text = systemd_unit(plan.exp, plan.launcher, plan.env, a.cwd)
    if a.dry_run:
        print("\n--dry-run: the systemd unit that would be written:\n")
        print(unit_text)
        print("(dry run — omit --dry-run to start it)")
        return 0

    # I2: seed the warm-start ckpt so the launcher's own resume logic (highest-numbered ckpt in
    # runs/<exp>/) actually picks it up. Without this, --warm-start's help text ("ckpt to seed and
    # to preflight against") was half true: preflight measured the warm policy, but the unit still
    # trained from scratch.
    if a.warm_start:
        os.makedirs(run_dir, exist_ok=True)
        seeded = os.path.join(run_dir, "ckpt_1.pt")
        shutil.copyfile(a.warm_start, seeded)
        print(f"seeded {seeded} from --warm-start {a.warm_start}")

    handle = install_and_start(f"bridle-{plan.exp}", unit_text)
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

    from bridle.lineage import check_ckpt_pins, compare_records, resolve_env
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
            eff_repr = m.effective if m.effective is not None else "(not set at all by the live env)"
            print(f"MISMATCH   {m.record}: {m.key} claims {m.claimed}, live env resolves to "
                  f"{eff_repr}")
            bad.append(m.key)

    # 3. Every app's ckpt exists, and where the live env pins that primitive, they agree. One
    #    unparsable manifest, or one with a malformed `ckpt` field, must not abort the scan of
    #    every other manifest — it is reported as its own violation and the scan continues.
    manifest_ckpts = []  # [(app_name, absolute_ckpt_path), ...], fed to check_ckpt_pins below
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
        manifest_ckpts.append((m.get("name"), full))

    # 4. Where the live env pins a primitive's checkpoint (COORD_CKPT_<prim>), some manifest must
    #    agree — this is the check step 3's own comment has always claimed and, until now, never
    #    ran (see bridle.lineage.check_ckpt_pins for the live 2026-08-12/13 grab defect this catches).
    for v in check_ckpt_pins(effective, manifest_ckpts):
        label = "MISMATCH" if v.kind == "mismatch" else "UNREGISTERED"
        print(f"{label:<10} {v.key}: live env resolves to {v.effective} — {v.note}")
        bad.append(f"{v.key}:{v.kind}")

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
    r.add_argument("--preflight", default=None,
                   help="path to preflight.yaml; overrides the manifest's own `preflight:` key")
    r.add_argument("--warm-start", default=None, help="ckpt to seed and to preflight against")
    r.add_argument("--from-scratch", action="store_true", help="skip asserts needing a warm start")
    r.add_argument("--allow-vacuous-preflight", action="store_true",
                   help="proceed even though the resolved preflight contributed zero asserts")
    r.add_argument("--resume", action="store_true",
                   help="allow writing into an existing, non-empty runs/<exp>/ directory")
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
