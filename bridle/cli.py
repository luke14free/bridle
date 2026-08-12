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


def main(argv=None):
    ap = argparse.ArgumentParser(prog="bridle", description=__doc__.split("\n")[0])
    ap.add_argument("--store", default="~/.bridle/apps")
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

    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
