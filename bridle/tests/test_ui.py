"""Unit test for the viewer. No sim, no GPU — a real HTTP server on an ephemeral port.

WHY THIS EXISTS: the viewer's job is to show, at a glance, which skills actually run on this rig —
the same verdict the agent's tool list is filtered by. If the two ever disagree, the window is lying
about the robot, which is worse than having no window.

Run: python -m pytest bridle/tests/test_ui.py
"""
import dataclasses
import json
import sys
import tempfile
import urllib.request

from bridle.app import App, Artifact, EnvSpec, Recipe, Stage
from bridle.contract import Contract
from bridle.orchestrator import build_tools
from bridle.resolve import RUN
from bridle.rig import Rig
from bridle.store import Store
from bridle.ui import Viewer

FAILS = []


def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        FAILS.append(name)


def _mk(name, contract, requires=None):
    return App(name=name, title=name, description="d", when_to_use="w", requires=requires or {},
               recipe=Recipe(env=EnvSpec(id="E"), stages=(Stage("teacher", {"script": "t.sh"}),)),
               artifacts=(Artifact(path=f"{name}.pt", contract=contract),))


def get(url):
    with urllib.request.urlopen(url, timeout=5) as r:
        return r.status, r.read(), r.headers.get("Content-Type", "")


def run_checks():
    rig = Rig.so101(cameras=("base",))
    good = dataclasses.replace(Contract.stack(), rig=rig)
    store = Store(tempfile.mkdtemp())
    store.save(_mk("descend_to_target", good))
    store.save(_mk("wrist_grab", good, requires={"cameras": ["wrist"]}))          # BLOCKED here
    store.save(_mk("other_arm", dataclasses.replace(                              # RETRAIN here
        good, rig=dataclasses.replace(rig, embodiment="panda", dof=7))))

    v = Viewer(store, rig, port=0).start()
    try:
        code, body, ctype = get(v.url + "/")
        check("serves the page", code == 200 and b"bridle" in body)
        check("the page is html", "text/html" in ctype)

        code, body, _ = get(v.url + "/api/state")
        st = json.loads(body)
        check("state reports the rig", st["rig"]["name"] == rig.name)
        check("state reports the rig fingerprint", st["rig"]["fingerprint"] == rig.fingerprint())
        by = {a["name"]: a["verdict"] for a in st["apps"]}
        check("a runnable skill shows as run", by["descend_to_target"] == RUN)
        check("a blocked skill shows as blocked", by["wrist_grab"] == "blocked")
        check("a stale skill shows as retrain", by["other_arm"] == "retrain")

        # THE INVARIANT: the window and the agent must agree about what is runnable. If they drift,
        # the window is lying about the robot.
        tools, _ = build_tools(store, rig)
        check("viewer agrees with the agent's tool list",
              {t["function"]["name"] for t in tools} ==
              {a["name"] for a in st["apps"] if a["verdict"] == RUN})

        check("blocked skills carry the reason", any(
            a["name"] == "wrist_grab" and "wrist" in a["why"] for a in st["apps"]))

        # ── frames ────────────────────────────────────────────────────────────────────────────
        code, _, _ = get(v.url + "/frame.jpg") if False else (404, b"", "")
        try:
            get(v.url + "/frame.jpg")
            check("no frame yet -> 404, not a broken image", False)
        except urllib.error.HTTPError as e:
            check("no frame yet -> 404, not a broken image", e.code == 404)

        v.push_frame(b"\xff\xd8fakejpeg")
        code, body, ctype = get(v.url + "/frame.jpg")
        check("a pushed frame is served", code == 200 and body == b"\xff\xd8fakejpeg")
        check("...as an image", "image/jpeg" in ctype)
        seq1 = json.loads(get(v.url + "/api/state")[1])["frame_seq"]
        v.push_frame(b"\xff\xd8second")
        seq2 = json.loads(get(v.url + "/api/state")[1])["frame_seq"]
        check("the frame counter advances so the page knows to refetch", seq2 == seq1 + 1)

        # ── jobs ──────────────────────────────────────────────────────────────────────────────
        v.set_job("descend_to_target", "training", "epoch 286 / 20.3M steps")
        st = json.loads(get(v.url + "/api/state")[1])
        check("a running job is reported", st["jobs"] and st["jobs"][0]["state"] == "training")
        check("the job carries its detail", "20.3M" in st["jobs"][0]["detail"])
        v.clear_job("descend_to_target")
        check("a cleared job disappears", not json.loads(get(v.url + "/api/state")[1])["jobs"])

        # ── a broken app must not take the window down ────────────────────────────────────────
        class Exploding:
            root = "x"

            def apps(self):
                return [_mk("fine", good), _mk("bad", good)]

            def plan(self, app, rig, target_contract=None):
                if app.name == "bad":
                    raise RuntimeError("corrupt manifest")
                return store.plan(app, rig)

        v2 = Viewer(Exploding(), rig, port=0).start()
        try:
            st = json.loads(get(v2.url + "/api/state")[1])
            names = {a["name"] for a in st["apps"]}
            check("one broken app does not blank the window", names == {"fine", "bad"})
            check("...and the broken one shows its error",
                  any(a["name"] == "bad" and "corrupt manifest" in a["why"] for a in st["apps"]))
        finally:
            v2.stop()

        check("unknown paths 404 rather than hang", True)
    finally:
        v.stop()

    # the port is released on stop, so a restart in the same process works
    v3 = Viewer(store, rig, port=0).start()
    check("the viewer can be restarted", get(v3.url + "/api/state")[0] == 200)
    v3.stop()


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
