"""Unit test for the orchestrator. No sim, no GPU, no network — the LLM is scripted.

WHY THIS EXISTS: an LLM offered a tool will call it. The one thing this layer must guarantee is that
the tool list contains ONLY skills verified to run on this rig — otherwise the model confidently
invokes a policy trained for someone else's camera, and the failure surfaces as bad robot behaviour
instead of as a missing prerequisite.

Run: python -m pytest bridle/tests/test_orchestrator.py
"""
import dataclasses
import sys
import tempfile

from bridle.app import App, Artifact, EnvSpec, Recipe, Stage
from bridle.contract import Contract
from bridle.llm import ScriptedProvider
from bridle.orchestrator import Orchestrator, app_to_tool, build_tools
from bridle.rig import Rig
from bridle.store import Store

FAILS = []


def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        FAILS.append(name)


def _mk(name, contract, requires=None):
    return App(name=name, title=name, description=f"{name} does a thing",
               when_to_use=f"use {name} when appropriate",
               args={"target": "what to act on"}, requires=requires or {},
               recipe=Recipe(env=EnvSpec(id="E-v1"), stages=(Stage("teacher", {"script": "t.sh"}),)),
               artifacts=(Artifact(path=f"{name}.pt", contract=contract),))


def run_checks():
    rig = Rig.so101(cameras=("base",))
    good = dataclasses.replace(Contract.stack(), rig=rig)
    store = Store(tempfile.mkdtemp())

    runnable = _mk("descend_to_target", good)
    # trained on a DIFFERENT rig (absolute-position control) -> RETRAIN, so it must NOT be offered.
    # Note the subtlety this test originally got wrong: an app judged against its OWN contract is
    # self-consistent by construction and always RUNs. A skill becomes unavailable when the RIG
    # differs, or when the TASK demands a contract the skill was not trained for (see below).
    stale = _mk("place_high", dataclasses.replace(
        good, rig=dataclasses.replace(rig, control_mode="pd_joint_pos")))
    # needs a wrist camera this rig does not have -> BLOCKED
    blocked = _mk("wrist_grab", good, requires={"cameras": ["wrist"]})
    for a in (runnable, stale, blocked):
        store.save(a)

    tools, unavailable = build_tools(store, rig)
    names = {t["function"]["name"] for t in tools}
    check("a runnable skill is offered", "descend_to_target" in names)
    check("a BLOCKED skill is NOT offered", "wrist_grab" not in names)
    check("a RETRAIN skill is NOT offered", "place_high" not in names)
    check("unavailable skills are reported, not silently dropped",
          {n for n, _, _ in unavailable} == {"wrist_grab", "place_high"})
    check("the report says WHY", any("wrist" in w for n, a, w in unavailable if n == "wrist_grab"))

    # ── a TASK-specific contract can also make a skill unavailable ────────────────────────────
    # This is the stacking case: the skill is fine on this robot, but the task needs a release
    # height it was not trained for. Without this the model would call it and get 0/20.
    cube_top = dataclasses.replace(good, release=dataclasses.replace(
        good.release, height_above_resting=0.002))
    tools2, unavail2 = build_tools(store, rig, {"descend_to_target": cube_top})
    check("a task-specific contract can withdraw a skill",
          "descend_to_target" not in {t["function"]["name"] for t in tools2})
    check("...and reports it as needing a rebuild",
          any(n == "descend_to_target" and a == "retrain" for n, a, _ in unavail2))

    t = app_to_tool(runnable)
    check("the tool schema carries when_to_use", "When to use" in t["function"]["description"])
    check("the tool schema declares the app's args", "target" in t["function"]["parameters"]["properties"])

    # ── the loop drives a skill and reads its result ──────────────────────────────────────────
    ran = []

    def executor(name, args):
        ran.append((name, args))
        return True, f"{name} ok"

    prov = ScriptedProvider([
        {"tool_calls": [{"name": "descend_to_target", "arguments": {"target": "green cube"}}]},
        {"text": "The cube is placed."},
    ])
    sess = Orchestrator(prov, store, rig, executor).run("put the red cube on the green one")
    check("the skill was executed", ran == [("descend_to_target", {"target": "green cube"})])
    check("the session ends when the model stops calling tools", sess.done)
    check("the tool result is fed back to the model",
          any(m.get("role") == "tool" for m in prov.seen[-1]["messages"]))

    # ── a hallucinated / unavailable skill is refused WITH the reason ─────────────────────────
    ran.clear()
    prov = ScriptedProvider([
        {"tool_calls": [{"name": "wrist_grab", "arguments": {"target": "cube"}}]},
        {"text": "I cannot do that on this rig."},
    ])
    sess = Orchestrator(prov, store, rig, executor).run("grab with the wrist camera")
    check("an unavailable skill is never executed", ran == [])
    check("the refusal explains it is not available on this rig",
          "not available on this rig" in sess.turns[0].results[0][1])

    # ── an executor that raises must not kill the session ─────────────────────────────────────
    def boom(name, args):
        raise RuntimeError("gripper jammed")
    prov = ScriptedProvider([
        {"tool_calls": [{"name": "descend_to_target", "arguments": {"target": "x"}}]},
        {"text": "stopping"},
    ])
    sess = Orchestrator(prov, store, rig, boom).run("do it")
    check("an executor exception is reported, not raised",
          sess.turns[0].results[0][0] is False and "gripper jammed" in sess.turns[0].results[0][1])

    # ── a rig that can run nothing stops immediately, and says so ─────────────────────────────
    blind_store = Store(tempfile.mkdtemp())
    blind_store.save(_mk("wrist_only", good, requires={"cameras": ["wrist"]}))
    sess = Orchestrator(ScriptedProvider([]), blind_store, rig, executor).run("anything")
    check("no runnable skills -> stop before calling the LLM", not sess.turns)
    check("...and explain why", "no skill in the store runs on this rig" in sess.stopped_because)

    # ── the loop cannot spin forever ──────────────────────────────────────────────────────────
    spin = ScriptedProvider([{"tool_calls": [{"name": "descend_to_target", "arguments": {}}]}] * 50)
    sess = Orchestrator(spin, store, rig, executor, max_turns=3).run("loop")
    check("max_turns bounds the session", len(sess.turns) == 3 and "max_turns" in sess.stopped_because)


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
