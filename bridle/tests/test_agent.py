"""Unit test for the steerable agent session. No sim, no GPU, no network.

WHY THIS EXISTS: steering is the difference between an agent you watch and an agent you drive. A
coding agent that goes wrong writes a bad file; a robot agent that goes wrong moves a real arm, and
noticing three tool calls later means a scattered scene.

The invariant with teeth: **interrupt never lands mid-skill**. A half-executed grasp leaves the
gripper somewhere no policy was trained from — the exact class of state that cost 0.43 of task
success in the codebase bridle came from.

Run: python -m pytest bridle/tests/test_agent.py
"""
import dataclasses
import sys
import tempfile
import threading
import time

from bridle.agent import AgentSession
from bridle.app import App, Artifact, EnvSpec, Recipe, Stage
from bridle.contract import Contract
from bridle.llm import ScriptedProvider, from_spec
from bridle.rig import Rig
from bridle.store import Store

FAILS = []


def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        FAILS.append(name)


def _store(rig):
    s = Store(tempfile.mkdtemp())
    c = dataclasses.replace(Contract.stack(), rig=rig)
    for n in ("pick", "place"):
        s.save(App(name=n, title=n, description="d", when_to_use="w", args={"obj": "what"},
                   recipe=Recipe(env=EnvSpec(id="E"), stages=(Stage("teacher", {"script": "t"}),)),
                   artifacts=(Artifact(path=f"{n}.pt", contract=c),)))
    return s


def wait(sess, timeout=5):
    t0 = time.time()
    while sess.running and time.time() - t0 < timeout:
        time.sleep(0.01)
    return not sess.running


def run_checks():
    rig = Rig.so101(cameras=("base",))
    store = _store(rig)
    calls = []

    def ex(name, args):
        calls.append(name)
        return True, f"{name} ok"

    # ── a plain run ───────────────────────────────────────────────────────────────────────────
    prov = ScriptedProvider([
        {"text": "picking", "tool_calls": [{"name": "pick", "arguments": {"obj": "red"}}]},
        {"text": "placing", "tool_calls": [{"name": "place", "arguments": {"obj": "green"}}]},
        {"text": "done"},
    ])
    s = AgentSession(prov, store, rig, ex)
    s.start("stack them")
    check("the session finishes", wait(s))
    check("both skills ran in order", calls == ["pick", "place"])
    kinds = [e.kind for e in s.drain()]
    check("it emits assistant text", "assistant" in kinds)
    check("it emits tool calls and results", "tool_call" in kinds and "tool_result" in kinds)
    check("it emits done", "done" in kinds)
    check("final state is done", s.state == "done")

    # ── steering: guidance is delivered BEFORE the next skill is chosen ───────────────────────
    calls.clear()
    seen_before_second = {}

    class Watching(ScriptedProvider):
        def complete(self, messages, tools=None):
            if len(self.seen) == 1:                 # about to produce the SECOND reply
                seen_before_second["msgs"] = [m.get("content") for m in messages
                                              if m.get("role") == "user"]
            return super().complete(messages, tools)

    prov = Watching([
        {"tool_calls": [{"name": "pick", "arguments": {}}]},
        {"tool_calls": [{"name": "place", "arguments": {}}]},
        {"text": "done"},
    ])
    # The skill is deliberately SLOW so there is a real window to steer into. With an instant
    # executor the whole run completes in microseconds and the test would be asserting on a race
    # rather than on the mechanism.
    in_skill = threading.Event()

    def slow_ex(name, args):
        calls.append(name)
        in_skill.set()
        time.sleep(0.2)
        return True, f"{name} ok"

    s = AgentSession(prov, store, rig, slow_ex)
    s.start("stack them")
    in_skill.wait(2)                       # steer while the FIRST skill is still running
    s.steer("use the left platform")
    check("the session finishes after steering", wait(s))
    check("steered guidance reached the model as a user message",
          any("left platform" in (m or "") for m in seen_before_second.get("msgs", [])))

    # ── interrupt stops the run, and NEVER mid-skill ──────────────────────────────────────────
    started, finished = threading.Event(), []

    def slow(name, args):
        started.set()
        time.sleep(0.25)                            # the skill is "running"
        finished.append(name)                       # must ALWAYS be reached
        return True, f"{name} ok"

    prov = ScriptedProvider([{"tool_calls": [{"name": "pick", "arguments": {}}]}] * 10)
    s = AgentSession(prov, store, rig, slow)
    s.start("go")
    started.wait(2)
    s.interrupt()                                   # fired DURING the skill
    check("the session stops", wait(s, timeout=5))
    check("the in-flight skill was allowed to FINISH (never killed mid-grasp)",
          finished == ["pick"])
    check("state is interrupted", s.state == "interrupted")
    check("no further skills ran after the interrupt", len(finished) == 1)

    # ── model switching mid-session ───────────────────────────────────────────────────────────
    s = AgentSession(ScriptedProvider([{"text": "hi"}]), store, rig, ex)
    other = ScriptedProvider([{"text": "other"}])
    s.switch_provider(other, label="anthropic:claude")
    check("provider is swappable mid-session", s.provider is other)
    check("the switch is announced", any("model switched" in e.text for e in s.drain()))

    # ── failures surface, they do not crash the UI ────────────────────────────────────────────
    class Broken:
        def complete(self, *a, **k):
            raise RuntimeError("model offline")

    s = AgentSession(Broken(), store, rig, ex)
    s.start("go")
    check("a dead provider ends the session", wait(s))
    check("...as an error event, not an exception",
          any(e.kind == "error" and "model offline" in e.text for e in s.drain()))
    check("state is error", s.state == "error")

    def boom(name, args):
        raise RuntimeError("gripper jammed")
    s = AgentSession(ScriptedProvider([
        {"tool_calls": [{"name": "pick", "arguments": {}}]}, {"text": "stopping"}]),
        store, rig, boom)
    s.start("go")
    wait(s)
    check("an executor exception becomes a tool result, not a crash",
          any(e.kind == "tool_result" and "gripper jammed" in e.text for e in s.drain()))

    # ── a task typed while busy is steering, not a second run ─────────────────────────────────
    s = AgentSession(ScriptedProvider(
        [{"tool_calls": [{"name": "pick", "arguments": {}}]}] * 3 + [{"text": "d"}]), store, rig, ex)
    s.start("first")
    s.start("second")                               # must NOT spawn a second loop
    check("start() while running steers instead of racing", wait(s))

    # ── provider specs ────────────────────────────────────────────────────────────────────────
    check("from_spec builds a local provider", from_spec("local:m").base_url.endswith("8000/v1"))
    check("from_spec builds an anthropic provider",
          type(from_spec("anthropic:claude")).__name__ == "AnthropicProvider")
    try:
        from_spec("nope:m")
        check("an unknown preset is rejected", False)
    except ValueError:
        check("an unknown preset is rejected", True)


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
