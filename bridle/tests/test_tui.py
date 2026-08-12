"""Unit test for the TUI's logic, against a fake curses screen. No TTY needed.

WHY A FAKE SCREEN: `curses.wrapper` needs a real terminal, which CI and this repo's test runner do
not have. Untested render code in the flagship UI is not acceptable, so the drawing is exercised
against a stub that records what would be written — which catches the failures that actually happen
(index errors at small sizes, writes past the edge, a pane that renders nothing).

Run: python -m pytest bridle/tests/test_tui.py
"""
import curses
import dataclasses
import sys
import tempfile

from bridle.agent import AgentSession
from bridle.app import App, Artifact, EnvSpec, Recipe, Stage
from bridle.contract import Contract
from bridle.llm import ScriptedProvider
from bridle.rig import Rig
from bridle.store import Store
from bridle.tui import TUI

FAILS = []


def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        FAILS.append(name)


class FakeScreen:
    """Records what would be drawn, and enforces the bounds curses would enforce."""

    def __init__(self, h=24, w=90):
        self.h, self.w = h, w
        self.writes = []

    def getmaxyx(self):
        return self.h, self.w

    def erase(self):
        self.writes.clear()

    def addnstr(self, y, x, s, n, attr=0):
        if not (0 <= y < self.h and 0 <= x < self.w):
            raise curses.error(f"addnstr out of bounds at ({y},{x}) on {self.h}x{self.w}")
        self.writes.append((y, x, s[:max(0, n)]))

    def addch(self, y, x, ch, attr=0):
        if not (0 <= y < self.h and 0 <= x < self.w):
            raise curses.error(f"addch out of bounds at ({y},{x})")

    def move(self, y, x):
        if not (0 <= y < self.h and 0 <= x < self.w):
            raise curses.error("move out of bounds")

    def refresh(self):
        pass

    def text(self):
        return "\n".join(s for _, _, s in self.writes)


def _store(rig):
    s = Store(tempfile.mkdtemp())
    c = dataclasses.replace(Contract.stack(), rig=rig)
    mk = lambda n, req=None: App(
        name=n, title=n, description="d", when_to_use="w", args={"obj": "what"}, requires=req or {},
        recipe=Recipe(env=EnvSpec(id="E"), stages=(Stage("teacher", {"script": "t"}),)),
        artifacts=(Artifact(path=f"{n}.pt", contract=c),))
    s.save(mk("pick"))
    s.save(mk("place"))
    s.save(mk("wrist_grab", {"cameras": ["wrist"]}))          # BLOCKED on a base-only rig
    return s


def _tui(rig, store):
    sess = AgentSession(ScriptedProvider([{"text": "hi"}]), store, rig, lambda n, a: (True, "ok"))
    return TUI(sess, viewer_url="http://127.0.0.1:8799", model_label="local:qwen3-32b",
               models=["local:qwen3-32b", "anthropic:claude-sonnet-4"])


def run_checks():
    rig = Rig.so101(cameras=("base",))
    store = _store(rig)
    t = _tui(rig, store)
    t.lines = [("status", "3 skills available"), ("assistant", "I will pick the red cube."),
               ("tool_call", "pick(obj='red')"), ("tool_result", "picked (14.2 N)"),
               ("error", "could not reach the target")]

    scr = FakeScreen()
    t.draw(scr)
    body = scr.text()
    check("the header names the rig", rig.name in body)
    check("the header names the model", "local:qwen3-32b" in body)
    check("the conversation is rendered", "pick the red cube" in body)
    check("tool calls are rendered", "pick(obj=" in body)
    check("the skills pane lists a runnable skill", "pick" in body and "ready" in body)
    check("a blocked skill is shown as unavailable", "rig can't run it" in body)
    check("the viewer url is offered", "8799" in body)
    check("keybindings are shown", "esc interrupt" in body)

    # ── the pane must survive hostile geometry ────────────────────────────────────────────────
    for h, w in [(24, 90), (10, 40), (60, 200), (8, 30), (5, 20)]:
        try:
            t.draw(FakeScreen(h, w))
            check(f"renders at {h}x{w} without writing out of bounds", True)
        except curses.error as e:
            check(f"renders at {h}x{w} without writing out of bounds ({e})", False)

    # ── input handling ────────────────────────────────────────────────────────────────────────
    t2 = _tui(rig, store)
    for c in "hello":
        t2.key(ord(c))
    check("typing accumulates", t2.input == "hello")
    t2.key(curses.KEY_BACKSPACE)
    check("backspace deletes", t2.input == "hell")
    t2.key(10)
    check("enter clears the input line", t2.input == "")
    check("enter starts or steers the session", any("hell" in txt for _, txt in t2.lines))

    check("^N cycles the model", (t2.key(14), t2.model_label)[1] == "anthropic:claude-sonnet-4")
    check("^C quits", t2.key(3) is False)
    check("other keys do not quit", t2.key(ord("x")) is not False)

    # ESC must ask the SESSION to stop, not kill anything itself — a half-executed grasp is worse
    # than either finishing or not starting.
    t3 = _tui(rig, store)
    t3.key(27)
    check("esc requests an interrupt on the session", t3.s._interrupt.is_set())

    # ── events become lines ───────────────────────────────────────────────────────────────────
    t4 = _tui(rig, store)
    t4.s._emit("assistant", "thinking about it")
    t4.s._emit("tool_call", "pick", args={"obj": "red"})
    t4.s._emit("tool_result", "picked", ok=True)
    t4.s._emit("tool_result", "dropped it", ok=False)
    t4.pump()
    kinds = [k for k, _ in t4.lines]
    check("assistant events become lines", "assistant" in kinds)
    check("tool calls render with their args", any("obj=" in txt for _, txt in t4.lines))
    check("a FAILED tool result is rendered as an error, not a success",
          kinds.count("error") == 1 and kinds.count("tool_result") == 1)

    # ── the skills pane agrees with the store ─────────────────────────────────────────────────
    verdicts = dict(t4.skills())
    check("the pane's verdicts come from plan()", verdicts["pick"] == "run"
          and verdicts["wrist_grab"] == "blocked")


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
