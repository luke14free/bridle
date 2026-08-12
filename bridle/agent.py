"""A steerable agent session: the loop a TUI can watch, interrupt, and redirect mid-flight.

`Orchestrator.run()` is a blocking loop that returns when it is done — fine for a script, useless
for an interface. This is the same loop turned inside out: it runs on a thread, emits events as it
goes, and reads two channels the operator controls.

    sess = AgentSession(provider, store, rig, executor)
    sess.start("stack the red cube on the green one")
    sess.steer("no — use the left platform")      # queued, delivered at the next safe point
    sess.interrupt()                              # stop after the current skill returns
    for ev in sess.drain(): ...                   # assistant / tool_call / tool_result / done

WHY STEERING MATTERS MORE FOR A ROBOT THAN FOR CODE. A coding agent that goes wrong writes a bad
file and you undo it. A robot agent that goes wrong moves a real arm, and the cost of noticing three
tool calls later is a scattered scene or a dropped object. So the two operator channels are:

  steer()     add guidance; the agent keeps going, and sees it before choosing its next skill
  interrupt() stop cleanly — never mid-skill, because a half-executed grasp is a worse state than
              either finishing or not starting

INTERRUPT IS NOT A KILL. It is checked between turns and after a skill returns, never during one.
Killing a rollout halfway leaves the gripper wherever it happened to be, holding whatever it happened
to hold — a state no policy was trained from, which is exactly how this project once handed `grab` a
state it had never seen and lost 0.43 of task success.

stdlib only: threading and queue.
"""
import json
import queue
import threading
import time
from dataclasses import dataclass, field

from bridle.orchestrator import SYSTEM, build_tools


@dataclass
class Event:
    """Something the operator should see. `kind` drives how a UI renders it."""

    kind: str          # status | assistant | tool_call | tool_result | steer | error | done
    text: str = ""
    data: dict = field(default_factory=dict)
    t: float = field(default_factory=time.time)


class AgentSession:
    """An agent loop that can be watched, steered and stopped."""

    def __init__(self, provider, store, rig, executor, system=SYSTEM, max_turns=40,
                 target_contracts=None):
        self.provider = provider           # swappable mid-session — see switch_provider()
        self.store, self.rig, self.executor = store, rig, executor
        self.system, self.max_turns = system, max_turns
        self.target_contracts = target_contracts or {}
        self.messages = []
        self.events = queue.Queue()
        self._steer = queue.Queue()
        self._interrupt = threading.Event()
        self._thread = None
        self.state = "idle"                # idle | thinking | acting | done | error | interrupted
        self.tools, self.unavailable = [], []
        self.turn = 0

    # ── operator channels ─────────────────────────────────────────────────────────────────────
    def steer(self, text: str) -> None:
        """Add guidance. Delivered before the agent chooses its next skill, not mid-skill."""
        self._steer.put(text)
        self._emit("steer", text)

    def interrupt(self) -> None:
        """Stop cleanly at the next safe point. Never mid-skill."""
        self._interrupt.set()
        self._emit("status", "interrupt requested — stopping after the current skill")

    def switch_provider(self, provider, label="") -> None:
        """Change model mid-session. The conversation is provider-agnostic, so history carries over."""
        self.provider = provider
        self._emit("status", f"model switched{': ' + label if label else ''}")

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ── lifecycle ─────────────────────────────────────────────────────────────────────────────
    def start(self, task: str) -> None:
        if self.running:
            self.steer(task)               # a task typed while busy is guidance, not a new run
            return
        self._interrupt.clear()
        self.messages = [{"role": "system", "content": self.system},
                         {"role": "user", "content": task}]
        self._thread = threading.Thread(target=self._run, name="bridle-agent", daemon=True)
        self._thread.start()

    def drain(self):
        """All events emitted since the last call. Non-blocking, for a UI's render tick."""
        out = []
        while True:
            try:
                out.append(self.events.get_nowait())
            except queue.Empty:
                return out

    def _emit(self, kind, text="", **data):
        self.events.put(Event(kind, text, data))

    # ── the loop ──────────────────────────────────────────────────────────────────────────────
    def _run(self):
        try:
            self.tools, self.unavailable = build_tools(self.store, self.rig, self.target_contracts)
            if not self.tools:
                self.state = "error"
                self._emit("error", "no skill in the store runs on this rig — nothing to drive")
                self._emit("done")
                return
            self._emit("status", f"{len(self.tools)} skills available on {self.rig.name}")

            names = {t["function"]["name"] for t in self.tools}
            for self.turn in range(1, self.max_turns + 1):
                if self._take_steer() and self._interrupt.is_set():
                    break
                if self._interrupt.is_set():
                    self.state = "interrupted"
                    self._emit("status", "interrupted")
                    break

                self.state = "thinking"
                try:
                    reply = self.provider.complete(self.messages, tools=self.tools)
                except Exception as e:
                    self.state = "error"
                    self._emit("error", f"{type(e).__name__}: {e}")
                    break

                text, calls = reply.get("text", ""), reply.get("tool_calls", [])
                if text:
                    self._emit("assistant", text)
                if not calls:
                    self.state = "done"
                    self._emit("status", "the model stopped without calling a skill")
                    break

                self.messages.append({
                    "role": "assistant", "content": text or None,
                    "tool_calls": [{"type": "function", "id": f"c{i}",
                                    "function": {"name": c["name"],
                                                 "arguments": json.dumps(c.get("arguments", {}))}}
                                   for i, c in enumerate(calls)]})

                self.state = "acting"
                for i, call in enumerate(calls):
                    name, args = call["name"], call.get("arguments", {})
                    self._emit("tool_call", name, args=args)
                    if name not in names:
                        why = next((f"{a}: {w}" for n, a, w in self.unavailable if n == name),
                                   "no such skill")
                        ok, msg = False, f"skill {name!r} is not available on this rig ({why})"
                    else:
                        try:
                            ok, msg = self.executor(name, args)
                        except Exception as e:
                            ok, msg = False, f"{type(e).__name__}: {e}"
                    self._emit("tool_result", msg, name=name, ok=ok)
                    self.messages.append({"role": "tool", "tool_call_id": f"c{i}", "name": name,
                                          "content": msg})
                    # Checked AFTER the skill returns, never during: a half-executed grasp is a
                    # worse state than either finishing or not starting.
                    if self._interrupt.is_set():
                        self.state = "interrupted"
                        self._emit("status", "interrupted after the skill returned")
                        break
                if self.state == "interrupted":
                    break
            else:
                self.state = "done"
                self._emit("status", f"hit max_turns={self.max_turns}")
        except Exception as e:                       # a UI must never be taken down by the loop
            self.state = "error"
            self._emit("error", f"{type(e).__name__}: {e}")
        finally:
            if self.state not in ("error", "interrupted"):
                self.state = "done"
            self._emit("done", self.state)

    def _take_steer(self) -> bool:
        """Fold any queued guidance into the conversation. True if anything was delivered."""
        got = False
        while True:
            try:
                text = self._steer.get_nowait()
            except queue.Empty:
                return got
            self.messages.append({"role": "user", "content": text})
            got = True
