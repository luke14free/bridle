"""bridle's terminal UI: an agentic loop you can watch, steer, and stop.

    bridle tui --model local:qwen3-32b

    ┌ bridle · so101-default · local:qwen3-32b ····························· acting ┐
    │ conversation                          │ skills (16 ready)                    │
    │ › stack the red cube on the green one │  ● reach            ready            │
    │ ⋯ I'll pick the red cube first.       │  ● grab             ready            │
    │ → pick {"obj": "red cube"}            │  ● descend_to_target ready           │
    │ ✓ picked the red cube (14.2 N)        │  ▲ compact_grasp    needs re-distil  │
    │ → place {"dest": "green cube"}        │  ✕ sphere_grab      rig can't run it │
    │                                       ├──────────────────────────────────────┤
    │                                       │ jobs                                 │
    │                                       │ descend_to_target  training  ep 24   │
    ├───────────────────────────────────────┴──────────────────────────────────────┤
    │ › _                        enter send · esc interrupt · ^N model · ^C quit    │
    └──────────────────────────────────────────────────────────────────────────────┘

WHY A TUI AND NOT JUST THE WEB VIEWER. They answer different questions. The terminal is where you
TALK to the agent and steer it; the browser window is where you WATCH the robot. Video does not
belong in a terminal and a text conversation does not belong in a video pane. `bridle.ui.Viewer`
runs alongside this and shows the simulator.

STEERING IS THE POINT. A coding agent that goes wrong writes a bad file. A robot agent that goes
wrong moves a real arm, and noticing three tool calls later means a scattered scene. So typing while
the agent runs queues guidance it will see before choosing its next skill, and ESC stops it cleanly
after the current skill returns — never mid-skill, because a half-executed grasp is a state no policy
was trained from.

stdlib only: curses ships with Python. No framework, no build step, no dependency.
"""
import curses
import textwrap
import time

from bridle.agent import AgentSession
from bridle.llm import PRESETS, from_spec
from bridle.resolve import ADAPT, RETRAIN, RUN
from bridle.store import BLOCKED

GLYPH = {RUN: ("●", "ready"), ADAPT: ("▲", "needs re-distil"),
         RETRAIN: ("✕", "needs rebuild"), BLOCKED: ("·", "rig can't run it")}
KIND_GLYPH = {"assistant": ("⋯", 1), "tool_call": ("→", 4), "tool_result": ("✓", 2),
              "steer": ("»", 5), "status": ("·", 6), "error": ("!", 3), "done": ("·", 6)}


def _pair(n):
    """`curses.color_pair`, tolerant of an uninitialised or colourless terminal.

    Colour is decoration; a UI that refuses to draw without it is worse than a monochrome one. This
    also lets the rendering be tested against a fake screen with no initscr() — see tests/test_tui.py.
    """
    try:
        return curses.color_pair(n)
    except curses.error:
        return 0


class TUI:
    def __init__(self, session: AgentSession, viewer_url=None, model_label="", models=()):
        self.s = session
        self.viewer_url = viewer_url
        self.model_label = model_label
        self.models = list(models)
        self.lines = []                 # (kind, text)
        self.input = ""
        self.scroll = 0
        self._skills = (0.0, [])

    # ── data ──────────────────────────────────────────────────────────────────────────────────
    def skills(self):
        """Skill list with verdicts, cached — plan() walks the whole store and this redraws often."""
        now = time.time()
        if now - self._skills[0] < 2.0:
            return self._skills[1]
        out = []
        for app in self.s.store.apps():
            try:
                p = self.s.store.plan(app, self.s.rig)
                out.append((app.name, p.action))
            except Exception:
                out.append((app.name, RETRAIN))
        order = {RUN: 0, ADAPT: 1, RETRAIN: 2, BLOCKED: 3}
        out.sort(key=lambda x: (order.get(x[1], 9), x[0]))
        self._skills = (now, out)
        return out

    def pump(self):
        for ev in self.s.drain():
            if ev.kind == "tool_call":
                args = ev.data.get("args") or {}
                brief = ", ".join(f"{k}={v!r}" for k, v in list(args.items())[:3])
                self.lines.append(("tool_call", f"{ev.text}({brief})"))
            elif ev.kind == "tool_result":
                self.lines.append(("tool_result" if ev.data.get("ok") else "error", ev.text))
            elif ev.kind in ("assistant", "status", "error", "steer"):
                self.lines.append((ev.kind, ev.text))
            elif ev.kind == "done":
                self.lines.append(("status", f"— {ev.text or self.s.state} —"))

    # ── drawing ───────────────────────────────────────────────────────────────────────────────
    def draw(self, scr):
        scr.erase()
        h, w = scr.getmaxyx()
        # A narrow terminal drops the side pane rather than forcing a negative origin. `max(28, ...)`
        # used to win over the width and produce left = -9 at 20 columns, which curses rejects — a
        # crash on a small window is not a cosmetic bug, it takes the operator's only interface out
        # mid-task.
        right = max(24, min(40, w // 3)) if w >= 56 else 0
        left = w - right - 1 if right else w

        state = self.s.state
        head = f" bridle · {self.s.rig.name} · {self.model_label or 'no model'} "
        scr.addnstr(0, 0, head + "·" * max(0, w - len(head) - len(state) - 2), w - 1,
                    curses.A_BOLD)
        scr.addnstr(0, max(0, w - len(state) - 1), state, len(state) + 1,
                    _pair(2 if state in ("done", "idle") else
                                      3 if state == "error" else 4) | curses.A_BOLD)

        # conversation
        body = h - 3
        wrapped = []
        for kind, text in self.lines:
            g, col = KIND_GLYPH.get(kind, ("·", 6))
            for i, chunk in enumerate(textwrap.wrap(text, max(10, left - 3)) or [""]):
                wrapped.append((f"{g if i == 0 else ' '} {chunk}", col))
        view = wrapped[max(0, len(wrapped) - body + self.scroll):][:body - 1]
        for r, (txt, col) in enumerate(view, start=2):
            scr.addnstr(r, 1, txt, left - 1, _pair(col))

        # divider. ACS_VLINE only exists after initscr(), so fall back to a plain bar — the same
        # tolerance as _pair(): a UI that will not draw without a fully-initialised terminal cannot
        # be tested, and untested render code in the flagship interface is not acceptable.
        if right:
            vline = getattr(curses, "ACS_VLINE", ord("|"))
            for r in range(1, h - 2):
                scr.addch(r, left, vline)

        # skills
        sk = self.skills() if right else []
        if right:
            ready = sum(1 for _, v in sk if v == RUN)
            scr.addnstr(1, left + 2, f"skills ({ready} ready / {len(sk)})", right - 3,
                        curses.A_BOLD | _pair(6))
        cut = (h - 3) // 2
        # The VERDICT is the point of this pane, so the NAME is what gets truncated when space is
        # short — never the label. Truncating "rig can't run it" to "rig can't ru" would hide the
        # one thing the operator needs to see.
        # width bookkeeping: the rendered line is "G name<pad> label" = 1+1+namew+1+labw, and the
        # write is clipped at right-3. Solve for namew so the LABEL always survives intact.
        labw = max(len(l) for _, l in GLYPH.values())
        namew = max(6, (right - 3) - 3 - labw)
        for r, (name, verdict) in enumerate(sk[:max(0, cut - 2)], start=2):
            g, label = GLYPH.get(verdict, ("·", verdict))
            col = {RUN: 2, ADAPT: 4, RETRAIN: 3, BLOCKED: 6}.get(verdict, 6)
            scr.addnstr(r, left + 2, f"{g} {name[:namew]:<{namew}} {label}", right - 3, _pair(col))

        # jobs
        if right:
            jr = cut + 1
            scr.addnstr(jr, left + 2, "jobs", right - 3, curses.A_BOLD | _pair(6))
            jobs = getattr(self.s, "jobs", None) or {}
            for r, (name, j) in enumerate(list(jobs.items())[:max(0, h - jr - 4)], start=jr + 1):
                scr.addnstr(r, left + 2, f"{name[:18]:<18} {j}", right - 3, _pair(4))

        # input + keys
        scr.addnstr(h - 2, 0, "─" * (w - 1), w - 1)
        prompt = f"› {self.input}"
        scr.addnstr(h - 1, 1, prompt, w - 2, curses.A_BOLD)
        keys = "enter send · esc interrupt · ^N model · ^C quit"
        if self.viewer_url:
            keys = f"{self.viewer_url} · " + keys
        if w - len(keys) - 2 > len(prompt) + 2:
            scr.addnstr(h - 1, w - len(keys) - 2, keys, len(keys), _pair(6))
        scr.move(h - 1, min(w - 2, 1 + len(prompt)))
        scr.refresh()

    # ── input ─────────────────────────────────────────────────────────────────────────────────
    def key(self, ch):
        """Returns False to quit."""
        if ch in (3,):                                   # ^C
            return False
        if ch == 27:                                     # ESC — stop cleanly, do not kill
            self.s.interrupt()
        elif ch == 14:                                   # ^N — next model
            if len(self.models) > 1:
                self.models.append(self.models.pop(0))
                spec = self.models[0]
                try:
                    self.s.switch_provider(from_spec(spec), label=spec)
                    self.model_label = spec
                except Exception as e:
                    self.lines.append(("error", f"could not switch to {spec}: {e}"))
        elif ch in (curses.KEY_ENTER, 10, 13):
            text, self.input = self.input.strip(), ""
            if text:
                # Typing while the agent runs is STEERING, not a new task — the session decides,
                # because only it knows whether a loop is in flight.
                self.lines.append(("steer" if self.s.running else "status", f"› {text}"))
                self.s.start(text) if not self.s.running else self.s.steer(text)
        elif ch in (curses.KEY_BACKSPACE, 127, 8):
            self.input = self.input[:-1]
        elif ch == curses.KEY_PPAGE:
            self.scroll -= 5
        elif ch == curses.KEY_NPAGE:
            self.scroll = min(0, self.scroll + 5)
        elif 32 <= ch < 127:
            self.input += chr(ch)
        return True

    def loop(self, scr):
        curses.curs_set(1)
        curses.use_default_colors()
        for i, c in enumerate([curses.COLOR_WHITE, curses.COLOR_GREEN, curses.COLOR_RED,
                               curses.COLOR_YELLOW, curses.COLOR_MAGENTA, curses.COLOR_CYAN], 1):
            curses.init_pair(i, c, -1)
        scr.nodelay(True)
        scr.keypad(True)
        while True:
            self.pump()
            self.draw(scr)
            try:
                ch = scr.getch()
            except curses.error:
                ch = -1
            if ch != -1 and not self.key(ch):
                return
            time.sleep(0.03)


def run(session, viewer_url=None, model_label="", models=()):
    curses.wrapper(TUI(session, viewer_url, model_label, models).loop)
