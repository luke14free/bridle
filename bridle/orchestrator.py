"""The LLM loop: tools built from the store, filtered by what your rig can actually run.

    tools, unavailable = build_tools(store, rig)
    Orchestrator(provider, store, rig, executor).run("stack the red cube on the green one")

THE ONE IDEA HERE. An LLM offered a tool will call it. So the tool list must not contain skills that
do not work on this robot — otherwise the model confidently invokes a policy trained for someone
else's camera and the failure surfaces as bad robot behaviour rather than as a missing prerequisite.

`build_tools` therefore offers ONLY skills that `plan()` resolves to RUN. Everything else is returned
separately, with its verdict and the reason, so the caller can say "three skills need building
first" instead of discovering it as a 0/20 two days later. That filtering is the entire reason this
module is more than a for-loop around a chat API — and it is why the agent loop itself stays thin
(the spec's own instruction: the orchestrator is the least differentiated part).
"""
import json
from dataclasses import dataclass, field

from bridle.resolve import RUN
from bridle.store import BLOCKED


def app_to_tool(app) -> dict:
    """One app as an OpenAI-style tool schema.

    `when_to_use` is folded into the description because that is what makes a model choose CORRECTLY
    rather than merely plausibly — a name and a one-liner produce confident misuse.
    """
    props, required = {}, []
    for arg, desc in (app.args or {}).items():
        props[arg] = {"type": "string", "description": str(desc)[:300]}
        required.append(arg)
    desc = app.description or app.title
    if app.when_to_use:
        desc = f"{desc}\n\nWhen to use: {app.when_to_use}"
    return {"type": "function",
            "function": {"name": app.name, "description": desc[:900],
                         "parameters": {"type": "object", "properties": props,
                                        "required": required}}}


def build_tools(store, rig, target_contracts=None):
    """(tools, unavailable) — only RUN-able skills become callable.

    `target_contracts` optionally maps app name -> the Contract the TASK demands. Without it, each
    app is judged against its own contract re-pointed at your rig, which answers "does this skill
    work on this robot?". With it, the question sharpens to "does this skill work on this robot FOR
    THIS TASK?" — which is what distinguishes a descend trained for platform tops from one the task
    needs for cube tops. The former is the common case; the latter is how a task can demand a
    rebuild rather than silently accepting a skill trained for a different problem.

    `unavailable` is a list of (app_name, action, reason) so a caller can explain the gap. It is
    deliberately not silent: a skill missing from the tool list with no explanation looks like a
    bug, and a user who cannot see WHY will not know that `Foundry` would fix it.
    """
    tools, unavailable = [], []
    for app in store.apps():
        try:
            plan = store.plan(app, rig, target_contract=(target_contracts or {}).get(app.name))
        except Exception as e:                       # a malformed app must not sink the whole store
            unavailable.append((app.name, "error", f"{type(e).__name__}: {e}"))
            continue
        if plan.action == RUN:
            tools.append(app_to_tool(app))
        else:
            why = plan.reason
            if plan.action == BLOCKED:
                why = f"{why}: {', '.join(plan.blockers)}"
            unavailable.append((app.name, plan.action, why))
    return tools, unavailable


SYSTEM = """You control a robot arm. You have a set of SKILLS, each of which has been verified to run
on THIS robot — if a skill is not in your tool list, it is not available and you must not pretend it
is. Call one skill at a time and read its result before deciding the next.

Report honestly. If a skill reports failure, say so and stop rather than continuing as though it
succeeded; a wrong belief about what the robot did is worse than a stalled task."""


@dataclass
class Turn:
    text: str = ""
    calls: list = field(default_factory=list)
    results: list = field(default_factory=list)


@dataclass
class Session:
    task: str
    turns: list = field(default_factory=list)
    done: bool = False
    stopped_because: str = ""


class Orchestrator:
    """Runs an LLM against the rig's available skills.

    `executor(app_name, arguments) -> (ok: bool, message: str)` is supplied by the host: bridle knows
    which skills are valid, not how to drive your simulator.
    """

    def __init__(self, provider, store, rig, executor, system=SYSTEM, max_turns=12,
                 target_contracts=None):
        self.provider, self.store, self.rig = provider, store, rig
        self.executor, self.system, self.max_turns = executor, system, max_turns
        self.target_contracts = target_contracts or {}

    def run(self, task: str) -> Session:
        tools, unavailable = build_tools(self.store, self.rig, self.target_contracts)
        sess = Session(task=task)
        if not tools:
            sess.stopped_because = (
                "no skill in the store runs on this rig. "
                + "; ".join(f"{n}: {a}" for n, a, _ in unavailable[:5]))
            return sess

        messages = [{"role": "system", "content": self.system},
                    {"role": "user", "content": task}]
        for _ in range(self.max_turns):
            reply = self.provider.complete(messages, tools=tools)
            turn = Turn(text=reply.get("text", ""), calls=reply.get("tool_calls", []))
            sess.turns.append(turn)
            if not turn.calls:
                sess.done = True
                sess.stopped_because = "the model returned no tool call"
                return sess
            messages.append({"role": "assistant", "content": turn.text or None,
                             "tool_calls": [{"type": "function", "id": f"c{i}",
                                             "function": {"name": c["name"],
                                                          "arguments": json.dumps(c["arguments"])}}
                                            for i, c in enumerate(turn.calls)]})
            for i, call in enumerate(turn.calls):
                name = call["name"]
                if name not in {t["function"]["name"] for t in tools}:
                    # The model invented a skill, or reached for one this rig cannot run. Say which,
                    # rather than failing opaquely — this is the exact moment the contract check pays
                    # for itself, and the model can recover if it is told.
                    reason = next((f"{a}: {w}" for n, a, w in unavailable if n == name),
                                  "no such skill")
                    out = (False, f"skill {name!r} is not available on this rig ({reason})")
                else:
                    try:
                        out = self.executor(name, call["arguments"])
                    except Exception as e:
                        out = (False, f"{type(e).__name__}: {e}")
                turn.results.append(out)
                messages.append({"role": "tool", "tool_call_id": f"c{i}", "name": name,
                                 "content": out[1]})
        sess.stopped_because = f"hit max_turns={self.max_turns}"
        return sess
