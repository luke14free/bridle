# Using bridle with Pi (or any agent harness)

**bridle does not ship a coding agent, and should not.**

[Pi](https://pi.dev) (MIT, `earendil-works/pi`) already is one: a TUI with mid-session model
switching across 15+ providers, tree-structured shareable sessions, real-time steering, and — the
part that matters here — a first-class extension API and a language-agnostic RPC mode. Its design
premise is that you extend rather than fork.

Forking it would mean maintaining a coding agent forever as a tax on the robotics work, and would
buy nothing bridle needs. So the split is:

| | owns |
|---|---|
| **Pi** (unforked) | the conversation, the model, the session, the terminal |
| **bridle** | the robot: which skills run on your rig, executing them, and the simulator window |

A TUI cannot render a robot anyway, which is why `bridle.ui.Viewer` exists as a separate window
rather than as a pane inside the agent.

## The shape

```
┌─ terminal ─────────────┐   ┌─ browser ──────────────────────┐
│ Pi (TUI)               │   │ bridle Viewer                  │
│  you: stack the cubes  │   │  so101-default · base cam      │
│  > descend_to_target   │   │  ready(16) needs-rebuild(2)    │
│  < ok, force 3.5N      │   │  ┌──────────────────────────┐  │
│                        │   │  │   [simulator frames]     │  │
│                        │   │  └──────────────────────────┘  │
│                        │   │  descend_to_target  training   │
└────────────────────────┘   └────────────────────────────────┘
        │                                   ▲
        └── tools filtered by ──► bridle ───┘
            resolve(app, rig)
```

The agent's tool list and the viewer's skill list come from the **same** `plan(app, rig)` call, so
they cannot disagree about what the robot can do. `bridle/tests/test_ui.py` asserts exactly that.

## Running the viewer

```python
from bridle import Rig, Store
from bridle.ui import Viewer

viewer = Viewer(Store("~/.bridle/apps"), Rig.so101(cameras=("base",))).start()
print(viewer.url)                      # http://127.0.0.1:8799

viewer.push_frame(jpeg_bytes)          # whatever your sim just rendered
viewer.set_job("descend_to_target", "training", "epoch 286 / 20.3M steps")
```

Read-only by design: it shows state and streams frames. Anything that *moves the robot* goes through
the agent, where the contract checks are.

## The Pi extension

Pi extensions are TypeScript modules in `~/.pi/agent/extensions/` that register tools. The extension
is thin on purpose — it forwards to a local bridle server and lets bridle decide what is runnable:

```typescript
// ~/.pi/agent/extensions/bridle.ts
export default async function (pi: ExtensionAPI) {
  const BASE = process.env.BRIDLE_URL ?? "http://127.0.0.1:8799";
  const state = await (await fetch(`${BASE}/api/state`)).json();

  // Only skills bridle says RUN on this rig become tools. A model offered a tool WILL call it, so
  // a skill that cannot run must never appear — that is the whole point of the contract check.
  for (const app of state.apps.filter((a: any) => a.verdict === "run")) {
    pi.registerTool({
      name: app.name,
      description: app.why ? `${app.name} — ${app.why}` : app.name,
      run: async (args: any) =>
        (await (await fetch(`${BASE}/api/execute`, {
          method: "POST",
          body: JSON.stringify({ app: app.name, args }),
        })).json()).message,
    });
  }

  // Surface the skills that are NOT available, and why. Silence here reads as "that skill doesn't
  // exist"; the truth is usually "it needs a rebuild for your rig", which is actionable.
  pi.registerCommand("bridle", {
    run: async () => {
      const s = await (await fetch(`${BASE}/api/state`)).json();
      const bad = s.apps.filter((a: any) => a.verdict !== "run");
      return [`rig: ${s.rig.name} (${s.rig.fingerprint})`,
              `ready: ${s.counts.run}`,
              ...bad.map((a: any) => `  ${a.verdict.padEnd(8)} ${a.name} — ${a.why}`)].join("\n");
    },
  });
}
```

**Not yet implemented:** `POST /api/execute`. The viewer is deliberately read-only today, and the
executor that actually drives a skill is host-specific — it needs a live simulator session, which
lives in the consuming project, not in the library. `bridle.Orchestrator` already takes an
`executor(name, args) -> (ok, message)` callback for exactly this; wiring it to an HTTP endpoint is
the remaining step.

## If you use a different agent

Nothing above is Pi-specific. Any harness that can call a local HTTP endpoint can consume
`/api/state` and filter its tools the same way; `bridle.Orchestrator` is a working reference
implementation of that loop in ~80 lines, and Pi's RPC mode (LF-delimited JSONL over stdio) is an
alternative for non-JavaScript hosts.
