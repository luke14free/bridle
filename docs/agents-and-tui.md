# Agents, providers and the TUI

[← README](../README.md)

## Only offer what works

An LLM offered a tool will call it. So the tool list contains only skills that resolve to `run` on
your rig; everything else comes back separately, with its verdict and the reason — visible to you,
invisible to the model.

```python
from bridle import Orchestrator, build_tools, from_spec

provider = from_spec("local:qwen3-32b")     # or "anthropic:...", "openai:...", "ollama:..."
tools, unavailable = build_tools(store, rig)
# unavailable -> [('sphere_grab', 'blocked',
#                  "the rig does not meet this skill's hard requirements: "
#                  "camera 'wrist' (rig has ['base'])")]

def executor(app_name, arguments):
    ...                                      # you drive your simulator here
    return True, "picked the red cube"

session = Orchestrator(provider, store, rig, executor).run("stack the red cube on the green one")
```

## Providers are one method

```python
Provider.complete(messages, tools) -> {"text": ..., "tool_calls": [...]}
```

An OpenAI-compatible HTTP client, an Anthropic Messages client and a scripted fake ship with the
library — all over `urllib`, no SDK, no dependency. `from_spec("local:qwen3-32b")` points at a local
vLLM or Ollama's compatibility endpoint.

The scripted fake is not a toy: it makes the orchestrator's control flow testable with no network, no
GPU and no model, which is why the agent's interrupt guarantee has real coverage.

## The TUI

```bash
bridle tui --model local:qwen3-32b
```

Two surfaces, because they answer different questions.

**The terminal** is where you talk to the agent. Typing while it runs queues guidance it sees before
choosing its next skill — you do not have to interrupt to redirect. ESC stops it cleanly **after the
current skill returns**, never mid-grasp: a robot halted between "closed the jaw" and "lifted" is in
a state nothing in the system knows how to recover.

**The browser** (`bridle.ui.Viewer`, default `http://127.0.0.1:8799`) is where you watch — the skill
list with live verdicts, running jobs, and simulator frames you push yourself with
`viewer.push_frame(jpeg_bytes)`.

Without `--executor module:function` the TUI runs in **dry mode**: skill calls are reported, nothing
moves. That is the default on purpose.

### Or don't use ours

bridle does not ship a coding agent and should not. If you already have an agent harness you like,
expose bridle's skills to it instead — see
**[docs/pi-extension.md](pi-extension.md)** for a worked integration with
[Pi](https://pi.dev), which applies to any harness that can call tools.

## The CLI

```bash
bridle skills                     # what runs on this rig, and what doesn't
bridle plan <app>                 # why a skill needs adapting or rebuilding
bridle tui --model <spec>         # agent TUI + viewer

bridle skill vocab                # the authorable surface — the model's prompt payload
bridle skill check   <f.yaml>     # schema, then compile. Exit 1 on the first refusal
bridle skill compile <f.yaml>     # the resolved plan + the plan fingerprint
```

**`skill` and `skills` are two different commands and the `s` is the whole difference.** `skills`
(plural) lists what is already trained and whether it runs here. `skill` (singular) is the authoring
side — see [docs/skill-yaml.md](skill-yaml.md).

The subcommands construct an SO-101 rig with only the camera list configurable; other rigs go through
the Python API.
