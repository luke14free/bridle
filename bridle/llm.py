"""Bring your own LLM. A deliberately tiny provider interface — no SDK, no vendor lock.

Two implementations ship: an OpenAI-compatible HTTP client (which covers a local vLLM/Ollama server
and most hosted APIs), and a scripted fake for tests. That is the whole surface.

WHY SO THIN. The agent loop is the least differentiated part of a robotics harness — plenty of
projects wrap an LLM around a robot. bridle's defensible piece is the contract spine: knowing which
skills actually work on your rig. So this module does the minimum needed to let any model drive
those skills, and deliberately does not grow a prompt framework, a memory system, or a router.

stdlib only: urllib, not requests. Core stays dependency-free.
"""
import json
import os
import urllib.error
import urllib.request


class Provider:
    """Anything that can turn (messages, tools) into a reply, possibly a tool call."""

    def complete(self, messages, tools=None) -> dict:
        """Return {"text": str, "tool_calls": [{"name": str, "arguments": dict}, ...]}."""
        raise NotImplementedError


class OpenAICompatProvider(Provider):
    """Any server speaking the OpenAI chat-completions shape.

    Covers a local vLLM (`--api-key` optional), Ollama's compat endpoint, and the hosted APIs. The
    base URL and model are yours; bridle never assumes a vendor.
    """

    def __init__(self, base_url, model, api_key=None, timeout=120, temperature=0.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or os.environ.get("BRIDLE_LLM_API_KEY")
        self.timeout = timeout
        self.temperature = temperature

    def complete(self, messages, tools=None) -> dict:
        payload = {"model": self.model, "messages": messages, "temperature": self.temperature}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json",
                     **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {})})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                body = json.loads(r.read().decode())
        except urllib.error.URLError as e:
            raise RuntimeError(f"LLM provider unreachable at {self.base_url}: {e}") from None
        msg = (body.get("choices") or [{}])[0].get("message", {})
        calls = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", {})
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    args = {"_raw": args}
            calls.append({"name": fn.get("name"), "arguments": args or {}})
        return {"text": msg.get("content") or "", "tool_calls": calls}


class AnthropicProvider(Provider):
    """Anthropic's Messages API — a different wire shape, same Provider contract.

    Worth carrying natively rather than via a compatibility shim: tools, system prompt and tool
    results are all shaped differently enough that a shim silently drops tool calls, and an agent
    that cannot call a skill looks like a model that will not use the robot.
    """

    def __init__(self, model, api_key=None, base_url="https://api.anthropic.com/v1",
                 timeout=120, max_tokens=4096, temperature=0.0):
        self.model, self.base_url, self.timeout = model, base_url.rstrip("/"), timeout
        self.max_tokens, self.temperature = max_tokens, temperature
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")

    def complete(self, messages, tools=None) -> dict:
        system = " ".join(m["content"] for m in messages if m.get("role") == "system" and m.get("content"))
        conv = []
        for m in messages:
            role = m.get("role")
            if role == "system":
                continue
            if role == "tool":
                conv.append({"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": m.get("tool_call_id", "c0"),
                     "content": m.get("content", "")}]})
            elif role == "assistant" and m.get("tool_calls"):
                blocks = ([{"type": "text", "text": m["content"]}] if m.get("content") else [])
                for tc in m["tool_calls"]:
                    fn = tc["function"]
                    args = fn.get("arguments")
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except ValueError:
                            args = {}
                    blocks.append({"type": "tool_use", "id": tc.get("id", "c0"),
                                   "name": fn["name"], "input": args or {}})
                conv.append({"role": "assistant", "content": blocks})
            else:
                conv.append({"role": role, "content": m.get("content") or ""})
        payload = {"model": self.model, "max_tokens": self.max_tokens,
                   "temperature": self.temperature, "messages": conv}
        if system:
            payload["system"] = system
        if tools:
            payload["tools"] = [{"name": t["function"]["name"],
                                 "description": t["function"].get("description", ""),
                                 "input_schema": t["function"].get("parameters", {})}
                                for t in tools]
        req = urllib.request.Request(
            f"{self.base_url}/messages", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", "anthropic-version": "2023-06-01",
                     **({"x-api-key": self.api_key} if self.api_key else {})})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                body = json.loads(r.read().decode())
        except urllib.error.URLError as e:
            raise RuntimeError(f"Anthropic API unreachable: {e}") from None
        text, calls = "", []
        for block in body.get("content") or []:
            if block.get("type") == "text":
                text += block.get("text", "")
            elif block.get("type") == "tool_use":
                calls.append({"name": block.get("name"), "arguments": block.get("input") or {}})
        return {"text": text, "tool_calls": calls}


class ScriptedProvider(Provider):
    """A fixed sequence of replies. For tests, and for replaying a session deterministically."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.seen = []

    def complete(self, messages, tools=None) -> dict:
        self.seen.append({"messages": messages, "tools": tools})
        if not self.replies:
            return {"text": "done", "tool_calls": []}
        r = self.replies.pop(0)
        return {"text": r.get("text", ""), "tool_calls": r.get("tool_calls", [])}


#: Shorthand -> (provider, default base_url). `bridle tui --model local:qwen` should not require
#: knowing that vLLM speaks the OpenAI shape on :8000/v1.
PRESETS = {
    "local": ("openai", "http://127.0.0.1:8000/v1"),
    "vllm": ("openai", "http://127.0.0.1:8000/v1"),
    "ollama": ("openai", "http://127.0.0.1:11434/v1"),
    "openai": ("openai", "https://api.openai.com/v1"),
    "openrouter": ("openai", "https://openrouter.ai/api/v1"),
    "anthropic": ("anthropic", "https://api.anthropic.com/v1"),
}


def from_spec(spec, base_url=None, api_key=None) -> Provider:
    """Build a provider from "<preset>:<model>", e.g. "local:qwen3-32b", "anthropic:claude-sonnet-4".

    A bare string with no preset is treated as a local model, because the common case for a robot in
    a lab is a model on the same machine as the simulator.
    """
    preset, _, model = spec.partition(":")
    if not model:
        preset, model = "local", preset
    if preset not in PRESETS:
        raise ValueError(f"unknown provider {preset!r}; known: {sorted(PRESETS)}")
    kind, default_url = PRESETS[preset]
    url = base_url or os.environ.get("BRIDLE_LLM_URL") or default_url
    if kind == "anthropic":
        return AnthropicProvider(model=model, api_key=api_key, base_url=url)
    return OpenAICompatProvider(base_url=url, model=model, api_key=api_key)
