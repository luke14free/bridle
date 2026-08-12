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
