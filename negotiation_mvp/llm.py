"""Minimal OpenAI-compatible chat-completions client (standard library only)."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from copy import deepcopy
from typing import Protocol


class ChatModel(Protocol):
    def complete(self, model: str, messages: list[dict[str, str]]) -> str: ...


class RecordingChat:
    """Keep the exact message payload and response for each model call."""

    def __init__(self, client: ChatModel):
        self.client = client
        self.calls: list[dict] = []

    def complete(self, model: str, messages: list[dict[str, str]]) -> str:
        call = {"model": model, "messages": deepcopy(messages)}
        self.calls.append(call)
        try:
            response = self.client.complete(model, messages)
        except Exception as exc:
            call["error"] = str(exc)
            raise
        call["response"] = response
        return response


class OpenAICompatibleChat:
    def __init__(self, base_url: str | None = None, api_key: str | None = None, timeout: int | None = None):
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL", "")).rstrip("/")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.timeout = timeout if timeout is not None else int(os.getenv("OPENAI_TIMEOUT", "360"))
        if not self.base_url:
            raise ValueError("OPENAI_BASE_URL is required")

    def complete(self, model: str, messages: list[dict[str, str]]) -> str:
        payload = json.dumps({"model": model, "messages": messages}, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            headers=headers,
            method="POST",
        )
        for attempt in range(2):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    result = json.load(response)
                content = result["choices"][0]["message"]["content"]
                if not isinstance(content, str):
                    raise ValueError("LLM returned non-text content")
                return content.strip()
            except (TimeoutError, urllib.error.URLError) as exc:
                timed_out = isinstance(exc, TimeoutError) or isinstance(getattr(exc, "reason", None), TimeoutError)
                if timed_out and attempt == 0:
                    continue
                detail = f"timed out twice after {self.timeout} seconds per attempt" if timed_out else str(exc)
                raise RuntimeError(f"LLM request failed: {detail}") from exc
            except (KeyError, IndexError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"LLM request failed: {exc}") from exc
