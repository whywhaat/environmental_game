"""Minimal OpenAI-compatible chat-completions client (standard library only)."""

from __future__ import annotations

import json
import os
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from copy import deepcopy
from typing import Protocol
from uuid import uuid4


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
        started = time.perf_counter()
        try:
            response = self.client.complete(model, messages)
        except Exception as exc:
            call["error"] = str(exc)
            raise
        else:
            call["response"] = response
            return response
        finally:
            call["duration_seconds"] = round(time.perf_counter() - started, 3)


class SplitChat:
    """Route converter calls to one API and player calls to another."""

    def __init__(self, player_client: ChatModel, converter_client: ChatModel, converter_model: str):
        self.player_client = player_client
        self.converter_client = converter_client
        self.converter_model = converter_model

    def complete(self, model: str, messages: list[dict[str, str]]) -> str:
        client = self.converter_client if model == self.converter_model else self.player_client
        return client.complete(model, messages)


class OpenAICompatibleChat:
    def __init__(self, base_url: str | None = None, api_key: str | None = None, timeout: int | None = None, verify_ssl: bool | None = None):
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL", "")).rstrip("/")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.timeout = timeout if timeout is not None else int(os.getenv("OPENAI_TIMEOUT", "360"))
        self.verify_ssl = verify_ssl if verify_ssl is not None else os.getenv("OPENAI_SSL_VERIFY", "true").lower() not in {"0", "false", "no"}
        if not self.base_url:
            raise ValueError("OPENAI_BASE_URL is required")

    def complete(self, model: str, messages: list[dict[str, str]]) -> str:
        payload = json.dumps({"model": model, "messages": messages}, ensure_ascii=False).encode("utf-8")
        ssl_context = None if self.verify_ssl else ssl._create_unverified_context()
        for attempt in range(2):
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            request = urllib.request.Request(
                f"{self.base_url}/chat/completions",
                data=payload,
                headers=headers,
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout, context=ssl_context) as response:
                    result = json.load(response)
                content = result["choices"][0]["message"]["content"]
                if not isinstance(content, str):
                    raise ValueError("LLM returned non-text content")
                return content.strip()
            except urllib.error.HTTPError as exc:
                if exc.code == 401 and attempt == 0 and urllib.parse.urlparse(self.base_url).hostname == "api.giga.chat":
                    self._refresh_gigachat_token()
                    continue
                raise RuntimeError(f"LLM request failed: HTTP Error {exc.code}: {exc.reason}") from exc
            except (TimeoutError, urllib.error.URLError) as exc:
                timed_out = isinstance(exc, TimeoutError) or isinstance(getattr(exc, "reason", None), TimeoutError)
                if timed_out and attempt == 0:
                    continue
                detail = f"timed out twice after {self.timeout} seconds per attempt" if timed_out else str(exc)
                raise RuntimeError(f"LLM request failed: {detail}") from exc
            except (KeyError, IndexError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"LLM request failed: {exc}") from exc

    def _refresh_gigachat_token(self) -> None:
        auth_key = os.getenv("GIGACHAT_AUTH_KEY", "").strip()
        if not auth_key:
            raise RuntimeError("GigaChat returned 401; set GIGACHAT_AUTH_KEY in .env for automatic token renewal")
        if auth_key.lower().startswith("basic "):
            auth_key = auth_key[6:].strip()
        request = urllib.request.Request(
            "https://ngw.devices.sberbank.ru:9443/api/v2/oauth",
            data=urllib.parse.urlencode({"scope": "GIGACHAT_API_PERS"}).encode("ascii"),
            headers={"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json",
                     "RqUID": str(uuid4()), "Authorization": f"Basic {auth_key}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout, context=ssl._create_unverified_context()) as response:
                token = json.load(response)["access_token"]
        except (urllib.error.URLError, ValueError, KeyError) as exc:
            raise RuntimeError(f"GigaChat token renewal failed: {exc}") from exc
        if not isinstance(token, str) or not token:
            raise RuntimeError("GigaChat token renewal returned no access_token")
        self.api_key = token
