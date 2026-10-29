# -*- coding: utf-8 -*-
import py_compile

file_path = "langflow/custom_components/my_components/dialog.py"

with open(file_path, "r", encoding="utf-8") as f:
    code = f.read()

# 1. Add time and random imports if missing
if "import time" not in code:
    code = code.replace("import re\n", "import random\nimport re\nimport time\n")
    print("Added time and random imports!")

# 2. Update _invoke_model with retry & backoff
old_invoke = """    def _invoke_model(self, current: dict, phase: str, turn: int, messages: list):
        call = {
            "phase": phase,
            "turn": turn,
            "player": current["name"],
            "messages": [{"role": message.type, "content": str(message.content)} for message in messages],
        }
        self._debug_calls.append(call)
        try:
            response = current["llm"].invoke(messages)
        except Exception as exc:
            call["error"] = f"{type(exc).__name__}: {exc}"
            raise
        content = getattr(response, "content", response)
        call["response"] = content if isinstance(content, str) else str(content)
        return response"""

new_invoke = """    def _invoke_model(self, current: dict, phase: str, turn: int, messages: list, max_retries: int = 5, base_delay: float = 2.0):
        call = {
            "phase": phase,
            "turn": turn,
            "player": current["name"],
            "messages": [{"role": message.type, "content": str(message.content)} for message in messages],
        }
        self._debug_calls.append(call)

        delay = base_delay
        for attempt in range(1, max_retries + 1):
            try:
                response = current["llm"].invoke(messages)
                content = getattr(response, "content", response)
                call["response"] = content if isinstance(content, str) else str(content)
                # Пауза между запросами для предотвращения превышения burst-лимитов API (GigaChat / OpenAI)
                time.sleep(1.0)
                return response
            except Exception as exc:
                err_str = str(exc).lower()
                err_name = type(exc).__name__

                # Определяем ошибки rate limit (429) или временные сбои шлюза / сети
                is_rate_limit = (
                    "429" in err_str
                    or "rate limit" in err_str
                    or "ratelimit" in err_name.lower()
                    or "too many requests" in err_str
                )
                is_transient = is_rate_limit or any(
                    sig in err_str
                    for sig in [
                        "500", "502", "503", "504",
                        "timeout", "timed out", "connection",
                        "remote disconnected", "server disconnected", "overloaded"
                    ]
                )

                if attempt < max_retries and is_transient:
                    # Извлекаем retry-after если сервер вернул конкретное время ожидания
                    retry_after = None
                    resp_obj = getattr(exc, "response", None)
                    if resp_obj is not None:
                        hdrs = getattr(resp_obj, "headers", None)
                        if hdrs:
                            ra_hdr = hdrs.get("retry-after") or hdrs.get("Retry-After")
                            if ra_hdr:
                                try:
                                    retry_after = float(ra_hdr)
                                except (ValueError, TypeError):
                                    pass

                    if retry_after is not None and retry_after > 0:
                        wait_time = retry_after + random.uniform(0.5, 1.5)
                    else:
                        wait_time = delay + random.uniform(0.5, 1.5)
                        delay *= 2.0

                    call.setdefault("retries", []).append({
                        "attempt": attempt,
                        "error": f"{err_name}: {exc}",
                        "wait_time": round(wait_time, 2),
                    })
                    time.sleep(wait_time)
                else:
                    call["error"] = f"{err_name}: {exc}"
                    if is_rate_limit:
                        raise RuntimeError(
                            f"Превышен лимит запросов к модели ({err_name}: 429 Rate Limit) для игрока '{current['name']}'. "
                            f"Было выполнено {attempt} попыток с паузами. Пожалуйста, подождите 1-2 минуты перед повторным запуском "
                            f"или проверьте квоту/баланс токенов в личном кабинете GigaChat."
                        ) from exc
                    raise"""

if old_invoke in code:
    code = code.replace(old_invoke, new_invoke)
    print("Replaced _invoke_model with LF!")
else:
    old_invoke_crlf = old_invoke.replace("\n", "\r\n")
    new_invoke_crlf = new_invoke.replace("\n", "\r\n")
    if old_invoke_crlf in code:
        code = code.replace(old_invoke_crlf, new_invoke_crlf)
        print("Replaced _invoke_model with CRLF!")
    else:
        print("ERROR: old_invoke not found!")
        exit(1)

with open(file_path, "w", encoding="utf-8") as f:
    f.write(code)

py_compile.compile(file_path, doraise=True)
print("dialog.py compiled successfully with py_compile!")
