# -*- coding: utf-8 -*-
import time
import random

class MockRateLimitError(Exception):
    def __init__(self, message="Error code: 429 - {'detail': {'error': {'message': '429 https://api.giga.chat/v1/chat/completions'}}}", status_code=429):
        super().__init__(message)
        self.status_code = status_code

class MockResponse:
    def __init__(self, content):
        self.content = content

class MockLLM:
    def __init__(self, fail_times=2):
        self.attempts = 0
        self.fail_times = fail_times

    def invoke(self, messages):
        self.attempts += 1
        print(f"MockLLM invoke attempt {self.attempts}...")
        if self.attempts <= self.fail_times:
            raise MockRateLimitError()
        return MockResponse("Ответ от модели успешно получен!")

def test_retry():
    mock_llm = MockLLM(fail_times=2)
    current = {"name": "Промкомплекс", "llm": mock_llm}
    messages = [{"role": "system", "content": "test"}]
    
    max_retries = 5
    base_delay = 1.0
    debug_calls = []
    
    call = {
        "phase": "Вступление",
        "turn": 1,
        "player": current["name"],
    }
    debug_calls.append(call)
    
    delay = base_delay
    last_exc = None
    
    for attempt in range(1, max_retries + 1):
        try:
            response = current["llm"].invoke(messages)
            content = getattr(response, "content", response)
            call["response"] = content if isinstance(content, str) else str(content)
            time.sleep(0.1)
            print("SUCCESS:", call["response"])
            return response
        except Exception as exc:
            last_exc = exc
            err_str = str(exc).lower()
            err_name = type(exc).__name__
            
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
                    "remote disconnected", "server disconnected"
                ]
            )
            
            if attempt < max_retries and is_transient:
                wait_time = delay + random.uniform(0.1, 0.3)
                print(f"Attempt {attempt} failed with {err_name}. Waiting {wait_time:.2f}s before retry {attempt+1}...")
                call.setdefault("retries", []).append({
                    "attempt": attempt,
                    "error": f"{err_name}: {exc}",
                    "wait_time": round(wait_time, 2),
                })
                time.sleep(wait_time)
                delay *= 2.0
            else:
                call["error"] = f"{err_name}: {exc}"
                raise

if __name__ == "__main__":
    resp = test_retry()
    assert resp.content == "Ответ от модели успешно получен!"
    print("Test passed successfully!")
