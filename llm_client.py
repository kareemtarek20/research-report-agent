"""
Thin wrapper around the OpenAI python client, pointed at whatever
OpenAI-compatible endpoint is serving the model.

Adds (on top of the original chat/chat_json API):
  - retries with backoff on timeouts / rate limits / transient API errors
  - robust JSON extraction (handles prose-wrapped or fenced JSON, lists)
  - a usage meter (call count, estimated tokens) for observability
"""

import json
import re
import threading
import time
from openai import OpenAI, APIConnectionError, APIStatusError, APITimeoutError, RateLimitError

from config import (LLM_BASE_URL, LLM_MODEL, LLM_API_KEY, LLM_TEMPERATURE,
                    LLM_TEMP_PRECISE, LLM_TIMEOUT_SECONDS, LLM_MAX_RETRIES)

# The client only builds a connection object; a placeholder key keeps imports
# working for tests/offline use, and real calls fail fast with a clear
# provider error if LLM_API_KEY was never configured.
_client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY or "missing-llm-api-key",
                 timeout=LLM_TIMEOUT_SECONDS, max_retries=0)


class UsageMeter:
    """Thread-safe counter of LLM calls and approximate token usage."""

    def __init__(self):
        self._lock = threading.Lock()
        self.calls = 0
        self.retries = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def record_call(self, prompt_chars: int, completion: str, usage=None):
        with self._lock:
            self.calls += 1
            if usage is not None:
                self.prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
                self.completion_tokens += getattr(usage, "completion_tokens", 0) or 0
            else:  # rough estimate when the provider omits usage
                self.prompt_tokens += prompt_chars // 4
                self.completion_tokens += len(completion) // 4

    def record_retry(self):
        with self._lock:
            self.retries += 1

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "llm_calls": self.calls,
                "llm_retries": self.retries,
                "prompt_tokens_est": self.prompt_tokens,
                "completion_tokens_est": self.completion_tokens,
            }


usage_meter = UsageMeter()


def chat(messages: list[dict], temperature: float = None, precise: bool = False) -> str:
    """Send messages to the LLM and return the text of the reply.

    precise=True forces the deterministic low temperature used for planning,
    extraction, scoring and fact checking.
    """
    temp = LLM_TEMP_PRECISE if precise else (
        temperature if temperature is not None else LLM_TEMPERATURE)
    prompt_chars = sum(len(m.get("content", "")) for m in messages)

    last_error = None
    for attempt in range(LLM_MAX_RETRIES + 1):
        try:
            response = _client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                temperature=temp,
            )
            content = response.choices[0].message.content or ""
            usage_meter.record_call(prompt_chars, content, getattr(response, "usage", None))
            return content
        except (APITimeoutError, RateLimitError, APIConnectionError) as e:
            last_error = e
            usage_meter.record_retry()
            if attempt < LLM_MAX_RETRIES:
                time.sleep(2 * (attempt + 1))
        except APIStatusError as e:
            if e.status_code in (429, 500, 502, 503) and attempt < LLM_MAX_RETRIES:
                last_error = e
                usage_meter.record_retry()
                time.sleep(2 * (attempt + 1))
            else:
                raise
    raise last_error


def chat_json(messages: list[dict], temperature: float = None, precise: bool = True) -> dict:
    """
    Ask the LLM for a JSON object and parse it. Retries once with a
    stricter instruction if the first parse fails (small/local models
    sometimes wrap JSON in prose or code fences).
    """
    raw = chat(messages, temperature, precise=precise)
    parsed = extract_json(raw)
    if isinstance(parsed, dict):
        return parsed

    retry_messages = messages + [
        {"role": "assistant", "content": raw},
        {
            "role": "user",
            "content": (
                "That was not valid JSON. Reply with ONLY a valid JSON object, "
                "no prose, no markdown code fences, nothing else."
            ),
        },
    ]
    raw2 = chat(retry_messages, temperature, precise=precise)
    parsed2 = extract_json(raw2)
    if isinstance(parsed2, dict):
        return parsed2

    raise ValueError(f"Model did not return valid JSON after retry. Last output:\n{raw2}")


def chat_json_list(messages: list[dict], temperature: float = None,
                   precise: bool = True) -> list:
    """Ask the LLM for a JSON array (or an object wrapping one). Never raises
    on malformed output — returns [] so one bad node can't kill the run."""
    try:
        raw = chat(messages, temperature, precise=precise)
        parsed = extract_json(raw)
    except Exception:
        return []
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        # common shapes: {"items": [...]} / {"results": [...]}
        for v in parsed.values():
            if isinstance(v, list):
                return v
    return []


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def extract_json(text: str):
    """Extract the first JSON object/array from arbitrary LLM output.
    Handles code fences, leading prose and trailing commentary. Returns None
    when nothing parseable is found."""
    if not text:
        return None
    candidates = [text.strip()]
    fenced = _FENCE_RE.findall(text)
    candidates = fenced + candidates

    for cand in candidates:
        obj = _loads(cand)
        if obj is not None:
            return obj
        # scan for the first { or [ and try raw_decode from there
        for i, ch in enumerate(cand):
            if ch in "{[":
                obj = _raw_decode(cand, i)
                if obj is not None:
                    return obj
                break
    return None


def _loads(text: str):
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None


def _raw_decode(text: str, start: int):
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
        return obj
    except (json.JSONDecodeError, ValueError):
        return None


# Backwards-compatible alias used previously
def _try_parse_json(text: str):
    obj = extract_json(text)
    return obj if isinstance(obj, dict) else None
