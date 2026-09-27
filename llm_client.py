"""
Thin wrapper around the OpenAI python client, pointed at whatever
OpenAI-compatible endpoint is serving Qwen (or any other model).
"""

import json
from openai import OpenAI
from config import LLM_BASE_URL, LLM_MODEL, LLM_API_KEY, LLM_TEMPERATURE

_client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)


def chat(messages: list[dict], temperature: float = None) -> str:
    """Send messages to the LLM and return the text of the reply."""
    response = _client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=temperature if temperature is not None else LLM_TEMPERATURE,
    )
    return response.choices[0].message.content


def chat_json(messages: list[dict], temperature: float = None) -> dict:
    """
    Ask the LLM for a JSON object and parse it. Retries once with a
    stricter instruction if the first parse fails (small/local models
    sometimes wrap JSON in prose or code fences).
    """
    raw = chat(messages, temperature)
    parsed = _try_parse_json(raw)
    if parsed is not None:
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
    raw2 = chat(retry_messages, temperature)
    parsed2 = _try_parse_json(raw2)
    if parsed2 is not None:
        return parsed2

    raise ValueError(f"Model did not return valid JSON after retry. Last output:\n{raw2}")


def _try_parse_json(text: str):
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
