"""Technical researcher node (docs/benchmark-focused, parallel fan-out branch)."""

from agents.base_researcher import run_researcher


def technical_researcher_node(state: dict) -> dict:
    return run_researcher("technical", state)
