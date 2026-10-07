"""Academic researcher node (Semantic Scholar-backed, parallel fan-out branch)."""

from agents.base_researcher import run_researcher


def academic_researcher_node(state: dict) -> dict:
    return run_researcher("academic", state)
