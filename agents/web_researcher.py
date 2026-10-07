"""Web researcher node (runs in parallel with academic/technical researchers)."""

from agents.base_researcher import run_researcher


def web_researcher_node(state: dict) -> dict:
    return run_researcher("web", state)
