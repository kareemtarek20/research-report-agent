"""
Conditional-edge routing for the research workflow.

All routing decisions (which researchers to fan out to, whether to research
more, whether to revise the report) live here, and every loop has a hard
counter bound so the graph always terminates.
"""

from research.modes import get_mode


def route_researchers(state: dict) -> list[str]:
    """Fan-out: run in parallel every researcher that (a) the mode enables
    and (b) the planner assigned at least one task to. Always returns at
    least one node so the graph cannot stall."""
    mode = get_mode(state.get("mode", "deep"))
    assigned = {t.get("researcher", "web") for t in state.get("tasks", [])}
    active = [f"{kind}_researcher" for kind in mode.researchers
              if kind in assigned and kind in ("web", "academic", "technical")]
    return active or ["web_researcher"]


def after_critic(state: dict) -> str:
    """Route to more research only for high-priority gaps, within the round
    cap. Quick modes and the hard MAX rounds always go to the writer."""
    mode = get_mode(state.get("mode", "deep"))
    rnd = state.get("round", 0)
    assessment = state.get("assessment", {}) or {}
    gaps = state.get("gaps", [])

    has_high = any(g.get("importance") == "high" for g in gaps)
    if assessment.get("sufficient"):
        return "write"
    if not has_high:
        return "write"
    if rnd >= mode.max_rounds:
        return "write"
    return "research"


def after_report_quality(state: dict) -> str:
    """Allow at most MAX_REPORT_REVISIONS revision passes."""
    from config import MAX_REPORT_REVISIONS

    quality = state.get("report_quality", {}) or {}
    if quality.get("passed"):
        return "finalize"
    if state.get("revisions", 0) >= MAX_REPORT_REVISIONS:
        return "finalize"
    if not state.get("report_feedback"):
        return "finalize"
    return "revise"
