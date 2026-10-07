"""
Research Planner node.

Decomposes the question into subquestions and assigns each a focused query
and a researcher (web / academic / technical). On later rounds it plans
specifically to close the gaps the Research Critic found.
"""

import logging

from config import MAX_SEARCH_ITERATIONS
from graph.state import ResearchTask, model_from_dict
from llm_client import chat_json
from research.modes import get_mode

log = logging.getLogger("agent.planner")

SYSTEM = ("You are a research planner for an evidence-driven research agent. "
          "Decompose the question into subquestions, each with one focused "
          "search query and the best researcher to handle it. JSON only.")


def plan_node(state: dict) -> dict:
    mode = get_mode(state.get("mode", "deep"))
    rnd = state.get("round", 0)
    question = state["question"]
    prior_queries = state.get("queries_done", [])
    gaps = state.get("gaps", [])
    evidence_preview = _evidence_preview(state.get("evidence", []))

    if rnd == 0 or not gaps:
        focus = (f"{mode.planner_guidance}\n"
                 f"Only these researchers are available for this mode: "
                 f"{', '.join(mode.researchers)}.")
        user = (f"RESEARCH QUESTION: {question}\n\n"
                f"Guidance: {focus}\n\n"
                f"Queries already used (avoid repeats): {prior_queries or '(none)'}\n\n"
                "Reply ONLY with JSON:\n"
                '{"tasks": [{"subquestion": "...", "query": "...", '
                '"researcher": "web|academic|technical", "rationale": "..."}]}')
    else:
        user = (f"RESEARCH QUESTION: {question}\n\n"
                "A research critic found GAPS in the current evidence. Plan the "
                "next round of research to close ONLY the high-importance gaps.\n\n"
                f"GAPS (JSON): {gaps}\n\n"
                f"Existing evidence summary:\n{evidence_preview}\n\n"
                f"Queries already used (avoid repeats): {prior_queries}\n\n"
                f"Available researchers: {', '.join(mode.researchers)}.\n"
                "Reply ONLY with JSON:\n"
                '{"tasks": [{"subquestion": "...", "query": "...", '
                '"researcher": "web|academic|technical", "rationale": "..."}]}')

    try:
        result = chat_json([{"role": "system", "content": SYSTEM},
                            {"role": "user", "content": user}])
        raw_tasks = result.get("tasks", [])
    except Exception as e:
        log.warning("planner LLM failed (%s); using fallback plan", e)
        raw_tasks = []

    tasks = []
    for t in raw_tasks:
        task = model_from_dict(ResearchTask, t)
        if task is None or not task.query.strip():
            continue
        if task.researcher not in ("web", "academic", "technical"):
            task.researcher = "web"
        if task.researcher not in mode.researchers:
            task.researcher = mode.researchers[0]
        tasks.append(task.to_dict())

    # Hard caps keep the fan-out bounded no matter what the model returns.
    cap = min(MAX_SEARCH_ITERATIONS, mode.max_queries_per_round * 3)
    tasks = tasks[:cap]

    if not tasks:  # never dead-end the graph
        first = mode.researchers[0] if mode.researchers else "web"
        tasks = [{"subquestion": question, "query": question,
                  "researcher": first, "rationale": "fallback"}]

    new_queries = [t["query"] for t in tasks]
    log.info("round=%s tasks=%s queries=%s", rnd, len(tasks), new_queries)
    return {
        "tasks": tasks,
        "queries_done": prior_queries + new_queries,
        "round": rnd + 1,
        "log": [f"🧭 Planner: round {rnd + 1} — {len(tasks)} subquestions: "
                + "; ".join(t['query'] for t in tasks[:4])],
    }


def _evidence_preview(evidence: list[dict], limit: int = 15) -> str:
    if not evidence:
        return "(no evidence yet)"
    lines = []
    for e in evidence[:limit]:
        mark = "" if e.get("supported", True) else " [UNSUPPORTED]"
        lines.append(f"- ({e.get('source_id', '?')}) {e.get('claim', '')[:140]}{mark}")
    return "\n".join(lines)
