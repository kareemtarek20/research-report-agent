"""
Research Intelligence Agent — LangGraph workflow.

    USER QUESTION
      -> planner
      -> FAN-OUT: web / academic / technical researchers (real parallel
         supersteps; each writes its own findings channel)
      -> evidence extraction + quality scoring (convergence)
      -> fact checker (independent re-verification)
      -> contradiction detector
      -> research critic ---[high gaps & rounds left]--> planner (bounded loop)
      -> report writer
      -> report quality evaluator ---[fail & revisions left]--> writer (bounded)
      -> finalize (metadata + optional memory store) -> END

Every loop has a hard counter bound, so the graph always terminates.
"""

import logging
import time

from langgraph.graph import StateGraph, END

from graph.state import ResearchState, ReportMetadata
from graph.routing import route_researchers, after_critic, after_report_quality
from llm_client import usage_meter
from logging_setup import setup_logging, NodeTimer
from research.modes import get_mode

from agents.planner import plan_node
from agents.web_researcher import web_researcher_node
from agents.academic_researcher import academic_researcher_node
from agents.technical_researcher import technical_researcher_node
from agents.evidence_agent import evidence_node
from agents.fact_checker import fact_check_node
from agents.contradiction_agent import contradiction_node
from agents.critic import critic_node
from agents.writer import writer_node
from agents.report_critic import evaluate_report_node

log = logging.getLogger("agent.workflow")


def resilient(name: str, fn):
    """A crashing node must degrade the run, not end it (Phase 17)."""

    def wrapper(state: dict) -> dict:
        try:
            return fn(state)
        except Exception as e:
            log.exception("node %s failed", name)
            return {"log": [f"⚠️ {name} failed ({type(e).__name__}: {str(e)[:120]}) "
                            "— continuing with what we have"],
                    "errors": [f"{name}: {type(e).__name__}: {str(e)[:200]}"]}
    wrapper.__name__ = name
    return wrapper


def finalize_node(state: dict, stats: dict) -> dict:
    """Compile report metadata, persist research memory, log the run summary."""
    from memory.vector_store import get_memory

    quality = state.get("report_quality", {}) or {}
    meta = ReportMetadata(
        question=state["question"],
        mode=state.get("mode", "deep"),
        source_count=len(state.get("sources", [])),
        evidence_count=len(state.get("evidence", [])),
        claims_verified=len(state.get("fact_checks", [])),
        contradictions_found=len(state.get("contradictions", [])),
        gaps_found=len(state.get("gaps", [])),
        research_rounds=state.get("round", 0),
        quality=quality,
    )
    stored = get_memory().store_run(
        state["question"], meta.mode, state.get("evidence", []),
        state.get("sources", []), report_summary=state.get("report", "")[:2000])

    stats["duration_s"] = round(time.time() - stats.pop("t0", time.time()), 1)
    stats["retry_count"] = usage_meter.snapshot().get("llm_retries", 0)
    stats["memory_items_stored"] = stored
    meta_dict = meta.to_dict()
    meta_dict["stats"] = {k: v for k, v in stats.items() if k != "node_times"}
    meta_dict["node_times_s"] = stats.get("node_times", {})
    meta_dict["llm_usage"] = usage_meter.snapshot()

    lines = [f"🏁 Finalized: {meta.source_count} sources, {meta.evidence_count} "
             f"evidence items, quality {quality.get('overall', 0):.0%}, "
             f"{stats['duration_s']}s, {stored} items saved to memory"]
    log.info("FINAL question=%r sources=%s evidence=%s quality=%.2f "
             "duration_s=%s errors=%s", state["question"][:80], meta.source_count,
             meta.evidence_count, quality.get("overall", 0), stats["duration_s"],
             len(state.get("errors", [])))
    return {"metadata": meta_dict, "log": lines}


def build_graph(stats: dict):
    graph = StateGraph(ResearchState)

    timed_nodes = {
        "plan": plan_node,
        "web_researcher": web_researcher_node,
        "academic_researcher": academic_researcher_node,
        "technical_researcher": technical_researcher_node,
        "evidence": evidence_node,
        "fact_check": fact_check_node,
        "contradictions": contradiction_node,
        "critic": critic_node,
        "writer": writer_node,
        "evaluate": evaluate_report_node,
    }
    for name, fn in timed_nodes.items():
        def make(nm, f):
            def timed(state, _f=f, _nm=nm):
                with NodeTimer(_nm, stats):
                    return _f(state)
            return timed
        graph.add_node(name, resilient(name, make(name, fn)))

    graph.add_node("finalize", resilient("finalize",
                                         lambda s: finalize_node(s, stats)))

    graph.set_entry_point("plan")
    graph.add_conditional_edges("plan", route_researchers,
                                ["web_researcher", "academic_researcher",
                                 "technical_researcher"])
    graph.add_edge("web_researcher", "evidence")
    graph.add_edge("academic_researcher", "evidence")
    graph.add_edge("technical_researcher", "evidence")
    graph.add_edge("evidence", "fact_check")
    graph.add_edge("fact_check", "contradictions")
    graph.add_edge("contradictions", "critic")
    graph.add_conditional_edges("critic", after_critic,
                                {"research": "plan", "write": "writer"})
    graph.add_edge("writer", "evaluate")
    graph.add_conditional_edges("evaluate", after_report_quality,
                                {"revise": "writer", "finalize": "finalize"})
    graph.add_edge("finalize", END)
    return graph.compile()


def run_agent(question: str, mode: str = None, documents: list = None):
    """Run research on a question. Yields state snapshots with a
    `current_nodes` hint so a UI can show live progress. The final snapshot
    contains `report`, `metadata` and all structured research objects.

    documents: optional list of (filename, bytes) uploaded files.
    """
    setup_logging()
    from config import DEFAULT_MODE, VALID_MODES
    from tools.document_loader import load_document

    if mode not in VALID_MODES:
        mode = DEFAULT_MODE

    doc_findings = []
    for filename, data in documents or []:
        try:
            doc_findings.extend(load_document(filename, data))
        except Exception as e:
            log.warning("document %s failed to load: %s", filename, e)

    stats = {"t0": time.time(), "node_times": {}, "mode": mode}
    app = build_graph(stats)
    initial: ResearchState = {
        "question": question, "mode": mode, "documents": doc_findings,
        "round": 0, "tasks": [], "queries_done": [],
        "web_findings": [], "academic_findings": [], "technical_findings": [],
        "doc_findings": [],
        "sources": [], "evidence": [], "fact_checks": [], "contradictions": [],
        "gaps": [], "assessment": {},
        "report": "", "report_feedback": [], "revisions": 0,
        "report_quality": {}, "quality_evaluations": [], "metadata": {},
        "log": [f"🚀 Research mode: {get_mode(mode).label} — {question}"],
        "errors": [],
    }

    current_nodes: list[str] = []
    for kind, chunk in app.stream(initial, stream_mode=["updates", "values"]):
        if kind == "updates":
            current_nodes = [k for k in chunk.keys() if k != "__interrupt__"]
        else:
            snapshot = dict(chunk)
            snapshot["current_nodes"] = current_nodes
            yield snapshot
