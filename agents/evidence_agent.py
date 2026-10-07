"""
Evidence Extraction + Evidence Quality Scoring node.

Convergence point of the parallel researchers. Builds the deduplicated
source registry (with deterministic quality scores), extracts structured
evidence from every NEW source, folds in uploaded documents and previous
research memory, and enforces hard caps on sources and evidence count.
"""

import logging

from config import MAX_SOURCES, MAX_TOTAL_EVIDENCE, MAX_EVIDENCE_PER_SOURCE
from graph.state import Evidence
from research.evidence import extract_evidence, dedupe_evidence
from research.sources import findings_to_sources
from memory.vector_store import get_memory

log = logging.getLogger("agent.evidence")


def evidence_node(state: dict) -> dict:
    question = state["question"]
    tasks = state.get("tasks", [])

    # 1. Gather findings from all branches (+ user documents + memory)
    findings = []
    for key in ("web_findings", "academic_findings", "technical_findings",
                "doc_findings"):
        findings.extend(state.get(key) or [])
    doc_findings = state.get("documents") or []
    if doc_findings and not state.get("doc_findings"):
        findings.extend(doc_findings)

    memory_findings = get_memory().recall(question, top_k=5)
    if memory_findings:
        findings.extend(memory_findings)

    # 2. Merge into the persistent, scored source registry (capped)
    sources = findings_to_sources(findings, question, existing=state.get("sources", []))
    if len(sources) > MAX_SOURCES:
        sources = _cap_sources(sources)

    by_id = {s["source_id"]: s for s in sources}
    already_extracted = {e.get("source_id") for e in state.get("evidence", [])}

    # 3. Extract structured evidence from each not-yet-processed source
    new_evidence: list[dict] = []
    log_lines = [f"📚 Sources registry: {len(sources)} total"]
    for src in sources:
        if src["source_id"] in already_extracted:
            continue
        if src["source_type"] == "previous_research":
            new_evidence.extend(_evidence_from_memory(src))
            continue
        content = _content_for(findings, src)
        if not content:
            content = src.get("snippet", "")
        items = extract_evidence(src, content, question, tasks, MAX_EVIDENCE_PER_SOURCE)
        if items:
            new_evidence.extend(items)
            log_lines.append(f"   {src['source_id']} ({src['quality_tier']}) → "
                             f"{len(items)} evidence items")

    # 4. Quality-score evidence and keep the best up to the hard cap
    scored = [_score_item(e, by_id.get(e.get("source_id", ""), {})) for e in new_evidence]
    combined = state.get("evidence", []) + scored
    combined = dedupe_evidence(combined)
    if len(combined) > MAX_TOTAL_EVIDENCE:
        combined.sort(key=lambda e: e.get("confidence", 0) * e.get("authority_score", 0.5),
                      reverse=True)
        combined = combined[:MAX_TOTAL_EVIDENCE]

    log_lines.append(
        f"🧪 Evidence: +{len(scored)} new, {len(combined)} total "
        f"(incl. {len(doc_findings)} uploaded doc chunks, "
        f"{len(memory_findings)} memory hits)")
    log.info("sources=%s new_evidence=%s total=%s", len(sources), len(scored), len(combined))
    return {"sources": sources, "evidence": combined,
            "doc_findings": doc_findings or state.get("doc_findings") or [],
            "log": log_lines}


def _cap_sources(sources: list[dict]) -> list[dict]:
    """Keep all previously-registered sources; trim weakest NEW ones."""
    keep_high = [s for s in sources if s["quality_score"] >= 0.5
                 or s["source_type"] in ("user_document", "academic")]
    rest = [s for s in sources if s not in keep_high]
    rest.sort(key=lambda s: s["quality_score"], reverse=True)
    room = max(0, MAX_SOURCES - len(keep_high))
    kept = keep_high + rest[:room]
    kept.sort(key=lambda s: int(s["source_id"][1:]))
    return kept


def _content_for(findings: list[dict], source: dict) -> str:
    from research.sources import normalize_url
    key = normalize_url(source["url"])
    chunks = []
    for f in findings:
        if normalize_url(f.get("url", "")) == key and f.get("content"):
            chunks.append(f["content"])
    return "\n".join(chunks)


def _evidence_from_memory(source: dict) -> list[dict]:
    """Memory hits arrive as claim+passage text; wrap them as low-confidence
    prior-research evidence rather than re-running extraction."""
    text = source.get("snippet", "")
    claim = text.split("\n")[0].strip() if text else ""
    if not claim:
        return []
    ev = Evidence(
        claim=claim,
        supporting_passage=text[:600],
        supporting_text=text[:600],
        source_id=source["source_id"],
        source_title=source["title"],
        source_url=source["url"],
        source_type="previous_research",
        confidence=0.4,
        supported=True,
        authority_score=source.get("authority_score", 0.5),
        relevance_score=source.get("relevance_score", 0.5),
        recency_score=source.get("recency_score", 0.5),
    )
    return [ev.to_dict()]


def _score_item(e: dict, source: dict) -> dict:
    """Evidence quality = extraction confidence gated by source quality."""
    src_quality = source.get("quality_score", 0.5)
    if source:
        e["authority_score"] = source.get("authority_score", e.get("authority_score", 0.5))
        e["recency_score"] = source.get("recency_score", e.get("recency_score", 0.5))
        e["relevance_score"] = max(e.get("relevance_score", 0.5),
                                   source.get("relevance_score", 0.0))
    base_conf = e.get("confidence", 0.5)
    e["confidence"] = round(base_conf * (0.5 + 0.5 * src_quality), 3)
    if not e.get("supported", True):
        e["confidence"] = min(e["confidence"], 0.2)
    return e
