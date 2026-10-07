"""
Document researcher node.

Runs BEFORE the web/academic/technical fan-out whenever the user uploads
files, so the run is grounded in the user's own material first: the most
relevant chunks are turned into `user_document` sources and their evidence
is extracted immediately. Web research then only supplements what the
documents already answer.
"""

import logging
import re
from concurrent.futures import ThreadPoolExecutor

from config import MAX_DOC_CHUNKS, MAX_EVIDENCE_PER_SOURCE, RESEARCH_WORKERS
from agents.evidence_agent import _score_item
from research.evidence import extract_evidence
from research.sources import findings_to_sources

log = logging.getLogger("agent.documents")


def document_node(state: dict) -> dict:
    question = state["question"]
    tasks = state.get("tasks", [])
    docs = state.get("documents") or []
    if not docs:
        return {"doc_findings": [], "log": ["📄 No uploaded documents to search"]}

    ranked = sorted(docs, key=lambda f: _relevance(question, f), reverse=True)
    selected = ranked[:MAX_DOC_CHUNKS]
    files = sorted({f.get("title", "?") for f in selected})

    sources = findings_to_sources(selected, question,
                                  existing=state.get("sources", []))
    existing_ids = {s["source_id"] for s in state.get("sources", [])}
    doc_sources = [s for s in sources if s["source_id"] not in existing_ids]

    log_lines = [f"📄 Document researcher: searched {len(docs)} chunks in "
                 f"{len(files)} uploaded file(s) — answering from your documents first",
                 f"   🔎 kept the {len(selected)} most relevant chunks "
                 f"({', '.join(files)})"]

    by_id = {s["source_id"]: s for s in sources}
    content_by_url = {s["url"]: _content_of(s["url"], selected) for s in doc_sources}
    with ThreadPoolExecutor(max_workers=max(1, RESEARCH_WORKERS)) as ex:
        batches = list(ex.map(
            lambda src: extract_evidence(src, content_by_url.get(src["url"], ""),
                                         question, tasks, MAX_EVIDENCE_PER_SOURCE),
            doc_sources))
    doc_evidence: list[dict] = []
    for src, items in zip(doc_sources, batches):
        if items:
            doc_evidence.extend(items)
            log_lines.append(f"   {src['source_id']} ({src['title']}) → "
                             f"{len(items)} evidence items from your documents")

    doc_evidence = [_score_item(e, by_id.get(e.get("source_id", ""), {}))
                    for e in doc_evidence]
    log.info("documents=%s chunks_used=%s evidence=%s", len(files),
             len(selected), len(doc_evidence))
    return {"doc_findings": selected,
            "sources": sources,
            "evidence": state.get("evidence", []) + doc_evidence,
            "log": log_lines}


def _relevance(question: str, chunk: dict) -> float:
    """Fraction of distinct question terms that appear in the chunk."""
    text = ((chunk.get("title") or "") + " " +
            (chunk.get("content") or "")).lower()
    terms = {t for t in re.findall(r"[a-z0-9]+", question.lower()) if len(t) > 2}
    if not terms:
        return 0.0
    return sum(1 for t in terms if t in text) / len(terms)


def _content_of(url: str, chunks: list[dict]) -> str:
    for c in chunks:
        if c.get("url") == url:
            return c.get("content") or c.get("snippet", "")
    return ""
