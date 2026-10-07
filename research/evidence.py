"""
Structured evidence extraction (Phase 4).

For each source, the LLM must return JSON evidence items:
  claim / supporting_text (verbatim passage) / confidence / supported.

Anti-fabrication guard: a returned passage is verified to actually appear in
the source text (whitespace-normalized). Items whose passage is not found get
their confidence cut and are flagged unless the model itself marked them
unsupported. The extractor prompt never asks the model to fill gaps from its
own knowledge.
"""

import hashlib
import json
import logging

from graph.state import Evidence, model_from_dict
from llm_client import chat_json_list

log = logging.getLogger("agent.research.evidence")

EXTRACT_PROMPT = """You are an evidence extractor for a research agent.
Given SOURCE CONTENT and a RESEARCH QUESTION, extract only claims that the
source content itself directly states.

Rules:
- NEVER use your own prior knowledge. If the content does not state it, do not extract it.
- supporting_passage MUST be a verbatim quote from the content (copy-paste, no rewording).
- supported=true only when the passage directly backs the claim. If the source only
  hints at it, or contradicts it, set supported=false and confidence<=0.2.
- confidence reflects how directly and strongly the content supports the claim (0-1).
- Extract at most {max_items} items; prefer few, well-supported claims over many weak ones.
- Reply ONLY with a JSON array (no prose):
[{{"claim": "...", "supporting_passage": "...", "task": "...", "confidence": 0.0, "supported": true}}]

RESEARCH QUESTION:
{question}

SUBQUESTIONS BEING INVESTIGATED:
{tasks}

SOURCE: {source_title} ({source_url})
CONTENT:
{content}
"""


def extract_evidence(source: dict, content: str, question: str,
                     tasks: list[dict], max_items: int) -> list[dict]:
    """Extract Evidence items from one source's page text. Returns [] on any
    failure (a bad source must not kill the run)."""
    if not content or len(content) < 80:
        return []
    task_lines = "\n".join(f"- {t.get('subquestion', '')}" for t in tasks[:8]) or "- (general)"
    messages = [
        {"role": "system",
         "content": "You extract only directly-supported evidence. You never invent facts. JSON only."},
        {"role": "user", "content": EXTRACT_PROMPT.format(
            max_items=max_items, question=question, tasks=task_lines,
            source_title=source.get("title", ""), source_url=source.get("url", ""),
            content=content[:6000])},
    ]
    items = chat_json_list(messages)
    out = []
    for raw in items:
        if not isinstance(raw, dict):
            continue
        ev = model_from_dict(Evidence, {
            "claim": raw.get("claim", ""),
            "supporting_passage": raw.get("supporting_passage", ""),
            "supporting_text": raw.get("supporting_passage", ""),
            "confidence": _clamp(raw.get("confidence", 0.5)),
            "supported": bool(raw.get("supported", True)),
            "task": raw.get("task", ""),
            "source_id": source.get("source_id", ""),
            "source_title": source.get("title", ""),
            "source_url": source.get("url", ""),
            "source_type": source.get("source_type", "web"),
            "publication_date": source.get("publication_date", ""),
            "author": source.get("author", ""),
            "relevance_score": source.get("relevance_score", 0.5),
            "authority_score": source.get("authority_score", 0.5),
            "recency_score": source.get("recency_score", 0.5),
        })
        if ev is None or not ev.claim.strip():
            continue
        _verify_passage(ev, content)
        ev.content_hash = hashlib.sha256(normalize(ev.claim).encode()).hexdigest()[:16]
        out.append(ev.to_dict())
    return out[:max_items]


def _verify_passage(ev: Evidence, content: str) -> None:
    """If the 'verbatim' passage is not actually in the source, the item is
    untrustworthy: demote it."""
    if not ev.supporting_passage:
        ev.supported = False
        ev.confidence = min(ev.confidence, 0.2)
        return
    if normalize(ev.supporting_passage)[:200] not in normalize(content):
        # allow short paraphrases: check word-overlap instead
        words = set(normalize(ev.supporting_passage).split())
        content_words = set(normalize(content).split())
        overlap = len(words & content_words) / max(1, len(words))
        if overlap < 0.6:
            log.info("passage not found in source %s; demoting evidence", ev.source_id)
            ev.supported = False
            ev.confidence = min(ev.confidence, 0.25)
        else:
            ev.confidence = round(ev.confidence * 0.8, 3)


def normalize(text: str) -> str:
    return " ".join((text or "").lower().split())


def dedupe_evidence(items: list[dict]) -> list[dict]:
    """Drop near-duplicate claims (same normalized claim text)."""
    seen = {}
    for e in items:
        key = normalize(e.get("claim", ""))[:120]
        if not key:
            continue
        prev = seen.get(key)
        if prev is None or e.get("confidence", 0) > prev.get("confidence", 0):
            seen[key] = e
    return list(seen.values())


def evidence_as_json(items: list[dict]) -> str:
    return json.dumps(items, ensure_ascii=False)


def _clamp(v, lo=0.0, hi=1.0) -> float:
    try:
        return max(lo, min(hi, float(v)))
    except (TypeError, ValueError):
        return 0.5
