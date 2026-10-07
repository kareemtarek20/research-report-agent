"""
Contradiction detection (Phase 6).

Different sources disagreeing about a fact must be surfaced, not silently
resolved. Claims are grouped by subquestion; the LLM identifies genuinely
conflicting pairs (numbers, requirements, dates, causal claims) and reasons
for the disagreement (versions, hardware, datasets, methodology, dates).
"""

import logging

from graph.state import Contradiction, model_from_dict
from llm_client import chat_json_list

log = logging.getLogger("agent.research.contradictions")

DETECT_PROMPT = """You are a contradiction detector for a research agent.
Below is a list of evidence claims, each tied to a source id.

Find PAIRS of claims that genuinely conflict about the SAME fact
(different numbers, requirements, capabilities, dates, causal claims).
Do NOT flag claims that merely differ in topic or phrasing.

For each conflict, list plausible reasons (different product versions,
different hardware/configurations, different datasets, different measurement
methodology, different publication dates, one source being a misreading).

Reply ONLY with a JSON array:
[{{"topic": "...", "claim_a": "...", "source_a": "S1", "claim_b": "...",
  "source_b": "S2", "possible_reasons": ["..."], "status": "unresolved"}}]
Return [] if there are no genuine conflicts.

CLAIMS:
{claims}
"""


def detect_contradictions(evidence: list[dict]) -> list[dict]:
    """Ask the LLM to find conflicts among claims. Never raises."""
    if len(evidence) < 2:
        return []
    lines = []
    for e in evidence:
        flag = "" if e.get("supported", True) else " [marked unsupported]"
        lines.append(f"- ({e.get('source_id', '?')}) {e.get('claim', '')}{flag}")
    messages = [
        {"role": "system", "content": "You detect genuine factual conflicts. JSON only."},
        {"role": "user", "content": DETECT_PROMPT.format(claims="\n".join(lines))},
    ]
    items = chat_json_list(messages)
    out = []
    for raw in items:
        c = model_from_dict(Contradiction, raw) if isinstance(raw, dict) else None
        if c is None or not c.claim_a or not c.claim_b:
            continue
        if c.status not in ("resolved", "unresolved"):
            c.status = "unresolved"
        out.append(c.to_dict())
    return out


def compute_agreement(evidence: list[dict], sources: list[dict]) -> None:
    """Cross-source agreement: for each distinct claim, count how many
    DIFFERENT sources support it. Update each source's agreement_score and
    quality in place (list-of-dicts state form)."""
    from research.sources import normalize_url
    from research.scoring import combined_quality, quality_tier

    claim_support: dict[str, set] = {}
    for e in evidence:
        if not e.get("supported", True):
            continue
        key = " ".join(e.get("claim", "").lower().split())[:80]
        claim_support.setdefault(key, set()).add(e.get("source_url", ""))

    multi = sum(1 for urls in claim_support.values() if len(urls) >= 2)
    total = max(1, len(claim_support))
    by_url: dict[str, list[dict]] = {}
    for s in sources:
        by_url.setdefault(normalize_url(s["url"]), []).append(s)

    for urls in claim_support.values():
        if len(urls) >= 2:
            for u in urls:
                for s in by_url.get(normalize_url(u), []):
                    s["agreement_score"] = round(min(1.0, s.get("agreement_score", 0)
                                                     + 1.0 / total), 3)
    for s in sources:
        s["quality_score"] = round(0.85 * s["quality_score"]
                                   + 0.15 * s.get("agreement_score", 0.0), 3)
        s["quality_tier"] = quality_tier(s["quality_score"])
