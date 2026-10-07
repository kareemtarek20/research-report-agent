"""
Fact Checker node (Phase 5).

Independently re-verifies the most important claims: for each claim it runs
a FRESH targeted web/academic search and asks the LLM to judge the claim
against the newly retrieved snippets and evidence from OTHER sources —
explicitly not trusting the original researcher. Results never raise: a
failed check degrades to "unverifiable".
"""

import logging
from concurrent.futures import ThreadPoolExecutor

from config import MAX_FACT_CHECKS, RESEARCH_WORKERS
from graph.state import FactCheckResult, model_from_dict
from llm_client import chat_json
from research.evidence import normalize
from tools.web_search import web_search

log = logging.getLogger("agent.factchecker")

SYSTEM = ("You are an independent fact checker. You do NOT trust the researcher "
          "who collected a claim. Judge the claim ONLY against freshly retrieved "
          "material and evidence from OTHER sources. If material is insufficient, "
          "say 'unverifiable'. JSON only.")

PROMPT = """CLAIM: {claim}

FRESHLY RETRIEVED SEARCH RESULTS (independent of the original research):
{search_hits}

EVIDENCE FROM OTHER SOURCES IN OUR POOL:
{other_evidence}

Classify the claim. Reply ONLY with JSON:
{{"status": "supported|partially_supported|contradicted|unverifiable",
  "confidence": 0.0,
  "supporting_sources": ["url or S-id", ...],
  "contradicting_sources": ["url or S-id", ...],
  "explanation": "one or two sentences"}}
"""


def fact_check_node(state: dict) -> dict:
    evidence = state.get("evidence", [])
    existing_checks = state.get("fact_checks", [])
    checked_hashes = {c.get("claim_hash", "") for c in existing_checks}
    n = _remaining_checks(state.get("mode", "deep"))

    # Rank by importance: confidence x authority; skip already-checked claims.
    candidates = [e for e in evidence
                  if _hash(e.get("claim", "")) not in checked_hashes]
    candidates.sort(key=lambda e: e.get("confidence", 0) * e.get("authority_score", 0.5),
                    reverse=True)

    new_checks = []
    log_lines = []
    batch = candidates[:n]
    if batch:
        with ThreadPoolExecutor(max_workers=max(1, RESEARCH_WORKERS)) as ex:
            results = list(ex.map(_check_one_safe, batch,
                                  [evidence] * len(batch)))
        for e, result in zip(batch, results):
            if result is None:
                continue
            result["claim_hash"] = _hash(e.get("claim", ""))
            new_checks.append(result)
            log_lines.append(f"🔍 Fact check [{result['status']}] {e.get('claim', '')[:80]}")

    if not log_lines:
        log_lines = [f"🔍 Fact checker: {len(existing_checks)} claims already verified, "
                     "nothing new to check"]
    log.info("fact_checks_new=%s total=%s", len(new_checks),
             len(existing_checks) + len(new_checks))
    return {"fact_checks": existing_checks + new_checks, "log": log_lines}


def _check_one_safe(e: dict, all_evidence: list[dict]) -> dict | None:
    try:
        return _check_one(e, all_evidence)
    except Exception as ex:
        log.warning("fact check crashed for %r: %s", e.get("claim", "")[:60], ex)
        return None


def _check_one(e: dict, all_evidence: list[dict]) -> dict | None:
    claim = e.get("claim", "")
    try:
        hits = web_search(claim[:200])[:4]
    except Exception as ex:
        log.warning("fact-check search failed for %r: %s", claim[:60], ex)
        hits = []
    search_txt = "\n".join(
        f"- {h.get('title', '')} ({h.get('url', '')}): {h.get('snippet', '')[:300]}"
        for h in hits) or "(no fresh results — search unavailable)"

    own_url = normalize(e.get("source_url", ""))
    others = [o for o in all_evidence
              if normalize(o.get("source_url", "")) != own_url
              and o.get("supported", True)][:6]
    others_txt = "\n".join(
        f"- ({o.get('source_id', '?')}) {o.get('claim', '')[:160]}" for o in others
    ) or "(none)"

    try:
        raw = chat_json([
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": PROMPT.format(
                claim=claim, search_hits=search_txt[:2500],
                other_evidence=others_txt[:1500])},
        ])
    except Exception as ex:
        log.warning("fact-check LLM failed for %r: %s", claim[:60], ex)
        raw = {"status": "unverifiable", "confidence": 0.0,
               "explanation": f"Checker unavailable ({type(ex).__name__}); not verified."}

    fc = model_from_dict(FactCheckResult, {
        "claim": claim,
        "status": raw.get("status", "unverifiable")
        if raw.get("status") in ("supported", "partially_supported",
                                 "contradicted", "unverifiable") else "unverifiable",
        "confidence": _clamp(raw.get("confidence", 0.0)),
        "supporting_sources": _as_list(raw.get("supporting_sources")),
        "contradicting_sources": _as_list(raw.get("contradicting_sources")),
        "explanation": raw.get("explanation", ""),
        "evidence_source_id": e.get("source_id", ""),
    })
    return fc.to_dict() if fc else None


def _remaining_checks(mode_key: str) -> int:
    from research.modes import get_mode
    return max(0, min(MAX_FACT_CHECKS, get_mode(mode_key).fact_checks))


def _hash(claim: str) -> str:
    return normalize(claim)[:120]


def _as_list(v):
    if isinstance(v, list):
        return [str(x)[:200] for x in v][:6]
    if isinstance(v, str) and v:
        return [v[:200]]
    return []


def _clamp(v):
    try:
        return max(0.0, min(1.0, float(v)))
    except (TypeError, ValueError):
        return 0.0
