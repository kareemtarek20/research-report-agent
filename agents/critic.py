"""
Research Critic node (Phase 7).

Judges evidence sufficiency: unanswered subquestions, source diversity,
weak/unsupported claims, contradictions, fact-check failures and outdated
evidence. Combines deterministic checks with an LLM assessment and returns
structured research gaps. High-priority gaps route the graph back to the
planner (bounded by the mode's max rounds).
"""

import logging

from graph.state import ResearchAssessment, ResearchGap, model_from_dict
from llm_client import chat_json

log = logging.getLogger("agent.critic")

SYSTEM = ("You are a research critic. You assess whether collected evidence "
          "answers a research question and identify concrete gaps. You are "
          "skeptical of weak, single-sourced, or outdated claims. JSON only.")

PROMPT = """RESEARCH QUESTION: {question}

SUBQUESTIONS PLANNED:
{subquestions}

EVIDENCE SUMMARY ({n_evidence} items from {n_sources} sources):
{evidence_lines}

SOURCE TYPES: {source_types}
FACT CHECK RESULTS: {fact_summary}
CONTRADICTIONS: {contradictions_summary}

Assess sufficiency. Reply ONLY with JSON:
{{"sufficient": true|false, "reason": "...",
 "gaps": [{{"gap": "...", "importance": "high|medium|low",
           "recommended_query": "...", "related_subquestion": "..."}}],
 "unanswered_subquestions": ["..."],
 "weak_claims": ["..."]}}
Return at most 4 gaps; only mark importance=high when the report would be
materially wrong or misleading without it."""


def critic_node(state: dict) -> dict:
    question = state["question"]
    evidence = state.get("evidence", [])
    sources = state.get("sources", [])
    tasks = state.get("tasks", [])
    fact_checks = state.get("fact_checks", [])
    contradictions = state.get("contradictions", [])

    det_gaps = _deterministic_gaps(state)
    assessment = _llm_assessment(state, det_gaps)

    merged_gaps = _dedupe_gaps(
        [g.to_dict() for g in assessment.gaps] + det_gaps)
    assessment.source_diversity = len({s.get("source_type") for s in sources}) / 4.0
    if fact_checks:
        bad = [fc for fc in fact_checks
               if fc.get("status") in ("contradicted", "unverifiable")]
        assessment.needs_fact_check_attention = len(bad) > len(fact_checks) / 2

    result = assessment.to_dict()
    result["gaps"] = merged_gaps
    high = [g for g in merged_gaps if g.get("importance") == "high"]
    log.info("sufficient=%s gaps=%s high=%s", assessment.sufficient,
             len(merged_gaps), len(high))
    lines = [f"🧐 Critic: {'sufficient' if assessment.sufficient else 'gaps found'} — "
             f"{len(merged_gaps)} gaps ({len(high)} high-priority)"]
    return {"assessment": result, "gaps": merged_gaps, "log": lines}


def _deterministic_gaps(state: dict) -> list[dict]:
    """Checks that don't need an LLM and can't be hallucinated away."""
    gaps = []
    evidence = state.get("evidence", [])
    tasks = state.get("tasks", [])
    sources = state.get("sources", [])

    covered = {normalize_task(e.get("task", "")) for e in evidence if e.get("supported", True)}
    for t in tasks:
        if normalize_task(t.get("subquestion", "")) not in covered and \
           not any(_overlap(t.get("subquestion", ""), e.get("task", "")) for e in evidence):
            gaps.append(ResearchGap(
                gap=f"No evidence collected for subquestion: {t.get('subquestion', '')[:120]}",
                importance="high",
                recommended_query=t.get("query", ""),
                related_subquestion=t.get("subquestion", ""),
            ).to_dict())

    types = {s.get("source_type") for s in sources}
    if "user_document" not in types and len(types) < 2 and sources:
        gaps.append(ResearchGap(
            gap="Evidence comes from a single source type; cross-source verification is weak.",
            importance="medium", recommended_query="", related_subquestion="").to_dict())

    for e in evidence:
        if e.get("recency_score", 1.0) < 0.3 and e.get("supported", True):
            gaps.append(ResearchGap(
                gap=f"Potentially outdated evidence: {e.get('claim', '')[:100]}",
                importance="low", recommended_query="",
                related_subquestion=e.get("task", "")).to_dict())
    return gaps[:4]


def _llm_assessment(state: dict, det_gaps: list[dict]) -> ResearchAssessment:
    question = state["question"]
    evidence = state.get("evidence", [])
    sources = state.get("sources", [])
    tasks = state.get("tasks", [])
    fact_checks = state.get("fact_checks", [])
    contradictions = state.get("contradictions", [])

    ev_lines = "\n".join(
        f"- ({e.get('source_id', '?')}, conf {e.get('confidence', 0):.2f}"
        f"{', UNSUPPORTED' if not e.get('supported', True) else ''}) {e.get('claim', '')[:150]}"
        for e in evidence[:25]) or "(none)"
    fc_summary = "\n".join(
        f"- [{fc.get('status')}] {fc.get('claim', '')[:100]}" for fc in fact_checks[:10]
    ) or "(none)"
    con_summary = "\n".join(
        f"- ({c.get('status')}): {c.get('claim_a', '')[:80]} VS {c.get('claim_b', '')[:80]}"
        for c in contradictions[:6]) or "(none)"

    prompt = PROMPT.format(
        question=question,
        subquestions="\n".join(f"- {t.get('subquestion', '')}" for t in tasks) or "(none)",
        n_evidence=len(evidence), n_sources=len(sources),
        evidence_lines=ev_lines or "(none)",
        source_types=", ".join(sorted({s.get('source_type', '') for s in sources})) or "(none)",
        fact_summary=fc_summary, contradictions_summary=con_summary)

    try:
        raw = chat_json([{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": prompt}])
    except Exception as e:
        log.warning("critic LLM failed (%s); relying on deterministic checks", e)
        raw = {}

    gaps = []
    for g in raw.get("gaps", []):
        m = model_from_dict(ResearchGap, g)
        if m and m.gap.strip():
            gaps.append(m)
    assessment = ResearchAssessment(
        sufficient=bool(raw.get("sufficient", False)),
        reason=raw.get("reason", ""),
        gaps=gaps,
        unanswered_subquestions=[str(x) for x in raw.get("unanswered_subquestions", [])][:6],
        weak_claims=[str(x) for x in raw.get("weak_claims", [])][:6],
    )
    # Hard rules the LLM cannot overrule: no supported evidence at all, or a
    # deterministic high-priority gap, means NOT sufficient.
    has_supported = any(e.get("supported", True) for e in evidence)
    if (det_gaps and any(g["importance"] == "high" for g in det_gaps)) or not has_supported:
        assessment.sufficient = False
    return assessment


def _dedupe_gaps(gaps: list[dict]) -> list[dict]:
    seen, out = set(), []
    order = {"high": 0, "medium": 1, "low": 2}
    gaps = sorted(gaps, key=lambda g: order.get(g.get("importance", "medium"), 1))
    for g in gaps:
        key = normalize_task(g.get("gap", ""))[:80]
        if key and key not in seen:
            seen.add(key)
            out.append(g)
    return out[:8]


def normalize_task(text: str) -> str:
    return " ".join((text or "").lower().split())


def _overlap(a: str, b: str) -> bool:
    ta = {w for w in normalize_task(a).split() if len(w) > 3}
    tb = {w for w in normalize_task(b).split() if len(w) > 3}
    if not ta or not tb:
        return False
    return len(ta & tb) / len(ta) >= 0.5
