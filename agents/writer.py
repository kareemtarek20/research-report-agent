"""
Report Writer node (Phase 11).

Writes the final report ONLY from structured evidence, using inline [S#]
citations that reference the source registry. The Sources section is
appended programmatically from the registry, so cited URLs are always real
retrieved sources — the writer cannot fabricate them. Supports a revision
pass driven by the Report Quality Evaluator's feedback.
"""

import logging

from research.modes import get_mode
from research.sources import format_source_registry

log = logging.getLogger("agent.writer")

SYSTEM = ("You are a rigorous research report writer. Every factual statement "
          "must cite retrieved evidence with [S#] markers. You never add facts "
          "from your own knowledge that are not in the evidence. If evidence is "
          "missing, weak or conflicting, you say so plainly.")

PROMPT = """Write a research report in Markdown.

RESEARCH QUESTION: {question}
MODE: {mode_label} — {mode_desc}

REQUIRED SECTIONS (use these exact '##' headings, in order; omit a section only
if instructed otherwise by the mode guidance):
{sections}

RULES:
- Cite evidence inline as [S1], [S2]... immediately after each factual claim.
  Example: "Transformer models use self-attention [S3]."
- Use ONLY the evidence provided below and ONLY these citation ids.
- Do NOT write a Sources list — it is appended automatically.
- Claims marked CONTRADICTED or UNSUPPORTED by fact checks may appear only in
  'Conflicting Evidence' or 'Limitations', clearly flagged as such.
- Describe contradictions transparently (both sides + possible reasons);
  never silently pick one source.
- Write Methodology from the facts given (mode, rounds, queries, source types).
- Do not invent anything not present in the evidence.

{writer_guidance}

EVIDENCE (id | confidence | status | claim | source):
{evidence_block}

FACT CHECKS:
{factcheck_block}

CONTRADICTIONS:
{contradiction_block}

RESEARCH GAPS / LIMITATIONS TO DISCLOSE:
{gap_block}

METHOD FACTS: rounds={rounds}; queries executed: {queries};
sources by type: {source_type_counts}

{revision_block}

Report:
"""

REVISION_BLOCK = """PREVIOUS DRAFT (revise it — fix the reviewer issues, keep
everything already well-supported):

REVIEWER ISSUES:
{issues}

PREVIOUS DRAFT:
{draft}
"""


def writer_node(state: dict) -> dict:
    mode = get_mode(state.get("mode", "deep"))
    evidence = state.get("evidence", [])
    sources = state.get("sources", [])
    fact_checks = state.get("fact_checks", [])
    contradictions = state.get("contradictions", [])
    gaps = state.get("gaps", [])

    status_by_hash = {fc.get("claim", ""): fc.get("status") for fc in fact_checks}
    ev_lines = []
    for e in sorted(evidence, key=lambda x: (x.get("source_id", "S99"),
                                             -x.get("confidence", 0))):
        fc_status = status_by_hash.get(e.get("claim", ""), "")
        flag = e.get("confidence", 0) >= 0.6 and e.get("supported", True)
        tag = "SUPPORTED" if flag else "WEAK"
        if fc_status:
            tag = f"FACT-CHECK: {fc_status.upper()}"
        ev_lines.append(
            f"- [{e.get('source_id', '?')}] (conf {e.get('confidence', 0):.2f}; {tag}) "
            f"{e.get('claim', '')}\n    passage: {e.get('supporting_passage', '')[:200]}")
    evidence_block = "\n".join(ev_lines) or "(no evidence — write an honest " \
        "limitations-focused report)"

    fc_block = "\n".join(
        f"- [{fc.get('status')}] {fc.get('claim', '')[:120]} — {fc.get('explanation', '')[:160]}"
        for fc in fact_checks) or "(none)"

    con_block = "\n".join(
        f"- {c.get('topic', '')}: \"{c.get('claim_a', '')[:100]}\" [{c.get('source_a', '')}] "
        f"vs \"{c.get('claim_b', '')[:100]}\" [{c.get('source_b', '')}] "
        f"(reasons: {', '.join(c.get('possible_reasons', [])) or 'unknown'}; {c.get('status')})"
        for c in contradictions) or "(no contradictions detected)"

    gap_block = "\n".join(
        f"- ({g.get('importance', 'medium')}) {g.get('gap', '')}" for g in gaps) or "(none)"

    from collections import Counter
    stc = ", ".join(f"{k}: {v}" for k, v in
                    Counter(s.get("source_type", "") for s in sources).items()) or "(none)"

    revision_block = ""
    feedback = state.get("report_feedback") or []
    if state.get("report", "") and feedback:
        revision_block = REVISION_BLOCK.format(
            issues="\n".join(f"- {i}" for i in feedback),
            draft=state["report"][:12000])

    prompt = PROMPT.format(
        question=state["question"], mode_label=mode.label,
        mode_desc=mode.description,
        sections="\n".join(f"## {s}" for s in mode.report_sections),
        writer_guidance=mode.writer_guidance or "-",
        evidence_block=evidence_block[:14000], factcheck_block=fc_block[:3000],
        contradiction_block=con_block[:3000], gap_block=gap_block[:2500],
        rounds=state.get("round", 1),
        queries=", ".join(state.get("queries_done", [])[:12]) or "(none)",
        source_type_counts=stc, revision_block=revision_block)

    from llm_client import chat
    try:
        body = chat([{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": prompt}])
    except Exception as e:
        log.error("writer LLM failed: %s", e)
        body = ("## Executive Summary\n\nReport generation failed "
                f"({type(e).__name__}). Evidence collected: {len(evidence)} items.")

    body = _strip_fake_sources_section(body)
    registry = format_source_registry(_cited_sources(sources, body)) if sources else ""
    report = body.rstrip() + "\n\n## Sources\n\n" + (registry or "(no sources retrieved)") \
        + "\n"

    revisions = state.get("revisions", 0) + (1 if feedback else 0)
    log.info("report drafted (chars=%s, revision=%s)", len(report), revisions)
    return {"report": report, "revisions": revisions,
            "report_feedback": [],
            "log": [f"✍️ Writer: report drafted ({len(report)} chars, "
                    f"revision {revisions})"]}


def _cited_sources(sources: list[dict], report: str) -> list[dict]:
    """Only include sources actually cited in the report; if the writer cited
    nothing, include the top-quality sources so evidence stays traceable."""
    cited = [s for s in sources if f"[{s['source_id']}]" in report]
    if cited:
        return cited
    return sorted(sources, key=lambda s: s.get("quality_score", 0), reverse=True)[:10]


def _strip_fake_sources_section(body: str) -> str:
    """Remove any LLM-written '## Sources' block; ours is appended from the
    real registry so URLs can't be hallucinated."""
    import re
    return re.split(r"^##\s+Sources\b.*", body, flags=re.DOTALL | re.MULTILINE)[0].rstrip()
