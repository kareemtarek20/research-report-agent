"""
Report Quality Evaluator (Phase 12).

Deterministic metrics — citation coverage, evidence coverage, source quality
of cited sources, unsupported-claim handling — plus one LLM pass for
question coverage. Failing thresholds sends the report back to the writer for
ONE revision; a revision counter guarantees termination.
"""

import logging
import re

from config import (QUALITY_EVIDENCE_COVERAGE, QUALITY_CITATION_COVERAGE,
                    QUALITY_QUESTION_COVERAGE, QUALITY_SOURCE_QUALITY,
                    MAX_REPORT_REVISIONS)
from graph.state import ReportQuality
from llm_client import chat_json

log = logging.getLogger("agent.report_critic")

CITE_RE = re.compile(r"\[S(\d+)\]")
SENT_RE = re.compile(r"[^.!\n]+\.\s")


def evaluate_report_node(state: dict) -> dict:
    report = state.get("report", "")
    evidence = state.get("evidence", [])
    sources = state.get("sources", [])
    fact_checks = state.get("fact_checks", [])
    contradictions = state.get("contradictions", [])
    tasks = state.get("tasks", [])

    q = ReportQuality()

    # --- citation coverage: body sentences (excl. headings & Sources list) ---
    body = _body_without_sources(report)
    sentences = [s.strip() for s in SENT_RE.findall(body) if len(s.strip()) > 60]
    factual = [s for s in sentences if re.search(r"\d|\b(is|are|was|were|requires?|uses?)\b", s)]
    cited = [s for s in factual if CITE_RE.search(s)]
    q.citation_coverage = round(len(cited) / len(factual), 3) if factual else 1.0

    # --- evidence coverage: strong evidence claims reflected in the report ---
    strong = [e for e in evidence if e.get("supported", True)
              and e.get("confidence", 0) >= 0.5]
    rl = report.lower()
    covered = sum(1 for e in strong if _claim_in_report(e.get("claim", ""), rl))
    q.evidence_coverage = round(covered / len(strong), 3) if strong else 0.0

    # --- source quality of cited sources ---
    cited_ids = {f"S{n}" for n in CITE_RE.findall(report)}
    cited_src = [s for s in sources if s["source_id"] in cited_ids]
    q.source_quality = round(sum(s.get("quality_score", 0) for s in cited_src)
                             / len(cited_src), 3) if cited_src else 0.0

    # --- unsupported / contradicted handling ---
    unsup = [e for e in evidence if not e.get("supported", True)]
    q.unsupported_claims = len(unsup)
    q.contradictions_handled = (not contradictions
                                or ("onflict" in report or "isagree" in report
                                    or "esolve" in report))

    # --- question coverage (LLM, cheap single call, never fatal on error) ---
    q.question_coverage = _question_coverage(report, tasks, state["question"])

    q.issues = _issues(q)
    q.overall = round((0.30 * q.evidence_coverage + 0.30 * q.citation_coverage
                       + 0.20 * q.source_quality + 0.20 * q.question_coverage), 3)
    q.passed = (q.evidence_coverage >= QUALITY_EVIDENCE_COVERAGE
                and q.citation_coverage >= QUALITY_CITATION_COVERAGE
                and q.question_coverage >= QUALITY_QUESTION_COVERAGE
                and q.source_quality >= QUALITY_SOURCE_QUALITY
                and q.contradictions_handled)

    revs = state.get("revisions", 0)
    will_revise = (not q.passed) and revs < MAX_REPORT_REVISIONS
    log.info("report quality overall=%.2f passed=%s revise=%s", q.overall,
             q.passed, will_revise)
    lines = [f"📏 Report quality: evidence {q.evidence_coverage:.0%} | "
             f"citations {q.citation_coverage:.0%} | source quality "
             f"{q.source_quality:.0%} | question coverage {q.question_coverage:.0%}"
             + (" → PASS" if q.passed else f" → {'revision requested' if will_revise else 'accepted (revision limit)'}")]
    return {"report_quality": q.to_dict(),
            "report_feedback": q.issues if will_revise else [],
            "quality_evaluations": state.get("quality_evaluations", []) + [q.to_dict()],
            "log": lines}


def _issues(q: ReportQuality) -> list[str]:
    issues = []
    if q.evidence_coverage < QUALITY_EVIDENCE_COVERAGE:
        issues.append(f"Report omits too much collected evidence "
                      f"(coverage {q.evidence_coverage:.0%}, need "
                      f"{QUALITY_EVIDENCE_COVERAGE:.0%}). Incorporate the strong "
                      "evidence items into Key Findings/Evidence.")
    if q.citation_coverage < QUALITY_CITATION_COVERAGE:
        issues.append(f"Too many factual sentences lack [S#] citations "
                      f"({q.citation_coverage:.0%} coverage). Add inline "
                      "citations to every factual claim.")
    if q.question_coverage < QUALITY_QUESTION_COVERAGE:
        issues.append("Some subquestions are not addressed; cover them or "
                      "state explicitly in Limitations why evidence is missing.")
    if q.source_quality < QUALITY_SOURCE_QUALITY:
        issues.append("Cited sources are low quality; prefer the high/medium "
                      "tier sources when making key claims.")
    if not q.contradictions_handled:
        issues.append("Detected contradictions are not transparently discussed "
                      "in a Conflicting Evidence section.")
    return issues


def _body_without_sources(report: str) -> str:
    idx = report.rfind("\n## Sources")
    return report[:idx] if idx > 0 else report


def _claim_in_report(claim: str, report_lower: str) -> bool:
    """A claim counts as covered when a distinctive word-run of it appears."""
    words = [w for w in re.findall(r"[a-z0-9]+", claim.lower()) if len(w) > 3]
    if len(words) < 3:
        return claim.lower()[:40] in report_lower
    run = words[:5]
    if any(" ".join(run[i:i + 3]) in report_lower for i in range(len(run) - 2)):
        return True
    # fall back: most distinctive words present anywhere in the report
    return sum(1 for w in set(words) if w in report_lower) / len(set(words)) >= 0.7


def _question_coverage(report: str, tasks: list[dict], question: str) -> float:
    if not tasks:
        return 1.0
    try:
        raw = chat_json([
            {"role": "system", "content": "You check whether a report answers "
             "given subquestions. JSON only."},
            {"role": "user", "content":
                "For each subquestion, does the report address it (even to say "
                "evidence is missing)? Reply ONLY:\n"
                '{"answered": [{"subquestion": "...", "addressed": true|false}]}\n\n'
                f"SUBQUESTIONS:\n" + "\n".join(f"- {t.get('subquestion', '')}"
                                              for t in tasks[:10])
                + "\n\nREPORT:\n" + report[:12000]}])
        answers = raw.get("answered", [])
        if not answers:
            return 1.0
        yes = sum(1 for a in answers if isinstance(a, dict) and a.get("addressed"))
        return round(yes / len(answers), 3)
    except Exception as e:
        log.warning("question-coverage LLM check failed (%s); assuming 1.0", e)
        return 1.0
