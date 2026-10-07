"""Structured research objects for the Research Intelligence Agent.

Replaces the old loose "notes" strings with typed evidence, sources, claims,
contradictions, gaps and assessments. Pydantic models are serialized to plain
dicts (`to_dict`) at LangGraph channel boundaries so state stays checkpoint-
friendly and exportable to JSON.
"""

import operator
from datetime import datetime, timezone
from typing import Annotated, Literal, Optional, TypedDict

from pydantic import BaseModel, ConfigDict, Field

ResearcherKind = Literal["web", "academic", "technical", "document"]
SourceType = Literal["web", "academic", "technical", "user_document", "previous_research"]
FactStatus = Literal["supported", "partially_supported", "contradicted", "unverifiable"]
Importance = Literal["high", "medium", "low"]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class _Model(BaseModel):
    model_config = ConfigDict(extra="ignore")

    def to_dict(self) -> dict:
        return self.model_dump()


class ResearchTask(_Model):
    """One subquestion the planner wants answered, assigned to a researcher."""

    subquestion: str
    query: str
    researcher: ResearcherKind = "web"
    rationale: str = ""


class ResearchSource(_Model):
    """A retrieved document (web page, paper, uploaded file, memory hit)."""

    source_id: str = ""  # "S1", "S2", ... assigned at aggregation
    title: str = ""
    url: str = ""
    source_type: SourceType = "web"
    domain: str = ""
    snippet: str = ""
    author: str = ""
    publication_date: str = ""
    # Quality dimensions (0.0 - 1.0), filled by research.scoring
    authority_score: float = 0.5
    relevance_score: float = 0.5
    recency_score: float = 0.5
    evidence_quality: float = 0.5
    agreement_score: float = 0.0  # cross-source agreement, filled after extraction
    is_primary: bool = False
    quality_score: float = 0.5
    quality_tier: str = "unknown"  # high | medium | low | weak
    retrieved_at: str = Field(default_factory=_now)


class Evidence(_Model):
    """A single extracted claim tied to one source. Never fabricated: if the
    source does not directly support the claim, confidence is 0 and
    supported=False."""

    claim: str
    supporting_text: str = ""
    supporting_passage: str = ""
    source_id: str = ""
    source_title: str = ""
    source_url: str = ""
    source_type: SourceType = "web"
    publication_date: str = ""
    author: str = ""
    relevance_score: float = 0.5
    authority_score: float = 0.5
    recency_score: float = 0.5
    confidence: float = 0.5
    supported: bool = True
    task: str = ""  # subquestion this evidence addresses
    content_hash: str = ""


class FactCheckResult(_Model):
    claim: str
    status: FactStatus = "unverifiable"
    confidence: float = 0.0
    supporting_sources: list[str] = Field(default_factory=list)
    contradicting_sources: list[str] = Field(default_factory=list)
    explanation: str = ""
    evidence_source_id: str = ""


class Contradiction(_Model):
    topic: str
    claim_a: str
    claim_b: str
    source_a: str = ""
    source_b: str = ""
    possible_reasons: list[str] = Field(default_factory=list)
    status: str = "unresolved"  # resolved | unresolved


class ResearchGap(_Model):
    gap: str
    importance: Importance = "medium"
    recommended_query: str = ""
    related_subquestion: str = ""


class ResearchAssessment(_Model):
    """Output of the Research Critic."""

    sufficient: bool = False
    reason: str = ""
    gaps: list[ResearchGap] = Field(default_factory=list)
    unanswered_subquestions: list[str] = Field(default_factory=list)
    weak_claims: list[str] = Field(default_factory=list)
    source_diversity: float = 0.0
    needs_fact_check_attention: bool = False


class ReportQuality(_Model):
    """Output of the Report Quality Evaluator (Phase 12)."""

    evidence_coverage: float = 0.0
    citation_coverage: float = 0.0
    source_quality: float = 0.0
    question_coverage: float = 0.0
    unsupported_claims: int = 0
    contradictions_handled: bool = True
    overall: float = 0.0
    passed: bool = False
    issues: list[str] = Field(default_factory=list)


class ReportMetadata(_Model):
    question: str
    mode: str
    generated_at: str = Field(default_factory=_now)
    source_count: int = 0
    evidence_count: int = 0
    claims_verified: int = 0
    contradictions_found: int = 0
    gaps_found: int = 0
    research_rounds: int = 0
    quality: Optional[dict] = None


def model_from_dict(model, data):
    """Best-effort parse of LLM JSON into a typed model; returns None instead
    of raising so a single malformed item never kills a research run."""
    try:
        if isinstance(data, model):
            return data
        return model(**data)
    except Exception:
        return None


class ResearchState(TypedDict, total=False):
    """LangGraph channels for the full research workflow.

    Nodes return PARTIAL updates. Parallel researchers each write their own
    findings channel, so fan-out is genuinely parallel without write
    conflicts; `log` and `errors` use an add-reducer because every node
    (including parallel ones) appends to them.
    """

    question: str
    mode: str
    documents: list[dict]            # uploaded-doc findings (input channel)

    round: int                         # research rounds executed
    tasks: list[dict]                  # ResearchTask dumps
    queries_done: list[str]

    web_findings: list[dict]           # raw per-researcher output (parallel)
    academic_findings: list[dict]
    technical_findings: list[dict]
    doc_findings: list[dict]

    sources: list[dict]                # ResearchSource dumps (scored registry)
    evidence: list[dict]               # Evidence dumps
    fact_checks: list[dict]            # FactCheckResult dumps (+claim_hash)
    contradictions: list[dict]         # Contradiction dumps
    gaps: list[dict]                   # ResearchGap dumps
    assessment: dict                   # ResearchAssessment dump

    report: str
    report_feedback: list[str]         # writer revision instructions
    revisions: int                     # completed report revisions
    report_quality: dict               # latest ReportQuality dump
    quality_evaluations: list[dict]    # every evaluation pass
    metadata: dict                     # ReportMetadata dump

    log: Annotated[list[str], operator.add]
    errors: Annotated[list[str], operator.add]
