"""
Research modes (Phase 10).

One workflow, seven modes. A mode only changes configuration: which
researchers fan out, how many rounds/queries are allowed, extra planning
guidance, the report section schema, and verification intensity.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ResearchMode:
    key: str
    label: str
    description: str
    researchers: tuple[str, ...]          # web | academic | technical
    max_queries_per_round: int
    max_rounds: int
    fact_checks: int                      # claims independently re-verified
    planner_guidance: str
    report_sections: tuple[str, ...]
    writer_guidance: str = ""


_COMMON_SECTIONS = (
    "Executive Summary", "Research Question", "Methodology", "Key Findings",
    "Evidence", "Conflicting Evidence", "Limitations", "Research Gaps",
    "Conclusions", "Sources",
)

MODES: dict[str, ResearchMode] = {
    "quick": ResearchMode(
        key="quick", label="Quick Research",
        description="Fast single-pass overview from web sources.",
        researchers=("web",),
        max_queries_per_round=2, max_rounds=1, fact_checks=3,
        planner_guidance="Produce only the 2 most decisive queries that cover the core of the question.",
        report_sections=("Executive Summary", "Research Question", "Key Findings",
                         "Evidence", "Limitations", "Sources"),
        writer_guidance="Keep it concise (800-1200 words). Skip Methodology and Conflicting Evidence sections unless contradictions exist.",
    ),
    "deep": ResearchMode(
        key="deep", label="Deep Research",
        description="Full multi-researcher pipeline with fact checking and reflection.",
        researchers=("web", "academic", "technical"),
        max_queries_per_round=3, max_rounds=2, fact_checks=8,
        planner_guidance="Decompose the question into 4-8 subquestions across the assigned research angles, each with one focused query.",
        report_sections=_COMMON_SECTIONS,
    ),
    "academic": ResearchMode(
        key="academic", label="Academic Literature Review",
        description="Prioritizes papers and scholarly sources; survey-style report.",
        researchers=("academic", "web"),
        max_queries_per_round=3, max_rounds=2, fact_checks=6,
        planner_guidance="Frame subquestions as literature-review questions (key methods, findings, debates, open problems). Prefer queries naming papers, authors, or venues.",
        report_sections=("Executive Summary", "Research Question", "Methodology",
                         "Key Findings", "Thematic Analysis", "Evidence",
                         "Conflicting Evidence", "Limitations", "Research Gaps",
                         "Conclusions", "Sources"),
        writer_guidance="Write as a literature review: organize by theme, attribute findings to papers, highlight methodological differences and citation chains.",
    ),
    "technical": ResearchMode(
        key="technical", label="Technical Investigation",
        description="Docs, benchmarks, implementations and constraints.",
        researchers=("technical", "web"),
        max_queries_per_round=3, max_rounds=2, fact_checks=6,
        planner_guidance="Target official documentation, benchmarks, requirements, APIs, known issues and implementation guidance.",
        report_sections=("Executive Summary", "Research Question", "Methodology",
                         "Key Findings", "Technical Details", "Evidence",
                         "Conflicting Evidence", "Limitations", "Research Gaps",
                         "Conclusions", "Sources"),
        writer_guidance="Emphasize concrete numbers, version requirements, configuration and caveats. Quote exact values from documentation.",
    ),
    "competitive": ResearchMode(
        key="competitive", label="Competitive Analysis",
        description="Compare entities (products, approaches, vendors) on consistent dimensions.",
        researchers=("web", "technical"),
        max_queries_per_round=3, max_rounds=2, fact_checks=5,
        planner_guidance="Identify the entities being compared and 4-6 comparison dimensions (capabilities, cost, performance, maturity, ecosystem).",
        report_sections=("Executive Summary", "Research Question", "Methodology",
                         "Comparison Matrix", "Key Findings", "Evidence",
                         "Conflicting Evidence", "Limitations", "Conclusions", "Sources"),
        writer_guidance="Include a Markdown comparison table over the entities and dimensions; state evidence quality per cell where sources disagree.",
    ),
    "fact_check": ResearchMode(
        key="fact_check", label="Fact Check",
        description="Verdict-focused: does the evidence support the claim?",
        researchers=("web", "academic"),
        max_queries_per_round=3, max_rounds=1, fact_checks=10,
        planner_guidance="Generate queries that would CONFIRM and REFUTE the claim, plus queries about its original source and context.",
        report_sections=("Verdict", "The Claim", "Evidence For", "Evidence Against",
                         "Conflicting Evidence", "Reasoning", "Confidence", "Sources"),
        writer_guidance="Lead with an explicit verdict: Supported / Partially supported / Contradicted / Unverifiable, with confidence and the strongest sources each way.",
    ),
    "gap_analysis": ResearchMode(
        key="gap_analysis", label="Research Gap Analysis",
        description="Maps what is known, what is weakly supported, and what is missing.",
        researchers=("web", "academic", "technical"),
        max_queries_per_round=3, max_rounds=2, fact_checks=4,
        planner_guidance="Cover the question broadly enough that absences become visible; include queries about surveys, reviews and 'open problems'.",
        report_sections=("Executive Summary", "Research Question", "Methodology",
                         "State of the Evidence", "Weakly Supported Claims",
                         "Conflicting Evidence", "Research Gaps",
                         "Recommended Next Queries", "Sources"),
        writer_guidance="The value is in what is MISSING: characterize each gap's importance and what a future study would need to answer it.",
    ),
}


def get_mode(key: str) -> ResearchMode:
    return MODES.get(key, MODES["deep"])
