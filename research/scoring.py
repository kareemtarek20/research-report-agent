"""
Source quality scoring (Phase 3).

Deterministic, explainable scores on 0-1 for authority, relevance, recency
and evidence quality, combined into one normalized quality score and tier.
Sources are NOT ranked by search-engine position - a domain tier table plus
content heuristics decide authority.
"""

import re
from urllib.parse import urlparse

# Domain authority tiers (0-1). Substring match on registered-ish domain.
AUTHORITY_TIERS = {
    # government / intergovernmental
    ".gov": 0.95, ".gov.": 0.95, ".int": 0.9, "who.int": 0.9, "europa.eu": 0.85,
    # academia / research
    ".edu": 0.9, "arxiv.org": 0.88, "nature.com": 0.88, "sciencedirect.com": 0.8,
    "ieee.org": 0.85, "acm.org": 0.85, "springer": 0.8, "pnas.org": 0.85,
    "nih.gov": 0.92, "pubmed": 0.9, "doi.org": 0.8, "openreview.net": 0.75,
    # standards bodies
    "ietf.org": 0.9, "rfc-editor.org": 0.9, "w3.org": 0.88, "iso.org": 0.85,
    "ecma-international.org": 0.8, "unicode.org": 0.8,
    # official vendor / project documentation
    "developer.mozilla.org": 0.9, "docs.python.org": 0.9, "python.org": 0.85,
    "platform.openai.com": 0.9, "openai.com": 0.8, "ai.google": 0.82,
    "deepmind.com": 0.8, "anthropic.com": 0.8, "mistral.ai": 0.72,
    "pytorch.org": 0.85, "tensorflow.org": 0.85, "huggingface.co": 0.7,
    "nvidia.com": 0.78, "developer.nvidia.com": 0.85,
    "kubernetes.io": 0.85, "docs.aws.amazon.com": 0.85, "learn.microsoft.com": 0.85,
    "cloud.google.com": 0.82, "developers.google.com": 0.82,
    "github.com": 0.65, "gitlab.com": 0.6,
    # reputable tech media / encyclopedic
    "wikipedia.org": 0.6,
    "arstechnica.com": 0.65, "infoq.com": 0.6, "thenextweb.com": 0.4,
    "techtarget.com": 0.45, "geeksforgeeks.org": 0.35,
}
# Content farms / SEO-heavy indicators (applied as caps, not floors)
LOW_QUALITY_SIGNALS = {
    "blogspot.com": 0.2, "medium.com": 0.3, "substack.com": 0.35,
    "wixsite.com": 0.15, "wordpress.com": 0.2, "pinterest.com": 0.1,
    "quora.com": 0.25, "reddit.com": 0.3, "fandom.com": 0.35,
    "craiyon": 0.1,
}
DEFAULT_AUTHORITY = 0.4  # unknown domain: neither trusted nor dismissed

PRIMARY_PATTERNS = (
    "arxiv.org", "doi.org", "pubmed", "ieee.org", "acm.org", "nature.com",
    "openreview.net", "nih.gov", "rfc-editor.org", "ietf.org", "w3.org",
    "/papers/", "/proceedings/", "/abstract", "research.google", "openai.com/research",
)
DOC_PATTERNS = ("docs.", "/docs/", "/documentation", "/reference/", "developer.",
                "platform.openai.com", "learn.microsoft.com", "developer.mozilla.org")

_YEAR_RE = re.compile(r"(19|20)\d{2}")


def domain_of(url: str) -> str:
    try:
        netloc = urlparse(url).netloc.lower()
        return netloc.removeprefix("www.")
    except Exception:
        return ""


def authority_score(url: str, source_type: str = "web") -> float:
    """Domain-tier authority. Type boosts for academic/user docs."""
    dom = domain_of(url) or url.lower()
    if source_type == "user_document":
        return 0.75  # authoritative for its own content, unverifiable externally
    if source_type == "previous_research":
        return 0.5

    best = None
    for tier_dom, score in AUTHORITY_TIERS.items():
        if tier_dom in dom and (best is None or len(tier_dom) > best[0]):
            best = (len(tier_dom), score)
    score = best[1] if best else DEFAULT_AUTHORITY
    for cap_dom, cap in LOW_QUALITY_SIGNALS.items():
        if cap_dom in dom:
            score = min(score, cap)
    if dom.endswith(".gov") or ".gov." in dom:
        score = max(score, 0.92)
    if dom.endswith(".edu"):
        score = max(score, 0.88)
    return round(min(score, 1.0), 3)


def is_primary_source(url: str, source_type: str = "web") -> bool:
    if source_type in ("academic", "user_document", "previous_research"):
        return source_type in ("academic", "user_document")
    low = (url or "").lower()
    return any(p in low for p in PRIMARY_PATTERNS)


def is_documentation(url: str) -> bool:
    low = (url or "").lower()
    return any(p in low for p in DOC_PATTERNS)


def recency_score(publication_date: str = "", url: str = "",
                  text: str = "", current_year: int = None) -> float:
    """Score from an explicit date, else a year found in URL/text. Unknown
    dates land mid-scale, not at an extreme."""
    if current_year is None:
        from datetime import datetime
        current_year = datetime.now().year
    year = None
    m = re.search(r"(19|20)\d{2}", publication_date or "")
    if m:
        year = int(m.group(0))
    if year is None:
        m = _YEAR_RE.search(url or "")
        if m:
            candidate = int(m.group(0))
            if current_year - 40 <= candidate <= current_year + 1:
                year = candidate
    if year is None:
        # earliest-looking recent year in first part of text (publication lines)
        for m in _YEAR_RE.finditer((text or "")[:600]):
            candidate = int(m.group(0))
            if current_year - 30 <= candidate <= current_year + 1:
                year = candidate
                break
    if year is None:
        return 0.5
    age = max(0, current_year - year)
    if age <= 1:
        return 1.0
    if age <= 3:
        return 0.8
    if age <= 5:
        return 0.6
    if age <= 8:
        return 0.4
    return 0.2


def relevance_score(query_terms: str, title: str, snippet: str) -> float:
    """Deterministic lexical relevance: term overlap against title (2x) and
    snippet (1x)."""
    terms = {t for t in re.findall(r"[a-z0-9]+", (query_terms or "").lower())
             if len(t) > 2}
    if not terms:
        return 0.5
    title_l = (title or "").lower()
    snippet_l = (snippet or "").lower()
    hits = sum(2.0 if t in title_l else (1.0 if t in snippet_l else 0.0) for t in terms)
    max_hits = 3.0 * len(terms)
    return round(min(1.0, 0.25 + 0.75 * (hits / max_hits)), 3)


def combined_quality(authority: float, relevance: float, recency: float,
                     evidence_quality: float = 0.5) -> float:
    return round(0.35 * authority + 0.30 * relevance + 0.15 * recency
                 + 0.20 * evidence_quality, 3)


def quality_tier(score: float) -> str:
    if score >= 0.75:
        return "high"
    if score >= 0.55:
        return "medium"
    if score >= 0.4:
        return "low"
    return "weak"


def score_source(source) -> None:
    """Fill scoring fields on a ResearchSource (pydantic model, in place)."""
    source.authority_score = authority_score(source.url, source.source_type)
    source.is_primary = is_primary_source(source.url, source.source_type)
    if source.is_primary:
        source.authority_score = min(1.0, source.authority_score + 0.05)
    source.recency_score = recency_score(source.publication_date, source.url,
                                         source.snippet)
    if source.relevance_score == 0.5 and source.snippet:
        source.relevance_score = relevance_score(source.title, source.title,
                                                 source.snippet)
    source.quality_score = combined_quality(source.authority_score,
                                            source.relevance_score,
                                            source.recency_score,
                                            source.evidence_quality)
    source.quality_tier = quality_tier(source.quality_score)
