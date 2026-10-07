"""
Source registry: turns raw researcher findings into deduplicated,
ID-assigned, scored ResearchSource objects (Phases 2-3).
"""

import logging

from graph.state import ResearchSource
from research.scoring import score_source, relevance_score

log = logging.getLogger("agent.research.sources")


def findings_to_sources(findings: list[dict], question: str,
                        start_index: int = 1,
                        existing: list[dict] | None = None) -> list[dict]:
    """Merge new findings into the existing source registry.

    Deduplicates by normalized URL (uploaded docs dedupe by url+chunk).
    New sources get sequential IDs (S1, S2...) and deterministic scores.
    Returns the full updated list as dicts.
    """
    registry = [ResearchSource(**s) for s in (existing or [])]
    by_url = {normalize_url(s.url): s for s in registry}
    next_index = start_index + len(registry)

    for f in findings:
        url = f.get("url", "") or ""
        key = normalize_url(url)
        if not key:
            continue
        if key in by_url:
            # richer metadata from a later sighting wins (e.g. a real date)
            src = by_url[key]
            src.snippet = src.snippet or f.get("snippet", "")
            if f.get("published_date") and not src.publication_date:
                src.publication_date = f["published_date"]
            if f.get("author") and not src.author:
                src.author = f["author"]
            continue
        source = ResearchSource(
            source_id=f"S{next_index}",
            title=f.get("title", "") or url,
            url=url,
            source_type=f.get("source_type", "web"),
            domain=_domain(url),
            snippet=(f.get("snippet") or "")[:700],
            author=f.get("author", ""),
            publication_date=f.get("published_date", ""),
            relevance_score=relevance_score(question, f.get("title", ""),
                                            f.get("snippet", "")),
        )
        score_source(source)
        registry.append(source)
        by_url[key] = source
        next_index += 1

    return [s.to_dict() for s in registry]


def normalize_url(url: str) -> str:
    u = (url or "").strip().lower()
    u = u.removeprefix("https://").removeprefix("http://").removeprefix("www.")
    return u.rstrip("/")


def _domain(url: str) -> str:
    from research.scoring import domain_of
    return domain_of(url or "")


def sources_by_id(sources: list[dict]) -> dict[str, dict]:
    return {s["source_id"]: s for s in sources}


def format_source_registry(sources: list[dict]) -> str:
    """Human/LLM-readable registry used by the writer and critic."""
    lines = []
    for s in sources:
        tier = s.get("quality_tier", "unknown")
        primary = " [primary]" if s.get("is_primary") else ""
        date = s.get("publication_date") or "n.d."
        author = s.get("author") or "unknown"
        lines.append(
            f"[{s['source_id']}] {s['title']} — {s['url']} "
            f"({s['source_type']}, {tier} quality{primary}, {date}, by {author})"
        )
    return "\n".join(lines)
