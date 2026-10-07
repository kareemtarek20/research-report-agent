"""
Academic search tool.

Primary backend: Semantic Scholar Graph API (free, no key required; setting
SEMANTIC_SCHOLAR_API_KEY raises rate limits). Fallback: DuckDuckGo search
restricted to scholarly domains. Failures return [] rather than raising.
"""

import logging

import requests

from config import SEMANTIC_SCHOLAR_API_KEY, MAX_RESULTS_PER_SEARCH, TAVILY_API_KEY
from tools.web_search import _dedupe

log = logging.getLogger("agent.tools.academic")

S2_FIELDS = "title,url,abstract,year,authors,venue,publicationDate,isOpenAccess,externalIds"


def academic_search(query: str) -> list[dict]:
    """Return academic results as {title, url, snippet, author,
    published_date} dicts. Never raises."""
    try:
        results = _semantic_scholar(query)
        if results:
            return _dedupe(results)
        log.info("Semantic Scholar returned nothing for %r, using fallback", query)
        return _dedupe(_scholarly_web(query))
    except Exception as e:
        log.warning("academic search failed for %r: %s", query, e)
        return []


def _semantic_scholar(query: str) -> list[dict]:
    params = {"query": query, "limit": MAX_RESULTS_PER_SEARCH, "fields": S2_FIELDS}
    headers = {}
    if SEMANTIC_SCHOLAR_API_KEY:
        headers["x-api-key"] = SEMANTIC_SCHOLAR_API_KEY
    resp = requests.get(
        "https://api.semanticscholar.org/graph/v1/paper/search",
        params=params,
        headers=headers,
        timeout=20,
    )
    if resp.status_code != 200:
        log.info("Semantic Scholar status %s for %r", resp.status_code, query)
        return []
    papers = resp.json().get("data", [])
    out = []
    for p in papers:
        authors = ", ".join(a.get("name", "") for a in (p.get("authors") or [])[:3])
        url = p.get("url") or ""
        doi = (p.get("externalIds") or {}).get("DOI")
        if doi:
            url = url or f"https://doi.org/{doi}"
        out.append({
            "title": p.get("title", ""),
            "url": url,
            "snippet": (p.get("abstract") or p.get("venue") or "")[:800],
            "author": authors,
            "published_date": p.get("publicationDate") or str(p.get("year") or ""),
        })
    return out


def _scholarly_web(query: str) -> list[dict]:
    """Fallback: general search with scholarly-domain restriction."""
    q = f"{query} (site:arxiv.org OR site:nature.com OR site:pubmed.ncbi.nlm.nih.gov OR filetype:pdf)"
    if TAVILY_API_KEY:
        from tools.web_search import _tavily_search
        return _tavily_search(q)
    from tools.web_search import _duckduckgo_search
    return _duckduckgo_search(q)
