"""
Web search tool.

Uses Tavily if TAVILY_API_KEY is set (recommended - built for agents,
returns clean snippets). Falls back to DuckDuckGo's free HTML endpoint
(no API key needed, but noisier) if no Tavily key is configured.

Failures degrade gracefully: any search error returns an empty list (logged)
instead of raising, so one failed search never terminates a research run.
"""

import logging

import requests

from config import TAVILY_API_KEY, MAX_RESULTS_PER_SEARCH

log = logging.getLogger("agent.tools.search")


def web_search(query: str) -> list[dict]:
    """Search the web and return a list of {title, url, snippet} dicts."""
    try:
        if TAVILY_API_KEY:
            results = _tavily_search(query)
        else:
            results = _duckduckgo_search(query)
    except Exception as e:  # network/API failure -> keep researching with other sources
        log.warning("search failed for %r: %s", query, e)
        return []
    return _dedupe(results)


def _tavily_search(query: str) -> list[dict]:
    resp = requests.post(
        "https://api.tavily.com/search",
        json={
            "api_key": TAVILY_API_KEY,
            "query": query,
            "max_results": MAX_RESULTS_PER_SEARCH,
            "search_depth": "basic",
        },
        timeout=20,
    )
    resp.raise_for_status()
    data = resp.json()
    return [
        {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "snippet": r.get("content", ""),
        }
        for r in data.get("results", [])
    ]


def _duckduckgo_search(query: str) -> list[dict]:
    """Free fallback with no API key. Scrapes DuckDuckGo's HTML results page."""
    from bs4 import BeautifulSoup

    resp = requests.post(
        "https://html.duckduckgo.com/html/",
        data={"q": query},
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=20,
    )
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    results = []
    for result in soup.select(".result")[:MAX_RESULTS_PER_SEARCH]:
        link = result.select_one(".result__a")
        snippet = result.select_one(".result__snippet")
        if link:
            url = _clean_ddg_url(link.get("href", ""))
            if url.startswith("http"):
                results.append(
                    {
                        "title": link.get_text(strip=True),
                        "url": url,
                        "snippet": snippet.get_text(strip=True) if snippet else "",
                    }
                )
    return results


def _clean_ddg_url(url: str) -> str:
    """DuckDuckGo wraps result URLs in a redirect: //duckduckgo.com/l/?uddg=<real>."""
    import urllib.parse

    if url.startswith("//"):
        url = "https:" + url
    parsed = urllib.parse.urlparse(url)
    if parsed.netloc.endswith("duckduckgo.com") and parsed.path.startswith("/l/"):
        qs = urllib.parse.parse_qs(parsed.query)
        if "uddg" in qs:
            return qs["uddg"][0]
    return url


def _dedupe(results: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for r in results:
        url = (r.get("url") or "").rstrip("/").lower()
        if not url or url in seen:
            continue
        seen.add(url)
        out.append(r)
    return out
