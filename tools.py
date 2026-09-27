"""
Tools the agent can call:
  1. web_search(query) -> list of {title, url, snippet}
  2. read_url(url)      -> extracted plain text of a page

Search uses Tavily if TAVILY_API_KEY is set (recommended - built for agents,
returns clean snippets). Falls back to DuckDuckGo's free HTML endpoint
(no API key needed, but noisier) if no Tavily key is configured.
"""

import requests
from bs4 import BeautifulSoup
from config import TAVILY_API_KEY, MAX_RESULTS_PER_SEARCH


def web_search(query: str) -> list[dict]:
    """Search the web and return a list of {title, url, snippet} dicts."""
    if TAVILY_API_KEY:
        return _tavily_search(query)
    return _duckduckgo_search(query)


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
            results.append(
                {
                    "title": link.get_text(strip=True),
                    "url": link.get("href", ""),
                    "snippet": snippet.get_text(strip=True) if snippet else "",
                }
            )
    return results


def read_url(url: str, max_chars: int = 4000) -> str:
    """Fetch a URL and return cleaned, truncated plain text content."""
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=15,
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        return f"[Could not fetch {url}: {e}]"

    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    text = " ".join(soup.get_text(separator=" ").split())
    return text[:max_chars]
