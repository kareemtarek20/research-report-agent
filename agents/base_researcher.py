"""
Shared research engine used by the web / academic / technical researcher
nodes. Each researcher: runs its assigned queries with the matching search
backend, deep-reads a bounded number of result pages, and emits raw findings
under its OWN state channel — which is what makes LangGraph's fan-out
parallelism safe (no two parallel nodes write the same channel).
"""

import logging
from concurrent.futures import ThreadPoolExecutor

from config import RESEARCH_WORKERS
from research.modes import get_mode
from tools.web_search import web_search
from tools.academic_search import academic_search
from tools.url_reader import read_url

log = logging.getLogger("agent.researcher")

PAGES_PER_QUERY = 3
FINDINGS_KEY = {"web": "web_findings", "academic": "academic_findings",
                "technical": "technical_findings"}


def _search(kind: str, query: str) -> list[dict]:
    if kind == "academic":
        return academic_search(query)
    if kind == "technical":
        return web_search(f"{query} documentation")
    return web_search(query)


def run_researcher(kind: str, state: dict) -> dict:
    """Execute this researcher's tasks; return {findings_key, log}."""
    mode = get_mode(state.get("mode", "deep"))
    tasks = [t for t in state.get("tasks", []) if t.get("researcher") == kind]
    tasks = tasks[: mode.max_queries_per_round]
    out_key = FINDINGS_KEY[kind]
    label = {"web": "🌐 Web researcher", "academic": "🎓 Academic researcher",
             "technical": "🔧 Technical researcher"}[kind]

    findings: list[dict] = []
    lines: list[str] = []
    if not tasks:
        return {out_key: [], "log": [f"{label}: no tasks this round"]}

    for t in tasks:
        query = t["query"]
        try:
            results = _search(kind, query)
        except Exception as e:  # defensive: search tools already degrade to []
            log.warning("%s search failed for %r: %s", kind, query, e)
            results = []
        lines.append(f"{label}: \"{query}\" → {len(results)} results")

        pages = results[:PAGES_PER_QUERY]
        contents = _read_pages(pages)
        for r, content in zip(pages, contents):
            if not content and r.get("snippet"):
                content = r["snippet"]  # abstracts count as content for papers
            if not content:
                lines.append(f"   ⚠️ could not read {r.get('url', '?')} (skipped)")
                continue
            findings.append({
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": r.get("snippet", ""),
                "content": content,
                "source_type": kind,
                "query": query,
                "subquestion": t.get("subquestion", ""),
                "author": r.get("author", ""),
                "published_date": r.get("published_date", ""),
            })

    lines.append(f"   → {len(findings)} pages collected")
    log.info("researcher=%s findings=%s", kind, len(findings))
    return {out_key: findings, "log": lines}


def _read_pages(pages: list[dict]) -> list[str]:
    """Fetch several result pages concurrently (IO-bound). Order preserved;
    a failing page yields '' and never breaks the batch."""
    if not pages:
        return []
    if len(pages) == 1 or RESEARCH_WORKERS <= 1:
        return [read_url(p.get("url", "")) for p in pages]

    def _safe(p):
        try:
            return read_url(p.get("url", ""))
        except Exception as e:
            log.warning("read_url failed for %s: %s", p.get("url", "?"), e)
            return ""

    with ThreadPoolExecutor(max_workers=min(RESEARCH_WORKERS, len(pages))) as ex:
        return list(ex.map(_safe, pages))
