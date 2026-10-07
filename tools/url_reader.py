"""
URL reading tool: fetch a page and return cleaned plain text.

Handles failures gracefully (returns "" and logs) so one inaccessible page
never terminates the research process. PDFs served directly from a URL are
parsed with pypdf when available.
"""

import logging

import requests

from config import MAX_PAGE_CHARS

log = logging.getLogger("agent.tools.reader")


def read_url(url: str, max_chars: int = MAX_PAGE_CHARS) -> str:
    """Fetch a URL and return cleaned, truncated plain text content."""
    if not url or not url.startswith("http"):
        return ""
    try:
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as e:
        log.warning("could not fetch %s: %s", url, e)
        return ""

    ctype = resp.headers.get("Content-Type", "").lower()
    if "pdf" in ctype or url.lower().endswith(".pdf"):
        return _read_pdf_bytes(resp.content, url, max_chars)
    if "html" not in ctype and "text" not in ctype and ctype:
        log.info("skipping non-text content at %s (%s)", url, ctype)
        return ""
    return _html_to_text(resp.text, max_chars)


def _html_to_text(html: str, max_chars: int) -> str:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "aside"]):
        tag.decompose()
    text = " ".join(soup.get_text(separator=" ").split())
    return text[:max_chars]


def _read_pdf_bytes(data: bytes, url: str, max_chars: int) -> str:
    try:
        from tools.document_loader import _pdf_text
        return _pdf_text(data)[:max_chars]
    except Exception as e:
        log.warning("PDF parse failed for %s: %s", url, e)
        return ""
