"""
Document loader for user-uploaded research files (Phase 9).

Supports PDF, plain text and Markdown. Uploaded documents are chunked and
turned into the same "finding" shape the web researchers produce, but tagged
with source_type="user_document" so evidence stays clearly distinguishable.
"""

import io
import logging

log = logging.getLogger("agent.tools.documents")

CHUNK_SIZE = 4000


def load_document(filename: str, data: bytes) -> list[dict]:
    """Parse an uploaded file into findings: [{title,url,snippet,content,
    source_type,query,chunk_index}]. Raises ValueError for unsupported types."""
    name = (filename or "").lower()
    text = ""
    if name.endswith(".pdf"):
        text = _pdf_text(data)
    elif name.endswith((".txt", ".md", ".markdown", ".rst")):
        text = data.decode("utf-8", errors="replace")
    else:
        raise ValueError(f"Unsupported document type: {filename}")

    if not text.strip():
        log.info("document %s produced no extractable text", filename)
        return []

    findings = []
    for i, chunk in enumerate(chunk_text(text)):
        findings.append({
            "title": filename,
            "url": f"upload://{filename}#chunk{i}",
            "snippet": chunk[:300],
            "content": chunk,
            "source_type": "user_document",
            "query": "user document",
            "author": "",
            "published_date": "",
            "chunk_index": i,
        })
    return findings


def chunk_text(text: str, size: int = CHUNK_SIZE) -> list[str]:
    """Split text into roughly-paragraph-sized chunks of at most `size` chars."""
    text = " ".join(text.split())
    if not text:
        return []
    chunks = []
    paragraphs = text.split(". ")
    current = ""
    for para in paragraphs:
        candidate = (current + ". " + para).strip() if current else para
        if len(candidate) > size:
            if current:
                chunks.append(current)
            # hard-split oversized single paragraphs
            while len(para) > size:
                chunks.append(para[:size])
                para = para[size:]
            current = para
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def _pdf_text(data: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    pages = [(page.extract_text() or "") for page in reader.pages]
    return "\n".join(pages)
