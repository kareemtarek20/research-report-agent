"""
Optional persistent research memory (Phase 8).

Uses ChromaDB when installed and ENABLE_MEMORY is on. Everything degrades
gracefully: if chromadb is missing, the DB is locked, or calls fail, memory
becomes a no-op and the agent keeps working exactly as before.
"""

import logging
from pathlib import Path

from config import ENABLE_MEMORY, MEMORY_DIR

log = logging.getLogger("agent.memory")

_COLLECTION = "research_evidence"


class ResearchMemory:
    """Thin, failure-tolerant wrapper around a Chroma collection."""

    def __init__(self, enabled: bool = None, persist_dir: str = None):
        self.enabled = ENABLE_MEMORY if enabled is None else enabled
        self.persist_dir = persist_dir or MEMORY_DIR
        self._collection = None
        self._unavailable_reason = ""
        if not self.enabled:
            self._unavailable_reason = "memory disabled (ENABLE_MEMORY=false)"
            return
        try:
            import chromadb  # optional dependency

            Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=self.persist_dir)
            self._collection = client.get_or_create_collection(_COLLECTION)
            log.info("memory=enabled backend=chroma dir=%s", self.persist_dir)
        except Exception as e:
            self.enabled = False
            self._unavailable_reason = f"chromadb unavailable: {e}"
            log.warning("memory disabled - %s (pip install chromadb to enable)", e)

    # -- write path -----------------------------------------------------
    def store_run(self, question: str, mode: str, evidence: list[dict],
                  sources: list[dict], report_summary: str = "") -> int:
        """Persist evidence items + a run summary. Returns count stored."""
        if not self.enabled or self._collection is None:
            return 0
        import hashlib

        stored = 0
        try:
            for e in evidence:
                text = f"{e.get('claim', '')}\n{e.get('supporting_passage', '')}"
                if not text.strip():
                    continue
                doc_id = hashlib.sha256(
                    f"{question}|{e.get('source_url')}|{e.get('claim')}".encode()
                ).hexdigest()[:24]
                self._collection.upsert(
                    ids=[doc_id],
                    documents=[text[:1500]],
                    metadatas=[{
                        "question": question[:200],
                        "mode": mode,
                        "source_url": e.get("source_url", ""),
                        "source_title": e.get("source_title", "")[:200],
                        "source_type": e.get("source_type", "web"),
                        "kind": "evidence",
                    }],
                )
                stored += 1
            if report_summary.strip():
                import time as _t

                self._collection.upsert(
                    ids=[f"summary-{int(_t.time())}-{abs(hash(question)) % 10**8}"],
                    documents=[report_summary[:3000]],
                    metadatas=[{"question": question[:200], "mode": mode,
                                "kind": "summary", "source_type": "previous_research"}],
                )
                stored += 1
            log.info("memory=stored items=%s question=%r", stored, question[:80])
        except Exception as e:
            log.warning("memory store failed (continuing without): %s", e)
        return stored

    # -- read path ------------------------------------------------------
    def recall(self, question: str, top_k: int = 5) -> list[dict]:
        """Return previous-research findings shaped like researcher findings,
        tagged source_type='previous_research'. Empty list when unavailable."""
        if not self.enabled or self._collection is None:
            return []
        try:
            res = self._collection.query(query_texts=[question],
                                         n_results=min(top_k, max(1, self._collection.count())))
            out = []
            docs = (res.get("documents") or [[]])[0]
            metas = (res.get("metadatas") or [[]])[0]
            for doc, meta in zip(docs, metas):
                meta = meta or {}
                if meta.get("kind") == "evidence":
                    out.append({
                        "title": meta.get("source_title") or "Previous research",
                        "url": meta.get("source_url") or "memory://previous_research",
                        "snippet": doc[:300],
                        "content": doc,
                        "source_type": "previous_research",
                        "query": question,
                    })
            return out
        except Exception as e:
            log.warning("memory recall failed (continuing without): %s", e)
            return []


_singleton = None


def get_memory() -> ResearchMemory:
    global _singleton
    if _singleton is None:
        _singleton = ResearchMemory()
    return _singleton
