"""Optional persistent research memory (Chroma-backed, degrades gracefully)."""

from memory.vector_store import ResearchMemory, get_memory

__all__ = ["ResearchMemory", "get_memory"]
