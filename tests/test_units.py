"""Unit tests: JSON parsing, source scoring, source registry, evidence
extraction (incl. anti-fabrication), contradiction detection, documents,
memory fallback. No API keys or network needed."""

import time

from llm_client import extract_json
from research.scoring import (authority_score, recency_score, relevance_score,
                              combined_quality, quality_tier, is_primary_source)
from research.sources import findings_to_sources, normalize_url
from research.evidence import extract_evidence, dedupe_evidence
from research.contradictions import compute_agreement
from tools.document_loader import load_document, chunk_text
from memory.vector_store import ResearchMemory
from graph.state import ResearchSource


# ---------------------------------------------------------- JSON parsing --
def test_extract_json_plain():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_fenced():
    assert extract_json('```json\n{"a": [1, 2]}\n```') == {"a": [1, 2]}


def test_extract_json_prose_wrapped():
    assert extract_json('Sure! Here you go: {"query": "x"} — hope that helps') \
        == {"query": "x"}


def test_extract_json_list_in_prose():
    assert extract_json('results: [{"claim": "a"}] trailing') == [{"claim": "a"}]


def test_extract_json_garbage():
    assert extract_json("no json at all") is None
    assert extract_json("") is None


# --------------------------------------------------------- source scoring --
def test_authority_tiers():
    assert authority_score("https://www.nist.gov/guide") >= 0.9
    assert authority_score("https://arxiv.org/abs/1") >= 0.85
    assert authority_score("https://randomblog.blogspot.com/post") <= 0.25
    assert 0.3 <= authority_score("https://unknown-domain.example/x") <= 0.5


def test_recency():
    year = time.localtime().tm_year
    assert recency_score(str(year)) > recency_score(str(year - 9))
    assert recency_score("", "") == 0.5  # unknown lands mid-scale


def test_relevance_monotonic():
    q = "quantum computing cryptography"
    good = relevance_score(q, "Quantum computing and cryptography",
                           "Shor breaks RSA")
    bad = relevance_score(q, "Cooking pasta perfectly", "Boil water")
    assert good > bad


def test_quality_normalized_and_tiered():
    s = combined_quality(1.0, 1.0, 1.0, 1.0)
    assert s <= 1.0 and s == 1.0
    assert quality_tier(0.8) == "high"
    assert quality_tier(0.1) == "weak"


def test_primary_detection():
    assert is_primary_source("https://arxiv.org/abs/1234", "academic")
    assert not is_primary_source("https://medium.com/x", "web")


# ------------------------------------------------------ source registry --
def test_dedupe_by_url():
    f1 = {"title": "A", "url": "https://a.com/page", "snippet": "s1",
          "source_type": "web"}
    f2 = {"title": "A again", "url": "https://www.a.com/page/", "snippet": "s2",
          "source_type": "web"}
    sources = findings_to_sources([f1, f2], "test question")
    assert len(sources) == 1


def test_sequential_ids_and_scoring():
    findings = [{"title": f"t{i}", "url": f"https://x{i}.com",
                 "snippet": "s"} for i in range(3)]
    sources = findings_to_sources(findings, "question")
    assert [s["source_id"] for s in sources] == ["S1", "S2", "S3"]
    assert all(0.0 <= s["quality_score"] <= 1.0 for s in sources)


def test_incremental_round_numbering():
    r1 = findings_to_sources([{"title": "a", "url": "https://a.com"}], "q")
    r2 = findings_to_sources([{"title": "b", "url": "https://b.com"}], "q",
                             existing=r1)
    assert [s["source_id"] for s in r2] == ["S1", "S2"]


def test_normalize_url():
    assert normalize_url("https://WWW.Example.com/path/") == "example.com/path"


# ------------------------------------------------- evidence extraction --
SOURCE = {"source_id": "S1", "title": "Guide", "url": "https://a.com/g",
          "source_type": "web", "quality_score": 0.6,
          "authority_score": 0.6, "recency_score": 0.6, "relevance_score": 0.6}
CONTENT = "Local 7B models need 8GB VRAM with 4-bit quantization. Plenty more words here for length."


def test_evidence_with_verbatim_passage(monkeypatch):
    import research.evidence as r_evidence
    monkeypatch.setattr(r_evidence, "chat_json_list", lambda msgs, **kw: [{
        "claim": "7B models need 8GB VRAM when quantized",
        "supporting_passage": "8GB VRAM with 4-bit quantization",
        "task": "vr", "confidence": 0.9, "supported": True}])

    items = extract_evidence(SOURCE, CONTENT, "q", [], 5)
    assert len(items) == 1
    assert items[0]["supported"] is True
    assert items[0]["confidence"] >= 0.7


def test_fabricated_passage_demoted(monkeypatch):
    import research.evidence as r_evidence
    monkeypatch.setattr(r_evidence, "chat_json_list", lambda msgs, **kw: [{
        "claim": "Models need 12GB VRAM",
        "supporting_passage": "12GB VRAM is required per the study",
        "confidence": 0.95, "supported": True}])
    items = extract_evidence(SOURCE, CONTENT, "q", [], 5)
    assert items[0]["supported"] is False
    assert items[0]["confidence"] <= 0.25  # never trust the unverifiable quote


def test_malformed_items_skipped(monkeypatch):
    import research.evidence as r_evidence
    monkeypatch.setattr(r_evidence, "chat_json_list", lambda msgs, **kw: [
        "not a dict", {"no_claim": 1}, {"claim": "", "supporting_passage": ""}])
    assert extract_evidence(SOURCE, CONTENT, "q", [], 5) == []


def test_dedupe_evidence_keeps_best_confidence():
    a = {"claim": "X needs 8GB", "confidence": 0.5, "source_id": "S1"}
    b = {"claim": "x needs  8GB", "confidence": 0.8, "source_id": "S2"}
    out = dedupe_evidence([a, b])
    assert len(out) == 1 and out[0]["confidence"] == 0.8


# ------------------------------------------------- contradiction effects --
def test_agreement_raises_quality():
    evidence = [
        {"claim": "7B needs 8GB", "supported": True, "source_url": "https://a.com"},
        {"claim": "7B needs 8GB", "supported": True, "source_url": "https://b.com"},
    ]
    sources = [{"source_id": "S1", "url": "https://a.com", "quality_score": 0.5,
                "agreement_score": 0.0, "source_type": "web"},
               {"source_id": "S2", "url": "https://b.com", "quality_score": 0.5,
                "agreement_score": 0.0, "source_type": "web"}]
    compute_agreement(evidence, sources)
    assert sources[0]["quality_score"] > 0.5
    assert sources[0]["agreement_score"] > 0


def test_contradictions_need_two_claims():
    from research.contradictions import detect_contradictions
    assert detect_contradictions([]) == []
    assert detect_contradictions([{"claim": "only one"}]) == []


# ----------------------------------------------------------- documents --
def test_load_txt_and_md():
    findings = load_document("notes.txt", b"VRAM requirement is 8GB. " * 50)
    assert findings and findings[0]["source_type"] == "user_document"
    assert findings[0]["url"].startswith("upload://")


def test_load_unsupported_raises():
    try:
        load_document("x.docx", b"junk")
        assert False, "should have raised"
    except ValueError:
        pass


def test_chunking_bounds():
    chunks = chunk_text("word " * 20000)
    assert all(len(c) <= 4000 for c in chunks)
    assert len(chunks) > 1


# ------------------------------------------------------- memory fallback --
def test_memory_disabled_is_noop():
    m = ResearchMemory(enabled=False)
    assert m.recall("q") == []
    assert m.store_run("q", "deep", [], []) == 0


def test_memory_survives_broken_chromadb(monkeypatch):
    m = ResearchMemory(enabled=True, persist_dir=".pytest_missing_db")
    # either real chromadb works, or it degraded — both acceptable, crash is not
    assert m.enabled or "chromadb unavailable" in m._unavailable_reason
