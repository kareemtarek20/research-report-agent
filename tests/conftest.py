"""Shared fixtures: canned fake LLM/search/reader so tests need no API keys."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pytest

SOURCE_CONTENT = ("Local 7B models need 8GB VRAM with 4-bit quantization. "
                  "Full fp16 inference requires about 14GB VRAM. "
                  "CPU offload is slow but works with 6GB.")


def canned_evidence(passage_ok=True):
    passage = "8GB VRAM with 4-bit quantization" if passage_ok else "12GB VRAM required"
    return [{"claim": "7B models need 8GB VRAM when quantized to 4-bit",
             "supporting_passage": passage, "task": "VRAM requirements",
             "confidence": 0.9, "supported": True}]


@pytest.fixture
def fake_llm_payloads():
    return {
        "planner": {"tasks": [
            {"subquestion": "VRAM needed for 7B models?", "query": "7B LLM VRAM requirements",
             "researcher": "web", "rationale": ""},
            {"subquestion": "Academic measurements?", "query": "LLM inference memory paper",
             "researcher": "academic", "rationale": ""},
            {"subquestion": "Official docs?", "query": "llama.cpp requirements",
             "researcher": "technical", "rationale": ""}]},
        "fact_check": {"status": "supported", "confidence": 0.8,
                        "supporting_sources": ["https://b.edu/paper"],
                        "contradicting_sources": [], "explanation": "Matches papers."},
        "critic": {"sufficient": True, "reason": "solid coverage", "gaps": [],
                    "unanswered_subquestions": [], "weak_claims": []},
        "question_coverage": {"answered": [
            {"subquestion": s, "addressed": True}
            for s in ("VRAM needed for 7B models?", "Academic measurements?",
                      "Official docs?")]},
    }


@pytest.fixture
def fake_search_results():
    return [{"title": "Local LLM guide", "url": "https://a.com/guide",
             "snippet": "7B needs 8GB VRAM quantized"},
            {"title": "Bad link page", "url": "https://dead.example/x",
             "snippet": "irrelevant marketing copy"}]


@pytest.fixture
def patched_pipeline(monkeypatch, fake_llm_payloads, fake_search_results):
    """Patch EVERY LLM + search + reader entry point used by the graph."""
    import llm_client
    import agents.planner as planner
    import agents.base_researcher as base_r
    import agents.fact_checker as fact_checker
    import agents.critic as critic
    import agents.report_critic as report_critic
    import research.evidence as r_evidence
    import research.contradictions as r_contra

    calls = {"plan": 0, "write": 0}

    def planner_chat_json(messages, **kw):
        calls["plan"] += 1
        return fake_llm_payloads["planner"]

    def fact_chat_json(messages, **kw):
        return fake_llm_payloads["fact_check"]

    def critic_chat_json(messages, **kw):
        return fake_llm_payloads["critic"]

    def coverage_chat_json(messages, **kw):
        return fake_llm_payloads["question_coverage"]

    def evidence_chat_json_list(messages, **kw):
        return canned_evidence()

    def contradiction_chat_json_list(messages, **kw):
        return []

    def writer_chat(messages, **kw):
        calls["write"] += 1
        return ("## Executive Summary\n\nQuantized 7B models run in 8GB "
                "VRAM [S1].\n\n## Key Findings\n\nfp16 requires about 14GB "
                "VRAM [S1]. Conflicting evidence: none.\n\n## Limitations\n\n"
                "Single-source coverage.\n\n## Sources\n\nMADE UP: https://evil.example\n")

    monkeypatch.setattr(planner, "chat_json", planner_chat_json)
    monkeypatch.setattr(fact_checker, "chat_json", fact_chat_json)
    monkeypatch.setattr(critic, "chat_json", critic_chat_json)
    monkeypatch.setattr(report_critic, "chat_json", coverage_chat_json)
    monkeypatch.setattr(r_evidence, "chat_json_list", evidence_chat_json_list)
    monkeypatch.setattr(r_contra, "chat_json_list", contradiction_chat_json_list)
    monkeypatch.setattr(llm_client, "chat", writer_chat)

    monkeypatch.setattr(base_r, "web_search", lambda q: list(fake_search_results))
    monkeypatch.setattr(base_r, "academic_search",
                        lambda q: [{"title": "Paper", "url": "https://b.edu/paper",
                                    "snippet": "We measure VRAM: 8GB quantized",
                                    "author": "Smith", "published_date": "2025"}])
    monkeypatch.setattr(base_r, "read_url",
                        lambda url: SOURCE_CONTENT if "a.com" in url or "b.edu" in url else "")
    monkeypatch.setattr(fact_checker, "web_search",
                        lambda q: [{"title": "Check source", "url": "https://c.gov/x",
                                    "snippet": "8GB is typical for 4-bit 7B"}])
    return calls
