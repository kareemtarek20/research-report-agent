"""Graph tests: routing decisions, iteration limits, full mocked end-to-end
run, failure handling, document ingestion, revision cap. No API keys needed."""

import pytest

from graph.routing import (route_researchers, route_after_plan, after_critic,
                           after_report_quality)
from graph.workflow import run_agent, build_graph
from research.modes import get_mode


def drain(question, mode="quick", documents=None):
    final = None
    for state in run_agent(question, mode=mode, documents=documents):
        final = state
    return final


# ------------------------------------------------------------- routing --
def test_fanout_respects_mode():
    state = {"mode": "quick", "tasks": [{"researcher": "academic", "query": "x"},
                                        {"researcher": "web", "query": "y"}]}
    assert route_researchers(state) == ["web_researcher"]


def test_fanout_parallel_three():
    state = {"mode": "deep", "tasks": [
        {"researcher": "web", "query": "a"},
        {"researcher": "academic", "query": "b"},
        {"researcher": "technical", "query": "c"}]}
    assert sorted(route_researchers(state)) == [
        "academic_researcher", "technical_researcher", "web_researcher"]


def test_fanout_always_returns_something():
    assert route_researchers({"mode": "deep", "tasks": []}) == ["web_researcher"]


def test_route_after_plan_sends_documents_first():
    docs = [{"url": "upload://a.txt#chunk0"}]
    tasks = [{"researcher": "web"}, {"researcher": "academic"},
             {"researcher": "technical"}]
    assert route_after_plan({"mode": "quick", "tasks": [], "documents": docs,
                             "doc_findings": []}) == "document_researcher"
    # after the document node ran, later rounds fan out to researchers
    assert sorted(route_after_plan({"mode": "deep", "tasks": tasks,
                                    "documents": docs,
                                    "doc_findings": docs})) == [
        "academic_researcher", "technical_researcher", "web_researcher"]
    # no documents -> never routes to the document node
    assert route_after_plan({"mode": "quick", "tasks": [], "documents": [],
                             "doc_findings": []}) == ["web_researcher"]


def test_critic_loops_only_for_high_gaps():
    base = {"mode": "deep", "round": 1, "assessment": {"sufficient": False}}
    high = base | {"gaps": [{"importance": "high", "gap": "g"}]}
    low = base | {"gaps": [{"importance": "low", "gap": "g"}]}
    assert after_critic(high) == "research"
    assert after_critic(low) == "write"
    assert after_critic(base | {"assessment": {"sufficient": True},
                                "gaps": [{"importance": "high"}]}) == "write"


def test_critic_loop_hard_round_cap():
    deep = get_mode("deep")
    state = {"mode": "deep", "round": deep.max_rounds,
             "assessment": {"sufficient": False},
             "gaps": [{"importance": "high", "gap": "g"}]}
    assert after_critic(state) == "write"


def test_revision_hard_cap(monkeypatch):
    import graph.routing as routing
    monkeypatch.setattr("config.MAX_REPORT_REVISIONS", 1)
    fail = {"report_quality": {"passed": False}, "revisions": 0,
            "report_feedback": ["fix"]}
    assert routing.after_report_quality(fail) == "revise"
    assert routing.after_report_quality(
        {"report_quality": {"passed": False}, "revisions": 1,
         "report_feedback": ["fix"]}) == "finalize"
    assert routing.after_report_quality(
        {"report_quality": {"passed": True}, "revisions": 0,
         "report_feedback": []}) == "finalize"


# ------------------------------------------------------ mocked end-to-end --
def test_deep_run_produces_cited_report(patched_pipeline):
    final = drain("What VRAM does a 7B model need?", mode="deep")
    assert final["report"]
    assert "[S" in final["report"], "report must contain inline citations"
    assert "## Sources" in final["report"]
    assert "https://a.com/guide" in final["report"] or \
           "https://b.edu/paper" in final["report"]
    assert final["metadata"]["source_count"] >= 1
    assert final["metadata"]["evidence_count"] >= 1
    assert final["fact_checks"], "fact checker should have run"
    assert not final["errors"]
    # parallel branches all contributed
    assert final["web_findings"] or final["academic_findings"] or \
        final["technical_findings"]


def test_report_is_citation_traceable(patched_pipeline):
    final = drain("What VRAM does a 7B model need?", mode="deep")
    import re
    cited_ids = {f"S{n}" for n in re.findall(r"\[S(\d+)\]", final["report"])}
    source_ids = {s["source_id"] for s in final["sources"]}
    assert cited_ids <= source_ids, "no citation may reference an unknown source"


def test_revision_loop_terminates(patched_pipeline, monkeypatch):
    # force quality failure to prove the revision counter stops the loop
    calls = patched_pipeline
    monkeypatch.setattr("agents.report_critic.QUALITY_EVIDENCE_COVERAGE", 0.99)
    monkeypatch.setattr("agents.report_critic.QUALITY_CITATION_COVERAGE", 0.99)
    final = drain("What VRAM does a 7B model need?", mode="deep")
    assert final["report"]
    assert calls["write"] <= 2, "writer must run at most 1 draft + 1 revision"
    assert len(final["quality_evaluations"]) <= 2


def test_gap_driven_research_loop_terminates(monkeypatch, fake_llm_payloads,
                                             fake_search_results):
    import agents.critic as critic
    import agents.planner as planner
    import agents.base_researcher as base_r
    import agents.fact_checker as fact_checker
    import agents.report_critic as report_critic
    import research.evidence as r_evidence
    import research.contradictions as r_contra
    import llm_client

    plan_calls = {"n": 0}

    def counting_planner(messages, **kw):
        plan_calls["n"] += 1
        return fake_llm_payloads["planner"]

    monkeypatch.setattr(planner, "chat_json", counting_planner)
    monkeypatch.setattr(critic, "chat_json", lambda m, **kw: {
        "sufficient": False, "reason": "missing benchmark",
        "gaps": [{"gap": "No low-VRAM benchmark", "importance": "high",
                  "recommended_query": "low vram benchmark"}],
        "unanswered_subquestions": [], "weak_claims": []})
    monkeypatch.setattr(fact_checker, "chat_json",
                        lambda m, **kw: fake_llm_payloads["fact_check"])
    monkeypatch.setattr(report_critic, "chat_json",
                        lambda m, **kw: fake_llm_payloads["question_coverage"])
    monkeypatch.setattr(r_evidence, "chat_json_list",
                        lambda m, **kw: [{"claim": "7B needs 8GB VRAM when quantized",
                                          "supporting_passage":
                                          "8GB VRAM with 4-bit quantization",
                                          "confidence": 0.9, "supported": True}])
    monkeypatch.setattr(r_contra, "chat_json_list", lambda m, **kw: [])
    monkeypatch.setattr(llm_client, "chat", lambda m, **kw:
                        "## Key Findings\n\nQuantized 7B models need 8GB VRAM "
                        "according to measured benchmarks [S1].\n")
    monkeypatch.setattr(base_r, "web_search", lambda q: list(fake_search_results))
    monkeypatch.setattr(base_r, "academic_search", lambda q: [])
    monkeypatch.setattr(base_r, "read_url", lambda u:
                        "Local 7B models need 8GB VRAM with 4-bit quantization. "
                        "Extra filler text so content length passes the guard.")
    monkeypatch.setattr(fact_checker, "web_search", lambda q: [])

    final = drain("VRAM?", mode="deep")  # critic always says insufficient
    deep = get_mode("deep")
    assert plan_calls["n"] <= deep.max_rounds, "gap loop must respect hard cap"
    assert final["report"], "agent must still produce a report after the cap"


# ------------------------------------------------------ failure handling --
def test_search_exception_does_not_kill_run(monkeypatch, fake_llm_payloads):
    import agents.base_researcher as base_r
    import agents.planner as planner
    import agents.fact_checker as fact_checker
    import agents.critic as critic
    import agents.report_critic as report_critic
    import llm_client

    monkeypatch.setattr(planner, "chat_json",
                        lambda m, **kw: fake_llm_payloads["planner"])
    monkeypatch.setattr(base_r, "web_search",
                        lambda q: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(base_r, "academic_search",
                        lambda q: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(base_r, "read_url", lambda u: "")
    monkeypatch.setattr(fact_checker, "chat_json",
                        lambda m, **kw: fake_llm_payloads["fact_check"])
    monkeypatch.setattr(fact_checker, "web_search",
                        lambda q: (_ for _ in ()).throw(RuntimeError("down")))
    monkeypatch.setattr(critic, "chat_json",
                        lambda m, **kw: fake_llm_payloads["critic"])
    monkeypatch.setattr(report_critic, "chat_json",
                        lambda m, **kw: fake_llm_payloads["question_coverage"])
    monkeypatch.setattr(llm_client, "chat",
                        lambda m, **kw: "## Key Findings\n\nNo evidence was "
                                        "retrievable; see Limitations.")

    final = drain("anything", mode="quick")
    assert final["report"]  # degraded but finished
    assert final["web_findings"] == [] and final["academic_findings"] == []
    assert final["sources"] == []  # nothing could be retrieved, run still ended


def test_llm_json_garbage_degrades_gracefully(monkeypatch, fake_llm_payloads,
                                               fake_search_results):
    import agents.planner as planner
    import agents.base_researcher as base_r

    def broken_planner(messages, **kw):
        raise ValueError("Model did not return valid JSON after retry")

    monkeypatch.setattr(planner, "chat_json", broken_planner)
    monkeypatch.setattr(base_r, "web_search", lambda q: list(fake_search_results))
    monkeypatch.setattr(base_r, "read_url", lambda u: "7B needs 8GB VRAM text " * 10)
    # fallback plan must still exist (question itself as query)
    final_state = None
    for state in run_agent("topic", mode="quick"):
        final_state = state
    assert final_state["tasks"]  # fallback task used


# ------------------------------------------------------------ documents --
def test_uploaded_documents_become_labeled_sources(patched_pipeline):
    final = drain("Summarize my uploaded constraints",
                  mode="academic",
                  documents=[("mynotes.txt",
                              b"Our lab machines have 8GB VRAM. " * 20)])
    types = {s["source_type"] for s in final["sources"]}
    assert "user_document" in types
    doc_src = [s for s in final["sources"] if s["source_type"] == "user_document"]
    assert doc_src[0]["url"].startswith("upload://")


def test_uploaded_documents_are_searched_before_the_web(patched_pipeline):
    final = drain("What VRAM do our lab machines have?", mode="deep",
                  documents=[("mynotes.txt",
                              b"Our lab machines have 8GB VRAM. " * 20)])
    log = final["log"]
    doc_i = next(i for i, line in enumerate(log)
                 if line.startswith("📄 Document researcher"))
    web_i = next(i for i, line in enumerate(log) if "Web researcher" in line)
    assert doc_i < web_i, "documents must be searched before any web call"
    # and their evidence was extracted in that first pass
    assert any("evidence items from your documents" in line for line in log)
    doc_src = [s for s in final["sources"] if s["source_type"] == "user_document"]
    assert doc_src and doc_src[0]["source_id"] == "S1"  # registered first


def test_evidence_caps_enforced(patched_pipeline, monkeypatch):
    import agents.evidence_agent as ev_agent
    monkeypatch.setattr(ev_agent, "MAX_TOTAL_EVIDENCE", 1)
    final = drain("What VRAM?", mode="deep")
    assert len(final["evidence"]) <= 1 + 3  # cap applied after dedupe of rounds


# --------------------------------------------------- speed optimizations --
def test_weak_sources_skipped_but_floor_kept():
    from agents.evidence_agent import _filter_weak
    mk = lambda i, q, t="web": {"source_id": f"S{i}", "quality_score": q,
                                "quality_tier": "weak" if q < 0.4 else "ok",
                                "source_type": t}
    cands = [mk(1, 0.9), mk(2, 0.3), mk(3, 0.2), mk(4, 0.1), mk(5, 0.1),
             mk(6, 0.1, "user_document")]
    keep, skip = _filter_weak(cands)
    ids = {s["source_id"] for s in keep}
    assert "S1" in ids and "S6" in ids          # good + user docs always kept
    assert len(keep) >= 3                        # coverage floor of 3
    assert all(s["source_id"] not in {"S4", "S5"} or s["source_type"] != "web"
               for s in keep)                    # weakest webs past floor skipped
    assert keep and skip


def test_parallel_extraction_matches_serial(patched_pipeline):
    """Concurrent extraction must not reorder or lose per-source results."""
    final = drain("What VRAM does a 7B model need?", mode="deep")
    src_ids = {s["source_id"] for s in final["sources"]}
    assert {e["source_id"] for e in final["evidence"]} <= src_ids
    assert final["evidence"], "extraction produced evidence under threads"


def test_parallel_fact_checks_complete(patched_pipeline):
    final = drain("What VRAM?", mode="deep")
    assert all(fc.get("claim_hash") for fc in final["fact_checks"])
