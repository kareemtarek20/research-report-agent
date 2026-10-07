"""
Streamlit UI for the Research Intelligence Agent.

Run with:  streamlit run app.py

Shows only concise action summaries and observable results (no internal
chain-of-thought): current agent stage, queries executed, structured
evidence, fact checks, contradictions, gaps and quality metrics.
"""

import json

import streamlit as st

from config import (LLM_MODEL, LLM_BASE_URL, TAVILY_API_KEY, MAX_RESEARCH_ROUNDS,
                    MAX_SOURCES, MAX_TOTAL_EVIDENCE, MAX_FACT_CHECKS,
                    MAX_REPORT_REVISIONS)
from graph.workflow import run_agent
from research.modes import MODES

st.set_page_config(page_title="Research Intelligence Agent", page_icon="🧠",
                   layout="wide")

st.title("🧠 Research Intelligence Agent")
st.caption("Evidence-driven multi-agent research: parallel researchers, source "
           "quality scoring, independent fact checking, contradiction detection, "
           "gap reflection and citation-traceable reports.")

# ---------------------------------------------------------------- sidebar --
with st.sidebar:
    st.header("⚙️ Advanced settings")
    st.caption("Limits are hard-coded safety rails so every run terminates.")
    st.markdown(
        f"- **Research rounds:** up to {MAX_RESEARCH_ROUNDS}\n"
        f"- **Max sources:** {MAX_SOURCES}\n"
        f"- **Max evidence items:** {MAX_TOTAL_EVIDENCE}\n"
        f"- **Claims re-verified:** {MAX_FACT_CHECKS}\n"
        f"- **Report revisions:** {MAX_REPORT_REVISIONS}")
    st.divider()
    st.subheader("Status")
    st.markdown(f"**LLM:** `{LLM_MODEL}`\n\n**Endpoint:** `{LLM_BASE_URL}`")
    st.markdown("**Search:** " +
                ("Tavily API" if TAVILY_API_KEY else "DuckDuckGo fallback"))
    from memory.vector_store import get_memory
    mem = get_memory()
    if mem.enabled:
        st.success("Persistent research memory: ON")
    else:
        st.caption(f"Persistent memory: off ({mem._unavailable_reason.split(':')[0]}). "
                   "Set ENABLE_MEMORY=true and install chromadb to enable.")

# ------------------------------------------------------------------ input --
mode_keys = list(MODES.keys())
mode = st.selectbox(
    "Research mode", mode_keys, index=mode_keys.index("deep"),
    format_func=lambda k: f"{MODES[k].label}",
    help=" | ".join(f"{MODES[k].label}: {MODES[k].description}" for k in mode_keys))
st.caption(MODES[mode].description)

question = st.text_input(
    "Research question",
    placeholder="e.g. What are the real VRAM requirements for running 7B LLMs locally?")

documents = st.file_uploader(
    "Upload documents (PDF / TXT / Markdown) — used as clearly-labelled "
    "user-document evidence alongside web sources",
    type=["pdf", "txt", "md", "markdown"], accept_multiple_files=True,
    help="Uploaded files become first-class evidence sources, tagged "
         "'user_document' and kept distinct from web evidence.")

run_button = st.button("🔬 Run research", type="primary")
# Deliberately not `disabled=not question`: a text_input only commits on
# Enter or blur, so a disabled-until-typed button silently swallows the click.

# ------------------------------------------------------------------- run --
if run_button:
    if not question.strip():
        st.warning("Type a research question first.")
        st.stop()

    doc_payloads = [(f.name, f.getvalue()) for f in documents or []]
    final = None
    seen_lines = 0

    with st.status("Research agents working...", expanded=True) as status:
        progress = st.progress(0.0, text="starting")
        for state in run_agent(question, mode=mode, documents=doc_payloads):
            for line in state["log"][seen_lines:]:
                st.write(line)
            seen_lines = len(state["log"])
            nodes = state.get("current_nodes") or []
            if nodes:
                progress.progress(min(0.95, seen_lines / 25),
                                  text=f"🤖 Active: {', '.join(nodes)}")
            final = state
        progress.progress(1.0, text="done")
        status.update(label="Research complete", state="complete", expanded=False)

    if final:
        st.session_state["research_result"] = final

# ---------------------------------------------------------------- results --
final = st.session_state.get("research_result")
if final and final.get("report"):
    sources = final.get("sources", [])
    evidence = final.get("evidence", [])
    fact_checks = final.get("fact_checks", [])
    contradictions = final.get("contradictions", [])
    gaps = final.get("gaps", [])
    quality = final.get("report_quality", {}) or {}
    metadata = final.get("metadata", {}) or {}

    st.subheader("📊 Run overview")
    c = st.columns(6)
    c[0].metric("Sources", len(sources))
    c[1].metric("Evidence", len(evidence))
    c[2].metric("Claims verified", len(fact_checks))
    c[3].metric("Contradictions", len(contradictions))
    c[4].metric("Research gaps", len(gaps))
    dur = (metadata.get("stats") or {}).get("duration_s")
    c[5].metric("Duration", f"{dur}s" if dur else "-")

    if quality:
        st.subheader("📏 Report quality")
        qc = st.columns(5)
        for col, (label, val) in zip(qc, [
                ("Evidence coverage", quality.get("evidence_coverage", 0)),
                ("Citation coverage", quality.get("citation_coverage", 0)),
                ("Source quality", quality.get("source_quality", 0)),
                ("Question coverage", quality.get("question_coverage", 0)),
                ("Overall", quality.get("overall", 0))]):
            col.metric(label, f"{val:.0%}")
        st.markdown("**PASS** ✅" if quality.get("passed") else
                    "**Below threshold — delivered with disclosed limitations** ⚠️")
        if quality.get("issues"):
            st.caption("Reviewer notes: " + "; ".join(quality["issues"]))

    st.divider()
    st.subheader("📄 Final Report")
    st.markdown(final["report"])

    # ------------------------------------------------------------- exports --
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in question[:40])
    e1, e2 = st.columns(2)
    e1.download_button("⬇️ Download Markdown", data=final["report"],
                        file_name=f"report_{safe}.md", mime="text/markdown")
    export_payload = {
        "question": final.get("question"), "mode": final.get("mode"),
        "metadata": metadata,
        "sources": sources, "evidence": evidence, "fact_checks": fact_checks,
        "contradictions": contradictions, "gaps": gaps,
        "assessment": final.get("assessment"), "report": final["report"],
        "trace": final.get("log"), "errors": final.get("errors"),
    }
    e2.download_button("⬇️ Download JSON evidence dataset",
                        data=json.dumps(export_payload, indent=2, ensure_ascii=False,
                                        default=str),
                        file_name=f"research_{safe}.json", mime="application/json")

    # ------------------------------------------------------ detailed traces --
    with st.expander("🧭 Research trace (action log)"):
        for line in final.get("log", []):
            st.text(line)
        if final.get("errors"):
            st.error("Node errors (gracefully handled): " + " | ".join(final["errors"]))

    with st.expander(f"🌐 Sources ({len(sources)})"):
        if sources:
            ordered = sorted(sources, key=lambda s: s.get("quality_score", 0),
                             reverse=True)
            for s in ordered:
                tier_icon = {"high": "🟢", "medium": "🟡", "low": "🟠",
                             "weak": "🔴"}.get(s.get("quality_tier", ""), "⚪")
                primary = " · primary" if s.get("is_primary") else ""
                st.markdown(
                    f"{tier_icon} **[{s.get('source_id')}] "
                    f"[{s.get('title', '')}]({s.get('url', '')})** — "
                    f"{s.get('source_type')} · quality {s.get('quality_score', 0):.2f} "
                    f"(authority {s.get('authority_score', 0):.2f}, "
                    f"relevance {s.get('relevance_score', 0):.2f}, "
                    f"recency {s.get('recency_score', 0):.2f}){primary}")

    with st.expander(f"🧪 Evidence ({len(evidence)})"):
        for ev in evidence:
            ok = "✅" if ev.get("supported", True) else "⚠️ unsupported"
            st.markdown(f"**{ev.get('claim', '')}** · conf "
                        f"{ev.get('confidence', 0):.2f} {ok} · "
                        f"`{ev.get('source_id')}` — {ev.get('source_url', '')}")
            if ev.get("supporting_passage"):
                st.caption(f"“{ev['supporting_passage'][:300]}”")

    with st.expander(f"🔍 Fact checks ({len(fact_checks)})"):
        for fc in fact_checks:
            icon = {"supported": "✅", "partially_supported": "🟡",
                    "contradicted": "❌", "unverifiable": "❓"}.get(
                        fc.get("status", ""), "❓")
            st.markdown(f"{icon} **{fc.get('status')}** ({fc.get('confidence', 0):.0%}): "
                        f"{fc.get('claim', '')}")
            st.caption(fc.get("explanation", ""))

    with st.expander(f"⚡ Contradictions ({len(contradictions)})"):
        for cdict in contradictions:
            st.markdown(f"**{cdict.get('topic', '')}** — {cdict.get('status')}")
            st.markdown(f"- Claim A [{cdict.get('source_a', '')}]: "
                        f"{cdict.get('claim_a', '')}")
            st.markdown(f"- Claim B [{cdict.get('source_b', '')}]: "
                        f"{cdict.get('claim_b', '')}")
            if cdict.get("possible_reasons"):
                st.caption("Possible reasons: " + ", ".join(cdict["possible_reasons"]))

    with st.expander(f"🕳️ Research gaps ({len(gaps)})"):
        for gdict in gaps:
            icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(
                gdict.get("importance", ""), "⚪")
            st.markdown(f"{icon} **{gdict.get('importance', '')}** — "
                        f"{gdict.get('gap', '')}")
            if gdict.get("recommended_query"):
                st.caption(f"Recommended query: `{gdict['recommended_query']}`")
else:
    st.info("Pick a research mode, ask a question (optionally upload documents), "
            "and click **Run research**. Example: *“What VRAM is required to run "
            "7B-parameter LLMs locally, and how do quantization methods change "
            "that?”* in Deep Research mode demonstrates parallel research, "
            "contradiction detection and citation-traceable reporting.")
