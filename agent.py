"""
Research & Report Agent
========================

Graph:  Plan -> Search -> Extract -> Evaluate -> (loop back to Plan) -> Write Report

- Plan:      decide the next search query given what's been learned so far
- Search:    call the web_search tool
- Extract:   fetch top result pages and summarize relevant facts into notes
- Evaluate:  ask the LLM "do I have enough to write a good report yet?"
- Write:     produce the final structured report

A hard iteration cap (config.MAX_SEARCH_ITERATIONS) guarantees the loop
terminates even if the model never says "sufficient".
"""

import sys
from typing import TypedDict
from langgraph.graph import StateGraph, END

from llm_client import chat, chat_json
from tools import web_search, read_url
from config import MAX_SEARCH_ITERATIONS, MAX_RESULTS_PER_SEARCH


class AgentState(TypedDict):
    topic: str
    iteration: int
    queries_done: list[str]
    notes: list[str]
    sufficient: bool
    report: str
    log: list[str]  # human-readable trace of what the agent did, for the UI
    # Scratch keys threaded between adjacent nodes. LangGraph only persists
    # channels declared in the schema, so these must be declared here.
    next_query: str
    search_results: list[dict]


def plan_node(state: AgentState) -> AgentState:
    state["log"].append(f"🧭 Planning search #{state['iteration'] + 1}...")

    prior = "\n".join(state["notes"]) or "(nothing yet)"
    done = ", ".join(state["queries_done"]) or "(none)"

    messages = [
        {
            "role": "system",
            "content": (
                "You are a research planner. Given a topic and notes gathered so far, "
                "produce the single best next web search query to fill the biggest "
                "remaining gap in knowledge. Avoid repeating prior queries. "
                'Reply as JSON: {"query": "..."}'
            ),
        },
        {
            "role": "user",
            "content": (
                f"Topic: {state['topic']}\n\n"
                f"Queries already tried: {done}\n\n"
                f"Notes gathered so far:\n{prior}"
            ),
        },
    ]
    result = chat_json(messages)
    query = result.get("query", state["topic"])
    state["queries_done"].append(query)
    state["next_query"] = query
    state["log"].append(f"   → next query: \"{query}\"")
    return state


def search_node(state: AgentState) -> AgentState:
    query = state["next_query"]
    state["log"].append(f"🔎 Searching: \"{query}\"")
    results = web_search(query)
    state["search_results"] = results[:MAX_RESULTS_PER_SEARCH]
    state["log"].append(f"   → found {len(state['search_results'])} results")
    return state


def extract_node(state: AgentState) -> AgentState:
    results = state.get("search_results", [])
    if not results:
        state["log"].append("⚠️  No results to extract from this round.")
        return state

    state["log"].append("📄 Reading top results and extracting facts...")
    pages_text = []
    for r in results[:3]:  # only deep-read the top 3 to keep this fast
        content = read_url(r["url"])
        pages_text.append(f"SOURCE: {r['title']} ({r['url']})\n{content}\n")

    combined = "\n---\n".join(pages_text)
    messages = [
        {
            "role": "system",
            "content": (
                "You extract factual, relevant notes from web page content for a "
                "research report. Write concise bullet points. Include the source "
                "title in parentheses after each fact. Ignore ads, navigation, and "
                "irrelevant content."
            ),
        },
        {
            "role": "user",
            "content": f"Topic: {state['topic']}\n\nPage content:\n{combined}",
        },
    ]
    new_notes = chat(messages)
    state["notes"].append(new_notes)
    state["log"].append("   → notes updated")
    return state


def evaluate_node(state: AgentState) -> AgentState:
    state["log"].append("🤔 Evaluating whether we have enough information...")
    all_notes = "\n".join(state["notes"])
    messages = [
        {
            "role": "system",
            "content": (
                "You judge whether enough information has been gathered to write "
                "a solid, well-rounded report on the topic. "
                'Reply as JSON: {"sufficient": true/false, "reason": "..."}'
            ),
        },
        {
            "role": "user",
            "content": f"Topic: {state['topic']}\n\nNotes so far:\n{all_notes}",
        },
    ]
    result = chat_json(messages)
    state["sufficient"] = bool(result.get("sufficient", False))
    state["log"].append(f"   → sufficient: {state['sufficient']} ({result.get('reason', '')})")
    state["iteration"] += 1
    return state


def write_node(state: AgentState) -> AgentState:
    state["log"].append("✍️  Writing final report...")
    all_notes = "\n".join(state["notes"])
    messages = [
        {
            "role": "system",
            "content": (
                "You are a research report writer. Using ONLY the notes provided, "
                "write a clear, well-structured report with headings, in Markdown. "
                "Include a short 'Sources reviewed' note at the end based on the "
                "sources mentioned in the notes. Do not invent facts not in the notes."
            ),
        },
        {
            "role": "user",
            "content": f"Topic: {state['topic']}\n\nNotes:\n{all_notes}",
        },
    ]
    state["report"] = chat(messages)
    state["log"].append("✅ Report complete.")
    return state


def should_continue(state: AgentState) -> str:
    if state["sufficient"] or state["iteration"] >= MAX_SEARCH_ITERATIONS:
        return "write"
    return "plan"


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("plan", plan_node)
    graph.add_node("search", search_node)
    graph.add_node("extract", extract_node)
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("write", write_node)

    graph.set_entry_point("plan")
    graph.add_edge("plan", "search")
    graph.add_edge("search", "extract")
    graph.add_edge("extract", "evaluate")
    graph.add_conditional_edges("evaluate", should_continue, {"plan": "plan", "write": "write"})
    graph.add_edge("write", END)

    return graph.compile()


def run_agent(topic: str):
    """
    Run the agent on a topic. Yields state snapshots after each step so a UI
    can show live progress (see app.py). The final yield has a populated
    'report' field.
    """
    app = build_graph()
    initial_state: AgentState = {
        "topic": topic,
        "iteration": 0,
        "queries_done": [],
        "notes": [],
        "sufficient": False,
        "report": "",
        "log": [],
        "next_query": "",
        "search_results": [],
    }
    for step_state in app.stream(initial_state, stream_mode="values"):
        yield step_state


if __name__ == "__main__":
    # Log lines are emoji-prefixed; Windows consoles default to cp1252 and raise
    # UnicodeEncodeError without this.
    sys.stdout.reconfigure(encoding="utf-8")

    topic = input("Research topic: ")
    final_state = None
    for state in run_agent(topic):
        for line in state["log"][len(final_state["log"]) if final_state else 0:]:
            print(line)
        final_state = state
    print("\n\n===== REPORT =====\n")
    print(final_state["report"])
