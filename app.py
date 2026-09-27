"""
Streamlit UI for the Research & Report Agent.

Run with:  streamlit run app.py
"""

import streamlit as st
from agent import run_agent

st.set_page_config(page_title="Research & Report Agent", page_icon="🧠", layout="centered")

st.title("🧠 Research & Report Agent")
st.caption("Autonomous multi-step research agent — plans searches, reads sources, "
           "decides when it knows enough, and writes a report.")

topic = st.text_input("Research topic", placeholder="e.g. Impact of quantum computing on cryptography")
# Deliberately not `disabled=not topic`: a text_input only commits on Enter or
# blur, so a disabled-until-typed button silently swallows the first click.
run_button = st.button("Run agent", type="primary")

if run_button:
    if not topic.strip():
        st.warning("Type a topic first, then click Run agent.")
        st.stop()

    seen_lines = 0
    final_state = None

    with st.status("Agent is working...", expanded=True) as status:
        for state in run_agent(topic):
            new_lines = state["log"][seen_lines:]
            for line in new_lines:
                st.write(line)
            seen_lines = len(state["log"])
            final_state = state
        status.update(label="Done!", state="complete", expanded=False)

    if final_state and final_state.get("report"):
        st.divider()
        st.subheader("📄 Final Report")
        st.markdown(final_state["report"])
        st.download_button(
            "Download report as Markdown",
            data=final_state["report"],
            file_name=f"report_{topic[:30].replace(' ', '_')}.md",
            mime="text/markdown",
        )
