"""
Research Intelligence Agent — CLI entry point.

`python agent.py` runs the full evidence-driven research graph and prints a
live action trace, then the final report. The web UI lives in app.py
(`streamlit run app.py`). The graph itself is in graph/workflow.py.
"""

import sys

from graph.workflow import run_agent
from research.modes import MODES, get_mode


def main():
    # Log lines are emoji-prefixed; Windows consoles default to cp1252 and raise
    # UnicodeEncodeError without this.
    sys.stdout.reconfigure(encoding="utf-8")

    topic = input("Research question: ").strip()
    if not topic:
        print("Nothing to research.")
        return
    print("\nResearch modes: " + ", ".join(f"{k} ({m.label})" for k, m in MODES.items()))
    mode = input("Mode [deep]: ").strip() or "deep"
    print(f"\nRunning {get_mode(mode).label}...\n")

    final_state = None
    for state in run_agent(topic, mode=mode):
        new_lines = state["log"][len(final_state["log"]) if final_state else 0:]
        for line in new_lines:
            print(line)
        final_state = state

    if final_state and final_state.get("report"):
        print("\n\n===== REPORT =====\n")
        print(final_state["report"])
        quality = final_state.get("report_quality", {})
        if quality:
            print(f"\nQuality: evidence {quality.get('evidence_coverage', 0):.0%} | "
                  f"citations {quality.get('citation_coverage', 0):.0%} | "
                  f"source quality {quality.get('source_quality', 0):.0%} | "
                  f"question coverage {quality.get('question_coverage', 0):.0%}")
    else:
        print("\nThe agent finished without producing a report. "
              "Check the log above for failures.")


if __name__ == "__main__":
    main()
