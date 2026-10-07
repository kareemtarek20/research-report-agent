"""
Contradiction Detector node (Phase 6).

Finds claims from different sources that genuinely conflict, records possible
reasons for disagreement, and never silently picks a winner — unresolved
contradictions are carried into the report. Also folds cross-source agreement
back into source quality scores.
"""

import logging

from research.contradictions import detect_contradictions, compute_agreement

log = logging.getLogger("agent.contradictions")


def contradiction_node(state: dict) -> dict:
    evidence = state.get("evidence", [])
    sources = [dict(s) for s in state.get("sources", [])]

    found = detect_contradictions(evidence)
    compute_agreement(evidence, sources)

    unresolved = [c for c in found if c.get("status") != "resolved"]
    log.info("contradictions=%s unresolved=%s", len(found), len(unresolved))
    if found:
        lines = [f"⚡ Contradictions: {len(found)} found "
                 f"({len(unresolved)} unresolved) — will be reported transparently"]
    else:
        lines = ["⚡ Contradictions: none detected across sources"]
    return {"contradictions": found, "sources": sources, "log": lines}
