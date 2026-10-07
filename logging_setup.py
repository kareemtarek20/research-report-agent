"""
Structured logging / observability (Phase 15).

Console + optional rotating file log. Log lines are structured key=value
style so node execution, queries, counts, timings, retries and final quality
scores are easy to grep. API keys are never logged.
"""

import logging
import os
import time
from logging.handlers import RotatingFileHandler

from config import LOG_LEVEL, LOG_FILE

_configured = False


def setup_logging(force: bool = False) -> None:
    global _configured
    if _configured and not force:
        return
    root = logging.getLogger("agent")
    root.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))
    root.handlers.clear()

    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s | %(message)s")
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(console)

    if LOG_FILE and os.getenv("DISABLE_LOG_FILE", "").lower() not in ("1", "true"):
        try:
            fileh = RotatingFileHandler(LOG_FILE, maxBytes=2_000_000,
                                        backupCount=2, encoding="utf-8")
            fileh.setFormatter(fmt)
            root.addHandler(fileh)
        except OSError:
            pass  # log file is a nicety; never crash the agent over it

    root.propagate = False
    _configured = True


class NodeTimer:
    """Context manager that logs node execution time and updates stats."""

    def __init__(self, node_name: str, stats: dict):
        self.node = node_name
        self.stats = stats
        self.t0 = 0.0

    def __enter__(self):
        self.t0 = time.time()
        logging.getLogger("agent.node").info("node=%s event=start", self.node)
        return self

    def __exit__(self, exc_type, exc, tb):
        elapsed = round(time.time() - self.t0, 2)
        nodes = self.stats.setdefault("node_times", {})
        nodes[self.node] = round(nodes.get(self.node, 0.0) + elapsed, 2)
        if exc_type:
            logging.getLogger("agent.node").warning(
                "node=%s event=error duration_s=%s error=%s", self.node, elapsed, exc)
        else:
            logging.getLogger("agent.node").info(
                "node=%s event=done duration_s=%s", self.node, elapsed)
        return False  # don't swallow exceptions
