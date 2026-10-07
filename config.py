"""
Central configuration for the Research Intelligence Agent.

The agent talks to your LLM through an OpenAI-compatible /chat/completions
endpoint. This works for:
  - Ollama            -> base_url="http://localhost:11434/v1"
  - LM Studio         -> base_url="http://localhost:1234/v1"
  - vLLM server       -> base_url="http://localhost:8000/v1"
  - Alibaba DashScope  -> base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
  - Any other OpenAI-compatible provider

Just change LLM_BASE_URL / LLM_MODEL / LLM_API_KEY below or via a .env file.
"""

import os
from dotenv import load_dotenv

load_dotenv()


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _bool_env(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


# --- LLM settings ---
# Defaults below point at Xkiro's OpenAI-compatible gateway. Any other
# OpenAI-compatible endpoint works by changing these three values in .env.
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.xkiro.com/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen/qwen3.8-omni-flash:free")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")  # client requires a non-empty string
LLM_TEMPERATURE = _float_env("LLM_TEMPERATURE", 0.3)
# Deterministic temperature for planning, extraction, fact checking, scoring.
LLM_TEMP_PRECISE = _float_env("LLM_TEMP_PRECISE", 0.0)
LLM_TIMEOUT_SECONDS = _int_env("LLM_TIMEOUT_SECONDS", 90)
LLM_MAX_RETRIES = _int_env("LLM_MAX_RETRIES", 2)  # retries on timeout/rate limit

# --- Web search settings ---
# Tavily has a generous free tier and is built for LLM agents.
# Get a free key at https://tavily.com
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# --- Academic search ---
# Semantic Scholar works without a key (rate-limited); a key raises limits.
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")

# --- Agent behavior (hard limits keep the graph terminating) ---
MAX_SEARCH_ITERATIONS = _int_env("MAX_SEARCH_ITERATIONS", 4)   # queries per researcher per round
MAX_RESULTS_PER_SEARCH = _int_env("MAX_RESULTS_PER_SEARCH", 5)
MAX_RESEARCH_ROUNDS = _int_env("MAX_RESEARCH_ROUNDS", 2)       # gap-driven re-research loops
MAX_SOURCES = _int_env("MAX_SOURCES", 24)                      # total deep-read sources per run
MAX_TOTAL_EVIDENCE = _int_env("MAX_TOTAL_EVIDENCE", 40)
MAX_EVIDENCE_PER_SOURCE = _int_env("MAX_EVIDENCE_PER_SOURCE", 5)
MAX_PAGE_CHARS = _int_env("MAX_PAGE_CHARS", 6000)              # text fed per page to extractor
MAX_FACT_CHECKS = _int_env("MAX_FACT_CHECKS", 8)               # claims independently re-verified
MAX_REPORT_REVISIONS = _int_env("MAX_REPORT_REVISIONS", 1)     # 0 disables the revision loop

# --- Report quality thresholds (Phase 12) ---
QUALITY_EVIDENCE_COVERAGE = _float_env("QUALITY_EVIDENCE_COVERAGE", 0.60)
QUALITY_CITATION_COVERAGE = _float_env("QUALITY_CITATION_COVERAGE", 0.70)
QUALITY_QUESTION_COVERAGE = _float_env("QUALITY_QUESTION_COVERAGE", 0.70)
QUALITY_SOURCE_QUALITY = _float_env("QUALITY_SOURCE_QUALITY", 0.45)

# --- Research memory (Phase 8, optional) ---
ENABLE_MEMORY = _bool_env("ENABLE_MEMORY", False)
MEMORY_DIR = os.getenv("MEMORY_DIR", ".memory_db")

# --- Observability ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", "research_agent.log")

# --- UI defaults ---
VALID_MODES = ("quick", "deep", "academic", "technical", "competitive",
               "fact_check", "gap_analysis")
DEFAULT_MODE = os.getenv("DEFAULT_MODE", "deep")
