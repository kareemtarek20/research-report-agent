"""
Central configuration for the Research & Report Agent.

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

# --- LLM settings ---
# Defaults below point at Xkiro's OpenAI-compatible gateway. Any other
# OpenAI-compatible endpoint works by changing these three values in .env.
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.xkiro.com/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen/qwen3.8-omni-flash:free")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")  # client requires a non-empty string
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))

# --- Web search settings ---
# Tavily has a generous free tier and is built for LLM agents.
# Get a free key at https://tavily.com
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# --- Agent behavior ---
MAX_SEARCH_ITERATIONS = int(os.getenv("MAX_SEARCH_ITERATIONS", "4"))
MAX_RESULTS_PER_SEARCH = int(os.getenv("MAX_RESULTS_PER_SEARCH", "5"))
