import os

LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "gemini")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "gemini-2.0-flash")
PORT = int(os.environ.get("PORT", "8000"))

# Per-attempt LLM timeouts: first try, then retry. 12 + 0.5 (backoff) + 8 keeps the
# worst case near 20.5 s — safely inside the judge's 30 s per-request hard timeout,
# with time to spare for the keyword fallback + LP solve (~5 ms).
LLM_ATTEMPT_TIMEOUT_SECONDS = (12.0, 8.0)
