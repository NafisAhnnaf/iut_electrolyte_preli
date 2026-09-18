import os

LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "gemini")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "gemini-2.0-flash")
PORT = int(os.environ.get("PORT", "8000"))

LLM_REQUEST_TIMEOUT_SECONDS = 20.0
