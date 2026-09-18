import os
from pathlib import Path


def _load_env_file() -> None:
    candidates = [
        Path(".env"),
        Path("../.env"),
        Path(__file__).resolve().parent / ".env",
        Path(__file__).resolve().parent.parent / ".env",
        Path(__file__).resolve().parent.parent.parent / ".env",
    ]
    seen = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
            if resolved in seen or not resolved.is_file():
                continue
            seen.add(resolved)
            with resolved.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip("\"'")
                    if k and k not in os.environ:
                        os.environ[k] = v
        except Exception:
            pass


_load_env_file()

LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "gemini")
LLM_API_KEY = os.environ.get("LLM_API_KEY") or os.environ.get("GEMINI_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "gemini-3.5-flash-lite")
PORT = int(os.environ.get("PORT", "8000"))

# Per-attempt LLM timeouts: up to 3 tries. 7+7+7 = 21s of network time, leaving
# room for backoff sleeps between attempts while staying under the judge's 30s
# per-request hard timeout (with margin for guardrails + LP solve, ~5ms).
LLM_ATTEMPT_TIMEOUT_SECONDS = (7.0, 7.0, 7.0)

# Total wall-clock budget for the whole interpret_notes() call (network time +
# backoff sleeps). Kept below the 30s hard timeout so main.py/guardrails/optimize
# always have time to run afterward.
LLM_TOTAL_BUDGET_SECONDS = 26.0

# Fallback backoff (used only when the provider gives no retry hint): exponential
# with jitter, capped so a single sleep can't eat the whole remaining budget.
LLM_BACKOFF_BASE_SECONDS = 1.0
LLM_BACKOFF_MAX_SECONDS = 8.0

