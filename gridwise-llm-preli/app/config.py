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

# Per-attempt LLM timeouts: first try, then retry. 12 + 0.5 (backoff) + 8 keeps the
# worst case near 20.5 s — safely inside the judge's 30 s per-request hard timeout,
# with time to spare for the keyword fallback + LP solve (~5 ms).
LLM_ATTEMPT_TIMEOUT_SECONDS = (12.0, 8.0)

