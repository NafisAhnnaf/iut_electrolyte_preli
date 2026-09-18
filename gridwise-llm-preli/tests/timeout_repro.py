"""Reproduce the LLM-provider worst cases that threaten the judge's 30 s hard timeout.

No request body can trigger these: the input is irrelevant. The worst case happens when
the LLM provider is slow or unresponsive. This script fakes that by running the service
behind a local "bad proxy" (via HTTPS_PROXY, which httpx honours) and timing a normal
request.

Scenarios:
  hang     proxy accepts the TCP connection and never replies        (provider hangs)
  trickle  proxy answers the CONNECT, then drips one byte every 5 s   (provider stalls
           during the TLS handshake)
  drip     provider sends response headers, then drips the body one byte every 2 s.
           Every individual read succeeds, so httpx's per-read timeout NEVER fires --
           only the per-attempt asyncio.wait_for ceiling stops it. (In-process test:
           the interpreter's HTTP call is pointed at a local drip server.)

Usage (no real API key needed -- a dummy key is used and nothing reaches Google):
    python tests/timeout_repro.py hang
    python tests/timeout_repro.py trickle
    python tests/timeout_repro.py drip

Pass: HTTP 200 with a schema-valid body in < 30 s (the degraded keyword fallback answers).
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROXY_PORT = 18999
SERVICE_PORT = 18998
JUDGE_TIMEOUT_S = 30.0


def _serve(mode: str) -> None:
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", PROXY_PORT))
    srv.listen(16)

    def handle(conn: socket.socket) -> None:
        try:
            conn.recv(4096)  # the CONNECT request
            if mode == "trickle":
                conn.sendall(b"HTTP/1.1 200 Connection established\r\n\r\n")
                while True:  # never completes the TLS handshake, but never goes silent
                    time.sleep(5)
                    conn.sendall(b"\x16")
            else:  # hang
                time.sleep(3600)
        except OSError:
            pass
        finally:
            conn.close()

    while True:
        c, _ = srv.accept()
        threading.Thread(target=handle, args=(c,), daemon=True).start()


def _drip_server(port: int) -> None:
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", port))
    srv.listen(16)

    def handle(conn: socket.socket) -> None:
        try:
            conn.recv(65536)
            conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                         b"Content-Length: 1000\r\n\r\n")
            for _ in range(1000):
                time.sleep(2)
                conn.sendall(b" ")
        except OSError:
            pass
        finally:
            conn.close()

    while True:
        c, _ = srv.accept()
        threading.Thread(target=handle, args=(c,), daemon=True).start()


def run_drip() -> int:
    """In-process: route the interpreter's LLM HTTP call to a local drip server."""
    import asyncio

    import httpx

    sys.path.insert(0, str(ROOT))
    from app import interpreter  # noqa: E402

    port = PROXY_PORT + 2
    threading.Thread(target=_drip_server, args=(port,), daemon=True).start()
    time.sleep(0.3)

    async def dripping_llm(user_message: str, timeout: float) -> str:
        # Same httpx usage as the real provider call, only the URL differs.
        async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
            resp = await client.post(f"http://127.0.0.1:{port}/", json={"q": user_message})
            return resp.text

    interpreter._call_llm = dripping_llm
    case = json.loads((ROOT / "sample_cases.json").read_text(encoding="utf-8"))["cases"][0]["input"]
    print("[drip] provider sends headers, then 1 byte every 2 s...")
    t0 = time.time()
    out = asyncio.run(interpreter.interpret_notes(case["operator_notes"]))
    elapsed = time.time() - t0
    ok = len(out) == len(case["operator_notes"]) and elapsed < JUDGE_TIMEOUT_S
    print(f"[drip] interpreter returned {len(out)} entries in {elapsed:.1f}s -> "
          f"{'PASS (under the 30 s judge timeout)' if ok else 'FAIL'}")
    return 0 if ok else 1


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "hang"
    if mode == "drip":
        return run_drip()
    if mode not in ("hang", "trickle"):
        print("mode must be 'hang', 'trickle' or 'drip'")
        return 2

    threading.Thread(target=_serve, args=(mode,), daemon=True).start()

    env = dict(os.environ)
    env.update({
        "LLM_PROVIDER": "gemini",
        "LLM_API_KEY": "dummy-not-a-real-key",
        "HTTPS_PROXY": f"http://127.0.0.1:{PROXY_PORT}",
        "HTTP_PROXY": f"http://127.0.0.1:{PROXY_PORT}",
        "NO_PROXY": "127.0.0.1,localhost",
    })
    for k in ("https_proxy", "http_proxy", "no_proxy", "ALL_PROXY", "all_proxy"):
        env.pop(k, None)

    service = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app",
         "--host", "127.0.0.1", "--port", str(SERVICE_PORT), "--log-level", "warning"],
        cwd=ROOT, env=env,
    )
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # direct
        for _ in range(60):
            try:
                opener.open(f"http://127.0.0.1:{SERVICE_PORT}/health", timeout=1)
                break
            except OSError:
                time.sleep(0.5)

        case = json.loads((ROOT / "sample_cases.json").read_text(encoding="utf-8"))["cases"][0]["input"]
        req = urllib.request.Request(
            f"http://127.0.0.1:{SERVICE_PORT}/optimize-energy",
            data=json.dumps(case).encode(), headers={"Content-Type": "application/json"},
            method="POST",
        )
        print(f"[{mode}] sending one request; the provider is unresponsive...")
        t0 = time.time()
        try:
            resp = opener.open(req, timeout=90)
            status, body = resp.status, json.load(resp)
        except Exception as exc:  # noqa: BLE001
            status, body = f"error ({exc})", None
        elapsed = time.time() - t0

        ok = status == 200 and body is not None and elapsed < JUDGE_TIMEOUT_S
        print(f"[{mode}] status={status} elapsed={elapsed:.1f}s  -> "
              f"{'PASS (under the 30 s judge timeout)' if ok else 'FAIL'}")
        return 0 if ok else 1
    finally:
        service.terminate()
        try:
            service.wait(timeout=5)
        except subprocess.TimeoutExpired:
            service.kill()


if __name__ == "__main__":
    raise SystemExit(main())
