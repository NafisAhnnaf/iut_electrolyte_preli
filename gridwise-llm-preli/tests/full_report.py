"""Full-visibility test run: POSTs every case in one or more packs to a running
service, records latency, detects whether the LLM path or the emergency keyword
fallback answered (by watching the service's log file), and saves every raw
response body to a JSON file for inspection.

    python tests/full_report.py --base-url http://localhost:8000 \\
        --cases sample_cases.json tests/extra_cases.json \\
        --log-file /tmp/gridwise_server.log \\
        --out /tmp/gridwise_full_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402


def tail_new_lines(log_path: Path, offset: int) -> tuple[list[str], int]:
    if not log_path.exists():
        return [], offset
    data = log_path.read_text(encoding="utf-8", errors="replace")
    new = data[offset:]
    return new.splitlines(), len(data)


def classify_llm_usage(log_lines: list[str]) -> str:
    joined = "\n".join(log_lines)
    if "using keyword fallback" in joined:
        return "FALLBACK (LLM failed twice)"
    if "LLM interpretation attempt 1 failed" in joined:
        return "LLM (succeeded on retry)"
    return "LLM"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--cases", nargs="+", required=True, help="one or more case pack JSON files")
    ap.add_argument("--log-file", default="/tmp/gridwise_server.log",
                    help="server stdout/stderr log, used to detect LLM-vs-fallback per request")
    ap.add_argument("--out", default="/tmp/gridwise_full_report.json")
    ap.add_argument("--delay", type=float, default=0.0,
                    help="seconds to sleep between requests, to avoid self-inflicted rate limiting")
    args = ap.parse_args()

    base = args.base_url.rstrip("/")
    log_path = Path(args.log_file)
    log_offset = len(log_path.read_text(encoding="utf-8", errors="replace")) if log_path.exists() else 0

    all_cases: list[tuple[str, dict[str, Any]]] = []
    for cases_path in args.cases:
        pack = json.loads(Path(cases_path).read_text(encoding="utf-8"))
        pack_name = Path(cases_path).stem
        for case in pack["cases"]:
            all_cases.append((pack_name, case))

    results: list[dict[str, Any]] = []
    print(f"{'PACK':<14} {'CASE':<12} {'STATUS':<8} {'LATENCY':>9}  {'LLM PATH':<24} {'NOTES -> TYPES'}")
    print("-" * 110)

    with httpx.Client(timeout=35.0) as client:
        for i, (pack_name, case) in enumerate(all_cases):
            if i > 0 and args.delay > 0:
                time.sleep(args.delay)
            cid = case["id"]
            payload = case["input"]
            t0 = time.perf_counter()
            try:
                r = client.post(f"{base}/optimize-energy", json=payload)
                elapsed_ms = (time.perf_counter() - t0) * 1000
                status = r.status_code
                body = r.json()
            except Exception as exc:  # noqa: BLE001
                elapsed_ms = (time.perf_counter() - t0) * 1000
                status = "ERR"
                body = {"error": f"{type(exc).__name__}: {exc}"}

            new_lines, log_offset = tail_new_lines(log_path, log_offset)
            llm_path = classify_llm_usage(new_lines) if status == 200 else "n/a"

            types = "-"
            if status == 200 and isinstance(body.get("directive_interpretation"), list):
                types = ",".join(
                    d.get("directive_type", "?") for d in body["directive_interpretation"]
                )

            results.append({
                "pack": pack_name,
                "id": cid,
                "status": status,
                "latency_ms": round(elapsed_ms, 1),
                "llm_path": llm_path,
                "directive_types": types,
                "request": payload,
                "response": body,
                "log_lines": new_lines,
            })
            print(f"{pack_name:<14} {cid:<12} {str(status):<8} {elapsed_ms:>7.0f}ms  {llm_path:<24} {types}")

    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")

    n = len(results)
    ok = sum(1 for r in results if r["status"] == 200)
    fallback = sum(1 for r in results if "FALLBACK" in r["llm_path"])
    latencies = [r["latency_ms"] for r in results if r["status"] == 200]
    print("-" * 110)
    print(f"{ok}/{n} returned HTTP 200")
    if fallback:
        print(f"** {fallback} request(s) used the keyword FALLBACK, not the LLM **")
    else:
        print("all successful requests were answered by the real LLM (no fallback triggered)")
    if latencies:
        latencies.sort()
        p50 = latencies[len(latencies) // 2]
        p95 = latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))]
        print(f"latency: min {min(latencies):.0f}ms  p50 {p50:.0f}ms  p95 {p95:.0f}ms  max {max(latencies):.0f}ms")
    print(f"full request/response bodies + relevant log lines saved to {args.out}")
    return 0 if ok == n else 1


if __name__ == "__main__":
    raise SystemExit(main())
