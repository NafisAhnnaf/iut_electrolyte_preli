# GridWise LLM — Smart Campus Energy Optimization (BUP CSE Fest 2026, Online Preliminary)

An HTTP API that reads 1–3 natural-language campus operator notes, interprets them with an
LLM into machine-checkable directives, validates that interpretation with deterministic
guardrails, and then solves a 24-hour least-cost energy schedule (grid + rooftop solar +
battery) with a linear program. Every response is replay-checked against the full GridWise
rule set before it leaves the service.

## Architecture

```
operator_notes ──► LLM interpreter ──► deterministic guardrails ──► LP optimizer ──► replay validator ──► JSON response
                   (Gemini, temp 0,     (enum/window/numeric         (scipy HiGHS)     (all judge checks
                    JSON mode, up to     normalization; never                            re-run in-process)
                    3 attempts)          invents a directive)
```

- **LLM interpreter** (`app/interpreter.py`): one batched call for all notes, temperature 0,
  JSON output. The model extracts *intermediate* fields only (directive type, start/end hour,
  raw value + value kind); it never does arithmetic or unit conversion. Up to 3 attempts of
  7 s each, waiting between attempts by the provider's own rate-limit hint (Retry-After /
  RetryInfo) or exponential backoff with jitter, inside a hard 26 s total budget so every
  request finishes under the judge's 30 s limit (each attempt also has a hard wall-clock
  ceiling, so a provider that stalls mid-response cannot hang the request). If every attempt
  fails, or no key is configured, an emergency deterministic keyword classifier keeps the
  service answering (degraded mode — the primary interpretation path is always the LLM).
- **Guardrails** (`app/guardrails.py`): converts intermediate fields into the exact
  `structured_adjustment` shapes — end-exclusive windows ("1 PM to 3 PM" → `[13,14]`),
  percent→factor ("80% reduction" → `factor 0.2`), percent-of-capacity reserves, MWh→kWh,
  "until midnight" → hour 24, and windows that cross midnight ("10 PM to 2 AM" →
  `[0,1,22,23]`) — and enforces enum membership, hour ranges (unique, ascending, 0–23), and
  numeric bounds (0 ≤ factor ≤ 1, 0 ≤ reserve ≤ capacity, finite cap ≥ 0). Loosely typed model
  output (`"60"`, `18.0`) is normalized; anything unusable is coerced to a logged `no_op`.
  Nothing is invented and nothing crashes.
- **Optimizer** (`app/optimizer.py`): LP over grid/solar/charge/discharge per hour —
  energy balance, battery envelope (with directive reserves), rate limits, grid caps,
  end-of-day neutrality; objective = Σ grid_kwh × tariff. Rounded flows are recomputed so
  the returned numbers balance exactly, and the reported totals are recalculated from the
  final `hourly_plan`.
- **Replay validator** (`app/validator.py`): re-implements every judge check and runs on our
  own response before it is returned.

## Model / provider

- Default: **Google Gemini**, model `gemini-3.5-flash-lite` (set by `LLM_MODEL`).
- Also supported: `LLM_PROVIDER=openai` (api.openai.com) and `LLM_PROVIDER=groq`
  (OpenAI-compatible endpoint) with the corresponding key in `LLM_API_KEY`.

## Environment variables (names only — never commit values)

| Variable | Meaning | Default |
|---|---|---|
| `LLM_PROVIDER` | `gemini` \| `openai` \| `groq` | `gemini` |
| `LLM_API_KEY` | Provider API key (supplied at runtime). `GEMINI_API_KEY` is accepted as an alias | — |
| `LLM_MODEL` | Model identifier | `gemini-3.5-flash-lite` |
| `PORT` | HTTP port to bind | `8000` |

For local runs the variables may also be placed in a `.env` file next to the project (read at
startup; real environment variables take precedence). `.env` is gitignored and excluded from
the Docker image by `.dockerignore` — never commit it.

## Local quickstart (clean environment)

```bash
git clone <this repository>
cd gridwise-llm-preli
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt

export LLM_API_KEY=<your key>          # Windows: set LLM_API_KEY=<your key>
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

One sample request (SAMPLE-01 from the public case pack):

```bash
python - <<'EOF' > /tmp/sample01.json
import json; print(json.dumps(json.load(open("sample_cases.json"))["cases"][0]["input"]))
EOF
curl -s -X POST http://localhost:8000/optimize-energy \
     -H "Content-Type: application/json" -d @/tmp/sample01.json
```

Expected shape: `scenario_id` echo, one `directive_interpretation` entry per note in
`note_index` order, a 24-entry `hourly_plan`, and recalculated
`total_grid_kwh` / `total_cost_bdt` / `peak_grid_kwh` (SAMPLE-01 optimal cost: 38365 BDT).

## Public-sample test procedure

```bash
# Engine only (no LLM, no network): optimizer + validator against the 10 public cases
python tests/run_public_cases.py --engine-only

# Full end-to-end against a running service (schema, interpretation semantics,
# ground-truth replay, totals, cost vs reference, paraphrase + robustness checks)
python tests/run_public_cases.py --base-url http://localhost:8000
```

Expected result: a PASS table ending `.../... checks passed`, exit code 0 (engine-only:
`30/30 checks passed`; every case matches the organizer's optimal cost exactly).

Additional suites (all accept `--engine-only` or `--base-url` where noted):

```bash
# Extra packs: 8 merge/edge cases and 45 judge-style cases (paraphrases, window edges,
# distractors, numeric/battery/profile edges, combinations)
python tests/run_public_cases.py --engine-only --cases tests/extra_cases.json
python tests/run_public_cases.py --engine-only --cases tests/judge_cases.json
python tests/run_public_cases.py --base-url http://localhost:8000 --cases tests/judge_cases.json

# HTTP robustness + response-contract suite (invalid input, unusual valid input,
# determinism, concurrency, latency bands, keep-alive after errors)
python tests/test_judge_robustness.py --base-url http://localhost:8000

# Engine-level checks
python tests/test_validator_rejects.py      # validator catches hand-broken plans
python tests/test_engine_soak.py            # 300 random + edge scenarios

# Slow/unresponsive LLM provider scenarios (no real key needed): each must stay < 30 s
python tests/timeout_repro.py hang
python tests/timeout_repro.py drip
```

## Docker fallback

- **Registry Reference:** `nfs996/iut-electrolyte-preli:v1`
- **Digest:** `sha256:74ea68c6f0677524d1d55e6407c33c8e0f6d9357dda84284e26e2a1e50691c43`

```bash
docker pull nfs996/iut-electrolyte-preli:v1
docker run -d --name gridwise-fallback -p 8000:8000 -e LLM_API_KEY=<your key> nfs996/iut-electrolyte-preli:v1
curl http://localhost:8000/health
# {"status":"ok"}
```

The image binds `0.0.0.0`, exposes port 8000 (override with `-e PORT=...`), and contains
**no baked-in credentials** — the key is supplied at `docker run` time.

## Error handling & security

- Malformed JSON or a structurally invalid body (wrong types, 0 or >3 notes, not exactly
  hours 0–23, negative/NaN/infinite numbers, inconsistent battery) → HTTP 400 with a short
  JSON error. Unexpected internal failures → controlled 500 (`{"error":"internal error"}`)
  returned from inside the endpoint, so stack traces never reach callers and the keep-alive
  connection stays usable for the next request.
- LLM/provider failures never take the service down: bounded retries (above), then a
  deterministic keyword fallback answers in the same schema.
- The API key is sent in a request header (never in URLs), log messages are scrubbed of the
  key defensively, `.env` is gitignored, and no secrets exist in the repo or image.

## Dependencies (credited)

FastAPI, uvicorn, pydantic (API + validation) · scipy/numpy (HiGHS linear programming) ·
httpx (async LLM calls) · Google Gemini API (operator-note interpretation). AI coding
assistants were used during development per the official rulebook; architecture and logic
are the team's own.

## Known limitations

- The keyword fallback recognizes common phrasings only (all 10 public samples and the 8
  extra cases; about 70% of the harder judge-style pack); it is an availability safety net,
  not the graded interpretation path (that is always the LLM when the provider is up).
- A window crossing midnight is interpreted within the single 24-hour schedule
  (hours 22, 23, 0, 1 → `[0,1,22,23]`); the problem statement does not specify this case.
- If a directive set were genuinely infeasible, the optimizer relaxes the most likely
  misread directive family and logs it rather than failing the request (organizer scoring
  scenarios are guaranteed feasible, so this path is not expected during judging).
- Grid export/sell-back is out of scope per the problem statement; surplus solar is curtailed.
