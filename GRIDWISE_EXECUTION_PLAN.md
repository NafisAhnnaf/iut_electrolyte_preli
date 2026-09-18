# GridWise LLM — Execution Plan (Agent Handoff)

**Event:** BUP CSE Fest 2026 Hackathon — Online Preliminary (7:00–11:00 PM, 4 hours)
**Goal:** Deploy a public HTTP API that (1) interprets 1–3 natural-language operator notes with an LLM, (2) validates the interpretation with deterministic guardrails, (3) solves a 24-hour energy-cost LP, and (4) returns both in the exact required JSON schema.

**Scoring (100 pts):** Interpretation 25 · Directive application & constraints 25 · Optimization 10 · API/schema 10 · Performance/reliability 10 · Deployment/Docker 10 · Docs/reproducibility 10. Video = tie-break only.
**Hard rule:** an LLM MUST be in the operator-note → directive path (not just for `plan_summary`), or the team is disqualified from the shortlist.

---

## 0. Ground rules for the agent

- Create a **new private GitHub repo** now (it must be created after question reveal). Name: `gridwise-llm-preli`. Never commit secrets — API key goes in env var only. Add `.env` to `.gitignore`.
- Stack: **Python 3.11+, FastAPI + uvicorn, scipy (HiGHS linprog), pydantic, httpx**. No heavy frameworks.
- Reference files (already available): Problem Statement PDF, Participant Guide PDF, `BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json` (10 worked cases — use as the local test suite).
- LLM provider: read from env. Default plan: `LLM_PROVIDER` ∈ {`openai`,`gemini`,`groq`}; env vars `LLM_API_KEY`, `LLM_MODEL`. Use temperature 0, JSON-mode/structured output. **One LLM call per request covering all notes** (latency: p95 ≤ 5s earns full points; hard timeout 30s per request).
- Build order matters: **optimizer + validator first (deterministic, testable), LLM layer second, deployment third, docs fourth.** Do not gold-plate; correctness > cost.

## 1. Repository layout

```
gridwise-llm-preli/
├── app/
│   ├── main.py            # FastAPI app: /health, /optimize-energy, error handlers
│   ├── schemas.py         # pydantic request/response models (exact contract)
│   ├── interpreter.py     # LLM call + prompt + raw-output parsing
│   ├── guardrails.py      # deterministic validation + normalization of LLM output
│   ├── optimizer.py       # LP formulation + rounding/recompute
│   ├── validator.py       # full replay checker (same rules as judge)
│   └── config.py          # env vars: LLM_PROVIDER, LLM_API_KEY, LLM_MODEL, PORT
├── tests/
│   └── run_public_cases.py  # POSTs all 10 public cases, checks everything locally
├── sample_cases.json      # copy of the public case pack
├── Dockerfile
├── requirements.txt
├── README.md
└── .gitignore             # includes .env
```

## 2. Exact API contract (do not deviate)

### GET /health
`200` → `{"status": "ok"}`

### POST /optimize-energy — request
```json
{
  "scenario_id": "string",
  "operator_notes": ["1–3 non-empty strings"],
  "hours": [ {"hour": 0..23, "demand_kwh": num, "solar_kwh": num, "tariff_bdt_per_kwh": num} × 24 ],
  "battery": {
    "capacity_kwh": num, "initial_energy_kwh": num, "minimum_energy_kwh": num,
    "max_charge_kwh_per_hour": num, "max_discharge_kwh_per_hour": num
  }
}
```
Malformed JSON / structurally invalid → **400** (controlled JSON error body, no stack traces). Optionally 422 for well-formed-but-semantically-invalid. Never 500 on bad input.

### Response
```json
{
  "scenario_id": "<echo request>",
  "directive_interpretation": [
    {
      "note_index": 0,
      "applies": true|false,
      "directive_type": "solar_reduction|minimum_battery_reserve|no_charge_window|no_discharge_window|max_grid_window|no_op",
      "structured_adjustment": { ... } | null,
      "explanation": "short string"
    }
  ],
  "hourly_plan": [
    {"hour": h, "grid_kwh": num, "solar_used_kwh": num,
     "battery_action": "charge|discharge|idle", "battery_kwh": num,
     "battery_energy_after_kwh": num} × 24
  ],
  "total_grid_kwh": num,
  "total_cost_bdt": num,
  "peak_grid_kwh": num,
  "plan_summary": "short string"
}
```

`structured_adjustment` shapes (exact):
| directive_type | shape |
|---|---|
| solar_reduction | `{"hours":[...], "factor": number}` (factor = **fraction remaining**, 0–1) |
| minimum_battery_reserve | `{"hours":[...], "minimum_energy_kwh": number}` |
| no_charge_window | `{"hours":[...]}` |
| no_discharge_window | `{"hours":[...]}` |
| max_grid_window | `{"hours":[...], "max_grid_kwh": number}` |
| no_op | `null`, and `applies` must be `false` |

Rules: one entry per note, in `note_index` order 0..N-1; every non-no_op has `applies=true`; hours arrays are unique ints 0–23 **ascending**; windows are start-inclusive, end-exclusive ("1 PM to 3 PM" → `[13,14]`; "noon until 2 PM" → `[12,13]`; "from 6 PM until 9 PM" → `[18,19,20]`).

## 3. Optimizer (build first) — `optimizer.py`

LP via `scipy.optimize.linprog(method="highs")`. Per hour h (0..23), variables:
- `g[h]` grid import ≥ 0
- `s[h]` solar used, `0 ≤ s[h] ≤ eff_solar[h]` where `eff_solar[h] = solar_kwh[h] * factor` if h in a solar_reduction window else `solar_kwh[h]`
- `c[h]` charge, `0 ≤ c[h] ≤ max_charge` (0 if h in no_charge_window)
- `d[h]` discharge, `0 ≤ d[h] ≤ max_discharge` (0 if h in no_discharge_window)

Constraints:
- **Balance (equality):** `g[h] + s[h] + d[h] = demand[h] + c[h]` for every h.
- **Battery state:** `E[h] = initial + Σ_{k≤h}(c[k] − d[k])`; enforce `res[h] ≤ E[h] ≤ capacity` where `res[h] = max(minimum_energy_kwh, directive reserve if h in reserve window)`.
- **Neutrality:** `E[23] = initial_energy_kwh` (equality).
- **Grid cap:** `g[h] ≤ max_grid_kwh` for h in max_grid_window.
- **Objective:** minimize `Σ g[h] * tariff[h]`.

Notes:
- Simultaneous charge+discharge is never optimal with positive tariffs, so a c/d split is safe; still, after solving, if both are >ε in an hour, net them.
- **Rounding procedure (critical):** round `c,d,s` to 4 decimals, recompute `E[h]` cumulatively from rounded flows, then set `g[h] = demand[h] + c[h] − d[h] − s[h]` (clip tiny negatives to 0 and reduce s accordingly) so the balance holds *exactly* on the returned numbers. Derive `battery_action` (`charge` if c>ε, `discharge` if d>ε, else `idle`, with `battery_kwh=0` when idle). Recompute `total_grid_kwh`, `total_cost_bdt` (= Σ g·tariff), `peak_grid_kwh` (= max g) **from the final hourly_plan**. Tolerance everywhere: 0.01.
- If LP infeasible (shouldn't happen on judge cases): drop soft assumptions? No — judge scenarios are feasible; as a safety net, retry once with directives from failed guardrails removed, and if still infeasible return a controlled 422/500-safe JSON (never crash).

## 4. Replay validator — `validator.py`

Re-implements every judge check; run it on our own response **before returning** and in tests:
1. 24 unique hours 0–23 in both `hours` and `hourly_plan`; all values finite, non-negative.
2. Per hour: `grid + solar_used + discharge == demand + charge` (±0.01).
3. `solar_used ≤ eff_solar` (±0.01), battery transitions correct, `res[h] ≤ E_after ≤ capacity`, rate limits, `battery_kwh == 0` when idle.
4. Directive checks: charge=0 in no_charge hours, discharge=0 in no_discharge hours, `grid ≤ cap` in capped hours, `E_after ≥ reserve` in reserve hours.
5. `E_after[23] == initial` (±0.01).
6. Reported totals match recomputed (±0.01).

If the self-check fails, fall back to a **degraded-but-valid plan** (see §7) rather than returning an invalid one.

## 5. LLM interpreter — `interpreter.py`

One call, temperature 0, JSON output. The LLM extracts *intermediate* fields; deterministic code does all arithmetic and window math.

Prompt skeleton (system):
```
You convert campus energy operator notes into structured directives.
Directive types: solar_reduction, minimum_battery_reserve, no_charge_window,
no_discharge_window, max_grid_window, no_op.
For EACH note return JSON:
{"note_index": int,
 "directive_type": one of the types,
 "start_hour": int 0-23 or null,   // window start, 24h clock, inclusive
 "end_hour": int 1-24 or null,     // window end, EXCLUSIVE ("until 9 PM" -> 21)
 "value": number or null,          // the numeric quantity mentioned
 "value_kind": "fraction_remaining" | "fraction_reduced" | "percent_remaining" |
               "percent_reduced" | "kwh" | "percent_of_capacity" | null}
Rules:
- Notes about future days, other departments, menus, deadlines, or anything not
  affecting TODAY's 24-hour electricity schedule are no_op (all other fields null).
- "drops to 20%" => percent_remaining 20. "80% reduction" => percent_reduced 80.
  "one-fifth of normal output" => fraction_remaining 0.2.
- "keep at least 120 kWh" => kwh 120. "keep 50% of capacity" => percent_of_capacity 50.
- Times: noon=12, midnight=0, "2 AM"=2, "6 PM"=18, "13:00"=13.
Return {"interpretations":[ ... one per note, in order ... ]}. JSON only.
```
Include 3–4 few-shot examples covering: solar reduction with %, reserve as % of capacity, maintenance no-charge window, and a distractor no_op. User message: the numbered notes plus `battery.capacity_kwh` (needed context for nothing — capacity math is done in code, but harmless to include note count only). Keep the prompt compact for latency.

Retry policy: 1 retry on non-JSON/HTTP error with short backoff. On total failure, fall back to a minimal deterministic keyword classifier **only as an emergency degraded mode** (documented in README as fallback; primary path is the LLM as required).

## 6. Guardrails — `guardrails.py`

Deterministic conversion of LLM output → final `directive_interpretation` + optimizer constraints:
- Exactly one entry per note; sort/verify note_index 0..N-1, no dupes.
- `directive_type` must be in the allowed enum, else coerce that note to safe handling (retry once; if still bad → treat as no_op and log — never invent a type, never crash).
- Window: `hours = list(range(start_hour, end_hour))`, require ints, 0 ≤ start < end ≤ 24, result non-empty, unique, ascending.
- Numeric normalization:
  - solar_reduction factor: `percent_remaining p → p/100`; `percent_reduced p → 1 − p/100`; `fraction_reduced f → 1 − f`; `fraction_remaining f → f`. Clamp check 0 ≤ factor ≤ 1 else reject→retry.
  - minimum_battery_reserve: `kwh v → v`; `percent_of_capacity p → p/100 * battery.capacity_kwh`. Require 0 ≤ reserve ≤ capacity.
  - max_grid_kwh: finite, ≥ 0.
- Build final entries: no_op → `applies=false, structured_adjustment=null`; others → `applies=true` with the exact shape from §2. Write short `explanation` strings in code (free text isn't byte-matched).
- Merge multiple directives of the same type across notes correctly (e.g., two reserve windows: per-hour max of reserves; two grid caps on same hour: min of caps; unions of no-charge/no-discharge hours; solar factors: if two reductions hit the same hour, multiply? — judge scenarios won't contradict; apply per-hour with later-note-wins avoided by taking the *more restrictive* value: min factor, max reserve, min cap).

## 7. FastAPI app — `main.py`

- Pydantic-validated request; validation error → 400 JSON `{"error": "..."} `; generic exception handler → 500 JSON with generic message, log details server-side only. Never echo secrets or stack traces.
- Flow: parse → interpret (LLM) → guardrails → optimize → self-validate (replay) → respond.
- Degraded modes (never crash, always return schema-valid JSON): LLM fails twice → emergency keyword classifier; a single note unguardrailable → no_op that note; LP infeasible after directives → re-solve without soft interpretation? (judge cases feasible; just log and return best-effort).
- `plan_summary`: one templated sentence built in code from the applied directives (fast, deterministic).
- Concurrency: uvicorn default workers=1 is fine; make the LLM call with `httpx.AsyncClient` (timeout 20s) so repeated judge requests don't block.

## 8. Local test harness — `tests/run_public_cases.py`

For each of the 10 public cases:
1. POST `case.input` to the running service.
2. Assert response schema (all required fields, enums).
3. Compare `directive_interpretation` semantics against `expected_output.directive_interpretation` (type, applies, hours, numerics ±0.01 — NOT explanation text).
4. Run the §4 replay validator on our `hourly_plan` **using the expected ground-truth directives**.
5. Compare `total_cost_bdt` to the reference (ours must be ≤ reference + 0.01; equal-or-better is expected from a true LP optimum).
Print a PASS/FAIL table. **All 10 must pass before deployment.**

Also test: malformed JSON → 400; empty notes array → 400; garbage note text → schema-valid response with no_op; paraphrase set ("PV production will drop to about 20% between 13:00 and 15:00", "Panel washing from one until three will leave roughly one-fifth of normal solar output", "Expect an 80% reduction in rooftop solar during the 1-3 PM maintenance window") → identical directive `{hours:[13,14], factor:0.2}`.

## 9. Docker & deployment

`Dockerfile`: `python:3.11-slim`, copy app, `pip install -r requirements.txt`, `EXPOSE 8000`, `CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}`. **Bind 0.0.0.0, no baked-in secrets.**
- Build, run locally with `-e LLM_API_KEY=...`, curl /health + one sample case through the container.
- Push to Docker Hub or GHCR with an exact tag (e.g., `:v1`); record the digest. Image must stay pullable through judging.
- Deploy to Render / Railway / Fly.io (whichever the team has ready; free tier is fine but **disable sleep/spin-down** or keep it warm — judge hits must not time out). Set env vars in the platform dashboard.
- From a network outside dev (phone hotspot ok): `curl BASE/health` and POST 2–3 sample cases. Record the public base URL.

## 10. README.md (10 pts — write it fully)

Must be self-contained; judges reproduce from a clean machine. Sections:
1. What it does (2 lines) + architecture: **LLM → deterministic guardrails → LP optimizer → replay validator** (small ASCII diagram).
2. Model/provider used (exact model id) and the LLM's role.
3. Environment variables (**names only**, no values): `LLM_PROVIDER`, `LLM_API_KEY`, `LLM_MODEL`, `PORT`.
4. Local quickstart (copy-paste): clone → `pip install -r requirements.txt` → export env vars → `uvicorn app.main:app --port 8000` → `curl localhost:8000/health` → one `curl -X POST` with a real sample body + expected-shape response.
5. Public-sample test: `python tests/run_public_cases.py --base-url http://localhost:8000` and what PASS output looks like.
6. Docker fallback: exact `docker pull <ref>` and `docker run -p 8000:8000 -e LLM_API_KEY=... <ref>` commands.
7. Guardrails list, optimizer/solver (scipy HiGHS LP), dependencies credited (FastAPI, scipy, pydantic, httpx, provider SDK), known limitations, secret-handling note.

## 11. Video (3 min, tie-break only — do last)

Screen recording: 30s problem, 60s architecture diagram walkthrough (LLM → guardrails → optimizer → replay), 60s demo (health curl, one sample POST, test harness green), 30s repo/README/Docker tour. Upload MP4 or accessible link.

## 12. Timeline (4-hour window)

| Time | Milestone | Definition of done |
|---|---|---|
| 0:00–0:15 | Repo (private), scaffold, /health, schemas.py | /health 200 locally |
| 0:15–1:15 | optimizer.py + validator.py | Fed with *expected* directives from the sample pack, all 10 plans valid and cost ≤ reference |
| 1:15–2:00 | interpreter.py + guardrails.py + full pipeline | All 10 public cases pass end-to-end incl. paraphrase tests |
| 2:00–2:40 | Docker build/push + cloud deploy + external tests | Public URL passes /health and 3 sample cases from outside |
| 2:40–3:20 | README + robustness (400s, LLM failure fallback, no stack traces) | Clean-env quickstart verified; malformed-input tests pass |
| 3:20–3:50 | Video + final checklist | All checklist boxes ticked |
| 3:50–4:00 | Submit; flip repo public after deadline | — |

If behind schedule: cut video polish first, never cut the replay validator or public-case pass.

## 13. Final checklist

- [ ] /health returns `{"status":"ok"}` externally
- [ ] All 10 public cases pass (interpretation semantics + replay + cost ≤ reference)
- [ ] Paraphrase trio resolves identically; distractor notes → no_op
- [ ] no_op: `applies=false` + `null`; all others `applies=true` + exact shape; entries in note_index order
- [ ] Hours ascending unique 0–23; end-exclusive windows; factor = fraction remaining; % reserve × capacity
- [ ] Totals recomputed from hourly_plan; balance/battery/neutrality hold within 0.01
- [ ] Malformed JSON → 400; LLM outage → controlled fallback, no crash, no 500 storm
- [ ] p95 latency ≤ 5s (single batched LLM call); per-request < 30s
- [ ] No secrets in repo, image, logs, or responses; `.env` gitignored
- [ ] Docker image pushed, pullable, documented run command verified
- [ ] README quickstart works on a clean environment
- [ ] Repo created after reveal, private now, public after deadline; video ≤ 3:00 accessible
