# GridWise — Two-Person Work Split

Companion to **GRIDWISE_EXECUTION_PLAN.md** (the plan stays canonical for all specs, schemas, and rules; section numbers below refer to it). This document only divides the work and fixes the merge contract.

---

## 0. The merge contract — agree on this in the first 10 minutes, then never change it

The two halves meet at exactly **two frozen interfaces**. Write these two files together at 0:00–0:10, commit them to `main`, and treat them as read-only afterward. Everything else is disjoint files, so merges are trivial.

### Frozen file 1: `app/schemas.py`
All pydantic models for the request and response, exactly as in Plan §2, **plus** the internal handoff type:

```python
# The ONLY object that crosses from Person B's side to Person A's side.
@dataclass
class Directives:
    solar_factor: dict[int, float]      # hour -> factor (fraction remaining); absent hour = 1.0
    min_reserve: dict[int, float]       # hour -> reserve kWh (already the max of base + directives)
    no_charge_hours: set[int]
    no_discharge_hours: set[int]
    grid_cap: dict[int, float]          # hour -> max grid kWh (already the min if multiple)
```

### Frozen file 2: `app/engine_api.py`
Two function signatures — Person A implements them, Person B calls them, neither changes them:

```python
def optimize(scenario: ScenarioRequest, directives: Directives) -> list[HourPlan]:
    """Returns 24 schema-ready hourly_plan entries. Raises InfeasibleError only."""

def validate_plan(scenario: ScenarioRequest, directives: Directives,
                  plan: list[HourPlan]) -> list[str]:
    """Replay check (Plan §4). Empty list = valid; else human-readable violations."""
```

Merge rules:
- Person A owns `optimizer.py`, `validator.py`, `tests/run_public_cases.py` — Person B never edits them.
- Person B owns `interpreter.py`, `guardrails.py`, `main.py`, `config.py`, `Dockerfile`, `README.md` — Person A never edits them.
- Both push directly to `main` frequently (no long-lived branches — disjoint files can't conflict). If either genuinely needs a change to a frozen file, it's a 2-minute pair decision, announced in chat, one commit.
- Totals (`total_grid_kwh`, `total_cost_bdt`, `peak_grid_kwh`) and `plan_summary` are computed in **B's** `main.py` from the plan A returns (trivial sums — keeps A's API minimal).

---

## Person A — "Engine": optimizer, validator, test harness

**You own correctness of the math. You need no LLM key and no network.** Plan sections: §3, §4, §8.

| When | Task |
|---|---|
| 0:00–0:10 | With B: create repo, write the two frozen files, scaffold layout (Plan §1). |
| 0:10–1:10 | `optimizer.py` per Plan §3: LP (scipy HiGHS), variables g/s/c/d, balance equality, battery state with per-hour reserve from `Directives.min_reserve`, neutrality, grid caps, directive-forced zero bounds. Then the **rounding-and-recompute** procedure so the returned numbers balance exactly (±0.01). |
| 1:10–1:40 | `validator.py` per Plan §4 — every judge check, driven by a `Directives` object. |
| 1:40–2:10 | `tests/run_public_cases.py` per Plan §8 — **two modes**: `--engine-only` (build `Directives` directly from each case's `expected_output.directive_interpretation`, call `optimize` + `validate_plan` in-process, compare cost ≤ reference) and `--base-url` (full HTTP end-to-end, also checks B's interpretation semantics vs expected). Get engine-only mode green on all 10 cases — this unblocks nothing of B's and proves your half independently. |
| 2:10–3:20 | Support: run the full `--base-url` suite against B's deployed service, fix engine bugs, add the malformed-input and paraphrase tests (Plan §8 last paragraph — the HTTP assertions live in your harness even though the behavior is B's code). |
| 3:20–3:50 | **Record the 3-minute video** (Plan §11) while B finishes the README — you know the architecture and your demo is the green test table. |

**Definition of done:** engine-only mode passes 10/10 with cost ≤ reference +0.01; validator rejects a hand-broken plan; full-suite failures clearly attributed (interpretation vs plan) so B can act on them.

---

## Person B — "Service": LLM, guardrails, API, deploy, docs

**You own everything that touches language, HTTP, and the outside world.** Plan sections: §5, §6, §7, §9, §10.

| When | Task |
|---|---|
| 0:00–0:10 | With A: repo + frozen files. Then immediately: `config.py`, `main.py` skeleton with `/health` → deploy this stub to the platform **now** (Render/Railway/Fly) so the deployment pipeline is proven early, not at hour 3. |
| 0:10–1:00 | `interpreter.py` per Plan §5: one batched LLM call, temp 0, JSON mode, the intermediate-fields prompt with few-shots, 1 retry, emergency keyword fallback. Test it standalone against the sample-pack notes + the paraphrase trio (you don't need A's code for this — just compare your parsed fields to `expected_output.directive_interpretation`). |
| 1:00–1:45 | `guardrails.py` per Plan §6: enum/window/numeric normalization, %-of-capacity reserve math, merge rules (min factor, max reserve, min cap, union windows). Output = the response's `directive_interpretation` list **and** a `Directives` object for A's engine. |
| 1:45–2:15 | Wire `main.py` full flow (Plan §7): parse → interpret → guardrails → `optimize()` → `validate_plan()` self-check → totals + `plan_summary` → respond. 400 on malformed input, controlled degraded modes, no stack traces. |
| 2:15–2:50 | Docker (Plan §9): build, run with env vars, push tagged image to Hub/GHCR, redeploy real service, set platform env vars, disable spin-down. Test /health + 3 cases from an outside network while A runs the full suite. |
| 2:50–3:35 | `README.md` — the full Plan §10 checklist, then verify the quickstart in a clean venv/container. |
| 3:35–4:00 | With A: final checklist (Plan §13), submit endpoint + repo + image ref + video; repo public after deadline. |

**Definition of done:** deployed URL passes the full 10-case suite from outside; paraphrase trio → identical directive; LLM outage path returns valid JSON; no secrets anywhere; README quickstart reproduced clean.

---

## Why this merges cleanly

- **File ownership is disjoint** — after the first commit, no file is edited by both people.
- **One typed seam** (`Directives` + two function signatures) instead of "we'll integrate later."
- **Both halves are independently testable**: A against expected interpretations (engine-only mode), B against expected `directive_interpretation` (parser-only), before ever combining.
- **Deployment starts at minute 10** with a stub, so the riskiest external dependency is de-risked first.
- Sync points: minute 10 (freeze interfaces), ~1:45 (first end-to-end run), ~2:50 (full suite on prod), 3:35 (submission).

**Fallback if one person stalls:** the seam makes the halves swappable — either person can stub the other side (A can hand-build `Directives` from expected outputs; B can return a naive greedy plan) to keep testing, but a stub never ships: the checklist in Plan §13 gates submission.
