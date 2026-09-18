"""Adversarial checks: the replay validator must reject hand-broken plans.

A validator that never fails is worthless as a pre-return self-check, so every
rule in Plan section 4 gets a deliberately corrupted plan here.

    python tests/test_validator_rejects.py
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.engine_api import optimize, validate_plan          # noqa: E402
from app.schemas import Directives, HourPlan, ScenarioRequest  # noqa: E402
from run_public_cases import directives_from_interpretation  # noqa: E402

Breaker = Callable[[list[HourPlan]], list[HourPlan]]


def _find(plan: list[HourPlan], action: str) -> HourPlan:
    for row in plan:
        if row.battery_action == action:
            return row
    raise AssertionError(f"no {action} hour in the reference plan")


def break_balance(plan):
    plan[5].grid_kwh += 25.0
    return plan


def break_solar_cap(plan):
    # Claim more solar than the forecast allows, paying for it out of the grid draw
    # so the hour still balances and only the solar ceiling is violated.
    row = max((p for p in plan if p.grid_kwh > 1.0), key=lambda p: p.grid_kwh)
    shift = min(row.grid_kwh, 30.0)
    row.solar_used_kwh += shift
    row.grid_kwh -= shift
    return plan


def break_idle_amount(plan):
    row = _find(plan, "idle")
    row.battery_kwh = 12.0
    return plan


def break_rate_limit(plan):
    row = _find(plan, "charge")
    row.battery_kwh += 500.0
    row.grid_kwh += 500.0
    return plan


def break_state_transition(plan):
    for row in plan[7:]:
        row.battery_energy_after_kwh += 15.0
    return plan


def break_neutrality(plan):
    plan[23].battery_energy_after_kwh += 30.0
    return plan


BREAKERS: list[tuple[str, Breaker, Directives | None]] = [
    ("energy balance", break_balance, None),
    ("solar above effective", break_solar_cap, None),
    ("idle with battery_kwh", break_idle_amount, None),
    ("charge rate limit", break_rate_limit, None),
    ("battery state transition", break_state_transition, None),
    ("end-of-day neutrality", break_neutrality, None),
]


def scenario_breakers(scenario: ScenarioRequest,
                      plan: list[HourPlan]) -> list[tuple[str, ScenarioRequest]]:
    """Same untouched plan, judged against a battery it no longer fits. Isolates the
    envelope and rate rules without disturbing balance or the state transitions."""
    peak_store = max(p.battery_energy_after_kwh for p in plan)
    biggest_move = max(p.battery_kwh for p in plan)

    # Leave initial_energy_kwh alone: lowering it would break the very first
    # transition and neutrality as well, which is not what this case is testing.
    tight_cap = scenario.model_dump()
    tight_cap["battery"]["capacity_kwh"] = max(
        scenario.battery.initial_energy_kwh, peak_store - 5.0)

    slow = scenario.model_dump()
    slow["battery"]["max_charge_kwh_per_hour"] = max(0.0, biggest_move - 5.0)
    slow["battery"]["max_discharge_kwh_per_hour"] = max(0.0, biggest_move - 5.0)

    return [
        ("capacity ceiling", ScenarioRequest(**tight_cap)),
        ("charge/discharge rates", ScenarioRequest(**slow)),
    ]


def directive_breakers(plan: list[HourPlan]) -> list[tuple[str, Directives]]:
    """Same untouched plan, but judged against directives it cannot satisfy."""
    charge_hour = _find(plan, "charge").hour
    discharge_hour = _find(plan, "discharge").hour
    peak = max(plan, key=lambda p: p.grid_kwh)
    return [
        ("no-charge window", Directives(no_charge_hours={charge_hour})),
        ("no-discharge window", Directives(no_discharge_hours={discharge_hour})),
        ("grid cap", Directives(grid_cap={peak.hour: max(0.0, peak.grid_kwh - 20.0)})),
        ("reserve floor", Directives(
            min_reserve={p.hour: p.battery_energy_after_kwh + 25.0 for p in plan})),
        ("solar factor cut", Directives(
            solar_factor={p.hour: 0.0 for p in plan if p.solar_used_kwh > 0})),
    ]


def main() -> int:
    pack = json.loads((ROOT / "sample_cases.json").read_text(encoding="utf-8"))
    case = pack["cases"][0]
    scenario = ScenarioRequest(**case["input"])
    truth = directives_from_interpretation(
        case["expected_output"]["directive_interpretation"], scenario.battery.capacity_kwh
    )
    good = optimize(scenario, truth)

    results: list[tuple[str, bool, str]] = []

    clean = validate_plan(scenario, truth, good)
    results.append(("untouched plan accepted", not clean, "; ".join(clean[:2])))

    for name, breaker, _ in BREAKERS:
        broken = breaker(copy.deepcopy(good))
        errs = validate_plan(scenario, truth, broken)
        results.append((f"rejects: {name}", bool(errs), errs[0] if errs else "accepted!"))

    for name, directives in directive_breakers(good):
        errs = validate_plan(scenario, directives, copy.deepcopy(good))
        results.append((f"rejects: {name}", bool(errs), errs[0] if errs else "accepted!"))

    for name, tweaked in scenario_breakers(scenario, good):
        errs = validate_plan(tweaked, truth, copy.deepcopy(good))
        results.append((f"rejects: {name}", bool(errs), errs[0] if errs else "accepted!"))

    # Structural corruption must be caught too, not raise.
    short = copy.deepcopy(good)[:20]
    errs = validate_plan(scenario, truth, short)
    results.append(("rejects: only 20 hours", bool(errs), errs[0] if errs else "accepted!"))

    dupe = copy.deepcopy(good)
    dupe[4].hour = 5
    errs = validate_plan(scenario, truth, dupe)
    results.append(("rejects: duplicate hour", bool(errs), errs[0] if errs else "accepted!"))

    width = max(len(n) for n, _, _ in results)
    failed = 0
    print()
    for name, ok, detail in results:
        failed += 0 if ok else 1
        print(f"{name.ljust(width)}  {'PASS' if ok else 'FAIL'}  {detail[:88]}")
    print(f"\n{len(results) - failed}/{len(results)} validator checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
