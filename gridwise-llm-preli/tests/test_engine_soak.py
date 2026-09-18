"""Randomised soak + edge cases for the engine.

The public pack is only ten scenarios and they all share a shape. Hidden judge
cases will not, so this hammers optimize() with randomly generated batteries,
tariffs, solar curves and directive combinations and insists that every plan it
produces passes the replay validator.

    python tests/test_engine_soak.py [--n 300] [--seed 7]
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.engine_api import validate_plan                     # noqa: E402
from app.optimizer import optimize_detailed                  # noqa: E402
from app.schemas import Directives, InfeasibleError, ScenarioRequest  # noqa: E402
from app.validator import totals                            # noqa: E402


def random_scenario(rng: random.Random) -> ScenarioRequest:
    capacity = rng.choice([80, 150, 220, 400, 1000])
    minimum = round(capacity * rng.uniform(0.0, 0.3), 2)
    initial = round(rng.uniform(minimum, capacity), 2)
    peak_sun = rng.choice([0.0, 30.0, 120.0, 300.0])
    hours = []
    for h in range(24):
        daylight = max(0.0, 1.0 - abs(h - 12) / 6.0)
        hours.append({
            "hour": h,
            "demand_kwh": round(rng.uniform(0.0, 250.0), 2),
            "solar_kwh": round(peak_sun * daylight * rng.uniform(0.6, 1.0), 2),
            "tariff_bdt_per_kwh": round(rng.uniform(0.0, 40.0), 2),
        })
    return ScenarioRequest(
        scenario_id=f"SOAK-{rng.randrange(10**6):06d}",
        operator_notes=["synthetic"],
        hours=hours,
        battery={
            "capacity_kwh": capacity,
            "initial_energy_kwh": initial,
            "minimum_energy_kwh": minimum,
            "max_charge_kwh_per_hour": round(rng.uniform(0.0, capacity / 2), 2),
            "max_discharge_kwh_per_hour": round(rng.uniform(0.0, capacity / 2), 2),
        },
    )


def random_window(rng: random.Random) -> list[int]:
    start = rng.randrange(0, 23)
    end = rng.randrange(start + 1, 25)
    return list(range(start, end))


def random_directives(rng: random.Random, scenario: ScenarioRequest) -> Directives:
    d = Directives()
    bat = scenario.battery
    if rng.random() < 0.5:
        factor = rng.choice([0.0, 0.2, 0.25, 0.5, 0.8])
        for h in random_window(rng):
            d.solar_factor[h] = factor
    if rng.random() < 0.5:
        reserve = round(rng.uniform(bat.minimum_energy_kwh, bat.capacity_kwh), 2)
        for h in random_window(rng):
            d.min_reserve[h] = reserve
    if rng.random() < 0.4:
        d.no_charge_hours.update(random_window(rng))
    if rng.random() < 0.4:
        d.no_discharge_hours.update(random_window(rng))
    if rng.random() < 0.4:
        cap = round(rng.uniform(0.0, 300.0), 2)
        for h in random_window(rng):
            d.grid_cap[h] = cap
    return d


def edge_cases() -> list[tuple[str, ScenarioRequest, Directives]]:
    """Degenerate shapes that a random generator rarely produces."""
    flat = [{"hour": h, "demand_kwh": 100.0, "solar_kwh": 0.0,
             "tariff_bdt_per_kwh": 10.0} for h in range(24)]

    no_battery = ScenarioRequest(
        scenario_id="EDGE-dead-battery", operator_notes=["x"], hours=flat,
        battery={"capacity_kwh": 100.0, "initial_energy_kwh": 100.0,
                 "minimum_energy_kwh": 100.0, "max_charge_kwh_per_hour": 0.0,
                 "max_discharge_kwh_per_hour": 0.0})

    zero_demand = ScenarioRequest(
        scenario_id="EDGE-zero-demand", operator_notes=["x"],
        hours=[{"hour": h, "demand_kwh": 0.0, "solar_kwh": 50.0,
                "tariff_bdt_per_kwh": 5.0} for h in range(24)],
        battery={"capacity_kwh": 200.0, "initial_energy_kwh": 50.0,
                 "minimum_energy_kwh": 0.0, "max_charge_kwh_per_hour": 40.0,
                 "max_discharge_kwh_per_hour": 40.0})

    free_grid = ScenarioRequest(
        scenario_id="EDGE-zero-tariff", operator_notes=["x"],
        hours=[{"hour": h, "demand_kwh": 120.0, "solar_kwh": 10.0,
                "tariff_bdt_per_kwh": 0.0} for h in range(24)],
        battery={"capacity_kwh": 200.0, "initial_energy_kwh": 100.0,
                 "minimum_energy_kwh": 20.0, "max_charge_kwh_per_hour": 50.0,
                 "max_discharge_kwh_per_hour": 50.0})

    return [
        ("battery frozen at its floor", no_battery, Directives()),
        ("no demand all day", zero_demand, Directives()),
        ("tariff is zero everywhere", free_grid, Directives()),
        ("reserve at full capacity", no_battery,
         Directives(min_reserve={h: 100.0 for h in range(24)})),
        ("solar wiped out entirely", zero_demand,
         Directives(solar_factor={h: 0.0 for h in range(24)})),
        ("no charging or discharging at all", free_grid,
         Directives(no_charge_hours=set(range(24)), no_discharge_hours=set(range(24)))),
        ("grid capped below demand", free_grid,
         Directives(grid_cap={h: 0.0 for h in range(24)})),
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    failures: list[str] = []
    infeasible = 0
    relaxed = 0

    for name, scenario, directives in edge_cases():
        try:
            plan, enforced, dropped = optimize_detailed(scenario, directives)
        except InfeasibleError:
            infeasible += 1
            print(f"edge  {name}: InfeasibleError (allowed)")
            continue
        # Judge the plan against what the engine could actually enforce; a dropped
        # family means the directive set itself had no solution.
        errs = validate_plan(scenario, enforced, plan)
        if errs:
            failures.append(f"edge '{name}': {errs[0]}")
        note = "OK" if not errs else "FAIL " + errs[0]
        if dropped:
            relaxed += 1
            note += f" (infeasible, dropped {dropped})"
        print(f"edge  {name}: {note}")

    for i in range(args.n):
        scenario = random_scenario(rng)
        directives = random_directives(rng, scenario)
        try:
            plan, enforced, dropped = optimize_detailed(scenario, directives)
        except InfeasibleError:
            infeasible += 1
            continue
        except Exception as exc:  # noqa: BLE001 - any other exception is a bug
            failures.append(f"random #{i} {scenario.scenario_id}: "
                            f"unexpected {type(exc).__name__}: {exc}")
            continue
        if dropped:
            relaxed += 1
        errs = validate_plan(scenario, enforced, plan)
        if errs:
            failures.append(f"random #{i} {scenario.scenario_id}: {errs[0]}"
                            + (f" [after dropping {dropped}]" if dropped else ""))
            continue
        t = totals(scenario, plan)
        if t["total_cost_bdt"] < -0.01 or t["peak_grid_kwh"] < -0.01:
            failures.append(f"random #{i}: negative totals {t}")

    print(f"\n{args.n} random scenarios + {len(edge_cases())} edge cases")
    print(f"directive sets with no solution (safety net dropped a family): {relaxed}")
    print(f"infeasible even after the full relaxation ladder: {infeasible}")
    if failures:
        print(f"FAILURES: {len(failures)}")
        for f in failures[:15]:
            print(f"  {f}")
        return 1
    print("all plans passed the replay validator")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
