"""Replay validator (Plan section 4).

Re-implements every check the judge is expected to run, driven by a `Directives`
object rather than by our own interpretation, so it is equally usable on our
response, on a hand-broken plan, and on the sample pack's reference schedules.

Returns a list of human-readable violations. Empty list = valid.
"""

from __future__ import annotations

import math

from app.schemas import TOL, Directives, HourPlan, ScenarioRequest

H = 24


def _finite(x: float) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def _gt(a: float, b: float) -> bool:
    """a is meaningfully greater than b."""
    return a > b + TOL


def validate_plan(
    scenario: ScenarioRequest,
    directives: Directives,
    plan: list[HourPlan],
) -> list[str]:
    """Replay check (Plan section 4). Empty list = valid; else human-readable violations."""
    errors: list[str] = []

    # 1. structural -----------------------------------------------------------
    if len(plan) != H:
        return [f"hourly_plan has {len(plan)} entries, expected {H}"]
    seen = [p.hour for p in plan]
    if sorted(seen) != list(range(H)):
        return [f"hourly_plan hours are not exactly 0..23 (got {sorted(seen)})"]

    rows = sorted(plan, key=lambda p: p.hour)
    hours = scenario.by_hour()
    bat = scenario.battery
    base_min = bat.minimum_energy_kwh

    for p in rows:
        for name, val in (
            ("grid_kwh", p.grid_kwh),
            ("solar_used_kwh", p.solar_used_kwh),
            ("battery_kwh", p.battery_kwh),
            ("battery_energy_after_kwh", p.battery_energy_after_kwh),
        ):
            if not _finite(val):
                errors.append(f"hour {p.hour}: {name} is not a finite number ({val!r})")
            elif val < -TOL:
                errors.append(f"hour {p.hour}: {name} is negative ({val})")
    if errors:
        return errors

    energy = bat.initial_energy_kwh
    for p in rows:
        h = p.hour
        src = hours[h]
        charge = p.battery_kwh if p.battery_action == "charge" else 0.0
        discharge = p.battery_kwh if p.battery_action == "discharge" else 0.0

        # 2. energy balance ---------------------------------------------------
        supply = p.grid_kwh + p.solar_used_kwh + discharge
        draw = src.demand_kwh + charge
        if abs(supply - draw) > TOL:
            errors.append(
                f"hour {h}: balance off by {supply - draw:+.4f} "
                f"(grid {p.grid_kwh} + solar {p.solar_used_kwh} + discharge {discharge} "
                f"!= demand {src.demand_kwh} + charge {charge})"
            )

        # 3. solar / battery physics ------------------------------------------
        factor = min(1.0, max(0.0, float(directives.solar_factor.get(h, 1.0))))
        eff_solar = src.solar_kwh * factor
        if _gt(p.solar_used_kwh, eff_solar):
            errors.append(
                f"hour {h}: solar_used_kwh {p.solar_used_kwh} exceeds effective solar "
                f"{eff_solar:.4f} (forecast {src.solar_kwh} x factor {factor})"
            )
        if p.battery_action == "idle" and abs(p.battery_kwh) > TOL:
            errors.append(f"hour {h}: battery_action is idle but battery_kwh is {p.battery_kwh}")
        if _gt(charge, bat.max_charge_kwh_per_hour):
            errors.append(
                f"hour {h}: charge {charge} exceeds max_charge_kwh_per_hour "
                f"{bat.max_charge_kwh_per_hour}"
            )
        if _gt(discharge, bat.max_discharge_kwh_per_hour):
            errors.append(
                f"hour {h}: discharge {discharge} exceeds max_discharge_kwh_per_hour "
                f"{bat.max_discharge_kwh_per_hour}"
            )

        energy = energy + charge - discharge
        if abs(p.battery_energy_after_kwh - energy) > TOL:
            errors.append(
                f"hour {h}: battery_energy_after_kwh {p.battery_energy_after_kwh} does not "
                f"follow from the previous state ({energy:.4f} expected)"
            )
        energy = p.battery_energy_after_kwh

        reserve = max(base_min, float(directives.min_reserve.get(h, base_min)))
        if energy < reserve - TOL:
            errors.append(
                f"hour {h}: stored energy {energy} is below the required reserve {reserve}"
            )
        if _gt(energy, bat.capacity_kwh):
            errors.append(
                f"hour {h}: stored energy {energy} exceeds capacity {bat.capacity_kwh}"
            )

        # 4. directive compliance ---------------------------------------------
        if h in directives.no_charge_hours and charge > TOL:
            errors.append(f"hour {h}: charged {charge} kWh inside a no-charge window")
        if h in directives.no_discharge_hours and discharge > TOL:
            errors.append(f"hour {h}: discharged {discharge} kWh inside a no-discharge window")
        cap = directives.grid_cap.get(h)
        if cap is not None and _gt(p.grid_kwh, float(cap)):
            errors.append(f"hour {h}: grid draw {p.grid_kwh} exceeds the cap {cap}")

    # 5. end-of-day neutrality ------------------------------------------------
    final = rows[-1].battery_energy_after_kwh
    if abs(final - bat.initial_energy_kwh) > TOL:
        errors.append(
            f"end of day: stored energy {final} does not return to the initial "
            f"{bat.initial_energy_kwh}"
        )

    return errors


def totals(scenario: ScenarioRequest, plan: list[HourPlan]) -> dict[str, float]:
    """Recompute the reported totals from the plan. Person B calls this from
    main.py; the harness uses it to check whatever a service reported."""
    hours = scenario.by_hour()
    rows = sorted(plan, key=lambda p: p.hour)
    total_grid = sum(p.grid_kwh for p in rows)
    total_cost = sum(p.grid_kwh * hours[p.hour].tariff_bdt_per_kwh for p in rows)
    return {
        "total_grid_kwh": round(total_grid, 4),
        "total_cost_bdt": round(total_cost, 4),
        "peak_grid_kwh": round(max((p.grid_kwh for p in rows), default=0.0), 4),
    }


def check_totals(scenario: ScenarioRequest, plan: list[HourPlan],
                 reported: dict[str, float]) -> list[str]:
    """Check 6: reported totals match the ones recomputed from hourly_plan."""
    errors: list[str] = []
    expected = totals(scenario, plan)
    for key, want in expected.items():
        got = reported.get(key)
        if got is None:
            errors.append(f"totals: {key} missing from the response")
        elif not _finite(got):
            errors.append(f"totals: {key} is not a finite number ({got!r})")
        elif abs(float(got) - want) > TOL:
            errors.append(f"totals: {key} reported {got}, recomputed {want}")
    return errors
