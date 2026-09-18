"""One-off generator for tests/extra_cases.json.

Each case's `directive_interpretation` is authored by hand (that is the part a
human must get right per the spec). Its `hourly_plan` / totals are then
produced by calling the project's own optimize() + validate_plan() -- the same
engine already proven to reproduce the 10 official reference costs exactly
(see run_public_cases.py --engine-only) -- so the ground truth is internally
consistent and mechanically re-derivable, not hand-computed.

Run once to (re)write tests/extra_cases.json:

    python tests/generate_extra_cases.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.engine_api import optimize, validate_plan  # noqa: E402
from app.schemas import Directives, ScenarioRequest  # noqa: E402
from app.validator import totals  # noqa: E402
from run_public_cases import directives_from_interpretation  # noqa: E402

BASE_HOURS = [
    {"hour": 0, "demand_kwh": 90, "solar_kwh": 0, "tariff_bdt_per_kwh": 6},
    {"hour": 1, "demand_kwh": 85, "solar_kwh": 0, "tariff_bdt_per_kwh": 6},
    {"hour": 2, "demand_kwh": 80, "solar_kwh": 0, "tariff_bdt_per_kwh": 5},
    {"hour": 3, "demand_kwh": 80, "solar_kwh": 0, "tariff_bdt_per_kwh": 5},
    {"hour": 4, "demand_kwh": 85, "solar_kwh": 0, "tariff_bdt_per_kwh": 5},
    {"hour": 5, "demand_kwh": 95, "solar_kwh": 0, "tariff_bdt_per_kwh": 6},
    {"hour": 6, "demand_kwh": 110, "solar_kwh": 5, "tariff_bdt_per_kwh": 8},
    {"hour": 7, "demand_kwh": 130, "solar_kwh": 20, "tariff_bdt_per_kwh": 10},
    {"hour": 8, "demand_kwh": 150, "solar_kwh": 50, "tariff_bdt_per_kwh": 12},
    {"hour": 9, "demand_kwh": 165, "solar_kwh": 90, "tariff_bdt_per_kwh": 14},
    {"hour": 10, "demand_kwh": 175, "solar_kwh": 130, "tariff_bdt_per_kwh": 16},
    {"hour": 11, "demand_kwh": 180, "solar_kwh": 160, "tariff_bdt_per_kwh": 16},
    {"hour": 12, "demand_kwh": 185, "solar_kwh": 180, "tariff_bdt_per_kwh": 15},
    {"hour": 13, "demand_kwh": 180, "solar_kwh": 170, "tariff_bdt_per_kwh": 14},
    {"hour": 14, "demand_kwh": 170, "solar_kwh": 140, "tariff_bdt_per_kwh": 13},
    {"hour": 15, "demand_kwh": 165, "solar_kwh": 90, "tariff_bdt_per_kwh": 14},
    {"hour": 16, "demand_kwh": 170, "solar_kwh": 45, "tariff_bdt_per_kwh": 18},
    {"hour": 17, "demand_kwh": 185, "solar_kwh": 10, "tariff_bdt_per_kwh": 22},
    {"hour": 18, "demand_kwh": 205, "solar_kwh": 0, "tariff_bdt_per_kwh": 28},
    {"hour": 19, "demand_kwh": 215, "solar_kwh": 0, "tariff_bdt_per_kwh": 30},
    {"hour": 20, "demand_kwh": 205, "solar_kwh": 0, "tariff_bdt_per_kwh": 26},
    {"hour": 21, "demand_kwh": 175, "solar_kwh": 0, "tariff_bdt_per_kwh": 18},
    {"hour": 22, "demand_kwh": 135, "solar_kwh": 0, "tariff_bdt_per_kwh": 10},
    {"hour": 23, "demand_kwh": 105, "solar_kwh": 0, "tariff_bdt_per_kwh": 7},
]
BASE_BATTERY = {
    "capacity_kwh": 220, "initial_energy_kwh": 110, "minimum_energy_kwh": 40,
    "max_charge_kwh_per_hour": 50, "max_discharge_kwh_per_hour": 50,
}


def no_op(note_index: int, explanation: str) -> dict[str, Any]:
    return {
        "note_index": note_index, "applies": False, "directive_type": "no_op",
        "structured_adjustment": None, "explanation": explanation,
    }


def entry(note_index: int, directive_type: str, adjustment: dict[str, Any],
          explanation: str) -> dict[str, Any]:
    return {
        "note_index": note_index, "applies": True, "directive_type": directive_type,
        "structured_adjustment": adjustment, "explanation": explanation,
    }


CASES: list[dict[str, Any]] = [
    {
        "id": "EXTRA-01",
        "label": "Overlapping solar_reduction: more-restrictive factor wins",
        "rationale": (
            "Two notes cut solar in overlapping hours with different severity. Guardrails "
            "must report each note's own window/factor untouched (one entry per note) while "
            "the ENGINE applies the more restrictive (lower) factor per hour: 0.5 at hour 12, "
            "min(0.5,0.2)=0.2 at hour 13 (both windows), 0.2 at hour 14."
        ),
        "notes": [
            "Panel washing from noon until 2 PM will leave usable solar at about 50% of forecast.",
            "Separately, inverter recalibration from 1 PM to 3 PM will cut usable solar to 20% of forecast.",
        ],
        "directive_interpretation": [
            entry(0, "solar_reduction", {"hours": [12, 13], "factor": 0.5},
                  "Usable solar cut to 50% during hours [12, 13]."),
            entry(1, "solar_reduction", {"hours": [13, 14], "factor": 0.2},
                  "Usable solar cut to 20% during hours [13, 14]."),
        ],
    },
    {
        "id": "EXTRA-02",
        "label": "Overlapping minimum_battery_reserve: more-restrictive (max) reserve wins",
        "rationale": (
            "Two reserve directives overlap at hour 20. percent_of_capacity 40% of 220 kWh = "
            "88 kWh (hours 18-20); a flat 120 kWh reserve (hours 20-22) is stricter. The engine "
            "must take the per-hour max: 88 at 18-19, max(88,120)=120 at 20, 120 at 21-22."
        ),
        "notes": [
            "Keep at least 40% of battery capacity in reserve from 6 PM to 9 PM for the campus radio relay.",
            "A VIP evening event additionally requires at least 120 kWh reserved from 8 PM to 11 PM.",
        ],
        "directive_interpretation": [
            entry(0, "minimum_battery_reserve", {"hours": [18, 19, 20], "minimum_energy_kwh": 88.0},
                  "Reserve of 88.0 kWh during hours [18, 19, 20]."),
            entry(1, "minimum_battery_reserve", {"hours": [20, 21, 22], "minimum_energy_kwh": 120.0},
                  "Reserve of 120.0 kWh during hours [20, 21, 22]."),
        ],
    },
    {
        "id": "EXTRA-03",
        "label": "Overlapping no_charge_window: union of hours, not intersection",
        "rationale": (
            "Two independent maintenance windows both forbid charging, hours 2-4 and 4-5, "
            "overlapping only at hour 4. The engine must block charging on the UNION "
            "{2,3,4,5}, not just the shared hour."
        ),
        "notes": [
            "The primary charger will be offline from 2 AM to 5 AM for scheduled maintenance.",
            "A separate feeder fault will also block charging from 4 AM to 6 AM.",
        ],
        "directive_interpretation": [
            entry(0, "no_charge_window", {"hours": [2, 3, 4]}, "No charging during hours [2, 3, 4]."),
            entry(1, "no_charge_window", {"hours": [4, 5]}, "No charging during hours [4, 5]."),
        ],
    },
    {
        "id": "EXTRA-04",
        "label": "Overlapping max_grid_window: more-restrictive (min) cap wins",
        "rationale": (
            "Two feeders cap grid import over overlapping early-morning windows (chosen so "
            "the caps stay feasible given max_discharge_kwh_per_hour=50 and zero solar at "
            "night: demand-50 is the loosest possible cap per hour). The engine must take "
            "the per-hour minimum: 60 at hour 2, min(60,50)=50 at hours 3-4, 50 at hour 5."
        ),
        "notes": [
            "Feeder A substation work limits grid import to 60 kWh from 2 AM to 5 AM.",
            "Feeder B separately caps grid import at 50 kWh from 3 AM to 6 AM.",
        ],
        "directive_interpretation": [
            entry(0, "max_grid_window", {"hours": [2, 3, 4], "max_grid_kwh": 60.0},
                  "Grid capped at 60.0 kWh during hours [2, 3, 4]."),
            entry(1, "max_grid_window", {"hours": [3, 4, 5], "max_grid_kwh": 50.0},
                  "Grid capped at 50.0 kWh during hours [3, 4, 5]."),
        ],
    },
    {
        "id": "EXTRA-05",
        "label": "Bare hour-of-day without AM/PM must resolve from daytime context",
        "rationale": (
            "Regression case for a real bug found in manual testing: 'from one until three' "
            "has no explicit AM/PM, and the LLM (Groq openai/gpt-oss-120b at temperature 0) "
            "was observed resolving it to 1-3 AM instead of 1-3 PM despite the daytime-only "
            "activity ('panel washing', 'solar output') that should disambiguate it. This case "
            "pins the correct reading so a prompt regression is caught by --base-url mode."
        ),
        "notes": [
            "Panel washing from one until three will leave roughly one-fifth of normal solar output.",
        ],
        "directive_interpretation": [
            entry(0, "solar_reduction", {"hours": [13, 14], "factor": 0.2},
                  "Usable solar cut to 20% during hours [13, 14]."),
        ],
    },
    {
        "id": "EXTRA-06",
        "label": "Window edge cases: 'until midnight' (end=24) and a single-hour window",
        "rationale": (
            "'From 11 PM until midnight' must resolve to end_hour=24 (exclusive), i.e. only "
            "hour 23 -- not an empty or out-of-range window. 'At 3 AM for a one-hour test' is "
            "a single-hour window [3]."
        ),
        "notes": [
            "Grid import will be capped at 60 kWh from 11 PM until midnight for a controlled test.",
            "No charging is allowed at 3 AM for a one-hour relay test.",
        ],
        "directive_interpretation": [
            entry(0, "max_grid_window", {"hours": [23], "max_grid_kwh": 60.0},
                  "Grid capped at 60.0 kWh during hours [23]."),
            entry(1, "no_charge_window", {"hours": [3]}, "No charging during hours [3]."),
        ],
    },
    {
        "id": "EXTRA-07",
        "label": "All three notes are distractors (no_op)",
        "rationale": (
            "Every note is about something other than today's 24-hour electricity schedule. "
            "All three entries must be applies=false / no_op / null, and the plan must equal "
            "the unconstrained cost-optimal schedule (no directives applied)."
        ),
        "notes": [
            "The registrar's office moved next semester's add/drop deadline by two days.",
            "Facilities repainted the lines in the west parking lot over the weekend.",
            "The cafeteria will switch to a new lunch vendor starting next month.",
        ],
        "directive_interpretation": [
            no_op(0, "This note does not affect today's electricity schedule."),
            no_op(1, "This note does not affect today's electricity schedule."),
            no_op(2, "This note does not affect today's electricity schedule."),
        ],
    },
    {
        "id": "EXTRA-08",
        "label": "Battery starts exactly at its own minimum (boundary state)",
        "rationale": (
            "initial_energy_kwh == minimum_energy_kwh == 40, so E[h] >= minimum must hold "
            "from hour 0 onward with zero slack at the start, and end-of-day neutrality "
            "(E[23] == 40) must still be met exactly. Pure boundary/precision check, "
            "independent of any directive."
        ),
        "notes": [
            "The facilities newsletter will go out later this week.",
        ],
        "directive_interpretation": [
            no_op(0, "This note does not affect today's electricity schedule."),
        ],
        "battery_overrides": {"initial_energy_kwh": 40, "minimum_energy_kwh": 40},
    },
]


def build_case(spec: dict[str, Any]) -> dict[str, Any]:
    battery = {**BASE_BATTERY, **spec.get("battery_overrides", {})}
    scenario_input = {
        "scenario_id": spec["id"],
        "operator_notes": spec["notes"],
        "hours": BASE_HOURS,
        "battery": battery,
    }
    scenario = ScenarioRequest(**scenario_input)
    directives = directives_from_interpretation(spec["directive_interpretation"], battery["capacity_kwh"])

    plan = optimize(scenario, directives)
    violations = validate_plan(scenario, directives, plan)
    if violations:
        raise AssertionError(f"{spec['id']}: generated plan fails its own validator: {violations}")

    t = totals(scenario, plan)
    expected_output = {
        "scenario_id": spec["id"],
        "directive_interpretation": spec["directive_interpretation"],
        "hourly_plan": [
            {
                "hour": p.hour, "grid_kwh": p.grid_kwh, "solar_used_kwh": p.solar_used_kwh,
                "battery_action": p.battery_action, "battery_kwh": p.battery_kwh,
                "battery_energy_after_kwh": p.battery_energy_after_kwh,
            }
            for p in plan
        ],
        "total_grid_kwh": t["total_grid_kwh"],
        "total_cost_bdt": t["total_cost_bdt"],
        "peak_grid_kwh": t["peak_grid_kwh"],
        "plan_summary": "generated",
    }
    return {
        "id": spec["id"],
        "label": spec["label"],
        "input": scenario_input,
        "expected_output": expected_output,
        "rationale": spec["rationale"],
    }


def main() -> None:
    cases = [build_case(spec) for spec in CASES]
    pack = {
        "_meta": {
            "description": (
                "Hand-authored edge-case pack complementing the 10 official public samples: "
                "directive-merge rules, window edge cases, an interpretation-ambiguity "
                "regression, an all-distractor scenario, and a battery boundary state. "
                "hourly_plan/totals are generated and self-validated by the project's own "
                "optimizer, not hand-computed; directive_interpretation is hand-authored."
            ),
            "generated_by": "tests/generate_extra_cases.py",
        },
        "cases": cases,
    }
    out_path = ROOT / "tests" / "extra_cases.json"
    out_path.write_text(json.dumps(pack, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(cases)} cases to {out_path}")


if __name__ == "__main__":
    main()
