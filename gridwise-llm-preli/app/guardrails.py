"""Deterministic validation + normalization of LLM output (Plan §6).

Turns the interpreter's intermediate fields into:
  - the response's `directive_interpretation` list (schema-exact shapes), and
  - a `Directives` object for the engine (app/engine_api.py).

Never invents a directive type and never crashes: anything that fails a check
is coerced to a logged no_op for that one note.
"""

from __future__ import annotations

import logging
import math

from app.schemas import Directives, DirectiveInterpretation

logger = logging.getLogger("gridwise.guardrails")

VALID_TYPES = {
    "solar_reduction",
    "minimum_battery_reserve",
    "no_charge_window",
    "no_discharge_window",
    "max_grid_window",
    "no_op",
}


def _no_op(note_index: int, reason: str) -> DirectiveInterpretation:
    logger.warning("note %d coerced to no_op: %s", note_index, reason)
    return DirectiveInterpretation(
        note_index=note_index,
        applies=False,
        directive_type="no_op",
        structured_adjustment=None,
        explanation="No actionable directive found in this note.",
    )


def _as_int(x: object) -> int | None:
    """Accept 18, 18.0 and "18" (LLM JSON output is not always strictly typed);
    reject 18.5, booleans, NaN and anything non-numeric."""
    if isinstance(x, bool):
        return None
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        return int(x) if math.isfinite(x) and x.is_integer() else None
    if isinstance(x, str):
        try:
            return _as_int(float(x.strip()))
        except ValueError:
            return None
    return None


def _as_number(x: object) -> float | None:
    """Accept 60, 60.0, "60", "60%", "1,900"; reject booleans, NaN/inf and text."""
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return float(x) if math.isfinite(x) else None
    if isinstance(x, str):
        cleaned = x.strip().replace(",", "").rstrip("%").strip()
        try:
            v = float(cleaned)
        except ValueError:
            return None
        return v if math.isfinite(v) else None
    return None


def _window_hours(entry: dict) -> list[int] | None:
    """End-exclusive window -> ascending unique hours in 0..23.

    A window that crosses midnight inside the single 24-hour schedule
    ("10 PM to 2 AM" -> start 22, end 2) wraps to hours 22, 23, 0, 1 and is
    returned ascending as [0, 1, 22, 23], as the spec requires ascending order.
    """
    start, end = _as_int(entry.get("start_hour")), _as_int(entry.get("end_hour"))
    if start is None or end is None:
        return None
    if start == 24:
        start = 0
    if not (0 <= start <= 23 and 0 <= end <= 24):
        return None
    if start < end:
        return list(range(start, end))
    if end < start and end != 0:
        return sorted(set(range(start, 24)) | set(range(0, end)))
    if end == 0 and start > 0:        # "... until midnight" written as end 0
        return list(range(start, 24))
    return None                        # start == end: empty window


def _solar_factor(entry: dict) -> float | None:
    value, kind = _as_number(entry.get("value")), entry.get("value_kind")
    if value is None:
        return None
    if kind == "percent_remaining":
        factor = value / 100
    elif kind == "percent_reduced":
        factor = 1 - value / 100
    elif kind == "fraction_remaining":
        factor = value
    elif kind == "fraction_reduced":
        factor = 1 - value
    else:
        return None
    if -1e-9 < factor < 0:             # float noise from 1 - 100/100 etc.
        factor = 0.0
    if 1 < factor < 1 + 1e-9:
        factor = 1.0
    if not (0 <= factor <= 1):
        return None
    return factor


def _reserve_kwh(entry: dict, capacity_kwh: float) -> float | None:
    value, kind = _as_number(entry.get("value")), entry.get("value_kind")
    if value is None:
        return None
    if kind == "kwh":
        reserve = value
    elif kind == "mwh":
        reserve = value * 1000
    elif kind == "percent_of_capacity":
        reserve = value / 100 * capacity_kwh
    else:
        return None
    if not (0 <= reserve <= capacity_kwh + 1e-9):
        return None
    return min(reserve, capacity_kwh)


def _grid_cap_kwh(entry: dict) -> float | None:
    value, kind = _as_number(entry.get("value")), entry.get("value_kind")
    if value is None:
        return None
    if kind == "mwh":
        value *= 1000
    elif kind not in ("kwh", None):
        return None                    # a percentage is not a grid cap in kWh
    if value < 0:
        return None
    return value


def _build_one(entry: dict, capacity_kwh: float) -> DirectiveInterpretation:
    note_index = entry["note_index"]
    directive_type = entry.get("directive_type")

    if directive_type not in VALID_TYPES:
        return _no_op(note_index, f"invalid directive_type {directive_type!r}")
    if directive_type == "no_op":
        return DirectiveInterpretation(
            note_index=note_index,
            applies=False,
            directive_type="no_op",
            structured_adjustment=None,
            explanation="This note does not affect today's electricity schedule.",
        )

    hours = _window_hours(entry)
    if hours is None:
        return _no_op(note_index, "missing or invalid start_hour/end_hour window")

    if directive_type == "solar_reduction":
        factor = _solar_factor(entry)
        if factor is None:
            return _no_op(note_index, "unusable solar_reduction value/value_kind")
        return DirectiveInterpretation(
            note_index=note_index,
            applies=True,
            directive_type="solar_reduction",
            structured_adjustment={"hours": hours, "factor": round(factor, 6)},
            explanation=f"Usable solar cut to {factor * 100:.0f}% during hours {hours}.",
        )

    if directive_type == "minimum_battery_reserve":
        reserve = _reserve_kwh(entry, capacity_kwh)
        if reserve is None:
            return _no_op(note_index, "unusable minimum_battery_reserve value/value_kind")
        return DirectiveInterpretation(
            note_index=note_index,
            applies=True,
            directive_type="minimum_battery_reserve",
            structured_adjustment={"hours": hours, "minimum_energy_kwh": round(reserve, 4)},
            explanation=f"Battery must hold at least {reserve:.1f} kWh during hours {hours}.",
        )

    if directive_type == "no_charge_window":
        return DirectiveInterpretation(
            note_index=note_index,
            applies=True,
            directive_type="no_charge_window",
            structured_adjustment={"hours": hours},
            explanation=f"Charging disabled during hours {hours}.",
        )

    if directive_type == "no_discharge_window":
        return DirectiveInterpretation(
            note_index=note_index,
            applies=True,
            directive_type="no_discharge_window",
            structured_adjustment={"hours": hours},
            explanation=f"Discharging disabled during hours {hours}.",
        )

    if directive_type == "max_grid_window":
        cap = _grid_cap_kwh(entry)
        if cap is None:
            return _no_op(note_index, "unusable max_grid_window value")
        return DirectiveInterpretation(
            note_index=note_index,
            applies=True,
            directive_type="max_grid_window",
            structured_adjustment={"hours": hours, "max_grid_kwh": round(cap, 4)},
            explanation=f"Grid import capped at {cap:.1f} kWh during hours {hours}.",
        )

    return _no_op(note_index, "unreachable directive_type branch")


def _merge_directives(entries: list[DirectiveInterpretation]) -> Directives:
    """More-restrictive-wins merge: min factor, max reserve, union windows, min cap."""
    out = Directives()
    for entry in entries:
        if not entry.applies:
            continue
        adj = entry.structured_adjustment or {}
        hours = [int(h) for h in adj.get("hours", [])]
        if entry.directive_type == "solar_reduction":
            factor = float(adj["factor"])
            for h in hours:
                out.solar_factor[h] = min(out.solar_factor.get(h, 1.0), factor)
        elif entry.directive_type == "minimum_battery_reserve":
            reserve = float(adj["minimum_energy_kwh"])
            for h in hours:
                out.min_reserve[h] = max(out.min_reserve.get(h, 0.0), reserve)
        elif entry.directive_type == "no_charge_window":
            out.no_charge_hours.update(hours)
        elif entry.directive_type == "no_discharge_window":
            out.no_discharge_hours.update(hours)
        elif entry.directive_type == "max_grid_window":
            cap = float(adj["max_grid_kwh"])
            for h in hours:
                out.grid_cap[h] = min(out.grid_cap.get(h, float("inf")), cap)
    return out


def apply_guardrails(
    intermediate: list[dict], capacity_kwh: float
) -> tuple[list[DirectiveInterpretation], Directives]:
    """intermediate: interpreter.interpret_notes() output, one dict per note,
    note_index 0..N-1. Returns (directive_interpretation, Directives)."""
    by_index = {e["note_index"]: e for e in intermediate}
    n = len(intermediate)
    entries: list[DirectiveInterpretation] = []
    for i in range(n):
        entry = by_index.get(i)
        if entry is None:
            entries.append(_no_op(i, "missing note_index from interpreter output"))
        else:
            entries.append(_build_one(entry, capacity_kwh))
    return entries, _merge_directives(entries)
