"""Deterministic validation + normalization of LLM output (Plan §6).

Turns the interpreter's intermediate fields into:
  - the response's `directive_interpretation` list (schema-exact shapes), and
  - a `Directives` object for the engine (app/engine_api.py).

Never invents a directive type and never crashes: anything that fails a check
is coerced to a logged no_op for that one note.
"""

from __future__ import annotations

import logging

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


def _window_hours(entry: dict) -> list[int] | None:
    start, end = entry.get("start_hour"), entry.get("end_hour")
    if not isinstance(start, int) or not isinstance(end, int):
        return None
    if not (0 <= start < end <= 24):
        return None
    return list(range(start, end))


def _solar_factor(entry: dict) -> float | None:
    value, kind = entry.get("value"), entry.get("value_kind")
    if not isinstance(value, (int, float)):
        return None
    value = float(value)
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
    if not (0 <= factor <= 1):
        return None
    return factor


def _reserve_kwh(entry: dict, capacity_kwh: float) -> float | None:
    value, kind = entry.get("value"), entry.get("value_kind")
    if not isinstance(value, (int, float)):
        return None
    value = float(value)
    if kind == "kwh":
        reserve = value
    elif kind == "percent_of_capacity":
        reserve = value / 100 * capacity_kwh
    else:
        return None
    if not (0 <= reserve <= capacity_kwh):
        return None
    return reserve


def _grid_cap_kwh(entry: dict) -> float | None:
    value = entry.get("value")
    if not isinstance(value, (int, float)):
        return None
    value = float(value)
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
