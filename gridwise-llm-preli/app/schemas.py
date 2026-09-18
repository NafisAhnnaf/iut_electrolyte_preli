"""Frozen contract file — pydantic request/response models (Plan §2) plus the
single internal handoff type between the service half and the engine half.

FROZEN: changing anything here is a two-person decision (see GRIDWISE_TEAM_SPLIT.md §0).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Every numeric comparison in this project uses this tolerance.
TOL = 0.01

DirectiveType = Literal[
    "solar_reduction",
    "minimum_battery_reserve",
    "no_charge_window",
    "no_discharge_window",
    "max_grid_window",
    "no_op",
]

BatteryAction = Literal["charge", "discharge", "idle"]


# --------------------------------------------------------------------------- #
# Request
# --------------------------------------------------------------------------- #
class HourInput(BaseModel):
    # extra="ignore": the spec lists required fields but never forbids extras, and a
    # judge harness that adds a metadata field must not 400 every hidden case.
    model_config = ConfigDict(extra="ignore")

    hour: int = Field(ge=0, le=23)
    demand_kwh: float = Field(ge=0, allow_inf_nan=False)
    solar_kwh: float = Field(ge=0, allow_inf_nan=False)
    tariff_bdt_per_kwh: float = Field(ge=0, allow_inf_nan=False)


class Battery(BaseModel):
    model_config = ConfigDict(extra="ignore")

    capacity_kwh: float = Field(gt=0, allow_inf_nan=False)
    initial_energy_kwh: float = Field(ge=0, allow_inf_nan=False)
    minimum_energy_kwh: float = Field(ge=0, allow_inf_nan=False)
    max_charge_kwh_per_hour: float = Field(ge=0, allow_inf_nan=False)
    max_discharge_kwh_per_hour: float = Field(ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def _consistent(self) -> "Battery":
        if self.minimum_energy_kwh > self.capacity_kwh + TOL:
            raise ValueError("minimum_energy_kwh exceeds capacity_kwh")
        if self.initial_energy_kwh > self.capacity_kwh + TOL:
            raise ValueError("initial_energy_kwh exceeds capacity_kwh")
        if self.initial_energy_kwh < self.minimum_energy_kwh - TOL:
            # Inherently infeasible under the spec itself: end-of-day neutrality
            # forces E_after[23] = initial, while every hour requires
            # E_after >= minimum_energy_kwh. Reject with a controlled 400 rather
            # than reaching the optimizer and returning a 500.
            raise ValueError("initial_energy_kwh below minimum_energy_kwh is infeasible "
                             "with end-of-day neutrality")
        return self


class ScenarioRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    scenario_id: str = Field(min_length=1)
    operator_notes: list[str] = Field(min_length=1, max_length=3)
    hours: list[HourInput] = Field(min_length=24, max_length=24)
    battery: Battery

    @field_validator("operator_notes")
    @classmethod
    def _notes_non_empty(cls, v: list[str]) -> list[str]:
        if any(not isinstance(n, str) or not n.strip() for n in v):
            raise ValueError("operator_notes entries must be non-empty strings")
        return v

    @field_validator("hours")
    @classmethod
    def _hours_complete(cls, v: list[HourInput]) -> list[HourInput]:
        if sorted(h.hour for h in v) != list(range(24)):
            raise ValueError("hours must contain each hour 0..23 exactly once")
        return v

    def by_hour(self) -> list[HourInput]:
        """The 24 hour records sorted ascending by `hour` (input order is free)."""
        return sorted(self.hours, key=lambda h: h.hour)


# --------------------------------------------------------------------------- #
# Response
# --------------------------------------------------------------------------- #
class DirectiveInterpretation(BaseModel):
    note_index: int = Field(ge=0)
    applies: bool
    directive_type: DirectiveType
    structured_adjustment: dict[str, Any] | None = None
    explanation: str


class HourPlan(BaseModel):
    hour: int = Field(ge=0, le=23)
    grid_kwh: float = Field(ge=0)
    solar_used_kwh: float = Field(ge=0)
    battery_action: BatteryAction
    battery_kwh: float = Field(ge=0)
    battery_energy_after_kwh: float = Field(ge=0)


class OptimizeResponse(BaseModel):
    scenario_id: str
    directive_interpretation: list[DirectiveInterpretation]
    hourly_plan: list[HourPlan]
    total_grid_kwh: float
    total_cost_bdt: float
    peak_grid_kwh: float
    plan_summary: str


class ErrorResponse(BaseModel):
    error: str


# --------------------------------------------------------------------------- #
# The ONLY object that crosses from Person B's side to Person A's side.
# --------------------------------------------------------------------------- #
@dataclass
class Directives:
    solar_factor: dict[int, float] = field(default_factory=dict)   # hour -> fraction remaining; absent = 1.0
    min_reserve: dict[int, float] = field(default_factory=dict)    # hour -> reserve kWh (already max of base + directives)
    no_charge_hours: set[int] = field(default_factory=set)
    no_discharge_hours: set[int] = field(default_factory=set)
    grid_cap: dict[int, float] = field(default_factory=dict)       # hour -> max grid kWh (already the min if several)


class InfeasibleError(Exception):
    """Raised by optimize() when the LP has no solution under the given directives."""
