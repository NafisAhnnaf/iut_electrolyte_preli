"""Frozen contract file — the two functions that cross the A/B seam.

Person A implements these (in optimizer.py / validator.py), Person B calls them.
Neither side changes the signatures. See GRIDWISE_TEAM_SPLIT.md §0.

    from app.engine_api import optimize, validate_plan
"""

from __future__ import annotations

from app.optimizer import optimize as _optimize
from app.schemas import Directives, HourPlan, InfeasibleError, ScenarioRequest
from app.validator import validate_plan as _validate_plan

__all__ = ["optimize", "validate_plan", "InfeasibleError"]


def optimize(scenario: ScenarioRequest, directives: Directives) -> list[HourPlan]:
    """Returns 24 schema-ready hourly_plan entries. Raises InfeasibleError only."""
    return _optimize(scenario, directives)


def validate_plan(
    scenario: ScenarioRequest,
    directives: Directives,
    plan: list[HourPlan],
) -> list[str]:
    """Replay check (Plan §4). Empty list = valid; else human-readable violations."""
    return _validate_plan(scenario, directives, plan)
