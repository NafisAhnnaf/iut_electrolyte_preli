"""Public sample-case harness (Plan section 8).

Two modes:

  --engine-only              Build `Directives` straight from each case's expected
                             directive_interpretation, call optimize() +
                             validate_plan() in process. Proves the engine half on
                             its own: no LLM, no network, no service.

  --base-url http://host:8000
                             Full HTTP end-to-end against a running service: schema,
                             interpretation semantics vs expected, replay of the
                             returned plan against the *ground-truth* directives,
                             totals, cost vs reference, plus the malformed-input,
                             distractor and paraphrase checks.

Exit code 0 only when every selected check passes.

    python tests/run_public_cases.py --engine-only
    python tests/run_public_cases.py --base-url http://localhost:8000
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.engine_api import optimize, validate_plan          # noqa: E402
from app.schemas import TOL, Directives, HourPlan, ScenarioRequest  # noqa: E402
from app.validator import check_totals, totals              # noqa: E402

DEFAULT_CASES = ROOT / "sample_cases.json"

VALID_TYPES = {
    "solar_reduction", "minimum_battery_reserve", "no_charge_window",
    "no_discharge_window", "max_grid_window", "no_op",
}

PARAPHRASE_TRIO = [
    "PV production will drop to about 20% between 13:00 and 15:00",
    "Panel washing from one until three will leave roughly one-fifth of normal solar output",
    "Expect an 80% reduction in rooftop solar during the 1-3 PM maintenance window",
]
PARAPHRASE_EXPECTED = {"hours": [13, 14], "factor": 0.2}


# --------------------------------------------------------------------------- #
# Ground truth -> Directives
# --------------------------------------------------------------------------- #
def directives_from_interpretation(entries: list[dict[str, Any]],
                                   capacity_kwh: float) -> Directives:
    """Merge a directive_interpretation list into the engine's `Directives`, using
    the same more-restrictive-wins rules as guardrails.py (Plan section 6)."""
    out = Directives()
    for entry in entries:
        if not entry.get("applies"):
            continue
        kind = entry.get("directive_type")
        adj = entry.get("structured_adjustment") or {}
        hours = [int(h) for h in adj.get("hours", [])]
        if kind == "solar_reduction":
            factor = float(adj["factor"])
            for h in hours:
                out.solar_factor[h] = min(out.solar_factor.get(h, 1.0), factor)
        elif kind == "minimum_battery_reserve":
            reserve = float(adj["minimum_energy_kwh"])
            for h in hours:
                out.min_reserve[h] = max(out.min_reserve.get(h, 0.0), reserve)
        elif kind == "no_charge_window":
            out.no_charge_hours.update(hours)
        elif kind == "no_discharge_window":
            out.no_discharge_hours.update(hours)
        elif kind == "max_grid_window":
            cap = float(adj["max_grid_kwh"])
            for h in hours:
                out.grid_cap[h] = min(out.grid_cap.get(h, float("inf")), cap)
    del capacity_kwh  # percentages are already resolved to kWh in the expected output
    return out


# --------------------------------------------------------------------------- #
# Comparisons
# --------------------------------------------------------------------------- #
def compare_interpretation(got: Any, expected: list[dict[str, Any]]) -> list[str]:
    """Semantic comparison only -- explanation wording is never matched."""
    errs: list[str] = []
    if not isinstance(got, list):
        return ["directive_interpretation is not a list"]
    if len(got) != len(expected):
        errs.append(f"expected {len(expected)} interpretation entries, got {len(got)}")
    for i, want in enumerate(expected):
        if i >= len(got):
            break
        g = got[i]
        if not isinstance(g, dict):
            errs.append(f"note {i}: entry is not an object")
            continue
        if g.get("note_index") != i:
            errs.append(f"note {i}: note_index is {g.get('note_index')}, entries must be in order")
        if g.get("directive_type") not in VALID_TYPES:
            errs.append(f"note {i}: directive_type {g.get('directive_type')!r} is not in the enum")
        if g.get("directive_type") != want["directive_type"]:
            errs.append(
                f"note {i}: directive_type {g.get('directive_type')!r}, "
                f"expected {want['directive_type']!r}"
            )
        if bool(g.get("applies")) != bool(want["applies"]):
            errs.append(f"note {i}: applies {g.get('applies')}, expected {want['applies']}")
        if not isinstance(g.get("explanation"), str) or not g.get("explanation"):
            errs.append(f"note {i}: explanation must be a non-empty string")
        errs += compare_adjustment(i, g.get("structured_adjustment"),
                                   want.get("structured_adjustment"))
    return errs


def compare_adjustment(i: int, got: Any, want: Any) -> list[str]:
    errs: list[str] = []
    if want is None:
        if got is not None:
            errs.append(f"note {i}: structured_adjustment must be null for no_op, got {got!r}")
        return errs
    if not isinstance(got, dict):
        errs.append(f"note {i}: structured_adjustment must be an object, got {got!r}")
        return errs
    if "hours" in want:
        hours = got.get("hours")
        if not isinstance(hours, list) or [int(h) for h in hours] != want["hours"]:
            errs.append(f"note {i}: hours {got.get('hours')!r}, expected {want['hours']}")
        elif hours != sorted(set(hours)) or any(not (0 <= int(h) <= 23) for h in hours):
            errs.append(f"note {i}: hours must be unique ascending ints 0-23, got {hours}")
    for key in ("factor", "minimum_energy_kwh", "max_grid_kwh"):
        if key in want:
            val = got.get(key)
            if not isinstance(val, (int, float)) or abs(float(val) - float(want[key])) > TOL:
                errs.append(f"note {i}: {key} {val!r}, expected {want[key]}")
    return errs


def check_plan_shape(plan: Any) -> list[str]:
    if not isinstance(plan, list):
        return ["hourly_plan is not a list"]
    if len(plan) != 24:
        return [f"hourly_plan has {len(plan)} entries, expected 24"]
    errs = []
    required = {"hour", "grid_kwh", "solar_used_kwh", "battery_action",
                "battery_kwh", "battery_energy_after_kwh"}
    for row in plan:
        if not isinstance(row, dict):
            errs.append("hourly_plan entry is not an object")
            continue
        missing = required - row.keys()
        if missing:
            errs.append(f"hour {row.get('hour')}: missing fields {sorted(missing)}")
        if row.get("battery_action") not in {"charge", "discharge", "idle"}:
            errs.append(f"hour {row.get('hour')}: bad battery_action {row.get('battery_action')!r}")
    return errs


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
class Report:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, bool, str]] = []

    def add(self, case_id: str, label: str, ok: bool, detail: str = "") -> None:
        self.rows.append((case_id, label, ok, detail))

    @property
    def failed(self) -> int:
        return sum(1 for _, _, ok, _ in self.rows if not ok)

    def print(self) -> None:
        width = max((len(r[0]) for r in self.rows), default=10)
        lwidth = max((len(r[1]) for r in self.rows), default=20)
        print()
        print(f"{'CASE'.ljust(width)}  {'CHECK'.ljust(lwidth)}  RESULT")
        print("-" * (width + lwidth + 12))
        for case_id, label, ok, detail in self.rows:
            mark = "PASS" if ok else "FAIL"
            print(f"{case_id.ljust(width)}  {label.ljust(lwidth)}  {mark}"
                  + (f"  {detail}" if detail else ""))
        total = len(self.rows)
        print("-" * (width + lwidth + 12))
        print(f"{total - self.failed}/{total} checks passed")


def _fmt(errs: list[str], limit: int = 2) -> str:
    if not errs:
        return ""
    head = "; ".join(errs[:limit])
    return head + (f" (+{len(errs) - limit} more)" if len(errs) > limit else "")


# --------------------------------------------------------------------------- #
# Mode 1: engine only
# --------------------------------------------------------------------------- #
def run_engine_only(cases: list[dict[str, Any]], report: Report) -> None:
    for case in cases:
        cid = case["id"]
        scenario = ScenarioRequest(**case["input"])
        expected = case["expected_output"]
        directives = directives_from_interpretation(
            expected["directive_interpretation"], scenario.battery.capacity_kwh
        )

        t0 = time.perf_counter()
        try:
            plan = optimize(scenario, directives)
        except Exception as exc:  # noqa: BLE001 - harness reports, never crashes
            report.add(cid, "optimize", False, f"{type(exc).__name__}: {exc}")
            continue
        elapsed = (time.perf_counter() - t0) * 1000

        violations = validate_plan(scenario, directives, plan)
        report.add(cid, "replay validator", not violations, _fmt(violations))

        ours = totals(scenario, plan)
        ref = float(expected["total_cost_bdt"])
        ok = ours["total_cost_bdt"] <= ref + TOL
        report.add(cid, "cost <= reference", ok,
                   f"ours {ours['total_cost_bdt']:.2f} vs ref {ref:.2f} ({elapsed:.0f} ms)")

        # The reference schedule must itself pass our validator -- if it does not,
        # our reading of the rules disagrees with the judge's.
        ref_plan = [HourPlan(**row) for row in expected["hourly_plan"]]
        ref_violations = validate_plan(scenario, directives, ref_plan)
        report.add(cid, "reference replays", not ref_violations, _fmt(ref_violations))


# --------------------------------------------------------------------------- #
# Mode 2: full HTTP
# --------------------------------------------------------------------------- #
def run_http(cases: list[dict[str, Any]], base_url: str, report: Report) -> None:
    import httpx

    base = base_url.rstrip("/")
    with httpx.Client(timeout=35.0) as client:
        try:
            r = client.get(f"{base}/health")
            ok = r.status_code == 200 and r.json().get("status") == "ok"
            report.add("-", "GET /health", ok, f"HTTP {r.status_code}")
        except Exception as exc:  # noqa: BLE001
            report.add("-", "GET /health", False, f"{type(exc).__name__}: {exc}")
            return

        for case in cases:
            cid = case["id"]
            scenario = ScenarioRequest(**case["input"])
            expected = case["expected_output"]

            t0 = time.perf_counter()
            try:
                r = client.post(f"{base}/optimize-energy", json=case["input"])
            except Exception as exc:  # noqa: BLE001
                report.add(cid, "POST", False, f"{type(exc).__name__}: {exc}")
                continue
            elapsed = (time.perf_counter() - t0) * 1000
            if r.status_code != 200:
                report.add(cid, "POST", False, f"HTTP {r.status_code}: {r.text[:120]}")
                continue
            body = r.json()
            report.add(cid, "POST", True, f"{elapsed:.0f} ms")
            report.add(cid, "latency < 5s", elapsed < 5000, f"{elapsed:.0f} ms")

            if body.get("scenario_id") != case["input"]["scenario_id"]:
                report.add(cid, "scenario_id echoed", False, repr(body.get("scenario_id")))
            else:
                report.add(cid, "scenario_id echoed", True)

            # Interpretation is Person B's half -- report it separately so a failure
            # here is never confused with an engine failure.
            interp_errs = compare_interpretation(
                body.get("directive_interpretation"), expected["directive_interpretation"]
            )
            report.add(cid, "interpretation [B]", not interp_errs, _fmt(interp_errs))

            shape_errs = check_plan_shape(body.get("hourly_plan"))
            report.add(cid, "plan schema", not shape_errs, _fmt(shape_errs))
            if shape_errs:
                continue

            plan = [HourPlan(**row) for row in body["hourly_plan"]]
            truth = directives_from_interpretation(
                expected["directive_interpretation"], scenario.battery.capacity_kwh
            )
            violations = validate_plan(scenario, truth, plan)
            # The plan is replayed against the *expected* directives, so if the
            # interpretation was already wrong this will fail downstream of it --
            # say so, rather than sending Person A hunting in the optimizer.
            hint = " (downstream of the interpretation failure)" if interp_errs else ""
            report.add(cid, "replay vs truth [A]", not violations, _fmt(violations) + hint)

            total_errs = check_totals(scenario, plan, body)
            report.add(cid, "totals match plan", not total_errs, _fmt(total_errs))

            ref = float(expected["total_cost_bdt"])
            got_cost = float(body.get("total_cost_bdt", float("inf")))
            report.add(cid, "cost <= reference", got_cost <= ref + TOL,
                       f"ours {got_cost:.2f} vs ref {ref:.2f}")

            if not isinstance(body.get("plan_summary"), str) or not body["plan_summary"]:
                report.add(cid, "plan_summary", False, "missing or empty")
            else:
                report.add(cid, "plan_summary", True)

        run_robustness(client, base, cases, report)


def run_robustness(client: Any, base: str, cases: list[dict[str, Any]],
                   report: Report) -> None:
    """Malformed input, distractor and paraphrase checks (Plan section 8, last paragraph)."""
    url = f"{base}/optimize-energy"
    template = copy.deepcopy(cases[0]["input"])

    r = client.post(url, content=b"{not json", headers={"Content-Type": "application/json"})
    report.add("robust", "malformed JSON -> 400", r.status_code == 400, f"HTTP {r.status_code}")

    bad = copy.deepcopy(template)
    bad["operator_notes"] = []
    r = client.post(url, json=bad)
    report.add("robust", "empty notes -> 400", r.status_code == 400, f"HTTP {r.status_code}")

    bad = copy.deepcopy(template)
    del bad["battery"]
    r = client.post(url, json=bad)
    report.add("robust", "missing battery -> 400", r.status_code == 400, f"HTTP {r.status_code}")

    bad = copy.deepcopy(template)
    bad["hours"] = bad["hours"][:20]
    r = client.post(url, json=bad)
    report.add("robust", "20 hours -> 400", r.status_code == 400, f"HTTP {r.status_code}")

    garbage = copy.deepcopy(template)
    garbage["scenario_id"] = "ROBUST-GARBAGE"
    garbage["operator_notes"] = ["qwerty zxcvb lorem ipsum 4821"]
    r = client.post(url, json=garbage)
    if r.status_code != 200:
        report.add("robust", "garbage note -> no_op", False, f"HTTP {r.status_code}")
    else:
        body = r.json()
        entries = body.get("directive_interpretation") or [{}]
        ok = (entries[0].get("directive_type") == "no_op"
              and entries[0].get("applies") is False
              and entries[0].get("structured_adjustment") is None)
        report.add("robust", "garbage note -> no_op", ok, "" if ok else json.dumps(entries[0]))
        scenario = ScenarioRequest(**garbage)
        plan_errs = check_plan_shape(body.get("hourly_plan"))
        if not plan_errs:
            plan = [HourPlan(**row) for row in body["hourly_plan"]]
            violations = validate_plan(scenario, Directives(), plan)
            report.add("robust", "garbage plan valid", not violations, _fmt(violations))
        else:
            report.add("robust", "garbage plan valid", False, _fmt(plan_errs))

    seen: list[Any] = []
    for i, note in enumerate(PARAPHRASE_TRIO):
        payload = copy.deepcopy(template)
        payload["scenario_id"] = f"PARAPHRASE-{i + 1}"
        payload["operator_notes"] = [note]
        r = client.post(url, json=payload)
        if r.status_code != 200:
            report.add("robust", f"paraphrase {i + 1}", False, f"HTTP {r.status_code}")
            continue
        entry = (r.json().get("directive_interpretation") or [{}])[0]
        errs = compare_adjustment(i, entry.get("structured_adjustment"), PARAPHRASE_EXPECTED)
        if entry.get("directive_type") != "solar_reduction":
            errs.append(f"directive_type {entry.get('directive_type')!r}")
        report.add("robust", f"paraphrase {i + 1} [B]", not errs, _fmt(errs))
        seen.append(entry.get("structured_adjustment"))
    if len(seen) == 3:
        # All-null would be "identical" but is really three failures, so require a
        # real adjustment as well as agreement.
        ok = seen[0] == seen[1] == seen[2] and seen[0] is not None
        report.add("robust", "paraphrases identical [B]", ok, "" if ok else json.dumps(seen))


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="GridWise public sample-case harness")
    ap.add_argument("--engine-only", action="store_true",
                    help="run the optimizer/validator in process, no service needed")
    ap.add_argument("--base-url", help="run the full HTTP suite against this service")
    ap.add_argument("--cases", default=str(DEFAULT_CASES), help="path to the sample case pack")
    ap.add_argument("--only", help="run a single case id, e.g. SAMPLE-03")
    args = ap.parse_args()

    if not args.engine_only and not args.base_url:
        ap.error("pass --engine-only or --base-url (or both)")

    pack = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    cases = pack["cases"]
    if args.only:
        cases = [c for c in cases if c["id"] == args.only]
        if not cases:
            ap.error(f"no case with id {args.only}")

    report = Report()
    if args.engine_only:
        print(f"engine-only mode: {len(cases)} cases from {args.cases}")
        run_engine_only(cases, report)
    if args.base_url:
        print(f"HTTP mode: {len(cases)} cases against {args.base_url}")
        run_http(cases, args.base_url, report)

    report.print()
    return 1 if report.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
