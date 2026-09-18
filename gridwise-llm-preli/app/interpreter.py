"""LLM call + prompt + raw-output parsing (Plan §5).

Converts operator notes into the LLM's *intermediate* fields:
    {"note_index": int, "directive_type": str,
     "start_hour": int|None, "end_hour": int|None,
     "value": float|None, "value_kind": str|None}

All arithmetic (percent -> factor, window math, reserve math) happens later
in guardrails.py. This module only talks to the LLM (or, in the emergency
degraded path, a keyword classifier) and hands back those raw fields.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re

import httpx

from app.config import LLM_API_KEY, LLM_ATTEMPT_TIMEOUT_SECONDS, LLM_MODEL, LLM_PROVIDER

logger = logging.getLogger("gridwise.interpreter")

DIRECTIVE_TYPES = (
    "solar_reduction",
    "minimum_battery_reserve",
    "no_charge_window",
    "no_discharge_window",
    "max_grid_window",
    "no_op",
)

SYSTEM_PROMPT = """You convert campus energy operator notes into structured directives.
Directive types: solar_reduction, minimum_battery_reserve, no_charge_window,
no_discharge_window, max_grid_window, no_op.
For EACH note return JSON:
{"note_index": int,
 "directive_type": one of the types,
 "start_hour": int 0-23 or null,   // window start, 24h clock, inclusive
 "end_hour": int 1-24 or null,     // window end, EXCLUSIVE ("until 9 PM" -> 21)
 "value": number or null,          // the numeric quantity mentioned
 "value_kind": "fraction_remaining" | "fraction_reduced" | "percent_remaining" |
               "percent_reduced" | "kwh" | "percent_of_capacity" | null}
Rules:
- Notes about future days, other departments, menus, deadlines, or anything not
  affecting TODAY's 24-hour electricity schedule are no_op (all other fields null).
- "drops to 20%" => percent_remaining 20. "80% reduction" => percent_reduced 80.
  "one-fifth of normal output" => fraction_remaining 0.2.
- "keep at least 120 kWh" => kwh 120. "keep 50% of capacity" => percent_of_capacity 50.
- Times: noon=12, midnight=0, "2 AM"=2, "6 PM"=18, "13:00"=13.
Return {"interpretations":[ ... one per note, in order ... ]}. JSON only, no prose.

Examples:

Notes:
0. Facilities will wash the rooftop panels from noon until 2 PM, leaving usable solar at roughly 25% of forecast.
{"interpretations":[{"note_index":0,"directive_type":"solar_reduction","start_hour":12,"end_hour":14,"value":25,"value_kind":"percent_remaining"}]}

Notes:
0. Keep at least 50% of battery capacity stored from 6 PM until 9 PM for emergency operations.
{"interpretations":[{"note_index":0,"directive_type":"minimum_battery_reserve","start_hour":18,"end_hour":21,"value":50,"value_kind":"percent_of_capacity"}]}

Notes:
0. The battery charger will be isolated from 2 AM until 5 AM for electrical maintenance.
{"interpretations":[{"note_index":0,"directive_type":"no_charge_window","start_hour":2,"end_hour":5,"value":null,"value_kind":null}]}

Notes:
0. Expect an 80% reduction in rooftop solar between 11 AM and 2 PM because of inverter work.
1. The student affairs office will publish club notices tomorrow.
{"interpretations":[{"note_index":0,"directive_type":"solar_reduction","start_hour":11,"end_hour":14,"value":80,"value_kind":"percent_reduced"},{"note_index":1,"directive_type":"no_op","start_hour":null,"end_hour":null,"value":null,"value_kind":null}]}
"""


class InterpreterError(Exception):
    """Raised only internally; callers always get a best-effort list back."""


def _build_user_message(operator_notes: list[str]) -> str:
    lines = [f"{i}. {note}" for i, note in enumerate(operator_notes)]
    return "Notes:\n" + "\n".join(lines)


async def _call_gemini(client: httpx.AsyncClient, user_message: str) -> str:
    # Key goes in a header, NEVER in the URL: httpx error strings include the full
    # request URL, and those strings get logged on failure (e.g. a 429 during judging).
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{LLM_MODEL}:generateContent"
    body = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": [{"text": user_message}]}],
        "generationConfig": {"temperature": 0, "responseMimeType": "application/json"},
    }
    resp = await client.post(url, json=body, headers={"x-goog-api-key": LLM_API_KEY})
    resp.raise_for_status()
    data = resp.json()
    return data["candidates"][0]["content"]["parts"][0]["text"]


async def _call_openai_compatible(client: httpx.AsyncClient, user_message: str, url: str) -> str:
    body = {
        "model": LLM_MODEL,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    }
    headers = {"Authorization": f"Bearer {LLM_API_KEY}"}
    resp = await client.post(url, json=body, headers=headers)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def _safe_err(exc: Exception) -> str:
    """Defence in depth: never let the API key reach a log line, whatever the
    provider or httpx put in the exception message."""
    msg = f"{type(exc).__name__}: {exc}"
    return msg.replace(LLM_API_KEY, "***") if LLM_API_KEY else msg


async def _call_llm(user_message: str, timeout: float) -> str:
    async with httpx.AsyncClient(timeout=timeout) as client:
        if LLM_PROVIDER == "gemini":
            return await _call_gemini(client, user_message)
        if LLM_PROVIDER == "openai":
            return await _call_openai_compatible(
                client, user_message, "https://api.openai.com/v1/chat/completions"
            )
        if LLM_PROVIDER == "groq":
            return await _call_openai_compatible(
                client, user_message, "https://api.groq.com/openai/v1/chat/completions"
            )
        raise InterpreterError(f"unknown LLM_PROVIDER: {LLM_PROVIDER!r}")


def _parse_and_validate(raw_text: str, note_count: int) -> list[dict]:
    data = json.loads(raw_text)
    interpretations = data["interpretations"]
    if not isinstance(interpretations, list) or len(interpretations) != note_count:
        raise InterpreterError("interpretation count does not match note count")
    by_index: dict[int, dict] = {}
    for item in interpretations:
        idx = item["note_index"]
        if item.get("directive_type") not in DIRECTIVE_TYPES:
            raise InterpreterError(f"invalid directive_type: {item.get('directive_type')!r}")
        by_index[idx] = {
            "note_index": idx,
            "directive_type": item["directive_type"],
            "start_hour": item.get("start_hour"),
            "end_hour": item.get("end_hour"),
            "value": item.get("value"),
            "value_kind": item.get("value_kind"),
        }
    if set(by_index.keys()) != set(range(note_count)):
        raise InterpreterError("note_index values do not cover 0..N-1")
    return [by_index[i] for i in range(note_count)]


async def interpret_notes(operator_notes: list[str]) -> list[dict]:
    """Returns one intermediate-field dict per note, in order.

    Tries the real LLM (1 retry on failure), then falls back to a keyword
    classifier as an emergency degraded mode. Never raises.
    """
    if not LLM_API_KEY:
        logger.warning("LLM_API_KEY is not set; using keyword fallback")
        return _keyword_fallback(operator_notes)

    user_message = _build_user_message(operator_notes)
    last_error = ""
    for attempt, timeout in enumerate(LLM_ATTEMPT_TIMEOUT_SECONDS):
        try:
            raw_text = await _call_llm(user_message, timeout)
            return _parse_and_validate(raw_text, len(operator_notes))
        except Exception as exc:  # noqa: BLE001 - any failure triggers retry/fallback
            last_error = _safe_err(exc)
            logger.warning("LLM interpretation attempt %d failed: %s", attempt + 1, last_error)
            if attempt == 0:
                await asyncio.sleep(0.5)
    logger.error("LLM interpretation failed twice (%s); using keyword fallback", last_error)
    return _keyword_fallback(operator_notes)


# --- Emergency degraded mode: deterministic keyword classifier -------------
# Only used if the real LLM call fails twice. Documented in README as a
# fallback path; the primary interpretation path is always the LLM.

_HOUR_WORDS = {"noon": 12, "midnight": 0}
_TIME_RE = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b", re.IGNORECASE)


def _parse_hour(token: str) -> int | None:
    token = token.strip().lower()
    if token in _HOUR_WORDS:
        return _HOUR_WORDS[token]
    m = _TIME_RE.search(token)
    if not m:
        return None
    hour = int(m.group(1))
    meridiem = m.group(3)
    if meridiem == "pm" and hour != 12:
        hour += 12
    if meridiem == "am" and hour == 12:
        hour = 0
    if 0 <= hour <= 24:
        return hour
    return None


_WINDOW_RE = re.compile(
    r"(?:from|between)\s+([^,]+?)\s+(?:until|to|and)\s+([^,.]+?)(?:[,.]|$)",
    re.IGNORECASE,
)
_PERCENT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")
_KWH_RE = re.compile(r"(\d+(?:\.\d+)?)\s*kwh", re.IGNORECASE)
_FRACTION_WORDS = {"one-fifth": 0.2, "one-quarter": 0.25, "one-third": 1 / 3, "half": 0.5}


def _extract_window(note: str) -> tuple[int | None, int | None]:
    m = _WINDOW_RE.search(note)
    if not m:
        return None, None
    start_hour = _parse_hour(m.group(1))
    end_hour = _parse_hour(m.group(2))
    if start_hour is None or end_hour is None:
        return None, None
    return start_hour, end_hour


def _classify_one(note_index: int, note: str) -> dict:
    lowered = note.lower()
    start_hour, end_hour = _extract_window(note)
    empty = {
        "note_index": note_index,
        "directive_type": "no_op",
        "start_hour": None,
        "end_hour": None,
        "value": None,
        "value_kind": None,
    }
    if start_hour is None or end_hour is None:
        return empty

    percent_match = _PERCENT_RE.search(lowered)
    kwh_match = _KWH_RE.search(lowered)
    fraction_value = next((v for w, v in _FRACTION_WORDS.items() if w in lowered), None)

    if "solar" in lowered and (percent_match or fraction_value is not None):
        if "reduction" in lowered or "reduce" in lowered:
            value, kind = (float(percent_match.group(1)), "percent_reduced") if percent_match else (fraction_value, "fraction_reduced")
        else:
            value, kind = (float(percent_match.group(1)), "percent_remaining") if percent_match else (fraction_value, "fraction_remaining")
        return {**empty, "directive_type": "solar_reduction", "start_hour": start_hour, "end_hour": end_hour, "value": value, "value_kind": kind}

    if "reserve" in lowered or "keep at least" in lowered or "remain in the battery" in lowered:
        if kwh_match:
            value, kind = float(kwh_match.group(1)), "kwh"
        elif percent_match:
            value, kind = float(percent_match.group(1)), "percent_of_capacity"
        else:
            return empty
        return {**empty, "directive_type": "minimum_battery_reserve", "start_hour": start_hour, "end_hour": end_hour, "value": value, "value_kind": kind}

    if "not discharge" in lowered or "no discharge" in lowered or ("discharge" in lowered and ("disable" in lowered or "unavailable" in lowered)):
        return {**empty, "directive_type": "no_discharge_window", "start_hour": start_hour, "end_hour": end_hour}

    if "charg" in lowered and ("disable" in lowered or "isolat" in lowered or "unavailable" in lowered):
        return {**empty, "directive_type": "no_charge_window", "start_hour": start_hour, "end_hour": end_hour}

    if "grid" in lowered and ("exceed" in lowered or "cap" in lowered or "limit" in lowered or "constrained" in lowered):
        if kwh_match:
            return {**empty, "directive_type": "max_grid_window", "start_hour": start_hour, "end_hour": end_hour, "value": float(kwh_match.group(1)), "value_kind": "kwh"}

    return empty


def _keyword_fallback(operator_notes: list[str]) -> list[dict]:
    return [_classify_one(i, note) for i, note in enumerate(operator_notes)]
