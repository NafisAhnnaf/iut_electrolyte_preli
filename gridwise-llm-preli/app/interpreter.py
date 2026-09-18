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
import random
import re
import time

import httpx

from app.config import (
    LLM_API_KEY,
    LLM_ATTEMPT_TIMEOUT_SECONDS,
    LLM_BACKOFF_BASE_SECONDS,
    LLM_BACKOFF_MAX_SECONDS,
    LLM_MODEL,
    LLM_PROVIDER,
    LLM_TOTAL_BUDGET_SECONDS,
)

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
               "percent_reduced" | "kwh" | "mwh" | "percent_of_capacity" | null}
Rules:
- no_op (all other fields null) for anything that does not constrain TODAY's schedule:
  future days ("next Monday", "tomorrow"), past events ("yesterday"), other departments,
  announcements, meetings, and requests to change demand, tariffs, or to export power
  (those are NOT supported directive types - never invent a directive for them).
- "drops to 20%" => percent_remaining 20. "80% reduction" => percent_reduced 80.
  "one-fifth of normal output" => fraction_remaining 0.2. "cut in half" => fraction_remaining 0.5.
  "completely offline"/"no solar at all" => percent_remaining 0.
- "keep at least 120 kWh" => kwh 120. "keep 50% of capacity" => percent_of_capacity 50.
  "half the capacity" => percent_of_capacity 50. "completely full" => percent_of_capacity 100.
  A value in MWh => value as written, value_kind "mwh" (never convert units yourself).
- Numbers: write words as digits ("one hundred and twenty" => 120, "1,900" => 1900).
  value, start_hour and end_hour must be JSON numbers, never strings.
- Times: noon=12, midnight=0, "2 AM"=2, "6 PM"=18, "13:00"=13. A window ending at midnight
  has end_hour 24. "all day"/"entire day" => start_hour 0, end_hour 24.
  "for 3 hours starting at 3 PM" => start_hour 15, end_hour 18. A single hour "at 7 PM"
  => start_hour 19, end_hour 20. Bare hours like "from one until three" for solar work mean
  daytime (13 to 15). A window crossing midnight ("10 PM to 2 AM") => start_hour 22, end_hour 2.
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
        idx = int(item["note_index"])  # tolerate "0" / 0.0 from the model
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


# --- Retry-delay extraction: use the provider's own rate-limit hints instead of
# guessing with a fixed sleep. Falls back to exponential backoff with jitter when
# the response gives no hint at all. -----------------------------------------

_GO_DURATION_RE = re.compile(
    r"(?:(?P<h>\d+(?:\.\d+)?)h)?(?:(?P<m>\d+(?:\.\d+)?)m)?(?:(?P<s>\d+(?:\.\d+)?)s)?"
    r"(?:(?P<ms>\d+(?:\.\d+)?)ms)?$"
)


def _parse_duration(text: str) -> float | None:
    """Parses Go-style durations ("31s", "9.397s", "1h42m14.4s", "500ms") as used
    by Groq's x-ratelimit-reset-* headers and similar provider fields."""
    text = text.strip()
    m = _GO_DURATION_RE.fullmatch(text)
    if not m or not any(m.groups()):
        return None
    parts = m.groupdict()
    seconds = 0.0
    if parts["h"]:
        seconds += float(parts["h"]) * 3600
    if parts["m"]:
        seconds += float(parts["m"]) * 60
    if parts["s"]:
        seconds += float(parts["s"])
    if parts["ms"]:
        seconds += float(parts["ms"]) / 1000
    return seconds if seconds > 0 else None


def _retry_delay_seconds(exc: Exception) -> float | None:
    """Best-effort extraction of a provider-suggested retry delay. Returns None if
    the exception carries no usable hint (caller then uses backoff)."""
    resp = getattr(exc, "response", None)
    if resp is None:
        return None

    retry_after = resp.headers.get("retry-after")
    if retry_after:
        try:
            return max(0.0, float(retry_after))
        except ValueError:
            pass  # could be an HTTP-date; not worth parsing for this use case

    # Groq-style: x-ratelimit-reset-requests / x-ratelimit-reset-tokens (Go duration
    # strings). Either resource clearing is enough to plausibly succeed again, so
    # take the smaller of the two when both are present.
    candidates = []
    for header in ("x-ratelimit-reset-tokens", "x-ratelimit-reset-requests"):
        raw = resp.headers.get(header)
        if raw:
            parsed = _parse_duration(raw)
            if parsed is not None:
                candidates.append(parsed)
    if candidates:
        return min(candidates)

    # Gemini-style: JSON error body with a RetryInfo detail, e.g.
    # {"error": {"details": [{"@type": ".../RetryInfo", "retryDelay": "31s"}]}}
    try:
        body = resp.json()
        for detail in body.get("error", {}).get("details", []):
            if "retryDelay" in detail:
                parsed = _parse_duration(str(detail["retryDelay"]))
                if parsed is not None:
                    return parsed
    except Exception:  # noqa: BLE001 - body may not be JSON at all
        pass

    return None


async def interpret_notes(operator_notes: list[str]) -> list[dict]:
    """Returns one intermediate-field dict per note, in order.

    Tries the real LLM up to len(LLM_ATTEMPT_TIMEOUT_SECONDS) times, waiting
    between attempts according to the provider's own rate-limit hint (Retry-After,
    Groq's x-ratelimit-reset-* headers, or Gemini's RetryInfo.retryDelay) when one
    is present, falling back to exponential backoff with jitter otherwise. Never
    waits past the total wall-clock budget. Falls back to a keyword classifier as
    an emergency degraded mode only if every attempt fails. Never raises.
    """
    if not LLM_API_KEY:
        logger.warning("LLM_API_KEY is not set; using keyword fallback")
        return _keyword_fallback(operator_notes)

    user_message = _build_user_message(operator_notes)
    last_error = ""
    start = time.monotonic()
    attempts = LLM_ATTEMPT_TIMEOUT_SECONDS

    for attempt, timeout in enumerate(attempts):
        elapsed = time.monotonic() - start
        remaining = LLM_TOTAL_BUDGET_SECONDS - elapsed
        if remaining < timeout:
            timeout = remaining
        if timeout <= 0.5:
            logger.warning("LLM interpretation budget exhausted before attempt %d", attempt + 1)
            break

        try:
            # httpx's timeout applies to EACH read, not the whole request: a provider
            # that drips bytes slowly never trips it. wait_for is the hard wall-clock
            # ceiling per attempt, which is what keeps us inside the judge's 30 s.
            raw_text = await asyncio.wait_for(_call_llm(user_message, timeout), timeout=timeout)
            return _parse_and_validate(raw_text, len(operator_notes))
        except Exception as exc:  # noqa: BLE001 - any failure triggers retry/fallback
            last_error = _safe_err(exc)
            logger.warning("LLM interpretation attempt %d failed: %s", attempt + 1, last_error)

            if attempt == len(attempts) - 1:
                break  # no more attempts left, skip the sleep and go to fallback

            hinted = _retry_delay_seconds(exc)
            if hinted is not None:
                delay = hinted
                source = "provider hint"
            else:
                delay = min(LLM_BACKOFF_MAX_SECONDS,
                            LLM_BACKOFF_BASE_SECONDS * (2 ** attempt))
                delay += random.uniform(0, delay * 0.25)  # jitter
                source = "backoff"

            elapsed = time.monotonic() - start
            remaining = LLM_TOTAL_BUDGET_SECONDS - elapsed
            next_timeout = attempts[attempt + 1]
            sleep_for = max(0.0, min(delay, remaining - next_timeout))
            if sleep_for <= 0:
                logger.warning("no budget left to wait out a %s of %.1fs; giving up early",
                                source, delay)
                break
            logger.info("waiting %.1fs before retry (%s: %.1fs)", sleep_for, source, delay)
            await asyncio.sleep(sleep_for)

    logger.error("LLM interpretation failed after %d attempt(s) (%s); using keyword fallback",
                 attempt + 1, last_error)
    return _keyword_fallback(operator_notes)


# --- Emergency degraded mode: deterministic keyword classifier -------------
# Only used if every LLM attempt fails (or no key is configured). Documented in README as a
# fallback path; the primary interpretation path is always the LLM.

_HOUR_WORDS = {"noon": 12, "midday": 12, "midnight": 0}
_WORD_HOURS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
               "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12}
_TIME_RE = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b", re.IGNORECASE)


def _parse_hour(token: str, assume_meridiem: str | None = None) -> int | None:
    token = token.strip().lower()
    for word, hour in _HOUR_WORDS.items():
        if re.search(rf"\b{word}\b", token):  # \b keeps "afternoon" from matching "noon"
            return hour
    m = _TIME_RE.search(token)
    hour = None
    meridiem = None
    if m:
        hour = int(m.group(1))
        meridiem = m.group(3).lower() if m.group(3) else None
    else:
        # spelled-out hours: "from one until three" — earliest match by position,
        # so "three ... one-fifth" resolves to three, not one.
        best = None
        for word, value in _WORD_HOURS.items():
            m2 = re.search(rf"\b{word}\b", token)
            if m2 and (best is None or m2.start() < best[0]):
                best = (m2.start(), value)
        if best is not None:
            hour = best[1]
    if meridiem is None and ("afternoon" in token or "evening" in token):
        meridiem = "pm"
    elif meridiem is None and "morning" in token:
        meridiem = "am"
    if hour is None:
        return None
    meridiem = meridiem or assume_meridiem
    if meridiem == "pm" and hour != 12:
        hour += 12
    if meridiem == "am" and hour == 12:
        hour = 0
    if 0 <= hour <= 24:
        return hour
    return None


_WINDOW_RE = re.compile(
    r"(?:from|between)\s+([^,]+?)\s+(?:until|to|and|through)\s+([^,.;]+?)(?:[,.;]|$)",
    re.IGNORECASE,
)
_RANGE_RE = re.compile(  # "1-3 PM", "13:00-15:00"
    r"\b(\d{1,2})(?::\d{2})?\s*(am|pm)?\s*[-–]\s*(\d{1,2})(?::\d{2})?\s*(am|pm)?\b",
    re.IGNORECASE,
)
# Only a real clock time counts ("at 7 PM", "at 19:00", "at noon") -- never a quantity
# like "capped at 1.9 kWh".
_SINGLE_HOUR_RE = re.compile(r"\bat\s+(\d{1,2}(?::\d{2})?\s*(?:am|pm)\b|\d{1,2}:\d{2}\b|noon\b|midnight\b)", re.IGNORECASE)
_OTHER_DAY_RE = re.compile(
    r"\b(yesterday|tomorrow|last (night|week|month|year)|next (week|month|year|monday|tuesday|"
    r"wednesday|thursday|friday|saturday|sunday|semester)|previous day|the day before)\b"
)
_PERCENT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:%|percent)", re.IGNORECASE)
_KWH_RE = re.compile(r"(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)\s*(?:kwh|kilowatt[- ]hours?)", re.IGNORECASE)
_FRACTION_WORDS = {
    "one-fifth": 0.2, "one fifth": 0.2,
    "one-quarter": 0.25, "one quarter": 0.25, "a quarter": 0.25,
    "one-third": 1 / 3, "one third": 1 / 3, "a third": 1 / 3,
    "two-thirds": 2 / 3, "two thirds": 2 / 3,
    "three-quarters": 0.75, "three quarters": 0.75,
    "one-half": 0.5, "half": 0.5,
}


_MERIDIEM_MARKERS = re.compile(r"\bam\b|\bpm\b|\bnoon\b|\bmidday\b|\bmidnight\b|"
                               r"afternoon|evening|morning|\d:\d\d", re.IGNORECASE)


def _extract_window(note: str, daytime_bias: bool = False) -> tuple[int | None, int | None]:
    # Try every "from X until Y" span and keep the first whose both sides parse
    # as hours (skips false starts like "from the grid ...").
    for m in _WINDOW_RE.finditer(note):
        left, right = m.group(1), m.group(2)
        end_hour = _parse_hour(right)
        # "from one until three" / "1 to 3 PM": inherit the right side's meridiem.
        right_meridiem = "pm" if re.search(r"\bpm\b", right, re.IGNORECASE) else (
            "am" if re.search(r"\bam\b", right, re.IGNORECASE) else None)
        start_hour = _parse_hour(left, assume_meridiem=right_meridiem)
        if end_hour == 0 and start_hour is not None and start_hour > 0:
            end_hour = 24  # "... until midnight" ends the day, it is not hour 0
        if start_hour is not None and end_hour is not None and start_hour < end_hour:
            # Daytime assumption for solar notes with NO am/pm marker at all:
            # "panel washing from one until three" means 13-15, not 01-03.
            if (daytime_bias and 1 <= start_hour and end_hour <= 8
                    and not _MERIDIEM_MARKERS.search(m.group(0))):
                start_hour, end_hour = start_hour + 12, end_hour + 12
            return start_hour, end_hour
    m = _SINGLE_HOUR_RE.search(note)  # "at 3 AM for a one-hour test"
    if m and re.search(r"one[- ]hour|an hour|that hour|single hour|1[- ]hour", note, re.IGNORECASE):
        hour = _parse_hour(m.group(1))
        if hour is not None and 0 <= hour <= 23:
            return hour, hour + 1
    m = _RANGE_RE.search(note)  # "1-3 PM", "13:00-15:00"
    if m:
        mer_end = m.group(4).lower() if m.group(4) else None
        mer_start = (m.group(2).lower() if m.group(2) else None) or mer_end
        start_hour = _parse_hour(m.group(1) + " " + (mer_start or ""))
        end_hour = _parse_hour(m.group(3) + " " + (mer_end or ""))
        if start_hour is not None and end_hour is not None and start_hour < end_hour:
            return start_hour, end_hour
    return None, None


def _classify_one(note_index: int, note: str) -> dict:
    note = re.sub(r"\b([ap])\.m\.", r"\1m", note, flags=re.IGNORECASE)  # "p.m." -> "pm"
    lowered = note.lower()
    solar_context = bool(_any_in(lowered, "solar", "panel", "rooftop") or re.search(r"\bpv\b", lowered))
    start_hour, end_hour = _extract_window(note, daytime_bias=solar_context)
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
    if _OTHER_DAY_RE.search(lowered):
        return empty  # "yesterday", "next Monday", "tomorrow": not today's schedule

    percent_match = _PERCENT_RE.search(lowered)
    kwh_match = _KWH_RE.search(lowered)
    fraction_value = next((v for w, v in _FRACTION_WORDS.items() if w in lowered), None)

    def _any(*phrases: str) -> bool:
        return _any_in(lowered, *phrases)

    # Reserve first: reserve notes often mention kWh/percent AND words like "battery".
    # Falls through (does not return no_op) when the trigger matched but no usable
    # value was found, so other directive families still get a chance.
    if _any("reserve", "keep at least", "hold at least", "maintain at least",
            "store at least", "remain in the battery", "stored in the battery",
            "in storage"):
        if kwh_match:
            return {**empty, "directive_type": "minimum_battery_reserve", "start_hour": start_hour, "end_hour": end_hour, "value": float(kwh_match.group(1).replace(',', '')), "value_kind": "kwh"}
        if percent_match:
            return {**empty, "directive_type": "minimum_battery_reserve", "start_hour": start_hour, "end_hour": end_hour, "value": float(percent_match.group(1)), "value_kind": "percent_of_capacity"}
        if fraction_value is not None and "capacit" in lowered:
            return {**empty, "directive_type": "minimum_battery_reserve", "start_hour": start_hour, "end_hour": end_hour, "value": fraction_value * 100, "value_kind": "percent_of_capacity"}

    if (_any("solar", "panel", "rooftop") or re.search(r"\bpv\b", lowered)) and (percent_match or fraction_value is not None):
        if _any("reduction", "reduce", "cut by", "drop by", "down by"):
            value, kind = (float(percent_match.group(1)), "percent_reduced") if percent_match else (fraction_value, "fraction_reduced")
        else:
            value, kind = (float(percent_match.group(1)), "percent_remaining") if percent_match else (fraction_value, "fraction_remaining")
        return {**empty, "directive_type": "solar_reduction", "start_hour": start_hour, "end_hour": end_hour, "value": value, "value_kind": kind}

    if _any("not discharge", "no discharge", "must not be discharged") or (
        "discharg" in lowered and _any("disable", "unavailable", "suspend", "paused", "prohibit", "forbidden", "off-line", "offline", "block")
    ):
        return {**empty, "directive_type": "no_discharge_window", "start_hour": start_hour, "end_hour": end_hour}

    if _any("not charge", "no charging", "do not charge", "must not be charged", "charging is not") or (
        "charg" in lowered and _any("disable", "isolat", "unavailable", "suspend", "paused", "prohibit", "forbidden", "maintenance", "off-line", "offline", "block", "not allowed", "out of service")
    ):
        return {**empty, "directive_type": "no_charge_window", "start_hour": start_hour, "end_hour": end_hour}

    if _any("grid", "import", "feeder", "draw") and _any(
        "exceed", "cap", "limit", "constrained", "within", "no more than",
        "at most", "under", "below", "maximum", "max "
    ):
        if kwh_match:
            return {**empty, "directive_type": "max_grid_window", "start_hour": start_hour, "end_hour": end_hour, "value": float(kwh_match.group(1).replace(',', '')), "value_kind": "kwh"}

    return empty


def _any_in(lowered: str, *phrases: str) -> bool:
    return any(p in lowered for p in phrases)


def _keyword_fallback(operator_notes: list[str]) -> list[dict]:
    return [_classify_one(i, note) for i, note in enumerate(operator_notes)]
