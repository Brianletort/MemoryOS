#!/usr/bin/env python3
"""macOS Calendar.app -> Obsidian Markdown calendar extractor.

Queries Calendar.app via AppleScript (osascript) to extract calendar events.
Fully client-side -- no Graph API or Azure AD consent needed.

Requires the Exchange account to be configured via
System Settings > Internet Accounts with Calendars enabled.

Output:
  - 10_meetings/YYYY/MM/DD/calendar.md  (one file per day with events)
"""

from __future__ import annotations

import argparse
import logging
import re
import subprocess
import sys
import time as _time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.common.config import load_config, resolve_output_dir, setup_logging
from src.common.markdown import write_markdown, yaml_frontmatter
from src.common.state import get_cursor, load_state, save_state, set_cursor

logger = logging.getLogger("memoryos.calendar_app")

FIELD_SEP = "\x1e"
RECORD_SEP = "\x1f"


# ── Self-healing helpers ─────────────────────────────────────────────────────

def _is_calendar_app_running() -> bool:
    result = subprocess.run(
        ["pgrep", "-f", "Calendar.app/Contents/MacOS/Calendar$"],
        capture_output=True,
    )
    return result.returncode == 0


def _ensure_calendar_app() -> bool:
    """Launch Calendar.app if not running. Returns True if a launch was needed."""
    if _is_calendar_app_running():
        return False
    logger.warning("Calendar.app is not running — launching it")
    subprocess.run(["open", "-g", "-j", "-a", "Calendar"], timeout=10, capture_output=True)
    _time.sleep(10)
    logger.info("Calendar.app launched")
    return True


_HEAL_CAL_FILE = Path(__file__).resolve().parent.parent.parent / "config" / "heal_calendar.json"
_MAX_CAL_FAILURES_BEFORE_RESTART: int = 2


def _load_cal_heal() -> dict[str, Any]:
    if _HEAL_CAL_FILE.is_file():
        try:
            import json
            return json.loads(_HEAL_CAL_FILE.read_text())
        except Exception:
            pass
    return {}


def _save_cal_heal(hs: dict[str, Any]) -> None:
    import json
    _HEAL_CAL_FILE.parent.mkdir(parents=True, exist_ok=True)
    _HEAL_CAL_FILE.write_text(json.dumps(hs, indent=2, default=str))


def _restart_calendar_app(reason: str) -> None:
    """Force-quit and reopen Calendar.app to clear hung state."""
    logger.warning("Restarting Calendar.app (%s)", reason)
    subprocess.run(
        ["pkill", "-f", "Calendar.app/Contents/MacOS/Calendar"],
        timeout=10, capture_output=True,
    )
    _time.sleep(5)
    subprocess.run(["open", "-g", "-j", "-a", "Calendar"], timeout=10, capture_output=True)
    _time.sleep(10)
    logger.info("Calendar.app restarted")
    hs = _load_cal_heal()
    hs["consecutive_failures"] = 0
    hs["last_restart_reason"] = reason
    hs["last_restart_epoch"] = _time.time()
    _save_cal_heal(hs)


def _record_cal_failure(reason: str) -> bool:
    """Record a failure and return True if Calendar.app should be restarted."""
    hs = _load_cal_heal()
    count = hs.get("consecutive_failures", 0) + 1
    hs["consecutive_failures"] = count
    hs["last_failure_reason"] = reason
    _save_cal_heal(hs)
    logger.warning(
        "Calendar.app failure [%d/%d]: %s",
        count, _MAX_CAL_FAILURES_BEFORE_RESTART, reason,
    )
    return count >= _MAX_CAL_FAILURES_BEFORE_RESTART


def _reset_cal_failures() -> None:
    hs = _load_cal_heal()
    if hs.get("consecutive_failures", 0) > 0:
        hs["consecutive_failures"] = 0
        _save_cal_heal(hs)


# ── AppleScript helpers ──────────────────────────────────────────────────────

def _run_osascript(script: str, *, timeout: int = 120) -> str:
    wrapper = Path(__file__).resolve().parent.parent.parent / "scripts" / "osascript_wrapper.sh"
    cmd = [str(wrapper), "-e", script] if wrapper.is_file() else ["osascript", "-e", script]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        logger.warning(
            "AppleScript timed out after %ds (Calendar.app may still be syncing)",
            timeout,
        )
        if _record_cal_failure(f"AppleScript timeout after {timeout}s"):
            _restart_calendar_app(f"AppleScript timeout after {timeout}s")
        return ""
    if result.returncode != 0:
        stderr = result.stderr.strip()
        if "execution error" in stderr:
            if "-600" in stderr:
                logger.warning("Calendar.app not running — auto-launching")
                _ensure_calendar_app()
            else:
                logger.warning("AppleScript error: %s", stderr)
            return ""
        raise RuntimeError(f"osascript failed (rc={result.returncode}): {stderr}")
    return result.stdout.strip()


# ── Calendar.app data extraction ─────────────────────────────────────────────

def _reload_calendars() -> None:
    """Ask Calendar.app to refresh its Exchange/iCloud sync."""
    try:
        _run_osascript(
            'tell application "Calendar" to reload calendars',
            timeout=30,
        )
    except Exception:
        logger.debug("reload calendars failed (non-fatal)", exc_info=True)


def fetch_events(
    calendar_names: list[str],
    start: datetime,
    end: datetime,
) -> list[dict[str, str]]:
    """Fetch calendar events via AppleScript for the given date range.

    Returns a list of dicts with keys: summary, start_date, start_time,
    end_date, end_time, location, is_all_day, attendees, organizer, notes.
    """
    _reload_calendars()
    _time.sleep(2)

    start_str = start.strftime("%A, %B %d, %Y at %I:%M:%S %p")
    end_str = end.strftime("%A, %B %d, %Y at %I:%M:%S %p")

    cal_filter_lines = []
    for name in calendar_names:
        cal_filter_lines.append(
            f'if name of c is "{name}" then set targetCals to targetCals & {{c}}'
        )
    cal_filters = "\n            ".join(cal_filter_lines)

    script = f'''
tell application "Calendar" to launch
delay 1
tell application "Calendar"
    set fieldSep to ASCII character 30
    set recSep to ASCII character 31
    set output to ""
    set startDate to date "{start_str}"
    set endDate to date "{end_str}"

    set targetCals to {{}}
    repeat with c in every calendar
        {cal_filters}
    end repeat

    if (count of targetCals) is 0 then
        set targetCals to every calendar
    end if

    repeat with c in targetCals
        try
            set evts to (every event of c whose start date >= startDate and start date <= endDate)
            repeat with e in evts
                set summ to summary of e
                set sDate to start date of e as string
                set eDate to end date of e as string
                set loc to ""
                try
                    set loc to location of e
                    if loc is missing value then set loc to ""
                end try
                set allDay to "false"
                try
                    if allday event of e then set allDay to "true"
                end try
                set output to output & summ & fieldSep & sDate & fieldSep & eDate & fieldSep & loc & fieldSep & allDay & fieldSep & "" & fieldSep & "" & fieldSep & "" & recSep
            end repeat
        end try
    end repeat
    return output
end tell
'''
    raw = _run_osascript(script, timeout=180)
    if not raw:
        return []

    events: list[dict[str, str]] = []
    for record in raw.split(RECORD_SEP):
        record = record.strip()
        if not record:
            continue
        fields = record.split(FIELD_SEP)
        if len(fields) < 4:
            continue

        start_dt = _parse_applescript_date(fields[1].strip())
        end_dt = _parse_applescript_date(fields[2].strip())

        raw_attendees = fields[5].strip() if len(fields) > 5 else ""
        attendees = [
            a.strip() for a in raw_attendees.split(",") if a.strip()
        ]
        raw_notes = fields[6].strip() if len(fields) > 6 else ""
        organizer = fields[7].strip() if len(fields) > 7 else ""

        events.append({
            "summary": fields[0].strip(),
            "start_date": start_dt.strftime("%Y-%m-%d") if start_dt else "unknown",
            "start_time": start_dt.strftime("%H:%M") if start_dt else "??:??",
            "end_date": end_dt.strftime("%Y-%m-%d") if end_dt else "unknown",
            "end_time": end_dt.strftime("%H:%M") if end_dt else "??:??",
            "location": fields[3].strip() if len(fields) > 3 else "",
            "is_all_day": (fields[4].strip().lower() == "true") if len(fields) > 4 else False,
            "attendees": attendees,
            "organizer": organizer,
            "notes": raw_notes,
        })

    return events


def _parse_applescript_date(date_str: str) -> datetime | None:
    if not date_str:
        return None
    date_str = date_str.replace("\u202f", " ").replace("\u00a0", " ")
    fmts = [
        "%A, %B %d, %Y at %I:%M:%S %p",
        "%A, %B %d, %Y at %H:%M:%S",
        "%m/%d/%Y %I:%M:%S %p",
        "%m/%d/%Y %H:%M:%S",
    ]
    for fmt in fmts:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    logger.debug("Could not parse date: %s", date_str)
    return None


def _enrich_with_attendees(
    events: list[dict[str, Any]],
    calendar_names: list[str],
    date: datetime,
) -> None:
    """Fetch attendees for a single day's events (in-place update).

    Runs a separate, targeted AppleScript for just one day to avoid the
    performance penalty of fetching attendees across the full 21-day range.
    """
    start_str = date.replace(hour=0, minute=0).strftime("%A, %B %d, %Y at %I:%M:%S %p")
    end_str = (date.replace(hour=23, minute=59, second=59)).strftime("%A, %B %d, %Y at %I:%M:%S %p")

    cal_filter_lines = []
    for name in calendar_names:
        cal_filter_lines.append(
            f'if name of c is "{name}" then set targetCals to targetCals & {{c}}'
        )
    cal_filters = "\n            ".join(cal_filter_lines)

    script = f'''
tell application "Calendar"
    set fieldSep to ASCII character 30
    set recSep to ASCII character 31
    set output to ""
    set startDate to date "{start_str}"
    set endDate to date "{end_str}"

    set targetCals to {{}}
    repeat with c in every calendar
        {cal_filters}
    end repeat
    if (count of targetCals) is 0 then set targetCals to every calendar

    repeat with c in targetCals
        try
            set evts to (every event of c whose start date >= startDate and start date <= endDate)
            repeat with e in evts
                set summ to summary of e
                set sTime to start date of e as string
                set attendeeList to ""
                try
                    repeat with a in (attendees of e)
                        try
                            set attendeeList to attendeeList & display name of a & ", "
                        end try
                    end repeat
                end try
                set eventNotes to ""
                try
                    set eventNotes to description of e
                    if eventNotes is missing value then set eventNotes to ""
                end try
                set output to output & summ & fieldSep & sTime & fieldSep & attendeeList & fieldSep & eventNotes & recSep
            end repeat
        end try
    end repeat
    return output
end tell
'''
    raw = _run_osascript(script, timeout=60)
    if not raw:
        logger.warning("Attendee enrichment AppleScript returned no data for %s — trying fallback", date.strftime("%Y-%m-%d"))
    else:
        logger.info("Attendee enrichment AppleScript returned %d bytes for %s", len(raw), date.strftime("%Y-%m-%d"))

    lookup: dict[str, dict[str, Any]] = {}
    if raw:
        for record in raw.split(RECORD_SEP):
            record = record.strip()
            if not record:
                continue
            fields = record.split(FIELD_SEP)
            if len(fields) < 2:
                continue
            summ = fields[0].strip()
            att_str = fields[2].strip() if len(fields) > 2 else ""
            notes = fields[3].strip() if len(fields) > 3 else ""
            attendees = [a.strip() for a in att_str.split(",") if a.strip()]
            lookup[summ] = {"attendees": attendees, "notes": notes}

    enriched = 0
    for ev in events:
        info = lookup.get(ev.get("summary", ""))
        if info:
            if info["attendees"]:
                ev["attendees"] = info["attendees"]
            if info["notes"] and not ev.get("notes"):
                ev["notes"] = info["notes"]
            enriched += 1

    fallback_count = 0
    for ev in events:
        if ev.get("attendees"):
            continue
        notes = ev.get("notes", "")
        if not notes:
            continue
        import re
        name_patterns = re.findall(
            r'(?:Required|Optional):\s*([^\n]+)',
            notes, re.IGNORECASE,
        )
        if not name_patterns:
            name_patterns = re.findall(
                r'([A-Z][a-z]+ [A-Z][a-z]+)(?:\s*[,;]|\s*$)',
                notes[:2000],
            )
        if name_patterns:
            names = []
            for chunk in name_patterns:
                for n in chunk.split(";"):
                    n = n.strip().rstrip(",")
                    if n and len(n) > 3 and len(n) < 60:
                        names.append(n)
            if names:
                ev["attendees"] = names[:20]
                fallback_count += 1

    total = enriched + fallback_count
    if total:
        logger.info("Enriched %d events with attendees for %s (AppleScript: %d, fallback: %d)",
                     total, date.strftime("%Y-%m-%d"), enriched, fallback_count)
    else:
        logger.warning("No attendees found for any events on %s", date.strftime("%Y-%m-%d"))


# ── Markdown rendering ───────────────────────────────────────────────────────

_TEAMS_NOISE_PATTERNS = [
    re.compile(r"https?://teams\.microsoft\.com/\S+", re.IGNORECASE),
    re.compile(r"https?://aka\.ms/\S+", re.IGNORECASE),
    re.compile(r"Meeting\s+ID:\s*[\d\s]+", re.IGNORECASE),
    re.compile(r"Passcode:\s*\S+", re.IGNORECASE),
    re.compile(r"_{10,}"),
    re.compile(r"Need help\?.*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"System reference.*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"Microsoft Teams meeting\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"Join:?\s*$", re.MULTILINE),
    re.compile(r"\[​[^\]]*\]\s*", re.IGNORECASE),
]


def _clean_notes(text: str) -> str:
    """Strip Teams join URLs, meeting IDs, passcodes, and boilerplate from notes."""
    for pat in _TEAMS_NOISE_PATTERNS:
        text = pat.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def render_calendar_note(
    date_str: str,
    events: list[dict[str, Any]],
) -> str:
    """Render a daily calendar note with event details."""
    meta = {
        "date": date_str,
        "source": "calendar-app",
        "type": "calendar",
        "event_count": len(events),
    }
    parts = [yaml_frontmatter(meta), ""]
    parts.append(f"# Calendar -- {date_str}")
    parts.append("")

    for ev in events:
        start_str = ev.get("start_time", "??:??")
        end_str = ev.get("end_time", "??:??")
        subject = ev.get("summary", "Untitled Event")

        if ev.get("is_all_day"):
            parts.append(f"## All Day: {subject}")
        else:
            parts.append(f"## {start_str} - {end_str}: {subject}")
        parts.append("")

        if ev.get("location"):
            parts.append(f"- **Location:** {ev['location']}")
        if ev.get("organizer"):
            parts.append(f"- **Organizer:** {ev['organizer']}")
        if ev.get("attendees"):
            parts.append(f"- **Attendees:** {', '.join(ev['attendees'])}")
        if ev.get("notes"):
            cleaned = _clean_notes(ev["notes"])
            if cleaned:
                notes_text = cleaned[:500]
                if len(cleaned) > 500:
                    notes_text += "..."
                parts.append(f"- **Notes:** {notes_text}")

        parts.append("")

    return "\n".join(parts)


_EXISTING_EVENT_RE = re.compile(
    r"^##\s+(?:All Day:\s*(.+)|(\d{1,2}:\d{2})\s*-\s*(\d{1,2}:\d{2}):\s*(.+))",
    re.MULTILINE,
)
_EXISTING_SECTION_RE = re.compile(r"^## ", re.MULTILINE)


def _parse_existing_events(text: str) -> list[dict[str, Any]]:
    """Parse summary + start_time from an existing calendar.md file.

    Returns lightweight dicts with ``summary`` and ``start_time`` keys so
    we can merge with freshly-extracted events without losing old data.
    """
    events: list[dict[str, Any]] = []
    sections = _EXISTING_SECTION_RE.split(text)
    for m in _EXISTING_EVENT_RE.finditer(text):
        if m.group(1):
            events.append({
                "summary": m.group(1).strip(),
                "start_time": "00:00",
                "end_time": "23:59",
                "is_all_day": True,
            })
        else:
            events.append({
                "summary": m.group(4).strip(),
                "start_time": m.group(2),
                "end_time": m.group(3),
                "is_all_day": False,
            })
    return events


def _merge_events(
    new_events: list[dict[str, Any]],
    existing_text: str,
) -> list[dict[str, Any]]:
    """Merge freshly-extracted events with events already in the file.

    Strategy: keep all new events, then append any old events whose
    (summary, start_time) key is not present in the new set.  This
    prevents a flaky 1-event extraction from wiping the file while
    still allowing cancelled meetings to eventually disappear when a
    healthy extraction runs.
    """
    new_keys: set[tuple[str, str]] = set()
    for ev in new_events:
        new_keys.add((ev.get("summary", ""), ev.get("start_time", "")))

    old_events = _parse_existing_events(existing_text)
    carried = 0
    for old in old_events:
        key = (old.get("summary", ""), old.get("start_time", ""))
        if key not in new_keys:
            new_events.append(old)
            new_keys.add(key)
            carried += 1

    if carried:
        logger.info("Merged %d existing events not in new extraction", carried)

    return new_events


# ── Main extraction logic ───────────────────────────────────────────────────

def run(
    cfg: dict[str, Any],
    *,
    dry_run: bool = False,
    days_back: int = 7,
    days_forward: int = 14,
) -> None:
    """Run the Calendar.app extractor."""
    _ensure_calendar_app()

    state_path = cfg["state_file"]
    state = load_state(state_path)

    cal_dir = resolve_output_dir(cfg, "meetings")
    cal_cfg = cfg.get("calendar_app", {})
    calendar_names: list[str] = cal_cfg.get("calendars", ["Calendar"])
    days_back = cal_cfg.get("days_back", days_back)
    days_forward = cal_cfg.get("days_forward", days_forward)

    now = datetime.now()
    start = now - timedelta(days=days_back)
    end = now + timedelta(days=days_forward)

    logger.info(
        "Fetching events from %s to %s (calendars: %s)",
        start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"),
        ", ".join(calendar_names),
    )

    raw_events = fetch_events(calendar_names, start, end)

    date_range_days = (end - start).days
    min_expected = 5 if date_range_days >= 7 else 1
    _sync_glitch = len(raw_events) < min_expected
    if _sync_glitch:
        should_restart = _record_cal_failure(
            f"sync glitch: {len(raw_events)} events for {date_range_days}-day range"
        )
        logger.warning(
            "Only %d events returned for a %d-day range — sync glitch "
            "(restart_threshold=%s)",
            len(raw_events), date_range_days, "REACHED" if should_restart else "not yet",
        )

        if should_restart:
            _restart_calendar_app(f"sync glitch persisted ({len(raw_events)} events)")
            _time.sleep(15)
            retry_events = fetch_events(calendar_names, start, end)
            if len(retry_events) >= min_expected:
                logger.info("Retry after restart fixed: %d -> %d events", len(raw_events), len(retry_events))
                raw_events = retry_events
                _reset_cal_failures()
            else:
                logger.warning("Retry after restart still bad (%d events)", len(retry_events))
                if len(retry_events) > len(raw_events):
                    raw_events = retry_events
        else:
            _reload_calendars()
            _time.sleep(15)
            retry_events = fetch_events(calendar_names, start, end)
            if len(retry_events) >= min_expected:
                logger.info("Retry after reload fixed: %d -> %d events", len(raw_events), len(retry_events))
                raw_events = retry_events
                _reset_cal_failures()
            elif len(retry_events) > len(raw_events):
                logger.info("Retry after reload improved: %d -> %d events", len(raw_events), len(retry_events))
                raw_events = retry_events
            else:
                logger.warning("Retry did not improve (%d events); proceeding with best result", len(retry_events))

    if not raw_events:
        logger.info("No calendar events found")
        if not dry_run:
            set_cursor(
                state, "calendar_app", "last_sync_datetime",
                now.isoformat(),
            )
            save_state(state_path, state)
        return

    logger.info("Processing %d calendar events", len(raw_events))

    # Fix recurring events: Calendar.app sometimes returns the series start
    # date instead of the occurrence date.  Any event whose start_date falls
    # outside the query range is a recurring occurrence -- reassign it to the
    # nearest matching weekday inside the range.
    range_start_str = start.strftime("%Y-%m-%d")
    range_end_str = end.strftime("%Y-%m-%d")
    reassigned = 0
    for ev in raw_events:
        ds = ev.get("start_date", "unknown")
        if ds == "unknown" or range_start_str <= ds <= range_end_str:
            continue
        try:
            orig = datetime.strptime(ds, "%Y-%m-%d")
            target_weekday = orig.weekday()
            candidate = start
            while candidate <= end:
                if candidate.weekday() == target_weekday:
                    new_ds = candidate.strftime("%Y-%m-%d")
                    if new_ds not in [e.get("start_date") for e in raw_events
                                      if e.get("summary") == ev.get("summary")
                                      and range_start_str <= e.get("start_date", "") <= range_end_str]:
                        ev["start_date"] = new_ds
                        ev["end_date"] = new_ds
                        reassigned += 1
                        break
                candidate += timedelta(days=1)
        except ValueError:
            pass
    if reassigned:
        logger.info("Reassigned %d recurring events to correct occurrence dates", reassigned)

    daily_events: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ev in raw_events:
        date_str = ev.get("start_date", "unknown")
        if date_str == "unknown":
            continue
        daily_events[date_str].append(ev)

    for date_str in daily_events:
        seen: set[tuple[str, str]] = set()
        unique: list[dict[str, Any]] = []
        for ev in daily_events[date_str]:
            key = (ev.get("summary", ""), ev.get("start_time", ""))
            if key not in seen:
                seen.add(key)
                unique.append(ev)
        if len(unique) < len(daily_events[date_str]):
            logger.info(
                "Deduped %d -> %d events for %s",
                len(daily_events[date_str]), len(unique), date_str,
            )
        daily_events[date_str] = unique

    today_str = now.strftime("%Y-%m-%d")
    tomorrow_str = (now + timedelta(days=1)).strftime("%Y-%m-%d")
    enrich_dates = {today_str, tomorrow_str}

    for date_str in sorted(daily_events.keys()):
        events = sorted(daily_events[date_str], key=lambda e: e.get("start_time", ""))
        if date_str in enrich_dates:
            try:
                dt = datetime.strptime(date_str, "%Y-%m-%d")
                _enrich_with_attendees(events, calendar_names, dt)
            except Exception:
                logger.warning("Attendee enrichment failed for %s", date_str, exc_info=True)
        cal_content = render_calendar_note(date_str, events)

        parts = date_str.split("-")
        cal_date_path = (
            f"{parts[0]}/{parts[1]}/{parts[2]}" if len(parts) == 3 else date_str
        )
        cal_path = cal_dir / cal_date_path / "calendar.md"

        if cal_path.is_file():
            existing_text = cal_path.read_text(encoding="utf-8", errors="replace")
            events = _merge_events(events, existing_text)
            events = sorted(events, key=lambda e: (
                not e.get("is_all_day"),
                e.get("start_time", ""),
            ))
            cal_content = render_calendar_note(date_str, events)

        if dry_run:
            logger.info("DRY RUN: Would write %s (%d events)", cal_path, len(events))
        else:
            write_markdown(cal_path, cal_content)

    if not dry_run:
        set_cursor(
            state, "calendar_app", "last_sync_datetime",
            now.isoformat(),
        )
        save_state(state_path, state)
        logger.info("Updated calendar_app cursor: %s", now.isoformat())

    if not _sync_glitch:
        _reset_cal_failures()
    logger.info(
        "Calendar.app extraction complete: %d events across %d days",
        len(raw_events), len(daily_events),
    )


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calendar.app -> Obsidian Markdown (Calendar)",
    )
    parser.add_argument("--config", help="Path to config.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    parser.add_argument(
        "--days-back", type=int, default=7,
        help="Days to look back (default: 7)",
    )
    parser.add_argument(
        "--days-forward", type=int, default=14,
        help="Days forward (default: 14)",
    )
    parser.add_argument("--reset", action="store_true", help="Reset cursor")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg)

    if args.reset:
        state = load_state(cfg["state_file"])
        set_cursor(state, "calendar_app", "last_sync_datetime", "")
        save_state(cfg["state_file"], state)
        logger.info("Reset calendar_app cursor")

    run(
        cfg,
        dry_run=args.dry_run,
        days_back=args.days_back,
        days_forward=args.days_forward,
    )


if __name__ == "__main__":
    main()
