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
import subprocess
import sys
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


# ── AppleScript helpers ──────────────────────────────────────────────────────

def _run_osascript(script: str, *, timeout: int = 120) -> str:
    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        logger.warning(
            "AppleScript timed out after %ds (Calendar.app may still be syncing)",
            timeout,
        )
        return ""
    if result.returncode != 0:
        stderr = result.stderr.strip()
        if "execution error" in stderr:
            logger.warning("AppleScript error: %s", stderr)
            return ""
        raise RuntimeError(f"osascript failed (rc={result.returncode}): {stderr}")
    return result.stdout.strip()


# ── Calendar.app data extraction ─────────────────────────────────────────────

def fetch_events(
    calendar_names: list[str],
    start: datetime,
    end: datetime,
) -> list[dict[str, str]]:
    """Fetch calendar events via AppleScript for the given date range.

    Returns a list of dicts with keys: summary, start_date, start_time,
    end_date, end_time, location, is_all_day.
    """
    start_str = start.strftime("%A, %B %d, %Y at %I:%M:%S %p")
    end_str = end.strftime("%A, %B %d, %Y at %I:%M:%S %p")

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
                set output to output & summ & fieldSep & sDate & fieldSep & eDate & fieldSep & loc & fieldSep & allDay & recSep
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

        events.append({
            "summary": fields[0].strip(),
            "start_date": start_dt.strftime("%Y-%m-%d") if start_dt else "unknown",
            "start_time": start_dt.strftime("%H:%M") if start_dt else "??:??",
            "end_date": end_dt.strftime("%Y-%m-%d") if end_dt else "unknown",
            "end_time": end_dt.strftime("%H:%M") if end_dt else "??:??",
            "location": fields[3].strip() if len(fields) > 3 else "",
            "is_all_day": (fields[4].strip().lower() == "true") if len(fields) > 4 else False,
        })

    return events


def _parse_applescript_date(date_str: str) -> datetime | None:
    if not date_str:
        return None
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


# ── Markdown rendering ───────────────────────────────────────────────────────

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

        parts.append("")

    return "\n".join(parts)


# ── Main extraction logic ───────────────────────────────────────────────────

def run(
    cfg: dict[str, Any],
    *,
    dry_run: bool = False,
    days_back: int = 7,
    days_forward: int = 14,
) -> None:
    """Run the Calendar.app extractor."""
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

    daily_events: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ev in raw_events:
        date_str = ev.get("start_date", "unknown")
        if date_str == "unknown":
            continue
        daily_events[date_str].append(ev)

    for date_str in sorted(daily_events.keys()):
        events = sorted(daily_events[date_str], key=lambda e: e.get("start_time", ""))
        cal_content = render_calendar_note(date_str, events)

        parts = date_str.split("-")
        cal_date_path = (
            f"{parts[0]}/{parts[1]}/{parts[2]}" if len(parts) == 3 else date_str
        )
        cal_path = cal_dir / cal_date_path / "calendar.md"

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
