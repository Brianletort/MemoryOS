#!/usr/bin/env python3
"""Outlook local DB -> Obsidian Markdown extractor.

Reads email metadata from Outlook's local SQLite database and extracts full
email bodies from .olk15Message files using macOS mdimport.  Writes:
  - 00_inbox/YYYY/MM/DD/{sanitized-subject}_{record_id}.md  (one file per email)
  - 00_inbox/YYYY/MM/DD/_index.md                            (daily digest index)
  - 10_meetings/YYYY/MM/DD/calendar.md                       (calendar events)

Supports:
  - Incremental processing via record ID cursor
  - Full backfill mode for initial import of all emails
  - Conversation threading via Conversation_ConversationID
"""

from __future__ import annotations

import argparse
import logging
import re
import sqlite3
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.common.config import load_config, resolve_output_dir, setup_logging
from src.common.markdown import (
    sanitize_filename,
    write_markdown,
    yaml_frontmatter,
)
from src.common.outlook_body import extract_body
from src.common.state import get_cursor, load_state, save_state, set_cursor

logger = logging.getLogger("memoryos.outlook")

# ── Outlook SQLite queries ───────────────────────────────────────────────────

# Outlook on Mac stores email timestamps as Unix epoch (seconds since 1970-01-01 UTC).
# Calendar timestamps use a different format: 1.25-second ticks from a custom epoch.
# Empirically determined epoch: 2017-02-16 21:00:00 UTC (Unix timestamp 1487278800).
CALENDAR_EPOCH_UNIX = 1487278800
CALENDAR_TICK_SECONDS = 1.25

QUERY_MAIL = """
SELECT
    m.Record_RecordID,
    m.PathToDataFile,
    m.Message_NormalizedSubject,
    m.Message_SenderList,
    m.Message_SenderAddressList,
    m.Message_DisplayTo,
    m.Message_ToRecipientAddressList,
    m.Message_CCRecipientAddressList,
    m.Message_TimeReceived,
    m.Message_TimeSent,
    m.Message_Preview,
    m.Message_IsOutgoingMessage,
    m.Conversation_ConversationID,
    m.Record_FolderID,
    f.Folder_Name
FROM Mail m
LEFT JOIN Folders f ON m.Record_FolderID = f.Record_RecordID
WHERE m.Record_RecordID > ?
ORDER BY m.Record_RecordID ASC
LIMIT ?
"""

QUERY_MAIL_BACKFILL = """
SELECT
    m.Record_RecordID,
    m.PathToDataFile,
    m.Message_NormalizedSubject,
    m.Message_SenderList,
    m.Message_SenderAddressList,
    m.Message_DisplayTo,
    m.Message_ToRecipientAddressList,
    m.Message_CCRecipientAddressList,
    m.Message_TimeReceived,
    m.Message_TimeSent,
    m.Message_Preview,
    m.Message_IsOutgoingMessage,
    m.Conversation_ConversationID,
    m.Record_FolderID,
    f.Folder_Name
FROM Mail m
LEFT JOIN Folders f ON m.Record_FolderID = f.Record_RecordID
ORDER BY m.Record_RecordID ASC
"""

QUERY_CALENDAR = """
SELECT
    e.Record_RecordID,
    e.PathToDataFile,
    e.Calendar_StartDateUTC,
    e.Calendar_EndDateUTC,
    e.Calendar_AttendeeCount,
    e.Calendar_IsRecurring
FROM CalendarEvents e
WHERE e.Record_RecordID > ?
ORDER BY e.Calendar_StartDateUTC ASC
LIMIT ?
"""


# ── Data parsing ─────────────────────────────────────────────────────────────

def _epoch_to_datetime(ts: int | float | None) -> datetime | None:
    """Convert Unix epoch timestamp to Python datetime."""
    if ts is None or ts == 0:
        return None
    try:
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).astimezone()
    except (OSError, ValueError, OverflowError):
        return None


def _calendar_tick_to_datetime(ticks: int | float | None) -> datetime | None:
    """Convert Outlook calendar tick value to Python datetime.

    Calendar events use 1.25-second ticks from epoch ~2017-02-16 21:00 UTC.
    """
    if ticks is None or ticks == 0:
        return None
    try:
        unix_ts = CALENDAR_EPOCH_UNIX + (float(ticks) * CALENDAR_TICK_SECONDS)
        return datetime.fromtimestamp(unix_ts, tz=timezone.utc).astimezone()
    except (OSError, ValueError, OverflowError):
        return None


def _clean_folder_name(name: str | None) -> str:
    """Extract readable folder name from Outlook placeholder names."""
    if not name:
        return "Unknown"
    # Remove "Placeholder_" prefix and "_Placeholder" suffix
    name = re.sub(r"^Placeholder_", "", name)
    name = re.sub(r"_Placeholder$", "", name)
    # Convert underscores to spaces
    return name.replace("_", " ").strip() or "Unknown"


def _parse_address_list(addr_str: str | None) -> list[str]:
    """Parse Outlook recipient list string into individual entries."""
    if not addr_str:
        return []
    # May be semicolon or comma separated
    return [a.strip() for a in re.split(r"[;,]", addr_str) if a.strip()]


# ── Email markdown rendering ────────────────────────────────────────────────

def render_email_markdown(
    record_id: int,
    subject: str | None,
    sender: str | None,
    sender_addr: str | None,
    to_display: str | None,
    to_addrs: str | None,
    cc_addrs: str | None,
    time_received: datetime | None,
    time_sent: datetime | None,
    is_outgoing: bool,
    conversation_id: int | None,
    folder: str,
    body: str | None,
    preview: str | None,
) -> str:
    """Render a single email as markdown with YAML frontmatter."""
    subject = subject or "(No Subject)"
    effective_time = time_sent if is_outgoing else time_received
    date_str = effective_time.strftime("%Y-%m-%d") if effective_time else "unknown"
    time_str = effective_time.strftime("%H:%M") if effective_time else "unknown"

    to_list = _parse_address_list(to_addrs or to_display)
    cc_list = _parse_address_list(cc_addrs)

    meta = {
        "date": date_str,
        "source": "outlook-local",
        "type": "email",
        "subject": subject,
        "from": sender_addr or sender or "unknown",
        "to": to_list,
        "cc": cc_list,
        "time_sent": time_sent.isoformat() if time_sent else "",
        "time_received": time_received.isoformat() if time_received else "",
        "conversation_id": conversation_id or 0,
        "is_outgoing": is_outgoing,
        "folder": folder,
        "record_id": record_id,
    }

    parts = [yaml_frontmatter(meta), ""]
    parts.append(f"# {subject}")
    parts.append("")

    # Header block
    if is_outgoing:
        parts.append(f"**From:** {sender_addr or sender or 'You'}")
        if to_list:
            parts.append(f"**To:** {', '.join(to_list)}")
    else:
        parts.append(f"**From:** {sender or sender_addr or 'unknown'}")
        if to_list:
            parts.append(f"**To:** {', '.join(to_list)}")
    if cc_list:
        parts.append(f"**CC:** {', '.join(cc_list)}")
    parts.append(f"**Date:** {date_str} {time_str}")
    parts.append(f"**Folder:** {folder}")
    parts.append("")
    parts.append("---")
    parts.append("")

    # Body
    if body:
        parts.append(body)
    elif preview:
        parts.append(preview)
        parts.append("")
        parts.append("*[Body truncated to preview -- full text not available]*")
    else:
        parts.append("*[No body content available]*")

    return "\n".join(parts)


def render_daily_index(
    date_str: str,
    sent: list[dict[str, Any]],
    received: list[dict[str, Any]],
) -> str:
    """Render a daily email index with links to individual email files."""
    meta = {
        "date": date_str,
        "source": "outlook-local",
        "type": "email-index",
        "email_count": len(sent) + len(received),
    }
    parts = [yaml_frontmatter(meta), ""]
    parts.append(f"# Email Index -- {date_str}")
    parts.append("")

    if sent:
        parts.append(f"## Sent ({len(sent)})")
        parts.append("")
        for e in sent:
            to_str = ", ".join(e.get("to", [])[:2])
            parts.append(f"- [[{e['filename']}|{e['subject']}]] to {to_str} ({e['time']})")
        parts.append("")

    if received:
        parts.append(f"## Received ({len(received)})")
        parts.append("")
        for e in received:
            parts.append(f"- [[{e['filename']}|{e['subject']}]] from {e['from']} ({e['time']})")
        parts.append("")

    return "\n".join(parts)


def render_calendar_note(
    date_str: str,
    events: list[dict[str, Any]],
) -> str:
    """Render a daily calendar markdown note."""
    meta = {
        "date": date_str,
        "source": "outlook-local",
        "type": "calendar",
        "event_count": len(events),
    }
    parts = [yaml_frontmatter(meta), ""]
    parts.append(f"# Calendar -- {date_str}")
    parts.append("")

    for ev in events:
        start = ev.get("start", "??:??")
        end = ev.get("end", "??:??")
        parts.append(f"## {start} - {end}")
        parts.append("")
        if ev.get("attendees"):
            parts.append(f"- **Attendees:** {ev['attendees']}")
        if ev.get("recurring"):
            parts.append("- **Recurring:** Yes")
        parts.append("")

    return "\n".join(parts)


# ── Main extraction logic ───────────────────────────────────────────────────

def _is_new_outlook() -> bool:
    """Detect if Outlook is in 'New Outlook' mode (cloud-only, no local DB writes).

    Uses two signals: the macOS defaults key (authoritative when present) and
    DB staleness (fallback only when the defaults key is missing).
    """
    try:
        result = subprocess.run(
            ["defaults", "read", "com.microsoft.Outlook", "IsRunningNewOutlook"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip() == "1"
    except Exception:
        pass

    # Key missing or command failed -- fall back to DB staleness heuristic
    db = (
        Path.home()
        / "Library" / "Group Containers" / "UBF8T346G9.Office"
        / "Outlook" / "Outlook 15 Profiles" / "Main Profile"
        / "Data" / "Outlook.sqlite"
    )
    if db.is_file():
        import time as _time

        stale_seconds = _time.time() - db.stat().st_mtime
        if stale_seconds > 48 * 3600:
            return True
    return False


def run(cfg: dict[str, Any], *, dry_run: bool = False, backfill: bool = False) -> None:
    """Run the Outlook extractor."""
    if _is_new_outlook():
        logger.warning(
            "Outlook is in 'New Outlook' mode -- local SQLite DB is not updated. "
            "Email extraction will use mail_app extractor instead."
        )
        return

    db_path = cfg["outlook"]["db_path"]
    messages_dir = Path(cfg["outlook"]["messages_dir"])

    if not Path(db_path).is_file():
        logger.error("Outlook DB not found: %s", db_path)
        return

    state_path = cfg["state_file"]
    state = load_state(state_path)

    batch_size = cfg.get("outlook_settings", {}).get("batch_size", 500)
    last_mail_id = 0 if backfill else get_cursor(state, "outlook", "last_mail_id", 0)
    last_event_id = get_cursor(state, "outlook", "last_event_id", 0)

    logger.info("Querying Outlook DB from mail_id=%d (backfill=%s)", last_mail_id, backfill)

    # Open read-only
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        if backfill:
            mail_rows = conn.execute(QUERY_MAIL_BACKFILL).fetchall()
        else:
            mail_rows = conn.execute(QUERY_MAIL, (last_mail_id, batch_size)).fetchall()

        event_rows = conn.execute(QUERY_CALENDAR, (last_event_id, batch_size)).fetchall()
    finally:
        conn.close()

    if not mail_rows and not event_rows:
        logger.info("No new Outlook data")
        return

    logger.info("Processing %d emails, %d calendar events", len(mail_rows), len(event_rows))

    email_dir = resolve_output_dir(cfg, "email")
    cal_dir = resolve_output_dir(cfg, "meetings")

    # Track max IDs
    max_mail_id = last_mail_id
    # Group emails by date for index
    daily_sent: dict[str, list[dict]] = defaultdict(list)
    daily_received: dict[str, list[dict]] = defaultdict(list)

    for row in mail_rows:
        (record_id, path_to_data, subject, sender, sender_addr,
         display_to, to_addrs, cc_addrs, time_received_raw,
         time_sent_raw, preview, is_outgoing, conversation_id,
         folder_id, folder_name) = row

        max_mail_id = max(max_mail_id, record_id)

        time_received = _epoch_to_datetime(time_received_raw)
        time_sent = _epoch_to_datetime(time_sent_raw)
        folder = _clean_folder_name(folder_name)
        is_outgoing = bool(is_outgoing)

        # Determine date for filing
        effective_time = time_sent if is_outgoing else time_received
        if effective_time is None:
            effective_time = datetime.now()
        date_str = effective_time.strftime("%Y-%m-%d")
        # Nested YYYY/MM/DD path for 00_inbox
        date_path = effective_time.strftime("%Y/%m/%d")
        time_str = effective_time.strftime("%H:%M")

        # Extract full body
        body = None
        if path_to_data:
            msg_file = messages_dir.parent / path_to_data
            if msg_file.exists():
                body = extract_body(msg_file)
            else:
                logger.debug("Message file not found: %s", msg_file)

        # Render email markdown
        content = render_email_markdown(
            record_id=record_id,
            subject=subject,
            sender=sender,
            sender_addr=sender_addr,
            to_display=display_to,
            to_addrs=to_addrs,
            cc_addrs=cc_addrs,
            time_received=time_received,
            time_sent=time_sent,
            is_outgoing=is_outgoing,
            conversation_id=conversation_id,
            folder=folder,
            body=body,
            preview=preview,
        )

        # Write individual email file
        safe_subject = sanitize_filename(subject or "no-subject")
        filename = f"{safe_subject}_{record_id}"
        email_path = email_dir / date_path / f"{filename}.md"

        if dry_run:
            logger.info("DRY RUN: Would write %s (%d bytes)", email_path, len(content))
            if record_id == mail_rows[0][0]:  # Show first email
                print(f"\n{'='*60}\n{email_path}\n{'='*60}")
                print(content[:1500])
        else:
            write_markdown(email_path, content)

        # Track for index
        index_entry = {
            "filename": filename,
            "subject": subject or "(No Subject)",
            "from": sender or sender_addr or "unknown",
            "to": _parse_address_list(to_addrs or display_to),
            "time": time_str,
        }
        if is_outgoing:
            daily_sent[date_str].append(index_entry)
        else:
            daily_received[date_str].append(index_entry)

    # Write daily index files
    all_email_dates = set(daily_sent.keys()) | set(daily_received.keys())
    for date_str in sorted(all_email_dates):
        index_content = render_daily_index(
            date_str,
            daily_sent.get(date_str, []),
            daily_received.get(date_str, []),
        )
        # Convert YYYY-MM-DD to YYYY/MM/DD path
        parts = date_str.split("-")
        index_date_path = f"{parts[0]}/{parts[1]}/{parts[2]}" if len(parts) == 3 else date_str
        index_path = email_dir / index_date_path / "_index.md"
        if dry_run:
            logger.info("DRY RUN: Would write index %s", index_path)
        else:
            write_markdown(index_path, index_content)

    # Process calendar events
    max_event_id = last_event_id
    daily_events: dict[str, list[dict]] = defaultdict(list)

    for row in event_rows:
        (record_id, path_to_data, start_raw, end_raw,
         attendee_count, is_recurring) = row

        max_event_id = max(max_event_id, record_id)

        start_dt = _calendar_tick_to_datetime(start_raw)
        end_dt = _calendar_tick_to_datetime(end_raw)

        if start_dt is None:
            continue

        date_str = start_dt.strftime("%Y-%m-%d")
        daily_events[date_str].append({
            "start": start_dt.strftime("%H:%M"),
            "end": end_dt.strftime("%H:%M") if end_dt else "??:??",
            "attendees": attendee_count or 0,
            "recurring": bool(is_recurring),
        })

    for date_str, events in sorted(daily_events.items()):
        cal_content = render_calendar_note(date_str, events)
        # Nested YYYY/MM/DD path for 10_meetings
        parts = date_str.split("-")
        cal_date_path = f"{parts[0]}/{parts[1]}/{parts[2]}" if len(parts) == 3 else date_str
        cal_path = cal_dir / cal_date_path / "calendar.md"
        if dry_run:
            logger.info("DRY RUN: Would write calendar %s", cal_path)
        else:
            write_markdown(cal_path, cal_content)

    # Update cursors
    if not dry_run:
        set_cursor(state, "outlook", "last_mail_id", max_mail_id)
        set_cursor(state, "outlook", "last_event_id", max_event_id)
        save_state(state_path, state)
        logger.info("Updated Outlook cursor: mail_id=%d, event_id=%d", max_mail_id, max_event_id)

    logger.info(
        "Outlook extraction complete: %d emails, %d events across %d days",
        len(mail_rows), len(event_rows), len(all_email_dates | set(daily_events.keys())),
    )


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Outlook -> Obsidian Markdown")
    parser.add_argument("--config", help="Path to config.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Preview output without writing")
    parser.add_argument("--backfill", action="store_true",
                        help="Process ALL emails (initial import)")
    parser.add_argument("--reset", action="store_true", help="Reset cursor to re-process all data")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg)

    if args.reset:
        state = load_state(cfg["state_file"])
        set_cursor(state, "outlook", "last_mail_id", 0)
        set_cursor(state, "outlook", "last_event_id", 0)
        save_state(cfg["state_file"], state)
        logger.info("Reset Outlook cursor to 0")

    run(cfg, dry_run=args.dry_run, backfill=args.backfill or args.reset)


if __name__ == "__main__":
    main()
