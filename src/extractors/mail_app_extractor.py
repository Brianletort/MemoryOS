#!/usr/bin/env python3
"""macOS Mail.app -> Obsidian Markdown email extractor.

Queries Mail.app via AppleScript (osascript) to extract email metadata and
body text.  Fully client-side -- no Graph API or Azure AD consent needed.

Requires Mail.app to be configured with the Exchange account via
System Settings > Internet Accounts.

Output:
  - 00_inbox/YYYY/MM/DD/{sanitized-subject}_{id}.md  (per email)
  - 00_inbox/YYYY/MM/DD/_index.md                     (daily digest)
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
from src.common.markdown import sanitize_filename, write_markdown, yaml_frontmatter
from src.common.state import get_cursor, load_state, save_state, set_cursor

logger = logging.getLogger("memoryos.mail_app")

FIELD_SEP = "\x1e"  # ASCII record separator -- won't appear in email fields
RECORD_SEP = "\x1f"  # ASCII unit separator -- delimits messages

MAX_TIMEOUTS_BEFORE_RESTART: int = 3
_HEAL_STATE_FILE = Path(__file__).resolve().parent.parent.parent / "config" / "heal_mail.json"


# ── Self-healing helpers ─────────────────────────────────────────────────────

def _load_heal_state() -> dict[str, Any]:
    if _HEAL_STATE_FILE.is_file():
        try:
            import json
            return json.loads(_HEAL_STATE_FILE.read_text())
        except Exception:
            pass
    return {}


def _save_heal_state(state: dict[str, Any]) -> None:
    import json
    _HEAL_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = _HEAL_STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, default=str))
    tmp.rename(_HEAL_STATE_FILE)


def _get_timeout_count() -> int:
    return _load_heal_state().get("consecutive_timeouts", 0)


def _set_timeout_count(n: int) -> None:
    hs = _load_heal_state()
    hs["consecutive_timeouts"] = n
    hs["last_updated"] = datetime.now().isoformat()
    _save_heal_state(hs)


def _is_mail_app_running() -> bool:
    result = subprocess.run(
        ["pgrep", "-f", "Mail.app/Contents/MacOS/Mail$"],
        capture_output=True,
    )
    return result.returncode == 0


def _ensure_mail_app() -> bool:
    """Launch Mail.app if it is not running. Returns True if a launch was needed."""
    if _is_mail_app_running():
        return False
    logger.warning("Mail.app is not running — launching it")
    subprocess.run(["open", "-g", "-j", "-a", "Mail"], timeout=10, capture_output=True)
    _time.sleep(20)
    logger.info("Mail.app launched")
    return True


def _restart_mail_app(reason: str) -> None:
    """Force-quit and reopen Mail.app to clear a hung state."""
    count = _get_timeout_count()
    logger.warning("Restarting Mail.app (%s, %d consecutive timeouts)", reason, count)
    subprocess.run(
        ["pkill", "-f", "Mail.app/Contents/MacOS/Mail"],
        timeout=10, capture_output=True,
    )
    _time.sleep(5)
    subprocess.run(["open", "-g", "-j", "-a", "Mail"], timeout=10, capture_output=True)
    _time.sleep(20)
    _set_timeout_count(0)
    logger.info("Mail.app restarted — next extraction cycle should succeed")


# ── AppleScript helpers ──────────────────────────────────────────────────────

def _run_osascript(script: str, *, timeout: int = 120) -> str:
    """Execute an AppleScript via osascript and return stdout."""
    wrapper = Path(__file__).resolve().parent.parent.parent / "scripts" / "osascript_wrapper.sh"
    cmd = [str(wrapper), "-e", script] if wrapper.is_file() else ["osascript", "-e", script]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        count = _get_timeout_count() + 1
        _set_timeout_count(count)
        logger.warning(
            "AppleScript timed out after %ds [consecutive_timeouts=%d/%d]",
            timeout, count, MAX_TIMEOUTS_BEFORE_RESTART,
        )
        if count >= MAX_TIMEOUTS_BEFORE_RESTART:
            _restart_mail_app("subprocess timeout")
        return ""
    if result.returncode != 0:
        stderr = result.stderr.strip()
        if "execution error" in stderr:
            if "-1712" in stderr:
                count = _get_timeout_count() + 1
                _set_timeout_count(count)
                logger.warning(
                    "Mail.app AppleEvent timed out (busy syncing) "
                    "[consecutive_timeouts=%d/%d]",
                    count, MAX_TIMEOUTS_BEFORE_RESTART,
                )
                if count >= MAX_TIMEOUTS_BEFORE_RESTART:
                    _restart_mail_app("AppleEvent -1712")
            elif "-600" in stderr:
                logger.warning("Mail.app not running — auto-launching")
                _ensure_mail_app()
            else:
                logger.warning("AppleScript error: %s", stderr)
            return ""
        raise RuntimeError(f"osascript failed (rc={result.returncode}): {stderr}")
    _set_timeout_count(0)
    return result.stdout.strip()


def _applescript_date_literal(dt: datetime) -> str:
    """Format a datetime as an AppleScript date literal string.

    AppleScript ``date`` accepts natural-language strings that the system
    interprets in the current locale.  The long US-English form works
    reliably on macOS regardless of locale.
    """
    return dt.strftime("%A, %B %d, %Y at %I:%M:%S %p")


# ── Mail.app data extraction ────────────────────────────────────────────────

def _mailbox_source_expression(mailbox: str) -> str:
    """Return the AppleScript expression for a given mailbox name.

    ``Inbox`` uses the special ``inbox`` keyword (unified across accounts).
    Other mailboxes are resolved per-account by name.
    """
    if mailbox.lower() == "inbox":
        return "inbox"
    # Account-specific: iterate accounts (most users have one Exchange acct)
    return f'mailbox "{mailbox}" of acct'


def _build_metadata_script(
    mailbox: str,
    *,
    since: datetime | None = None,
    batch_size: int = 200,
) -> str:
    """Build AppleScript to extract message metadata (no body -- fast).

    Instead of filtering with ``whose date received > ...`` (which scans the
    entire mailbox and times out on large Exchange accounts), we iterate the
    first N messages (newest first) and stop when we hit one older than the
    cutoff.  This is dramatically faster.
    """
    date_guard = ""
    if since:
        date_guard = (
            f'if recvDate < date "{_applescript_date_literal(since)}" '
            f"then exit repeat"
        )

    extract_loop = f'''
    set batchLimit to {batch_size}
    set msgCount to count of msgs
    if msgCount > batchLimit then set msgCount to batchLimit

    repeat with i from 1 to msgCount
        set m to item i of msgs
        set recvDate to date received of m

        {date_guard}

        set mid to id of m
        set subj to subject of m
        set fromStr to sender of m
        set recvStr to recvDate as string

        set toList to ""
        repeat with r in (to recipients of m)
            if toList is not "" then set toList to toList & "; "
            set toList to toList & (address of r)
        end repeat

        set ccList to ""
        try
            repeat with r in (cc recipients of m)
                if ccList is not "" then set ccList to ccList & "; "
                set ccList to ccList & (address of r)
            end repeat
        end try

        set output to output & mid & fieldSep & subj & fieldSep & fromStr & fieldSep & toList & fieldSep & ccList & fieldSep & recvStr & recSep
    end repeat
'''

    if mailbox.lower() == "inbox":
        return f'''
tell application "Mail"
    set fieldSep to ASCII character 30
    set recSep to ASCII character 31
    set output to ""
    set msgs to (every message of inbox)
{extract_loop}
    return output
end tell
'''
    else:
        return f'''
tell application "Mail"
    set fieldSep to ASCII character 30
    set recSep to ASCII character 31
    set output to ""
    repeat with acct in every account
        try
            set msgs to (every message of mailbox "{mailbox}" of acct)
{extract_loop}
        end try
    end repeat
    return output
end tell
'''


def fetch_messages(
    mailbox: str,
    *,
    since: datetime | None = None,
    batch_size: int = 200,
) -> list[dict[str, str]]:
    """Fetch messages from a Mail.app mailbox via AppleScript.

    Extracts metadata only (subject, sender, recipients, date).  Body text
    is omitted because Exchange accounts in Mail.app trigger slow per-message
    server fetches that cause AppleScript timeouts.

    Returns a list of dicts with keys: id, subject, sender, to, cc,
    date_received, body (always empty).
    """
    script = _build_metadata_script(
        mailbox, since=since, batch_size=batch_size,
    )
    raw = _run_osascript(script, timeout=180)
    if not raw:
        return []

    messages: list[dict[str, str]] = []
    for record in raw.split(RECORD_SEP):
        record = record.strip()
        if not record:
            continue
        fields = record.split(FIELD_SEP)
        if len(fields) < 6:
            logger.debug("Skipping malformed record with %d fields", len(fields))
            continue
        messages.append({
            "id": fields[0].strip(),
            "subject": fields[1].strip(),
            "sender": fields[2].strip(),
            "to": fields[3].strip(),
            "cc": fields[4].strip(),
            "date_received": fields[5].strip(),
            "body": "",
        })

    return messages


def _parse_applescript_date(date_str: str) -> datetime | None:
    """Parse the natural-language date string returned by AppleScript."""
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


def _parse_sender(sender_str: str) -> tuple[str, str]:
    """Extract name and email from 'Name <email>' format."""
    match = re.match(r"^(.+?)\s*<(.+?)>$", sender_str)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    if "@" in sender_str:
        return "", sender_str.strip()
    return sender_str.strip(), ""


# ── Markdown rendering ───────────────────────────────────────────────────────

def render_email_markdown(
    msg: dict[str, str],
    *,
    is_outgoing: bool,
    effective_time: datetime,
) -> str:
    """Render a single email as Markdown with YAML frontmatter."""
    subject = msg["subject"] or "(No Subject)"
    sender_name, sender_addr = _parse_sender(msg["sender"])
    to_list = [a.strip() for a in msg["to"].split(";") if a.strip()]
    cc_list = [a.strip() for a in msg["cc"].split(";") if a.strip()]

    date_str = effective_time.strftime("%Y-%m-%d")
    time_str = effective_time.strftime("%H:%M")

    meta: dict[str, Any] = {
        "date": date_str,
        "source": "mail-app",
        "type": "email",
        "subject": subject,
        "from": sender_addr or sender_name or "unknown",
        "to": to_list,
        "cc": cc_list,
        "time_received": msg["date_received"],
        "is_outgoing": is_outgoing,
        "mail_app_id": msg["id"],
    }

    parts = [yaml_frontmatter(meta), ""]
    parts.append(f"# {subject}")
    parts.append("")

    if is_outgoing:
        parts.append("**From:** You")
    else:
        from_display = msg["sender"] or "unknown"
        parts.append(f"**From:** {from_display}")
    if to_list:
        parts.append(f"**To:** {', '.join(to_list)}")
    if cc_list:
        parts.append(f"**CC:** {', '.join(cc_list)}")
    parts.append(f"**Date:** {date_str} {time_str}")
    parts.append("")
    parts.append("---")
    parts.append("")

    body = msg.get("body", "").strip()
    if body:
        body = re.sub(r"\n{3,}", "\n\n", body)
        parts.append(body)
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
        "source": "mail-app",
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
            parts.append(
                f"- [[{e['filename']}|{e['subject']}]] to {to_str} ({e['time']})"
            )
        parts.append("")

    if received:
        parts.append(f"## Received ({len(received)})")
        parts.append("")
        for e in received:
            parts.append(
                f"- [[{e['filename']}|{e['subject']}]] from {e['from']} ({e['time']})"
            )
        parts.append("")

    return "\n".join(parts)


# ── Main extraction logic ───────────────────────────────────────────────────

SENT_MAILBOXES = {"sent items", "sent", "sent messages"}


def run(
    cfg: dict[str, Any],
    *,
    dry_run: bool = False,
    days_back: int | None = None,
) -> None:
    """Run the Mail.app email extractor."""
    _ensure_mail_app()

    state_path = cfg["state_file"]
    state = load_state(state_path)

    email_dir = resolve_output_dir(cfg, "email")

    mail_cfg = cfg.get("mail_app", {})
    mailboxes: list[str] = mail_cfg.get("mailboxes", ["Inbox", "Sent Items"])
    batch_size: int = mail_cfg.get("batch_size", 200)

    last_date_str = get_cursor(state, "mail_app", "last_message_date", "")
    if days_back is not None:
        since = datetime.now() - timedelta(days=days_back)
    elif last_date_str:
        since = _parse_applescript_date(last_date_str)
        if since:
            since = since - timedelta(minutes=5)
    else:
        since = datetime.now() - timedelta(days=2)

    logger.info(
        "Fetching from %s since %s",
        ", ".join(mailboxes),
        since.strftime("%Y-%m-%d %H:%M") if since else "all time",
    )

    daily_sent: dict[str, list[dict]] = defaultdict(list)
    daily_received: dict[str, list[dict]] = defaultdict(list)
    max_date: datetime | None = None
    processed = 0
    seen_ids: set[str] = set()

    for mailbox in mailboxes:
        is_outgoing = mailbox.lower() in SENT_MAILBOXES
        logger.info("Querying mailbox: %s", mailbox)

        messages = fetch_messages(mailbox, since=since, batch_size=batch_size)
        logger.info("  %d messages from %s", len(messages), mailbox)

        for msg in messages:
            msg_id = msg["id"]
            if msg_id in seen_ids:
                continue
            seen_ids.add(msg_id)

            effective_time = _parse_applescript_date(msg["date_received"])
            if effective_time is None:
                effective_time = datetime.now()

            if max_date is None or effective_time > max_date:
                max_date = effective_time

            date_str = effective_time.strftime("%Y-%m-%d")
            date_path = effective_time.strftime("%Y/%m/%d")
            time_str = effective_time.strftime("%H:%M")

            content = render_email_markdown(
                msg, is_outgoing=is_outgoing, effective_time=effective_time,
            )

            safe_subject = sanitize_filename(msg["subject"] or "no-subject")
            filename = f"{safe_subject}_{msg_id}"
            email_path = email_dir / date_path / f"{filename}.md"

            if dry_run:
                logger.info(
                    "DRY RUN: Would write %s (%d bytes)", email_path, len(content),
                )
                if processed == 0:
                    print(f"\n{'=' * 60}\n{email_path}\n{'=' * 60}")
                    print(content[:1500])
            else:
                write_markdown(email_path, content)

            _, sender_email = _parse_sender(msg["sender"])
            index_entry = {
                "filename": filename,
                "subject": msg["subject"] or "(No Subject)",
                "from": sender_email or msg["sender"],
                "to": [a.strip() for a in msg["to"].split(";") if a.strip()][:2],
                "time": time_str,
            }
            if is_outgoing:
                daily_sent[date_str].append(index_entry)
            else:
                daily_received[date_str].append(index_entry)

            processed += 1

    all_dates = set(daily_sent.keys()) | set(daily_received.keys())
    for date_str in sorted(all_dates):
        index_content = render_daily_index(
            date_str,
            daily_sent.get(date_str, []),
            daily_received.get(date_str, []),
        )
        parts = date_str.split("-")
        index_date_path = (
            f"{parts[0]}/{parts[1]}/{parts[2]}" if len(parts) == 3 else date_str
        )
        index_path = email_dir / index_date_path / "_index.md"
        if dry_run:
            logger.info("DRY RUN: Would write index %s", index_path)
        else:
            write_markdown(index_path, index_content)

    if not dry_run and max_date:
        set_cursor(
            state, "mail_app", "last_message_date",
            _applescript_date_literal(max_date),
        )
        save_state(state_path, state)
        logger.info("Updated mail_app cursor: %s", max_date)

    logger.info(
        "Mail.app extraction complete: %d processed, %d days",
        processed, len(all_dates),
    )


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mail.app -> Obsidian Markdown (Email)",
    )
    parser.add_argument("--config", help="Path to config.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    parser.add_argument(
        "--days-back", type=int, default=None,
        help="Fetch emails from N days ago (overrides cursor)",
    )
    parser.add_argument("--reset", action="store_true", help="Reset cursor")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg)

    if args.reset:
        state = load_state(cfg["state_file"])
        set_cursor(state, "mail_app", "last_message_date", "")
        save_state(cfg["state_file"], state)
        logger.info("Reset mail_app cursor")

    run(cfg, dry_run=args.dry_run, days_back=args.days_back)


if __name__ == "__main__":
    main()
