"""Auto-generate AI-ready context files from the memory index.

Produces Markdown summaries in ``_context/`` that any AI tool can read directly:
  - today.md          -- today's meetings, emails, activity
  - this_week.md      -- rolling 7-day summary
  - recent_emails.md  -- last 50 emails with previews
  - upcoming.md       -- next 7 days of calendar events
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger("memoryos.memory")

_MAX_PREVIEW_CHARS = 300


def _strip_frontmatter(text: str) -> str:
    """Remove YAML frontmatter from Markdown content."""
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            return text[end + 4:].lstrip("\n")
    return text


def _preview(content: str, max_chars: int = _MAX_PREVIEW_CHARS) -> str:
    """Return a short plain-text preview of Markdown content."""
    clean = _strip_frontmatter(content)
    lines = [ln for ln in clean.splitlines() if ln.strip() and not ln.startswith("#")]
    text = " ".join(lines)
    if len(text) > max_chars:
        return text[:max_chars].rsplit(" ", 1)[0] + "..."
    return text


def _meeting_preview(content: str, max_chars: int = 4000) -> str:
    """Return a rich preview of meeting content preserving markdown structure."""
    clean = _strip_frontmatter(content)
    if not clean.strip():
        return ""
    if len(clean) <= max_chars:
        return clean.strip()
    return clean[:max_chars].rsplit("\n", 1)[0] + "\n\n... [truncated]"


def _email_preview(content: str, max_chars: int = 200) -> str:
    """Extract from/subject/body-snippet from an email markdown document."""
    clean = _strip_frontmatter(content)
    if not clean.strip():
        return ""

    from_line = ""
    body_lines: list[str] = []
    for ln in clean.splitlines():
        lower = ln.strip().lower()
        if lower.startswith("**from") or lower.startswith("from:"):
            from_line = ln.strip()
        elif ln.strip() and not ln.startswith("#") and not ln.startswith("---"):
            body_lines.append(ln.strip())

    parts: list[str] = []
    if from_line:
        parts.append(from_line)
    body_text = " ".join(body_lines)
    if body_text:
        if len(body_text) > max_chars:
            body_text = body_text[:max_chars].rsplit(" ", 1)[0] + "..."
        parts.append(body_text)
    return " | ".join(parts) if parts else ""


def _write_if_changed(path: Path, content: str) -> bool:
    """Write file only if content actually changed. Returns True if written."""
    if path.exists():
        try:
            existing = path.read_text(encoding="utf-8")
            if existing == content:
                return False
        except OSError:
            pass
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return True


def generate_context_files(
    index: Any,
    vault: Path,
    context_dir: str,
    cfg: dict[str, Any],
) -> None:
    """Generate all context files from the index into vault/context_dir/."""
    out = vault / context_dir
    now = datetime.now()

    _generate_today(index, out, now)
    _generate_this_week(index, out, now)
    _generate_recent_emails(index, out, now)
    _generate_upcoming(index, out, now)
    _generate_core(vault, out, cfg, now)

    logger.info("Context files updated in %s", out)


def _generate_today(index: Any, out: Path, now: datetime) -> None:
    """Today's meetings, emails received, and activity summary."""
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    end = now

    lines = [
        f"# Today -- {now.strftime('%A, %B %d, %Y')}",
        "",
        f"*Auto-generated at {now.strftime('%H:%M')}. This file updates every 5 minutes.*",
        "",
    ]

    meetings = index.get_by_date_range(start, end, source_type="meetings")
    lines.append(f"## Meetings ({len(meetings)})")
    lines.append("")
    if meetings:
        for doc in meetings:
            title = doc["title"]
            path = doc["path"]
            content = doc.get("content", "")

            if "calendar" in title.lower():
                lines.append(f"### Calendar — `{path}`")
                lines.append("")
                preview = _meeting_preview(content, max_chars=12000)
                if preview:
                    lines.append(preview)
            elif "audio" in title.lower():
                lines.append(f"### Transcripts — `{path}`")
                lines.append("")
                preview = _meeting_preview(content, max_chars=8000)
                if preview:
                    lines.append(preview)
            else:
                lines.append(f"- **{title}** — `{path}`")
                p = _preview(content)
                if p:
                    lines.append(f"  {p}")
    else:
        lines.append("*No meetings today.*")
    lines.append("")

    emails = index.get_by_date_range(start, end, source_type="email")
    lines.append(f"## Emails ({len(emails)})")
    lines.append("")
    if emails:
        for doc in emails[:30]:
            lines.append(f"- **{doc['title']}** — `{doc['path']}`")
            ep = _email_preview(doc.get("content", ""))
            if ep:
                lines.append(f"  {ep}")
    else:
        lines.append("*No emails today.*")
    lines.append("")

    activity = index.get_by_date_range(start, end, source_type="activity")
    lines.append(f"## Activity ({len(activity)} files)")
    lines.append("")
    if activity:
        for doc in activity:
            lines.append(f"- `{doc['path']}`")
    else:
        lines.append("*No activity captured yet today.*")
    lines.append("")

    teams = index.get_by_date_range(start, end, source_type="teams")
    if teams:
        lines.append(f"## Teams ({len(teams)} files)")
        lines.append("")
        for doc in teams:
            lines.append(f"- `{doc['path']}`")
        lines.append("")

    _write_if_changed(out / "today.md", "\n".join(lines) + "\n")


def _generate_this_week(index: Any, out: Path, now: datetime) -> None:
    """Rolling 7-day summary across all sources."""
    start = (now - timedelta(days=7)).replace(hour=0, minute=0, second=0, microsecond=0)
    end = now

    lines = [
        f"# This Week -- {start.strftime('%b %d')} to {now.strftime('%b %d, %Y')}",
        "",
        f"*Auto-generated at {now.strftime('%Y-%m-%d %H:%M')}.*",
        "",
    ]

    for source_label, source_type in [
        ("Meetings", "meetings"),
        ("Emails", "email"),
        ("Teams", "teams"),
        ("Activity", "activity"),
        ("Documents", "knowledge"),
        ("Slides", "slides"),
    ]:
        docs = index.get_by_date_range(start, end, source_type=source_type, limit=100)
        if not docs:
            continue
        lines.append(f"## {source_label} ({len(docs)})")
        lines.append("")

        by_day: dict[str, list[dict]] = {}
        for doc in docs:
            day = doc["created_at"][:10]
            by_day.setdefault(day, []).append(doc)

        for day in sorted(by_day.keys(), reverse=True):
            lines.append(f"### {day}")
            for doc in by_day[day]:
                lines.append(f"- **{doc['title']}** — `{doc['path']}`")
            lines.append("")

    _write_if_changed(out / "this_week.md", "\n".join(lines) + "\n")


def _generate_recent_emails(index: Any, out: Path, now: datetime) -> None:
    """Last 50 emails with subjects and short previews."""
    emails = index.get_recent(hours=168, source_type="email", limit=50)

    lines = [
        "# Recent Emails",
        "",
        f"*Last 50 emails as of {now.strftime('%Y-%m-%d %H:%M')}.*",
        "",
    ]

    if not emails:
        lines.append("*No recent emails.*")
    else:
        current_day = ""
        for doc in emails:
            day = doc["created_at"][:10]
            if day != current_day:
                current_day = day
                lines.append(f"## {day}")
                lines.append("")
            lines.append(f"- **{doc['title']}** — `{doc['path']}`")
            p = _preview(doc["content"], 150)
            if p:
                lines.append(f"  {p}")
        lines.append("")

    _write_if_changed(out / "recent_emails.md", "\n".join(lines) + "\n")


_RE_MEETING_HEADER = re.compile(
    r"^##\s+(?:All Day:\s*(.+)|(\d{1,2}:\d{2})\s*-\s*(\d{1,2}:\d{2}):\s*(.+))",
    re.MULTILINE,
)
_RE_ATTENDEES = re.compile(r"^\- \*\*Attendees:\*\*\s*(.+)", re.MULTILINE)


def _calendar_summary(content: str) -> str:
    """Extract a compact summary of meetings from a calendar markdown file."""
    clean = _strip_frontmatter(content)
    if not clean.strip():
        return ""

    events: list[str] = []
    for m in _RE_MEETING_HEADER.finditer(clean):
        if m.group(1):
            events.append(f"  - All Day: {m.group(1).strip()}")
        else:
            events.append(f"  - {m.group(2)}-{m.group(3)}: {m.group(4).strip()}")

    att_match = _RE_ATTENDEES.findall(clean)
    all_names: set[str] = set()
    for att_str in att_match:
        for name in att_str.split(","):
            name = name.strip()
            if name:
                all_names.add(name)

    parts: list[str] = []
    if events:
        parts.extend(events)
    if all_names:
        parts.append(f"  - Key attendees: {', '.join(sorted(all_names)[:15])}")
    return "\n".join(parts)


def _generate_upcoming(index: Any, out: Path, now: datetime) -> None:
    """Next 7 days of calendar events with meeting titles, times, and attendees."""
    start = now
    end = now + timedelta(days=7)

    meetings = index.get_by_date_range(start, end, source_type="meetings", limit=100)

    lines = [
        f"# Upcoming -- Next 7 Days",
        "",
        f"*As of {now.strftime('%Y-%m-%d %H:%M')}.*",
        "",
    ]

    if not meetings:
        lines.append("*No upcoming meetings in the index.*")
        lines.append("")
        lines.append("Note: calendar events appear here after the calendar extractor runs.")
    else:
        current_day = ""
        for doc in meetings:
            day = doc["created_at"][:10]
            if day != current_day:
                current_day = day
                lines.append(f"## {day}")
                lines.append("")
            lines.append(f"- **{doc['title']}** — `{doc['path']}`")
            content = doc.get("content", "")
            if "calendar" in doc["title"].lower() and content:
                summary = _calendar_summary(content)
                if summary:
                    lines.append(summary)
            else:
                p = _preview(content, 200)
                if p:
                    lines.append(f"  {p}")
        lines.append("")

    _write_if_changed(out / "upcoming.md", "\n".join(lines) + "\n")


def _generate_core(vault: Path, out: Path, cfg: dict[str, Any], now: datetime) -> None:
    """Concatenate pinned knowledge files into a single core.md context file."""
    memory_cfg = cfg.get("memory", {})
    pinned = memory_cfg.get("pinned_files", [])
    if not pinned:
        return

    lines = [
        "# Core Work Context",
        "",
        f"*Auto-generated at {now.strftime('%Y-%m-%d %H:%M')} from "
        f"{len(pinned)} pinned knowledge files.*",
        "",
    ]

    loaded = 0
    for rel_path in pinned:
        full = vault / rel_path
        if not full.is_file():
            continue
        try:
            raw = full.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        content = _strip_frontmatter(raw)
        if not content.strip():
            continue
        lines.append(content.strip())
        lines.append("")
        lines.append("---")
        lines.append("")
        loaded += 1

    if loaded == 0:
        return

    lines[2] = (
        f"*Auto-generated at {now.strftime('%Y-%m-%d %H:%M')} from "
        f"{loaded} pinned knowledge files.*"
    )

    _write_if_changed(out / "core.md", "\n".join(lines) + "\n")
