#!/usr/bin/env python3
"""Screenpipe -> Obsidian Markdown extractor.

Reads screen OCR and audio transcriptions from Screenpipe's SQLite DB,
deduplicates near-identical frames, and writes to multiple Obsidian folders:
  - 85_activity/YYYY/MM/DD/daily.md  (full activity timeline: all OCR + audio)
  - 20_teams-chat/YYYY/MM/DD/teams.md (Teams-specific content + meeting audio)
  - 10_meetings/YYYY/MM/DD/audio.md   (meeting audio transcriptions)

The 85_activity folder is the catch-all timeline of everything seen and heard.
Audio transcriptions include speaker names when available.

Privacy / noise filtering (configured in config.yaml -> privacy):
  - Privacy-mode flag file: if present, ALL audio is filtered out
  - Minimum word count: skip tiny fragments (TV blips, single words)
  - Work-app correlation: keep audio only when a work app was on-screen nearby
  - Work hours: optionally restrict to a time window
  Filtered audio is preserved in a collapsible section, never permanently lost.
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from bisect import bisect_left, bisect_right
from collections import defaultdict
from datetime import datetime, time as dt_time, timezone
from pathlib import Path
from typing import Any

# Allow running as a script from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.common.config import load_config, resolve_output_dir, setup_logging
from src.common.markdown import (
    append_markdown,
    clean_ocr_text,
    text_similarity,
    write_markdown,
    yaml_frontmatter,
)
from src.common.state import get_cursor, load_state, save_state, set_cursor

logger = logging.getLogger("memoryos.screenpipe")


# ── SQL queries ──────────────────────────────────────────────────────────────

QUERY_OCR = """
SELECT f.id AS frame_id,
       f.timestamp,
       COALESCE(f.app_name, '') AS app_name,
       COALESCE(f.window_name, '') AS window_name,
       COALESCE(f.browser_url, '') AS browser_url,
       o.text,
       COALESCE(o.focused, 0) AS focused
FROM frames f
JOIN ocr_text o ON o.frame_id = f.id
WHERE f.id > ?
ORDER BY f.timestamp ASC
"""

QUERY_AUDIO = """
SELECT at.id,
       at.timestamp,
       at.transcription,
       COALESCE(at.device, '') AS device,
       COALESCE(at.is_input_device, 1) AS is_input_device,
       at.speaker_id,
       s.name AS speaker_name
FROM audio_transcriptions at
LEFT JOIN speakers s ON at.speaker_id = s.id
WHERE at.id > ?
ORDER BY at.timestamp ASC
"""


# ── Data structures ──────────────────────────────────────────────────────────

class OcrEntry:
    __slots__ = ("frame_id", "timestamp", "app_name", "window_name",
                 "browser_url", "text", "focused")

    def __init__(self, row: tuple) -> None:
        (self.frame_id, ts, self.app_name, self.window_name,
         self.browser_url, self.text, self.focused) = row
        self.timestamp = _parse_ts(ts)


class AudioEntry:
    __slots__ = ("audio_id", "timestamp", "transcription", "device",
                 "is_input_device", "speaker_id", "speaker_name")

    def __init__(self, row: tuple) -> None:
        (self.audio_id, ts, self.transcription, self.device,
         self.is_input_device, self.speaker_id, self.speaker_name) = row
        self.timestamp = _parse_ts(ts)


def _parse_ts(ts_str: str) -> datetime:
    """Parse ISO 8601 timestamp from Screenpipe (may have tz offset)."""
    from dateutil import parser as dtparser
    dt = dtparser.isoparse(ts_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone()  # Convert to local timezone


# ── Deduplication ────────────────────────────────────────────────────────────

def deduplicate_ocr(entries: list[OcrEntry], threshold: float = 0.85) -> list[OcrEntry]:
    """Remove near-duplicate consecutive OCR entries for the same app/window."""
    if not entries:
        return entries

    result: list[OcrEntry] = [entries[0]]
    for entry in entries[1:]:
        prev = result[-1]
        same_context = (entry.app_name == prev.app_name
                        and entry.window_name == prev.window_name)
        if same_context and text_similarity(entry.text, prev.text) >= threshold:
            continue  # Skip duplicate
        result.append(entry)

    logger.info("Deduplication: %d -> %d OCR entries", len(entries), len(result))
    return result


# ── Audio noise filtering ─────────────────────────────────────────────────

class AudioFilterResult:
    """Result of filtering audio entries into kept vs. filtered buckets."""
    __slots__ = ("kept", "filtered", "stats")

    def __init__(
        self,
        kept: list[AudioEntry],
        filtered: list[AudioEntry],
        stats: dict[str, int],
    ) -> None:
        self.kept = kept
        self.filtered = filtered
        self.stats = stats


def _privacy_mode_active(cfg: dict[str, Any]) -> bool:
    """Return True if the privacy-mode flag file exists."""
    flag = cfg.get("privacy", {}).get("flag_file", "")
    return bool(flag) and Path(flag).exists()


def _parse_time(s: str) -> dt_time:
    """Parse 'HH:MM' into a datetime.time."""
    h, m = s.split(":")
    return dt_time(int(h), int(m))


def _build_work_app_timeline(
    ocr_entries: list[OcrEntry],
    work_apps: set[str],
) -> list[float]:
    """Return sorted list of POSIX timestamps where a work app was on-screen.

    Used for fast binary-search lookups when correlating audio with screen context.
    """
    timestamps: list[float] = []
    for e in ocr_entries:
        if e.app_name in work_apps:
            timestamps.append(e.timestamp.timestamp())
    timestamps.sort()
    return timestamps


def _work_app_nearby(
    audio_ts: float,
    work_timeline: list[float],
    window_seconds: int = 120,
) -> bool:
    """Check if any work-app OCR entry falls within +-window_seconds of audio_ts."""
    if not work_timeline:
        return False
    lo = bisect_left(work_timeline, audio_ts - window_seconds)
    hi = bisect_right(work_timeline, audio_ts + window_seconds)
    return lo < hi


def filter_audio_entries(
    audio_entries: list[AudioEntry],
    ocr_entries: list[OcrEntry],
    cfg: dict[str, Any],
) -> AudioFilterResult:
    """Apply privacy and noise filters to audio entries.

    Returns AudioFilterResult with kept/filtered lists and statistics.
    """
    privacy_cfg = cfg.get("privacy", {})
    filter_cfg = privacy_cfg.get("audio_filter", {})

    # If privacy mode is active, filter ALL audio
    if _privacy_mode_active(cfg):
        logger.info("Privacy mode active — filtering all %d audio entries", len(audio_entries))
        return AudioFilterResult(
            kept=[],
            filtered=audio_entries,
            stats={"total": len(audio_entries), "privacy_mode": len(audio_entries)},
        )

    min_words = filter_cfg.get("min_words", 5)
    work_hours_only = filter_cfg.get("work_hours_only", False)
    work_start = _parse_time(filter_cfg.get("work_hours_start", "07:00"))
    work_end = _parse_time(filter_cfg.get("work_hours_end", "19:00"))

    work_apps = set(privacy_cfg.get("work_apps", []))
    work_timeline = _build_work_app_timeline(ocr_entries, work_apps) if work_apps else []

    kept: list[AudioEntry] = []
    filtered: list[AudioEntry] = []
    stats: dict[str, int] = {
        "total": len(audio_entries),
        "too_short": 0,
        "outside_hours": 0,
        "no_work_context": 0,
    }

    for ae in audio_entries:
        text = ae.transcription.strip()
        word_count = len(text.split())

        # Filter 1: minimum word count
        if word_count < min_words:
            stats["too_short"] += 1
            filtered.append(ae)
            continue

        # Filter 2: work hours
        if work_hours_only:
            local_time = ae.timestamp.time()
            if not (work_start <= local_time <= work_end):
                stats["outside_hours"] += 1
                filtered.append(ae)
                continue

        # Filter 3: work-app correlation (only if we have work app data)
        if work_apps and work_timeline:
            if not _work_app_nearby(ae.timestamp.timestamp(), work_timeline):
                stats["no_work_context"] += 1
                filtered.append(ae)
                continue

        kept.append(ae)

    logger.info(
        "Audio filter: %d total -> %d kept, %d filtered "
        "(short=%d, hours=%d, no_context=%d)",
        stats["total"], len(kept), len(filtered),
        stats["too_short"], stats["outside_hours"], stats["no_work_context"],
    )
    return AudioFilterResult(kept=kept, filtered=filtered, stats=stats)


# ── Grouping ─────────────────────────────────────────────────────────────────

def group_by_date_hour_app(
    entries: list[OcrEntry],
) -> dict[str, dict[str, dict[str, list[OcrEntry]]]]:
    """Group OCR entries into date -> hour -> app_name -> entries."""
    grouped: dict[str, dict[str, dict[str, list[OcrEntry]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for e in entries:
        date_key = e.timestamp.strftime("%Y-%m-%d")
        hour_key = e.timestamp.strftime("%H:00 -- %H:59")
        app_key = e.app_name or "Unknown"
        if e.window_name:
            app_key = f"{app_key} - {e.window_name}"
        grouped[date_key][hour_key][app_key].append(e)
    return grouped


def group_audio_by_date(
    entries: list[AudioEntry],
) -> dict[str, list[AudioEntry]]:
    """Group audio entries by date."""
    grouped: dict[str, list[AudioEntry]] = defaultdict(list)
    for e in entries:
        date_key = e.timestamp.strftime("%Y-%m-%d")
        grouped[date_key].append(e)
    return grouped


# ── Markdown rendering ───────────────────────────────────────────────────────

def render_daily_note(
    date_str: str,
    ocr_by_hour_app: dict[str, dict[str, list[OcrEntry]]],
    audio_entries: list[AudioEntry],
    apps_used: set[str],
    filtered_audio: list[AudioEntry] | None = None,
) -> str:
    """Render a full daily activity note for 85_activity/.

    Filtered audio (noise) is placed in a collapsible <details> section at the
    bottom of the note so nothing is permanently lost.
    """
    meta = {
        "date": date_str,
        "source": "screenpipe",
        "type": "daily-activity",
        "last_updated": datetime.now().isoformat(timespec="seconds"),
        "apps_used": sorted(apps_used),
    }
    parts = [yaml_frontmatter(meta), "", f"# {date_str} Activity Log", ""]

    for hour_key in sorted(ocr_by_hour_app.keys()):
        parts.append(f"## {hour_key}")
        parts.append("")
        apps = ocr_by_hour_app[hour_key]
        for app_key in sorted(apps.keys()):
            entries = apps[app_key]
            parts.append(f"### {app_key}")
            parts.append("")
            for e in entries:
                cleaned = clean_ocr_text(e.text)
                if cleaned:
                    for line in cleaned.split("\n"):
                        parts.append(f"> {line}")
                    parts.append("")
            parts.append("")

    if audio_entries:
        parts.append("---")
        parts.append("")
        parts.append("## Audio Transcriptions")
        parts.append("")
        for ae in audio_entries:
            time_str = ae.timestamp.strftime("%H:%M")
            speaker = _format_speaker(ae)
            parts.append(f"**{speaker} ({time_str}):** {ae.transcription.strip()}")
            parts.append("")

    if filtered_audio:
        parts.append("---")
        parts.append("")
        parts.append("<details>")
        parts.append(f"<summary>Filtered audio ({len(filtered_audio)} entries — background/noise)</summary>")
        parts.append("")
        for ae in filtered_audio:
            time_str = ae.timestamp.strftime("%H:%M")
            speaker = _format_speaker(ae)
            parts.append(f"*{speaker} ({time_str}):* {ae.transcription.strip()}")
            parts.append("")
        parts.append("</details>")
        parts.append("")

    return "\n".join(parts)


def render_meeting_audio_note(
    date_str: str,
    audio_entries: list[AudioEntry],
    filtered_audio: list[AudioEntry] | None = None,
) -> str:
    """Render meeting audio transcriptions for 10_meetings/."""
    meta = {
        "date": date_str,
        "source": "screenpipe-audio",
        "type": "meeting-transcript",
        "last_updated": datetime.now().isoformat(timespec="seconds"),
    }
    parts = [yaml_frontmatter(meta), "", f"# Meeting Audio -- {date_str}", ""]

    for ae in audio_entries:
        time_str = ae.timestamp.strftime("%H:%M")
        speaker = _format_speaker(ae)
        parts.append(f"**{speaker} ({time_str}):** {ae.transcription.strip()}")
        parts.append("")

    if filtered_audio:
        parts.append("---")
        parts.append("")
        parts.append("<details>")
        parts.append(f"<summary>Filtered audio ({len(filtered_audio)} entries — background/noise)</summary>")
        parts.append("")
        for ae in filtered_audio:
            time_str = ae.timestamp.strftime("%H:%M")
            speaker = _format_speaker(ae)
            parts.append(f"*{speaker} ({time_str}):* {ae.transcription.strip()}")
            parts.append("")
        parts.append("</details>")
        parts.append("")

    return "\n".join(parts)


def render_teams_note(
    date_str: str,
    ocr_entries: list[OcrEntry],
    audio_entries: list[AudioEntry],
) -> str:
    """Render a Teams-specific daily note."""
    meta = {
        "date": date_str,
        "source": "screenpipe-teams",
        "type": "teams-capture",
        "last_updated": datetime.now().isoformat(timespec="seconds"),
    }
    parts = [yaml_frontmatter(meta), "", f"# Teams Activity -- {date_str}", ""]

    # Group OCR by window (which contains meeting/chat name)
    by_window: dict[str, list[OcrEntry]] = defaultdict(list)
    for e in ocr_entries:
        window = e.window_name or "Teams"
        by_window[window].append(e)

    for window, entries in by_window.items():
        time_range = f"{entries[0].timestamp.strftime('%H:%M')}"
        if len(entries) > 1:
            time_range += f" - {entries[-1].timestamp.strftime('%H:%M')}"
        parts.append(f"## {time_range} | {window}")
        parts.append("")
        parts.append("### Screen Content")
        parts.append("")
        for e in entries:
            cleaned = clean_ocr_text(e.text)
            if cleaned:
                for line in cleaned.split("\n"):
                    parts.append(f"> {line}")
                parts.append("")

    # Audio during Teams usage
    if audio_entries:
        parts.append("### Meeting Audio")
        parts.append("")
        for ae in audio_entries:
            time_str = ae.timestamp.strftime("%H:%M")
            speaker = _format_speaker(ae)
            parts.append(f"> **{speaker} ({time_str}):** {ae.transcription.strip()}")
            parts.append("")

    return "\n".join(parts)


def _format_speaker(ae: AudioEntry) -> str:
    """Format speaker label for audio entry."""
    if ae.speaker_name:
        return ae.speaker_name
    if ae.is_input_device:
        return "You"
    return f"Speaker {ae.speaker_id or '?'}"


# ── Main extraction logic ───────────────────────────────────────────────────

def _date_to_path(date_str: str) -> str:
    """Convert YYYY-MM-DD to YYYY/MM/DD nested path."""
    parts = date_str.split("-")
    return f"{parts[0]}/{parts[1]}/{parts[2]}" if len(parts) == 3 else date_str


def run(cfg: dict[str, Any], *, dry_run: bool = False) -> None:
    """Run the Screenpipe extractor.

    Writes to three locations:
      - 85_activity/YYYY/MM/DD/daily.md  (full timeline: all OCR + all audio)
      - 20_teams-chat/YYYY/MM/DD/teams.md (Teams OCR + meeting audio)
      - 10_meetings/YYYY/MM/DD/audio.md   (all audio transcriptions)
    """
    db_path = cfg["screenpipe"]["db_path"]
    if not Path(db_path).is_file():
        logger.error("Screenpipe DB not found: %s", db_path)
        return

    state_path = cfg["state_file"]
    state = load_state(state_path)

    last_frame_id = get_cursor(state, "screenpipe", "last_frame_id", 0)
    last_audio_id = get_cursor(state, "screenpipe", "last_audio_id", 0)

    logger.info("Querying Screenpipe DB from frame_id=%d, audio_id=%d", last_frame_id, last_audio_id)

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        ocr_rows = conn.execute(QUERY_OCR, (last_frame_id,)).fetchall()
        audio_rows = conn.execute(QUERY_AUDIO, (last_audio_id,)).fetchall()
    finally:
        conn.close()

    if not ocr_rows and not audio_rows:
        logger.info("No new Screenpipe data")
        return

    logger.info("Found %d new OCR entries, %d new audio entries", len(ocr_rows), len(audio_rows))

    # Parse
    ocr_entries = [OcrEntry(r) for r in ocr_rows]
    audio_entries = [AudioEntry(r) for r in audio_rows]

    # Dedup OCR
    threshold = cfg.get("screenpipe_settings", {}).get("dedup_threshold", 0.85)
    ocr_entries = deduplicate_ocr(ocr_entries, threshold)

    # ── Audio noise filtering ──
    filter_result = filter_audio_entries(audio_entries, ocr_entries, cfg)
    audio_kept = filter_result.kept
    audio_filtered = filter_result.filtered

    # Separate Teams OCR
    teams_names = set(cfg.get("screenpipe_settings", {}).get("teams_app_names", ["Microsoft Teams"]))
    teams_ocr = [e for e in ocr_entries if e.app_name in teams_names]

    # Group all data
    ocr_grouped = group_by_date_hour_app(ocr_entries)
    audio_kept_grouped = group_audio_by_date(audio_kept)
    audio_filtered_grouped = group_audio_by_date(audio_filtered)
    teams_by_date: dict[str, list[OcrEntry]] = defaultdict(list)
    for e in teams_ocr:
        teams_by_date[e.timestamp.strftime("%Y-%m-%d")].append(e)

    # Collect all dates across OCR and audio (both kept and filtered)
    all_dates = (
        set(ocr_grouped.keys())
        | set(audio_kept_grouped.keys())
        | set(audio_filtered_grouped.keys())
    )

    # Track max IDs for cursor update
    max_frame_id = max((e.frame_id for e in [OcrEntry(r) for r in ocr_rows]), default=last_frame_id)
    max_audio_id = max((e.audio_id for e in [AudioEntry(r) for r in audio_rows]), default=last_audio_id)

    activity_dir = resolve_output_dir(cfg, "activity")
    teams_dir = resolve_output_dir(cfg, "teams")
    meetings_dir = resolve_output_dir(cfg, "meetings")

    # 1. Write FULL daily activity notes -> 85_activity/YYYY/MM/DD/daily.md
    for date_str in sorted(all_dates):
        apps = set()
        for hour_apps in ocr_grouped.get(date_str, {}).values():
            for app_key in hour_apps:
                apps.add(app_key.split(" - ")[0])

        content = render_daily_note(
            date_str,
            ocr_grouped.get(date_str, {}),
            audio_kept_grouped.get(date_str, []),
            apps,
            filtered_audio=audio_filtered_grouped.get(date_str, []),
        )
        date_path = _date_to_path(date_str)
        daily_path = activity_dir / date_path / "daily.md"

        if dry_run:
            logger.info("DRY RUN: Would write %d bytes to %s", len(content), daily_path)
        else:
            write_markdown(daily_path, content)
            logger.info("Wrote daily activity: %s", daily_path)

    # 2. Write Teams notes -> 20_teams-chat/YYYY/MM/DD/teams.md
    for date_str in sorted(teams_by_date.keys()):
        teams_audio = [a for a in audio_kept_grouped.get(date_str, [])
                       if not a.is_input_device]
        teams_content = render_teams_note(
            date_str, teams_by_date[date_str], teams_audio,
        )
        teams_path = teams_dir / _date_to_path(date_str) / "teams.md"
        if dry_run:
            logger.info("DRY RUN: Would write %d bytes to %s", len(teams_content), teams_path)
        else:
            write_markdown(teams_path, teams_content)
            logger.info("Wrote Teams note: %s", teams_path)

    # 3. Write meeting audio -> 10_meetings/YYYY/MM/DD/audio.md
    for date_str in sorted(audio_kept_grouped.keys()):
        audio_list = audio_kept_grouped[date_str]
        if not audio_list and not audio_filtered_grouped.get(date_str):
            continue
        audio_content = render_meeting_audio_note(
            date_str,
            audio_list,
            filtered_audio=audio_filtered_grouped.get(date_str, []),
        )
        audio_path = meetings_dir / _date_to_path(date_str) / "audio.md"
        if dry_run:
            logger.info("DRY RUN: Would write %d bytes to %s", len(audio_content), audio_path)
        else:
            write_markdown(audio_path, audio_content)
            logger.info("Wrote meeting audio: %s", audio_path)

    # Update cursors
    if not dry_run:
        set_cursor(state, "screenpipe", "last_frame_id", max_frame_id)
        set_cursor(state, "screenpipe", "last_audio_id", max_audio_id)
        save_state(state_path, state)
        logger.info("Updated cursor: frame_id=%d, audio_id=%d", max_frame_id, max_audio_id)


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Screenpipe -> Obsidian Markdown")
    parser.add_argument("--config", help="Path to config.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Preview output without writing")
    parser.add_argument("--reset", action="store_true", help="Reset cursor to re-process all data")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg)

    if args.reset:
        state = load_state(cfg["state_file"])
        set_cursor(state, "screenpipe", "last_frame_id", 0)
        set_cursor(state, "screenpipe", "last_audio_id", 0)
        save_state(cfg["state_file"], state)
        logger.info("Reset Screenpipe cursor to 0")

    run(cfg, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
