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
import re
import sqlite3
import subprocess
import sys
from bisect import bisect_left, bisect_right
from collections import defaultdict
from datetime import datetime, time as dt_time, timedelta, timezone
from pathlib import Path
from typing import Any, NamedTuple

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

QUERY_OCR_BY_DATE = """
SELECT f.id AS frame_id,
       f.timestamp,
       COALESCE(f.app_name, '') AS app_name,
       COALESCE(f.window_name, '') AS window_name,
       COALESCE(f.browser_url, '') AS browser_url,
       o.text,
       COALESCE(o.focused, 0) AS focused
FROM frames f
JOIN ocr_text o ON o.frame_id = f.id
WHERE date(f.timestamp) = ?
ORDER BY f.timestamp ASC
"""

QUERY_AUDIO_BY_DATE = """
SELECT at.id,
       at.timestamp,
       at.transcription,
       COALESCE(at.device, '') AS device,
       COALESCE(at.is_input_device, 1) AS is_input_device,
       at.speaker_id,
       s.name AS speaker_name
FROM audio_transcriptions at
LEFT JOIN speakers s ON at.speaker_id = s.id
WHERE date(at.timestamp) = ?
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


def deduplicate_audio(entries: list[AudioEntry], threshold: float = 0.85) -> list[AudioEntry]:
    """Remove near-duplicate consecutive audio entries from the same speaker."""
    if not entries:
        return entries

    result: list[AudioEntry] = [entries[0]]
    for entry in entries[1:]:
        prev = result[-1]
        if (entry.speaker_id == prev.speaker_id
                and text_similarity(entry.transcription, prev.transcription) >= threshold):
            continue
        result.append(entry)

    logger.info("Audio dedup: %d -> %d entries", len(entries), len(result))
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


# ── Calendar correlation ──────────────────────────────────────────────────────

_CALENDAR_HEADING_RE = re.compile(
    r"^##\s+(\d{1,2}:\d{2})\s*-\s*(\d{1,2}:\d{2}):\s*(.+)$"
)


class CalendarEvent(NamedTuple):
    start: datetime
    end: datetime
    title: str


def parse_calendar_events(calendar_path: Path, date_str: str) -> list[CalendarEvent]:
    """Parse event time ranges and titles from a calendar.md file.

    Expects headings like ``## 11:00 - 12:00: Event Title``.
    All-day events (no time range) are skipped.
    """
    if not calendar_path.is_file():
        return []

    events: list[CalendarEvent] = []
    text = calendar_path.read_text(encoding="utf-8")
    for line in text.splitlines():
        m = _CALENDAR_HEADING_RE.match(line.strip())
        if not m:
            continue
        start_str, end_str, title = m.group(1), m.group(2), m.group(3).strip()
        try:
            start_t = datetime.strptime(f"{date_str} {start_str}", "%Y-%m-%d %H:%M")
            end_t = datetime.strptime(f"{date_str} {end_str}", "%Y-%m-%d %H:%M")
            start_t = start_t.astimezone()
            end_t = end_t.astimezone()
        except ValueError:
            continue
        events.append(CalendarEvent(start=start_t, end=end_t, title=title))

    return events


class MeetingAudioMatch(NamedTuple):
    time_range: str
    title: str
    entries: list[AudioEntry]


def match_audio_to_meetings(
    audio_entries: list[AudioEntry],
    events: list[CalendarEvent],
    buffer_minutes: int = 5,
) -> tuple[list[MeetingAudioMatch], list[AudioEntry]]:
    """Bucket audio entries into calendar event windows.

    Returns (matched_meetings, uncorrelated) where each matched meeting
    contains a time range string, title, and list of audio entries whose
    timestamps fall within the event window (expanded by buffer_minutes).

    When multiple events overlap, the narrowest (shortest duration) event
    wins so that specific meetings are preferred over broad blocks like
    "out of office" or all-day holds.
    """
    if not events:
        return [], list(audio_entries)

    buf = timedelta(minutes=buffer_minutes)

    indexed_events = sorted(
        enumerate(events),
        key=lambda ie: (ie[1].end - ie[1].start),
    )

    buckets: dict[int, list[AudioEntry]] = defaultdict(list)
    uncorrelated: list[AudioEntry] = []

    for ae in audio_entries:
        placed = False
        for idx, ev in indexed_events:
            if (ev.start - buf) <= ae.timestamp <= (ev.end + buf):
                buckets[idx].append(ae)
                placed = True
                break
        if not placed:
            uncorrelated.append(ae)

    matched: list[MeetingAudioMatch] = []
    for idx, ev in enumerate(events):
        if idx not in buckets:
            continue
        time_range = f"{ev.start.strftime('%H:%M')} - {ev.end.strftime('%H:%M')}"
        matched.append(MeetingAudioMatch(
            time_range=time_range,
            title=ev.title,
            entries=buckets[idx],
        ))

    return matched, uncorrelated


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
    matched_meetings: list[MeetingAudioMatch] | None = None,
    uncorrelated: list[AudioEntry] | None = None,
) -> str:
    """Render meeting audio transcriptions for 10_meetings/.

    When calendar correlation data is provided (matched_meetings / uncorrelated),
    audio is grouped under meeting headings.  Otherwise falls back to a flat list.
    """
    meta = {
        "date": date_str,
        "source": "screenpipe-audio",
        "type": "meeting-transcript",
        "last_updated": datetime.now().isoformat(timespec="seconds"),
    }
    parts = [yaml_frontmatter(meta), "", f"# Meeting Audio -- {date_str}", ""]

    if matched_meetings is not None:
        for meeting in matched_meetings:
            parts.append(f"## {meeting.time_range}: {meeting.title}")
            parts.append("")
            for ae in meeting.entries:
                time_str = ae.timestamp.strftime("%H:%M")
                speaker = _format_speaker(ae)
                parts.append(f"**{speaker} ({time_str}):** {ae.transcription.strip()}")
                parts.append("")

        if uncorrelated:
            parts.append("## Uncorrelated Audio")
            parts.append("")
            for ae in uncorrelated:
                time_str = ae.timestamp.strftime("%H:%M")
                speaker = _format_speaker(ae)
                parts.append(f"**{speaker} ({time_str}):** {ae.transcription.strip()}")
                parts.append("")
    else:
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


_PERSONAL_MIC_KEYWORDS = {"macbook", "airpods", "iphone microphone"}


def _format_speaker(ae: AudioEntry) -> str:
    """Format speaker label for audio entry.

    Only labels as "You" for personal microphones (MacBook, AirPods).
    Shared input devices like speakerphones and virtual audio capture
    both sides of a conversation, so they use speaker IDs.
    """
    if ae.speaker_name:
        return ae.speaker_name
    if ae.is_input_device:
        device_lower = (ae.device or "").lower()
        if any(kw in device_lower for kw in _PERSONAL_MIC_KEYWORDS):
            return "You"
    return f"Speaker {ae.speaker_id or '?'}"


# ── Self-healing helpers ─────────────────────────────────────────────────────

import json as _json
import time as _time
import urllib.request as _urllib_request

_HEAL_STATE_FILE = Path(__file__).resolve().parent.parent.parent / "config" / "heal_screenpipe.json"
SCREENPIPE_STALE_SECONDS: int = 600
MAX_FRAME_DROP_RATE: float = 0.95
RESTART_COOLDOWN: int = 600  # 10 min (was 30 min)
CONSECUTIVE_FAILURES_BEFORE_RESTART: int = 3


def _load_heal_state() -> dict[str, Any]:
    if _HEAL_STATE_FILE.is_file():
        try:
            return _json.loads(_HEAL_STATE_FILE.read_text())
        except Exception:
            pass
    return {}


def _save_heal_state(state: dict[str, Any]) -> None:
    _HEAL_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = _HEAL_STATE_FILE.with_suffix(".tmp")
    tmp.write_text(_json.dumps(state, indent=2, default=str))
    tmp.rename(_HEAL_STATE_FILE)


def _is_screenpipe_running() -> bool:
    result = subprocess.run(
        ["pgrep", "-f", "screenpipe-app"],
        capture_output=True,
    )
    return result.returncode == 0


def _ensure_preferred_audio_device() -> None:
    """Activate the preferred audio input device after a Screenpipe restart.

    Waits for the API to become available, then starts the configured device.
    """
    preferred: str | None = None
    try:
        cfg = load_config()
        preferred = cfg.get("screenpipe", {}).get("preferred_input_device")
    except Exception:
        pass
    if not preferred:
        return

    for attempt in range(6):
        _time.sleep(5)
        if _screenpipe_post("/audio/device/start", {"device_name": preferred}):
            logger.info("Activated preferred audio device: %s", preferred)
            return
        logger.debug("Waiting for Screenpipe API (attempt %d/6)", attempt + 1)

    logger.warning("Failed to activate preferred audio device '%s' after restart", preferred)


def _restart_screenpipe(reason: str) -> None:
    """Kill and relaunch Screenpipe."""
    logger.warning("Restarting Screenpipe — %s", reason)
    subprocess.run(["pkill", "-f", "screenpipe-app"], timeout=10, capture_output=True)
    _time.sleep(5)
    subprocess.run(["open", "-a", "screenpipe"], timeout=10, capture_output=True)
    _time.sleep(10)
    _ensure_preferred_audio_device()
    hs = _load_heal_state()
    hs["consecutive_no_data"] = 0
    hs["consecutive_degraded"] = 0
    hs["last_restart_epoch"] = _time.time()
    hs["last_restart_reason"] = reason
    _save_heal_state(hs)
    logger.info("Screenpipe restarted")


def _can_restart() -> bool:
    hs = _load_heal_state()
    return (_time.time() - hs.get("last_restart_epoch", 0)) > RESTART_COOLDOWN


def _query_screenpipe_health() -> dict[str, Any] | None:
    """Query the Screenpipe health API at localhost:3030."""
    try:
        req = _urllib_request.Request("http://localhost:3030/health", method="GET")
        with _urllib_request.urlopen(req, timeout=5) as resp:
            return _json.loads(resp.read())
    except Exception:
        return None


def _screenpipe_post(path: str, body: dict[str, Any]) -> bool:
    """POST JSON to a Screenpipe API endpoint."""
    url = f"http://localhost:3030{path}"
    data = _json.dumps(body).encode()
    req = _urllib_request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    try:
        with _urllib_request.urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except Exception:
        return False


def _cycle_audio_devices() -> None:
    """Stop and restart audio devices via the Screenpipe API.

    When a preferred_input_device is configured in config.yaml, only that
    device is cycled.  Otherwise all devices are cycled (legacy behaviour).
    """
    try:
        req = _urllib_request.Request("http://localhost:3030/audio/list", method="GET")
        with _urllib_request.urlopen(req, timeout=5) as resp:
            devices = _json.loads(resp.read())
    except Exception:
        logger.warning("Cannot list audio devices — Screenpipe API unreachable")
        return

    preferred: str | None = None
    try:
        cfg = load_config()
        preferred = cfg.get("screenpipe", {}).get("preferred_input_device")
    except Exception:
        pass

    cycled = 0
    for dev in devices:
        name = dev.get("name", "")
        if not name:
            continue
        if preferred and name != preferred:
            continue
        _screenpipe_post("/audio/device/stop", {"device_name": name})
        _time.sleep(1)
        _screenpipe_post("/audio/device/start", {"device_name": name})
        _time.sleep(1)
        cycled += 1

    logger.info(
        "Cycled %d audio device(s) to reset VAD state%s",
        cycled,
        f" (preferred: {preferred})" if preferred else "",
    )


def _enforce_preferred_device_only() -> None:
    """Stop all input devices except the preferred one.

    Screenpipe's device_monitor re-adds devices when system defaults change,
    causing USB reconnect sounds.  This prunes unwanted devices each cycle.
    """
    preferred: str | None = None
    try:
        cfg = load_config()
        preferred = cfg.get("screenpipe", {}).get("preferred_input_device")
    except Exception:
        return
    if not preferred:
        return

    try:
        req = _urllib_request.Request("http://localhost:3030/audio/list", method="GET")
        with _urllib_request.urlopen(req, timeout=5) as resp:
            devices = _json.loads(resp.read())
    except Exception:
        return

    stopped = 0
    preferred_found = False
    for dev in devices:
        name = dev.get("name", "")
        if not name or "(output)" in name:
            continue
        if name == preferred:
            preferred_found = True
            continue
        if _screenpipe_post("/audio/device/stop", {"device_name": name}):
            stopped += 1

    if not preferred_found:
        _screenpipe_post("/audio/device/start", {"device_name": preferred})

    if stopped:
        logger.info(
            "Enforced single-device mode: stopped %d unwanted input device(s)",
            stopped,
        )


AUDIO_STALE_CYCLES: int = 2  # 2 cycles * 5 min = 10 min of silence triggers reset
AUDIO_CYCLE_COOLDOWN: int = 600  # 10 min (was 30 min)


def _is_work_hours() -> bool:
    """Return True if current local time is within work hours (7 AM - 7 PM)."""
    hour = datetime.now().hour
    return 7 <= hour < 19


def _check_audio_staleness(had_audio: bool) -> None:
    """Track consecutive no-audio cycles and cycle devices if stale during work hours."""
    hs = _load_heal_state()

    if had_audio:
        if hs.get("consecutive_no_audio", 0) > 0:
            hs["consecutive_no_audio"] = 0
            _save_heal_state(hs)
        return

    if not _is_work_hours():
        return

    no_audio = hs.get("consecutive_no_audio", 0) + 1
    hs["consecutive_no_audio"] = no_audio
    _save_heal_state(hs)

    if no_audio >= AUDIO_STALE_CYCLES:
        last_cycle = hs.get("last_audio_cycle_epoch", 0)
        if (_time.time() - last_cycle) > AUDIO_CYCLE_COOLDOWN:
            logger.warning(
                "No audio for %d consecutive cycles during work hours — "
                "cycling audio devices", no_audio,
            )
            _cycle_audio_devices()
            _enforce_preferred_device_only()
            hs["consecutive_no_audio"] = 0
            hs["last_audio_cycle_epoch"] = _time.time()
            _save_heal_state(hs)


def _check_screenpipe_health(db_path: str) -> None:
    """Check multiple health signals and restart Screenpipe if needed."""
    hs = _load_heal_state()

    # Signal 1: Is the app running at all?
    if not _is_screenpipe_running():
        logger.warning("Screenpipe not running — launching")
        subprocess.run(["open", "-a", "screenpipe"], timeout=10, capture_output=True)
        _time.sleep(10)
        return

    # Signal 2: Health API check (frame drops, stalls)
    health = _query_screenpipe_health()
    if health:
        pipeline = health.get("pipeline", {})
        drop_rate = pipeline.get("frame_drop_rate", 0)
        stall_count = pipeline.get("pipeline_stall_count", 0)
        frame_status = health.get("frame_status", "unknown")
        audio_status = health.get("audio_status", "unknown")

        if frame_status == "stale" and drop_rate > MAX_FRAME_DROP_RATE:
            degraded_count = hs.get("consecutive_degraded", 0) + 1
            hs["consecutive_degraded"] = degraded_count
            _save_heal_state(hs)
            logger.warning(
                "Screenpipe degraded: %.0f%% frame drop, %d stalls "
                "[consecutive_degraded=%d/%d]",
                drop_rate * 100, stall_count,
                degraded_count, CONSECUTIVE_FAILURES_BEFORE_RESTART,
            )
            if degraded_count >= CONSECUTIVE_FAILURES_BEFORE_RESTART and _can_restart():
                _restart_screenpipe(
                    f"persistent frame drops ({drop_rate:.0%}) "
                    f"with {stall_count} stalls"
                )
                return
        else:
            if hs.get("consecutive_degraded", 0) > 0:
                hs["consecutive_degraded"] = 0
                _save_heal_state(hs)

    # Signal 3: DB-timestamp freshness (catches "healthy API but no DB writes")
    try:
        _conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5)
        row = _conn.execute("SELECT MAX(timestamp) FROM frames").fetchone()
        _conn.close()
        if row and row[0]:
            from datetime import timezone as _tz
            ts_str = row[0]
            if "+" in ts_str or ts_str.endswith("Z"):
                last_frame_dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            else:
                last_frame_dt = datetime.fromisoformat(ts_str).replace(tzinfo=_tz.utc)
            db_age = (datetime.now(_tz.utc) - last_frame_dt).total_seconds()
            if db_age > SCREENPIPE_STALE_SECONDS:
                no_data_count = hs.get("consecutive_no_data", 0)
                if no_data_count >= CONSECUTIVE_FAILURES_BEFORE_RESTART and _can_restart():
                    _restart_screenpipe(
                        f"DB frames stale for {db_age / 60:.0f}m (API may report healthy), "
                        f"{no_data_count} cycles with no data"
                    )
                    return
    except Exception as e:
        logger.debug("DB-timestamp check failed: %s", e)

    # Signal 4: DB file-mtime staleness (fallback)
    db = Path(db_path)
    if not db.is_file():
        return

    stale_seconds = _time.time() - db.stat().st_mtime
    no_data_count = hs.get("consecutive_no_data", 0)

    if stale_seconds > SCREENPIPE_STALE_SECONDS and no_data_count >= CONSECUTIVE_FAILURES_BEFORE_RESTART:
        if _can_restart():
            _restart_screenpipe(
                f"DB file stale for {stale_seconds / 60:.0f}m, "
                f"{no_data_count} cycles with no data"
            )
            return

    _enforce_preferred_device_only()


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
        _check_screenpipe_health(db_path)
        return

    state_path = cfg["state_file"]
    state = load_state(state_path)

    last_frame_id = get_cursor(state, "screenpipe", "last_frame_id", 0)
    last_audio_id = get_cursor(state, "screenpipe", "last_audio_id", 0)

    logger.info("Querying Screenpipe DB from frame_id=%d, audio_id=%d", last_frame_id, last_audio_id)

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        max_frame = conn.execute("SELECT MAX(id) FROM frames").fetchone()[0] or 0
        max_audio = conn.execute("SELECT MAX(id) FROM audio_transcriptions").fetchone()[0] or 0
        if last_frame_id > max_frame:
            logger.warning(
                "Frame cursor %d ahead of DB max %d — resetting to 0 (DB may have been rebuilt)",
                last_frame_id, max_frame,
            )
            last_frame_id = 0
        if last_audio_id > max_audio:
            logger.warning(
                "Audio cursor %d ahead of DB max %d — resetting to 0",
                last_audio_id, max_audio,
            )
            last_audio_id = 0

        ocr_rows = conn.execute(QUERY_OCR, (last_frame_id,)).fetchall()
        audio_rows = conn.execute(QUERY_AUDIO, (last_audio_id,)).fetchall()

        if not ocr_rows and not audio_rows:
            logger.info("No new Screenpipe data")
            hs = _load_heal_state()
            hs["consecutive_no_data"] = hs.get("consecutive_no_data", 0) + 1
            _save_heal_state(hs)
            _check_audio_staleness(had_audio=False)
            _check_screenpipe_health(db_path)
            return

        _check_audio_staleness(had_audio=bool(audio_rows))

        # Reset no-data counter on success
        hs = _load_heal_state()
        if hs.get("consecutive_no_data", 0) > 0:
            hs["consecutive_no_data"] = 0
            _save_heal_state(hs)

        max_frame_id = max((r[0] for r in ocr_rows), default=last_frame_id)
        max_audio_id = max((r[0] for r in audio_rows), default=last_audio_id)

        logger.info("Found %d new OCR entries, %d new audio entries", len(ocr_rows), len(audio_rows))

        # Determine which dates were affected by new entries
        affected_dates: set[str] = set()
        for r in ocr_rows:
            affected_dates.add(_parse_ts(r[1]).strftime("%Y-%m-%d"))
        for r in audio_rows:
            affected_dates.add(_parse_ts(r[1]).strftime("%Y-%m-%d"))

        # Re-query ALL entries for affected dates so writes contain complete data
        all_ocr_entries: list[OcrEntry] = []
        all_audio_entries: list[AudioEntry] = []
        for date_str in affected_dates:
            for r in conn.execute(QUERY_OCR_BY_DATE, (date_str,)).fetchall():
                all_ocr_entries.append(OcrEntry(r))
            for r in conn.execute(QUERY_AUDIO_BY_DATE, (date_str,)).fetchall():
                all_audio_entries.append(AudioEntry(r))

        logger.info(
            "Loaded full-day data for %d date(s): %d OCR, %d audio entries",
            len(affected_dates), len(all_ocr_entries), len(all_audio_entries),
        )
    finally:
        conn.close()

    ocr_entries = all_ocr_entries
    audio_entries = all_audio_entries

    threshold = cfg.get("screenpipe_settings", {}).get("dedup_threshold", 0.85)
    ocr_entries = deduplicate_ocr(ocr_entries, threshold)
    audio_entries = deduplicate_audio(audio_entries, threshold)

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

    all_dates = (
        set(ocr_grouped.keys())
        | set(audio_kept_grouped.keys())
        | set(audio_filtered_grouped.keys())
    )

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

        date_path = _date_to_path(date_str)
        calendar_path = meetings_dir / date_path / "calendar.md"
        events = parse_calendar_events(calendar_path, date_str)

        matched, uncorrelated = match_audio_to_meetings(audio_list, events)
        if events:
            logger.info(
                "Calendar correlation for %s: %d events, %d matched, %d uncorrelated",
                date_str, len(events), len(matched), len(uncorrelated),
            )

        audio_content = render_meeting_audio_note(
            date_str,
            audio_list,
            filtered_audio=audio_filtered_grouped.get(date_str, []),
            matched_meetings=matched if events else None,
            uncorrelated=uncorrelated if events else None,
        )
        audio_path = meetings_dir / date_path / "audio.md"
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
