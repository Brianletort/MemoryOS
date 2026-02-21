"""Tests for screenpipe extractor: audio dedup, calendar parsing, meeting matching."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from textwrap import dedent

import pytest

from src.extractors.screenpipe_extractor import (
    AudioEntry,
    CalendarEvent,
    MeetingAudioMatch,
    deduplicate_audio,
    match_audio_to_meetings,
    parse_calendar_events,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _audio(
    audio_id: int,
    ts: str,
    text: str,
    speaker_id: int = 1,
    device: str = "MacBook Pro Microphone",
    is_input: int = 1,
    speaker_name: str | None = None,
) -> AudioEntry:
    """Build an AudioEntry from minimal fields."""
    return AudioEntry((audio_id, ts, text, device, is_input, speaker_id, speaker_name))


# ── deduplicate_audio ────────────────────────────────────────────────────────

class TestDeduplicateAudio:
    def test_empty_list(self) -> None:
        assert deduplicate_audio([]) == []

    def test_single_entry(self) -> None:
        entry = _audio(1, "2026-02-20T11:00:00Z", "Hello there")
        assert deduplicate_audio([entry]) == [entry]

    def test_removes_consecutive_duplicates(self) -> None:
        a = _audio(1, "2026-02-20T11:00:00Z", "Hello everyone how are you today")
        b = _audio(2, "2026-02-20T11:00:01Z", "Hello everyone how are you today")
        result = deduplicate_audio([a, b])
        assert len(result) == 1
        assert result[0].audio_id == 1

    def test_keeps_different_speakers(self) -> None:
        a = _audio(1, "2026-02-20T11:00:00Z", "Hello everyone", speaker_id=1)
        b = _audio(2, "2026-02-20T11:00:01Z", "Hello everyone", speaker_id=2)
        result = deduplicate_audio([a, b])
        assert len(result) == 2

    def test_keeps_different_content(self) -> None:
        a = _audio(1, "2026-02-20T11:00:00Z", "Let's discuss the migration plan")
        b = _audio(2, "2026-02-20T11:00:30Z", "We should review the budget first")
        result = deduplicate_audio([a, b])
        assert len(result) == 2

    def test_respects_threshold(self) -> None:
        a = _audio(1, "2026-02-20T11:00:00Z", "Hello everyone how are you today")
        b = _audio(2, "2026-02-20T11:00:01Z", "Hello everyone how are you today friend")
        assert len(deduplicate_audio([a, b], threshold=0.99)) == 2
        assert len(deduplicate_audio([a, b], threshold=0.5)) == 1


# ── parse_calendar_events ────────────────────────────────────────────────────

class TestParseCalendarEvents:
    def test_parses_standard_events(self, tmp_path: Path) -> None:
        cal = tmp_path / "calendar.md"
        cal.write_text(dedent("""\
            ---
            date: 2026-02-20
            source: calendar-app
            type: calendar
            event_count: 2
            ---

            # Calendar -- 2026-02-20

            ## 11:00 - 12:00: [External] Candidate Interview

            - **Location:** Microsoft Teams Meeting

            ## 12:00 - 12:30: Project Alpha Review (Phase 2)

            - **Location:** Microsoft Teams Meeting
        """))

        events = parse_calendar_events(cal, "2026-02-20")
        assert len(events) == 2
        assert events[0].title == "[External] Candidate Interview"
        assert events[0].start.hour == 11
        assert events[0].end.hour == 12
        assert events[1].title == "Project Alpha Review (Phase 2)"
        assert events[1].start.hour == 12
        assert events[1].end.minute == 30

    def test_skips_all_day_events(self, tmp_path: Path) -> None:
        cal = tmp_path / "calendar.md"
        cal.write_text(dedent("""\
            # Calendar -- 2026-02-20

            ## All Day: Company Holiday

            ## 09:00 - 10:00: Standup

            - **Location:** Teams
        """))

        events = parse_calendar_events(cal, "2026-02-20")
        assert len(events) == 1
        assert events[0].title == "Standup"

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        cal = tmp_path / "nonexistent.md"
        assert parse_calendar_events(cal, "2026-02-20") == []

    def test_no_events_returns_empty(self, tmp_path: Path) -> None:
        cal = tmp_path / "calendar.md"
        cal.write_text("# Calendar -- 2026-02-20\n\nNo events today.\n")
        assert parse_calendar_events(cal, "2026-02-20") == []


# ── match_audio_to_meetings ──────────────────────────────────────────────────

def _make_events() -> list[CalendarEvent]:
    """Two events: 11:00-12:00 and 12:00-12:30 on 2026-02-20."""
    return [
        CalendarEvent(
            start=datetime(2026, 2, 20, 11, 0).astimezone(),
            end=datetime(2026, 2, 20, 12, 0).astimezone(),
            title="Team Standup",
        ),
        CalendarEvent(
            start=datetime(2026, 2, 20, 12, 0).astimezone(),
            end=datetime(2026, 2, 20, 12, 30).astimezone(),
            title="Project Review",
        ),
    ]


class TestMatchAudioToMeetings:
    def test_matches_audio_to_correct_meeting(self) -> None:
        events = _make_events()
        entries = [
            _audio(1, "2026-02-20T11:05:00-06:00", "Good morning everyone"),
            _audio(2, "2026-02-20T11:30:00-06:00", "Let me share my screen"),
            _audio(3, "2026-02-20T12:10:00-06:00", "Here is the status update"),
        ]
        matched, uncorrelated = match_audio_to_meetings(entries, events)
        assert len(matched) == 2
        assert matched[0].title == "Team Standup"
        assert len(matched[0].entries) == 2
        assert matched[1].title == "Project Review"
        assert len(matched[1].entries) == 1
        assert len(uncorrelated) == 0

    def test_uncorrelated_audio(self) -> None:
        events = _make_events()
        entries = [
            _audio(1, "2026-02-20T14:00:00-06:00", "This is an ad-hoc call"),
        ]
        matched, uncorrelated = match_audio_to_meetings(entries, events)
        assert len(matched) == 0
        assert len(uncorrelated) == 1

    def test_buffer_captures_early_join(self) -> None:
        events = _make_events()
        entries = [
            _audio(1, "2026-02-20T10:57:00-06:00", "Hi, joining a bit early"),
        ]
        matched, uncorrelated = match_audio_to_meetings(entries, events, buffer_minutes=5)
        assert len(matched) == 1
        assert matched[0].title == "Team Standup"
        assert len(uncorrelated) == 0

    def test_no_events_all_uncorrelated(self) -> None:
        entries = [
            _audio(1, "2026-02-20T11:00:00-06:00", "Some audio"),
        ]
        matched, uncorrelated = match_audio_to_meetings(entries, [])
        assert len(matched) == 0
        assert len(uncorrelated) == 1

    def test_empty_audio(self) -> None:
        events = _make_events()
        matched, uncorrelated = match_audio_to_meetings([], events)
        assert len(matched) == 0
        assert len(uncorrelated) == 0
