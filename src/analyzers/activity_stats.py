"""Deterministic parser for daily activity markdown files.

Parses the structured ``85_activity/{date}/daily.md`` files produced by
the screenpipe extractor and computes per-app time, total active hours,
and context-switch counts.  These metrics feed the focus-audit skill so
the LLM receives pre-computed numbers instead of trying (and failing) to
parse a 30k-line file within a truncated context window.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

HOUR_RE = re.compile(r"^## (\d{2}:\d{2}) -- \d{2}:\d{2}")
APP_RE = re.compile(r"^### (.+)$")
OCR_LINE_RE = re.compile(r"^> ")
TRUNCATED_LINE_RE = re.compile(r"^> \[\.\.\.truncated\]")

NOISE_APPS = frozenset({
    "Unknown",
    "loginwindow",
    "UserNotificationCenter",
    "SystemUIServer",
})

APP_CATEGORIES: dict[str, str] = {
    "Microsoft Teams": "Meeting/Communication",
    "Zoom": "Meeting/Communication",
    "Google Meet": "Meeting/Communication",
    "Microsoft Outlook": "Communication",
    "Mail": "Communication",
    "Slack": "Communication",
    "Cursor": "Deep Work",
    "VS Code": "Deep Work",
    "Visual Studio Code": "Deep Work",
    "Terminal": "Deep Work",
    "iTerm2": "Deep Work",
    "Warp": "Deep Work",
    "Microsoft Word": "Deep Work",
    "Microsoft PowerPoint": "Deep Work",
    "Microsoft Excel": "Deep Work",
    "Notion": "Deep Work",
    "Obsidian": "Deep Work",
    "Google Chrome": "Research",
    "Safari": "Research",
    "Firefox": "Research",
    "Arc": "Research",
    "ChatGPT": "Research",
    "Claude": "Research",
    "Finder": "Admin",
    "System Settings": "Admin",
    "System Preferences": "Admin",
    "Preview": "Admin",
    "Calendar": "Admin",
}


def _base_app_name(full_header: str) -> str:
    """Extract the base app name from a ``### App - WindowTitle`` header."""
    parts = full_header.split(" - ", 1)
    return parts[0].strip()


def _categorize(app_name: str) -> str:
    return APP_CATEGORIES.get(app_name, "Other")


@dataclass
class HourStats:
    hour: str
    app_captures: dict[str, int] = field(default_factory=dict)
    app_order: list[str] = field(default_factory=list)

    @property
    def context_switches(self) -> int:
        """Count distinct app transitions (consecutive different base apps)."""
        if len(self.app_order) <= 1:
            return 0
        switches = 0
        for i in range(1, len(self.app_order)):
            if self.app_order[i] != self.app_order[i - 1]:
                switches += 1
        return switches

    def scaled_minutes(self) -> dict[str, float]:
        """Distribute 60 minutes across apps proportionally to captures.

        Screenpipe deduplicates ~85-90% of frames, so raw capture counts
        severely undercount actual usage.  Scaling to fill the hour keeps
        the relative proportions accurate while producing realistic totals.
        """
        total = sum(self.app_captures.values())
        if total == 0:
            return {}
        return {
            app: (captures / total) * 60.0
            for app, captures in self.app_captures.items()
        }


@dataclass
class DailyStats:
    """Pre-computed activity statistics for one day."""

    date: str
    hours: list[HourStats] = field(default_factory=list)
    apps_used: list[str] = field(default_factory=list)

    @property
    def total_active_hours(self) -> float:
        """Hours where at least one non-noise app had captures."""
        return float(len([h for h in self.hours if h.app_captures]))

    @property
    def app_minutes(self) -> dict[str, float]:
        """Aggregate minutes per base app, scaled proportionally per hour."""
        totals: dict[str, float] = {}
        for h in self.hours:
            for app, mins in h.scaled_minutes().items():
                totals[app] = totals.get(app, 0.0) + mins
        return totals

    @property
    def hourly_app_minutes(self) -> dict[str, dict[str, float]]:
        """hour -> app -> minutes."""
        result: dict[str, dict[str, float]] = {}
        for h in self.hours:
            scaled = h.scaled_minutes()
            if scaled:
                result[h.hour] = scaled
        return result

    @property
    def context_switches_per_hour(self) -> dict[str, int]:
        return {h.hour: h.context_switches for h in self.hours if h.app_captures}

    @property
    def total_context_switches(self) -> int:
        return sum(h.context_switches for h in self.hours)

    def to_app_breakdown(self) -> list[dict[str, Any]]:
        """Format compatible with the focus-audit JSON ``app_breakdown`` field."""
        minutes = self.app_minutes
        if not minutes:
            return []
        # Filter out apps with <1 minute of usage
        significant = {a: m for a, m in minutes.items() if m >= 1.0}
        if not significant:
            return []
        total = sum(significant.values()) or 1.0
        breakdown = []
        for app, mins in sorted(significant.items(), key=lambda x: -x[1]):
            breakdown.append({
                "name": app,
                "minutes": round(mins),
                "percent": round((mins / total) * 100),
                "category": _categorize(app),
            })
        return breakdown

    def to_top_apps(self) -> list[dict[str, Any]]:
        """Format compatible with the focus-audit JSON ``top_apps`` field."""
        minutes = self.app_minutes
        return [
            {
                "name": app,
                "hours": round(mins / 60.0, 1),
                "category": _categorize(app),
            }
            for app, mins in sorted(minutes.items(), key=lambda x: -x[1])
        ]

    def to_compact_summary(self) -> str:
        """Human-readable summary small enough to fit in an LLM prompt."""
        lines = [
            f"Total active hours: {self.total_active_hours:.0f}",
            f"Total context switches: {self.total_context_switches}",
            "",
            "Per-app time (minutes):",
        ]
        for app, mins in sorted(self.app_minutes.items(), key=lambda x: -x[1]):
            lines.append(f"  {app}: {mins:.0f}m ({_categorize(app)})")

        lines.append("")
        lines.append("Per-hour breakdown:")
        for h in self.hours:
            scaled = h.scaled_minutes()
            if not scaled:
                continue
            apps_str = ", ".join(
                f"{a}={m:.0f}m"
                for a, m in sorted(scaled.items(), key=lambda x: -x[1])
            )
            lines.append(f"  {h.hour}: switches={h.context_switches} | {apps_str}")
        return "\n".join(lines)

    def to_metrics_dict(self) -> dict[str, Any]:
        """Pre-computed metrics to merge into the focus-audit JSON."""
        return {
            "total_active_hours": self.total_active_hours,
            "total_context_switches": self.total_context_switches,
            "context_switches_per_hour": self.context_switches_per_hour,
            "app_breakdown": self.to_app_breakdown(),
            "top_apps": self.to_top_apps(),
        }


def parse_daily_activity(text: str) -> DailyStats:
    """Parse a daily activity markdown file into structured stats.

    Expects the format produced by ``screenpipe_extractor.render_daily_note``:
    frontmatter, then ``## HH:00 -- HH:59`` sections containing
    ``### App - Window`` subsections with ``> OCR text`` lines.
    """
    date = ""
    apps_used_raw: list[str] = []
    hours: list[HourStats] = []
    current_hour: HourStats | None = None
    current_app: str | None = None
    in_frontmatter = False
    in_audio_section = False

    for line in text.splitlines():
        stripped = line.strip()

        if stripped == "---":
            if in_frontmatter:
                in_frontmatter = False
                continue
            if not date:
                in_frontmatter = True
                continue
            # Separator before Audio Transcriptions -- stop parsing OCR
            in_audio_section = True
            continue

        if in_frontmatter:
            if stripped.startswith("date:"):
                date = stripped.split(":", 1)[1].strip().strip('"').strip("'")
            elif stripped.startswith("- ") and apps_used_raw is not None:
                apps_used_raw.append(stripped[2:].strip())
            continue

        if in_audio_section:
            continue

        hour_match = HOUR_RE.match(stripped)
        if hour_match:
            current_hour = HourStats(hour=hour_match.group(1))
            hours.append(current_hour)
            current_app = None
            continue

        app_match = APP_RE.match(stripped)
        if app_match and current_hour is not None:
            base = _base_app_name(app_match.group(1))
            if base in NOISE_APPS:
                current_app = None
                continue
            current_app = base
            current_hour.app_order.append(base)
            continue

        if (
            OCR_LINE_RE.match(line)
            and not TRUNCATED_LINE_RE.match(line)
            and current_hour is not None
            and current_app is not None
        ):
            current_hour.app_captures[current_app] = (
                current_hour.app_captures.get(current_app, 0) + 1
            )

    return DailyStats(
        date=date,
        hours=hours,
        apps_used=[a for a in apps_used_raw if a not in NOISE_APPS],
    )
