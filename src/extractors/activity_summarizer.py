#!/usr/bin/env python3
"""Activity summarizer -- LLM-powered post-processor for Screenpipe data.

Reads raw activity markdown from 85_activity/YYYY/MM/DD/daily.md,
sends recent time blocks to an LLM for task-level summarization,
and writes structured recall notes to 87_recall/YYYY/MM/DD/recall.md.

Uses the existing llm_provider.py / LiteLLM infrastructure with
model tiers configured in config.yaml under screenpipe_settings.summarizer.

Usage::

    python3 -m src.extractors.activity_summarizer
    python3 -m src.extractors.activity_summarizer --dry-run
    python3 -m src.extractors.activity_summarizer --full-day
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.agents import llm_provider
from src.common.config import load_config, resolve_output_dir, setup_logging
from src.common.markdown import append_markdown, write_markdown, yaml_frontmatter
from src.common.state import get_cursor, load_state, save_state, set_cursor

logger = logging.getLogger("memoryos.activity_summarizer")

REPO_DIR = Path(__file__).resolve().parent.parent.parent
ENV_LOCAL = REPO_DIR / ".env.local"


def _load_env() -> None:
    """Load .env.local into os.environ if present."""
    if ENV_LOCAL.is_file():
        for line in ENV_LOCAL.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())


SUMMARY_SYSTEM_PROMPT = """\
You are an activity analyst. You receive raw screen OCR text and audio \
transcriptions captured over a time period. Your job is to convert this \
raw data into a concise, structured TASK-LEVEL recall note.

Rules:
- Group activity into logical TASKS, not just apps. Identify what the \
  user was actually doing (e.g. "Reviewing PR #452", "Writing migration plan").
- For each task block output:
  ## HH:MM - HH:MM | Brief Task Description
  **Apps:** comma-separated list
  **Tasks:**
  - Specific action taken
  - Another action
  **Artifacts:**
  - URLs, file paths, PR numbers, document names, email subjects found
- If audio transcriptions are present, summarize meeting discussions \
  as a task block with key points and decisions.
- Omit OCR noise (UI chrome, repeated menu text, status bars).
- Be concise. Each task block should be 3-8 lines.
- Output ONLY the markdown task blocks, no preamble or commentary.
- If the data is empty or contains only noise, output: "(no significant activity)"
"""


def _get_summarizer_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Extract summarizer settings from config, with defaults."""
    defaults = {
        "enabled": True,
        "model_summary": "gpt-5-mini",
        "model_cleanup": "gpt-5-nano",
        "model_digest": "gpt-5.2-pro",
        "model_weekly": "gpt-5.2-pro",
        "reasoning_summary": "none",
        "reasoning_digest": "medium",
        "reasoning_weekly": "high",
        "max_input_chars": 50_000,
        "max_input_chars_digest": 500_000,
        "max_input_chars_weekly": 1_000_000,
    }
    summ_cfg = cfg.get("screenpipe_settings", {}).get("summarizer", {})
    if summ_cfg:
        defaults.update(summ_cfg)
    return defaults


def _extract_time_window(
    daily_content: str,
    window_start: datetime,
    window_end: datetime,
) -> str:
    """Extract content from daily.md that falls within a time window.

    Parses ## HH:00 -- HH:59 headings and returns only blocks within range.
    """
    hour_pattern = re.compile(r"^## (\d{2}):00 -- \d{2}:59", re.MULTILINE)
    sections: list[tuple[int, str]] = []

    for m in hour_pattern.finditer(daily_content):
        hour = int(m.group(1))
        sections.append((hour, m.start()))

    if not sections:
        return daily_content

    result_parts: list[str] = []
    start_hour = window_start.hour
    end_hour = window_end.hour

    for i, (hour, start_pos) in enumerate(sections):
        if hour < start_hour or hour > end_hour:
            continue
        end_pos = sections[i + 1][1] if i + 1 < len(sections) else len(daily_content)
        result_parts.append(daily_content[start_pos:end_pos])

    return "\n".join(result_parts) if result_parts else ""


def _extract_full_day(daily_content: str) -> str:
    """Return the full daily.md content, skipping YAML frontmatter."""
    if daily_content.startswith("---"):
        end = daily_content.find("---", 3)
        if end != -1:
            return daily_content[end + 3:].strip()
    return daily_content


def summarize_window(
    raw_text: str,
    cfg: dict[str, Any],
    *,
    model_key: str = "model_summary",
    reasoning_key: str = "reasoning_summary",
    max_chars_key: str = "max_input_chars",
) -> str:
    """Send raw activity text to the LLM and return structured summary."""
    summ_cfg = _get_summarizer_config(cfg)
    model = summ_cfg.get(model_key, "gpt-5-mini")
    reasoning = summ_cfg.get(reasoning_key, "none")
    max_chars = summ_cfg.get(max_chars_key, 50_000)

    if len(raw_text) > max_chars:
        raw_text = raw_text[:max_chars] + "\n\n[... truncated for context window ...]\n"

    messages = [
        {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
        {"role": "user", "content": raw_text},
    ]

    return llm_provider.complete(
        messages,
        model_override=model,
        provider_override=cfg.get("agents", {}).get("provider", "openai"),
    )


def _date_to_path(date_str: str) -> str:
    """Convert YYYY-MM-DD to YYYY/MM/DD nested path."""
    parts = date_str.split("-")
    return f"{parts[0]}/{parts[1]}/{parts[2]}" if len(parts) == 3 else date_str


_CATCHUP_CHUNK_HOURS: int = 1
_CATCHUP_MAX_GAP_HOURS: int = 12


def _safe_save_state(state_path: str, state: dict, cursor_ts: str) -> None:
    """Save state with error handling and retry."""
    set_cursor(state, "activity_summarizer", "last_processed_ts", cursor_ts)
    for attempt in range(2):
        try:
            save_state(state_path, state)
            logger.info("Updated cursor to %s", cursor_ts)
            return
        except Exception as e:
            logger.error("State save failed (attempt %d): %s", attempt + 1, e)
            if attempt == 0:
                import time as _t
                _t.sleep(1)
    logger.error("State save failed after retries — cursor may be lost")


def _infer_cursor_from_recall(recall_dir: Path, date_path: str) -> str:
    """If recall file exists but state is missing, infer cursor from it."""
    recall_file = recall_dir / date_path / "recall.md"
    if not recall_file.is_file():
        return ""
    try:
        for line in recall_file.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("last_updated:"):
                ts = line.split(":", 1)[1].strip().strip('"').strip("'")
                logger.warning("Inferred cursor from recall file: %s", ts)
                return ts
    except Exception:
        pass
    return ""


def _process_single_window(
    daily_content: str,
    window_start: datetime,
    window_end: datetime,
    recall_dir: Path,
    date_path: str,
    today_str: str,
    cfg: dict[str, Any],
    now: datetime,
) -> bool:
    """Process a single time window. Returns True if summary was written."""
    raw_text = _extract_time_window(daily_content, window_start, window_end)
    window_label = f"{window_start.strftime('%H:%M')} - {window_end.strftime('%H:%M')}"

    if not raw_text.strip():
        logger.info("No activity data in window %s", window_label)
        return False

    logger.info("Summarizing %s (%d chars of raw data)", window_label, len(raw_text))
    summary = summarize_window(raw_text, cfg)

    if not summary or summary.strip() == "(no significant activity)":
        logger.info("LLM returned no significant activity for %s", window_label)
        return False

    recall_file = recall_dir / date_path / "recall.md"
    if not recall_file.exists():
        meta = {
            "date": today_str,
            "source": "activity-summarizer",
            "type": "task-recall",
            "last_updated": now.isoformat(timespec="seconds"),
        }
        header = yaml_frontmatter(meta) + f"\n\n# Task Recall -- {today_str}\n\n"
        write_markdown(recall_file, header + summary + "\n\n")
    else:
        append_markdown(recall_file, summary + "\n\n")

    logger.info("Wrote recall summary to %s", recall_file)
    return True


def run(cfg: dict[str, Any], *, dry_run: bool = False, full_day: bool = False) -> None:
    """Run the activity summarizer.

    Reads 85_activity/ raw data, summarizes via LLM, writes to 87_recall/.
    Supports catch-up mode: if the gap since last processing is > 2 hours,
    processes in 1-hour chunks to produce better recall blocks.
    """
    summ_cfg = _get_summarizer_config(cfg)
    if not summ_cfg.get("enabled", True):
        logger.info("Summarizer disabled in config")
        return

    activity_dir = resolve_output_dir(cfg, "activity")
    recall_dir = resolve_output_dir(cfg, "recall")
    state_path = cfg["state_file"]
    state = load_state(state_path)

    now = datetime.now()
    today_str = now.strftime("%Y-%m-%d")
    date_path = _date_to_path(today_str)

    daily_file = activity_dir / date_path / "daily.md"
    if not daily_file.is_file():
        logger.info("No daily activity file for %s: %s", today_str, daily_file)
        return

    daily_content = daily_file.read_text(encoding="utf-8", errors="replace")
    if not daily_content.strip():
        logger.info("Empty daily activity file for %s", today_str)
        return

    if full_day:
        raw_text = _extract_full_day(daily_content)
        window_label = f"Full day: {today_str}"

        if not raw_text.strip():
            logger.info("No activity data for full day %s", today_str)
            return

        logger.info("Summarizing %s (%d chars of raw data)", window_label, len(raw_text))
        summary = summarize_window(raw_text, cfg)

        if not summary or summary.strip() == "(no significant activity)":
            logger.info("LLM returned no significant activity for %s", window_label)
            return

        recall_file = recall_dir / date_path / "recall.md"
        if dry_run:
            print(f"--- Recall for {window_label} ---")
            print(summary)
            return

        if not recall_file.exists():
            meta = {
                "date": today_str, "source": "activity-summarizer",
                "type": "task-recall",
                "last_updated": now.isoformat(timespec="seconds"),
            }
            header = yaml_frontmatter(meta) + f"\n\n# Task Recall -- {today_str}\n\n"
            write_markdown(recall_file, header + summary + "\n\n")
        else:
            append_markdown(recall_file, summary + "\n\n")
        logger.info("Wrote recall summary to %s", recall_file)
        return

    last_ts = get_cursor(state, "activity_summarizer", "last_processed_ts", "")
    if not last_ts:
        inferred = _infer_cursor_from_recall(recall_dir, date_path)
        if inferred:
            last_ts = inferred
            logger.warning("State cursor missing — recovered from recall file: %s", inferred)

    if last_ts:
        window_start = datetime.fromisoformat(last_ts)
    else:
        window_start = now - timedelta(minutes=30)

    gap_hours = (now - window_start).total_seconds() / 3600
    gap_hours = min(gap_hours, _CATCHUP_MAX_GAP_HOURS)

    if gap_hours > 2 and not dry_run:
        logger.info(
            "Large gap detected (%.1fh since %s) — processing in %dh chunks",
            gap_hours, window_start.strftime('%H:%M'), _CATCHUP_CHUNK_HOURS,
        )
        chunk_start = window_start
        while chunk_start < now:
            chunk_end = min(chunk_start + timedelta(hours=_CATCHUP_CHUNK_HOURS), now)
            _process_single_window(
                daily_content, chunk_start, chunk_end,
                recall_dir, date_path, today_str, cfg, now,
            )
            chunk_start = chunk_end
            _safe_save_state(state_path, state, chunk_start.isoformat(timespec="seconds"))
        return

    window_end = now
    raw_text = _extract_time_window(daily_content, window_start, window_end)
    window_label = f"{window_start.strftime('%H:%M')} - {window_end.strftime('%H:%M')}"

    if not raw_text.strip():
        logger.info("No activity data in window %s", window_label)
        if not dry_run:
            _safe_save_state(state_path, state, now.isoformat(timespec="seconds"))
        return

    logger.info("Summarizing %s (%d chars of raw data)", window_label, len(raw_text))
    summary = summarize_window(raw_text, cfg)

    if not summary or summary.strip() == "(no significant activity)":
        logger.info("LLM returned no significant activity for %s", window_label)
        if not dry_run:
            _safe_save_state(state_path, state, now.isoformat(timespec="seconds"))
        return

    recall_file = recall_dir / date_path / "recall.md"

    if dry_run:
        print(f"--- Recall for {window_label} ---")
        print(summary)
        return

    if not recall_file.exists():
        meta = {
            "date": today_str, "source": "activity-summarizer",
            "type": "task-recall",
            "last_updated": now.isoformat(timespec="seconds"),
        }
        header = yaml_frontmatter(meta) + f"\n\n# Task Recall -- {today_str}\n\n"
        write_markdown(recall_file, header + summary + "\n\n")
    else:
        append_markdown(recall_file, summary + "\n\n")

    logger.info("Wrote recall summary to %s", recall_file)

    if not dry_run:
        _safe_save_state(state_path, state, now.isoformat(timespec="seconds"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM-powered activity summarizer for Screenpipe data",
    )
    parser.add_argument("--config", help="Path to config.yaml")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print summary to stdout without writing files",
    )
    parser.add_argument(
        "--full-day", action="store_true",
        help="Summarize the entire day instead of the last window",
    )
    args = parser.parse_args()

    _load_env()
    cfg = load_config(args.config)
    setup_logging(cfg)

    run(cfg, dry_run=args.dry_run, full_day=args.full_day)


if __name__ == "__main__":
    main()
