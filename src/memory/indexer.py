"""Vault scanner that keeps the SQLite FTS5 index in sync with Markdown files.

Designed to run on a schedule (launchd every 5 min) or on-demand.
Only reindexes files whose mtime has changed since the last run.
"""

from __future__ import annotations

import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from src.common.config import load_config, setup_logging
from src.memory.index import MemoryIndex
from src.memory.tier import classify, reclassify_all
from src.memory.context import generate_context_files

logger = logging.getLogger("memoryos.memory")

# Maps vault subfolder prefixes to source_type labels
_SOURCE_MAP: dict[str, str] = {
    "00_inbox": "email",
    "10_meetings": "meetings",
    "20_teams-chat": "teams",
    "40_slides": "slides",
    "50_knowledge": "knowledge",
    "85_activity": "activity",
}

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def _detect_source_type(rel_path: str, output_cfg: dict[str, str]) -> str:
    """Infer source_type from the file's position in the vault folder hierarchy."""
    inv = {v: k for k, v in output_cfg.items()}
    first_part = rel_path.split("/")[0] if "/" in rel_path else ""
    return inv.get(first_part, _SOURCE_MAP.get(first_part, "unknown"))


def _parse_frontmatter(text: str) -> dict[str, Any]:
    """Extract YAML frontmatter from Markdown, returning empty dict if absent."""
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}
    try:
        return yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError:
        return {}


def _extract_title(text: str, frontmatter: dict[str, Any], file_path: Path) -> str:
    """Extract title from frontmatter, first heading, or filename."""
    if "title" in frontmatter:
        return str(frontmatter["title"])
    for line in text.splitlines()[:20]:
        if line.startswith("# "):
            return line.lstrip("# ").strip()
    return file_path.stem.replace("_", " ").replace("-", " ")


def _extract_created_at(frontmatter: dict[str, Any], file_path: Path) -> datetime:
    """Get creation date from frontmatter or infer from path (YYYY/MM/DD pattern)."""
    for key in ("date", "created", "created_at"):
        if key in frontmatter:
            val = frontmatter[key]
            if isinstance(val, datetime):
                return val
            try:
                return datetime.fromisoformat(str(val))
            except (ValueError, TypeError):
                pass

    parts = file_path.parts
    for i, part in enumerate(parts):
        if re.match(r"^\d{4}$", part) and i + 2 < len(parts):
            month = parts[i + 1]
            day = parts[i + 2]
            if re.match(r"^\d{2}$", month) and re.match(r"^\d{2}$", day):
                try:
                    return datetime(int(part), int(month), int(day))
                except ValueError:
                    pass

    try:
        stat = file_path.stat()
        return datetime.fromtimestamp(stat.st_birthtime)
    except (OSError, AttributeError):
        return datetime.fromtimestamp(file_path.stat().st_mtime)


def scan_vault(cfg: dict[str, Any], *, full: bool = False) -> dict[str, int]:
    """Scan the vault and update the index. Returns stats dict.

    Args:
        cfg: Loaded MemoryOS config dict.
        full: If True, reindex all files regardless of mtime.
    """
    vault = Path(cfg["obsidian_vault"])
    mem_cfg = cfg.get("memory", {})
    db_path = mem_cfg.get("index_db", "config/memory.db")
    context_dir = mem_cfg.get("context_dir", "_context")
    tier_cfg = mem_cfg.get("tiers", {})
    hot_days = tier_cfg.get("hot_days", 7)
    warm_days = tier_cfg.get("warm_days", 90)
    output_cfg = cfg.get("output", {})

    stats = {"scanned": 0, "indexed": 0, "skipped": 0, "removed": 0, "errors": 0}

    with MemoryIndex(db_path) as idx:
        valid_paths: set[str] = set()

        for md_file in vault.rglob("*.md"):
            rel = str(md_file.relative_to(vault))

            if rel.startswith(context_dir):
                continue

            valid_paths.add(rel)
            stats["scanned"] += 1

            current_mtime = md_file.stat().st_mtime_ns
            if not full:
                stored_mtime = idx.get_mtime(rel)
                if stored_mtime is not None and stored_mtime == current_mtime:
                    stats["skipped"] += 1
                    continue

            try:
                text = md_file.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                logger.warning("Cannot read %s: %s", md_file, exc)
                stats["errors"] += 1
                continue

            fm = _parse_frontmatter(text)
            title = _extract_title(text, fm, md_file)
            source_type = _detect_source_type(rel, output_cfg)
            created_at = _extract_created_at(fm, md_file)
            tier = classify(created_at, hot_days=hot_days, warm_days=warm_days)

            idx.upsert(
                path=rel,
                title=title,
                source_type=source_type,
                content=text,
                created_at=created_at,
                modified_at=datetime.fromtimestamp(md_file.stat().st_mtime),
                tier=tier,
                mtime_ns=current_mtime,
            )
            stats["indexed"] += 1

        stats["removed"] = idx.remove_missing(valid_paths)
        reclassify_all(idx, hot_days=hot_days, warm_days=warm_days)

        logger.info("Index stats: %s", idx.stats())
        logger.info("Scan results: %s", stats)

        generate_context_files(idx, vault, context_dir, cfg)

    return stats


def main() -> None:
    """Entry point for ``python -m src.memory.indexer``."""
    cfg = load_config()
    setup_logging(cfg)

    full = "--full" in sys.argv
    logger.info("Starting vault indexer (full=%s)", full)
    stats = scan_vault(cfg, full=full)
    logger.info("Indexer complete: %s", stats)


if __name__ == "__main__":
    main()
