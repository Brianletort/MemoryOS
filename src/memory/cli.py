"""Command-line interface for querying the MemoryOS index.

Usage:
    python3 -m src.memory.cli search "budget meeting"
    python3 -m src.memory.cli recent --hours 48
    python3 -m src.memory.cli recent --hours 24 --type email
    python3 -m src.memory.cli meetings --date 2026-02-19
    python3 -m src.memory.cli stats
    python3 -m src.memory.cli reindex
    python3 -m src.memory.cli reindex --full
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

from src.common.config import load_config, setup_logging
from src.memory.index import MemoryIndex


def _open_index(cfg: dict) -> MemoryIndex:
    db_path = cfg.get("memory", {}).get("index_db", "config/memory.db")
    return MemoryIndex(db_path)


def cmd_search(args: argparse.Namespace, cfg: dict) -> None:
    query = " ".join(args.query)
    with _open_index(cfg) as idx:
        results = idx.search(
            query,
            source_type=args.type,
            limit=args.limit,
        )
    if not results:
        print(f"No results for: {query}")
        return
    print(f"Found {len(results)} result(s) for: {query}\n")
    for doc in results:
        print(f"  [{doc['tier']:4s}] [{doc['source_type']:10s}] {doc['title']}")
        print(f"         {doc['path']}")
        print(f"         {doc['created_at'][:10]}")
        print()


def cmd_recent(args: argparse.Namespace, cfg: dict) -> None:
    with _open_index(cfg) as idx:
        results = idx.get_recent(hours=args.hours, source_type=args.type, limit=args.limit)
    if not results:
        print(f"No documents in the last {args.hours} hours.")
        return
    print(f"{len(results)} document(s) in the last {args.hours} hours:\n")
    for doc in results:
        print(f"  [{doc['source_type']:10s}] {doc['title']}")
        print(f"         {doc['path']}")
        print()


def cmd_meetings(args: argparse.Namespace, cfg: dict) -> None:
    date = datetime.strptime(args.date, "%Y-%m-%d") if args.date else datetime.now()
    start = date.replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=1)

    with _open_index(cfg) as idx:
        results = idx.get_by_date_range(start, end, source_type="meetings")
    if not results:
        print(f"No meetings found for {start.strftime('%Y-%m-%d')}.")
        return
    print(f"Meetings for {start.strftime('%Y-%m-%d')}:\n")
    for doc in results:
        print(f"  {doc['title']}")
        print(f"    {doc['path']}")
        print()


def cmd_stats(args: argparse.Namespace, cfg: dict) -> None:
    with _open_index(cfg) as idx:
        s = idx.stats()
    print(f"Total indexed documents: {s['total']}\n")
    print("By source type:")
    for src, n in sorted(s["by_type"].items()):
        print(f"  {src:12s}  {n:,}")
    print("\nBy tier:")
    for tier, n in sorted(s["by_tier"].items()):
        print(f"  {tier:6s}  {n:,}")


def cmd_reindex(args: argparse.Namespace, cfg: dict) -> None:
    setup_logging(cfg)
    from src.memory.indexer import scan_vault
    stats = scan_vault(cfg, full=args.full)
    print(f"Reindex complete: {stats}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="memory", description="Query the MemoryOS index")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")

    sub = parser.add_subparsers(dest="command", required=True)

    p_search = sub.add_parser("search", help="Full-text search")
    p_search.add_argument("query", nargs="+", help="Search query")
    p_search.add_argument("--type", default=None, help="Filter by source type")
    p_search.add_argument("--limit", type=int, default=25)

    p_recent = sub.add_parser("recent", help="Recent documents")
    p_recent.add_argument("--hours", type=int, default=24)
    p_recent.add_argument("--type", default=None, help="Filter by source type")
    p_recent.add_argument("--limit", type=int, default=50)

    p_meet = sub.add_parser("meetings", help="Meetings for a date")
    p_meet.add_argument("--date", default=None, help="Date (YYYY-MM-DD), default today")

    sub.add_parser("stats", help="Index statistics")

    p_reidx = sub.add_parser("reindex", help="Rebuild the index from vault files")
    p_reidx.add_argument("--full", action="store_true", help="Reindex all files, not just changed")

    args = parser.parse_args()
    cfg = load_config(args.config)

    dispatch = {
        "search": cmd_search,
        "recent": cmd_recent,
        "meetings": cmd_meetings,
        "stats": cmd_stats,
        "reindex": cmd_reindex,
    }
    dispatch[args.command](args, cfg)


if __name__ == "__main__":
    main()
