#!/usr/bin/env python3
"""Manage Screenpipe speaker identities.

List unnamed speakers with sample transcripts, name them, find similar voices,
and merge duplicates -- all via the Screenpipe REST API.

Usage:
    python3 scripts/name_speakers.py list              # show unnamed speakers
    python3 scripts/name_speakers.py list --all        # show all speakers
    python3 scripts/name_speakers.py name 17 "Sean"    # name speaker 17
    python3 scripts/name_speakers.py similar 17        # find voices like speaker 17
    python3 scripts/name_speakers.py merge 5 3         # merge speaker 5 into speaker 3
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
import urllib.request
import urllib.error
from typing import Any

BASE_URL = "http://localhost:3030"


def _api_get(path: str, params: dict[str, Any] | None = None, timeout: int = 30) -> Any:
    url = f"{BASE_URL}{path}"
    if params:
        qs = "&".join(f"{k}={v}" for k, v in params.items() if v is not None)
        url = f"{url}?{qs}"
    req = urllib.request.Request(url)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except (urllib.error.URLError, TimeoutError) as exc:
        print(f"Error: cannot reach Screenpipe at {url}")
        print(f"  {exc}")
        sys.exit(1)


def _api_post(path: str, body: dict[str, Any]) -> Any:
    url = f"{BASE_URL}{path}"
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except (urllib.error.URLError, TimeoutError) as exc:
        print(f"Error: API call failed -- {exc}")
        sys.exit(1)


def _parse_samples(metadata_str: str) -> list[dict[str, Any]]:
    """Extract audio_samples from speaker metadata JSON."""
    if not metadata_str:
        return []
    try:
        meta = json.loads(metadata_str)
        return meta.get("audio_samples", [])
    except (json.JSONDecodeError, TypeError):
        return []


def _format_sample(sample: dict[str, Any], indent: int = 4) -> str:
    transcript = sample.get("transcript", "").strip()
    path = sample.get("path", "")
    prefix = " " * indent
    lines = [f'{prefix}"{transcript}"']
    if path:
        lines.append(f"{prefix}  file: {path}")
    return "\n".join(lines)


def cmd_list(args: argparse.Namespace) -> None:
    """List speakers with sample transcripts."""
    if args.all:
        speakers = _api_get("/speakers/search")
    else:
        speakers = _api_get("/speakers/unnamed", {"limit": 50, "offset": 0})

    if not speakers:
        print("No speakers found.")
        return

    for spk in speakers:
        sid = spk["id"]
        name = spk.get("name") or "(unnamed)"
        samples = _parse_samples(spk.get("metadata", ""))

        print(f"\n  Speaker {sid}: {name}")
        print(f"  {'─' * 40}")
        if samples:
            for s in samples[:3]:
                print(_format_sample(s))
        else:
            print("    (no audio samples)")

    print(f"\n  Total: {len(speakers)} speaker(s)")
    if not args.all:
        print("  Use --all to include named speakers.")
    print("  Use 'name <id> <name>' to assign a name.")


def cmd_name(args: argparse.Namespace) -> None:
    """Name a speaker."""
    result = _api_post("/speakers/update", {"id": args.speaker_id, "name": args.name})
    name = result.get("name", args.name)
    print(f"  Speaker {args.speaker_id} named: {name}")


def cmd_similar(args: argparse.Namespace) -> None:
    """Find speakers with similar voice prints."""
    results = _api_get(
        "/speakers/similar",
        {"speaker_id": args.speaker_id, "limit": 5},
        timeout=60,
    )
    if not results:
        print(f"  No similar speakers found for speaker {args.speaker_id}.")
        return

    print(f"\n  Speakers similar to speaker {args.speaker_id}:")
    for item in results:
        spk = item if isinstance(item, dict) and "id" in item else {}
        sid = spk.get("id", "?")
        name = spk.get("name") or "(unnamed)"
        score = spk.get("similarity_score", spk.get("score", ""))
        score_str = f" (similarity: {score:.3f})" if isinstance(score, (int, float)) else ""
        print(f"    Speaker {sid}: {name}{score_str}")


def cmd_merge(args: argparse.Namespace) -> None:
    """Merge two speakers (keep first, merge second into it)."""
    result = _api_post("/speakers/merge", {
        "speaker_to_keep_id": args.keep_id,
        "speaker_to_merge_id": args.merge_id,
    })
    if result.get("success"):
        print(f"  Merged speaker {args.merge_id} into speaker {args.keep_id}.")
    else:
        print(f"  Merge failed: {result}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Manage Screenpipe speaker identities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            examples:
              %(prog)s list              Show unnamed speakers with samples
              %(prog)s list --all        Show all speakers (named + unnamed)
              %(prog)s name 17 "Sean"    Name speaker 17 as Sean
              %(prog)s similar 17        Find voices similar to speaker 17
              %(prog)s merge 5 3         Merge speaker 5 into speaker 3
        """),
    )
    sub = parser.add_subparsers(dest="command")

    p_list = sub.add_parser("list", help="List speakers")
    p_list.add_argument("--all", action="store_true", help="Include named speakers")

    p_name = sub.add_parser("name", help="Name a speaker")
    p_name.add_argument("speaker_id", type=int, help="Speaker ID")
    p_name.add_argument("name", help="Name to assign")

    p_sim = sub.add_parser("similar", help="Find similar speakers")
    p_sim.add_argument("speaker_id", type=int, help="Speaker ID to compare")

    p_merge = sub.add_parser("merge", help="Merge two speakers")
    p_merge.add_argument("keep_id", type=int, help="Speaker ID to keep")
    p_merge.add_argument("merge_id", type=int, help="Speaker ID to merge into the kept one")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)

    dispatch = {
        "list": cmd_list,
        "name": cmd_name,
        "similar": cmd_similar,
        "merge": cmd_merge,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
