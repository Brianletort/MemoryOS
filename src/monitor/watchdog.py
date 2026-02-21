#!/usr/bin/env python3
"""MemoryOS Watchdog -- monitors pipeline health and sends macOS notifications.

Checks each component on every run and notifies on state transitions:
  - HEALTHY -> DEGRADED/DOWN: sends an alert notification
  - DEGRADED/DOWN -> HEALTHY: sends a recovery notification

State is persisted to a JSON file so alerts are not repeated every cycle.
Designed to run every 5 minutes via launchd.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.common.config import load_config, setup_logging

logger = logging.getLogger("memoryos.watchdog")

REPO_DIR = Path(__file__).resolve().parent.parent.parent
WATCHDOG_STATE_FILE = REPO_DIR / "config" / "watchdog_state.json"

STALE_THRESHOLDS = {
    "email": 4 * 3600,
    "meetings": 8 * 3600,
    "activity": 2 * 3600,
    "teams": 24 * 3600,
    "screenpipe_frames": 1800,
    "screenpipe_audio": 1800,
    "dashboard": 30,
}

EXPECTED_AGENTS = [
    "com.memoryos.screenpipe",
    "com.memoryos.outlook",
    "com.memoryos.onedrive",
    "com.memoryos.dashboard",
    "com.memoryos.indexer",
    "com.memoryos.mail-app",
    "com.memoryos.calendar-app",
    "com.memoryos.watchdog",
]


# ── macOS Notifications ──────────────────────────────────────────────────────

def notify(title: str, subtitle: str, message: str) -> None:
    """Send a macOS notification via osascript."""
    script = (
        f'display notification "{_escape(message)}" '
        f'with title "{_escape(title)}" '
        f'subtitle "{_escape(subtitle)}"'
    )
    try:
        subprocess.run(
            ["osascript", "-e", script],
            capture_output=True, timeout=10,
        )
    except Exception as e:
        logger.warning("Failed to send notification: %s", e)


def _escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


# ── State persistence ────────────────────────────────────────────────────────

def load_watchdog_state() -> dict[str, Any]:
    if WATCHDOG_STATE_FILE.is_file():
        try:
            return json.loads(WATCHDOG_STATE_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def save_watchdog_state(state: dict[str, Any]) -> None:
    WATCHDOG_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = WATCHDOG_STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, default=str))
    tmp.rename(WATCHDOG_STATE_FILE)


# ── Health checks ────────────────────────────────────────────────────────────

def check_screenpipe() -> dict[str, str]:
    """Check Screenpipe API health."""
    results = {}
    try:
        import urllib.request
        req = urllib.request.Request(
            "http://localhost:3030/health", method="GET",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())

        frame_status = data.get("frame_status", "unknown")
        audio_status = data.get("audio_status", "unknown")
        drop_rate = data.get("pipeline", {}).get("frame_drop_rate", 0)

        if frame_status == "ok":
            results["screenpipe_frames"] = "healthy"
        elif frame_status == "stale" and drop_rate < 0.95:
            results["screenpipe_frames"] = "stale"
            results["screenpipe_frames_detail"] = (
                f"Vision degraded ({drop_rate:.0%} drop rate)"
            )
        else:
            results["screenpipe_frames"] = "down"
            results["screenpipe_frames_detail"] = data.get("message", "")

        results["screenpipe_audio"] = (
            "healthy" if audio_status == "ok" else "down"
        )
        if audio_status != "ok":
            results["screenpipe_audio_detail"] = data.get("message", "")

    except Exception as e:
        results["screenpipe_frames"] = "down"
        results["screenpipe_audio"] = "down"
        results["screenpipe_frames_detail"] = f"API unreachable: {e}"
    return results


def check_folder_freshness(
    vault_path: Path,
    folder: str,
    threshold_seconds: int,
) -> tuple[str, str]:
    """Check freshness of the newest file in a vault subfolder."""
    folder_path = vault_path / folder
    if not folder_path.is_dir():
        return "down", f"Folder {folder} does not exist"

    newest_mtime = 0.0
    newest_name = ""
    for f in folder_path.rglob("*.md"):
        mt = f.stat().st_mtime
        if mt > newest_mtime:
            newest_mtime = mt
            newest_name = str(f.relative_to(vault_path))

    if newest_mtime == 0:
        return "down", f"No files in {folder}"

    age = time.time() - newest_mtime
    if age > threshold_seconds:
        hours = age / 3600
        return "stale", f"Newest: {newest_name} ({hours:.1f}h ago)"
    return "healthy", newest_name


def check_dashboard() -> tuple[str, str]:
    """Check if the dashboard is responding."""
    try:
        import urllib.request
        req = urllib.request.Request(
            "http://localhost:8765/api/status", method="GET",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status == 200:
                return "healthy", "Dashboard responding"
    except Exception as e:
        return "down", f"Dashboard unreachable: {e}"
    return "down", "Non-200 response"


def check_launchd_agents() -> tuple[str, str]:
    """Check that all expected launchd agents are loaded."""
    try:
        result = subprocess.run(
            ["launchctl", "list"],
            capture_output=True, text=True, timeout=10,
        )
        loaded = result.stdout
    except Exception:
        return "down", "Cannot query launchctl"

    missing = [a for a in EXPECTED_AGENTS if a not in loaded]
    if missing:
        return "stale", f"Missing agents: {', '.join(missing)}"
    return "healthy", "All agents loaded"


# ── Orchestrator ─────────────────────────────────────────────────────────────

def run_checks(cfg: dict[str, Any]) -> dict[str, dict[str, str]]:
    """Run all health checks and return results."""
    vault = Path(cfg["obsidian_vault"])
    results: dict[str, dict[str, str]] = {}

    sp = check_screenpipe()
    results["screenpipe_frames"] = {
        "status": sp["screenpipe_frames"],
        "detail": sp.get("screenpipe_frames_detail", ""),
    }
    results["screenpipe_audio"] = {
        "status": sp["screenpipe_audio"],
        "detail": sp.get("screenpipe_audio_detail", ""),
    }

    for component, folder, threshold in [
        ("email", cfg["output"]["email"], STALE_THRESHOLDS["email"]),
        ("meetings", cfg["output"]["meetings"], STALE_THRESHOLDS["meetings"]),
        ("activity", cfg["output"]["activity"], STALE_THRESHOLDS["activity"]),
        ("teams", cfg["output"]["teams"], STALE_THRESHOLDS["teams"]),
    ]:
        status, detail = check_folder_freshness(vault, folder, threshold)
        results[component] = {"status": status, "detail": detail}

    status, detail = check_dashboard()
    results["dashboard"] = {"status": status, "detail": detail}

    status, detail = check_launchd_agents()
    results["launchd"] = {"status": status, "detail": detail}

    return results


FRIENDLY_NAMES = {
    "screenpipe_frames": "Screen Recording",
    "screenpipe_audio": "Audio Transcription",
    "email": "Email Pipeline",
    "meetings": "Meeting Transcripts",
    "activity": "Activity Tracking",
    "teams": "Teams Chat",
    "dashboard": "Dashboard",
    "launchd": "Background Agents",
}


def evaluate_and_notify(results: dict[str, dict[str, str]]) -> None:
    """Compare with previous state, notify on transitions."""
    prev_state = load_watchdog_state()
    prev_statuses = prev_state.get("statuses", {})
    new_statuses: dict[str, str] = {}
    alerts: list[str] = []
    recoveries: list[str] = []

    for component, result in results.items():
        status = result["status"]
        new_statuses[component] = status
        prev = prev_statuses.get(component, "unknown")
        name = FRIENDLY_NAMES.get(component, component)

        if prev in ("healthy", "unknown") and status in ("down", "stale"):
            alerts.append(f"{name}: {result['detail']}")
            logger.warning("ALERT: %s is %s -- %s", name, status, result["detail"])
        elif prev in ("down", "stale") and status == "healthy":
            recoveries.append(name)
            logger.info("RECOVERY: %s is healthy again", name)

    if alerts:
        body = "\n".join(alerts[:4])
        notify(
            "MemoryOS Alert",
            f"{len(alerts)} component(s) need attention",
            body,
        )

    if recoveries:
        notify(
            "MemoryOS Recovered",
            f"{len(recoveries)} component(s) back online",
            ", ".join(recoveries),
        )

    overall = "healthy"
    for s in new_statuses.values():
        if s == "down":
            overall = "down"
            break
        if s == "stale":
            overall = "degraded"

    save_watchdog_state({
        "statuses": new_statuses,
        "results": {k: v for k, v in results.items()},
        "overall": overall,
        "last_check": datetime.now(timezone.utc).isoformat(),
        "alerts_sent": len(alerts),
        "recoveries_sent": len(recoveries),
    })

    logger.info(
        "Watchdog check complete: overall=%s, alerts=%d, recoveries=%d",
        overall, len(alerts), len(recoveries),
    )


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="MemoryOS Watchdog")
    parser.add_argument("--config", help="Path to config.yaml")
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress notifications (check only, update state)",
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Print current status and exit",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg)

    if args.status:
        state = load_watchdog_state()
        if not state:
            print("No watchdog state yet. Run without --status first.")
            return
        print(f"Overall: {state.get('overall', 'unknown')}")
        print(f"Last check: {state.get('last_check', 'never')}")
        for comp, status in state.get("statuses", {}).items():
            name = FRIENDLY_NAMES.get(comp, comp)
            detail = state.get("results", {}).get(comp, {}).get("detail", "")
            indicator = {"healthy": "+", "stale": "~", "down": "!"}
            print(f"  [{indicator.get(status, '?')}] {name}: {status}  {detail}")
        return

    results = run_checks(cfg)

    if args.quiet:
        save_watchdog_state({
            "statuses": {k: v["status"] for k, v in results.items()},
            "results": results,
            "overall": "check-only",
            "last_check": datetime.now(timezone.utc).isoformat(),
        })
        for comp, result in results.items():
            name = FRIENDLY_NAMES.get(comp, comp)
            print(f"  [{result['status']}] {name}: {result['detail']}")
    else:
        evaluate_and_notify(results)


if __name__ == "__main__":
    main()
