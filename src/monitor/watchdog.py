#!/usr/bin/env python3
"""MemoryOS Watchdog -- monitors pipeline health and sends macOS notifications.

Checks each component on every run and notifies on state transitions:
  - HEALTHY -> DEGRADED/DOWN: sends an alert notification
  - DEGRADED/DOWN -> HEALTHY: sends a recovery notification

State is persisted to a JSON file so alerts are not repeated every cycle.
Runs as a persistent daemon (KeepAlive via launchd) with a 300s internal loop.
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
    "dashboard": 60,
}

SCREENPIPE_DB_PATH = Path.home() / ".screenpipe" / "db.sqlite"
SCREENPIPE_DB_STALE_SECONDS = 600

ESCALATION_THRESHOLDS = [
    (5 * 60, "MemoryOS: data collection down 5 min", "Attempting automatic recovery"),
    (15 * 60, "DATA COLLECTION DOWN 15 min", "May need manual intervention"),
    (30 * 60, "DATA COLLECTION DOWN 30 min", "You are losing data. Consider restarting your machine."),
]

EXPECTED_AGENTS = [
    "com.memoryos.screenpipe",
    "com.memoryos.outlook",
    "com.memoryos.onedrive",
    "com.memoryos.dashboard",
    "com.memoryos.indexer",
    "com.memoryos.mail-app",
    "com.memoryos.calendar-app",
    "com.memoryos.watchdog",
    "com.memoryos.sentinel",
    "com.memoryos.resource-monitor",
    "com.memoryos.activity-summarizer",
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


def check_screenpipe_db_freshness() -> tuple[str, str]:
    """Check Screenpipe DB for actual data freshness (catches healthy-API-but-no-data)."""
    if not SCREENPIPE_DB_PATH.is_file():
        return "down", "Screenpipe DB not found"
    try:
        import sqlite3
        conn = sqlite3.connect(str(SCREENPIPE_DB_PATH), timeout=5)
        row = conn.execute("SELECT MAX(timestamp) FROM frames").fetchone()
        conn.close()
        if not row or not row[0]:
            return "down", "No frames in Screenpipe DB"
        from datetime import datetime, timezone
        ts = row[0]
        if "+" in ts or ts.endswith("Z"):
            last_frame = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        else:
            last_frame = datetime.fromisoformat(ts).replace(tzinfo=timezone.utc)
        age = (datetime.now(timezone.utc) - last_frame).total_seconds()
        if age > SCREENPIPE_DB_STALE_SECONDS:
            return "stale", f"Last frame {age/60:.0f}m ago (DB stale)"
        return "healthy", f"Last frame {age:.0f}s ago"
    except Exception as e:
        return "stale", f"Cannot query Screenpipe DB: {e}"


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
    """Check if the dashboard is responding (retries once on timeout)."""
    import urllib.request
    import time as _t

    for attempt in range(2):
        try:
            req = urllib.request.Request(
                "http://localhost:8765/api/status", method="GET",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status == 200:
                    return "healthy", "Dashboard responding"
            return "down", "Non-200 response"
        except Exception as e:
            if attempt == 0:
                _t.sleep(3)
                continue
            return "down", f"Dashboard unreachable: {e}"
    return "down", "Dashboard unreachable after retries"


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


RESOURCE_MONITOR_STATE_FILE = REPO_DIR / "config" / "resource_monitor_state.json"


def check_resource_health() -> tuple[str, str]:
    """Check system resource health via the resource monitor's state file."""
    if not RESOURCE_MONITOR_STATE_FILE.is_file():
        return "unknown", "Resource monitor not yet running"
    try:
        data = json.loads(RESOURCE_MONITOR_STATE_FILE.read_text())
        age = time.time() - datetime.fromisoformat(data["last_check"]).timestamp()
        if age > 300:
            return "stale", f"Resource monitor data is {age / 60:.0f}m old"
        level = data.get("level", "unknown")
        if level == "critical":
            alerts = data.get("alerts", [])
            critical = [a["message"] for a in alerts if a.get("severity") == "critical"]
            return "down", "; ".join(critical[:2]) or "Critical resource pressure"
        if level == "warning":
            alerts = data.get("alerts", [])
            warnings = [a["message"] for a in alerts if a.get("severity") == "warning"]
            return "stale", "; ".join(warnings[:2]) or "Resource usage elevated"
        sys_info = data.get("system", {})
        ram = sys_info.get("ram_available_gb", "?")
        return "healthy", f"RAM: {ram} GB free"
    except Exception as e:
        return "stale", f"Cannot read resource state: {e}"


# ── Active Recovery ───────────────────────────────────────────────────────────

RECOVERY_COOLDOWN = 120  # 2 min between recovery attempts


def _recovery_allowed(component: str, prev_state: dict[str, Any]) -> bool:
    """Check if enough time has passed since the last recovery attempt."""
    last = prev_state.get("recovery_attempts", {}).get(component, 0)
    return (time.time() - last) > RECOVERY_COOLDOWN


def _record_recovery(component: str, prev_state: dict[str, Any]) -> None:
    prev_state.setdefault("recovery_attempts", {})[component] = time.time()


def _recover_app(app_name: str, bundle_pattern: str) -> bool:
    """Attempt to launch a macOS app if it isn't running."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", bundle_pattern],
            capture_output=True, timeout=5,
        )
        if result.returncode == 0:
            return False  # already running
    except Exception:
        pass
    logger.info("Recovery: launching %s", app_name)
    try:
        subprocess.run(["open", "-a", app_name], timeout=10, capture_output=True)
        return True
    except Exception as e:
        logger.warning("Recovery: failed to launch %s: %s", app_name, e)
        return False


def _activate_preferred_audio_device() -> None:
    """Activate the preferred audio input device and stop all others.

    Screenpipe's device_monitor re-adds devices when system defaults change,
    causing USB reconnect sounds.  After starting the preferred device we
    explicitly stop every other input device to prevent cascading restarts.
    """
    try:
        cfg = load_config()
        preferred = cfg.get("screenpipe", {}).get("preferred_input_device")
    except Exception:
        return
    if not preferred:
        return

    import urllib.request

    activated = False
    for attempt in range(6):
        time.sleep(5)
        try:
            data = json.dumps({"device_name": preferred}).encode()
            req = urllib.request.Request(
                "http://localhost:3030/audio/device/start",
                data=data, method="POST",
            )
            req.add_header("Content-Type", "application/json")
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status == 200:
                    logger.info("Activated preferred audio device: %s", preferred)
                    activated = True
                    break
        except Exception:
            pass

    if not activated:
        logger.warning("Failed to activate preferred audio device '%s' after restart", preferred)
        return

    try:
        req = urllib.request.Request("http://localhost:3030/audio/list", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            devices = json.loads(resp.read())
        stopped = 0
        for dev in devices:
            name = dev.get("name", "")
            if not name or "(output)" in name or name == preferred:
                continue
            try:
                stop_data = json.dumps({"device_name": name}).encode()
                stop_req = urllib.request.Request(
                    "http://localhost:3030/audio/device/stop",
                    data=stop_data, method="POST",
                )
                stop_req.add_header("Content-Type", "application/json")
                with urllib.request.urlopen(stop_req, timeout=10):
                    stopped += 1
            except Exception:
                pass
        if stopped:
            logger.info("Stopped %d unwanted input device(s)", stopped)
    except Exception:
        pass


def _recover_screenpipe() -> bool:
    """Kill stale Screenpipe and relaunch, then activate preferred audio device."""
    logger.info("Recovery: restarting Screenpipe")
    try:
        subprocess.run(
            ["pkill", "-f", "screenpipe-app"], timeout=10, capture_output=True,
        )
        time.sleep(5)
        subprocess.run(
            ["open", "-a", "screenpipe"], timeout=10, capture_output=True,
        )
        time.sleep(10)
        _activate_preferred_audio_device()
        return True
    except Exception as e:
        logger.warning("Recovery: Screenpipe restart failed: %s", e)
        return False


def _recover_mail_app() -> bool:
    """Force-restart Mail.app to clear hung AppleEvent state."""
    logger.info("Recovery: restarting Mail.app")
    try:
        subprocess.run(
            ["pkill", "-f", "Mail.app/Contents/MacOS/Mail"],
            timeout=10, capture_output=True,
        )
        time.sleep(5)
        subprocess.run(["open", "-a", "Mail"], timeout=10, capture_output=True)
        return True
    except Exception as e:
        logger.warning("Recovery: Mail.app restart failed: %s", e)
        return False


def _recover_calendar_app() -> bool:
    """Force-restart Calendar.app to clear hung AppleEvent state."""
    logger.info("Recovery: restarting Calendar.app")
    try:
        subprocess.run(
            ["pkill", "-f", "Calendar.app/Contents/MacOS/Calendar"],
            timeout=10, capture_output=True,
        )
        time.sleep(5)
        subprocess.run(["open", "-a", "Calendar"], timeout=10, capture_output=True)
        return True
    except Exception as e:
        logger.warning("Recovery: Calendar.app restart failed: %s", e)
        return False


def _recover_dashboard() -> bool:
    """Restart the MemoryOS dashboard."""
    logger.info("Recovery: restarting dashboard")
    try:
        subprocess.run(
            ["pkill", "-f", "src/dashboard/app.py"],
            timeout=10, capture_output=True,
        )
        time.sleep(2)
        venv_python = REPO_DIR / ".venv" / "bin" / "python"
        python_cmd = str(venv_python) if venv_python.exists() else "python3"
        subprocess.Popen(
            [python_cmd, str(REPO_DIR / "src" / "dashboard" / "app.py")],
            cwd=str(REPO_DIR),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except Exception as e:
        logger.warning("Recovery: dashboard restart failed: %s", e)
        return False


LAUNCHD_AGENTS_DIR = Path.home() / "Library" / "LaunchAgents"


def _recover_launchd_agents(detail: str) -> list[str]:
    """Reload missing launchd agents from installed plist files."""
    reloaded: list[str] = []
    for agent in EXPECTED_AGENTS:
        if agent not in detail:
            continue
        if agent == "com.memoryos.watchdog":
            continue
        plist = LAUNCHD_AGENTS_DIR / f"{agent}.plist"
        if not plist.exists():
            logger.warning("Cannot reload %s: plist not found", agent)
            continue
        try:
            subprocess.run(
                ["launchctl", "load", str(plist)],
                capture_output=True, text=True, timeout=10,
            )
            reloaded.append(agent)
            logger.info("Recovery: reloaded agent %s", agent)
        except Exception as e:
            logger.warning("Recovery: failed to reload %s: %s", agent, e)
    return reloaded


def attempt_recovery(
    results: dict[str, dict[str, str]],
    prev_state: dict[str, Any],
) -> list[str]:
    """Attempt self-healing actions for down/stale components.

    Returns a list of components where recovery was attempted.
    """
    recovered: list[str] = []

    sp_frames = results.get("screenpipe_frames", {}).get("status", "")
    sp_audio = results.get("screenpipe_audio", {}).get("status", "")
    if sp_frames == "down" or sp_audio == "down":
        if _recovery_allowed("screenpipe", prev_state):
            if _recover_screenpipe():
                _record_recovery("screenpipe", prev_state)
                recovered.append("Screenpipe")

    email_status = results.get("email", {}).get("status", "")
    if email_status in ("down", "stale"):
        if _recovery_allowed("mail_app", prev_state):
            if _recover_app("Mail", r"Mail.app/Contents/MacOS/Mail$"):
                _record_recovery("mail_app", prev_state)
                recovered.append("Mail.app (launched)")
            elif _recover_mail_app():
                _record_recovery("mail_app", prev_state)
                recovered.append("Mail.app (restarted)")

    meetings_status = results.get("meetings", {}).get("status", "")
    if meetings_status in ("down", "stale"):
        if _recovery_allowed("calendar_app", prev_state):
            if _recover_app("Calendar", r"Calendar.app/Contents/MacOS/Calendar$"):
                _record_recovery("calendar_app", prev_state)
                recovered.append("Calendar.app (launched)")
            elif _recover_calendar_app():
                _record_recovery("calendar_app", prev_state)
                recovered.append("Calendar.app (restarted)")

    dash_status = results.get("dashboard", {}).get("status", "")
    if dash_status == "down":
        if _recovery_allowed("dashboard", prev_state):
            if _recover_dashboard():
                _record_recovery("dashboard", prev_state)
                recovered.append("Dashboard")

    activity_status = results.get("activity", {}).get("status", "")
    if activity_status in ("down", "stale"):
        if _recovery_allowed("activity", prev_state):
            uid = os.getuid()
            restarted: list[str] = []
            for agent in ("com.memoryos.screenpipe", "com.memoryos.activity-summarizer"):
                try:
                    subprocess.run(
                        ["launchctl", "kickstart", "-k", f"gui/{uid}/{agent}"],
                        capture_output=True, timeout=15,
                    )
                    restarted.append(agent)
                except Exception as e:
                    logger.warning("Recovery: failed to kickstart %s: %s", agent, e)
            if restarted:
                _record_recovery("activity", prev_state)
                recovered.append(f"Activity pipeline ({', '.join(restarted)})")

    launchd_status = results.get("launchd", {}).get("status", "")
    if launchd_status in ("stale", "down"):
        if _recovery_allowed("launchd", prev_state):
            detail = results.get("launchd", {}).get("detail", "")
            reloaded = _recover_launchd_agents(detail)
            if reloaded:
                _record_recovery("launchd", prev_state)
                recovered.extend(f"Agent: {a}" for a in reloaded)

    return recovered


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

    db_status, db_detail = check_screenpipe_db_freshness()
    if db_status != "healthy" and results["screenpipe_frames"]["status"] == "healthy":
        results["screenpipe_frames"]["status"] = db_status
        results["screenpipe_frames"]["detail"] = db_detail
        logger.warning("Screenpipe API healthy but DB stale: %s", db_detail)

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

    status, detail = check_resource_health()
    results["resources"] = {"status": status, "detail": detail}

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
    "resources": "System Resources",
}


def evaluate_and_notify(results: dict[str, dict[str, str]]) -> None:
    """Compare with previous state, notify on transitions, and attempt recovery."""
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

    # ── Active recovery ──
    healed = attempt_recovery(results, prev_state)
    if healed:
        logger.info("Self-healing attempted: %s", ", ".join(healed))
        notify(
            "MemoryOS Self-Healing",
            f"Restarted {len(healed)} component(s)",
            ", ".join(healed),
        )

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

    # ── Escalation: track how long critical collectors have been unhealthy ──
    critical_components = ("screenpipe_frames", "screenpipe_audio")
    now_ts = time.time()
    down_since = prev_state.get("down_since", {})
    last_escalation_level = prev_state.get("last_escalation_level", {})

    for comp in critical_components:
        status = new_statuses.get(comp, "healthy")
        if status in ("down", "stale"):
            if comp not in down_since:
                down_since[comp] = now_ts
            down_duration = now_ts - down_since[comp]
            for threshold_secs, title, message in ESCALATION_THRESHOLDS:
                prev_level = last_escalation_level.get(comp, 0)
                if down_duration >= threshold_secs and prev_level < threshold_secs:
                    name = FRIENDLY_NAMES.get(comp, comp)
                    notify(title, name, message)
                    logger.warning(
                        "ESCALATION: %s down for %dm -- %s",
                        name, down_duration / 60, message,
                    )
                    last_escalation_level[comp] = threshold_secs
        else:
            if comp in down_since:
                del down_since[comp]
            if comp in last_escalation_level:
                del last_escalation_level[comp]

    save_watchdog_state({
        "statuses": new_statuses,
        "results": {k: v for k, v in results.items()},
        "overall": overall,
        "last_check": datetime.now(timezone.utc).isoformat(),
        "alerts_sent": len(alerts),
        "recoveries_sent": len(recoveries),
        "self_healed": healed,
        "recovery_attempts": prev_state.get("recovery_attempts", {}),
        "down_since": down_since,
        "last_escalation_level": last_escalation_level,
    })

    logger.info(
        "Watchdog check complete: overall=%s, alerts=%d, recoveries=%d, healed=%d",
        overall, len(alerts), len(recoveries), len(healed),
    )


# ── CLI ──────────────────────────────────────────────────────────────────────

DAEMON_INTERVAL = 300  # seconds between check cycles


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
    parser.add_argument(
        "--once", action="store_true",
        help="Run a single check cycle then exit (legacy mode)",
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

    if args.once or args.quiet:
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
        return

    logger.info("Watchdog starting as persistent daemon (interval=%ds)", DAEMON_INTERVAL)
    while True:
        try:
            results = run_checks(cfg)
            evaluate_and_notify(results)
        except Exception:
            logger.exception("Watchdog cycle failed, will retry next cycle")
        time.sleep(DAEMON_INTERVAL)


if __name__ == "__main__":
    main()
