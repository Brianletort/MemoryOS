#!/usr/bin/env python3
"""MemoryOS Resource Monitor -- tracks system RAM/CPU/swap and per-process memory.

Runs as a persistent daemon (60s interval via launchd) to:
  - Monitor system-wide memory pressure, swap, CPU, and disk
  - Track per-process RSS for MemoryOS services and key apps
  - Detect memory leaks via rolling-window RSS growth analysis
  - Proactively restart bloated processes before the system locks up
  - Send macOS notifications on warning/critical thresholds

State is persisted to config/resource_monitor_state.json.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.common.config import load_config, setup_logging

logger = logging.getLogger("memoryos.resource_monitor")

REPO_DIR = Path(__file__).resolve().parent.parent.parent
STATE_FILE = REPO_DIR / "config" / "resource_monitor_state.json"

# ── Defaults (overridden by config.yaml resource_monitor section) ────────────

DEFAULT_INTERVAL = 60
DEFAULT_HISTORY_WINDOW = 20

DEFAULT_THRESHOLDS: dict[str, float] = {
    "system_ram_warning_gb": 2.0,
    "system_ram_critical_gb": 1.0,
    "process_rss_warning_mb": 500,
    "process_rss_critical_mb": 800,
    "dashboard_rss_critical_mb": 600,
    "unmanaged_rss_warning_mb": 2000,
    "unmanaged_rss_critical_mb": 4000,
    "swap_warning_gb": 4.0,
    "swap_critical_gb": 8.0,
    "disk_warning_gb": 10.0,
    "disk_critical_gb": 5.0,
    "rss_growth_rate_mb_per_10min": 50,
}

RECOVERY_COOLDOWN = 600  # seconds between recovery attempts per component
MAX_RECOVERIES_PER_CYCLE = 2

MONITORED_PROCESSES: dict[str, dict[str, Any]] = {
    "dashboard": {
        "pattern": "src/dashboard/app.py",
        "agent": "com.memoryos.dashboard",
    },
    "screenpipe_ext": {
        "pattern": "screenpipe_extractor",
        "agent": "com.memoryos.screenpipe",
    },
    "mail_ext": {
        "pattern": "mail_app_extractor",
        "agent": "com.memoryos.mail-app",
    },
    "calendar_ext": {
        "pattern": "calendar_app_extractor",
        "agent": "com.memoryos.calendar-app",
    },
    "outlook_ext": {
        "pattern": "outlook_extractor",
        "agent": "com.memoryos.outlook",
    },
    "onedrive_ext": {
        "pattern": "onedrive_extractor",
        "agent": "com.memoryos.onedrive",
    },
    "indexer": {
        "pattern": "src/memory/indexer",
        "agent": "com.memoryos.indexer",
    },
    "watchdog": {
        "pattern": "watchdog.py",
        "agent": "com.memoryos.watchdog",
    },
    "mail_app": {
        "pattern": "Mail.app/Contents/MacOS/Mail",
        "agent": None,
    },
    "screenpipe_app": {
        "pattern": "screenpipe-app",
        "agent": None,
    },
}

# ── macOS Notifications ──────────────────────────────────────────────────────


def notify(title: str, subtitle: str, message: str) -> None:
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


def load_state() -> dict[str, Any]:
    if STATE_FILE.is_file():
        try:
            return json.loads(STATE_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def save_state(state: dict[str, Any]) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, default=str))
    tmp.rename(STATE_FILE)


# ── Threshold helpers ────────────────────────────────────────────────────────


def _get_thresholds(cfg: dict[str, Any]) -> dict[str, float]:
    rm_cfg = cfg.get("resource_monitor", {})
    configured = rm_cfg.get("thresholds", {})
    merged = dict(DEFAULT_THRESHOLDS)
    merged.update(configured)
    return merged


def _get_interval(cfg: dict[str, Any]) -> int:
    return cfg.get("resource_monitor", {}).get("interval_seconds", DEFAULT_INTERVAL)


def _get_history_window(cfg: dict[str, Any]) -> int:
    return cfg.get("resource_monitor", {}).get("history_window", DEFAULT_HISTORY_WINDOW)


# ── Data collection ──────────────────────────────────────────────────────────


def collect_system_metrics() -> dict[str, Any]:
    vm = psutil.virtual_memory()
    sw = psutil.swap_memory()
    cpu_pct = psutil.cpu_percent(interval=1)

    return {
        "ram_total_gb": round(vm.total / (1024**3), 2),
        "ram_available_gb": round(vm.available / (1024**3), 2),
        "ram_used_gb": round(vm.used / (1024**3), 2),
        "ram_percent": vm.percent,
        "swap_total_gb": round(sw.total / (1024**3), 2),
        "swap_used_gb": round(sw.used / (1024**3), 2),
        "swap_percent": sw.percent,
        "cpu_percent": cpu_pct,
    }


def collect_disk_metrics(vault_path: str) -> dict[str, Any]:
    try:
        usage = psutil.disk_usage(vault_path)
        return {
            "disk_total_gb": round(usage.total / (1024**3), 2),
            "disk_free_gb": round(usage.free / (1024**3), 2),
            "disk_percent": usage.percent,
        }
    except Exception as e:
        logger.warning("Disk metrics unavailable: %s", e)
        return {"disk_total_gb": 0, "disk_free_gb": 0, "disk_percent": 0}


def collect_process_metrics() -> dict[str, dict[str, Any]]:
    """Scan running processes and match against monitored patterns."""
    results: dict[str, dict[str, Any]] = {}

    for proc in psutil.process_iter(["pid", "name", "cmdline", "memory_info", "cpu_percent"]):
        try:
            cmdline = " ".join(proc.info.get("cmdline") or [])
            if not cmdline:
                continue
            for name, spec in MONITORED_PROCESSES.items():
                if name in results:
                    continue
                if spec["pattern"] in cmdline:
                    mem = proc.info.get("memory_info")
                    results[name] = {
                        "pid": proc.info["pid"],
                        "rss_mb": round(mem.rss / (1024**2), 1) if mem else 0,
                        "cpu_percent": proc.info.get("cpu_percent", 0),
                    }
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue

    return results


# ── Trending / leak detection ────────────────────────────────────────────────


_rss_history: dict[str, deque[tuple[float, float]]] = {}


def update_rss_history(
    process_metrics: dict[str, dict[str, Any]],
    window: int,
) -> dict[str, float]:
    """Append current RSS to rolling window. Return growth rate (MB/10min) per process."""
    now = time.time()
    growth_rates: dict[str, float] = {}

    for name, info in process_metrics.items():
        if name not in _rss_history:
            _rss_history[name] = deque(maxlen=window)
        _rss_history[name].append((now, info["rss_mb"]))

        history = _rss_history[name]
        if len(history) >= 5:
            t_start, rss_start = history[0]
            t_end, rss_end = history[-1]
            elapsed_min = (t_end - t_start) / 60
            if elapsed_min > 0:
                rate_per_10min = ((rss_end - rss_start) / elapsed_min) * 10
                growth_rates[name] = round(rate_per_10min, 1)

    return growth_rates


# ── Evaluation ───────────────────────────────────────────────────────────────


def evaluate(
    system: dict[str, Any],
    disk: dict[str, Any],
    processes: dict[str, dict[str, Any]],
    growth_rates: dict[str, float],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    """Evaluate metrics against thresholds. Return alerts and recommended actions."""
    alerts: list[dict[str, str]] = []
    actions: list[dict[str, str]] = []
    level = "healthy"

    ram_avail = system["ram_available_gb"]
    if ram_avail < thresholds["system_ram_critical_gb"]:
        level = "critical"
        alerts.append({
            "component": "system_ram",
            "severity": "critical",
            "message": f"Available RAM critically low: {ram_avail:.1f} GB",
        })
        actions.append({"action": "restart_largest", "reason": "critical RAM pressure"})
    elif ram_avail < thresholds["system_ram_warning_gb"]:
        level = max(level, "warning", key=_severity_rank)
        alerts.append({
            "component": "system_ram",
            "severity": "warning",
            "message": f"Available RAM low: {ram_avail:.1f} GB",
        })

    swap_used = system["swap_used_gb"]
    if swap_used > thresholds["swap_critical_gb"]:
        level = max(level, "critical", key=_severity_rank)
        alerts.append({
            "component": "swap",
            "severity": "critical",
            "message": f"Swap usage very high: {swap_used:.1f} GB",
        })
    elif swap_used > thresholds["swap_warning_gb"]:
        level = max(level, "warning", key=_severity_rank)
        alerts.append({
            "component": "swap",
            "severity": "warning",
            "message": f"Swap usage elevated: {swap_used:.1f} GB",
        })

    disk_free = disk["disk_free_gb"]
    if disk_free < thresholds["disk_critical_gb"]:
        level = max(level, "critical", key=_severity_rank)
        alerts.append({
            "component": "disk",
            "severity": "critical",
            "message": f"Disk space critically low: {disk_free:.1f} GB free",
        })
    elif disk_free < thresholds["disk_warning_gb"]:
        level = max(level, "warning", key=_severity_rank)
        alerts.append({
            "component": "disk",
            "severity": "warning",
            "message": f"Disk space low: {disk_free:.1f} GB free",
        })

    for name, info in processes.items():
        rss = info["rss_mb"]
        is_dashboard = name == "dashboard"
        is_unmanaged = MONITORED_PROCESSES.get(name, {}).get("agent") is None

        if is_dashboard:
            critical_mb = thresholds["dashboard_rss_critical_mb"]
            warning_mb = thresholds["process_rss_warning_mb"]
        elif is_unmanaged:
            critical_mb = thresholds["unmanaged_rss_critical_mb"]
            warning_mb = thresholds["unmanaged_rss_warning_mb"]
        else:
            critical_mb = thresholds["process_rss_critical_mb"]
            warning_mb = thresholds["process_rss_warning_mb"]

        if rss > critical_mb:
            level = max(level, "critical", key=_severity_rank)
            alerts.append({
                "component": name,
                "severity": "critical",
                "message": f"{name} RSS critical: {rss:.0f} MB (limit {critical_mb:.0f} MB)",
            })
            actions.append({"action": "restart_process", "target": name, "reason": "RSS critical"})
        elif rss > warning_mb:
            level = max(level, "warning", key=_severity_rank)
            alerts.append({
                "component": name,
                "severity": "warning",
                "message": f"{name} RSS elevated: {rss:.0f} MB",
            })

    growth_limit = thresholds["rss_growth_rate_mb_per_10min"]
    for name, rate in growth_rates.items():
        if rate > growth_limit * 2:
            level = max(level, "critical", key=_severity_rank)
            alerts.append({
                "component": name,
                "severity": "critical",
                "message": f"{name} memory leak: +{rate:.0f} MB/10min",
            })
            actions.append({
                "action": "restart_process",
                "target": name,
                "reason": f"memory leak ({rate:.0f} MB/10min)",
            })
        elif rate > growth_limit:
            level = max(level, "warning", key=_severity_rank)
            alerts.append({
                "component": name,
                "severity": "warning",
                "message": f"{name} RSS growing: +{rate:.0f} MB/10min",
            })

    return {"level": level, "alerts": alerts, "actions": actions}


_SEVERITY_ORDER = {"healthy": 0, "warning": 1, "critical": 2}


def _severity_rank(s: str) -> int:
    return _SEVERITY_ORDER.get(s, 0)


# ── Recovery actions ─────────────────────────────────────────────────────────


def _recovery_allowed(component: str, prev_state: dict[str, Any]) -> bool:
    last = prev_state.get("recovery_attempts", {}).get(component, 0)
    return (time.time() - last) > RECOVERY_COOLDOWN


def _record_recovery(component: str, prev_state: dict[str, Any]) -> None:
    prev_state.setdefault("recovery_attempts", {})[component] = time.time()


def _restart_managed_agent(agent_label: str) -> bool:
    """Restart a launchd-managed agent via kickstart."""
    uid = os.getuid()
    try:
        subprocess.run(
            ["launchctl", "kickstart", "-k", f"gui/{uid}/{agent_label}"],
            capture_output=True, timeout=15,
        )
        logger.info("Restarted managed agent: %s", agent_label)
        return True
    except Exception as e:
        logger.warning("Failed to restart agent %s: %s", agent_label, e)
        return False


def _restart_unmanaged_app(name: str, pattern: str) -> bool:
    """Kill and relaunch an unmanaged app."""
    try:
        subprocess.run(["pkill", "-f", pattern], timeout=10, capture_output=True)
        time.sleep(3)
        app_name = "Mail" if "Mail" in pattern else "screenpipe"
        subprocess.run(["open", "-a", app_name], timeout=10, capture_output=True)
        logger.info("Restarted unmanaged app: %s", name)
        return True
    except Exception as e:
        logger.warning("Failed to restart %s: %s", name, e)
        return False


def execute_recovery(
    actions: list[dict[str, str]],
    processes: dict[str, dict[str, Any]],
    prev_state: dict[str, Any],
) -> list[str]:
    """Execute recovery actions, respecting cooldowns and per-cycle limits."""
    recovered: list[str] = []
    restarts_this_cycle = 0

    restart_largest = any(a["action"] == "restart_largest" for a in actions)
    if restart_largest:
        managed = {
            n: info for n, info in processes.items()
            if MONITORED_PROCESSES.get(n, {}).get("agent")
        }
        if managed:
            largest = max(managed, key=lambda n: managed[n]["rss_mb"])
            actions = [{"action": "restart_process", "target": largest, "reason": "largest RSS under memory pressure"}] + [
                a for a in actions if a["action"] != "restart_largest"
            ]

    for action in actions:
        if restarts_this_cycle >= MAX_RECOVERIES_PER_CYCLE:
            break

        if action["action"] == "restart_process":
            target = action["target"]
            spec = MONITORED_PROCESSES.get(target)
            if not spec:
                continue
            if not _recovery_allowed(target, prev_state):
                logger.info("Recovery cooldown active for %s, skipping", target)
                continue

            agent_label = spec.get("agent")
            if agent_label:
                ok = _restart_managed_agent(agent_label)
            else:
                ok = _restart_unmanaged_app(target, spec["pattern"])

            if ok:
                _record_recovery(target, prev_state)
                recovered.append(f"{target} ({action.get('reason', '')})")
                restarts_this_cycle += 1

    return recovered


# ── Main loop ────────────────────────────────────────────────────────────────


def run_cycle(
    cfg: dict[str, Any],
    prev_state: dict[str, Any],
) -> dict[str, Any]:
    """Run one monitoring cycle. Returns the new state dict."""
    thresholds = _get_thresholds(cfg)
    window = _get_history_window(cfg)
    vault_path = cfg["obsidian_vault"]

    system = collect_system_metrics()
    disk = collect_disk_metrics(vault_path)
    processes = collect_process_metrics()
    growth_rates = update_rss_history(processes, window)

    evaluation = evaluate(system, disk, processes, growth_rates, thresholds)
    alerts = evaluation["alerts"]
    actions = evaluation["actions"]
    level = evaluation["level"]

    recovered: list[str] = []
    if actions:
        recovered = execute_recovery(actions, processes, prev_state)
        if recovered:
            notify(
                "MemoryOS Resource Monitor",
                f"Restarted {len(recovered)} process(es)",
                ", ".join(recovered),
            )

    prev_level = prev_state.get("level", "healthy")

    if level == "critical" and prev_level != "critical":
        critical_alerts = [a["message"] for a in alerts if a["severity"] == "critical"]
        notify(
            "MemoryOS Memory Alert",
            "System resources critical",
            "; ".join(critical_alerts[:3]),
        )
    elif level == "warning" and prev_level == "healthy":
        warning_alerts = [a["message"] for a in alerts if a["severity"] == "warning"]
        notify(
            "MemoryOS Memory Warning",
            "Resource usage elevated",
            "; ".join(warning_alerts[:3]),
        )
    elif level == "healthy" and prev_level in ("warning", "critical"):
        notify(
            "MemoryOS Resources OK",
            "System resources recovered",
            f"RAM available: {system['ram_available_gb']:.1f} GB",
        )

    new_state: dict[str, Any] = {
        "level": level,
        "system": system,
        "disk": disk,
        "processes": {
            name: info for name, info in processes.items()
        },
        "growth_rates": growth_rates,
        "alerts": alerts,
        "recovered": recovered,
        "last_check": datetime.now(timezone.utc).isoformat(),
        "recovery_attempts": prev_state.get("recovery_attempts", {}),
    }

    return new_state


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="MemoryOS Resource Monitor")
    parser.add_argument("--config", help="Path to config.yaml")
    parser.add_argument(
        "--once", action="store_true",
        help="Run a single check cycle then exit",
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Print current status and exit",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg)

    if args.status:
        state = load_state()
        if not state:
            print("No resource monitor state yet. Run without --status first.")
            return
        print(f"Level: {state.get('level', 'unknown')}")
        print(f"Last check: {state.get('last_check', 'never')}")
        sys_info = state.get("system", {})
        print(f"RAM: {sys_info.get('ram_available_gb', '?')} GB free "
              f"/ {sys_info.get('ram_total_gb', '?')} GB total "
              f"({sys_info.get('ram_percent', '?')}% used)")
        print(f"Swap: {sys_info.get('swap_used_gb', '?')} GB used")
        print(f"CPU: {sys_info.get('cpu_percent', '?')}%")
        disk_info = state.get("disk", {})
        print(f"Disk: {disk_info.get('disk_free_gb', '?')} GB free")
        print("Processes:")
        for name, info in state.get("processes", {}).items():
            rate = state.get("growth_rates", {}).get(name, 0)
            rate_str = f" (+{rate:.0f} MB/10min)" if rate > 0 else ""
            print(f"  {name}: {info.get('rss_mb', '?')} MB RSS{rate_str}")
        if state.get("alerts"):
            print("Active alerts:")
            for a in state["alerts"]:
                print(f"  [{a['severity']}] {a['message']}")
        return

    interval = _get_interval(cfg)

    if args.once:
        prev = load_state()
        new_state = run_cycle(cfg, prev)
        save_state(new_state)
        logger.info("Resource check complete: level=%s", new_state["level"])
        return

    logger.info(
        "Resource monitor starting as persistent daemon (interval=%ds)", interval,
    )
    while True:
        try:
            prev = load_state()
            new_state = run_cycle(cfg, prev)
            save_state(new_state)
            logger.info(
                "Resource check: level=%s, RAM=%.1fGB free, swap=%.1fGB, "
                "processes=%d tracked",
                new_state["level"],
                new_state["system"]["ram_available_gb"],
                new_state["system"]["swap_used_gb"],
                len(new_state["processes"]),
            )
        except Exception:
            logger.exception("Resource monitor cycle failed, will retry next cycle")
        time.sleep(interval)


if __name__ == "__main__":
    main()
