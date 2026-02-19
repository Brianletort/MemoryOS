#!/usr/bin/env python3
"""MemoryOS Control Panel -- FastAPI dashboard for monitoring all extractors.

Run with:
    python3 -m uvicorn src.dashboard.app:app --host 0.0.0.0 --port 8765
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, BackgroundTasks, Request
from fastapi.responses import HTMLResponse, JSONResponse

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.common.config import load_config
from src.common.state import load_state

logger = logging.getLogger("memoryos.dashboard")

app = FastAPI(title="MemoryOS Control Panel", version="2.0.0")

REPO_DIR = Path(__file__).resolve().parent.parent.parent

# Cache SSID lookups to avoid repeated subprocess calls on every refresh
_ssid_cache: dict[str, Any] = {"ssid": None, "ts": 0.0}
_SSID_CACHE_TTL = 15.0
VENV_PYTHON = REPO_DIR / ".venv" / "bin" / "python3"
CONFIG_PATH = REPO_DIR / "config" / "config.yaml"
LAUNCHD_DIR = REPO_DIR / "launchd"
AGENTS_DIR = Path.home() / "Library" / "LaunchAgents"


def _cfg() -> dict[str, Any]:
    return load_config(CONFIG_PATH)


# ══════════════════════════════════════════════════════════════════════════════
#  Helper functions
# ══════════════════════════════════════════════════════════════════════════════

def _count_files(directory: Path, pattern: str = "*.md") -> int:
    if not directory.exists():
        return 0
    return sum(1 for _ in directory.rglob(pattern))


def _dir_size_mb(directory: Path) -> float:
    if not directory.exists():
        return 0.0
    total = sum(f.stat().st_size for f in directory.rglob("*") if f.is_file())
    return round(total / (1024 * 1024), 1)


def _tail_log(log_path: Path, lines: int = 50) -> str:
    if not log_path.exists():
        return "(no log file yet)"
    try:
        with open(log_path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            read_size = min(size, lines * 200)
            f.seek(max(0, size - read_size))
            data = f.read().decode("utf-8", errors="replace")
            result_lines = data.split("\n")
            return "\n".join(result_lines[-lines:])
    except OSError:
        return "(error reading log)"


def _file_age_seconds(path: Path) -> float | None:
    if not path.exists():
        return None
    return time.time() - path.stat().st_mtime


def _extractor_health(last_run_age: float | None, max_age_seconds: int = 600) -> str:
    if last_run_age is None:
        return "unknown"
    if last_run_age < max_age_seconds:
        return "healthy"
    if last_run_age < max_age_seconds * 3:
        return "warning"
    return "error"


def _launchd_status() -> dict[str, dict[str, Any]]:
    agents: dict[str, dict[str, Any]] = {}
    try:
        result = subprocess.run(
            ["launchctl", "list"], capture_output=True, text=True, timeout=5,
        )
        for line in result.stdout.strip().split("\n"):
            parts = line.split("\t")
            if len(parts) >= 3 and "memoryos" in parts[2]:
                name = parts[2]
                pid = parts[0] if parts[0] != "-" else None
                exit_code = int(parts[1]) if parts[1] != "-" else None
                agents[name] = {
                    "pid": pid, "last_exit_code": exit_code,
                    "loaded": True, "healthy": exit_code == 0 or exit_code is None,
                }
    except Exception as exc:
        logger.warning("Failed to check launchd: %s", exc)

    for name in [
        "com.memoryos.screenpipe", "com.memoryos.outlook", "com.memoryos.onedrive",
        "com.memoryos.dashboard", "com.memoryos.wifi-monitor",
        "com.memoryos.mail-app", "com.memoryos.calendar-app",
    ]:
        if name not in agents:
            agents[name] = {"pid": None, "last_exit_code": None, "loaded": False, "healthy": False}
    return agents


def _recent_errors(log_dir: Path, log_file: str) -> int:
    p = log_dir / log_file
    if not p.exists():
        return 0
    try:
        text = p.read_text(errors="replace")
        return text.lower().count("error") + text.lower().count("traceback")
    except OSError:
        return 0


def _privacy_flag_path() -> Path:
    cfg = _cfg()
    return Path(cfg.get("privacy", {}).get(
        "flag_file", str(REPO_DIR / "config" / ".privacy_mode"),
    ))


def _detect_ssid() -> str | None:
    """Detect WiFi SSID via scutil (lightweight, no permission dialogs)."""
    try:
        result = subprocess.run(
            ["scutil"],
            input="show State:/Network/Interface/en0/AirPort\n",
            capture_output=True, text=True, timeout=5,
        )
        for line in result.stdout.splitlines():
            stripped = line.strip()
            if stripped.startswith("SSID_STR"):
                ssid = stripped.split(":", 1)[1].strip() if ":" in stripped else ""
                return ssid or None
    except Exception:
        pass
    return None


def _cached_ssid() -> str | None:
    """Return SSID with a TTL cache to avoid repeated ~5s system_profiler calls."""
    now = time.monotonic()
    if now - _ssid_cache["ts"] < _SSID_CACHE_TTL:
        return _ssid_cache["ssid"]
    ssid = _detect_ssid()
    _ssid_cache["ssid"] = ssid
    _ssid_cache["ts"] = now
    return ssid


def _privacy_status() -> dict[str, Any]:
    flag = _privacy_flag_path()
    active = flag.exists()
    sp_paused: bool | None = None
    try:
        import urllib.request
        resp = urllib.request.urlopen("http://localhost:3030/health", timeout=2)
        data = json.loads(resp.read())
        sp_paused = data.get("audio_pipeline", {}).get("transcription_paused", None)
    except Exception:
        pass
    wifi_ssid = _cached_ssid()
    return {"privacy_mode": active, "screenpipe_audio_paused": sp_paused, "wifi_ssid": wifi_ssid}


def _screenpipe_audio_api(*, start: bool) -> None:
    def _call() -> None:
        import urllib.request
        import urllib.error
        endpoint = f"http://localhost:3030/audio/{'start' if start else 'stop'}"
        try:
            req = urllib.request.Request(endpoint, method="POST")
            urllib.request.urlopen(req, timeout=3)
        except (urllib.error.URLError, OSError):
            pass
    threading.Thread(target=_call, daemon=True).start()


def _files_created_today(directory: Path) -> int:
    if not directory.exists():
        return 0
    today_str = datetime.now().strftime("%Y-%m-%d")
    today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    count = 0
    try:
        for f in directory.rglob("*.md"):
            if f.stat().st_mtime >= today_start:
                count += 1
    except OSError:
        pass
    return count


def _newest_file(directory: Path) -> dict[str, Any] | None:
    if not directory.exists():
        return None
    newest: Path | None = None
    newest_mtime = 0.0
    try:
        for f in directory.rglob("*.md"):
            mt = f.stat().st_mtime
            if mt > newest_mtime:
                newest_mtime = mt
                newest = f
    except OSError:
        pass
    if newest is None:
        return None
    return {
        "name": newest.name,
        "path": str(newest),
        "modified": datetime.fromtimestamp(newest_mtime, tz=timezone.utc).isoformat(),
        "age_seconds": round(time.time() - newest_mtime, 1),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  API: Status (existing, enhanced)
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/status")
async def api_status() -> JSONResponse:
    cfg = _cfg()
    state = load_state(cfg["state_file"])
    vault = Path(cfg["obsidian_vault"])
    log_dir = Path(cfg["log_dir"])
    state_age = _file_age_seconds(Path(cfg["state_file"]))

    sp_state = state.get("screenpipe", {})
    ol_state = state.get("outlook", {})
    od_state = state.get("onedrive", {})
    od_tracked = len(od_state.get("file_mtimes", {}))

    activity_dir = vault / cfg["output"]["activity"]
    teams_dir = vault / cfg["output"]["teams"]
    meetings_dir = vault / cfg["output"]["meetings"]
    email_dir = vault / cfg["output"]["email"]
    slides_dir = vault / cfg["output"]["slides"]
    knowledge_dir = vault / cfg["output"]["knowledge"]

    data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "state_file_age_seconds": round(state_age, 1) if state_age else None,
        "extractors": {
            "screenpipe": {
                "last_frame_id": sp_state.get("last_frame_id", 0),
                "last_audio_id": sp_state.get("last_audio_id", 0),
                "activity_notes": _count_files(activity_dir),
                "teams_notes": _count_files(teams_dir),
                "meeting_audio": _count_files(meetings_dir),
                "health": _extractor_health(state_age, 600),
                "errors": _recent_errors(log_dir, "screenpipe_launchd.err"),
            },
            "outlook": {
                "last_mail_id": ol_state.get("last_mail_id", 0),
                "last_event_id": ol_state.get("last_event_id", 0),
                "email_files": _count_files(email_dir),
                "calendar_files": _count_files(meetings_dir, "calendar.md"),
                "health": _extractor_health(state_age, 600),
                "errors": _recent_errors(log_dir, "outlook_launchd.err"),
            },
            "onedrive": {
                "tracked_files": od_tracked,
                "slides_files": _count_files(slides_dir),
                "knowledge_files": _count_files(knowledge_dir),
                "health": _extractor_health(state_age, 1800),
                "errors": _recent_errors(log_dir, "onedrive_launchd.err"),
            },
        },
        "vault": {
            "total_markdown_files": _count_files(vault, "*.md"),
            "size_mb": _dir_size_mb(vault),
        },
        "folders": {
            "00_inbox": _count_files(email_dir),
            "10_meetings": _count_files(meetings_dir),
            "20_teams-chat": _count_files(teams_dir),
            "40_slides": _count_files(slides_dir),
            "50_knowledge": _count_files(knowledge_dir),
            "85_activity": _count_files(activity_dir),
        },
        "launchd": _launchd_status(),
        "privacy": _privacy_status(),
    }
    return JSONResponse(data)


# ══════════════════════════════════════════════════════════════════════════════
#  API: Pipeline Health
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/pipeline-health")
async def api_pipeline_health() -> JSONResponse:
    """Per-folder pipeline health: files today, newest file, total count."""
    cfg = _cfg()
    vault = Path(cfg["obsidian_vault"])

    folders_cfg = {
        "00_inbox": cfg["output"]["email"],
        "10_meetings": cfg["output"]["meetings"],
        "20_teams-chat": cfg["output"]["teams"],
        "40_slides": cfg["output"]["slides"],
        "50_knowledge": cfg["output"]["knowledge"],
        "85_activity": cfg["output"]["activity"],
    }

    result: dict[str, Any] = {}
    for label, subdir in folders_cfg.items():
        d = vault / subdir
        total = _count_files(d)
        today = _files_created_today(d)
        newest = _newest_file(d)
        result[label] = {
            "total_files": total,
            "files_today": today,
            "newest_file": newest,
            "healthy": newest is not None and newest["age_seconds"] < 86400,
        }

    return JSONResponse({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "folders": result,
    })


# ══════════════════════════════════════════════════════════════════════════════
#  API: Timeline
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/timeline/{date}")
async def api_timeline(date: str) -> JSONResponse:
    """Hour-by-hour activity for a given date (YYYY-MM-DD)."""
    cfg = _cfg()
    vault = Path(cfg["obsidian_vault"])

    try:
        dt = datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        return JSONResponse({"error": "Invalid date format, use YYYY-MM-DD"}, status_code=400)

    hours: dict[str, dict[str, int]] = {}
    for h in range(24):
        hours[f"{h:02d}"] = {"email": 0, "activity": 0, "meetings": 0, "teams": 0, "slides": 0}

    # Check email files for this date (path: 00_inbox/YYYY/MM/DD/)
    email_day = vault / cfg["output"]["email"] / dt.strftime("%Y/%m/%d")
    if email_day.is_dir():
        for f in email_day.rglob("*.md"):
            try:
                mtime = datetime.fromtimestamp(f.stat().st_mtime)
                hours[f"{mtime.hour:02d}"]["email"] += 1
            except OSError:
                pass

    # Check activity files for this date
    activity_day = vault / cfg["output"]["activity"] / dt.strftime("%Y/%m/%d")
    if activity_day.is_dir():
        for f in activity_day.rglob("*.md"):
            hours["00"]["activity"] += 1

    # Check meetings
    meetings_day = vault / cfg["output"]["meetings"] / dt.strftime("%Y/%m/%d")
    if meetings_day.is_dir():
        for f in meetings_day.rglob("*.md"):
            hours["00"]["meetings"] += 1

    return JSONResponse({
        "date": date,
        "hours": hours,
        "has_data": any(
            sum(v.values()) > 0 for v in hours.values()
        ),
    })


# ══════════════════════════════════════════════════════════════════════════════
#  API: Agent Control
# ══════════════════════════════════════════════════════════════════════════════

AGENT_NAMES = {
    "screenpipe": "com.memoryos.screenpipe",
    "outlook": "com.memoryos.outlook",
    "onedrive": "com.memoryos.onedrive",
    "wifi-monitor": "com.memoryos.wifi-monitor",
    "dashboard": "com.memoryos.dashboard",
}


@app.post("/api/agent/{name}/{action}")
async def api_agent_control(name: str, action: str) -> JSONResponse:
    """Start, stop, or restart a launchd agent."""
    if name not in AGENT_NAMES:
        return JSONResponse({"error": f"Unknown agent: {name}"}, status_code=404)
    if action not in ("start", "stop", "restart"):
        return JSONResponse({"error": f"Unknown action: {action}"}, status_code=400)

    label = AGENT_NAMES[name]
    plist_src = LAUNCHD_DIR / f"{label}.plist"
    plist_dst = AGENTS_DIR / f"{label}.plist"

    try:
        if action in ("stop", "restart"):
            subprocess.run(
                ["launchctl", "unload", str(plist_dst)],
                capture_output=True, timeout=10,
            )

        if action in ("start", "restart"):
            if plist_src.exists():
                import shutil
                shutil.copy2(str(plist_src), str(plist_dst))
            subprocess.run(
                ["launchctl", "load", str(plist_dst)],
                capture_output=True, timeout=10,
            )

        return JSONResponse({"status": "ok", "agent": name, "action": action})
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


# ══════════════════════════════════════════════════════════════════════════════
#  API: Settings
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/settings")
async def api_settings_get() -> JSONResponse:
    """Return current config as JSON (excluding sensitive paths)."""
    cfg = _cfg()
    privacy = cfg.get("privacy", {})
    return JSONResponse({
        "trusted_networks": privacy.get("trusted_networks", []),
        "work_apps": privacy.get("work_apps", []),
        "audio_filter": privacy.get("audio_filter", {}),
        "screenpipe_settings": cfg.get("screenpipe_settings", {}),
        "outlook_settings": cfg.get("outlook_settings", {}),
        "onedrive_settings": cfg.get("onedrive_settings", {}),
        "log_level": cfg.get("log_level", "INFO"),
    })


@app.post("/api/settings")
async def api_settings_post(request: Request) -> JSONResponse:
    """Update specific settings in config.yaml.

    Accepts JSON with keys: trusted_networks, work_apps, audio_filter.
    Uses round-trip load/dump to preserve existing structure as much as possible.
    """
    body = await request.json()

    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        if raw is None:
            raw = {}

        changed = False

        if "trusted_networks" in body:
            raw.setdefault("privacy", {})["trusted_networks"] = body["trusted_networks"]
            changed = True
        if "work_apps" in body:
            raw.setdefault("privacy", {})["work_apps"] = body["work_apps"]
            changed = True
        if "audio_filter" in body:
            raw.setdefault("privacy", {})["audio_filter"] = body["audio_filter"]
            changed = True

        if changed:
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                yaml.dump(
                    raw, f,
                    default_flow_style=False,
                    sort_keys=False,
                    allow_unicode=True,
                    width=200,
                )

        return JSONResponse({"status": "ok", "changed": changed})
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


# ══════════════════════════════════════════════════════════════════════════════
#  API: Sync Status
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/sync-status")
async def api_sync_status() -> JSONResponse:
    """Check Obsidian sync state."""
    cfg = _cfg()
    vault = Path(cfg["obsidian_vault"])
    obsidian_dir = vault / ".obsidian"

    result: dict[str, Any] = {
        "obsidian_running": False,
        "sync_plugin_enabled": False,
        "vault_path": str(vault),
        "vault_exists": vault.is_dir(),
    }

    # Is Obsidian running?
    try:
        r = subprocess.run(["pgrep", "-f", "Obsidian"], capture_output=True, timeout=3)
        result["obsidian_running"] = r.returncode == 0
    except Exception:
        pass

    # Sync plugin enabled?
    core_plugins = obsidian_dir / "core-plugins.json"
    if core_plugins.exists():
        try:
            plugins = json.loads(core_plugins.read_text())
            result["sync_plugin_enabled"] = "sync" in plugins
        except Exception:
            pass

    # Sync config
    sync_json = obsidian_dir / "sync.json"
    if sync_json.exists():
        try:
            sync_cfg = json.loads(sync_json.read_text())
            result["sync_vault_id"] = sync_cfg.get("vaultId", "")
            result["sync_configured"] = True
        except Exception:
            result["sync_configured"] = False
    else:
        result["sync_configured"] = False

    # Vault freshness
    newest = _newest_file(vault)
    if newest:
        result["newest_file_age_seconds"] = newest["age_seconds"]
        result["newest_file"] = newest["name"]

    return JSONResponse(result)


# ══════════════════════════════════════════════════════════════════════════════
#  API: Logs, Run-now, Privacy, File Browser (existing)
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/logs/{extractor}")
async def api_logs(extractor: str) -> JSONResponse:
    cfg = _cfg()
    log_dir = Path(cfg["log_dir"])
    log_map = {
        "screenpipe": "screenpipe_launchd", "outlook": "outlook_launchd",
        "onedrive": "onedrive_launchd", "wifi_monitor": "wifi_monitor",
        "mail_app": "mail_app_launchd", "calendar_app": "calendar_app_launchd",
        "main": "memoryos",
    }
    base = log_map.get(extractor)
    if not base:
        return JSONResponse({"error": f"Unknown: {extractor}"}, status_code=404)
    stdout = _tail_log(log_dir / f"{base}.log", 80)
    stderr = _tail_log(log_dir / f"{base}.err", 40)
    return JSONResponse({"extractor": extractor, "stdout": stdout, "stderr": stderr})


def _run_extractor(name: str) -> None:
    script_map = {
        "screenpipe": "src/extractors/screenpipe_extractor.py",
        "outlook": "src/extractors/outlook_extractor.py",
        "onedrive": "src/extractors/onedrive_extractor.py",
        "mail_app": "src/extractors/mail_app_extractor.py",
        "calendar_app": "src/extractors/calendar_app_extractor.py",
    }
    script = script_map.get(name)
    if not script:
        return
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_DIR)
    try:
        subprocess.run(
            [str(VENV_PYTHON), str(REPO_DIR / script)],
            env=env, capture_output=True, timeout=300,
        )
    except Exception as exc:
        logger.error("Run-now failed for %s: %s", name, exc)


@app.post("/api/run/{extractor}")
async def api_run(extractor: str, background_tasks: BackgroundTasks) -> JSONResponse:
    if extractor not in ("screenpipe", "outlook", "onedrive", "mail_app", "calendar_app"):
        return JSONResponse({"error": f"Unknown: {extractor}"}, status_code=404)
    background_tasks.add_task(_run_extractor, extractor)
    return JSONResponse({"status": "started", "extractor": extractor})


@app.get("/api/privacy")
async def api_privacy_status() -> JSONResponse:
    return JSONResponse(_privacy_status())


@app.post("/api/privacy/toggle")
async def api_privacy_toggle() -> JSONResponse:
    flag = _privacy_flag_path()
    if flag.exists():
        flag.unlink(missing_ok=True)
        _screenpipe_audio_api(start=True)
        return JSONResponse({"privacy_mode": False, "message": "Privacy mode OFF"})
    else:
        flag.parent.mkdir(parents=True, exist_ok=True)
        flag.touch()
        _screenpipe_audio_api(start=False)
        return JSONResponse({"privacy_mode": True, "message": "Privacy mode ON"})


def _safe_vault_path(cfg: dict[str, Any], rel_path: str) -> Path | None:
    vault = Path(cfg["obsidian_vault"]).resolve()
    target = (vault / rel_path).resolve()
    if not str(target).startswith(str(vault)):
        return None
    return target


@app.get("/api/browse")
async def api_browse(path: str = "") -> JSONResponse:
    cfg = _cfg()
    target = _safe_vault_path(cfg, path)
    if target is None or not target.is_dir():
        return JSONResponse({"error": "Invalid path"}, status_code=400)
    vault = Path(cfg["obsidian_vault"]).resolve()
    folders: list[dict[str, Any]] = []
    files: list[dict[str, Any]] = []
    try:
        for entry in sorted(target.iterdir(), key=lambda e: e.name):
            if entry.name.startswith("."):
                continue
            rel = str(entry.relative_to(vault))
            if entry.is_dir():
                md_count = sum(1 for _ in entry.rglob("*.md"))
                folders.append({"name": entry.name, "path": rel, "file_count": md_count})
            elif entry.suffix == ".md":
                stat = entry.stat()
                files.append({
                    "name": entry.name, "path": rel, "size": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                })
    except PermissionError:
        return JSONResponse({"error": "Permission denied"}, status_code=403)
    crumbs = [{"name": "Vault", "path": ""}]
    if path:
        parts = Path(path).parts
        for i, part in enumerate(parts):
            crumbs.append({"name": part, "path": str(Path(*parts[:i + 1]))})
    return JSONResponse({"current_path": path, "breadcrumb": crumbs, "folders": folders, "files": files})


@app.get("/api/file")
async def api_file(path: str = "") -> JSONResponse:
    cfg = _cfg()
    target = _safe_vault_path(cfg, path)
    if target is None or not target.is_file():
        return JSONResponse({"error": "File not found"}, status_code=404)
    if target.suffix != ".md":
        return JSONResponse({"error": "Only .md files"}, status_code=400)
    max_bytes = 200_000
    try:
        raw = target.read_bytes()[:max_bytes]
        content = raw.decode("utf-8", errors="replace")
        truncated = len(raw) >= max_bytes
    except OSError as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)
    stat = target.stat()
    return JSONResponse({
        "path": path, "name": target.name, "content": content,
        "truncated": truncated, "size": stat.st_size,
        "modified": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
    })


# ══════════════════════════════════════════════════════════════════════════════
#  HTML Dashboard — Full Control Panel
# ══════════════════════════════════════════════════════════════════════════════

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MemoryOS Control Panel</title>
<style>
:root {
  --bg: #0f1117; --surface: #1a1d27; --surface2: #22252f; --border: #2a2d3a;
  --text: #e0e0e6; --muted: #8b8fa3; --accent: #6c7cff;
  --green: #4ade80; --yellow: #fbbf24; --red: #f87171; --blue: #60a5fa;
  --orange: #fb923c;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'SF Pro', 'Inter', system-ui, sans-serif;
  background: var(--bg); color: var(--text); line-height: 1.5;
}

/* ── Top nav ── */
.top-tabs {
  display: flex; gap: 0; border-bottom: 2px solid var(--border);
  background: var(--surface); padding: 0 24px; position: sticky; top: 0; z-index: 100;
}
.top-tab {
  padding: 12px 20px; font-size: .9rem; font-weight: 500;
  border: none; background: transparent; color: var(--muted); cursor: pointer;
  border-bottom: 2px solid transparent; margin-bottom: -2px; transition: all .15s;
}
.top-tab:hover { color: var(--text); }
.top-tab.active { color: var(--accent); border-bottom-color: var(--accent); }
.tab-pane { display: none; padding: 24px; max-width: 1400px; margin: 0 auto; }
.tab-pane.active { display: block; }

/* ── Shared ── */
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 20px; margin-bottom: 16px; }
.card-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px; }
.card-title { font-size: 1.05rem; font-weight: 600; }
.badge { display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: .72rem; font-weight: 600; text-transform: uppercase; letter-spacing: .5px; }
.badge-healthy { background: rgba(74,222,128,.15); color: var(--green); }
.badge-warning { background: rgba(251,191,36,.15); color: var(--yellow); }
.badge-error   { background: rgba(248,113,113,.15); color: var(--red); }
.badge-unknown { background: rgba(139,143,163,.15); color: var(--muted); }
.badge-off     { background: rgba(139,143,163,.1);  color: var(--muted); }
.stat-row { display: flex; justify-content: space-between; padding: 5px 0; border-bottom: 1px solid var(--border); font-size: .87rem; }
.stat-row:last-child { border-bottom: none; }
.stat-label { color: var(--muted); }
.stat-value { font-weight: 500; font-variant-numeric: tabular-nums; }
.btn { display: inline-block; padding: 6px 16px; border-radius: 8px; border: 1px solid var(--border); background: var(--surface); color: var(--accent); font-size: .83rem; font-weight: 500; cursor: pointer; transition: all .15s; }
.btn:hover { background: var(--accent); color: #fff; border-color: var(--accent); }
.btn:disabled { opacity: .4; cursor: not-allowed; }
.btn-sm { padding: 4px 12px; font-size: .78rem; }
.btn-red { color: var(--red); }
.btn-red:hover { background: var(--red); color: #fff; border-color: var(--red); }
.btn-green { color: var(--green); }
.btn-green:hover { background: var(--green); color: #000; border-color: var(--green); }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px; margin-bottom: 20px; }
.dot { width:8px;height:8px;border-radius:50%;display:inline-block;margin-right:6px; }
.dot-green { background: var(--green); }
.dot-red { background: var(--red); }
.dot-gray { background: var(--muted); }

/* ── Overview hero bar ── */
.hero-bar { display: flex; gap: 20px; flex-wrap: wrap; margin-bottom: 20px; }
.hero-card {
  flex: 1; min-width: 160px; background: var(--surface); border: 1px solid var(--border);
  border-radius: 12px; padding: 16px; text-align: center;
}
.hero-num { font-size: 2rem; font-weight: 700; color: var(--accent); line-height: 1.1; }
.hero-label { font-size: .78rem; color: var(--muted); margin-top: 4px; }

/* ── Privacy bar ── */
.privacy-bar {
  display: flex; align-items: center; gap: 16px; padding: 14px 20px;
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 12px; margin-bottom: 20px;
}
.privacy-bar .label { font-weight: 600; font-size: .92rem; }
.privacy-bar .wifi-info { color: var(--muted); font-size: .8rem; }
.privacy-toggle-btn {
  padding: 8px 20px; border-radius: 8px; font-weight: 600; font-size: .83rem;
  cursor: pointer; transition: all .2s; border: 2px solid transparent;
}
.privacy-toggle-btn.active { background: rgba(248,113,113,.15); color: var(--red); border-color: var(--red); }
.privacy-toggle-btn.inactive { background: rgba(74,222,128,.15); color: var(--green); border-color: var(--green); }
.privacy-toggle-btn:hover { opacity: .85; }
.priv-dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }
.priv-dot.on { background: var(--red); box-shadow: 0 0 8px rgba(248,113,113,.5); }
.priv-dot.off { background: var(--green); box-shadow: 0 0 8px rgba(74,222,128,.5); }

/* ── Agents table ── */
.agent-row {
  display: flex; justify-content: space-between; align-items: center;
  padding: 10px 0; border-bottom: 1px solid var(--border); font-size: .87rem;
}
.agent-row:last-child { border-bottom: none; }
.agent-actions { display: flex; gap: 6px; }

/* ── Pipeline folder row ── */
.folder-row {
  display: grid; grid-template-columns: 140px 100px 100px 1fr 140px;
  align-items: center; padding: 10px 0; border-bottom: 1px solid var(--border);
  font-size: .85rem; gap: 12px;
}
.folder-row:last-child { border-bottom: none; }
@media (max-width: 900px) {
  .folder-row { grid-template-columns: 1fr 1fr; gap: 6px; }
}

/* ── Timeline bars ── */
.timeline { display: flex; gap: 2px; align-items: end; height: 60px; margin: 12px 0; }
.timeline-bar {
  flex: 1; min-width: 0; border-radius: 3px 3px 0 0; transition: height .3s;
  position: relative; cursor: pointer;
}
.timeline-bar:hover::after {
  content: attr(data-tip); position: absolute; bottom: 100%; left: 50%;
  transform: translateX(-50%); padding: 4px 8px; background: var(--surface2);
  border: 1px solid var(--border); border-radius: 6px; font-size: .7rem;
  white-space: nowrap; z-index: 10; color: var(--text);
}
.timeline-labels { display: flex; justify-content: space-between; font-size: .65rem; color: var(--muted); }

/* ── Settings ── */
.settings-section { margin-bottom: 24px; }
.settings-section h3 { font-size: .95rem; margin-bottom: 12px; color: var(--accent); }
.tag-list { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 12px; }
.tag {
  display: inline-flex; align-items: center; gap: 6px; padding: 4px 12px;
  background: var(--surface2); border: 1px solid var(--border); border-radius: 6px;
  font-size: .82rem;
}
.tag .remove { cursor: pointer; color: var(--red); font-weight: 700; font-size: .9rem; }
.tag .remove:hover { color: #fff; }
.add-input { display: flex; gap: 8px; }
.add-input input {
  flex: 1; padding: 6px 12px; border-radius: 8px; border: 1px solid var(--border);
  background: var(--bg); color: var(--text); font-size: .85rem; outline: none;
}
.add-input input:focus { border-color: var(--accent); }

/* ── Sync status ── */
.sync-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
@media (max-width: 700px) { .sync-grid { grid-template-columns: 1fr; } }

/* ── Log viewer ── */
.log-tabs { display: flex; gap: 8px; flex-wrap: wrap; }
.log-tab { padding: 4px 14px; border-radius: 6px; border: 1px solid var(--border); background: transparent; color: var(--muted); font-size: .78rem; cursor: pointer; }
.log-tab.active { background: var(--accent); color: #fff; border-color: var(--accent); }
.log-box {
  background: #0a0c10; border: 1px solid var(--border); border-radius: 8px;
  padding: 16px; font-family: 'SF Mono','Menlo',monospace;
  font-size: .76rem; line-height: 1.6; max-height: 600px;
  overflow-y: auto; white-space: pre-wrap; word-break: break-all; color: var(--muted);
}
.log-box .log-error { color: var(--red); font-weight: 600; }
.log-box .log-warn  { color: var(--yellow); }
.log-box .log-info  { color: var(--green); }

/* ── File Browser ── */
.browser-layout { display: grid; grid-template-columns: 340px 1fr; gap: 16px; min-height: 600px; }
@media (max-width: 900px) { .browser-layout { grid-template-columns: 1fr; } }
.browser-sidebar { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; overflow: hidden; display: flex; flex-direction: column; }
.browser-breadcrumb { padding: 12px 16px; border-bottom: 1px solid var(--border); font-size: .8rem; color: var(--muted); display: flex; flex-wrap: wrap; gap: 4px; }
.browser-breadcrumb a { color: var(--accent); text-decoration: none; cursor: pointer; }
.browser-breadcrumb a:hover { text-decoration: underline; }
.browser-breadcrumb .sep { margin: 0 2px; color: var(--border); }
.browser-list { flex: 1; overflow-y: auto; padding: 8px 0; }
.browser-item { display: flex; align-items: center; padding: 7px 16px; cursor: pointer; font-size: .86rem; transition: background .1s; gap: 8px; }
.browser-item:hover { background: rgba(108,124,255,.08); }
.browser-item.selected { background: rgba(108,124,255,.15); }
.browser-item .icon { font-size: 1rem; flex-shrink: 0; width: 20px; text-align: center; }
.browser-item .name { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.browser-item .meta { color: var(--muted); font-size: .73rem; flex-shrink: 0; font-variant-numeric: tabular-nums; }
.browser-preview { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; overflow: hidden; display: flex; flex-direction: column; }
.preview-header { padding: 12px 16px; border-bottom: 1px solid var(--border); display: flex; justify-content: space-between; align-items: center; }
.preview-header .filename { font-weight: 600; font-size: .9rem; }
.preview-header .file-meta { color: var(--muted); font-size: .76rem; }
.preview-body { flex: 1; overflow-y: auto; padding: 20px; font-size: .86rem; line-height: 1.7; }
.preview-body .frontmatter { background: rgba(108,124,255,.06); border: 1px solid var(--border); border-radius: 8px; padding: 12px 16px; margin-bottom: 16px; font-family: 'SF Mono','Menlo',monospace; font-size: .76rem; color: var(--muted); white-space: pre-wrap; }
.preview-body h1 { font-size: 1.3rem; margin: 16px 0 8px; color: var(--accent); }
.preview-body h2 { font-size: 1.1rem; margin: 14px 0 6px; color: var(--blue); }
.preview-body h3 { font-size: .95rem; margin: 10px 0 4px; }
.preview-body blockquote { border-left: 3px solid var(--accent); padding-left: 12px; margin: 8px 0; color: var(--muted); }
.preview-body hr { border: none; border-top: 1px solid var(--border); margin: 12px 0; }
.preview-body strong { color: var(--text); }
.preview-body a { color: var(--accent); }
.preview-empty { flex: 1; display: flex; align-items: center; justify-content: center; color: var(--muted); font-size: .9rem; }

/* ── Alerts ── */
.alert { padding: 12px 16px; border-radius: 8px; margin-bottom: 12px; font-size: .85rem; display: flex; align-items: center; gap: 10px; }
.alert-warn { background: rgba(251,191,36,.1); border: 1px solid rgba(251,191,36,.3); color: var(--yellow); }
.alert-error { background: rgba(248,113,113,.1); border: 1px solid rgba(248,113,113,.3); color: var(--red); }
.alert-info { background: rgba(96,165,250,.1); border: 1px solid rgba(96,165,250,.3); color: var(--blue); }
</style>
</head>
<body>

<!-- ══════ TOP TABS ══════ -->
<div class="top-tabs">
  <button class="top-tab active" onclick="switchTab('overview',this)">Overview</button>
  <button class="top-tab" onclick="switchTab('pipeline',this)">Pipeline</button>
  <button class="top-tab" onclick="switchTab('settings',this)">Settings</button>
  <button class="top-tab" onclick="switchTab('browser',this)">File Browser</button>
  <button class="top-tab" onclick="switchTab('logs',this)">Logs</button>
</div>

<!-- ══════ TAB 1: OVERVIEW ══════ -->
<div class="tab-pane active" id="pane-overview">
<h1 style="margin-bottom:4px">MemoryOS Control Panel</h1>
<p style="color:var(--muted);font-size:.82rem;margin-bottom:20px">Auto-refreshes every 15s</p>

<div id="alerts"></div>

<!-- Privacy -->
<div class="privacy-bar" id="privacy-bar">
  <span class="priv-dot off" id="priv-dot"></span>
  <span class="label" id="priv-label">Audio: Recording</span>
  <span class="wifi-info" id="priv-wifi">WiFi: checking...</span>
  <div style="flex:1"></div>
  <button class="privacy-toggle-btn inactive" id="priv-btn" onclick="togglePrivacy()">Pause Audio</button>
</div>

<!-- Hero stats -->
<div class="hero-bar" id="hero-bar">
  <div class="hero-card"><div class="hero-num" id="h-total">--</div><div class="hero-label">Total Markdown</div></div>
  <div class="hero-card"><div class="hero-num" id="h-vault">--</div><div class="hero-label">Vault Size (MB)</div></div>
  <div class="hero-card"><div class="hero-num" id="h-today">--</div><div class="hero-label">Files Today</div></div>
  <div class="hero-card"><div class="hero-num" id="h-agents">--</div><div class="hero-label">Agents Running</div></div>
</div>

<!-- Extractor cards -->
<div class="grid" id="cards"></div>

<!-- Agents -->
<div class="card">
  <div class="card-header"><span class="card-title">LaunchD Agents</span>
    <button class="btn btn-sm" onclick="refreshAgents()">Refresh</button>
  </div>
  <div id="agent-list"></div>
</div>

<!-- Sync -->
<div class="card" id="sync-card">
  <div class="card-header"><span class="card-title">Obsidian Sync</span><span class="badge badge-unknown" id="sync-badge">checking</span></div>
  <div id="sync-info"></div>
</div>
</div><!-- /pane-overview -->

<!-- ══════ TAB 2: PIPELINE ══════ -->
<div class="tab-pane" id="pane-pipeline">
<h2 style="margin-bottom:16px">Pipeline Health</h2>

<div class="card">
  <div class="card-header"><span class="card-title">Folder Status</span>
    <button class="btn btn-sm" onclick="loadPipeline()">Refresh</button>
  </div>
  <div style="overflow-x:auto">
    <div class="folder-row" style="font-weight:600;color:var(--muted);font-size:.78rem;border-bottom:2px solid var(--border)">
      <span>Folder</span><span>Total</span><span>Today</span><span>Newest File</span><span>Status</span>
    </div>
    <div id="pipeline-rows"></div>
  </div>
</div>

<div class="card">
  <div class="card-header"><span class="card-title">Today's Activity Timeline</span></div>
  <div class="timeline" id="timeline"></div>
  <div class="timeline-labels"><span>00:00</span><span>06:00</span><span>12:00</span><span>18:00</span><span>23:00</span></div>
</div>
</div><!-- /pane-pipeline -->

<!-- ══════ TAB 3: SETTINGS ══════ -->
<div class="tab-pane" id="pane-settings">
<h2 style="margin-bottom:16px">Settings</h2>

<div class="card settings-section">
  <h3>Trusted WiFi Networks</h3>
  <p style="color:var(--muted);font-size:.8rem;margin-bottom:12px">Audio is captured on these networks. Use * suffix for prefix matching (e.g. "DLR*").</p>
  <div class="tag-list" id="net-tags"></div>
  <div class="add-input">
    <input id="net-input" placeholder="Add network SSID..." onkeydown="if(event.key==='Enter')addNetwork()">
    <button class="btn btn-sm" onclick="addNetwork()">Add</button>
  </div>
</div>

<div class="card settings-section">
  <h3>Work Apps</h3>
  <p style="color:var(--muted);font-size:.8rem;margin-bottom:12px">Audio is kept when these apps are on screen (even without privacy mode).</p>
  <div class="tag-list" id="app-tags"></div>
  <div class="add-input">
    <input id="app-input" placeholder="Add app name..." onkeydown="if(event.key==='Enter')addApp()">
    <button class="btn btn-sm" onclick="addApp()">Add</button>
  </div>
</div>

<div class="card settings-section">
  <h3>Audio Filter</h3>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;max-width:500px">
    <div>
      <label style="font-size:.8rem;color:var(--muted);display:block;margin-bottom:4px">Minimum Words</label>
      <input type="number" id="set-min-words" min="1" max="50" style="width:100%;padding:6px 10px;border-radius:8px;border:1px solid var(--border);background:var(--bg);color:var(--text);font-size:.85rem">
    </div>
    <div>
      <label style="font-size:.8rem;color:var(--muted);display:block;margin-bottom:4px">Work Hours Only</label>
      <select id="set-work-hours" style="width:100%;padding:6px 10px;border-radius:8px;border:1px solid var(--border);background:var(--bg);color:var(--text);font-size:.85rem">
        <option value="false">No (24h capture)</option>
        <option value="true">Yes (work hours only)</option>
      </select>
    </div>
    <div>
      <label style="font-size:.8rem;color:var(--muted);display:block;margin-bottom:4px">Work Start</label>
      <input type="time" id="set-work-start" style="width:100%;padding:6px 10px;border-radius:8px;border:1px solid var(--border);background:var(--bg);color:var(--text);font-size:.85rem">
    </div>
    <div>
      <label style="font-size:.8rem;color:var(--muted);display:block;margin-bottom:4px">Work End</label>
      <input type="time" id="set-work-end" style="width:100%;padding:6px 10px;border-radius:8px;border:1px solid var(--border);background:var(--bg);color:var(--text);font-size:.85rem">
    </div>
  </div>
  <button class="btn" style="margin-top:16px" onclick="saveFilterSettings()">Save Filter Settings</button>
  <span id="filter-save-msg" style="margin-left:12px;font-size:.82rem;color:var(--green)"></span>
</div>
</div><!-- /pane-settings -->

<!-- ══════ TAB 4: FILE BROWSER ══════ -->
<div class="tab-pane" id="pane-browser">
<div class="browser-layout">
  <div class="browser-sidebar">
    <div class="browser-breadcrumb" id="fb-bc"></div>
    <div class="browser-list" id="fb-list">Loading...</div>
  </div>
  <div class="browser-preview" id="fb-preview"><div class="preview-empty">Select a markdown file to preview</div></div>
</div>
</div><!-- /pane-browser -->

<!-- ══════ TAB 5: LOGS ══════ -->
<div class="tab-pane" id="pane-logs">
<div class="card">
  <div class="card-header">
    <span class="card-title">Extractor Logs</span>
    <div style="display:flex;gap:8px;align-items:center">
      <label style="font-size:.78rem;color:var(--muted)"><input type="checkbox" id="log-auto" checked style="margin-right:4px">Auto-refresh</label>
      <button class="btn btn-sm" onclick="refreshLogs()">Refresh</button>
    </div>
  </div>
  <div class="log-tabs" style="margin-bottom:12px">
    <button class="log-tab active" data-src="main" onclick="loadLog('main',this)">Main</button>
    <button class="log-tab" data-src="screenpipe" onclick="loadLog('screenpipe',this)">Screenpipe</button>
    <button class="log-tab" data-src="outlook" onclick="loadLog('outlook',this)">Outlook</button>
    <button class="log-tab" data-src="onedrive" onclick="loadLog('onedrive',this)">OneDrive</button>
    <button class="log-tab" data-src="mail_app" onclick="loadLog('mail_app',this)">Mail App</button>
    <button class="log-tab" data-src="calendar_app" onclick="loadLog('calendar_app',this)">Calendar App</button>
    <button class="log-tab" data-src="wifi_monitor" onclick="loadLog('wifi_monitor',this)">WiFi Monitor</button>
  </div>
  <div class="log-box" id="log-box">Loading...</div>
</div>
</div><!-- /pane-logs -->

<script>
/* ═══ Globals ═══ */
const R = 15000;
let logTab = 'main', browsePath = '';
let settingsData = {};

/* ═══ Utilities ═══ */
function escHtml(s) { return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
function badge(h) {
  const c = {healthy:'badge-healthy',warning:'badge-warning',error:'badge-error'}[h]||'badge-unknown';
  return `<span class="badge ${c}">${h||'unknown'}</span>`;
}
function fmtSize(b) { return b<1024?b+' B':b<1048576?(b/1024).toFixed(1)+' KB':(b/1048576).toFixed(1)+' MB'; }
function fmtDate(iso) { if(!iso)return''; const d=new Date(iso); return d.toLocaleDateString()+' '+d.toLocaleTimeString([],{hour:'2-digit',minute:'2-digit'}); }
function fmtAge(s) { if(s==null)return'never'; if(s<60)return Math.round(s)+'s ago'; if(s<3600)return Math.round(s/60)+'m ago'; if(s<86400)return Math.round(s/3600)+'h ago'; return Math.round(s/86400)+'d ago'; }
function todayStr() { const d=new Date(); return d.getFullYear()+'-'+String(d.getMonth()+1).padStart(2,'0')+'-'+String(d.getDate()).padStart(2,'0'); }

/* ═══ Tab switching ═══ */
function switchTab(name, btn) {
  document.querySelectorAll('.top-tab').forEach(t=>t.classList.remove('active'));
  document.querySelectorAll('.tab-pane').forEach(p=>p.classList.remove('active'));
  btn.classList.add('active');
  document.getElementById('pane-'+name).classList.add('active');
  if(name==='browser') browseTo(browsePath);
  if(name==='logs') loadLog(logTab);
  if(name==='pipeline') loadPipeline();
  if(name==='settings') loadSettings();
}

/* ═══ Overview: Status ═══ */
async function refresh() {
  try {
    const d = await (await fetch('/api/status')).json();
    renderHero(d); renderCards(d); renderAgents(d.launchd); renderAlerts(d);
    if(d.privacy) renderPrivacy(d.privacy);
  } catch(e) { console.error('Refresh failed',e); }
  if(document.getElementById('log-auto')?.checked && document.getElementById('pane-logs').classList.contains('active')) loadLog(logTab);
}

function renderHero(d) {
  document.getElementById('h-total').textContent = (d.vault?.total_markdown_files||0).toLocaleString();
  document.getElementById('h-vault').textContent = d.vault?.size_mb||'--';
  const loaded = Object.values(d.launchd||{}).filter(a=>a.loaded).length;
  document.getElementById('h-agents').textContent = loaded;
}

function renderCards(d) {
  const ex = d.extractors||{};
  let h = '';
  // Screenpipe
  const sp = ex.screenpipe||{};
  h += `<div class="card"><div class="card-header"><span class="card-title">Screenpipe</span>${badge(sp.health)}</div>
    <div class="stat-row"><span class="stat-label">Frames</span><span class="stat-value">${(sp.last_frame_id||0).toLocaleString()}</span></div>
    <div class="stat-row"><span class="stat-label">Audio</span><span class="stat-value">${(sp.last_audio_id||0).toLocaleString()}</span></div>
    <div class="stat-row"><span class="stat-label">Activity notes</span><span class="stat-value">${sp.activity_notes||0}</span></div>
    <div class="stat-row"><span class="stat-label">Teams</span><span class="stat-value">${sp.teams_notes||0}</span></div>
    <div class="stat-row"><span class="stat-label">Meeting audio</span><span class="stat-value">${sp.meeting_audio||0}</span></div>
    <div class="stat-row"><span class="stat-label">Errors</span><span class="stat-value">${sp.errors||0}</span></div>
    <div style="margin-top:10px"><button class="btn btn-sm" onclick="runNow('screenpipe',this)">Run Now</button></div></div>`;
  // Outlook
  const ol = ex.outlook||{};
  h += `<div class="card"><div class="card-header"><span class="card-title">Outlook (Classic)</span>${badge(ol.health)}</div>
    <div class="stat-row"><span class="stat-label">Emails (ID)</span><span class="stat-value">${(ol.last_mail_id||0).toLocaleString()}</span></div>
    <div class="stat-row"><span class="stat-label">Events (ID)</span><span class="stat-value">${(ol.last_event_id||0).toLocaleString()}</span></div>
    <div class="stat-row"><span class="stat-label">Email files</span><span class="stat-value">${(ol.email_files||0).toLocaleString()}</span></div>
    <div class="stat-row"><span class="stat-label">Calendar</span><span class="stat-value">${ol.calendar_files||0}</span></div>
    <div class="stat-row"><span class="stat-label">Errors</span><span class="stat-value">${ol.errors||0}</span></div>
    <div style="margin-top:10px"><button class="btn btn-sm" onclick="runNow('outlook',this)">Run Now</button></div></div>`;
  // OneDrive
  const od = ex.onedrive||{};
  h += `<div class="card"><div class="card-header"><span class="card-title">OneDrive</span>${badge(od.health)}</div>
    <div class="stat-row"><span class="stat-label">Tracked</span><span class="stat-value">${(od.tracked_files||0).toLocaleString()}</span></div>
    <div class="stat-row"><span class="stat-label">Slides</span><span class="stat-value">${(od.slides_files||0).toLocaleString()}</span></div>
    <div class="stat-row"><span class="stat-label">Knowledge</span><span class="stat-value">${(od.knowledge_files||0).toLocaleString()}</span></div>
    <div class="stat-row"><span class="stat-label">Errors</span><span class="stat-value">${od.errors||0}</span></div>
    <div style="margin-top:10px"><button class="btn btn-sm" onclick="runNow('onedrive',this)">Run Now</button></div></div>`;
  document.getElementById('cards').innerHTML = h;
}

function renderAgents(ld) {
  let h = '';
  for(const [name,info] of Object.entries(ld||{})) {
    const short = name.replace('com.memoryos.','');
    const dotCls = info.loaded?(info.healthy?'dot-green':'dot-red'):'dot-gray';
    const status = info.loaded?(info.healthy?'Running':`Exit ${info.last_exit_code}`):'Not loaded';
    h += `<div class="agent-row">
      <span><span class="dot ${dotCls}"></span><strong>${short}</strong></span>
      <span style="color:var(--muted);font-size:.82rem">${status}${info.pid?' (PID '+info.pid+')':''}</span>
      <div class="agent-actions">
        ${info.loaded
          ? `<button class="btn btn-sm btn-red" onclick="agentCtl('${short}','stop',this)">Stop</button>
             <button class="btn btn-sm" onclick="agentCtl('${short}','restart',this)">Restart</button>`
          : `<button class="btn btn-sm btn-green" onclick="agentCtl('${short}','start',this)">Start</button>`}
      </div>
    </div>`;
  }
  document.getElementById('agent-list').innerHTML = h;
}

function renderAlerts(d) {
  let alerts = '';
  const ex = d.extractors||{};
  for(const [k,v] of Object.entries(ex)) { if((v.errors||0)>5) alerts += `<div class="alert alert-error">${k} has ${v.errors} errors — check Logs tab</div>`; }
  if(d.privacy?.privacy_mode) alerts += `<div class="alert alert-info">Privacy mode is ON — audio transcriptions are being filtered</div>`;
  document.getElementById('alerts').innerHTML = alerts;
}

async function runNow(name, btn) {
  btn.disabled=true; btn.textContent='Running...';
  try { await fetch(`/api/run/${name}`,{method:'POST'}); btn.textContent='Started!';
    setTimeout(()=>{btn.disabled=false;btn.textContent='Run Now';refresh();},5000);
  } catch(e){ btn.textContent='Failed'; btn.disabled=false; }
}

async function agentCtl(name, action, btn) {
  btn.disabled=true; const orig=btn.textContent; btn.textContent='...';
  try {
    const r = await (await fetch(`/api/agent/${name}/${action}`,{method:'POST'})).json();
    if(r.error) { btn.textContent='Err'; } else { btn.textContent='OK'; }
    setTimeout(()=>{refresh();},1500);
  } catch(e){ btn.textContent='Err'; }
  setTimeout(()=>{btn.disabled=false;btn.textContent=orig;},2000);
}
function refreshAgents(){ refresh(); }

/* ═══ Privacy ═══ */
function renderPrivacy(p) {
  if(!p)return;
  const dot=document.getElementById('priv-dot'),lbl=document.getElementById('priv-label'),
        wifi=document.getElementById('priv-wifi'),btn=document.getElementById('priv-btn');
  if(p.privacy_mode){
    dot.className='priv-dot on'; lbl.textContent='Audio: PAUSED (Privacy Mode)'; lbl.style.color='var(--red)';
    btn.className='privacy-toggle-btn active'; btn.textContent='Resume Audio';
  } else {
    dot.className='priv-dot off'; lbl.textContent='Audio: Recording'; lbl.style.color='var(--green)';
    btn.className='privacy-toggle-btn inactive'; btn.textContent='Pause Audio';
  }
  let wt='WiFi: '+(p.wifi_ssid||'Not connected');
  if(p.screenpipe_audio_paused!=null) wt+=p.screenpipe_audio_paused?' | SP: paused':' | SP: active';
  wifi.textContent=wt;
}
async function togglePrivacy() {
  const btn=document.getElementById('priv-btn'); btn.disabled=true; btn.textContent='Toggling...';
  try { const d=await(await fetch('/api/privacy/toggle',{method:'POST'})).json(); renderPrivacy(d); setTimeout(refresh,1000); }
  catch(e){ btn.textContent='Error'; }
  btn.disabled=false;
}

/* ═══ Sync ═══ */
async function loadSync() {
  try {
    const d = await(await fetch('/api/sync-status')).json();
    const b = document.getElementById('sync-badge');
    if(d.obsidian_running && d.sync_plugin_enabled) { b.className='badge badge-healthy'; b.textContent='active'; }
    else if(d.obsidian_running) { b.className='badge badge-warning'; b.textContent='partial'; }
    else { b.className='badge badge-unknown'; b.textContent='offline'; }
    document.getElementById('sync-info').innerHTML = `
      <div class="stat-row"><span class="stat-label">Obsidian running</span><span class="stat-value">${d.obsidian_running?'Yes':'No'}</span></div>
      <div class="stat-row"><span class="stat-label">Sync plugin</span><span class="stat-value">${d.sync_plugin_enabled?'Enabled':'Disabled'}</span></div>
      <div class="stat-row"><span class="stat-label">Sync configured</span><span class="stat-value">${d.sync_configured?'Yes':'No'}</span></div>
      <div class="stat-row"><span class="stat-label">Newest file</span><span class="stat-value">${d.newest_file||'--'} (${fmtAge(d.newest_file_age_seconds)})</span></div>
      <div class="stat-row"><span class="stat-label">Vault path</span><span class="stat-value" style="font-size:.78rem">${d.vault_path||'--'}</span></div>`;
  } catch(e) { document.getElementById('sync-info').textContent='Failed to load'; }
}

/* ═══ Pipeline ═══ */
async function loadPipeline() {
  try {
    const d = await(await fetch('/api/pipeline-health')).json();
    let h = '', totalToday = 0;
    for(const [folder, info] of Object.entries(d.folders||{})) {
      const nf = info.newest_file;
      totalToday += info.files_today||0;
      const statusBadge = info.healthy ? '<span class="badge badge-healthy">active</span>'
        : info.total_files>0 ? '<span class="badge badge-warning">stale</span>'
        : '<span class="badge badge-off">empty</span>';
      h += `<div class="folder-row">
        <span style="font-weight:600">${folder}</span>
        <span class="stat-value">${(info.total_files||0).toLocaleString()}</span>
        <span class="stat-value" style="color:${info.files_today>0?'var(--green)':'var(--muted)'}">${info.files_today||0}</span>
        <span style="font-size:.78rem;color:var(--muted)">${nf?nf.name+' ('+fmtAge(nf.age_seconds)+')':'--'}</span>
        ${statusBadge}
      </div>`;
    }
    document.getElementById('pipeline-rows').innerHTML = h;
    document.getElementById('h-today').textContent = totalToday;
  } catch(e) { document.getElementById('pipeline-rows').innerHTML='<p style="color:var(--red)">Failed</p>'; }

  // Timeline
  try {
    const d = await(await fetch(`/api/timeline/${todayStr()}`)).json();
    let h = '';
    for(let i=0;i<24;i++) {
      const k = String(i).padStart(2,'0');
      const hr = d.hours?.[k]||{};
      const total = (hr.email||0)+(hr.activity||0)+(hr.meetings||0)+(hr.teams||0);
      const pct = total>0 ? Math.min(100, Math.max(8, total*2)) : 2;
      const color = total>0?'var(--accent)':'var(--border)';
      h += `<div class="timeline-bar" style="height:${pct}%;background:${color}" data-tip="${k}:00 — ${total} files"></div>`;
    }
    document.getElementById('timeline').innerHTML = h;
  } catch(e) {}
}

/* ═══ Settings ═══ */
async function loadSettings() {
  try {
    settingsData = await(await fetch('/api/settings')).json();
    renderNetworkTags(); renderAppTags();
    const af = settingsData.audio_filter||{};
    document.getElementById('set-min-words').value = af.min_words||5;
    document.getElementById('set-work-hours').value = String(af.work_hours_only||false);
    document.getElementById('set-work-start').value = af.work_hours_start||'07:00';
    document.getElementById('set-work-end').value = af.work_hours_end||'19:00';
  } catch(e) { console.error('Settings load failed',e); }
}
function renderNetworkTags() {
  const nets = settingsData.trusted_networks||[];
  document.getElementById('net-tags').innerHTML = nets.map((n,i)=>
    `<span class="tag">${escHtml(n)}<span class="remove" onclick="removeNetwork(${i})">×</span></span>`
  ).join('');
}
function renderAppTags() {
  const apps = settingsData.work_apps||[];
  document.getElementById('app-tags').innerHTML = apps.map((a,i)=>
    `<span class="tag">${escHtml(a)}<span class="remove" onclick="removeApp(${i})">×</span></span>`
  ).join('');
}
async function addNetwork() {
  const inp = document.getElementById('net-input');
  const v = inp.value.trim(); if(!v)return;
  const nets = settingsData.trusted_networks||[];
  if(!nets.includes(v)) { nets.push(v); settingsData.trusted_networks=nets; }
  await fetch('/api/settings',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({trusted_networks:nets})});
  inp.value=''; renderNetworkTags();
}
async function removeNetwork(i) {
  const nets = settingsData.trusted_networks||[];
  nets.splice(i,1); settingsData.trusted_networks=nets;
  await fetch('/api/settings',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({trusted_networks:nets})});
  renderNetworkTags();
}
async function addApp() {
  const inp = document.getElementById('app-input');
  const v = inp.value.trim(); if(!v)return;
  const apps = settingsData.work_apps||[];
  if(!apps.includes(v)) { apps.push(v); settingsData.work_apps=apps; }
  await fetch('/api/settings',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({work_apps:apps})});
  inp.value=''; renderAppTags();
}
async function removeApp(i) {
  const apps = settingsData.work_apps||[];
  apps.splice(i,1); settingsData.work_apps=apps;
  await fetch('/api/settings',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({work_apps:apps})});
  renderAppTags();
}
async function saveFilterSettings() {
  const af = {
    min_words: parseInt(document.getElementById('set-min-words').value)||5,
    work_hours_only: document.getElementById('set-work-hours').value==='true',
    work_hours_start: document.getElementById('set-work-start').value||'07:00',
    work_hours_end: document.getElementById('set-work-end').value||'19:00',
  };
  await fetch('/api/settings',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({audio_filter:af})});
  const msg = document.getElementById('filter-save-msg');
  msg.textContent='Saved!'; setTimeout(()=>msg.textContent='',3000);
}

/* ═══ Logs ═══ */
function highlightLog(text) {
  return text.replace(/^(.*)$/gm, line => {
    const ll=line.toLowerCase();
    if(ll.includes('error')||ll.includes('traceback')||ll.includes('exception')) return `<span class="log-error">${escHtml(line)}</span>`;
    if(ll.includes('warning')||ll.includes('warn')) return `<span class="log-warn">${escHtml(line)}</span>`;
    if(ll.includes('[info]')) return `<span class="log-info">${escHtml(line)}</span>`;
    return escHtml(line);
  });
}
async function loadLog(src, btn) {
  logTab=src;
  document.querySelectorAll('.log-tab').forEach(t=>t.classList.remove('active'));
  if(btn) btn.classList.add('active');
  else document.querySelector(`.log-tab[data-src="${src}"]`)?.classList.add('active');
  try {
    const d=await(await fetch(`/api/logs/${src}`)).json();
    const box=document.getElementById('log-box');
    let combined=d.stdout||'';
    if(d.stderr&&d.stderr!=='(no log file yet)') combined+='\n--- STDERR ---\n'+d.stderr;
    box.innerHTML=highlightLog(combined); box.scrollTop=box.scrollHeight;
  } catch(e) { document.getElementById('log-box').textContent='Failed'; }
}
function refreshLogs(){ loadLog(logTab); }

/* ═══ File Browser ═══ */
async function browseTo(path) {
  browsePath=path;
  try {
    const d=await(await fetch(`/api/browse?path=${encodeURIComponent(path)}`)).json();
    if(d.error){ document.getElementById('fb-list').textContent=d.error; return; }
    let bc='';
    d.breadcrumb.forEach((c,i)=>{ if(i>0) bc+='<span class="sep">/</span>'; bc+=`<a onclick="browseTo('${c.path.replace(/'/g,"\\'")}')">${escHtml(c.name)}</a>`; });
    document.getElementById('fb-bc').innerHTML=bc;
    let h='';
    if(path){ const p=path.split('/').slice(0,-1).join('/'); h+=`<div class="browser-item" onclick="browseTo('${p.replace(/'/g,"\\'")}')"><span class="icon">&#8593;</span><span class="name">..</span><span class="meta"></span></div>`; }
    for(const f of d.folders) h+=`<div class="browser-item" onclick="browseTo('${f.path.replace(/'/g,"\\'")}')"><span class="icon">&#128193;</span><span class="name">${escHtml(f.name)}</span><span class="meta">${f.file_count} files</span></div>`;
    for(const f of d.files) h+=`<div class="browser-item" data-path="${f.path}" onclick="previewFile('${f.path.replace(/'/g,"\\'")}',this)"><span class="icon">&#128196;</span><span class="name">${escHtml(f.name)}</span><span class="meta">${fmtSize(f.size)}</span></div>`;
    if(!d.folders.length&&!d.files.length) h='<div style="padding:16px;color:var(--muted);text-align:center">Empty</div>';
    document.getElementById('fb-list').innerHTML=h;
  } catch(e) { document.getElementById('fb-list').textContent='Failed'; }
}
function renderMarkdown(raw) {
  let body=raw, fm='';
  const m=body.match(/^---\n([\s\S]*?)\n---\n?/);
  if(m){ fm=m[1]; body=body.slice(m[0].length); }
  body=escHtml(body);
  body=body.replace(/```(\w*)\n([\s\S]*?)```/g,'<pre style="background:#0a0c10;border:1px solid var(--border);border-radius:6px;padding:12px;overflow-x:auto;font-size:.8rem">$2</pre>');
  body=body.replace(/^### (.+)$/gm,'<h3>$1</h3>');
  body=body.replace(/^## (.+)$/gm,'<h2>$1</h2>');
  body=body.replace(/^# (.+)$/gm,'<h1>$1</h1>');
  body=body.replace(/\*\*(.+?)\*\*/g,'<strong>$1</strong>');
  body=body.replace(/\*(.+?)\*/g,'<em>$1</em>');
  body=body.replace(/\[\[([^\]|]+)\|([^\]]+)\]\]/g,'<a>$2</a>');
  body=body.replace(/\[\[([^\]]+)\]\]/g,'<a>$1</a>');
  body=body.replace(/\[([^\]]+)\]\(([^)]+)\)/g,'<a href="$2">$1</a>');
  body=body.replace(/^&gt; (.+)$/gm,'<blockquote>$1</blockquote>');
  body=body.replace(/^---$/gm,'<hr>');
  body=body.replace(/^- (.+)$/gm,'<li>$1</li>');
  body=body.replace(/\n\n/g,'</p><p>');
  body='<p>'+body+'</p>';
  body=body.replace(/<p><\/p>/g,'');
  let html='';
  if(fm) html+=`<div class="frontmatter">${escHtml(fm)}</div>`;
  return html+body;
}
async function previewFile(path,el) {
  document.querySelectorAll('.browser-item.selected').forEach(e=>e.classList.remove('selected'));
  if(el) el.classList.add('selected');
  const pv=document.getElementById('fb-preview');
  pv.innerHTML='<div class="preview-empty">Loading...</div>';
  try {
    const d=await(await fetch(`/api/file?path=${encodeURIComponent(path)}`)).json();
    if(d.error){ pv.innerHTML=`<div class="preview-empty">${escHtml(d.error)}</div>`; return; }
    pv.innerHTML=`<div class="preview-header"><span class="filename">${escHtml(d.name)}</span><span class="file-meta">${fmtSize(d.size)} &middot; ${fmtDate(d.modified)}${d.truncated?' &middot; (truncated)':''}</span></div><div class="preview-body">${renderMarkdown(d.content)}</div>`;
  } catch(e) { pv.innerHTML='<div class="preview-empty">Failed</div>'; }
}

/* ═══ Init ═══ */
refresh(); loadSync(); loadPipeline();
setInterval(refresh, R);
setInterval(loadSync, 60000);
browseTo('');
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def dashboard() -> HTMLResponse:
    return HTMLResponse(DASHBOARD_HTML)


# ── Entrypoint ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.dashboard.app:app",
        host="0.0.0.0",
        port=8765,
        reload=False,
        log_level="info",
    )
