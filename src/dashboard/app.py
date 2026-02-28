#!/usr/bin/env python3
"""MemoryOS Control Panel -- FastAPI dashboard for monitoring all extractors.

Run with:
    python3 -m uvicorn src.dashboard.app:app --host 0.0.0.0 --port 8765
"""

from __future__ import annotations

import asyncio
import glob as globmod
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, BackgroundTasks, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.common.config import load_config
from src.common.state import load_state
from src.dashboard.report_fallback import extract_embedded_json, normalize_report, parse_project_brief_markdown

logger = logging.getLogger("memoryos.dashboard")

app = FastAPI(title="MemoryOS Control Panel", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from src.chat.routes import router as chat_router, prune_idle_sessions as _prune_chat_sessions
app.include_router(chat_router)

REPO_DIR = Path(__file__).resolve().parent.parent.parent

# Cache SSID lookups to avoid repeated subprocess calls on every refresh
_ssid_cache: dict[str, Any] = {"ssid": None, "ts": 0.0}
_SSID_CACHE_TTL = 15.0

# Track in-flight skill runs so the frontend can detect completion/failure
_skill_jobs: dict[str, dict[str, Any]] = {}
VENV_PYTHON = REPO_DIR / ".venv" / "bin" / "python3"
CONFIG_PATH = REPO_DIR / "config" / "config.yaml"
LAUNCHD_DIR = REPO_DIR / "launchd"
AGENTS_DIR = Path.home() / "Library" / "LaunchAgents"


def _cfg() -> dict[str, Any]:
    return load_config(CONFIG_PATH)


def _load_env_local() -> None:
    """Load .env.local into os.environ so agent modules can find API keys."""
    env_file = REPO_DIR / ".env.local"
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())


# ══════════════════════════════════════════════════════════════════════════════
#  TTL cache for expensive operations
# ══════════════════════════════════════════════════════════════════════════════

_ttl_cache: dict[str, tuple[float, Any]] = {}


def _cached(key: str, ttl: float, fn: Any, *args: Any) -> Any:
    """Return cached result if within TTL, otherwise call fn and cache."""
    now = time.time()
    entry = _ttl_cache.get(key)
    if entry and (now - entry[0]) < ttl:
        return entry[1]
    result = fn(*args)
    _ttl_cache[key] = (now, result)
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  Helper functions
# ══════════════════════════════════════════════════════════════════════════════

def _count_files_uncached(directory: Path, pattern: str = "*.md") -> int:
    if not directory.exists():
        return 0
    return sum(1 for _ in directory.rglob(pattern))


def _count_files(directory: Path, pattern: str = "*.md") -> int:
    return _cached(f"count:{directory}:{pattern}", 30.0, _count_files_uncached, directory, pattern)


def _dir_size_mb_uncached(directory: Path) -> float:
    if not directory.exists():
        return 0.0
    total = sum(f.stat().st_size for f in directory.rglob("*") if f.is_file())
    return round(total / (1024 * 1024), 1)


def _dir_size_mb(directory: Path) -> float:
    return _cached(f"size:{directory}", 30.0, _dir_size_mb_uncached, directory)


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


def _is_new_outlook() -> bool:
    """Detect if Outlook is in 'New Outlook' mode (cloud-only, no local DB writes).

    Uses two signals: the macOS defaults key (fast path) and DB staleness
    (fallback when the defaults key is absent or the command hangs).
    """
    try:
        result = subprocess.run(
            ["defaults", "read", "com.microsoft.Outlook", "IsRunningNewOutlook"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip() == "1":
            return True
    except Exception:
        pass

    db = (
        Path.home()
        / "Library" / "Group Containers" / "UBF8T346G9.Office"
        / "Outlook" / "Outlook 15 Profiles" / "Main Profile"
        / "Data" / "Outlook.sqlite"
    )
    if db.is_file():
        stale_seconds = time.time() - db.stat().st_mtime
        if stale_seconds > 48 * 3600:
            return True
    return False


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
                    "loaded": True,
                    "healthy": pid is not None or exit_code == 0 or exit_code is None,
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


def _recent_errors(log_dir: Path, log_file: str, max_age_hours: int = 4) -> int:
    """Count ERROR/Traceback lines only from the last *max_age_hours*."""
    p = log_dir / log_file
    if not p.exists():
        return 0
    try:
        from datetime import datetime, timedelta
        cutoff = datetime.now() - timedelta(hours=max_age_hours)
        cutoff_str = cutoff.strftime("%Y-%m-%d %H:%M")
        count = 0
        for line in p.read_text(errors="replace").splitlines():
            if len(line) >= 16 and line[:4].isdigit():
                if line[:16] < cutoff_str:
                    continue
            low = line.lower()
            if "error" in low or "traceback" in low:
                count += 1
        return count
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


def _screenpipe_device_api(path: str, body: dict[str, Any] | None = None) -> Any:
    """Call a Screenpipe device API endpoint synchronously."""
    import urllib.request
    import urllib.error
    url = f"http://localhost:3030{path}"
    if body is not None:
        data = json.dumps(body).encode()
        req = urllib.request.Request(url, data=data, method="POST")
        req.add_header("Content-Type", "application/json")
    else:
        req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=5) as resp:
        return json.loads(resp.read())


@app.get("/api/audio/devices")
async def api_audio_devices() -> JSONResponse:
    """List available audio input devices from Screenpipe."""
    cfg = _cfg()
    preferred = cfg.get("screenpipe", {}).get("preferred_input_device")
    try:
        devices = _screenpipe_device_api("/audio/list")
    except Exception:
        return JSONResponse({
            "devices": [],
            "preferred": preferred,
            "error": "Screenpipe API unreachable",
        })
    input_devices = [
        {"name": d.get("name", ""), "is_input": d.get("is_input", True)}
        for d in devices
        if d.get("name")
    ]
    return JSONResponse({
        "devices": input_devices,
        "preferred": preferred,
    })


@app.post("/api/audio/device")
async def api_audio_device_switch(request: Request) -> JSONResponse:
    """Switch the active audio input device in Screenpipe and persist preference."""
    body = await request.json()
    device_name = body.get("device_name", "").strip()
    if not device_name:
        return JSONResponse({"error": "device_name is required"}, status_code=400)

    errors: list[str] = []

    try:
        devices = _screenpipe_device_api("/audio/list")
    except Exception:
        return JSONResponse(
            {"error": "Screenpipe API unreachable"}, status_code=502,
        )

    known_names = {d.get("name", "") for d in devices if d.get("name")}
    if device_name not in known_names:
        return JSONResponse(
            {"error": f"Unknown device: {device_name}", "available": sorted(known_names)},
            status_code=404,
        )

    for d in devices:
        name = d.get("name", "")
        if not name or name == device_name:
            continue
        try:
            _screenpipe_device_api("/audio/device/stop", {"device_name": name})
        except Exception:
            errors.append(f"Failed to stop {name}")

    try:
        _screenpipe_device_api("/audio/device/start", {"device_name": device_name})
    except Exception:
        errors.append(f"Failed to start {device_name}")

    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        raw.setdefault("screenpipe", {})["preferred_input_device"] = device_name
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            yaml.dump(
                raw, f,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
                width=200,
            )
    except Exception as exc:
        errors.append(f"Config save failed: {exc}")

    return JSONResponse({
        "status": "ok" if not errors else "partial",
        "active_device": device_name,
        "errors": errors,
    })


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
                "health": "skipped" if _is_new_outlook() else _extractor_health(state_age, 600),
                "new_outlook": _is_new_outlook(),
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

@app.get("/api/watchdog")
async def api_watchdog() -> JSONResponse:
    """Return the latest watchdog state if available."""
    state_file = REPO_DIR / "config" / "watchdog_state.json"
    if not state_file.is_file():
        return JSONResponse({"error": "No watchdog data yet"}, status_code=404)
    try:
        data = json.loads(state_file.read_text())
        return JSONResponse(data)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


_gpu_name_cache: str | None = None


def _get_gpu_name() -> str:
    """Cache GPU name from system_profiler (expensive call, run once)."""
    global _gpu_name_cache
    if _gpu_name_cache is not None:
        return _gpu_name_cache
    try:
        import subprocess as _sp
        r = _sp.run(
            ["system_profiler", "SPDisplaysDataType", "-json"],
            capture_output=True, text=True, timeout=5,
        )
        gpus = json.loads(r.stdout).get("SPDisplaysDataType", [])
        _gpu_name_cache = gpus[0].get("sppci_model", "Unknown") if gpus else "N/A"
    except Exception:
        _gpu_name_cache = "N/A"
    return _gpu_name_cache


@app.get("/api/system-health")
async def api_system_health() -> JSONResponse:
    """Return resource monitor state enriched with live psutil metrics."""
    import psutil as _ps

    state_file = REPO_DIR / "config" / "resource_monitor_state.json"
    data: dict[str, Any] = {}
    if state_file.is_file():
        try:
            data = json.loads(state_file.read_text())
        except Exception:
            pass

    try:
        data["cpu_per_core"] = _ps.cpu_percent(interval=0.1, percpu=True)
        data["cpu_count"] = _ps.cpu_count(logical=True)
        freq = _ps.cpu_freq()
        data["cpu_freq_ghz"] = round(freq.current, 1) if freq else None

        data["load_avg"] = [round(x, 2) for x in _ps.getloadavg()]
        data["uptime_hours"] = round((time.time() - _ps.boot_time()) / 3600, 1)

        net = _ps.net_io_counters()
        data["net_io"] = {
            "sent_mb": round(net.bytes_sent / (1024 * 1024), 1),
            "recv_mb": round(net.bytes_recv / (1024 * 1024), 1),
        }

        bat = _ps.sensors_battery()
        if bat:
            data["battery"] = {"percent": bat.percent, "plugged": bat.power_plugged}
        else:
            data["battery"] = None

        data["gpu_name"] = _get_gpu_name()

        procs: list[dict[str, Any]] = []
        for p in _ps.process_iter(["pid", "name", "memory_info", "cpu_percent"]):
            try:
                pi = p.info
                rss = pi["memory_info"].rss / (1024 * 1024) if pi["memory_info"] else 0
                if rss > 50:
                    procs.append({
                        "pid": pi["pid"],
                        "name": (pi["name"] or "")[:40],
                        "rss_mb": round(rss, 1),
                        "cpu": pi["cpu_percent"] or 0,
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        procs.sort(key=lambda x: x["rss_mb"], reverse=True)
        data["top_processes"] = procs[:12]

        if "system" not in data:
            vm = _ps.virtual_memory()
            data["system"] = {
                "ram_total_gb": round(vm.total / (1024**3), 1),
                "ram_used_gb": round(vm.used / (1024**3), 1),
                "ram_percent": vm.percent,
                "cpu_percent": _ps.cpu_percent(),
            }
        if "disk" not in data:
            du = _ps.disk_usage("/")
            data["disk"] = {
                "disk_total_gb": round(du.total / (1024**3), 1),
                "disk_free_gb": round(du.free / (1024**3), 1),
                "disk_percent": du.percent,
            }

    except Exception as exc:
        logger.warning("Live psutil enrichment failed: %s", exc)

    if not data:
        return JSONResponse({"error": "No resource monitor data yet"}, status_code=404)

    return JSONResponse(data)


# ══════════════════════════════════════════════════════════════════════════════
#  API: Setup Wizard
# ══════════════════════════════════════════════════════════════════════════════

CONFIG_EXAMPLE = REPO_DIR / "config" / "config.yaml.example"
PLACEHOLDER_VAULT = "~/Documents/Obsidian/MyVault"


def _check(ok: bool, detail: str, fix: str = "", optional: bool = False) -> dict[str, str]:
    if ok:
        return {"status": "ok", "detail": detail, "fix": ""}
    return {"status": "warning" if optional else "missing", "detail": detail, "fix": fix}


def _detect_obsidian_vaults() -> list[str]:
    """Scan common locations for Obsidian vaults (dirs containing .obsidian/)."""
    candidates: list[str] = []
    home = Path.home()
    search_roots = [
        home / "Documents" / "Obsidian",
        home / "Documents",
        home / "Obsidian",
        home / "Data" / "Obsidian",
        home / "Library" / "Mobile Documents" / "iCloud~md~obsidian" / "Documents",
    ]
    for root in search_roots:
        try:
            if not root.is_dir():
                continue
            for child in root.iterdir():
                try:
                    if child.is_dir() and (child / ".obsidian").is_dir():
                        candidates.append(str(child))
                except (PermissionError, OSError):
                    continue
        except (PermissionError, OSError):
            continue
    return sorted(set(candidates))


def _detect_onedrive() -> list[str]:
    cloud = Path.home() / "Library" / "CloudStorage"
    try:
        if not cloud.is_dir():
            return []
        return sorted(
            str(p) for p in cloud.iterdir()
            if p.is_dir() and "OneDrive" in p.name
        )
    except (PermissionError, OSError):
        return []


@app.get("/api/setup/status")
async def api_setup_status() -> JSONResponse:
    """Return dependency health, config state, and auto-detected paths."""
    checks: dict[str, Any] = {}

    # Python
    v = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    checks["python"] = _check(True, f"Python {v}")

    # Homebrew
    checks["homebrew"] = _check(
        shutil.which("brew") is not None,
        "Homebrew installed" if shutil.which("brew") else "Homebrew not found",
        fix='Install: /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"',
    )

    # Screenpipe installed
    sp_dir = Path.home() / ".screenpipe"
    sp_installed = shutil.which("screenpipe") is not None or sp_dir.is_dir()
    checks["screenpipe_installed"] = _check(
        sp_installed,
        "Screenpipe installed" if sp_installed else "Screenpipe not found",
        fix="Install from https://screenpipe.com or: brew install screenpipe",
    )

    # Screenpipe running -- check main DB or WAL file (SQLite WAL mode
    # writes to the -wal file; main DB mtime only updates on checkpoint)
    sp_db = sp_dir / "db.sqlite"
    sp_wal = sp_dir / "db.sqlite-wal"
    sp_running = False
    if sp_db.is_file():
        newest_mtime = sp_db.stat().st_mtime
        if sp_wal.is_file():
            newest_mtime = max(newest_mtime, sp_wal.stat().st_mtime)
        age = time.time() - newest_mtime
        sp_running = age < 300
    checks["screenpipe_running"] = _check(
        sp_running,
        "Screenpipe active" if sp_running else "Screenpipe DB not updating",
        fix="Open the Screenpipe app or run: screenpipe",
    )

    # Pandoc
    checks["pandoc"] = _check(
        shutil.which("pandoc") is not None,
        "pandoc installed" if shutil.which("pandoc") else "pandoc not found",
        fix="brew install pandoc",
        optional=True,
    )

    # BlackHole
    bh_ok = False
    try:
        r = subprocess.run(["brew", "list", "blackhole-2ch"], capture_output=True, timeout=5)
        bh_ok = r.returncode == 0
    except Exception:
        pass
    checks["blackhole"] = _check(
        bh_ok,
        "BlackHole 2ch installed" if bh_ok else "BlackHole not installed (optional — needed for meeting audio)",
        fix="brew install blackhole-2ch",
        optional=True,
    )

    # Config file
    cfg_exists = CONFIG_PATH.is_file()
    cfg_placeholder = True
    if cfg_exists:
        try:
            raw = yaml.safe_load(CONFIG_PATH.read_text())
            cfg_placeholder = raw.get("obsidian_vault", "") == PLACEHOLDER_VAULT
        except Exception:
            pass
    checks["config_exists"] = _check(
        cfg_exists and not cfg_placeholder,
        "config.yaml configured" if (cfg_exists and not cfg_placeholder) else (
            "config.yaml has placeholder values" if cfg_exists else "config.yaml not created yet"
        ),
        fix="Complete Step 2 below to generate your config.",
    )

    # Obsidian vault reachable
    vault_ok = False
    vault_path = ""
    if cfg_exists and not cfg_placeholder:
        try:
            raw = yaml.safe_load(CONFIG_PATH.read_text())
            vault_path = str(Path(raw.get("obsidian_vault", "")).expanduser())
            vault_ok = Path(vault_path).is_dir()
        except Exception:
            pass
    checks["obsidian_vault"] = _check(
        vault_ok,
        f"Vault found: {vault_path}" if vault_ok else "Obsidian vault not reachable",
        fix="Set vault path in Step 2 or download Obsidian from https://obsidian.md",
    )

    # Launchd agents
    installed = list(AGENTS_DIR.glob("com.memoryos.*.plist")) if AGENTS_DIR.is_dir() else []
    checks["launchd_installed"] = _check(
        len(installed) > 0,
        f"{len(installed)} launchd agents installed" if installed else "No launchd agents installed",
        fix="Complete Step 3 to install background agents.",
    )

    # Memory index
    mem_db = REPO_DIR / "config" / "memory.db"
    checks["index_built"] = _check(
        mem_db.is_file() and mem_db.stat().st_size > 4096,
        "Search index built" if mem_db.is_file() else "Search index not built yet",
        fix="Complete Step 3 to build the index.",
    )

    # Auto-detected paths
    suggested: dict[str, Any] = {
        "obsidian_vaults": _detect_obsidian_vaults(),
        "onedrive_dirs": _detect_onedrive(),
        "screenpipe_db": str(sp_db) if sp_db.is_file() else None,
    }
    # Outlook DB
    ol_path = Path.home() / "Library" / "Group Containers" / "UBF8T346G9.Office" / "Outlook" / "Outlook 15 Profiles" / "Main Profile" / "Data" / "Outlook.sqlite"
    suggested["outlook_db"] = str(ol_path) if ol_path.is_file() else None

    return JSONResponse({
        "checks": checks,
        "suggested_paths": suggested,
        "setup_complete": all(
            c["status"] == "ok" for k, c in checks.items()
            if k not in ("pandoc", "blackhole", "screenpipe_running")
        ),
    })


@app.post("/api/setup/config")
async def api_setup_config(request: Request) -> JSONResponse:
    """Write config/config.yaml from the wizard form data."""
    body = await request.json()
    vault = body.get("obsidian_vault", "").strip()
    if not vault:
        return JSONResponse({"error": "obsidian_vault is required"}, status_code=400)
    vault_path = Path(vault).expanduser()
    if not vault_path.is_dir():
        return JSONResponse({"error": f"Vault directory not found: {vault}"}, status_code=400)

    if not CONFIG_EXAMPLE.is_file():
        return JSONResponse({"error": "config.yaml.example template missing"}, status_code=500)

    cfg = yaml.safe_load(CONFIG_EXAMPLE.read_text())
    cfg["obsidian_vault"] = str(vault_path)

    email_source = body.get("email_source", "mail_app")
    calendar_source = body.get("calendar_source", "calendar_app")

    if body.get("graph_client_id"):
        cfg["graph"]["client_id"] = body["graph_client_id"]

    onedrive_dir = body.get("onedrive_dir", "").strip()
    if onedrive_dir:
        cfg["onedrive"]["sync_dir"] = onedrive_dir

    cfg["_selected_email"] = email_source
    cfg["_selected_calendar"] = calendar_source

    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "w", encoding="utf-8") as fh:
        yaml.dump(cfg, fh, default_flow_style=False, sort_keys=False, allow_unicode=True)

    return JSONResponse({"status": "ok", "path": str(CONFIG_PATH)})


@app.post("/api/setup/install-agents")
async def api_setup_install_agents() -> JSONResponse:
    """Install launchd agents from plist templates."""
    results: dict[str, str] = {}
    AGENTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPO_DIR / "logs").mkdir(parents=True, exist_ok=True)
    venv_py = str(VENV_PYTHON)
    repo = str(REPO_DIR)

    for template in sorted(LAUNCHD_DIR.glob("com.memoryos.*.plist.template")):
        name = template.stem.replace(".plist", "")
        plist_name = f"{name}.plist"
        dest = AGENTS_DIR / plist_name
        try:
            if dest.is_file():
                subprocess.run(["launchctl", "unload", str(dest)], capture_output=True, timeout=10)
            content = template.read_text()
            content = content.replace("{{VENV_PYTHON}}", venv_py).replace("{{REPO_DIR}}", repo)
            dest.write_text(content)
            subprocess.run(["launchctl", "load", str(dest)], capture_output=True, timeout=10)
            results[name] = "installed"
        except Exception as exc:
            results[name] = f"error: {exc}"

    return JSONResponse({"status": "ok", "agents": results})


@app.post("/api/setup/reindex")
async def api_setup_reindex(background_tasks: BackgroundTasks) -> JSONResponse:
    """Trigger a full memory index rebuild as a background task."""
    def _reindex() -> None:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(REPO_DIR)
        try:
            subprocess.run(
                [str(VENV_PYTHON), "-m", "src.memory.cli", "reindex"],
                env=env, capture_output=True, timeout=600,
            )
        except Exception as exc:
            logger.error("Reindex failed: %s", exc)

    background_tasks.add_task(_reindex)
    return JSONResponse({"status": "started"})


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
#  API: My Day
# ══════════════════════════════════════════════════════════════════════════════

_CAL_EVENT_RE = re.compile(
    r"^## (?:All Day:\s*(?P<allday>.+)|(?P<start>\d{2}:\d{2})\s*-\s*(?P<end>\d{2}:\d{2}):\s*(?P<title>.+))$"
)
_RECALL_BLOCK_RE = re.compile(
    r"^## (?P<start>\d{2}:\d{2})\s*-\s*(?P<end>\d{2}:\d{2})\s*\|\s*(?P<title>.+)$"
)
_AUDIO_SPEAKER_RE = re.compile(r"^\*\*(?P<name>[^*]+)\s*\(")

_MONTH_ABBR = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def _parse_calendar_md(text: str) -> list[dict[str, Any]]:
    """Parse calendar.md into a list of event dicts."""
    events: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    for line in text.splitlines():
        m = _CAL_EVENT_RE.match(line.strip())
        if m:
            if current:
                events.append(current)
            if m.group("allday"):
                current = {
                    "summary": m.group("allday").strip(),
                    "start": None, "end": None,
                    "is_all_day": True, "location": "",
                    "attendees": [], "organizer": "",
                }
            else:
                current = {
                    "summary": (m.group("title") or "").strip(),
                    "start": m.group("start"),
                    "end": m.group("end"),
                    "is_all_day": False, "location": "",
                    "attendees": [], "organizer": "",
                }
            continue

        if current and line.strip().startswith("- **"):
            kv = line.strip()[4:]
            if kv.startswith("Location:**"):
                current["location"] = kv.split(":**", 1)[1].strip()
            elif kv.startswith("Organizer:**"):
                current["organizer"] = kv.split(":**", 1)[1].strip()
            elif kv.startswith("Attendees:**"):
                raw = kv.split(":**", 1)[1].strip()
                current["attendees"] = [a.strip() for a in raw.split(",") if a.strip()]

    if current:
        events.append(current)
    return events


def _parse_recall_md(text: str) -> list[dict[str, Any]]:
    """Parse recall.md into task-recall blocks."""
    blocks: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    for line in text.splitlines():
        m = _RECALL_BLOCK_RE.match(line.strip())
        if m:
            if current:
                blocks.append(current)
            current = {
                "start": m.group("start"),
                "end": m.group("end"),
                "title": m.group("title").strip(),
                "apps": [], "details": [], "artifacts": [],
            }
            continue

        if current:
            stripped = line.strip()
            if stripped.startswith("**Apps:**"):
                current["apps"] = [a.strip() for a in stripped[9:].split(",") if a.strip()]
            elif stripped.startswith("**Tasks:**"):
                pass
            elif stripped.startswith("**Artifacts:**"):
                pass
            elif stripped.startswith("- ") and current:
                item = stripped[2:].strip()
                if current["artifacts"] is not None and any(
                    kw in item.lower() for kw in ("http", "://", ".md", ".py", ".ts", "pid", "email")
                ):
                    current["artifacts"].append(item)
                else:
                    current["details"].append(item)

    if current:
        blocks.append(current)
    return blocks


_SPEAKER_N_RE = re.compile(r"^Speaker\s+\d+$", re.IGNORECASE)


def _extract_people(calendar_events: list[dict], audio_text: str) -> list[dict[str, Any]]:
    """Aggregate people from calendar attendees and audio speakers.

    Filters out anonymous 'Speaker N' placeholders from Screenpipe.
    Uses calendar attendees as a proxy for meeting participants when
    speaker diarization can't identify voices.
    """
    people: dict[str, dict[str, Any]] = {}

    for ev in calendar_events:
        if ev.get("is_all_day"):
            continue
        start_str = ev.get("start", "")
        end_str = ev.get("end", "")
        try:
            s = datetime.strptime(start_str, "%H:%M")
            e = datetime.strptime(end_str, "%H:%M")
            dur = max((e - s).seconds / 60, 0)
        except (ValueError, TypeError):
            dur = 30

        for name in ev.get("attendees", []):
            name = name.strip()
            if not name:
                continue
            if name not in people:
                people[name] = {"name": name, "meetings": 0, "total_minutes": 0}
            people[name]["meetings"] += 1
            people[name]["total_minutes"] += dur

        org = (ev.get("organizer") or "").strip()
        if org and org not in people:
            people[org] = {"name": org, "meetings": 0, "total_minutes": 0}
        if org and org in people:
            people[org]["meetings"] = max(people[org]["meetings"], 1)

    for line in audio_text.splitlines():
        m = _AUDIO_SPEAKER_RE.match(line.strip())
        if m:
            name = m.group("name").strip()
            if name.lower() in ("you", "speaker", "unknown"):
                continue
            if _SPEAKER_N_RE.match(name):
                continue
            if name not in people:
                people[name] = {"name": name, "meetings": 0, "total_minutes": 0}

    return sorted(people.values(), key=lambda p: -p["total_minutes"])


def _fuzzy_date_match(date_str: str, target: datetime) -> bool:
    """Check if a human date string like 'Feb 25' matches a target date."""
    if not date_str:
        return False
    date_str = date_str.strip().lower()
    parts = date_str.replace(",", " ").split()
    month = None
    day = None
    for p in parts:
        if p in _MONTH_ABBR:
            month = _MONTH_ABBR[p]
        elif p.isdigit():
            day = int(p)
    if month and day:
        return target.month == month and target.day == day
    return False


def _get_work_for_date(vault: Path, target: datetime) -> tuple[list[dict], list[dict]]:
    """Return (completed, in_progress) tasks for a given date from tasks.md."""
    tasks_path = vault / CONTEXT_DIR / "tasks.md"
    if not tasks_path.is_file():
        return [], []

    content = tasks_path.read_text(encoding="utf-8", errors="replace")
    parsed = _parse_tasks_md(content)

    completed: list[dict[str, Any]] = []
    in_progress: list[dict[str, Any]] = []

    def _scan_tasks(tasks: list[dict], project_name: str) -> None:
        for t in tasks:
            if t["status"] == "complete" and _fuzzy_date_match(t.get("due", ""), target):
                completed.append({
                    "task": t["task"], "priority": t.get("priority", "P1"),
                    "project": project_name,
                })
            elif t["status"] == "in_progress":
                in_progress.append({
                    "task": t["task"], "priority": t.get("priority", "P1"),
                    "project": project_name, "notes": t.get("notes", ""),
                })
            for sub in t.get("subtasks", []):
                if sub["status"] == "complete" and _fuzzy_date_match(sub.get("due", ""), target):
                    completed.append({
                        "task": sub["task"], "priority": sub.get("priority", "P1"),
                        "project": project_name,
                    })
                elif sub["status"] == "in_progress":
                    in_progress.append({
                        "task": sub["task"], "priority": sub.get("priority", "P1"),
                        "project": project_name, "notes": sub.get("notes", ""),
                    })

    for proj in parsed.get("projects", []):
        _scan_tasks(proj.get("tasks", []), proj.get("name", ""))

    return completed, in_progress


@app.get("/api/myday/{date}")
async def api_myday(date: str) -> JSONResponse:
    """Comprehensive day view: calendar, tasks, activity, people, work status."""
    cfg = _cfg()
    vault = Path(cfg["obsidian_vault"])

    try:
        dt = datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        return JSONResponse({"error": "Invalid date format, use YYYY-MM-DD"}, status_code=400)

    date_path = dt.strftime("%Y/%m/%d")

    calendar_events: list[dict] = []
    cal_file = vault / cfg["output"]["meetings"] / date_path / "calendar.md"
    if cal_file.is_file():
        calendar_events = _parse_calendar_md(
            cal_file.read_text(encoding="utf-8", errors="replace")
        )

    recall_tasks: list[dict] = []
    recall_dir = cfg["output"].get("recall", "87_recall")
    recall_file = vault / recall_dir / date_path / "recall.md"
    if recall_file.is_file():
        recall_tasks = _parse_recall_md(
            recall_file.read_text(encoding="utf-8", errors="replace")
        )

    activity_data: dict[str, Any] = {
        "total_active_hours": 0, "total_context_switches": 0,
        "app_breakdown": [], "hourly": {}, "context_switches_per_hour": {},
    }
    activity_file = vault / cfg["output"]["activity"] / date_path / "daily.md"
    if activity_file.is_file():
        from src.analyzers.activity_stats import parse_daily_activity
        stats = parse_daily_activity(
            activity_file.read_text(encoding="utf-8", errors="replace")
        )
        activity_data = {
            "total_active_hours": stats.total_active_hours,
            "total_context_switches": stats.total_context_switches,
            "app_breakdown": stats.to_app_breakdown(),
            "hourly": stats.hourly_app_minutes,
            "context_switches_per_hour": stats.context_switches_per_hour,
        }

    audio_text = ""
    audio_file = vault / cfg["output"]["meetings"] / date_path / "audio.md"
    if audio_file.is_file():
        audio_text = audio_file.read_text(encoding="utf-8", errors="replace")

    people = _extract_people(calendar_events, audio_text)

    work_completed, work_in_progress = _get_work_for_date(vault, dt)

    has_data = bool(calendar_events or recall_tasks or activity_data["app_breakdown"] or people)

    return JSONResponse({
        "date": date,
        "calendar": calendar_events,
        "tasks": recall_tasks,
        "activity": activity_data,
        "people": people,
        "work_completed": work_completed,
        "work_in_progress": work_in_progress,
        "has_data": has_data,
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
#  Skills API
# ══════════════════════════════════════════════════════════════════════════════

SKILLS_DIR = Path.home() / ".cursor" / "skills"


@app.get("/api/skills")
async def api_skills() -> JSONResponse:
    """Return metadata and content for all installed agent skills."""
    skills: list[dict[str, Any]] = []
    if not SKILLS_DIR.is_dir():
        return JSONResponse(skills)
    for skill_dir in sorted(SKILLS_DIR.iterdir()):
        skill_md = skill_dir / "SKILL.md"
        if not skill_md.is_file():
            continue
        try:
            raw = skill_md.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        name = skill_dir.name
        description = ""
        body = raw
        if raw.startswith("---"):
            end = raw.find("\n---", 3)
            if end != -1:
                fm_text = raw[3:end].strip()
                body = raw[end + 4:].lstrip("\n")
                for line in fm_text.splitlines():
                    if line.startswith("description:"):
                        description = line.split(":", 1)[1].strip().strip('"').strip("'")
                    elif line.startswith("name:"):
                        name = line.split(":", 1)[1].strip()
        sections: list[str] = []
        for s_line in body.splitlines():
            if s_line.startswith("## "):
                sections.append(s_line[3:].strip())
        has_scripts = (skill_dir / "scripts").is_dir()
        file_count = sum(1 for f in skill_dir.rglob("*") if f.is_file())
        manifest_path = skill_dir / "manifest.yaml"
        manifest_data: dict[str, Any] | None = None
        if manifest_path.is_file():
            try:
                manifest_data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        skills.append({
            "name": name,
            "dir_name": skill_dir.name,
            "description": description,
            "sections": sections,
            "has_scripts": has_scripts,
            "file_count": file_count,
            "body": body,
            "has_manifest": manifest_data is not None,
            "schedule": manifest_data.get("schedule") if manifest_data else None,
        })
    return JSONResponse(skills)


# ══════════════════════════════════════════════════════════════════════════════
#  API: Agent Skills (headless execution, config, scheduling)
# ══════════════════════════════════════════════════════════════════════════════

SCHEDULED_SKILLS = {
    "morning-brief": "com.memoryos.morning-brief",
    "plan-my-week": "com.memoryos.plan-my-week",
    "weekly-status": "com.memoryos.weekly-status",
    "news-pulse": "com.memoryos.news-pulse",
    "commitment-tracker": "com.memoryos.commitment-tracker",
    "project-brief": "com.memoryos.project-brief",
    "focus-audit": "com.memoryos.focus-audit",
    "relationship-crm": "com.memoryos.relationship-crm",
    "team-manager": "com.memoryos.team-manager",
    "approvals-queue": "com.memoryos.approvals-queue",
}


@app.get("/api/agents/config")
async def api_agents_config_get() -> JSONResponse:
    """Return current agents config (API keys masked)."""
    _load_env_local()
    cfg = _cfg()
    agents = cfg.get("agents", {})
    email_cfg = agents.get("email", {})
    return JSONResponse({
        "provider": agents.get("provider", "openai"),
        "model": agents.get("model", "gpt-5.2"),
        "reasoning_effort": agents.get("reasoning_effort", "high"),
        "api_base": agents.get("api_base"),
        "temperature": agents.get("temperature", 0.3),
        "email": {
            "enabled": email_cfg.get("enabled", False),
            "from": email_cfg.get("from", ""),
            "to": email_cfg.get("to", ""),
            "smtp_host": email_cfg.get("smtp_host", "smtp.office365.com"),
            "smtp_port": email_cfg.get("smtp_port", 587),
            "smtp_user": email_cfg.get("smtp_user", ""),
        },
        "has_api_key": bool(os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
                            or os.environ.get("GEMINI_API_KEY") or os.environ.get("AZURE_API_KEY")),
        "has_smtp_password": bool(os.environ.get("MEMORYOS_SMTP_PASSWORD")),
    })


@app.post("/api/agents/config")
async def api_agents_config_post(request: Request) -> JSONResponse:
    """Save agents config to config.yaml."""
    body = await request.json()
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        agents = raw.setdefault("agents", {})
        for key in ("provider", "model", "reasoning_effort", "api_base", "temperature"):
            if key in body:
                agents[key] = body[key]
        if "email" in body:
            email_sec = agents.setdefault("email", {})
            for ek in ("enabled", "from", "to", "smtp_host", "smtp_port", "smtp_user"):
                if ek in body["email"]:
                    email_sec[ek] = body["email"][ek]
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            yaml.dump(raw, f, default_flow_style=False, sort_keys=False, allow_unicode=True, width=200)
        return JSONResponse({"status": "ok"})
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.post("/api/agents/test-llm")
async def api_agents_test_llm(request: Request) -> JSONResponse:
    """Send a test prompt to the configured LLM."""
    _load_env_local()
    body = await request.json() if await request.body() else {}
    try:
        from src.agents.llm_provider import test_connection
        result = test_connection(
            provider=body.get("provider"),
            model=body.get("model"),
            api_base=body.get("api_base"),
            api_key=body.get("api_key"),
        )
        return JSONResponse(result)
    except Exception as exc:
        return JSONResponse({"ok": False, "detail": str(exc)[:300]})


@app.post("/api/agents/test-email")
async def api_agents_test_email() -> JSONResponse:
    """Send a test email via SMTP."""
    _load_env_local()
    try:
        from src.agents.emailer import send_test_email
        result = send_test_email()
        return JSONResponse(result)
    except Exception as exc:
        return JSONResponse({"ok": False, "detail": str(exc)[:300]})


@app.post("/api/agents/run-skill/{skill_name}")
async def api_agents_run_skill(skill_name: str, background_tasks: BackgroundTasks) -> JSONResponse:
    """Trigger a skill run as a background task."""
    skill_dir = SKILLS_DIR / skill_name
    if not (skill_dir / "SKILL.md").is_file():
        return JSONResponse({"error": f"Skill not found: {skill_name}"}, status_code=404)

    existing = _skill_jobs.get(skill_name)
    if existing and existing.get("status") == "running":
        return JSONResponse(
            {"error": "Skill is already running", "started_at": existing["started_at"]},
            status_code=409,
        )

    started_at = time.time()
    _skill_jobs[skill_name] = {"status": "running", "started_at": started_at, "error": None}

    runner_module = None
    manifest_path = skill_dir / "manifest.yaml"
    if manifest_path.is_file():
        try:
            _m = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
            if _m.get("runner") == "scripted" and _m.get("runner_module"):
                runner_module = _m["runner_module"]
        except Exception:
            pass

    def _run() -> None:
        try:
            env = os.environ.copy()
            env["PYTHONPATH"] = str(REPO_DIR)
            env_local = REPO_DIR / ".env.local"
            if env_local.is_file():
                for line in env_local.read_text().splitlines():
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, _, v = line.partition("=")
                        env.setdefault(k.strip(), v.strip())
            cmd = [str(VENV_PYTHON), "-m", runner_module] if runner_module else \
                  [str(VENV_PYTHON), "-m", "src.agents.skill_runner", "--skill", skill_name]
            result = subprocess.run(
                cmd,
                env=env, capture_output=True, text=True, timeout=600, cwd=str(REPO_DIR),
            )
            if result.returncode != 0:
                err_tail = (result.stderr or "")[-500:]
                logger.error("Skill run '%s' exited %d: %s", skill_name, result.returncode, err_tail)
                _skill_jobs[skill_name] = {
                    "status": "failed", "started_at": started_at,
                    "error": err_tail or f"exit code {result.returncode}",
                }
            else:
                _skill_jobs[skill_name] = {"status": "completed", "started_at": started_at, "error": None}
        except Exception as exc:
            logger.error("Skill run '%s' failed: %s", skill_name, exc)
            _skill_jobs[skill_name] = {"status": "failed", "started_at": started_at, "error": str(exc)[:500]}

    background_tasks.add_task(_run)
    return JSONResponse({"status": "started", "skill": skill_name, "started_at": started_at})


@app.get("/api/agents/run-status/{skill_name}")
async def api_agents_run_status(skill_name: str) -> JSONResponse:
    """Return the status of the most recent skill run."""
    job = _skill_jobs.get(skill_name)
    if not job:
        return JSONResponse({"status": "idle", "started_at": None, "error": None})
    return JSONResponse(job)


@app.get("/api/agents/status")
async def api_agents_status() -> JSONResponse:
    """Return schedule, last run time, launchd status for all scheduled skills."""
    cfg = _cfg()
    vault = Path(cfg.get("obsidian_vault", "")).expanduser()
    reports_dir_name = cfg.get("agents", {}).get("reports_dir", "90_reports")

    loaded_agents: set[str] = set()
    try:
        proc = await asyncio.create_subprocess_exec(
            "launchctl", "list",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
        for line in stdout.decode().splitlines():
            for label in SCHEDULED_SKILLS.values():
                if label in line:
                    loaded_agents.add(label)
    except Exception:
        pass

    results: list[dict[str, Any]] = []
    for skill_name, label in SCHEDULED_SKILLS.items():
        manifest_path = SKILLS_DIR / skill_name / "manifest.yaml"
        schedule = ""
        if manifest_path.is_file():
            try:
                m = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
                schedule = m.get("schedule", "")
            except Exception:
                pass

        reports_path = vault / reports_dir_name / skill_name
        last_run = None
        last_report = None
        if reports_path.is_dir():
            files = sorted(reports_path.glob("*.md"), reverse=True)
            if files:
                last_report = files[0].name
                last_run = datetime.fromtimestamp(files[0].stat().st_mtime).isoformat()

        is_loaded = label in loaded_agents
        plist_installed = (AGENTS_DIR / f"{label}.plist").is_file()

        results.append({
            "skill": skill_name,
            "label": label,
            "schedule": schedule,
            "last_run": last_run,
            "last_report": last_report,
            "launchd_loaded": is_loaded,
            "plist_installed": plist_installed,
            "status": "active" if is_loaded else ("installed" if plist_installed else "not installed"),
        })

    return JSONResponse(results)


@app.get("/api/agents/topics")
async def api_agents_topics_get() -> JSONResponse:
    """Return news-pulse topics.yaml contents."""
    topics_file = SKILLS_DIR / "news-pulse" / "topics.yaml"
    if not topics_file.is_file():
        return JSONResponse({"topics": []})
    try:
        data = yaml.safe_load(topics_file.read_text(encoding="utf-8")) or {}
        return JSONResponse(data)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.post("/api/agents/topics")
async def api_agents_topics_post(request: Request) -> JSONResponse:
    """Update news-pulse topics.yaml."""
    body = await request.json()
    topics_file = SKILLS_DIR / "news-pulse" / "topics.yaml"
    topics_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(topics_file, "w", encoding="utf-8") as f:
            yaml.dump(body, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        return JSONResponse({"status": "ok"})
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


SHARED_CONTEXT_PATH = SKILLS_DIR / "_shared" / "context.yaml"


@app.get("/api/agents/context")
async def api_agents_context_get() -> JSONResponse:
    """Return shared context (my_context + global_context)."""
    if not SHARED_CONTEXT_PATH.is_file():
        return JSONResponse({"my_context": {}, "global_context": {}})
    try:
        data = yaml.safe_load(SHARED_CONTEXT_PATH.read_text(encoding="utf-8")) or {}
        return JSONResponse(data)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.post("/api/agents/context")
async def api_agents_context_post(request: Request) -> JSONResponse:
    """Update shared context (my_context + global_context)."""
    body = await request.json()
    SHARED_CONTEXT_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(SHARED_CONTEXT_PATH, "w", encoding="utf-8") as f:
            yaml.dump(body, f, default_flow_style=False, sort_keys=False, allow_unicode=True, width=200)
        return JSONResponse({"status": "ok"})
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.get("/api/agents/skill-config/{skill_name}")
async def api_agents_skill_config_get(skill_name: str) -> JSONResponse:
    """Return a skill's manifest.yaml + topics.yaml (if present) as JSON."""
    skill_dir = SKILLS_DIR / skill_name
    if not skill_dir.is_dir():
        return JSONResponse({"error": f"Skill not found: {skill_name}"}, status_code=404)

    result: dict[str, Any] = {"skill": skill_name}

    manifest_path = skill_dir / "manifest.yaml"
    if manifest_path.is_file():
        try:
            result["manifest"] = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
        except Exception:
            result["manifest"] = {}
    else:
        result["manifest"] = {}

    topics_path = skill_dir / "topics.yaml"
    if topics_path.is_file():
        try:
            result["topics"] = yaml.safe_load(topics_path.read_text(encoding="utf-8")) or {}
        except Exception:
            result["topics"] = {}

    return JSONResponse(result)


@app.post("/api/agents/skill-config/{skill_name}")
async def api_agents_skill_config_post(skill_name: str, request: Request) -> JSONResponse:
    """Update a skill's manifest.yaml and/or topics.yaml."""
    skill_dir = SKILLS_DIR / skill_name
    if not skill_dir.is_dir():
        return JSONResponse({"error": f"Skill not found: {skill_name}"}, status_code=404)

    body = await request.json()
    errors: list[str] = []

    if "manifest" in body:
        manifest_path = skill_dir / "manifest.yaml"
        try:
            existing = {}
            if manifest_path.is_file():
                existing = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
            existing.update(body["manifest"])
            with open(manifest_path, "w", encoding="utf-8") as f:
                yaml.dump(existing, f, default_flow_style=False, sort_keys=False, allow_unicode=True, width=200)
        except Exception as exc:
            errors.append(f"manifest: {exc}")

    if "topics" in body:
        topics_path = skill_dir / "topics.yaml"
        try:
            with open(topics_path, "w", encoding="utf-8") as f:
                yaml.dump(body["topics"], f, default_flow_style=False, sort_keys=False, allow_unicode=True, width=200)
        except Exception as exc:
            errors.append(f"topics: {exc}")

    if errors:
        return JSONResponse({"status": "partial", "errors": errors})
    return JSONResponse({"status": "ok"})


@app.get("/api/agents/reports/{skill_name}/{date}")
async def api_agents_report(skill_name: str, date: str, request: Request) -> JSONResponse:
    """Return a generated report from 90_reports/.

    Optional query param ``after`` (unix timestamp): only return the report
    if its mtime is strictly greater than the given value.  This prevents the
    frontend from seeing a stale report that existed before "Run Now" fired.
    """
    cfg = _cfg()
    vault = Path(cfg.get("obsidian_vault", "")).expanduser()
    reports_dir_name = cfg.get("agents", {}).get("reports_dir", "90_reports")
    report_path = vault / reports_dir_name / skill_name / f"{date}.md"
    if not report_path.is_file():
        return JSONResponse({"error": "Report not found"}, status_code=404)
    after = request.query_params.get("after")
    if after:
        try:
            if report_path.stat().st_mtime <= float(after):
                return JSONResponse({"error": "Report not yet refreshed"}, status_code=404)
        except (ValueError, OSError):
            pass
    return JSONResponse({"content": report_path.read_text(encoding="utf-8", errors="replace")})


# ── Plan My Week: estimated-vs-actual enrichment ──

_CALENDAR_MEETING_RE = re.compile(
    r"^##\s+(\d{1,2}:\d{2})\s*-\s*(\d{1,2}:\d{2}):\s*(.+)",
    re.MULTILINE,
)


def _parse_calendar_meetings(calendar_text: str) -> list[dict[str, Any]]:
    """Extract meetings with start, end, title, and duration from calendar.md."""
    meetings: list[dict[str, Any]] = []
    for m in _CALENDAR_MEETING_RE.finditer(calendar_text):
        start_s, end_s, title = m.group(1), m.group(2), m.group(3).strip()
        try:
            sh, sm = map(int, start_s.split(":"))
            eh, em = map(int, end_s.split(":"))
            dur_h = (eh * 60 + em - sh * 60 - sm) / 60.0
        except ValueError:
            dur_h = 0.0
        if dur_h < 0:
            dur_h = 0.0
        meetings.append({"start": start_s, "end": end_s, "title": title, "hours": round(dur_h, 2)})
    return meetings


def _parse_audio_attendance(audio_text: str) -> list[str]:
    """Extract meeting titles from audio.md transcript section headers."""
    titles: list[str] = []
    for m in _CALENDAR_MEETING_RE.finditer(audio_text):
        titles.append(m.group(3).strip())
    return titles


def _enrich_plan_with_actuals(data: dict[str, Any], vault: Path, cfg: dict[str, Any]) -> None:
    """Overlay actual vault data onto the baseline plan-my-week JSON (in-place)."""
    today = datetime.now()
    today_str = today.strftime("%Y/%m/%d")
    meetings_base = cfg.get("output", {}).get("meetings", "10_meetings")
    email_base = cfg.get("output", {}).get("email", "00_inbox")

    total_actual_mtg = 0.0
    total_actual_focus = 0.0
    days_with_actuals = 0

    for day in data.get("days", []):
        date_str = day.get("date", "")
        if not date_str:
            continue

        if date_str > today_str:
            day["day_status"] = "future"
            continue

        is_today = date_str == today_str
        day["day_status"] = "in_progress" if is_today else "completed"

        cal_path = vault / meetings_base / date_str / "calendar.md"
        audio_path = vault / meetings_base / date_str / "audio.md"
        email_dir = vault / email_base / date_str

        actual_meetings: list[dict[str, Any]] = []
        if cal_path.is_file():
            try:
                actual_meetings = _parse_calendar_meetings(cal_path.read_text(encoding="utf-8", errors="replace"))
            except OSError:
                pass

        if is_today:
            now_minutes = today.hour * 60 + today.minute
            actual_meetings = [m for m in actual_meetings if _time_to_min(m["start"]) < now_minutes]

        attended: list[str] = []
        if audio_path.is_file():
            try:
                attended = _parse_audio_attendance(audio_path.read_text(encoding="utf-8", errors="replace"))
            except OSError:
                pass

        actual_mtg_hours = round(sum(m["hours"] for m in actual_meetings), 1)
        actual_mtg_count = len(actual_meetings)
        actual_mtg_names = [m["title"] for m in actual_meetings]
        actual_attended = [t for t in attended if any(
            _fuzzy_title_match(t, m["title"]) for m in actual_meetings
        )] if actual_meetings else attended
        actual_focus = round(max(0.0, 8.0 - actual_mtg_hours), 1)

        email_count = 0
        if email_dir.is_dir():
            try:
                email_count = sum(1 for _ in email_dir.rglob("*.md"))
            except OSError:
                pass

        day["actual_meeting_hours"] = actual_mtg_hours
        day["actual_meeting_count"] = actual_mtg_count
        day["actual_meeting_names"] = actual_mtg_names
        day["actual_meetings_attended"] = actual_attended
        day["actual_focus_hours"] = actual_focus
        day["actual_email_count"] = email_count

        total_actual_mtg += actual_mtg_hours
        total_actual_focus += actual_focus
        days_with_actuals += 1

    data["actual_total_meeting_hours"] = round(total_actual_mtg, 1)
    data["actual_total_focus_hours"] = round(total_actual_focus, 1)
    data["days_with_actuals"] = days_with_actuals

    week_days = data.get("days", [])
    total_days = len(week_days)
    completed_days = sum(1 for d in week_days if d.get("day_status") == "completed")
    in_progress = 1 if any(d.get("day_status") == "in_progress" for d in week_days) else 0
    data["week_progress"] = {
        "completed_days": completed_days,
        "current_day": completed_days + in_progress,
        "total_days": total_days,
        "percent": round((completed_days + in_progress * 0.5) / max(total_days, 1) * 100),
    }


def _time_to_min(t: str) -> int:
    """Convert 'HH:MM' to minutes since midnight."""
    try:
        h, m = map(int, t.split(":"))
        return h * 60 + m
    except (ValueError, AttributeError):
        return 0


def _fuzzy_title_match(audio_title: str, cal_title: str) -> bool:
    """Check if two meeting titles refer to the same meeting (case-insensitive substring)."""
    a = audio_title.lower().strip()
    b = cal_title.lower().strip()
    return a in b or b in a or a == b


def _enrich_weekly_status_with_actuals(data: dict[str, Any], vault: Path, cfg: dict[str, Any]) -> None:
    """Patch weekly-status daily_meetings with actual hours from vault calendar files."""
    meetings_base = cfg.get("output", {}).get("meetings", "10_meetings")

    for entry in data.get("daily_meetings", []):
        date_str = entry.get("date", "")
        if not date_str:
            continue
        vault_date = date_str.replace("-", "/")
        cal_path = vault / meetings_base / vault_date / "calendar.md"
        if not cal_path.is_file():
            continue
        try:
            actual = _parse_calendar_meetings(cal_path.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            continue
        entry["hours"] = round(sum(m["hours"] for m in actual), 1)
        entry["actual_count"] = len(actual)


@app.get("/api/agents/reports/{skill_name}/{date}/json")
async def api_agents_report_json(skill_name: str, date: str, request: Request) -> JSONResponse:
    """Return structured JSON report for rich dashboard rendering."""
    cfg = _cfg()
    vault = Path(cfg.get("obsidian_vault", "")).expanduser()
    reports_dir_name = cfg.get("agents", {}).get("reports_dir", "90_reports")
    json_path = vault / reports_dir_name / skill_name / f"{date}.json"
    md_path = vault / reports_dir_name / skill_name / f"{date}.md"

    source_path: Path | None = json_path if json_path.is_file() else md_path if md_path.is_file() else None
    if source_path is None:
        return JSONResponse({"error": "JSON report not found"}, status_code=404)
    after = request.query_params.get("after")
    if after:
        try:
            if source_path.stat().st_mtime <= float(after):
                return JSONResponse({"error": "Report not yet refreshed"}, status_code=404)
        except (ValueError, OSError):
            pass
    try:
        data: dict[str, Any] | None = None
        if source_path == json_path:
            data = json.loads(json_path.read_text(encoding="utf-8", errors="replace"))
        else:
            md = md_path.read_text(encoding="utf-8", errors="replace")
            embedded = extract_embedded_json(md)
            if embedded:
                data = normalize_report(skill_name, embedded, markdown=md)
            elif skill_name == "project-brief":
                data = parse_project_brief_markdown(md)
        if not data:
            return JSONResponse({"error": "JSON report not found"}, status_code=404)
        if skill_name == "plan-my-week":
            _enrich_plan_with_actuals(data, vault, cfg)
        elif skill_name == "weekly-status":
            _enrich_weekly_status_with_actuals(data, vault, cfg)
        return JSONResponse(data)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.get("/api/agents/images/{skill_name}/{date_dir}/{filename}")
async def api_agents_image(skill_name: str, date_dir: str, filename: str) -> FileResponse:
    """Serve generated images from the reports directory."""
    cfg = _cfg()
    vault = Path(cfg.get("obsidian_vault", "")).expanduser()
    reports_dir_name = cfg.get("agents", {}).get("reports_dir", "90_reports")
    image_path = vault / reports_dir_name / skill_name / "images" / date_dir / filename
    if not image_path.is_file():
        return JSONResponse({"error": "Image not found"}, status_code=404)
    media_type = "image/png"
    if filename.endswith(".jpg") or filename.endswith(".jpeg"):
        media_type = "image/jpeg"
    elif filename.endswith(".webp"):
        media_type = "image/webp"
    return FileResponse(image_path, media_type=media_type)


# ══════════════════════════════════════════════════════════════════════════════
#  API: Knowledge Files (Work Memory)
# ══════════════════════════════════════════════════════════════════════════════

KNOWLEDGE_DIR = "50_knowledge"
KNOWLEDGE_CATEGORIES = {
    "00_profile": "Profile",
    "01_programs": "Programs",
    "02_preferences": "Preferences",
    "03_patterns": "Patterns",
    "90_templates": "Templates",
}


@app.get("/api/knowledge/files")
async def api_knowledge_files() -> JSONResponse:
    """List all knowledge files with parsed frontmatter metadata."""
    cfg = _cfg()
    vault = Path(cfg.get("obsidian_vault", "")).expanduser()
    kdir = vault / KNOWLEDGE_DIR
    pinned_list = cfg.get("memory", {}).get("pinned_files", [])
    pinned_set = set(pinned_list)

    files: list[dict[str, Any]] = []
    if not kdir.is_dir():
        return JSONResponse(files)

    for cat_dir in sorted(kdir.iterdir()):
        if not cat_dir.is_dir():
            continue
        cat_key = cat_dir.name
        cat_label = KNOWLEDGE_CATEGORIES.get(cat_key, cat_key)
        for md_file in sorted(cat_dir.glob("*.md")):
            rel_path = f"{KNOWLEDGE_DIR}/{cat_key}/{md_file.name}"
            fm: dict[str, Any] = {}
            try:
                raw = md_file.read_text(encoding="utf-8", errors="replace")
                if raw.startswith("---"):
                    end = raw.find("\n---", 3)
                    if end != -1:
                        fm = yaml.safe_load(raw[3:end]) or {}
            except Exception:
                pass

            mtime = datetime.fromtimestamp(md_file.stat().st_mtime)
            age_days = (datetime.now() - mtime).days

            updated = fm.get("updated_at", mtime.strftime("%Y-%m-%d"))
            if hasattr(updated, "isoformat"):
                updated = updated.isoformat()

            files.append({
                "path": rel_path,
                "name": md_file.stem.replace("_", " ").replace("-", " "),
                "filename": md_file.name,
                "category": cat_label,
                "category_key": cat_key,
                "type": fm.get("type", ""),
                "title": fm.get("title", md_file.stem),
                "pinned": rel_path in pinned_set,
                "updated_at": str(updated),
                "age_days": age_days,
                "tags": fm.get("tags", []),
            })

    return JSONResponse(files)


@app.get("/api/knowledge/file")
async def api_knowledge_file_get(path: str = "") -> JSONResponse:
    """Read a single knowledge file's content."""
    cfg = _cfg()
    vault = Path(cfg.get("obsidian_vault", "")).expanduser()
    full = vault / path
    if not full.is_file() or KNOWLEDGE_DIR not in path:
        return JSONResponse({"error": "File not found"}, status_code=404)
    try:
        content = full.read_text(encoding="utf-8", errors="replace")
        return JSONResponse({"path": path, "content": content})
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.post("/api/knowledge/file")
async def api_knowledge_file_post(request: Request) -> JSONResponse:
    """Save edits to a knowledge file."""
    body = await request.json()
    path = body.get("path", "")
    content = body.get("content", "")
    cfg = _cfg()
    vault = Path(cfg.get("obsidian_vault", "")).expanduser()
    full = vault / path
    if KNOWLEDGE_DIR not in path:
        return JSONResponse({"error": "Invalid path"}, status_code=400)
    try:
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(content, encoding="utf-8")
        return JSONResponse({"status": "ok"})
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.post("/api/knowledge/new")
async def api_knowledge_new(request: Request) -> JSONResponse:
    """Create a new knowledge file from a template."""
    body = await request.json()
    template = body.get("template", "")
    filename = body.get("filename", "")
    category = body.get("category", "01_programs")
    cfg = _cfg()
    vault = Path(cfg.get("obsidian_vault", "")).expanduser()

    tmpl_path = vault / KNOWLEDGE_DIR / "90_templates" / template
    if not tmpl_path.is_file():
        return JSONResponse({"error": f"Template not found: {template}"}, status_code=404)

    if not filename.endswith(".md"):
        filename += ".md"

    dest = vault / KNOWLEDGE_DIR / category / filename
    if dest.is_file():
        return JSONResponse({"error": "File already exists"}, status_code=409)

    try:
        content = tmpl_path.read_text(encoding="utf-8")
        content = content.replace("TEMPLATE — ", "").replace("template.", f"mem.{category.split('_',1)[-1]}.")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding="utf-8")
        rel = f"{KNOWLEDGE_DIR}/{category}/{filename}"
        return JSONResponse({"status": "ok", "path": rel})
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.post("/api/knowledge/pin")
async def api_knowledge_pin(request: Request) -> JSONResponse:
    """Toggle pin status for a knowledge file in config.yaml."""
    body = await request.json()
    path = body.get("path", "")
    pinned = body.get("pinned", True)

    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        mem = raw.setdefault("memory", {})
        pins = mem.setdefault("pinned_files", [])

        if pinned and path not in pins:
            pins.append(path)
        elif not pinned and path in pins:
            pins.remove(path)

        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            yaml.dump(raw, f, default_flow_style=False, sort_keys=False, allow_unicode=True, width=200)
        return JSONResponse({"status": "ok", "pinned": pinned})
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


# ══════════════════════════════════════════════════════════════════════════════
#  API: Context Files (priorities.md, tasks.md)
# ══════════════════════════════════════════════════════════════════════════════

CONTEXT_DIR = "_context"
CONTEXT_EDITABLE = {"priorities.md", "tasks.md"}


@app.get("/api/context/file")
async def api_context_file_get(name: str = "") -> JSONResponse:
    """Read an editable context file (priorities.md or tasks.md)."""
    if name not in CONTEXT_EDITABLE:
        return JSONResponse({"error": f"Not editable: {name}"}, status_code=400)
    cfg = _cfg()
    vault = Path(cfg.get("obsidian_vault", "")).expanduser()
    full = (vault / CONTEXT_DIR / name).resolve()
    if not str(full).startswith(str(vault.resolve())):
        return JSONResponse({"error": "Invalid path"}, status_code=400)
    if not full.is_file():
        return JSONResponse({"name": name, "content": "", "exists": False})
    try:
        content = full.read_text(encoding="utf-8", errors="replace")
        stat = full.stat()
        return JSONResponse({
            "name": name,
            "content": content,
            "exists": True,
            "modified": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        })
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.post("/api/context/file")
async def api_context_file_post(request: Request) -> JSONResponse:
    """Save edits to an editable context file."""
    body = await request.json()
    name = body.get("name", "")
    content = body.get("content", "")
    if name not in CONTEXT_EDITABLE:
        return JSONResponse({"error": f"Not editable: {name}"}, status_code=400)
    cfg = _cfg()
    vault = Path(cfg.get("obsidian_vault", "")).expanduser()
    full = (vault / CONTEXT_DIR / name).resolve()
    if not str(full).startswith(str(vault.resolve())):
        return JSONResponse({"error": "Invalid path"}, status_code=400)
    try:
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(content, encoding="utf-8")
        return JSONResponse({"status": "ok"})
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


# ══════════════════════════════════════════════════════════════════════════════
#  API: Structured Task Table (parses tasks.md + priorities.md <-> JSON)
# ══════════════════════════════════════════════════════════════════════════════

_TASK_RE = re.compile(
    r"^- \[(?P<check>[ /xX])\]\s+"
    r"(?P<desc>.+?)"
    r"(?:\s+--\s+\*\*(?P<pri>P[012])\*\*)?"
    r"(?:\s+--\s+(?:due|completed):\s*(?P<date>[A-Za-z0-9 ,]+))?"
    r"(?:\s+//\s*(?P<notes>.+))?$"
)
_SUBTASK_RE = re.compile(
    r"^  - \[(?P<check>[ /xX])\]\s+"
    r"(?P<desc>.+?)"
    r"(?:\s+--\s+\*\*(?P<pri>P[012])\*\*)?"
    r"(?:\s+--\s+(?:due|completed):\s*(?P<date>[A-Za-z0-9 ,]+))?"
    r"(?:\s+//\s*(?P<notes>.+))?$"
)

_STATUS_MAP = {" ": "not_started", "/": "in_progress", "x": "complete", "X": "complete"}
_STATUS_TO_CHECK = {"not_started": " ", "in_progress": "/", "complete": "x",
                    "waiting": " ", "blocked": " "}


def _parse_tasks_md(content: str) -> dict:
    """Parse tasks.md into structured data."""
    projects: list[dict] = []
    waiting: list[dict] = []
    completed: list[dict] = []
    backlog: list[dict] = []

    section = "active"
    current_project = ""
    current_tasks: list[dict] = []
    task_idx = 0
    in_comment = False

    for line in content.split("\n"):
        if "<!--" in line:
            in_comment = True
        if "-->" in line:
            in_comment = False
            continue
        if in_comment:
            continue
        stripped = line.strip()
        if stripped.startswith("## Active Tasks"):
            section = "active"
            continue
        if stripped.startswith("## Waiting On Others"):
            if current_project and current_tasks:
                projects.append({"name": current_project, "tasks": current_tasks})
                current_tasks = []
                current_project = ""
            section = "waiting"
            continue
        if stripped.startswith("## Completed"):
            if current_project and current_tasks:
                projects.append({"name": current_project, "tasks": current_tasks})
                current_tasks = []
                current_project = ""
            section = "completed"
            continue
        if stripped.startswith("## Backlog"):
            if current_project and current_tasks:
                projects.append({"name": current_project, "tasks": current_tasks})
                current_tasks = []
                current_project = ""
            section = "backlog"
            continue

        if section == "active" and stripped.startswith("### "):
            if current_project and current_tasks:
                projects.append({"name": current_project, "tasks": current_tasks})
                current_tasks = []
            current_project = stripped[4:].strip()
            continue

        sm = _SUBTASK_RE.match(line)
        if sm:
            task_idx += 1
            sub: dict[str, Any] = {
                "id": f"t{task_idx}",
                "task": sm.group("desc").strip(),
                "status": _STATUS_MAP.get(sm.group("check"), "not_started"),
                "priority": sm.group("pri") or "P1",
                "due": sm.group("date") or "",
                "notes": sm.group("notes") or "",
                "section": section,
            }
            parent = current_tasks[-1] if current_tasks else None
            if parent:
                parent.setdefault("subtasks", []).append(sub)
            continue

        m = _TASK_RE.match(stripped)
        if m:
            task_idx += 1
            task: dict[str, Any] = {
                "id": f"t{task_idx}",
                "task": m.group("desc").strip(),
                "status": _STATUS_MAP.get(m.group("check"), "not_started"),
                "priority": m.group("pri") or "P1",
                "due": m.group("date") or "",
                "notes": m.group("notes") or "",
                "section": section,
                "subtasks": [],
            }
            if section == "active":
                current_tasks.append(task)
            elif section == "completed":
                completed.append(task)
            elif section == "backlog":
                backlog.append(task)
            continue

        if section == "waiting" and stripped.startswith("|") and "---" not in stripped:
            cells = [c.strip() for c in stripped.split("|")]
            cells = [c for c in cells if c]
            if len(cells) >= 2 and cells[0] not in ("Who", "") and not cells[0].startswith("<!--"):
                waiting.append({
                    "who": cells[0], "what": cells[1] if len(cells) > 1 else "",
                    "since": cells[2] if len(cells) > 2 else "",
                    "followup": cells[3] if len(cells) > 3 else "",
                })

    if current_project and current_tasks:
        projects.append({"name": current_project, "tasks": current_tasks})

    return {"projects": projects, "waiting": waiting, "completed": completed, "backlog": backlog}


def _parse_priorities_md(content: str) -> list[dict]:
    """Parse priorities.md into a list of priority objects."""
    priorities: list[dict] = []
    current: dict[str, Any] | None = None

    for line in content.split("\n"):
        stripped = line.strip()
        if stripped.startswith("### "):
            if current:
                priorities.append(current)
            name = re.sub(r"^\d+\.\s*", "", stripped[4:]).strip()
            current = {"name": name, "owner": "", "goal": "", "status": "",
                        "deadline": "", "dependencies": ""}
            continue
        if current and stripped.startswith("- **"):
            kv = stripped[2:]
            if kv.startswith("**Owner:**"):
                current["owner"] = kv.split(":**", 1)[1].strip()
            elif kv.startswith("**Goal:**"):
                current["goal"] = kv.split(":**", 1)[1].strip()
            elif kv.startswith("**Status:**"):
                current["status"] = kv.split(":**", 1)[1].strip()
            elif kv.startswith("**Deadline:**"):
                current["deadline"] = kv.split(":**", 1)[1].strip()
            elif kv.startswith("**Dependencies:**"):
                current["dependencies"] = kv.split(":**", 1)[1].strip()

    if current:
        priorities.append(current)
    return priorities


def _serialize_tasks_md(data: dict) -> str:
    """Serialize structured data back to tasks.md markdown."""
    lines = [
        "# Task List", "",
        f"*Last updated: {datetime.now().strftime('%Y-%m-%d')} via dashboard.*", "",
        "<!-- Conventions:",
        "  - [ ] Not started",
        "  - [/] In progress",
        '  - [x] Complete -- add "completed: {date}"',
        '  Progress notes: append "// {note}" to any task',
        "  Priority: **P0** (today/urgent), **P1** (this week), **P2** (later)",
        '  Move blocked items to "Waiting On Others" with who/since/follow-up',
        "  Examples:",
        "  - [ ] New task -- **P1** -- due: Mar 5",
        "  - [/] Started task -- **P1** -- due: Mar 5 // met with team, drafting doc",
        "  - [x] Done task -- **P1** -- completed: Feb 25",
        "-->", "",
        "## Active Tasks", "",
    ]

    for proj in data.get("projects", []):
        lines.append(f"### {proj['name']}")
        for t in proj.get("tasks", []):
            lines.append(_task_to_line(t))
            for st in t.get("subtasks", []):
                lines.append(_task_to_line(st, indent=1))
        lines.append("")

    lines.append("## Waiting On Others")
    lines.append("")
    lines.append("| Who | What | Since | Follow-up |")
    lines.append("|-----|------|-------|-----------|")
    for w in data.get("waiting", []):
        lines.append(f"| {w.get('who','')} | {w.get('what','')} | {w.get('since','')} | {w.get('followup','')} |")
    if not data.get("waiting"):
        lines.append("| <!-- e.g. Eva | Updated data model | Feb 18 | Ping Thursday --> |")
    lines.append("")

    lines.append("## Completed (last 7 days)")
    lines.append("")
    if data.get("completed"):
        for t in data["completed"]:
            lines.append(_task_to_line(t))
    else:
        lines.append("- *(none yet)*")
    lines.append("")

    lines.append("## Backlog")
    lines.append("")
    if data.get("backlog"):
        for t in data["backlog"]:
            lines.append(_task_to_line(t))
    else:
        lines.append("- *(add lower-priority or future tasks here)*")
    lines.append("")

    return "\n".join(lines)


def _task_to_line(t: dict, indent: int = 0) -> str:
    check = _STATUS_TO_CHECK.get(t.get("status", "not_started"), " ")
    pri = t.get("priority", "P1")
    due = t.get("due", "")
    notes = t.get("notes", "")
    desc = t.get("task", "")
    parts = [f"- [{check}] {desc}"]
    if pri:
        parts.append(f"**{pri}**")
    if due:
        prefix = "completed" if t.get("status") == "complete" else "due"
        parts.append(f"{prefix}: {due}")
    line = " -- ".join(parts)
    if notes:
        line += f" // {notes}"
    return ("  " * indent) + line


@app.get("/api/tasks")
async def api_tasks_get() -> JSONResponse:
    """Return structured task data parsed from tasks.md + priorities.md."""
    cfg = _cfg()
    vault = Path(cfg.get("obsidian_vault", "")).expanduser()
    tasks_path = vault / CONTEXT_DIR / "tasks.md"
    pri_path = vault / CONTEXT_DIR / "priorities.md"

    tasks_content = ""
    if tasks_path.is_file():
        tasks_content = tasks_path.read_text(encoding="utf-8", errors="replace")
    pri_content = ""
    if pri_path.is_file():
        pri_content = pri_path.read_text(encoding="utf-8", errors="replace")

    task_data = _parse_tasks_md(tasks_content)
    priorities = _parse_priorities_md(pri_content)

    pri_map: dict[str, dict] = {}
    for p in priorities:
        pri_map[p["name"]] = p
        base = re.sub(r"\s*\(.*?\)\s*$", "", p["name"])
        if base != p["name"]:
            pri_map[base] = p

    for proj in task_data["projects"]:
        pri = pri_map.get(proj["name"], {})
        if not pri:
            base = re.sub(r"\s*\(.*?\)\s*$", "", proj["name"])
            pri = pri_map.get(base, {})
        proj["owner"] = pri.get("owner", "")
        proj["goal"] = pri.get("goal", "")
        proj["priority_status"] = pri.get("status", "")
        for t in proj.get("tasks", []):
            if not t.get("owner"):
                t["owner"] = pri.get("owner", "")

    return JSONResponse({
        **task_data,
        "priorities": priorities,
        "project_names": [p["name"] for p in priorities],
    })


@app.post("/api/tasks")
async def api_tasks_post(request: Request) -> JSONResponse:
    """Save structured task data back to tasks.md (and update priorities.md status)."""
    body = await request.json()
    cfg = _cfg()
    vault = Path(cfg.get("obsidian_vault", "")).expanduser()
    tasks_path = (vault / CONTEXT_DIR / "tasks.md").resolve()
    pri_path = (vault / CONTEXT_DIR / "priorities.md").resolve()
    vault_resolved = vault.resolve()

    if not str(tasks_path).startswith(str(vault_resolved)):
        return JSONResponse({"error": "Invalid path"}, status_code=400)

    try:
        md = _serialize_tasks_md(body)
        tasks_path.parent.mkdir(parents=True, exist_ok=True)
        tasks_path.write_text(md, encoding="utf-8")

        if pri_path.is_file() and str(pri_path).startswith(str(vault_resolved)):
            pri_content = pri_path.read_text(encoding="utf-8", errors="replace")
            for proj in body.get("projects", []):
                tasks = proj.get("tasks", [])
                if not tasks:
                    continue
                all_done = all(t.get("status") == "complete" for t in tasks)
                any_progress = any(t.get("status") in ("in_progress", "complete") for t in tasks)
                if all_done:
                    new_status = "Complete"
                elif any_progress:
                    new_status = "In progress"
                else:
                    new_status = None
                if new_status:
                    pattern = re.compile(
                        r"(###\s+\d*\.?\s*" + re.escape(proj["name"]) + r".*?\n(?:.*?\n)*?- \*\*Status:\*\*)\s+.*",
                        re.MULTILINE,
                    )
                    pri_content = pattern.sub(r"\1 " + new_status, pri_content, count=1)
            pri_path.write_text(pri_content, encoding="utf-8")

        return JSONResponse({"status": "ok"})
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


# ══════════════════════════════════════════════════════════════════════════════
#  API: Approvals Queue (manual actions)
# ══════════════════════════════════════════════════════════════════════════════

APPROVALS_STATE_FILE = REPO_DIR / "config" / "approvals_state.json"


def _load_approvals_state() -> dict[str, Any]:
    if APPROVALS_STATE_FILE.is_file():
        try:
            return json.loads(APPROVALS_STATE_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {"items": {}, "last_email_scan": None, "last_evidence_scan_frame_id": 0}


def _save_approvals_state(state: dict[str, Any]) -> None:
    APPROVALS_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = APPROVALS_STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")
    tmp.rename(APPROVALS_STATE_FILE)


@app.get("/api/approvals/state")
async def api_approvals_state() -> JSONResponse:
    """Return current approvals queue state."""
    return JSONResponse(_load_approvals_state())


@app.post("/api/approvals/action")
async def api_approvals_action(request: Request, background_tasks: BackgroundTasks) -> JSONResponse:
    """Manually mark an approval as approved or dismissed."""
    body = await request.json()
    approval_id = body.get("approval_id", "")
    action = body.get("action", "")

    if action not in ("approved", "dismissed"):
        return JSONResponse({"error": "action must be 'approved' or 'dismissed'"}, status_code=400)

    state = _load_approvals_state()
    item = state.get("items", {}).get(approval_id)
    if not item:
        return JSONResponse({"error": f"Item not found: {approval_id}"}, status_code=404)

    item["status"] = action
    item["status_changed_at"] = datetime.now().isoformat(timespec="seconds")
    if action == "approved" and not item.get("evidence"):
        item["evidence"] = {"timestamp": item["status_changed_at"], "url": "", "snippet": "Manual approval via dashboard"}
    state["items"][approval_id] = item
    _save_approvals_state(state)

    def _regen_report() -> None:
        try:
            from src.monitor.approvals_monitor import generate_report, load_approvals_state as _load
            cfg = _cfg()
            vault = Path(cfg.get("obsidian_vault", "")).expanduser()
            fresh = _load()
            generate_report(fresh, vault, cfg)
        except Exception as exc:
            logger.error("Approvals report regen failed: %s", exc)

    background_tasks.add_task(_regen_report)
    return JSONResponse({"status": "ok", "approval_id": approval_id, "new_status": action})


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

/* ── Audio device picker ── */
.audio-device-picker {
  display: flex; align-items: center; gap: 6px;
}
.audio-device-picker select {
  padding: 6px 10px; border-radius: 8px; font-size: .83rem;
  border: 1px solid var(--border); background: var(--bg); color: var(--text);
  cursor: pointer; max-width: 220px;
}
.audio-device-picker select:disabled { opacity: .5; cursor: default; }
.btn-icon {
  background: none; border: 1px solid var(--border); border-radius: 6px;
  color: var(--muted); cursor: pointer; font-size: 1rem; padding: 4px 7px;
  line-height: 1; transition: color .2s;
}
.btn-icon:hover { color: var(--text); }
.audio-device-msg {
  font-size: .78rem; font-weight: 600; transition: opacity .3s;
}

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
@media (max-width: 900px) { #mywork-grid { grid-template-columns: 1fr !important; } }

/* ── Task Table ── */
.mw-topbar { display:flex; gap:10px; align-items:center; flex-wrap:wrap; margin-bottom:16px; }
.mw-toggle { display:inline-flex; border:1px solid var(--border); border-radius:8px; overflow:hidden; }
.mw-toggle button { padding:6px 14px; font-size:.8rem; font-weight:500; border:none; background:transparent; color:var(--muted); cursor:pointer; }
.mw-toggle button.active { background:var(--accent); color:#fff; }
.mw-filters { display:flex; gap:8px; align-items:center; margin-left:auto; }
.mw-filters select { padding:4px 8px; font-size:.78rem; border-radius:6px; border:1px solid var(--border); background:var(--bg); color:var(--text); }
.task-proj-hdr { display:flex; align-items:center; gap:10px; padding:10px 14px; background:var(--surface2); border-bottom:1px solid var(--border); cursor:pointer; user-select:none; }
.task-proj-hdr .proj-name { font-size:.82rem; font-weight:600; text-transform:uppercase; letter-spacing:.03em; color:var(--muted); }
.task-proj-hdr .proj-meta { font-size:.74rem; color:var(--muted); margin-left:auto; }
.task-proj-hdr .proj-toggle { font-size:.7rem; color:var(--muted); transition:transform .15s; }
.task-proj-hdr .proj-toggle.collapsed { transform:rotate(-90deg); }
.task-proj-hdr .proj-edit-btn { background:none; border:none; color:var(--muted); cursor:pointer; font-size:.72rem; padding:2px 4px; border-radius:4px; opacity:.4; }
.task-proj-hdr .proj-edit-btn:hover { opacity:1; color:var(--accent); }
.task-proj-hdr .proj-owner-lbl { cursor:pointer; border-bottom:1px dashed transparent; }
.task-proj-hdr .proj-owner-lbl:hover { border-bottom-color:var(--accent); color:var(--text); }
.task-row { display:grid; grid-template-columns:1fr 110px 70px 110px 105px 1fr 36px; gap:0; padding:6px 10px; border-bottom:1px solid var(--border); align-items:center; font-size:.82rem; }
.task-row input, .task-row select { padding:4px 6px; font-size:.78rem; border-radius:5px; border:1px solid transparent; background:transparent; color:var(--text); width:100%; }
.task-row input:hover, .task-row select:hover { border-color:var(--border); }
.task-row input:focus, .task-row select:focus { border-color:var(--accent); background:var(--bg); outline:none; }
.task-row .task-del { background:none; border:none; color:var(--muted); cursor:pointer; font-size:.9rem; padding:2px 6px; border-radius:4px; }
.task-row .task-del:hover { color:var(--red); background:rgba(248,113,113,.1); }
.task-hdr { display:grid; grid-template-columns:1fr 110px 70px 110px 105px 1fr 36px; gap:0; padding:6px 10px; font-size:.72rem; font-weight:600; color:var(--muted); text-transform:uppercase; letter-spacing:.04em; border-bottom:2px solid var(--border); }
.task-bottom { display:flex; gap:12px; align-items:center; margin-top:16px; }
.task-counts { font-size:.8rem; color:var(--muted); margin-left:auto; }
.task-counts span { margin-right:12px; }
select.pri-p0 { color:var(--red); } select.pri-p1 { color:var(--yellow); } select.pri-p2 { color:var(--muted); }
select.st-not_started { color:var(--muted); } select.st-in_progress { color:var(--blue); } select.st-complete { color:var(--green); } select.st-waiting { color:var(--orange); } select.st-blocked { color:var(--red); }
.task-row input.overdue { color:var(--red); font-weight:600; }
.task-row.subtask { padding-left:36px; background:rgba(255,255,255,.015); border-left:2px solid var(--accent); }
.task-row .task-actions { display:flex; gap:2px; align-items:center; }
.task-row .task-add-sub { background:none; border:none; color:var(--muted); cursor:pointer; font-size:.72rem; padding:2px 5px; border-radius:4px; opacity:.5; }
.task-row .task-add-sub:hover { color:var(--accent); background:rgba(99,102,241,.1); opacity:1; }
@media (max-width: 900px) {
  .task-row, .task-hdr { grid-template-columns: 1fr !important; gap:4px; }
  .task-hdr { display:none; }
  .task-row { padding:12px; border:1px solid var(--border); border-radius:8px; margin-bottom:8px; }
  .task-row input, .task-row select { border-color:var(--border); background:var(--bg); }
  .task-row > *::before { content:attr(data-label); display:block; font-size:.68rem; color:var(--muted); text-transform:uppercase; margin-bottom:2px; }
  .mw-filters { margin-left:0; width:100%; }
}

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

/* ── Setup wizard ── */
.wiz-step { font-size:.82rem; }
.wiz-step.active { background:var(--accent); color:#fff; border-color:var(--accent); }
.setup-row { display:flex; align-items:center; gap:12px; padding:10px 0; border-bottom:1px solid var(--border); font-size:.87rem; }
.setup-row:last-child { border-bottom:none; }
.setup-icon { width:22px; text-align:center; flex-shrink:0; font-size:1rem; }
.setup-detail { flex:1; }
.setup-fix { font-size:.78rem; color:var(--muted); margin-top:2px; }
.radio-card { display:inline-block; padding:12px 16px; border:1px solid var(--border); border-radius:10px; cursor:pointer; transition:border-color .15s; }
.radio-card:has(input:checked) { border-color:var(--accent); background:rgba(108,124,255,.08); }
.radio-card input { margin-right:6px; }
.chip { display:inline-block; padding:4px 12px; border-radius:999px; font-size:.78rem; background:var(--surface2); border:1px solid var(--border); cursor:pointer; transition:all .15s; }
.chip:hover { border-color:var(--accent); color:var(--accent); }

/* ── Help tab ── */
.tab-pane#pane-help .card { margin-bottom:16px; }
.tab-pane#pane-help code { background:var(--surface2); padding:2px 6px; border-radius:4px; font-size:.82rem; }
.tab-pane#pane-help details summary { padding:6px 0; }
.tab-pane#pane-help details[open] summary { margin-bottom:4px; }

/* ── Skills tab ── */
.skills-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(340px, 1fr)); gap: 16px; }
.skill-card {
  background: var(--surface); border: 1px solid var(--border); border-radius: 12px;
  padding: 20px; cursor: pointer; transition: border-color .15s, transform .1s;
}
.skill-card:hover { border-color: var(--accent); transform: translateY(-2px); }
.skill-card-name { font-size: 1.1rem; font-weight: 600; margin-bottom: 6px; color: var(--accent); }
.skill-card-desc { font-size: .82rem; color: var(--muted); line-height: 1.5; margin-bottom: 12px;
  display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden; }
.skill-card-meta { display: flex; gap: 8px; flex-wrap: wrap; }
.skill-tag {
  display: inline-block; padding: 2px 8px; border-radius: 6px; font-size: .7rem; font-weight: 500;
  background: rgba(108,124,255,.12); color: var(--accent);
}
.skill-tag-scripts { background: rgba(74,222,128,.12); color: var(--green); }

/* Skill config editor */
.skill-config-panel {
  display: none; padding: 16px; background: var(--surface2); border-bottom: 1px solid var(--border);
  animation: slideDown .15s ease-out;
}
.skill-config-panel.open { display: block; }
@keyframes slideDown { from { opacity: 0; max-height: 0; } to { opacity: 1; max-height: 2000px; } }
.topic-card {
  background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
  padding: 14px; margin-bottom: 10px; position: relative;
}
.topic-card .topic-del { position: absolute; top: 8px; right: 10px; background: none; border: none;
  color: var(--red); cursor: pointer; font-size: 1.1rem; opacity: .6; }
.topic-card .topic-del:hover { opacity: 1; }
.kw-pill {
  display: inline-flex; align-items: center; gap: 4px; padding: 2px 10px; border-radius: 12px;
  background: rgba(108,124,255,.12); color: var(--accent); font-size: .75rem; margin: 2px;
}
.kw-pill button { background: none; border: none; color: var(--accent); cursor: pointer;
  font-size: .85rem; padding: 0; opacity: .6; }
.kw-pill button:hover { opacity: 1; }
.cfg-input {
  width: 100%; padding: 6px 10px; border-radius: 8px; border: 1px solid var(--border);
  background: var(--bg); color: var(--text); font-size: .83rem;
}
.cfg-label { font-size: .78rem; color: var(--muted); display: block; margin-bottom: 4px; font-weight: 500; }
.cfg-textarea { width: 100%; min-height: 80px; padding: 8px 10px; border-radius: 8px;
  border: 1px solid var(--border); background: var(--bg); color: var(--text);
  font-size: .83rem; font-family: inherit; resize: vertical; }

/* Skills modal */
.skill-modal-backdrop {
  display: none; position: fixed; inset: 0; background: rgba(0,0,0,.6);
  z-index: 200; justify-content: center; align-items: center; padding: 24px;
}
.skill-modal-backdrop.open { display: flex; }
.skill-modal {
  background: var(--surface); border: 1px solid var(--border); border-radius: 14px;
  max-width: 900px; width: 100%; max-height: 90vh; display: flex; flex-direction: column;
}
.skill-modal-header {
  display: flex; justify-content: space-between; align-items: center;
  padding: 18px 28px; border-bottom: 1px solid var(--border); flex-shrink: 0;
}
.skill-modal-title { font-size: 1.2rem; font-weight: 600; color: var(--accent); }
.skill-modal-close { background: none; border: none; color: var(--muted); font-size: 1.5rem; cursor: pointer; padding: 0 4px; }
.skill-modal-close:hover { color: var(--text); }
.skill-modal-body {
  padding: 28px 32px; overflow-y: auto; font-size: .9rem; line-height: 1.75; color: var(--text);
}
.skill-modal-body h1 { font-size: 1.4rem; margin: 4px 0 12px; color: var(--accent); font-weight: 700; }
.skill-modal-body h2 { font-size: 1.1rem; margin: 28px 0 10px; color: var(--accent);
  border-bottom: 1px solid var(--border); padding-bottom: 6px; font-weight: 600; }
.skill-modal-body h3 { font-size: .95rem; margin: 18px 0 6px; color: var(--blue); font-weight: 600; }
.skill-modal-body h4 { font-size: .88rem; margin: 14px 0 4px; color: var(--text); font-weight: 600; }
.skill-modal-body p { margin: 8px 0; }
.skill-modal-body > .report-subtitle { color: var(--muted); font-style: italic; margin: -8px 0 16px; font-size: .86rem; }
.skill-modal-body ul { margin: 8px 0 8px 0; padding-left: 20px; list-style: disc; }
.skill-modal-body ol { margin: 8px 0 8px 0; padding-left: 20px; }
.skill-modal-body li { margin: 6px 0; line-height: 1.7; }
.skill-modal-body li li { margin: 3px 0; }
.skill-modal-body code {
  background: rgba(108,124,255,.08); padding: 2px 6px; border-radius: 4px; font-size: .82rem; color: var(--blue);
  font-family: 'SF Mono','Menlo',monospace;
}
.skill-modal-body pre {
  background: var(--bg); padding: 14px 18px; border-radius: 10px; overflow-x: auto;
  margin: 10px 0; border: 1px solid var(--border);
}
.skill-modal-body pre code { background: none; padding: 0; color: var(--green); font-size: .8rem; }
.skill-modal-body table { width: 100%; border-collapse: collapse; margin: 12px 0; font-size: .84rem;
  border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }
.skill-modal-body thead { background: var(--surface2); }
.skill-modal-body th { text-align: left; padding: 10px 14px; border-bottom: 2px solid var(--border);
  color: var(--accent); font-weight: 600; font-size: .78rem; text-transform: uppercase; letter-spacing: .03em; }
.skill-modal-body td { padding: 10px 14px; border-bottom: 1px solid var(--border); vertical-align: top; }
.skill-modal-body tr:last-child td { border-bottom: none; }
.skill-modal-body tr:hover td { background: rgba(108,124,255,.04); }
.skill-modal-body strong { color: #fff; }
.skill-modal-body em { color: var(--muted); }
.skill-modal-body a { color: var(--accent); text-decoration: none; }
.skill-modal-body a:hover { text-decoration: underline; }
.skill-modal-body blockquote { border-left: 3px solid var(--accent); padding: 8px 16px; margin: 12px 0;
  background: rgba(108,124,255,.04); border-radius: 0 8px 8px 0; color: var(--muted); }
.skill-modal-body hr { border: none; border-top: 1px solid var(--border); margin: 24px 0; }

/* ── Rich Skill Views ── */
.rich-view-overlay {
  display: none; position: fixed; inset: 0; z-index: 300;
  background: var(--bg); overflow-y: auto;
}
.rich-view-overlay.open { display: block; }
.rich-view-header {
  position: sticky; top: 0; z-index: 10;
  display: flex; align-items: center; gap: 16px;
  padding: 14px 28px; background: var(--surface);
  border-bottom: 1px solid var(--border);
}
.rich-view-back {
  background: none; border: 1px solid var(--border); color: var(--accent);
  padding: 6px 14px; border-radius: 8px; cursor: pointer; font-size: .84rem;
}
.rich-view-back:hover { background: rgba(108,124,255,.1); }
.rich-view-title { font-size: 1.15rem; font-weight: 600; color: var(--text); }
.rich-view-date { font-size: .84rem; color: var(--muted); margin-left: auto; }
.rich-view-body { padding: 28px; max-width: 1400px; margin: 0 auto; }

/* Metrics bar */
.rv-metrics { display: flex; gap: 16px; margin-bottom: 24px; flex-wrap: wrap; }
.rv-metric {
  flex: 1; min-width: 140px; background: var(--surface); border: 1px solid var(--border);
  border-radius: 12px; padding: 18px 20px; text-align: center;
}
.rv-metric-value { font-size: 2rem; font-weight: 700; color: var(--accent); }
.rv-metric-label { font-size: .78rem; color: var(--muted); margin-top: 4px; }

/* Hero image */
.rv-hero { width: 100%; border-radius: 14px; margin-bottom: 24px; object-fit: contain; max-height: 420px; }

/* Section headers */
.rv-section { margin-bottom: 28px; }
.rv-section-title {
  font-size: 1.1rem; font-weight: 600; color: var(--accent);
  border-bottom: 1px solid var(--border); padding-bottom: 8px; margin-bottom: 14px;
}

/* Cards */
.rv-card {
  background: var(--surface); border: 1px solid var(--border); border-radius: 12px;
  padding: 16px 20px; margin-bottom: 12px; transition: border-color .15s;
}
.rv-card:hover { border-color: var(--accent); }
.rv-card-title { font-weight: 600; font-size: .95rem; margin-bottom: 6px; }
.rv-card-meta { font-size: .78rem; color: var(--muted); }
.rv-card-body { font-size: .88rem; line-height: 1.6; margin-top: 8px; }

/* Status pills */
.rv-pill {
  display: inline-block; padding: 2px 10px; border-radius: 12px;
  font-size: .72rem; font-weight: 600; text-transform: uppercase;
}
.rv-pill-green { background: rgba(74,222,128,.15); color: var(--green); }
.rv-pill-yellow { background: rgba(250,204,21,.15); color: #facc15; }
.rv-pill-red { background: rgba(248,113,113,.15); color: var(--red); }
.rv-pill-blue { background: rgba(108,124,255,.15); color: var(--accent); }
.rv-pill-gray { background: rgba(128,128,128,.15); color: var(--muted); }

/* Severity badges */
.rv-severity-critical { border-left: 4px solid var(--red); }
.rv-severity-warning { border-left: 4px solid #facc15; }
.rv-severity-high { border-left: 4px solid var(--red); }
.rv-severity-medium { border-left: 4px solid #facc15; }
.rv-severity-low { border-left: 4px solid var(--green); }
.rv-severity-info { border-left: 4px solid var(--accent); }

/* Two-column layout */
.rv-cols { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
@media (max-width: 900px) { .rv-cols { grid-template-columns: 1fr; } }

/* Chart containers */
.rv-chart-container { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 20px; margin-bottom: 16px; }
.rv-chart-container canvas { max-height: 250px; }

/* News Pulse -- Single Column Inline Expand */
.np-feed { max-width: 920px; margin: 0 auto; }
.np-section-label {
  font-size: .72rem; font-weight: 700; text-transform: uppercase; letter-spacing: .1em;
  color: var(--accent); padding: 18px 0 6px; display: flex; align-items: center; gap: 8px;
  border-bottom: 2px solid var(--accent); margin-bottom: 12px; margin-top: 24px;
}
.np-section-label.internal { color: var(--green); border-color: var(--green); }
.np-section-label::before { content: ''; width: 10px; height: 10px; border-radius: 50%; background: currentColor; }
.np-editorial {
  font-size: .88rem; color: var(--muted); line-height: 1.6; font-style: italic;
  padding: 0 0 12px; margin-bottom: 4px;
}
.np-card {
  display: flex; gap: 16px; padding: 16px; border-radius: 14px; cursor: pointer;
  margin-bottom: 6px; transition: all .2s ease; border: 1px solid transparent;
  align-items: flex-start;
}
.np-card:hover { background: rgba(108,124,255,.06); border-color: var(--border); }
.np-card.expanded { background: var(--surface); border-color: var(--accent); cursor: default; flex-direction: column; gap: 0; }
.np-card-header { display: flex; gap: 16px; align-items: flex-start; width: 100%; }
.np-card-thumb {
  width: 80px; height: 80px; border-radius: 10px; object-fit: cover; flex-shrink: 0;
  background: var(--surface2);
}
.np-card-thumb.internal { border: 2px solid var(--green); }
.np-card-info { flex: 1; min-width: 0; }
.np-card-title { font-weight: 700; font-size: .92rem; line-height: 1.35; margin-bottom: 4px; }
.np-card-snippet { font-size: .8rem; color: var(--muted); line-height: 1.5;
  display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }
.np-card.expanded .np-card-snippet { display: none; }
.np-card-badges { display: flex; gap: 6px; margin-top: 6px; flex-wrap: wrap; }
.np-card-expand {
  display: none; width: 100%; padding-top: 16px; animation: npSlideIn .25s ease;
}
.np-card.expanded .np-card-expand { display: block; }
@keyframes npSlideIn { from { opacity: 0; transform: translateY(-8px); } to { opacity: 1; transform: translateY(0); } }
.np-card-hero {
  width: 100%; max-height: 280px; border-radius: 12px; margin-bottom: 16px;
  object-fit: contain; background: var(--surface2);
}
.np-card-body { font-size: .88rem; line-height: 1.7; color: var(--text); white-space: pre-line; }
.np-card-so-what {
  background: rgba(108,124,255,.08); border-left: 3px solid var(--accent);
  border-radius: 0 10px 10px 0; padding: 14px 18px; margin: 14px 0;
  font-size: .86rem; line-height: 1.6;
}
.np-card-so-what b { color: var(--accent); }
.np-card-so-what.internal { background: rgba(74,222,128,.08); border-color: var(--green); }
.np-card-so-what.internal b { color: var(--green); }
.np-card-source {
  font-size: .76rem; color: var(--muted); margin-top: 10px;
  display: flex; align-items: center; gap: 8px;
}
.np-card-source a { color: var(--accent); text-decoration: none; }
.np-card-source a:hover { text-decoration: underline; }

/* Day cards for Plan My Week */
.rv-day-strip { display: flex; gap: 10px; margin-bottom: 24px; overflow-x: auto; padding-bottom: 8px; }
.rv-day-card {
  min-width: 120px; flex: 1; background: var(--surface); border: 1px solid var(--border);
  border-radius: 12px; padding: 14px; text-align: center; cursor: pointer; transition: all .15s;
}
.rv-day-card:hover { border-color: var(--accent); transform: translateY(-2px); }
.rv-day-card.active { border-color: var(--accent); background: rgba(108,124,255,.08); }
.rv-day-card-name { font-size: .82rem; font-weight: 600; color: var(--text); }
.rv-day-card-date { font-size: .72rem; color: var(--muted); margin-top: 2px; }
.rv-day-card-meetings { font-size: 1.4rem; font-weight: 700; margin-top: 6px; }
.rv-day-card-hours { font-size: .72rem; color: var(--muted); }
.rv-day-density-light { border-top: 3px solid var(--green); }
.rv-day-density-moderate { border-top: 3px solid #facc15; }
.rv-day-density-heavy { border-top: 3px solid var(--red); }

/* Day status indicators (estimated vs actual) */
.rv-day-status {
  display: inline-flex; align-items: center; justify-content: center;
  width: 18px; height: 18px; border-radius: 50%; margin: 6px auto 2px; font-size: .7rem;
}
.rv-day-status-completed { background: var(--green); color: #000; }
.rv-day-status-completed::after { content: '\2713'; }
.rv-day-status-today {
  background: var(--accent); animation: pulseDay 2s ease-in-out infinite;
}
.rv-day-status-today::after { content: '\25CF'; color: #fff; font-size: .5rem; }
@keyframes pulseDay { 0%,100% { opacity:1; box-shadow: 0 0 0 0 rgba(108,124,255,.4); } 50% { opacity:.8; box-shadow: 0 0 0 6px rgba(108,124,255,0); } }
.rv-day-status-future { background: var(--border); }

.rv-day-actual {
  font-size: .68rem; color: var(--muted); margin-top: 4px;
  border-top: 1px dashed var(--border); padding-top: 4px;
}
.rv-day-actual-val { font-weight: 600; }
.rv-day-actual-good { color: var(--green); }
.rv-day-actual-bad { color: var(--red); }

/* Week progress bar */
.rv-week-progress {
  background: var(--surface); border: 1px solid var(--border); border-radius: 12px;
  padding: 14px 20px; margin-bottom: 20px; display: flex; align-items: center; gap: 16px;
}
.rv-week-progress-label { font-size: .84rem; font-weight: 600; color: var(--text); white-space: nowrap; }
.rv-week-progress-track {
  flex: 1; height: 8px; background: var(--surface2); border-radius: 4px; overflow: hidden; position: relative;
}
.rv-week-progress-fill {
  height: 100%; border-radius: 4px; background: linear-gradient(90deg, var(--accent), var(--green));
  transition: width .5s ease;
}
.rv-week-progress-pct { font-size: .78rem; color: var(--muted); white-space: nowrap; }

/* Estimated vs Actual metrics */
.rv-metric-compare {
  display: flex; gap: 8px; align-items: baseline; justify-content: center;
  margin-top: 4px; font-size: .74rem;
}
.rv-metric-planned { color: var(--muted); }
.rv-metric-actual { font-weight: 600; }
.rv-metric-delta { font-size: .68rem; margin-left: 2px; }
.rv-metric-delta-good { color: var(--green); }
.rv-metric-delta-bad { color: var(--red); }

/* Gantt actual bars */
.rv-gantt-actual-bar {
  height: 60%; border-radius: 4px; position: absolute; bottom: 2px;
  opacity: .7; border: 1px dashed rgba(255,255,255,.3);
}
.rv-gantt-today-marker {
  position: absolute; top: -4px; bottom: -4px; width: 2px;
  background: var(--accent); z-index: 2;
}
.rv-gantt-today-marker::before {
  content: 'Today'; position: absolute; top: -16px; left: 50%;
  transform: translateX(-50%); font-size: .58rem; color: var(--accent);
  white-space: nowrap; font-weight: 600;
}
.rv-gantt-row { position: relative; }
.rv-gantt-legend {
  display: flex; gap: 16px; margin-top: 8px; font-size: .7rem; color: var(--muted);
}
.rv-gantt-legend-dot {
  display: inline-block; width: 10px; height: 10px; border-radius: 3px; margin-right: 4px;
  vertical-align: middle;
}

/* Gantt-like deliverable bars */
.rv-gantt { margin-bottom: 24px; }
.rv-gantt-row { display: flex; align-items: center; gap: 10px; margin-bottom: 6px; font-size: .82rem; }
.rv-gantt-label { width: 200px; flex-shrink: 0; text-align: right; color: var(--muted); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.rv-gantt-track { flex: 1; height: 24px; background: var(--surface2); border-radius: 6px; position: relative; display: flex; }
.rv-gantt-bar { height: 100%; border-radius: 6px; display: flex; align-items: center; justify-content: center;
  font-size: .7rem; font-weight: 600; color: #fff; min-width: 30px; }

/* Meeting timeline for Morning Brief */
.rv-timeline { position: relative; padding-left: 32px; }
.rv-timeline::before { content: ''; position: absolute; left: 12px; top: 0; bottom: 0; width: 2px; background: var(--border); }
.rv-timeline-item { position: relative; margin-bottom: 16px; }
.rv-timeline-dot {
  position: absolute; left: -26px; top: 4px; width: 12px; height: 12px;
  border-radius: 50%; border: 2px solid var(--accent); background: var(--bg);
}
.rv-timeline-dot.high { border-color: var(--red); background: rgba(248,113,113,.2); }
.rv-timeline-dot.medium { border-color: #facc15; background: rgba(250,204,21,.2); }
.rv-timeline-time { font-size: .78rem; color: var(--muted); font-weight: 500; margin-bottom: 4px; }

/* Attendee avatars */
.rv-attendees { display: flex; gap: 6px; margin-top: 8px; flex-wrap: wrap; }
.rv-attendee {
  display: flex; align-items: center; gap: 6px; padding: 4px 10px;
  background: var(--surface2); border-radius: 20px; font-size: .75rem;
}
.rv-attendee-initial {
  width: 22px; height: 22px; border-radius: 50%; display: flex; align-items: center;
  justify-content: center; font-size: .7rem; font-weight: 700; color: #fff;
}
.rv-attendee-ooo { opacity: .5; text-decoration: line-through; }

/* Progress bars */
.rv-progress { height: 8px; background: var(--surface2); border-radius: 4px; overflow: hidden; margin-top: 6px; }
.rv-progress-bar { height: 100%; border-radius: 4px; transition: width .4s ease; }
.rv-progress-bar-green { background: var(--green); }
.rv-progress-bar-yellow { background: #facc15; }
.rv-progress-bar-red { background: var(--red); }
.rv-progress-bar-accent { background: var(--accent); }

/* ═══ Nuclear Overhaul: Score Gauge ═══ */
.rv-gauge-wrap { display:flex; align-items:center; gap:20px; margin-bottom:24px; }
.rv-gauge { position:relative; width:120px; height:120px; flex-shrink:0; }
.rv-gauge canvas { width:100%!important; height:100%!important; }
.rv-gauge-label { position:absolute; top:50%; left:50%; transform:translate(-50%,-50%); text-align:center; }
.rv-gauge-value { font-size:1.8rem; font-weight:800; color:var(--text); line-height:1; }
.rv-gauge-sub { font-size:.68rem; color:var(--muted); margin-top:2px; }
.rv-gauge-sm { width:64px; height:64px; }
.rv-gauge-sm .rv-gauge-value { font-size:1rem; }

/* ═══ Analysis Card ═══ */
.rv-analysis { background:linear-gradient(135deg,rgba(108,124,255,.08),rgba(108,124,255,.02)); border:1px solid rgba(108,124,255,.2); border-radius:14px; padding:20px 24px; margin-bottom:24px; }
.rv-analysis-insight { font-size:1.05rem; font-weight:600; color:var(--text); line-height:1.5; margin-bottom:12px; }
.rv-analysis-row { display:flex; gap:16px; flex-wrap:wrap; }
.rv-analysis-item { flex:1; min-width:180px; }
.rv-analysis-label { font-size:.68rem; text-transform:uppercase; letter-spacing:.06em; color:var(--muted); margin-bottom:4px; }
.rv-analysis-text { font-size:.84rem; color:var(--text); line-height:1.4; }

/* ═══ Kanban Board ═══ */
.rv-kanban { display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:14px; margin-bottom:24px; }
.rv-kanban-col { background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:14px; min-height:120px; }
.rv-kanban-col-title { font-size:.78rem; font-weight:700; text-transform:uppercase; letter-spacing:.06em; color:var(--muted); margin-bottom:12px; display:flex; align-items:center; gap:8px; }
.rv-kanban-col-title .rv-kanban-count { background:var(--surface2); border-radius:10px; padding:1px 8px; font-size:.7rem; }
.rv-kanban-item { background:var(--bg); border:1px solid var(--border); border-radius:8px; padding:10px 12px; margin-bottom:8px; font-size:.82rem; transition:border-color .15s; }
.rv-kanban-item:hover { border-color:var(--accent); }
.rv-kanban-item-title { font-weight:600; margin-bottom:4px; }
.rv-kanban-item-meta { font-size:.72rem; color:var(--muted); }

/* ═══ Heatmap Grid ═══ */
.rv-heatmap { display:grid; gap:3px; margin-bottom:24px; }
.rv-heatmap-cell { border-radius:4px; min-height:32px; display:flex; align-items:center; justify-content:center; font-size:.68rem; font-weight:600; color:rgba(255,255,255,.8); transition:transform .15s; cursor:default; }
.rv-heatmap-cell:hover { transform:scale(1.15); z-index:1; }
.rv-heatmap-label { font-size:.72rem; color:var(--muted); display:flex; align-items:center; justify-content:center; }

/* ═══ Energy Map Bar ═══ */
.rv-energy-map { display:flex; gap:2px; margin-bottom:24px; border-radius:8px; overflow:hidden; height:40px; }
.rv-energy-slot { flex:1; display:flex; align-items:center; justify-content:center; font-size:.6rem; font-weight:600; color:#fff; cursor:default; position:relative; }
.rv-energy-slot[data-tip]:hover::after { content:attr(data-tip); position:absolute; bottom:100%; left:50%; transform:translateX(-50%); background:var(--surface); border:1px solid var(--border); padding:4px 8px; border-radius:6px; font-size:.7rem; color:var(--text); white-space:nowrap; z-index:10; }
.rv-energy-open { background:rgba(74,222,128,.6); }
.rv-energy-meeting { background:rgba(248,113,113,.6); }
.rv-energy-transition { background:rgba(250,204,21,.4); }

/* ═══ Treemap ═══ */
.rv-treemap { display:grid; gap:4px; margin-bottom:24px; }
.rv-treemap-cell { border-radius:8px; padding:10px 12px; display:flex; flex-direction:column; justify-content:center; min-height:60px; transition:transform .15s; }
.rv-treemap-cell:hover { transform:scale(1.03); }
.rv-treemap-name { font-weight:700; font-size:.82rem; color:#fff; }
.rv-treemap-val { font-size:.72rem; color:rgba(255,255,255,.8); }

/* ═══ Network Graph ═══ */
.rv-network { width:100%; height:300px; background:var(--surface); border:1px solid var(--border); border-radius:14px; margin-bottom:24px; overflow:hidden; }
.rv-network svg { width:100%; height:100%; }

/* ═══ Milestone Timeline ═══ */
.rv-milestones { display:flex; align-items:center; gap:0; margin-bottom:24px; padding:16px 0; position:relative; overflow-x:auto; }
.rv-milestone { display:flex; flex-direction:column; align-items:center; flex-shrink:0; position:relative; min-width:100px; }
.rv-milestone-dot { width:16px; height:16px; border-radius:50%; border:3px solid var(--accent); background:var(--bg); z-index:1; }
.rv-milestone-dot.hit { background:var(--green); border-color:var(--green); }
.rv-milestone-dot.missed { background:var(--red); border-color:var(--red); }
.rv-milestone-dot.at-risk { background:#facc15; border-color:#facc15; }
.rv-milestone-name { font-size:.72rem; font-weight:600; color:var(--text); text-align:center; margin-top:8px; max-width:100px; }
.rv-milestone-date { font-size:.66rem; color:var(--muted); }
.rv-milestone-line { flex:1; height:3px; background:var(--border); min-width:20px; }

/* ═══ Team Grid ═══ */
.rv-team-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(150px,1fr)); gap:12px; margin-bottom:24px; }
.rv-team-member { background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:14px; text-align:center; transition:border-color .15s; }
.rv-team-member:hover { border-color:var(--accent); }
.rv-team-avatar { width:48px; height:48px; border-radius:50%; margin:0 auto 8px; display:flex; align-items:center; justify-content:center; font-size:1.1rem; font-weight:700; color:#fff; }
.rv-team-name { font-size:.84rem; font-weight:600; }
.rv-team-role { font-size:.7rem; color:var(--muted); margin-top:2px; }
.rv-team-badge { display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:4px; }

/* ═══ Filter Bar ═══ */
.rv-filter-bar { display:flex; gap:8px; margin-bottom:20px; flex-wrap:wrap; }
.rv-filter-chip { padding:5px 14px; border-radius:20px; font-size:.76rem; font-weight:500; border:1px solid var(--border); background:var(--surface); color:var(--muted); cursor:pointer; transition:all .15s; }
.rv-filter-chip:hover,.rv-filter-chip.active { background:rgba(108,124,255,.15); color:var(--accent); border-color:var(--accent); }

/* ═══ Calendar Grid ═══ */
.rv-cal-grid { display:grid; grid-template-columns:60px repeat(5,1fr); gap:2px; margin-bottom:24px; background:var(--surface); border:1px solid var(--border); border-radius:12px; overflow:hidden; padding:4px; }
.rv-cal-header { font-size:.72rem; font-weight:700; text-transform:uppercase; color:var(--muted); text-align:center; padding:8px 4px; }
.rv-cal-hour { font-size:.68rem; color:var(--muted); text-align:right; padding:4px 8px 4px 0; display:flex; align-items:start; justify-content:flex-end; min-height:32px; }
.rv-cal-block { border-radius:4px; padding:2px 6px; font-size:.62rem; font-weight:600; color:#fff; min-height:32px; display:flex; align-items:center; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.rv-cal-meeting { background:rgba(248,113,113,.5); }
.rv-cal-focus { background:rgba(74,222,128,.4); }
.rv-cal-admin { background:rgba(128,128,128,.3); }
.rv-cal-break { background:rgba(250,204,21,.3); }
.rv-cal-empty { background:transparent; }

/* ═══ Risk Matrix ═══ */
.rv-risk-matrix { display:grid; grid-template-columns:auto 1fr 1fr 1fr; grid-template-rows:auto 1fr 1fr 1fr; gap:4px; margin-bottom:24px; background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:16px; }
.rv-risk-axis { font-size:.7rem; font-weight:600; color:var(--muted); display:flex; align-items:center; justify-content:center; }
.rv-risk-cell { border-radius:6px; min-height:60px; padding:6px; font-size:.7rem; display:flex; flex-direction:column; gap:4px; }
.rv-risk-hh { background:rgba(248,113,113,.2); }
.rv-risk-hm,.rv-risk-mh { background:rgba(250,204,21,.15); }
.rv-risk-mm,.rv-risk-hl,.rv-risk-lh { background:rgba(250,204,21,.08); }
.rv-risk-ml,.rv-risk-lm,.rv-risk-ll { background:rgba(74,222,128,.08); }
.rv-risk-tag { background:var(--bg); border-radius:4px; padding:2px 6px; font-size:.66rem; }

/* ═══ Quick Win Cards ═══ */
.rv-quick-wins { display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:12px; margin-bottom:24px; }
.rv-quick-win { background:linear-gradient(135deg,rgba(74,222,128,.08),rgba(74,222,128,.02)); border:1px solid rgba(74,222,128,.2); border-radius:12px; padding:14px 16px; }
.rv-quick-win-action { font-weight:600; font-size:.88rem; margin-bottom:6px; }
.rv-quick-win-impact { font-size:.78rem; color:var(--muted); }
.rv-quick-win-time { display:inline-block; background:rgba(74,222,128,.15); color:var(--green); padding:2px 8px; border-radius:10px; font-size:.68rem; font-weight:600; margin-top:6px; }

/* ═══ Animated Counter ═══ */
.rv-counter { transition:all .3s; }

/* ═══ PDF Download Button ═══ */
.pdf-btn {
  background: var(--accent); color: #fff; border: none; border-radius: 8px;
  padding: 8px 18px; font-size: .82rem; font-weight: 600; cursor: pointer;
  transition: opacity .15s; margin-left: auto;
}
.pdf-btn:hover { opacity: .85; }

/* ═══════════════════════════════════════════════════════════════════════════ */
/* EXECUTIVE DESIGN SYSTEM                                                   */
/* ═══════════════════════════════════════════════════════════════════════════ */

/* ── Light Theme Variables (for PDF / Report pages) ── */
.exec-light {
  --bg: #FFFFFF; --surface: #F8F9FB; --surface2: #F0F1F5; --border: #E5E7EB;
  --text: #111827; --muted: #6B7280; --accent: #4F46E5;
  --green: #059669; --yellow: #D97706; --red: #DC2626; --blue: #2563EB;
  --orange: #EA580C;
  --shadow-sm: 0 1px 2px rgba(0,0,0,0.05);
  --shadow-md: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -2px rgba(0,0,0,0.1);
  --shadow-lg: 0 10px 15px -3px rgba(0,0,0,0.1), 0 4px 6px -4px rgba(0,0,0,0.1);
}
.exec-light body, .exec-light .report-body {
  color: var(--text); background: var(--bg);
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'SF Pro', system-ui, sans-serif;
}
.exec-light .report-header {
  background: #1E1B4B; color: #fff; border-bottom: none;
  padding: 20px 32px;
}
.exec-light .report-header h1 { color: #fff; font-size: 1.4rem; letter-spacing: -0.3px; }
.exec-light .report-header .date { color: rgba(255,255,255,0.7); }
.exec-light .rv-metric {
  background: var(--surface); border: 1px solid var(--border);
  box-shadow: var(--shadow-sm); border-radius: 12px;
}
.exec-light .rv-metric-value { color: var(--text); }
.exec-light .rv-card {
  background: #fff; border: 1px solid var(--border);
  box-shadow: var(--shadow-sm); border-radius: 12px;
}
.exec-light .rv-card:hover { border-color: var(--accent); box-shadow: var(--shadow-md); }
.exec-light .rv-section-title { color: var(--text); border-color: var(--border); }
.exec-light .rv-analysis {
  background: linear-gradient(135deg, rgba(79,70,229,0.06), rgba(79,70,229,0.02));
  border-color: rgba(79,70,229,0.2);
}
.exec-light .rv-chart-container {
  background: #fff; border: 1px solid var(--border); box-shadow: var(--shadow-sm);
}
.exec-light .rv-kanban-col {
  background: var(--surface); border: 1px solid var(--border);
}
.exec-light .rv-kanban-item {
  background: #fff; border: 1px solid var(--border); box-shadow: var(--shadow-sm);
}
.exec-light .np-card {
  background: #fff; border: 1px solid var(--border); box-shadow: var(--shadow-sm);
}
.exec-light .np-editorial { color: var(--muted); }
.exec-light .rv-gauge-value { color: var(--text); }
.exec-light .rv-gauge-sub { color: var(--muted); }
.exec-light .rv-energy-slot[data-tip]:hover::after {
  background: #fff; border-color: var(--border); color: var(--text);
}

/* ── Executive Header Component ── */
.exec-header {
  position: relative; margin-bottom: 28px; border-radius: 16px; overflow: hidden;
}
.exec-header-hero {
  width: 100%; height: 200px; object-fit: cover; display: block;
  filter: brightness(0.5);
}
.exec-header-overlay {
  position: absolute; top: 0; left: 0; right: 0; bottom: 0;
  background: linear-gradient(180deg, rgba(17,24,39,0.3) 0%, rgba(17,24,39,0.85) 100%);
  display: flex; flex-direction: column; justify-content: flex-end; padding: 24px 28px;
}
.exec-header-no-hero {
  background: linear-gradient(135deg, #1E1B4B 0%, #312E81 50%, #1E1B4B 100%);
  padding: 28px 32px; border-radius: 16px;
}
.exec-header-bluf {
  font-size: 1.15rem; font-weight: 600; color: #fff; line-height: 1.45;
  letter-spacing: -0.2px; max-width: 800px;
}
.exec-header-status {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 4px 12px; border-radius: 20px; font-size: .72rem; font-weight: 700;
  text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 10px;
}
.exec-header-status-green { background: rgba(5,150,105,0.25); color: #34D399; }
.exec-header-status-yellow { background: rgba(217,119,6,0.25); color: #FBBF24; }
.exec-header-status-red { background: rgba(220,38,38,0.25); color: #FCA5A5; }
.exec-header-metrics {
  display: flex; gap: 16px; margin-top: 20px; flex-wrap: wrap;
}
.exec-header-kpi {
  background: rgba(255,255,255,0.08); backdrop-filter: blur(8px);
  border: 1px solid rgba(255,255,255,0.12); border-radius: 12px;
  padding: 14px 20px; min-width: 120px; text-align: center;
  transition: transform 0.15s, background 0.15s;
}
.exec-header-kpi:hover { transform: translateY(-2px); background: rgba(255,255,255,0.12); }
.exec-header-kpi-value {
  font-size: 1.8rem; font-weight: 800; color: #fff; line-height: 1;
  font-variant-numeric: tabular-nums;
}
.exec-header-kpi-label {
  font-size: .7rem; color: rgba(255,255,255,0.65); margin-top: 4px;
  text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600;
}
.exec-header-kpi-trend {
  font-size: .72rem; font-weight: 700; margin-top: 4px;
}
.exec-header-kpi-trend-up { color: #34D399; }
.exec-header-kpi-trend-down { color: #FCA5A5; }

/* Light-theme executive header adjustments */
.exec-light .exec-header-no-hero {
  background: linear-gradient(135deg, #1E1B4B 0%, #312E81 50%, #1E1B4B 100%);
}

/* ── Three Moves / Action Cards ── */
.exec-moves {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 14px; margin-bottom: 28px;
}
.exec-move {
  background: linear-gradient(135deg, rgba(79,70,229,0.08), rgba(79,70,229,0.02));
  border: 1px solid rgba(79,70,229,0.2); border-radius: 14px;
  padding: 18px 20px; position: relative;
}
.exec-light .exec-move {
  background: linear-gradient(135deg, rgba(79,70,229,0.06), rgba(79,70,229,0.02));
  box-shadow: var(--shadow-sm);
}
.exec-move-num {
  width: 28px; height: 28px; border-radius: 50%; background: var(--accent);
  display: flex; align-items: center; justify-content: center;
  color: #fff; font-size: .8rem; font-weight: 800; margin-bottom: 10px;
}
.exec-move-title { font-weight: 700; font-size: .92rem; margin-bottom: 4px; }
.exec-move-detail { font-size: .82rem; color: var(--muted); line-height: 1.45; }
.exec-move-time {
  display: inline-block; margin-top: 8px; padding: 2px 10px;
  background: rgba(79,70,229,0.12); color: var(--accent);
  border-radius: 10px; font-size: .68rem; font-weight: 600;
}

/* ── Talking Points Callout ── */
.exec-talking-points {
  background: linear-gradient(135deg, rgba(37,99,235,0.08), rgba(37,99,235,0.02));
  border: 1px solid rgba(37,99,235,0.2); border-radius: 14px;
  padding: 20px 24px; margin-bottom: 28px;
}
.exec-talking-points-title {
  font-size: .72rem; font-weight: 700; text-transform: uppercase;
  letter-spacing: 0.5px; color: var(--blue); margin-bottom: 12px;
  display: flex; align-items: center; gap: 8px;
}
.exec-talking-point {
  font-size: .88rem; line-height: 1.55; padding: 8px 0;
  border-bottom: 1px solid rgba(37,99,235,0.1);
}
.exec-talking-point:last-child { border-bottom: none; }

/* ── Stakeholder Heat Map ── */
.exec-stakeholder-heat {
  display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
  gap: 12px; margin-bottom: 24px;
}
.exec-stakeholder-card {
  border-radius: 12px; padding: 14px; text-align: center;
  border: 1px solid var(--border); transition: transform 0.15s;
}
.exec-stakeholder-card:hover { transform: translateY(-2px); }
.exec-stakeholder-name { font-weight: 700; font-size: .86rem; margin-bottom: 4px; }
.exec-stakeholder-count {
  font-size: 1.6rem; font-weight: 800; line-height: 1;
}
.exec-stakeholder-label { font-size: .68rem; color: var(--muted); margin-top: 4px; }

/* ── Decision Needed Cards ── */
.exec-decision-card {
  background: linear-gradient(135deg, rgba(220,38,38,0.06), rgba(220,38,38,0.02));
  border: 1px solid rgba(220,38,38,0.2); border-radius: 14px;
  padding: 18px 20px; margin-bottom: 12px;
}
.exec-decision-title {
  font-weight: 700; font-size: .92rem; display: flex; align-items: center; gap: 8px;
}
.exec-decision-detail { font-size: .84rem; color: var(--muted); margin-top: 6px; line-height: 1.5; }
.exec-decision-action {
  margin-top: 10px; padding: 8px 14px; background: rgba(220,38,38,0.08);
  border-radius: 8px; font-size: .82rem; font-weight: 600; color: var(--red);
}

/* ── Aging Bar ── */
.exec-aging-bar {
  display: flex; align-items: center; gap: 8px; margin-top: 6px;
}
.exec-aging-track {
  flex: 1; height: 6px; background: var(--surface2); border-radius: 3px; overflow: hidden;
}
.exec-aging-fill { height: 100%; border-radius: 3px; transition: width 0.4s ease; }
.exec-aging-label { font-size: .68rem; color: var(--muted); white-space: nowrap; }

/* ── Milestone Runway (horizontal cross-project) ── */
.exec-runway {
  position: relative; padding: 20px 0; margin-bottom: 24px;
  overflow-x: auto;
}
.exec-runway-track {
  height: 4px; background: var(--border); border-radius: 2px;
  position: relative; min-width: 600px;
}
.exec-runway-item {
  position: absolute; top: -8px; transform: translateX(-50%);
  display: flex; flex-direction: column; align-items: center;
}
.exec-runway-dot {
  width: 20px; height: 20px; border-radius: 50%; border: 3px solid;
  background: var(--bg); z-index: 1;
}
.exec-runway-label {
  font-size: .68rem; font-weight: 600; margin-top: 8px; text-align: center;
  max-width: 100px; line-height: 1.3;
}
.exec-runway-date { font-size: .62rem; color: var(--muted); }
.exec-runway-project { font-size: .58rem; color: var(--accent); font-weight: 600; }

/* ═══ Print / PDF Styles ═══ */
@page {
  size: A4; margin: 18mm 15mm 20mm 15mm;
}
@page :first { margin-top: 0; }

@media print {
  html, body {
    background: #fff !important; color: #111 !important;
    -webkit-print-color-adjust: exact; print-color-adjust: exact;
    font-size: 11pt; line-height: 1.5;
  }

  .top-tabs, .pdf-btn, .rich-view-back, .report-header button.pdf-btn,
  .rv-filter-bar, .rich-view-header button, .exec-light-toggle { display: none !important; }

  .rich-view-overlay { position: static !important; background: #fff !important; overflow: visible !important; }
  .rich-view-overlay.open { display: block !important; }
  .rich-view-body, .report-body { padding: 8px !important; max-width: 100% !important; }

  /* Page break control */
  .rv-card, .rv-section, .rv-kanban-col, .rv-chart-container, .rv-analysis,
  .rv-quick-win, .rv-milestone, .rv-team-member, .exec-move, .exec-decision-card,
  .exec-stakeholder-card { break-inside: avoid; }
  .rv-section { break-before: auto; }
  .exec-header { break-after: avoid; }
  .exec-header + .rv-metrics, .exec-header + .exec-moves { break-before: avoid; }

  /* Light theme override for print */
  :root {
    --bg: #fff; --surface: #f8f9fb; --surface2: #f0f1f5; --border: #e5e7eb;
    --text: #111827; --muted: #6b7280; --accent: #4F46E5;
    --green: #059669; --yellow: #D97706; --red: #DC2626; --blue: #2563EB;
    --orange: #EA580C;
  }

  .rv-metrics { flex-wrap: wrap; }
  .rv-metric { border: 1px solid #e5e7eb; background: #f8f9fb; box-shadow: none; }
  .rv-metric-value { color: #111827; }
  .rv-card { border: 1px solid #e5e7eb; background: #fff; box-shadow: none; }
  .rv-card:hover { border-color: #e5e7eb; box-shadow: none; }
  .rv-hero { max-height: 180px; border-radius: 12px; }
  .rv-network { height: 200px; }
  .rv-heatmap-cell { border: 1px solid rgba(0,0,0,.05); }

  .rv-chart-container { background: #fff; border: 1px solid #e5e7eb; box-shadow: none; }
  .rv-analysis { background: #f8f9fb; border: 1px solid #e5e7eb; }
  .rv-kanban-col { background: #f8f9fb; border: 1px solid #e5e7eb; }
  .rv-kanban-item { background: #fff; box-shadow: none; }

  .np-card { flex-direction: column !important; gap: 0 !important; background: #f8f9fb !important;
    border: 1px solid #e5e7eb !important; page-break-inside: avoid; box-shadow: none !important; }
  .np-card .np-card-expand { display: block !important; }
  .np-card .np-card-snippet { display: none !important; }
  .np-card-hero { max-height: 160px; }

  /* Executive header print adjustments */
  .exec-header-no-hero { background: #1E1B4B !important; }
  .exec-header-kpi { background: rgba(255,255,255,0.1) !important; border-color: rgba(255,255,255,0.15) !important; }
  .exec-move { background: #f8f9fb; border: 1px solid #e5e7eb; box-shadow: none; }
  .exec-talking-points { background: #f8f9fb; border: 1px solid #e5e7eb; }
  .exec-decision-card { background: #fff8f8; border: 1px solid #fecaca; }

  canvas { max-width: 100%; height: auto !important; }

  .report-header {
    background: #1E1B4B !important; color: #fff !important;
    padding: 16px 24px; margin: -18mm -15mm 16px -15mm; width: calc(100% + 30mm);
  }
  .rich-view-header { border-bottom: 2px solid #4F46E5; padding: 12px 0; }

  /* Page footer */
  .report-footer-print {
    position: fixed; bottom: 0; left: 0; right: 0;
    font-size: 8pt; color: #9ca3af; padding: 8px 15mm;
    border-top: 1px solid #e5e7eb;
    display: flex; justify-content: space-between;
  }
}

/* Hide footer in screen mode */
.report-footer-print { display: none; }

/* ══════════════════════════════════════════════════════════════════════════
   MY DAY TAB
   ══════════════════════════════════════════════════════════════════════════ */

#pane-myday { overflow-y: auto; }
.md-header { margin-bottom: 20px; }
.md-health-indicator {
  display: inline-flex; align-items: center; gap: 6px;
  font-size: .75rem; font-weight: 600; padding: 3px 10px; border-radius: 12px;
}
.md-health-indicator .md-hdot {
  width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0;
}
.md-health-indicator.live { color: #4ade80; background: rgba(74,222,128,.1); }
.md-health-indicator.live .md-hdot { background: #4ade80; box-shadow: 0 0 6px #4ade80; }
.md-health-indicator.partial { color: #facc15; background: rgba(250,204,21,.1); }
.md-health-indicator.partial .md-hdot { background: #facc15; }
.md-health-indicator.offline { color: #f87171; background: rgba(248,113,113,.15); }
.md-health-indicator.offline .md-hdot { background: #f87171; animation: mdPulse 1.5s ease-in-out infinite; }
.md-title-row { display: flex; align-items: center; gap: 24px; flex-wrap: wrap; margin-top: 12px; }
.md-summary-stats { display: flex; gap: 16px; flex-wrap: wrap; }
.md-summary-stats .md-stat {
  display: flex; align-items: center; gap: 6px;
  font-size: .82rem; color: var(--muted);
  background: rgba(108,124,255,.08); padding: 4px 12px; border-radius: 20px;
}
.md-summary-stats .md-stat b { color: var(--text); font-weight: 600; }

/* Date pills */
.md-date-nav {
  display: flex; gap: 6px; flex-wrap: wrap;
}
.md-date-pill {
  padding: 6px 14px; border-radius: 20px; border: 1px solid var(--border);
  background: var(--surface); color: var(--muted); cursor: pointer;
  font-size: .78rem; font-weight: 500; transition: all .2s;
  display: flex; flex-direction: column; align-items: center; gap: 1px;
}
.md-date-pill:hover { border-color: var(--accent); color: var(--text); }
.md-date-pill.active {
  background: linear-gradient(135deg, var(--accent), #a78bfa);
  color: #fff; border-color: transparent; font-weight: 600;
  box-shadow: 0 2px 12px rgba(108,124,255,.35);
}
.md-date-pill .md-pill-day { font-size: .68rem; text-transform: uppercase; opacity: .7; }
.md-date-pill .md-pill-num { font-size: .95rem; font-weight: 700; }
.md-date-pill.today::after {
  content: ''; display: block; width: 4px; height: 4px; border-radius: 50%;
  background: var(--green); margin-top: 2px;
}

/* Glass cards */
.md-card {
  border-radius: 16px; padding: 20px; margin-bottom: 16px;
  position: relative;
}
.md-glass {
  background: rgba(26,29,39,.65);
  backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px);
  border: 1px solid rgba(108,124,255,.12);
  box-shadow: 0 4px 24px rgba(0,0,0,.2), inset 0 1px 0 rgba(255,255,255,.04);
}
.md-card-label {
  font-size: .72rem; text-transform: uppercase; letter-spacing: .08em;
  color: var(--muted); font-weight: 600; margin-bottom: 12px;
}

/* Metrics row */
.md-metrics-row {
  display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 16px;
}
.md-metric { text-align: center; min-height: 180px; max-height: 280px; overflow: hidden; }
.md-metric-value {
  font-size: 2.4rem; font-weight: 800; line-height: 1;
  background: linear-gradient(135deg, var(--accent), #a78bfa);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  background-clip: text;
}
.md-metric-sub { font-size: .78rem; color: var(--muted); margin-top: 4px; }

/* Two-column layout */
.md-two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 16px; }

/* Timeline SVG */
#md-timeline-container svg { display: block; }
.md-tl-event { cursor: pointer; transition: opacity .15s; }
.md-tl-event:hover { opacity: .85; }
.md-tl-tooltip {
  position: absolute; z-index: 100; padding: 10px 14px; border-radius: 10px;
  background: rgba(15,17,23,.95); border: 1px solid var(--border);
  color: var(--text); font-size: .78rem; pointer-events: none;
  max-width: 300px; box-shadow: 0 8px 24px rgba(0,0,0,.4);
  backdrop-filter: blur(8px);
}
.md-tl-tooltip b { color: var(--accent); }

/* Now-marker pulse */
@keyframes mdPulse {
  0%, 100% { opacity: 1; }
  50% { opacity: .4; }
}
.md-now-line { animation: mdPulse 2s ease-in-out infinite; }

/* Focus gauge arc animation */
@keyframes mdArcIn {
  from { stroke-dashoffset: 999; }
  to { stroke-dashoffset: 0; }
}

/* Work status items */
.md-work-item {
  display: flex; align-items: flex-start; gap: 10px; padding: 10px 0;
  border-bottom: 1px solid rgba(255,255,255,.04);
}
.md-work-item:last-child { border-bottom: none; }
.md-work-icon {
  width: 22px; height: 22px; border-radius: 50%; flex-shrink: 0;
  display: flex; align-items: center; justify-content: center; font-size: .7rem;
  margin-top: 2px;
}
.md-work-icon.done { background: rgba(74,222,128,.15); color: var(--green); }
.md-work-icon.wip { background: rgba(250,204,21,.15); color: var(--yellow); }
.md-work-task { font-size: .84rem; color: var(--text); line-height: 1.4; }
.md-work-task.done-text { text-decoration: line-through; opacity: .7; }
.md-pri-pill {
  display: inline-block; font-size: .65rem; font-weight: 700; padding: 1px 6px;
  border-radius: 4px; margin-left: 6px; vertical-align: middle;
}
.md-pri-pill.p0 { background: rgba(248,113,113,.2); color: #f87171; }
.md-pri-pill.p1 { background: rgba(250,204,21,.2); color: #facc15; }
.md-pri-pill.p2 { background: rgba(108,124,255,.2); color: var(--accent); }
.md-project-tag {
  font-size: .68rem; color: var(--muted); display: block; margin-top: 2px;
}

/* Task timeline */
.md-tl-item {
  display: flex; gap: 12px; padding: 12px 0;
  border-bottom: 1px solid rgba(255,255,255,.04);
}
.md-tl-item:last-child { border-bottom: none; }
.md-tl-time {
  flex-shrink: 0; width: 90px; font-size: .75rem; font-weight: 600;
  color: var(--accent); padding-top: 2px; text-align: right;
}
.md-tl-dot {
  width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0;
  background: var(--accent); margin-top: 5px;
  box-shadow: 0 0 8px rgba(108,124,255,.4);
}
.md-tl-body { flex: 1; }
.md-tl-title { font-size: .84rem; font-weight: 600; color: var(--text); margin-bottom: 4px; }
.md-tl-apps {
  display: flex; gap: 4px; flex-wrap: wrap; margin-bottom: 4px;
}
.md-tl-app {
  font-size: .65rem; padding: 2px 8px; border-radius: 10px;
  background: rgba(108,124,255,.1); color: var(--accent); font-weight: 500;
}
.md-tl-detail { font-size: .76rem; color: var(--muted); line-height: 1.5; }

/* App usage bars */
.md-app-bar-row {
  display: flex; align-items: center; gap: 8px; padding: 3px 0 3px 18px;
}
.md-app-bar-name {
  width: 130px; font-size: .75rem; color: var(--muted); text-align: right;
  flex-shrink: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.md-app-bar-track {
  flex: 1; height: 14px; background: rgba(255,255,255,.04); border-radius: 7px; overflow: hidden;
}
.md-app-bar-fill {
  height: 100%; border-radius: 7px; transition: width .6s cubic-bezier(.22,1,.36,1);
}
.md-app-bar-val {
  width: 40px; font-size: .72rem; color: var(--muted); flex-shrink: 0; text-align: right;
}

/* People bars */
.md-people-bar-row {
  display: flex; align-items: center; gap: 10px; padding: 6px 0;
}
.md-people-name { width: 120px; font-size: .78rem; color: var(--text); text-align: right; flex-shrink: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.md-people-bar-wrap { flex: 1; height: 20px; background: rgba(255,255,255,.04); border-radius: 10px; overflow: hidden; }
.md-people-bar {
  height: 100%; border-radius: 10px; transition: width .6s cubic-bezier(.22,1,.36,1);
  background: linear-gradient(90deg, var(--accent), #a78bfa);
}
.md-people-meta { font-size: .7rem; color: var(--muted); width: 80px; flex-shrink: 0; }

/* Sunburst center label */
.md-sunburst-center {
  position: absolute; top: 50%; left: 50%; transform: translate(-50%,-50%);
  text-align: center; pointer-events: none;
}
.md-sunburst-center .md-sc-val { font-size: 1.5rem; font-weight: 800; color: var(--text); }
.md-sunburst-center .md-sc-label { font-size: .7rem; color: var(--muted); }

/* No-data state */
.md-empty {
  display: flex; flex-direction: column; align-items: center; justify-content: center;
  min-height: 200px; color: var(--muted); font-size: .88rem; gap: 8px;
}
.md-empty-icon { font-size: 2rem; opacity: .3; }

/* Responsive */
@media (max-width: 900px) {
  .md-metrics-row { grid-template-columns: repeat(2, 1fr); }
  .md-two-col { grid-template-columns: 1fr; }
}
@media (max-width: 600px) {
  .md-metrics-row { grid-template-columns: 1fr; }
}
</style>
</head>
<body>

<!-- ══════ TOP TABS ══════ -->
<div class="top-tabs">
  <button class="top-tab" onclick="switchTab('setup',this)" id="tab-setup">Setup</button>
  <button class="top-tab" onclick="switchTab('myday',this)" id="tab-myday" style="color:var(--green);font-weight:600">My Day</button>
  <button class="top-tab active" onclick="switchTab('overview',this)">Overview</button>
  <button class="top-tab" onclick="switchTab('mywork',this)">My Work</button>
  <button class="top-tab" id="tab-chat" style="color:var(--accent);font-weight:600" onclick="window.open('http://localhost:3000','_blank')">Chat &#x2197;</button>
  <button class="top-tab" onclick="switchTab('pipeline',this)">Pipeline</button>
  <button class="top-tab" onclick="switchTab('knowledge',this)">Knowledge</button>
  <button class="top-tab" onclick="switchTab('skills',this)">Skills</button>
  <button class="top-tab" onclick="switchTab('settings',this)">Settings</button>
  <button class="top-tab" onclick="switchTab('browser',this)">File Browser</button>
  <button class="top-tab" onclick="switchTab('logs',this)">Logs</button>
  <button class="top-tab" onclick="switchTab('help',this)" id="tab-help">Help</button>
</div>

<!-- ══════ TAB 0: SETUP WIZARD ══════ -->
<div class="tab-pane" id="pane-setup">
<h1 style="margin-bottom:4px">Setup Wizard</h1>
<p style="color:var(--muted);font-size:.82rem;margin-bottom:20px">Get MemoryOS running in a few simple steps</p>

<!-- Wizard step indicators -->
<div style="display:flex;gap:8px;margin-bottom:24px;flex-wrap:wrap" id="wiz-steps">
  <button class="btn wiz-step active" onclick="wizGo(1)">1. Dependencies</button>
  <button class="btn wiz-step" onclick="wizGo(2)">2. Configure</button>
  <button class="btn wiz-step" onclick="wizGo(3)">3. AI Agents</button>
  <button class="btn wiz-step" onclick="wizGo(4)">4. Activate</button>
  <button class="btn wiz-step" onclick="wizGo(5)">5. Done</button>
</div>

<!-- Step 1: Dependencies -->
<div class="wiz-panel" id="wiz-1">
<div class="card">
  <div class="card-header"><span class="card-title">Dependency Check</span><button class="btn btn-sm" onclick="loadSetup()">Re-check</button></div>
  <div id="setup-checks">Loading...</div>
</div>
<div style="text-align:right;margin-top:16px"><button class="btn" onclick="wizGo(2)">Next &rarr;</button></div>
</div>

<!-- Step 2: Configure Paths -->
<div class="wiz-panel" id="wiz-2" style="display:none">
<div class="card">
  <div class="card-header"><span class="card-title">Configure Your Paths</span></div>
  <div style="margin-bottom:20px">
    <label style="font-size:.85rem;font-weight:600;display:block;margin-bottom:6px">Obsidian Vault Path <span style="color:var(--red)">*</span></label>
    <input id="cfg-vault" style="width:100%;max-width:600px;padding:8px 12px;border-radius:8px;border:1px solid var(--border);background:var(--bg);color:var(--text);font-size:.85rem" placeholder="/Users/you/Documents/Obsidian/MyVault">
    <div id="cfg-vault-suggestions" style="margin-top:8px;display:flex;gap:8px;flex-wrap:wrap"></div>
  </div>
  <div style="margin-bottom:20px">
    <label style="font-size:.85rem;font-weight:600;display:block;margin-bottom:6px">Email Source</label>
    <div style="display:flex;gap:16px;flex-wrap:wrap">
      <label class="radio-card"><input type="radio" name="email-src" value="mail_app" checked> <strong>Mail.app</strong><br><span style="color:var(--muted);font-size:.78rem">Easiest — zero config if email is in System Settings</span></label>
      <label class="radio-card"><input type="radio" name="email-src" value="outlook"> <strong>Outlook (Classic)</strong><br><span style="color:var(--muted);font-size:.78rem">Local SQLite DB — for Outlook for Mac users</span></label>
      <label class="radio-card"><input type="radio" name="email-src" value="graph"> <strong>Microsoft Graph</strong><br><span style="color:var(--muted);font-size:.78rem">Cloud API — requires Azure AD app registration</span></label>
    </div>
  </div>
  <div style="margin-bottom:20px">
    <label style="font-size:.85rem;font-weight:600;display:block;margin-bottom:6px">Calendar Source</label>
    <div style="display:flex;gap:16px;flex-wrap:wrap">
      <label class="radio-card"><input type="radio" name="cal-src" value="calendar_app" checked> <strong>Calendar.app</strong><br><span style="color:var(--muted);font-size:.78rem">Easiest — works with iCloud, Google, Exchange</span></label>
      <label class="radio-card"><input type="radio" name="cal-src" value="outlook"> <strong>Outlook</strong><br><span style="color:var(--muted);font-size:.78rem">Uses Outlook local database</span></label>
      <label class="radio-card"><input type="radio" name="cal-src" value="graph"> <strong>Microsoft Graph</strong><br><span style="color:var(--muted);font-size:.78rem">Cloud API — same Azure AD app as email</span></label>
    </div>
  </div>
  <div style="margin-bottom:20px" id="cfg-graph-row" style="display:none">
    <label style="font-size:.85rem;font-weight:600;display:block;margin-bottom:6px">Graph API Client ID <span style="color:var(--muted);font-weight:400">(only if using Graph)</span></label>
    <input id="cfg-graph-id" style="width:100%;max-width:600px;padding:8px 12px;border-radius:8px;border:1px solid var(--border);background:var(--bg);color:var(--text);font-size:.85rem" placeholder="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx">
  </div>
  <div style="margin-bottom:20px">
    <label style="font-size:.85rem;font-weight:600;display:block;margin-bottom:6px">OneDrive Sync Folder <span style="color:var(--muted);font-weight:400">(optional)</span></label>
    <input id="cfg-onedrive" style="width:100%;max-width:600px;padding:8px 12px;border-radius:8px;border:1px solid var(--border);background:var(--bg);color:var(--text);font-size:.85rem" placeholder="~/Library/CloudStorage/OneDrive-YourOrg">
    <div id="cfg-od-suggestions" style="margin-top:8px;display:flex;gap:8px;flex-wrap:wrap"></div>
  </div>
  <button class="btn" onclick="saveSetupConfig()" id="cfg-save-btn">Save Configuration</button>
  <span id="cfg-save-msg" style="margin-left:12px;font-size:.82rem"></span>
</div>
<div style="display:flex;justify-content:space-between;margin-top:16px"><button class="btn" onclick="wizGo(1)">&larr; Back</button><button class="btn" onclick="wizGo(3)">Next &rarr;</button></div>
</div>

<!-- Step 3: AI Agents -->
<div class="wiz-panel" id="wiz-3" style="display:none">
<div class="card">
  <div class="card-header"><span class="card-title">LLM Provider</span></div>
  <p style="color:var(--muted);font-size:.82rem;margin-bottom:16px">Configure the AI model used for headless skill execution (morning briefs, weekly plans, news pulse).</p>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;max-width:600px">
    <div>
      <label style="font-size:.8rem;color:var(--muted);display:block;margin-bottom:4px">Provider</label>
      <select id="wiz-llm-provider" onchange="wizProviderChanged()" style="width:100%;padding:8px 12px;border-radius:8px;border:1px solid var(--border);background:var(--bg);color:var(--text);font-size:.85rem">
        <option value="openai">OpenAI</option>
        <option value="anthropic">Anthropic</option>
        <option value="google">Google Gemini</option>
        <option value="azure">Azure AI Foundry</option>
        <option value="ollama">Ollama (Local)</option>
      </select>
    </div>
    <div>
      <label style="font-size:.8rem;color:var(--muted);display:block;margin-bottom:4px">Model</label>
      <input id="wiz-llm-model" value="gpt-5.2" style="width:100%;padding:8px 12px;border-radius:8px;border:1px solid var(--border);background:var(--bg);color:var(--text);font-size:.85rem">
    </div>
    <div id="wiz-llm-apibase-row" style="display:none">
      <label style="font-size:.8rem;color:var(--muted);display:block;margin-bottom:4px">API Base URL</label>
      <input id="wiz-llm-apibase" placeholder="http://localhost:11434" style="width:100%;padding:8px 12px;border-radius:8px;border:1px solid var(--border);background:var(--bg);color:var(--text);font-size:.85rem">
    </div>
    <div>
      <label style="font-size:.8rem;color:var(--muted);display:block;margin-bottom:4px">Reasoning Effort</label>
      <select id="wiz-llm-reasoning" style="width:100%;padding:8px 12px;border-radius:8px;border:1px solid var(--border);background:var(--bg);color:var(--text);font-size:.85rem">
        <option value="high">High</option>
        <option value="low">Low</option>
        <option value="none">None</option>
      </select>
    </div>
  </div>
  <div style="margin-top:16px;display:flex;align-items:center;gap:12px">
    <button class="btn" onclick="wizTestLLM(this)">Test Connection</button>
    <span id="wiz-llm-msg" style="font-size:.82rem"></span>
  </div>
</div>

<div class="card" style="margin-top:16px">
  <div class="card-header"><span class="card-title">Email Delivery</span></div>
  <p style="color:var(--muted);font-size:.82rem;margin-bottom:16px">Configure SMTP to receive scheduled skill reports by email. <span style="color:var(--muted);font-style:italic">(Optional -- reports are always saved to the vault.)</span></p>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;max-width:600px">
    <div>
      <label style="font-size:.8rem;color:var(--muted);display:block;margin-bottom:4px">SMTP Host</label>
      <input id="wiz-smtp-host" value="smtp.office365.com" style="width:100%;padding:8px 12px;border-radius:8px;border:1px solid var(--border);background:var(--bg);color:var(--text);font-size:.85rem">
    </div>
    <div>
      <label style="font-size:.8rem;color:var(--muted);display:block;margin-bottom:4px">SMTP Port</label>
      <input id="wiz-smtp-port" type="number" value="587" style="width:100%;padding:8px 12px;border-radius:8px;border:1px solid var(--border);background:var(--bg);color:var(--text);font-size:.85rem">
    </div>
    <div>
      <label style="font-size:.8rem;color:var(--muted);display:block;margin-bottom:4px">From / User</label>
      <input id="wiz-smtp-user" placeholder="you@company.com" style="width:100%;padding:8px 12px;border-radius:8px;border:1px solid var(--border);background:var(--bg);color:var(--text);font-size:.85rem">
    </div>
    <div>
      <label style="font-size:.8rem;color:var(--muted);display:block;margin-bottom:4px">Send To</label>
      <input id="wiz-smtp-to" placeholder="you@company.com" style="width:100%;padding:8px 12px;border-radius:8px;border:1px solid var(--border);background:var(--bg);color:var(--text);font-size:.85rem">
    </div>
  </div>
  <div style="margin-top:16px;display:flex;align-items:center;gap:12px">
    <button class="btn" onclick="wizSaveAgents(this)">Save Agent Config</button>
    <span id="wiz-agents-msg" style="font-size:.82rem"></span>
  </div>
</div>
<div style="display:flex;justify-content:space-between;margin-top:16px"><button class="btn" onclick="wizGo(2)">&larr; Back</button><button class="btn" onclick="wizGo(4)">Next &rarr;</button></div>
</div>

<!-- Step 4: Activate -->
<div class="wiz-panel" id="wiz-4" style="display:none">
<div class="card">
  <div class="card-header"><span class="card-title">Activate MemoryOS</span></div>
  <p style="color:var(--muted);font-size:.82rem;margin-bottom:20px">Install background agents, run your first extraction, and build the search index.</p>
  <div style="display:flex;flex-direction:column;gap:12px;max-width:500px">
    <div style="display:flex;align-items:center;gap:12px">
      <button class="btn" onclick="setupInstallAgents(this)" style="min-width:220px">Install Background Agents</button>
      <span id="act-agents" style="font-size:.82rem;color:var(--muted)"></span>
    </div>
    <div style="display:flex;align-items:center;gap:12px">
      <button class="btn" onclick="setupRunExtractors(this)" style="min-width:220px">Run First Extraction</button>
      <span id="act-extract" style="font-size:.82rem;color:var(--muted)"></span>
    </div>
    <div style="display:flex;align-items:center;gap:12px">
      <button class="btn" onclick="setupReindex(this)" style="min-width:220px">Build Search Index</button>
      <span id="act-index" style="font-size:.82rem;color:var(--muted)"></span>
    </div>
  </div>
</div>
<div style="display:flex;justify-content:space-between;margin-top:16px"><button class="btn" onclick="wizGo(3)">&larr; Back</button><button class="btn" onclick="wizGo(5)">Next &rarr;</button></div>
</div>

<!-- Step 5: Done -->
<div class="wiz-panel" id="wiz-5" style="display:none">
<div class="card" style="text-align:center;padding:40px">
  <div style="font-size:2.5rem;margin-bottom:12px">&#10003;</div>
  <h2 style="margin-bottom:8px">You're All Set</h2>
  <p style="color:var(--muted);margin-bottom:24px">MemoryOS is running. Your data will start flowing into the vault within minutes.</p>
  <div style="display:flex;gap:12px;justify-content:center;flex-wrap:wrap">
    <button class="btn" onclick="document.querySelector('.top-tab:nth-child(2)').click()">Go to Overview</button>
    <button class="btn" onclick="document.getElementById('tab-help').click()">Read the Help Docs</button>
  </div>
</div>
</div>

</div><!-- /pane-setup -->

<!-- ══════ MY DAY TAB ══════ -->
<div class="tab-pane" id="pane-myday">

<!-- Date navigation pills -->
<div class="md-header">
  <div class="md-date-nav" id="md-date-pills"></div>
  <div class="md-title-row">
    <h1 id="md-day-title" style="margin:0;font-size:1.6rem">My Day</h1>
    <div id="md-health-dot" class="md-health-indicator"></div>
    <div class="md-summary-stats" id="md-summary-stats"></div>
  </div>
</div>

<!-- Row 1: Timeline hero -->
<div class="md-card md-glass" id="md-timeline-card">
  <div class="md-card-label">Timeline</div>
  <div id="md-timeline-container" style="width:100%;overflow-x:auto"></div>
</div>

<!-- Row 2: Four metric cards -->
<div class="md-metrics-row">
  <div class="md-card md-glass md-metric" id="md-focus-card">
    <div class="md-card-label">Focus Score</div>
    <div id="md-focus-gauge"></div>
  </div>
  <div class="md-card md-glass md-metric" id="md-meetings-card">
    <div class="md-card-label">Meeting Load</div>
    <div id="md-meetings-content"></div>
  </div>
  <div class="md-card md-glass md-metric" id="md-switches-card">
    <div class="md-card-label">Context Switches</div>
    <div id="md-switches-content"></div>
  </div>
  <div class="md-card md-glass md-metric" id="md-completed-card">
    <div class="md-card-label">Work Completed</div>
    <div id="md-completed-content"></div>
  </div>
</div>

<!-- Row 3: Sunburst + People -->
<div class="md-two-col">
  <div class="md-card md-glass">
    <div class="md-card-label">App Usage</div>
    <div id="md-sunburst-container" style="display:flex;justify-content:center;align-items:center;min-height:320px"></div>
  </div>
  <div class="md-card md-glass">
    <div class="md-card-label">People</div>
    <div id="md-people-container" style="min-height:320px"></div>
  </div>
</div>

<!-- Row 4: Work Status + Activity Timeline -->
<div class="md-two-col">
  <div class="md-card md-glass">
    <div class="md-card-label">Work Status</div>
    <div id="md-work-status" style="max-height:500px;overflow-y:auto"></div>
  </div>
  <div class="md-card md-glass">
    <div class="md-card-label">Activity Timeline</div>
    <div id="md-task-timeline" style="max-height:500px;overflow-y:auto"></div>
  </div>
</div>

</div><!-- /pane-myday -->

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
  <div class="audio-device-picker">
    <select id="audio-device-select" onchange="switchAudioDevice(this.value)" disabled>
      <option value="">Loading devices...</option>
    </select>
    <button class="btn-icon" onclick="loadAudioDevices()" title="Refresh device list">&#x21bb;</button>
    <span class="audio-device-msg" id="audio-device-msg"></span>
  </div>
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

<!-- ══════ MY WORK TAB ══════ -->
<div class="tab-pane" id="pane-mywork">
<h2 style="margin-bottom:4px">My Work</h2>
<p style="color:var(--muted);font-size:.82rem;margin-bottom:12px">Your master task list. Edits save to tasks.md and priorities.md in the vault.</p>

<div class="mw-topbar">
  <div class="mw-toggle">
    <button class="active" onclick="toggleMyWorkView('table',this)">Table</button>
    <button onclick="toggleMyWorkView('raw',this)">Raw Markdown</button>
  </div>
  <div class="mw-filters">
    <label style="display:flex;align-items:center;gap:4px;font-size:.78rem;color:var(--muted);cursor:pointer;user-select:none"><input type="checkbox" id="mw-hide-done" onchange="filterTasks()" style="accent-color:var(--accent)"> Hide complete</label>
    <select id="mw-flt-status" onchange="filterTasks()"><option value="">All Status</option><option value="not_started">Not Started</option><option value="in_progress">In Progress</option><option value="complete">Complete</option><option value="waiting">Waiting</option></select>
    <select id="mw-flt-pri" onchange="filterTasks()"><option value="">All Priority</option><option value="P0">P0</option><option value="P1">P1</option><option value="P2">P2</option></select>
    <select id="mw-flt-proj" onchange="filterTasks()"><option value="">All Projects</option></select>
  </div>
  <button class="btn btn-sm btn-green" onclick="addNewProject()">+ Add Task</button>
</div>

<!-- Table view -->
<div id="mw-table-view">
  <div class="card" style="padding:0;overflow:hidden">
    <div class="task-hdr"><span>Task</span><span>Owner</span><span>Priority</span><span>Status</span><span>Due</span><span>Notes</span><span></span></div>
    <div id="mw-task-rows"></div>
  </div>
  <div class="task-bottom">
    <button class="btn" onclick="saveTaskTable()">Save All</button>
    <span id="mw-tbl-msg" style="font-size:.82rem"></span>
    <div class="task-counts"><span id="mw-cnt-open"></span><span id="mw-cnt-prog"></span><span id="mw-cnt-done"></span></div>
  </div>
</div>

<!-- Raw markdown view (hidden by default) -->
<div id="mw-raw-view" style="display:none">
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px" id="mywork-grid">
    <div class="card" style="display:flex;flex-direction:column">
      <div class="card-header" style="flex-shrink:0">
        <span class="card-title">Priorities</span>
        <div style="display:flex;gap:8px;align-items:center">
          <span id="mw-pri-msg" style="font-size:.78rem"></span>
          <button class="btn btn-sm" onclick="saveContextFile('priorities.md')">Save</button>
        </div>
      </div>
      <textarea class="cfg-textarea" id="mw-priorities" style="flex:1;min-height:400px;font-family:'SF Mono','Menlo',monospace;font-size:.8rem;line-height:1.6;resize:vertical" spellcheck="false"></textarea>
    </div>
    <div class="card" style="display:flex;flex-direction:column">
      <div class="card-header" style="flex-shrink:0">
        <span class="card-title">Tasks</span>
        <div style="display:flex;gap:8px;align-items:center">
          <span id="mw-task-msg" style="font-size:.78rem"></span>
          <button class="btn btn-sm" onclick="saveContextFile('tasks.md')">Save</button>
        </div>
      </div>
      <textarea class="cfg-textarea" id="mw-tasks" style="flex:1;min-height:400px;font-family:'SF Mono','Menlo',monospace;font-size:.8rem;line-height:1.6;resize:vertical" spellcheck="false"></textarea>
    </div>
  </div>
</div>

</div><!-- /pane-mywork -->

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

<!-- ══════ KNOWLEDGE TAB ══════ -->
<div class="tab-pane" id="pane-knowledge">
<h1 style="margin-bottom:4px">Work Knowledge</h1>
<p style="color:var(--muted);font-size:.82rem;margin-bottom:20px">Your identity, programs, patterns, and preferences. This context is injected into every agent skill.</p>

<div class="card settings-section">
  <h3>My Identity</h3>
  <p style="color:var(--muted);font-size:.8rem;margin-bottom:12px">Who you are. Injected into every agent skill so reports get attributions and relationships right.</p>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;max-width:600px;margin-bottom:14px">
    <div><label class="cfg-label">Name</label><input class="cfg-input" id="ctx-name"></div>
    <div><label class="cfg-label">Title</label><input class="cfg-input" id="ctx-title"></div>
    <div><label class="cfg-label">Company</label><input class="cfg-input" id="ctx-company"></div>
    <div><label class="cfg-label">Reports To</label><input class="cfg-input" id="ctx-reports-to"></div>
  </div>
  <label class="cfg-label" style="margin-bottom:6px">Key Relationships</label>
  <div id="ctx-rels"></div>
  <button class="btn btn-sm" style="margin-top:8px" onclick="addCtxRelationship()">+ Add Relationship</button>
  <div style="margin-top:14px;display:flex;gap:8px;align-items:center">
    <button class="btn" onclick="saveContext()">Save</button>
    <span id="ctx-save-msg" style="font-size:.82rem"></span>
  </div>
</div>

<div class="card settings-section">
  <h3>Work Environment</h3>
  <p style="color:var(--muted);font-size:.8rem;margin-bottom:12px">Company and industry context shared across all agent skills.</p>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;max-width:600px;margin-bottom:14px">
    <div><label class="cfg-label">Company</label><input class="cfg-input" id="gctx-company"></div>
    <div><label class="cfg-label">Industry</label><input class="cfg-input" id="gctx-industry"></div>
  </div>
  <div style="margin-bottom:14px">
    <label class="cfg-label">Competitors</label>
    <div style="display:flex;flex-wrap:wrap;gap:2px;margin-bottom:6px" id="gctx-competitors"></div>
    <div style="display:flex;gap:6px;max-width:400px"><input class="cfg-input" id="gctx-comp-input" placeholder="Add competitor..." onkeydown="if(event.key==='Enter')addGctxTag('competitors')"><button class="btn btn-sm" onclick="addGctxTag('competitors')">Add</button></div>
  </div>
  <div style="margin-bottom:14px">
    <label class="cfg-label">Strategic Priorities</label>
    <div id="gctx-priorities"></div>
    <div style="display:flex;gap:6px;max-width:500px;margin-top:6px"><input class="cfg-input" id="gctx-pri-input" placeholder="Add priority..." onkeydown="if(event.key==='Enter')addGctxTag('priorities')"><button class="btn btn-sm" onclick="addGctxTag('priorities')">Add</button></div>
  </div>
  <div style="margin-bottom:14px">
    <label class="cfg-label">Tools &amp; Platforms</label>
    <div style="display:flex;flex-wrap:wrap;gap:2px;margin-bottom:6px" id="gctx-tools"></div>
    <div style="display:flex;gap:6px;max-width:400px"><input class="cfg-input" id="gctx-tool-input" placeholder="Add tool..." onkeydown="if(event.key==='Enter')addGctxTag('tools')"><button class="btn btn-sm" onclick="addGctxTag('tools')">Add</button></div>
  </div>
  <div style="margin-top:14px;display:flex;gap:8px;align-items:center">
    <button class="btn" onclick="saveContext()">Save</button>
    <span id="gctx-save-msg" style="font-size:.82rem"></span>
  </div>
</div>

<div class="card">
  <div class="card-header">
    <span class="card-title">Pinned Knowledge Files</span>
    <div style="display:flex;gap:8px">
      <select id="kf-new-template" class="cfg-input" style="width:160px;font-size:.78rem">
        <option value="">New from template...</option>
      </select>
      <select id="kf-new-category" class="cfg-input" style="width:130px;font-size:.78rem">
        <option value="01_programs">Programs</option>
        <option value="00_profile">Profile</option>
        <option value="03_patterns">Patterns</option>
        <option value="02_preferences">Preferences</option>
      </select>
      <input id="kf-new-name" class="cfg-input" style="width:180px;font-size:.78rem" placeholder="filename...">
      <button class="btn btn-sm" onclick="createKnowledgeFile()">Create</button>
      <button class="btn btn-sm" onclick="loadKnowledgeFiles()">Refresh</button>
    </div>
  </div>
  <p style="color:var(--muted);font-size:.8rem;margin:0 0 12px;padding:0 12px">Your work memory: programs, patterns, preferences, and identity files. Click a row to view/edit. Pinned files feed into <code>_context/core.md</code> for all agents.</p>
  <div id="kf-list"></div>
</div>

</div><!-- /pane-knowledge -->

<!-- ══════ SETTINGS TAB ══════ -->
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

<div class="card settings-section">
  <h3>AI Provider</h3>
  <p style="color:var(--muted);font-size:.8rem;margin-bottom:12px">LLM used for headless skill execution (morning brief, weekly plan, news pulse).</p>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;max-width:500px">
    <div>
      <label style="font-size:.8rem;color:var(--muted);display:block;margin-bottom:4px">Provider</label>
      <select id="set-llm-provider" onchange="setProviderChanged()" style="width:100%;padding:6px 10px;border-radius:8px;border:1px solid var(--border);background:var(--bg);color:var(--text);font-size:.85rem">
        <option value="openai">OpenAI</option>
        <option value="anthropic">Anthropic</option>
        <option value="google">Google Gemini</option>
        <option value="azure">Azure AI Foundry</option>
        <option value="ollama">Ollama (Local)</option>
      </select>
    </div>
    <div>
      <label style="font-size:.8rem;color:var(--muted);display:block;margin-bottom:4px">Model</label>
      <input id="set-llm-model" style="width:100%;padding:6px 10px;border-radius:8px;border:1px solid var(--border);background:var(--bg);color:var(--text);font-size:.85rem">
    </div>
    <div id="set-llm-apibase-row" style="display:none">
      <label style="font-size:.8rem;color:var(--muted);display:block;margin-bottom:4px">API Base URL</label>
      <input id="set-llm-apibase" style="width:100%;padding:6px 10px;border-radius:8px;border:1px solid var(--border);background:var(--bg);color:var(--text);font-size:.85rem">
    </div>
    <div>
      <label style="font-size:.8rem;color:var(--muted);display:block;margin-bottom:4px">Reasoning Effort</label>
      <select id="set-llm-reasoning" style="width:100%;padding:6px 10px;border-radius:8px;border:1px solid var(--border);background:var(--bg);color:var(--text);font-size:.85rem">
        <option value="high">High</option>
        <option value="low">Low</option>
        <option value="none">None</option>
      </select>
    </div>
  </div>
  <div style="margin-top:16px;display:flex;gap:8px;align-items:center">
    <button class="btn" onclick="saveAgentSettings()">Save</button>
    <button class="btn btn-sm" onclick="testLLMFromSettings(this)">Test Connection</button>
    <span id="set-llm-msg" style="font-size:.82rem"></span>
  </div>
</div>

<div class="card settings-section">
  <h3>Email Delivery</h3>
  <p style="color:var(--muted);font-size:.8rem;margin-bottom:12px">SMTP settings for sending scheduled skill reports.</p>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;max-width:500px">
    <div>
      <label style="font-size:.8rem;color:var(--muted);display:block;margin-bottom:4px">SMTP Host</label>
      <input id="set-smtp-host" style="width:100%;padding:6px 10px;border-radius:8px;border:1px solid var(--border);background:var(--bg);color:var(--text);font-size:.85rem">
    </div>
    <div>
      <label style="font-size:.8rem;color:var(--muted);display:block;margin-bottom:4px">SMTP Port</label>
      <input id="set-smtp-port" type="number" style="width:100%;padding:6px 10px;border-radius:8px;border:1px solid var(--border);background:var(--bg);color:var(--text);font-size:.85rem">
    </div>
    <div>
      <label style="font-size:.8rem;color:var(--muted);display:block;margin-bottom:4px">From / User</label>
      <input id="set-smtp-user" style="width:100%;padding:6px 10px;border-radius:8px;border:1px solid var(--border);background:var(--bg);color:var(--text);font-size:.85rem">
    </div>
    <div>
      <label style="font-size:.8rem;color:var(--muted);display:block;margin-bottom:4px">Send To</label>
      <input id="set-smtp-to" style="width:100%;padding:6px 10px;border-radius:8px;border:1px solid var(--border);background:var(--bg);color:var(--text);font-size:.85rem">
    </div>
  </div>
  <div style="margin-top:16px;display:flex;gap:8px;align-items:center">
    <button class="btn" onclick="saveAgentSettings()">Save</button>
    <button class="btn btn-sm" onclick="testEmailFromSettings(this)">Send Test Email</button>
    <span id="set-email-msg" style="font-size:.82rem"></span>
  </div>
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

<!-- ══════ TAB 6: SKILLS ══════ -->
<div class="tab-pane" id="pane-skills">
<h1 style="margin-bottom:4px">Agent Skills</h1>
<p style="color:var(--muted);font-size:.82rem;margin-bottom:20px">Installed skills in ~/.cursor/skills/ &mdash; these extend what the AI agent can do</p>

<div class="card" id="scheduled-agents-card" style="margin-bottom:20px;display:none">
  <div class="card-header"><span class="card-title">Scheduled Agents</span>
    <button class="btn btn-sm" onclick="loadAgentStatus()">Refresh</button>
  </div>
  <div style="overflow-x:auto">
    <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr auto;gap:0;font-size:.78rem;font-weight:600;color:var(--muted);padding:8px 12px;border-bottom:2px solid var(--border)">
      <span>Skill</span><span>Schedule</span><span>Last Run</span><span>Status</span><span>Actions</span>
    </div>
    <div id="agent-status-rows"></div>
  </div>
</div>

<div id="skills-grid" class="skills-grid">Loading skills...</div>
</div><!-- /pane-skills -->

<!-- ══════ TAB 7: HELP ══════ -->
<div class="tab-pane" id="pane-help">
<h1 style="margin-bottom:4px">Help &amp; Documentation</h1>
<p style="color:var(--muted);font-size:.82rem;margin-bottom:20px">Everything you need to know about MemoryOS</p>

<div class="card">
<h3 style="margin-bottom:12px">Getting Started</h3>
<ol style="padding-left:20px;line-height:2;color:var(--muted);font-size:.88rem">
  <li><strong style="color:var(--text)">Clone the repo:</strong> <code>git clone https://github.com/Brianletort/MemoryOS.git</code></li>
  <li><strong style="color:var(--text)">Run setup:</strong> <code>cd MemoryOS && ./scripts/setup.sh</code></li>
  <li><strong style="color:var(--text)">Configure:</strong> Use the <a href="#" onclick="document.getElementById('tab-setup').click();return false" style="color:var(--accent)">Setup Wizard</a> to set your paths</li>
  <li><strong style="color:var(--text)">Activate:</strong> Install background agents and run your first extraction</li>
</ol>
</div>

<div class="card">
<h3 style="margin-bottom:12px">How It Works</h3>
<p style="color:var(--muted);font-size:.85rem;line-height:1.7;margin-bottom:12px">MemoryOS is a three-layer system:</p>
<div style="font-family:monospace;font-size:.78rem;line-height:1.6;background:var(--bg);padding:16px;border-radius:8px;overflow-x:auto;color:var(--muted)">
<pre style="margin:0">
Interface Layer:   Cursor (@context) | CLI | ChatGPT | Dashboard
                         |              |         |
Memory Layer:      SQLite FTS5 Index | Context Generator | Hot/Warm/Cold Tiers
                         |
Collection Layer:  Extractors -> Obsidian Vault (Markdown source of truth)
                   Screenpipe | Mail/Calendar | Outlook/Graph | OneDrive</pre>
</div>
<p style="color:var(--muted);font-size:.82rem;margin-top:12px">Extractors run every 5 minutes via launchd, writing structured Markdown. The memory layer indexes everything for instant search. Context files in <code>_context/</code> are auto-generated summaries optimized for AI tools.</p>
</div>

<div class="card">
<h3 style="margin-bottom:12px">Data Sources</h3>
<table style="width:100%;font-size:.82rem;border-collapse:collapse">
<tr style="border-bottom:1px solid var(--border)"><th style="text-align:left;padding:8px 4px;color:var(--muted)">Source</th><th style="text-align:left;padding:8px 4px;color:var(--muted)">How</th><th style="text-align:left;padding:8px 4px;color:var(--muted)">Output</th></tr>
<tr style="border-bottom:1px solid var(--border)"><td style="padding:8px 4px">Screen activity</td><td style="padding:8px 4px">OCR every ~2s via Screenpipe</td><td style="padding:8px 4px"><code>85_activity/YYYY/MM/DD/daily.md</code></td></tr>
<tr style="border-bottom:1px solid var(--border)"><td style="padding:8px 4px">Your voice</td><td style="padding:8px 4px">Microphone transcription</td><td style="padding:8px 4px"><code>10_meetings/YYYY/MM/DD/audio.md</code></td></tr>
<tr style="border-bottom:1px solid var(--border)"><td style="padding:8px 4px">Meeting audio</td><td style="padding:8px 4px">System audio via BlackHole</td><td style="padding:8px 4px"><code>10_meetings/YYYY/MM/DD/audio.md</code></td></tr>
<tr style="border-bottom:1px solid var(--border)"><td style="padding:8px 4px">Emails</td><td style="padding:8px 4px">Mail.app, Outlook, or Graph API</td><td style="padding:8px 4px"><code>00_inbox/YYYY/MM/DD/*.md</code></td></tr>
<tr style="border-bottom:1px solid var(--border)"><td style="padding:8px 4px">Calendar</td><td style="padding:8px 4px">Calendar.app, Outlook, or Graph</td><td style="padding:8px 4px"><code>10_meetings/YYYY/MM/DD/calendar.md</code></td></tr>
<tr style="border-bottom:1px solid var(--border)"><td style="padding:8px 4px">Teams chat</td><td style="padding:8px 4px">Screen OCR when Teams is active</td><td style="padding:8px 4px"><code>20_teams-chat/YYYY/MM/DD/teams.md</code></td></tr>
<tr><td style="padding:8px 4px">Documents</td><td style="padding:8px 4px">OneDrive files (docx, pptx, pdf)</td><td style="padding:8px 4px"><code>40_slides/</code> and <code>50_knowledge/</code></td></tr>
</table>
</div>

<div class="card">
<h3 style="margin-bottom:12px">Email &amp; Calendar Options</h3>
<table style="width:100%;font-size:.82rem;border-collapse:collapse">
<tr style="border-bottom:1px solid var(--border)"><th style="text-align:left;padding:8px 4px;color:var(--muted)">Option</th><th style="text-align:left;padding:8px 4px;color:var(--muted)">Best For</th><th style="text-align:left;padding:8px 4px;color:var(--muted)">Setup</th></tr>
<tr style="border-bottom:1px solid var(--border)"><td style="padding:8px 4px"><strong>Mail.app + Calendar.app</strong></td><td style="padding:8px 4px">Most users &mdash; works with any email provider configured in System Settings</td><td style="padding:8px 4px">Zero config</td></tr>
<tr style="border-bottom:1px solid var(--border)"><td style="padding:8px 4px"><strong>Outlook (Classic)</strong></td><td style="padding:8px 4px">Outlook for Mac users (not &ldquo;New Outlook&rdquo;)</td><td style="padding:8px 4px">Auto-detects DB path</td></tr>
<tr><td style="padding:8px 4px"><strong>Microsoft Graph API</strong></td><td style="padding:8px 4px">Full HTML email bodies, detailed calendar, New Outlook</td><td style="padding:8px 4px">Requires Azure AD app registration</td></tr>
</table>
</div>

<div class="card">
<h3 style="margin-bottom:12px">Audio Setup (BlackHole)</h3>
<p style="color:var(--muted);font-size:.82rem;margin-bottom:12px">To capture both sides of a meeting call (not just your microphone), you need a virtual audio loopback:</p>
<ol style="padding-left:20px;line-height:2.2;color:var(--muted);font-size:.85rem">
  <li><strong style="color:var(--text)">Install BlackHole:</strong> <code>brew install blackhole-2ch</code></li>
  <li><strong style="color:var(--text)">Open Audio MIDI Setup</strong> (Spotlight search)</li>
  <li><strong style="color:var(--text)">Create Multi-Output Device:</strong> Click <strong>+</strong> at bottom left &rarr; check your speakers <strong>and</strong> BlackHole 2ch</li>
  <li><strong style="color:var(--text)">Route system audio:</strong> System Settings &rarr; Sound &rarr; Output &rarr; select the Multi-Output Device</li>
  <li><strong style="color:var(--text)">Configure Screenpipe:</strong> Add <strong>BlackHole 2ch</strong> as an audio input device in Screenpipe settings</li>
</ol>
</div>

<div class="card">
<h3 style="margin-bottom:12px">Privacy Controls</h3>
<ul style="padding-left:20px;line-height:2;color:var(--muted);font-size:.85rem">
  <li><strong style="color:var(--text)">Privacy mode:</strong> Toggle on the Overview tab to instantly disable all audio capture</li>
  <li><strong style="color:var(--text)">WiFi-based auto-privacy:</strong> Automatically enables privacy mode on untrusted networks</li>
  <li><strong style="color:var(--text)">Work hours filter:</strong> Only capture audio during configured work hours</li>
  <li><strong style="color:var(--text)">Work app correlation:</strong> Only keep audio when work apps are active on screen</li>
  <li><strong style="color:var(--text)">All data stays local:</strong> Nothing leaves your machine unless you explicitly upload it</li>
</ul>
</div>

<div class="card">
<h3 style="margin-bottom:12px">CLI Reference</h3>
<div style="font-family:monospace;font-size:.8rem;line-height:1.8;background:var(--bg);padding:16px;border-radius:8px;overflow-x:auto">
<pre style="margin:0;color:var(--muted)">python3 -m src.memory.cli search "query"            # Full-text search
python3 -m src.memory.cli search "query" --type email  # Filter by type
python3 -m src.memory.cli recent --hours 24          # Last 24 hours
python3 -m src.memory.cli recent --type meetings     # Recent meetings
python3 -m src.memory.cli meetings --date 2026-02-19 # Specific date
python3 -m src.memory.cli stats                      # Index statistics
python3 -m src.memory.cli reindex                    # Incremental reindex
python3 -m src.memory.cli reindex --full             # Full reindex</pre>
</div>
<p style="color:var(--muted);font-size:.82rem;margin-top:8px">Source types: <code>email</code>, <code>meetings</code>, <code>teams</code>, <code>activity</code>, <code>knowledge</code>, <code>slides</code></p>
</div>

<div class="card">
<h3 style="margin-bottom:12px">Troubleshooting</h3>
<details style="margin-bottom:12px"><summary style="font-weight:600;cursor:pointer;font-size:.88rem">Extractors not running?</summary>
<ol style="padding-left:20px;line-height:1.8;color:var(--muted);font-size:.82rem;margin-top:8px">
  <li>Check launchd status: <code>launchctl list | grep memoryos</code></li>
  <li>Check logs: <code>tail -f logs/memoryos.log</code></li>
  <li>Run manually with <code>--dry-run</code> to test</li>
  <li>Verify paths in <code>config/config.yaml</code></li>
</ol>
</details>
<details style="margin-bottom:12px"><summary style="font-weight:600;cursor:pointer;font-size:.88rem">Missing data?</summary>
<ol style="padding-left:20px;line-height:1.8;color:var(--muted);font-size:.82rem;margin-top:8px">
  <li>Check <code>config/state.json</code> for cursor positions</li>
  <li>Run with <code>--reset</code> to reprocess</li>
  <li>Verify data source paths are accessible</li>
</ol>
</details>
<details style="margin-bottom:12px"><summary style="font-weight:600;cursor:pointer;font-size:.88rem">Audio not capturing?</summary>
<ol style="padding-left:20px;line-height:1.8;color:var(--muted);font-size:.82rem;margin-top:8px">
  <li>Verify Screenpipe is running</li>
  <li>Check BlackHole: <code>brew list blackhole-2ch</code></li>
  <li>Verify Audio MIDI Setup has the Multi-Output Device</li>
  <li>Check privacy mode on the Overview tab</li>
</ol>
</details>
<details><summary style="font-weight:600;cursor:pointer;font-size:.88rem">Index empty?</summary>
<ol style="padding-left:20px;line-height:1.8;color:var(--muted);font-size:.82rem;margin-top:8px">
  <li>Run <code>python3 -m src.memory.cli reindex --full</code></li>
  <li>Check vault path: <code>grep obsidian_vault config/config.yaml</code></li>
</ol>
</details>
</div>

</div><!-- /pane-help -->

<!-- ═══ Chat Pane ═══ -->
<div class="tab-pane" id="pane-chat">
<style>
.chat-container { display:flex; flex-direction:column; height:calc(100vh - 120px); max-width:900px; margin:0 auto; }
.chat-messages { flex:1; overflow-y:auto; padding:16px 0; display:flex; flex-direction:column; gap:12px; }
.chat-msg { max-width:85%; padding:12px 16px; border-radius:16px; font-size:.9rem; line-height:1.6; word-wrap:break-word; }
.chat-msg.user { align-self:flex-end; background:var(--accent); color:#fff; border-bottom-right-radius:4px; }
.chat-msg.assistant { align-self:flex-start; background:var(--surface); border:1px solid var(--border); border-bottom-left-radius:4px; }
.chat-msg.assistant pre { background:var(--bg); padding:8px 12px; border-radius:6px; overflow-x:auto; font-size:.82rem; margin:8px 0; }
.chat-msg.assistant code { background:var(--bg); padding:1px 5px; border-radius:3px; font-size:.84rem; }
.chat-msg.assistant pre code { background:none; padding:0; }
.chat-msg.assistant ul,.chat-msg.assistant ol { padding-left:20px; margin:6px 0; }
.chat-msg.assistant h1,.chat-msg.assistant h2,.chat-msg.assistant h3 { margin:10px 0 4px; font-size:1rem; }
.chat-msg.assistant a { color:var(--accent); }
.chat-tool { align-self:flex-start; max-width:85%; font-size:.8rem; padding:8px 14px; border-radius:10px; cursor:pointer; }
.chat-tool.call { background:rgba(139,92,246,.12); color:#a78bfa; border:1px solid rgba(139,92,246,.25); }
.chat-tool.result { background:rgba(139,92,246,.06); color:var(--muted); border:1px solid rgba(139,92,246,.15); }
.chat-tool .tool-output { display:none; margin-top:6px; white-space:pre-wrap; font-size:.78rem; max-height:200px; overflow-y:auto; color:var(--text); }
.chat-tool.expanded .tool-output { display:block; }
.chat-typing { align-self:flex-start; color:var(--muted); font-size:.82rem; padding:8px 16px; }
.chat-typing .dots { display:inline-block; }
.chat-typing .dots::after { content:'...'; animation:dotPulse 1.2s infinite; }
@keyframes dotPulse { 0%{opacity:.2} 50%{opacity:1} 100%{opacity:.2} }
.chat-input-bar { display:flex; gap:8px; padding:12px 0; border-top:1px solid var(--border); }
.chat-input-bar input { flex:1; padding:12px 16px; border-radius:12px; border:1px solid var(--border); background:var(--surface); color:var(--text); font-size:.9rem; outline:none; }
.chat-input-bar input:focus { border-color:var(--accent); }
.chat-input-bar button { padding:12px 24px; border-radius:12px; border:none; background:var(--accent); color:#fff; font-weight:600; font-size:.9rem; cursor:pointer; transition:opacity .15s; }
.chat-input-bar button:hover { opacity:.85; }
.chat-input-bar button:disabled { opacity:.4; cursor:not-allowed; }
.chat-status { text-align:center; padding:6px; font-size:.75rem; color:var(--muted); }
.chat-status.connected { color:var(--green); }
.chat-status.disconnected { color:var(--red); }
@media(max-width:700px) {
  .chat-msg,.chat-tool { max-width:95%; }
  .chat-container { height:calc(100vh - 100px); }
  .chat-input-bar input { font-size:16px; }
}
</style>
<div class="chat-container">
  <div class="chat-status disconnected" id="chat-status">Connecting...</div>
  <div class="chat-messages" id="chat-messages">
    <div class="chat-msg assistant">Welcome to MemoryOS Chat. Ask me anything about your emails, meetings, calendar, or run skills like <strong>morning-brief</strong> or <strong>meeting-prep</strong>.</div>
  </div>
  <div class="chat-input-bar">
    <input type="text" id="chat-input" placeholder="Ask MemoryOS..." autocomplete="off" />
    <button id="chat-send" onclick="chatSend()">Send</button>
  </div>
</div>
</div><!-- /pane-chat -->

<!-- Skills detail modal (markdown fallback) -->
<div class="skill-modal-backdrop" id="skill-modal-backdrop" onclick="closeSkillModal()">
  <div class="skill-modal" onclick="event.stopPropagation()">
    <div class="skill-modal-header">
      <span class="skill-modal-title" id="skill-modal-title"></span>
      <button class="skill-modal-close" onclick="closeSkillModal()">&times;</button>
    </div>
    <div class="skill-modal-body" id="skill-modal-body"></div>
  </div>
</div>

<!-- Rich skill view (full-page overlay for JSON-powered views) -->
<div class="rich-view-overlay" id="rich-view-overlay">
  <div class="rich-view-header">
    <button class="rich-view-back" onclick="closeRichView()">&larr; Back to Skills</button>
    <span class="rich-view-title" id="rich-view-title"></span>
    <span class="rich-view-date" id="rich-view-date"></span>
    <button class="pdf-btn" onclick="downloadPDF()">Download PDF</button>
  </div>
  <div class="rich-view-body" id="rich-view-body"></div>
</div>

<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/d3@7/dist/d3.min.js"></script>

<script>
/* ═══ Globals ═══ */
const R = 30000;
let logTab = 'main', browsePath = '';
let settingsData = {};

/* ═══ Utilities ═══ */
function escHtml(s) { return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
function badge(h) {
  const c = {healthy:'badge-healthy',warning:'badge-warning',error:'badge-error',skipped:'badge-unknown'}[h]||'badge-unknown';
  return `<span class="badge ${c}">${(h||'unknown').toUpperCase()}</span>`;
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
  if(name==='settings') { loadSettings(); loadAgentConfig(); }
  if(name==='knowledge') { loadContext(); loadKnowledgeFiles(); }
  if(name==='skills') { loadSkills(); loadAgentStatus(); }
  if(name==='myday') { loadMyDay(mdSelectedDate||todayStr()); mdStartAutoRefresh(); }
  else { mdStopAutoRefresh(); }
  if(name==='mywork') loadMyWork();
  if(name==='setup') loadSetup();
  if(name==='chat' && !chatConnected) chatConnect();
}

/* ═══════════════════════════════════════════════════════════════════════════
   MY DAY TAB
   ═══════════════════════════════════════════════════════════════════════════ */

let mdSelectedDate = todayStr();
let mdData = null;
let mdChartInstances = {};
let mdAutoRefreshTimer = null;
let mdLastLoadTime = 0;

function mdStartAutoRefresh() {
  mdStopAutoRefresh();
  if(mdSelectedDate === todayStr()) {
    mdAutoRefreshTimer = setInterval(function() {
      loadMyDay(todayStr());
    }, 300000);
  }
}
function mdStopAutoRefresh() {
  if(mdAutoRefreshTimer) { clearInterval(mdAutoRefreshTimer); mdAutoRefreshTimer = null; }
}

const MD_COLORS = {
  meeting: '#8b5cf6', meeting1on1: '#6c7cff', focus: '#4ade80',
  allDay: '#475569', comm: '#38bdf8', research: '#fbbf24', admin: '#94a3b8',
  deep: '#4ade80', other: '#6b7280',
};
const MD_CAT_COLORS = {
  'Deep Work': MD_COLORS.deep, 'Meeting/Communication': MD_COLORS.meeting,
  'Communication': MD_COLORS.comm, 'Research': MD_COLORS.research,
  'Admin': MD_COLORS.admin, 'Other': MD_COLORS.other,
};

function mdDateStr(offset) {
  const d = new Date(); d.setDate(d.getDate() + offset);
  return d.getFullYear()+'-'+String(d.getMonth()+1).padStart(2,'0')+'-'+String(d.getDate()).padStart(2,'0');
}
function mdDayName(ds) {
  return new Date(ds+'T12:00:00').toLocaleDateString('en-US',{weekday:'short'});
}
function mdDayNum(ds) { return ds.split('-')[2]; }
function mdFormatDate(ds) {
  return new Date(ds+'T12:00:00').toLocaleDateString('en-US',{weekday:'long',month:'long',day:'numeric'});
}
function mdTimeToMin(t) {
  if(!t) return 0;
  const [h,m] = t.split(':').map(Number);
  return h*60+(m||0);
}

function renderDatePills() {
  const el = document.getElementById('md-date-pills');
  let html = '';
  const today = todayStr();
  for(let i=-6; i<=0; i++) {
    const ds = mdDateStr(i);
    const isToday = ds === today;
    const isActive = ds === mdSelectedDate;
    html += `<div class="md-date-pill${isActive?' active':''}${isToday?' today':''}" onclick="mdSelectDate('${ds}')">
      <span class="md-pill-day">${mdDayName(ds)}</span>
      <span class="md-pill-num">${mdDayNum(ds)}</span>
    </div>`;
  }
  el.innerHTML = html;
}

function mdSelectDate(ds) {
  mdSelectedDate = ds;
  renderDatePills();
  loadMyDay(ds);
}

async function loadMyDay(date) {
  mdSelectedDate = date || todayStr();
  renderDatePills();
  document.getElementById('md-day-title').textContent = mdFormatDate(mdSelectedDate);

  try {
    const res = await fetch(`/api/myday/${mdSelectedDate}`);
    mdData = await res.json();
  } catch(e) {
    mdData = {calendar:[],tasks:[],activity:{total_active_hours:0,total_context_switches:0,app_breakdown:[],hourly:{},context_switches_per_hour:{}},people:[],work_completed:[],work_in_progress:[],has_data:false};
  }

  renderMdSummaryStats(mdData);
  renderTimeline(mdData);
  renderFocusGauge(mdData);
  renderMeetingCard(mdData);
  renderSwitchCard(mdData);
  renderCompletedCard(mdData);
  renderAppSunburst(mdData);
  renderPeopleChart(mdData);
  renderWorkStatus(mdData);
  renderTaskTimeline(mdData);

  mdLastLoadTime = Date.now();
  if(mdSelectedDate === todayStr()) {
    updateHealthDot();
    mdStartAutoRefresh();
  } else {
    document.getElementById('md-health-dot').className='md-health-indicator';
    document.getElementById('md-health-dot').innerHTML='';
    mdStopAutoRefresh();
  }
}

async function updateHealthDot() {
  const el = document.getElementById('md-health-dot');
  try {
    const res = await fetch('/api/status');
    const d = await res.json();
    const sp = d.extractors?.screenpipe?.health || 'unknown';
    const spFrameId = d.extractors?.screenpipe?.last_frame_id || 0;
    const wdState = d.watchdog_overall || '';

    if(sp === 'healthy' && spFrameId > 0) {
      el.className = 'md-health-indicator live';
      el.innerHTML = '<span class="md-hdot"></span>Live';
    } else if(sp === 'healthy') {
      el.className = 'md-health-indicator partial';
      el.innerHTML = '<span class="md-hdot"></span>Partial';
    } else {
      el.className = 'md-health-indicator offline';
      el.innerHTML = '<span class="md-hdot"></span>Offline';
    }
  } catch(e) {
    el.className = 'md-health-indicator offline';
    el.innerHTML = '<span class="md-hdot"></span>Offline';
  }
}

function mdHasActivity(d) {
  return d.activity && d.activity.app_breakdown && d.activity.app_breakdown.length > 0;
}

function renderMdSummaryStats(d) {
  const el = document.getElementById('md-summary-stats');
  const meetings = d.calendar.filter(e=>!e.is_all_day).length;
  const hasAct = mdHasActivity(d);
  const hours = d.activity.total_active_hours || 0;
  const deepMins = d.activity.app_breakdown.filter(a=>a.category==='Deep Work').reduce((s,a)=>s+a.minutes,0);
  const totalMins = d.activity.app_breakdown.reduce((s,a)=>s+a.minutes,0);
  const focusPct = totalMins > 0 ? Math.round(deepMins/totalMins*100) : 0;
  const topPerson = d.people.length > 0 ? d.people[0].name : '—';
  const completed = d.work_completed.length;
  let html = `<span class="md-stat"><b>${meetings}</b> meetings</span>`;
  if(hasAct) {
    html += `<span class="md-stat"><b>${hours}h</b> active</span>`;
    html += `<span class="md-stat"><b>${focusPct}%</b> focus</span>`;
  } else if(meetings > 0) {
    html += `<span class="md-stat" style="opacity:.5">Calendar only</span>`;
  }
  html += `<span class="md-stat"><b>${completed}</b> completed</span>`;
  if(d.people.length > 0) html += `<span class="md-stat">Top: <b>${escHtml(topPerson)}</b></span>`;
  el.innerHTML = html;
}

/* ─── Timeline (D3 swim-lane) ─── */

function mdAssignSubLanes(events, startHour, endHour) {
  const visible = events.filter(e => !e.is_all_day)
    .map(ev => ({ev, s: mdTimeToMin(ev.start), e: mdTimeToMin(ev.end)}))
    .filter(o => o.e > startHour*60 && o.s < endHour*60)
    .sort((a,b) => a.s - b.s || a.e - b.e);
  const lanes = [];
  visible.forEach(item => {
    let placed = false;
    for(let i = 0; i < lanes.length; i++) {
      if(item.s >= lanes[i]) { lanes[i] = item.e; item.lane = i; placed = true; break; }
    }
    if(!placed) { item.lane = lanes.length; lanes.push(item.e); }
  });
  return { items: visible, maxLanes: Math.max(lanes.length, 1) };
}

function renderTimeline(d) {
  const container = document.getElementById('md-timeline-container');
  container.innerHTML = '';

  if(!d.has_data && d.calendar.length === 0) {
    container.innerHTML = '<div class="md-empty"><div class="md-empty-icon">&#128197;</div>No data for this day</div>';
    return;
  }

  const W = Math.max(container.clientWidth || 900, 700);
  const marginL = 70, marginR = 20;
  const startHour = 6, endHour = 21;
  const subRowH = 26;
  const headerH = 24;
  const laneGap = 8;
  const hasActivity = d.tasks && d.tasks.length > 0;
  const hasCs = d.activity && d.activity.total_context_switches > 0;

  const calResult = mdAssignSubLanes(d.calendar, startHour, endHour);
  const meetingLaneH = Math.max(40, calResult.maxLanes * subRowH + 4);
  const activityLaneH = hasActivity ? 40 : 0;
  const switchLaneH = hasCs ? 36 : 0;

  const totalLanes = 1 + (hasActivity?1:0) + (hasCs?1:0);
  const H = headerH + meetingLaneH + (hasActivity ? laneGap + activityLaneH : 0) + (hasCs ? laneGap + switchLaneH : 0) + 40;

  const svg = d3.select(container).append('svg')
    .attr('width', W).attr('height', H)
    .style('font-family','Inter,system-ui,sans-serif');

  const x = d3.scaleLinear()
    .domain([startHour*60, endHour*60])
    .range([marginL, W - marginR]);

  // Alternating hour bands + grid
  for(let h = startHour; h <= endHour; h++) {
    const xp = x(h*60);
    if(h < endHour && h % 2 === 0) {
      const xn = x((h+1)*60);
      svg.append('rect').attr('x',xp).attr('y',headerH).attr('width',xn-xp).attr('height',H-headerH-30)
        .attr('fill','rgba(255,255,255,.015)');
    }
    svg.append('line').attr('x1',xp).attr('y1',headerH).attr('x2',xp).attr('y2',H-30)
      .attr('stroke','rgba(255,255,255,.06)').attr('stroke-width',1);
    svg.append('text').attr('x',xp).attr('y',headerH-6)
      .attr('text-anchor','middle').attr('fill','#6b7280').attr('font-size','11px').attr('font-weight','500')
      .text(h < 12 ? h+'a' : h === 12 ? '12p' : (h-12)+'p');
  }

  // Tooltip div
  let tooltip = container.querySelector('.md-tl-tooltip');
  if(!tooltip) { tooltip = document.createElement('div'); tooltip.className='md-tl-tooltip'; tooltip.style.display='none'; container.style.position='relative'; container.appendChild(tooltip); }

  // Lane 0: Meetings with sub-lane stacking
  const lane0y = headerH;
  svg.append('rect').attr('x',marginL).attr('y',lane0y)
    .attr('width',W-marginL-marginR).attr('height',meetingLaneH)
    .attr('rx',6).attr('fill','rgba(255,255,255,.02)');
  svg.append('text').attr('x',marginL-10).attr('y',lane0y+meetingLaneH/2+4)
    .attr('text-anchor','end').attr('fill','#6b7280').attr('font-size','10px').attr('font-weight','500').text('Meetings');

  // All-day banner
  const allDay = d.calendar.filter(e=>e.is_all_day);
  if(allDay.length) {
    svg.append('rect').attr('x',marginL).attr('y',lane0y).attr('width',W-marginL-marginR).attr('height',4)
      .attr('rx',2).attr('fill',MD_COLORS.allDay).attr('opacity',.5);
    svg.append('text').attr('x',marginL+6).attr('y',lane0y+3)
      .attr('fill','#94a3b8').attr('font-size','7px').attr('dominant-baseline','middle')
      .text(allDay.map(e=>e.summary).join(' · '));
  }

  calResult.items.forEach(item => {
    const ev = item.ev;
    const x1 = x(Math.max(item.s, startHour*60)), x2 = x(Math.min(item.e, endHour*60));
    const w = Math.max(x2-x1, 6);
    const rowY = lane0y + item.lane * subRowH + (allDay.length ? 5 : 2);
    const rh = subRowH - 4;
    const isSmall = ev.attendees && ev.attendees.length <= 2;
    const col = isSmall ? MD_COLORS.meeting1on1 : MD_COLORS.meeting;
    const g = svg.append('g').attr('class','md-tl-event');
    g.append('rect').attr('x',x1).attr('y',rowY).attr('width',w).attr('height',rh)
      .attr('rx',4).attr('fill',col).attr('opacity',.75);
    if(w > 35) {
      const maxChars = Math.max(Math.floor(w/6), 3);
      g.append('text').attr('x',x1+4).attr('y',rowY+rh/2+3.5)
        .attr('fill','#fff').attr('font-size','10px').attr('font-weight','600')
        .text(ev.summary.length > maxChars ? ev.summary.slice(0,maxChars)+'…' : ev.summary);
    }
    g.on('mouseover', function(event) {
      tooltip.style.display='block';
      tooltip.innerHTML = `<b>${escHtml(ev.summary)}</b><br>${ev.start}–${ev.end}${ev.location?'<br>'+escHtml(ev.location):''}${ev.attendees.length?'<br>'+ev.attendees.map(escHtml).join(', '):''}`;
    }).on('mousemove', function(event) {
      const r = container.getBoundingClientRect();
      tooltip.style.left = (event.clientX-r.left+12)+'px';
      tooltip.style.top = (event.clientY-r.top-10)+'px';
    }).on('mouseout', function() { tooltip.style.display='none'; });
  });

  // Lane 1: Activity blocks (only if data exists)
  let lane1y = lane0y + meetingLaneH + laneGap;
  if(hasActivity) {
    svg.append('rect').attr('x',marginL).attr('y',lane1y)
      .attr('width',W-marginL-marginR).attr('height',activityLaneH)
      .attr('rx',6).attr('fill','rgba(255,255,255,.02)');
    svg.append('text').attr('x',marginL-10).attr('y',lane1y+activityLaneH/2+4)
      .attr('text-anchor','end').attr('fill','#6b7280').attr('font-size','10px').attr('font-weight','500').text('Activity');

    d.tasks.forEach(t => {
      const s = mdTimeToMin(t.start), e2 = mdTimeToMin(t.end);
      if(e2 <= startHour*60 || s >= endHour*60) return;
      const x1 = x(Math.max(s, startHour*60)), x2 = x(Math.min(e2, endHour*60));
      const w = Math.max(x2-x1, 3);
      const cat = t.apps && t.apps.length ? (MD_CAT_COLORS[guessCat(t.apps[0])] || MD_COLORS.other) : MD_COLORS.other;
      const g = svg.append('g').attr('class','md-tl-event');
      g.append('rect').attr('x',x1).attr('y',lane1y+2).attr('width',w).attr('height',activityLaneH-4)
        .attr('rx',5).attr('fill',cat).attr('opacity',.6);
      if(w > 40) {
        g.append('text').attr('x',x1+4).attr('y',lane1y+activityLaneH/2+3)
          .attr('fill','#fff').attr('font-size','9px')
          .text(t.title.length > w/5 ? t.title.slice(0,Math.floor(w/5))+'…' : t.title);
      }
      g.on('mouseover', function(event) {
        tooltip.style.display='block';
        tooltip.innerHTML = `<b>${escHtml(t.title)}</b><br>${t.start}–${t.end}${t.apps.length?'<br>Apps: '+t.apps.map(escHtml).join(', '):''}`;
      }).on('mousemove', function(event) {
        const r = container.getBoundingClientRect();
        tooltip.style.left = (event.clientX-r.left+12)+'px';
        tooltip.style.top = (event.clientY-r.top-10)+'px';
      }).on('mouseout', function() { tooltip.style.display='none'; });
    });
    lane1y += activityLaneH + laneGap;
  }

  // Lane 2: Context switches sparkline (only if data exists)
  if(hasCs) {
    const lane2y = lane1y;
    svg.append('rect').attr('x',marginL).attr('y',lane2y)
      .attr('width',W-marginL-marginR).attr('height',switchLaneH)
      .attr('rx',6).attr('fill','rgba(255,255,255,.02)');
    svg.append('text').attr('x',marginL-10).attr('y',lane2y+switchLaneH/2+4)
      .attr('text-anchor','end').attr('fill','#6b7280').attr('font-size','10px').attr('font-weight','500').text('Switches');

    const csData = d.activity.context_switches_per_hour || {};
    const csEntries = [];
    for(let h = startHour; h < endHour; h++) {
      const key = String(h).padStart(2,'0')+':00';
      csEntries.push({hour: h, val: csData[key]||0});
    }
    const csMax = Math.max(d3.max(csEntries, e=>e.val)||1, 1);
    const csY = d3.scaleLinear().domain([0, csMax]).range([lane2y+switchLaneH-4, lane2y+4]);
    const area = d3.area().x(e => x(e.hour*60+30)).y0(lane2y+switchLaneH-2).y1(e => csY(e.val)).curve(d3.curveBasis);
    svg.append('path').datum(csEntries).attr('d',area)
      .attr('fill','rgba(250,204,21,.15)').attr('stroke','rgba(250,204,21,.5)').attr('stroke-width',1.5);
  }

  // Now marker (only for today)
  if(mdSelectedDate === todayStr()) {
    const now = new Date();
    const nowMin = now.getHours()*60+now.getMinutes();
    if(nowMin >= startHour*60 && nowMin <= endHour*60) {
      const nx = x(nowMin);
      svg.append('line').attr('class','md-now-line')
        .attr('x1',nx).attr('y1',headerH-2).attr('x2',nx).attr('y2',H-30)
        .attr('stroke','#ef4444').attr('stroke-width',2).attr('stroke-dasharray','4,3');
      svg.append('circle').attr('class','md-now-line')
        .attr('cx',nx).attr('cy',headerH-2).attr('r',4).attr('fill','#ef4444');
    }
  }

  // Legend
  const legendY = H - 22;
  const legendItems = [{label:'Meeting',color:MD_COLORS.meeting},{label:'1:1',color:MD_COLORS.meeting1on1},{label:'Deep Work',color:MD_COLORS.deep},{label:'Comms',color:MD_COLORS.comm},{label:'Research',color:MD_COLORS.research}];
  let lx = marginL;
  legendItems.forEach(li => {
    svg.append('rect').attr('x',lx).attr('y',legendY).attr('width',8).attr('height',8).attr('rx',2).attr('fill',li.color).attr('opacity',.7);
    svg.append('text').attr('x',lx+12).attr('y',legendY+7).attr('fill','#6b7280').attr('font-size','9px').text(li.label);
    lx += li.label.length * 6 + 24;
  });
}

function guessCat(appName) {
  const cats = {'Microsoft Teams':'Meeting/Communication','Zoom':'Meeting/Communication','Google Meet':'Meeting/Communication','Microsoft Outlook':'Communication','Mail':'Communication','Slack':'Communication','Cursor':'Deep Work','VS Code':'Deep Work','Terminal':'Deep Work','Microsoft Word':'Deep Work','Microsoft PowerPoint':'Deep Work','Microsoft Excel':'Deep Work','Obsidian':'Deep Work','Google Chrome':'Research','Safari':'Research','Firefox':'Research','Arc':'Research','ChatGPT':'Research','Claude':'Research','Finder':'Admin','System Settings':'Admin','Calendar':'Admin'};
  return cats[appName]||'Other';
}

/* ─── Focus Gauge (D3 arc) ─── */

function renderFocusGauge(d) {
  const el = document.getElementById('md-focus-gauge');
  el.innerHTML = '';
  if(!mdHasActivity(d)) {
    el.innerHTML = '<div class="md-metric-value" style="opacity:.25">--</div><div class="md-metric-sub">no activity data</div>';
    return;
  }
  const deepMins = d.activity.app_breakdown.filter(a=>a.category==='Deep Work').reduce((s,a)=>s+a.minutes,0);
  const totalMins = d.activity.app_breakdown.reduce((s,a)=>s+a.minutes,0);
  const pct = totalMins > 0 ? deepMins/totalMins : 0;
  const size = 120, stroke = 12;

  const svg = d3.select(el).append('svg').attr('width',size).attr('height',size)
    .style('display','block').style('margin','0 auto');
  const g = svg.append('g').attr('transform',`translate(${size/2},${size/2})`);
  const r = size/2 - stroke;
  const arc = d3.arc().innerRadius(r-stroke/2).outerRadius(r+stroke/2).startAngle(0).cornerRadius(6);

  g.append('path').attr('d', arc({endAngle: Math.PI*2}))
    .attr('fill','rgba(255,255,255,.06)');

  const fg = g.append('path')
    .attr('fill', pct > .4 ? '#4ade80' : pct > .2 ? '#facc15' : '#f87171');
  fg.transition().duration(1000).ease(d3.easeCubicOut)
    .attrTween('d', function() {
      const interp = d3.interpolate(0, pct * Math.PI * 2);
      return function(t) { return arc({endAngle: interp(t)}); };
    });

  g.append('text').attr('text-anchor','middle').attr('dy','-.1em')
    .attr('fill','var(--text)').attr('font-size','1.5rem').attr('font-weight','800')
    .text(Math.round(pct*100)+'%');
  g.append('text').attr('text-anchor','middle').attr('dy','1.2em')
    .attr('fill','var(--muted)').attr('font-size','.65rem')
    .text('deep work');
}

/* ─── Meeting Card ─── */

function renderMeetingCard(d) {
  const el = document.getElementById('md-meetings-content');
  const meetings = d.calendar.filter(e=>!e.is_all_day);
  const count = meetings.length;
  let totalMin = 0;
  const durations = [];
  meetings.forEach(m => {
    const dur = mdTimeToMin(m.end) - mdTimeToMin(m.start);
    totalMin += dur;
    durations.push({name: m.summary.slice(0,20), dur});
  });
  const hours = (totalMin/60).toFixed(1);

  el.innerHTML = `
    <div class="md-metric-value">${count}</div>
    <div class="md-metric-sub">${hours}h in meetings</div>
    <div style="height:60px;position:relative"><canvas id="md-meeting-bars"></canvas></div>
  `;

  if(durations.length) {
    const ctx = document.getElementById('md-meeting-bars');
    if(mdChartInstances.meetingBars) mdChartInstances.meetingBars.destroy();
    mdChartInstances.meetingBars = new Chart(ctx, {
      type:'bar',
      data:{
        labels: durations.map(d=>d.name),
        datasets:[{data:durations.map(d=>d.dur), backgroundColor:'rgba(139,92,246,.5)', borderRadius:4}]
      },
      options:{
        responsive:true, maintainAspectRatio:false,
        plugins:{legend:{display:false}},
        scales:{
          x:{display:false},
          y:{display:false}
        }
      }
    });
  }
}

/* ─── Context Switches Card ─── */

function renderSwitchCard(d) {
  const el = document.getElementById('md-switches-content');
  if(!mdHasActivity(d)) {
    el.innerHTML = '<div class="md-metric-value" style="opacity:.25">--</div><div class="md-metric-sub">no activity data</div>';
    return;
  }
  const total = d.activity.total_context_switches || 0;
  const csData = d.activity.context_switches_per_hour || {};
  const color = total < 30 ? 'var(--green)' : total < 60 ? 'var(--yellow)' : '#f87171';

  el.innerHTML = `
    <div class="md-metric-value" style="background:none;-webkit-text-fill-color:${color};color:${color}">${total}</div>
    <div class="md-metric-sub">total switches</div>
    <div style="height:50px;position:relative"><canvas id="md-switch-spark"></canvas></div>
  `;

  const labels = [], vals = [];
  for(let h=6; h<21; h++) {
    labels.push(h+'');
    const key = String(h).padStart(2,'0')+':00';
    vals.push(csData[key]||0);
  }
  const ctx = document.getElementById('md-switch-spark');
  if(mdChartInstances.switchSpark) mdChartInstances.switchSpark.destroy();
  mdChartInstances.switchSpark = new Chart(ctx, {
    type:'line',
    data:{
      labels,
      datasets:[{
        data:vals, fill:true,
        borderColor:'rgba(250,204,21,.6)', backgroundColor:'rgba(250,204,21,.08)',
        borderWidth:2, pointRadius:0, tension:.4,
      }]
    },
    options:{
      responsive:true, maintainAspectRatio:false,
      plugins:{legend:{display:false}},
      scales:{x:{display:false},y:{display:false}}
    }
  });
}

/* ─── Work Completed Card ─── */

function renderCompletedCard(d) {
  const el = document.getElementById('md-completed-content');
  const count = d.work_completed.length;
  if(count === 0) {
    el.innerHTML = `<div class="md-metric-value" style="opacity:.3">0</div><div class="md-metric-sub">No completions</div>`;
    return;
  }
  let html = `<div class="md-metric-value">${count}</div><div class="md-metric-sub">tasks shipped</div><div style="margin-top:8px;text-align:left">`;
  d.work_completed.slice(0,4).forEach(w => {
    html += `<div style="font-size:.72rem;color:var(--green);padding:2px 0">&#10003; ${escHtml(w.task.slice(0,40))}${w.task.length>40?'…':''}</div>`;
  });
  if(count > 4) html += `<div style="font-size:.68rem;color:var(--muted)">+${count-4} more</div>`;
  html += '</div>';
  el.innerHTML = html;
}

/* ─── App Sunburst (D3) ─── */

function renderAppSunburst(d) {
  const container = document.getElementById('md-sunburst-container');
  container.innerHTML = '';
  const apps = d.activity.app_breakdown || [];
  if(!apps.length) {
    container.innerHTML = '<div class="md-empty"><div class="md-empty-icon">&#128187;</div>No app data</div>';
    return;
  }

  const totalMins = apps.reduce((s,a)=>s+a.minutes,0);
  const catGroups = {};
  apps.forEach(a => {
    if(!catGroups[a.category]) catGroups[a.category] = {cat:a.category, apps:[], total:0};
    catGroups[a.category].apps.push(a);
    catGroups[a.category].total += a.minutes;
  });
  const sorted = Object.values(catGroups).sort((a,b)=>b.total-a.total);

  let html = '<div style="margin-bottom:12px;display:flex;align-items:baseline;gap:10px">';
  html += `<span style="font-size:1.8rem;font-weight:800;background:linear-gradient(135deg,var(--accent),#a78bfa);-webkit-background-clip:text;-webkit-text-fill-color:transparent">${totalMins}m</span>`;
  html += '<span style="font-size:.78rem;color:var(--muted)">total tracked</span></div>';

  sorted.forEach(group => {
    const col = MD_CAT_COLORS[group.cat] || MD_COLORS.other;
    const pct = Math.round(group.total/totalMins*100);
    html += `<div style="margin-bottom:14px">`;
    html += `<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">`;
    html += `<span style="width:10px;height:10px;border-radius:3px;background:${col};flex-shrink:0"></span>`;
    html += `<span style="font-size:.8rem;font-weight:600;color:var(--text)">${escHtml(group.cat)}</span>`;
    html += `<span style="font-size:.72rem;color:var(--muted);margin-left:auto">${group.total}m · ${pct}%</span>`;
    html += `</div>`;
    group.apps.sort((a,b)=>b.minutes-a.minutes).forEach(app => {
      const barPct = Math.max(app.minutes/sorted[0].apps[0].minutes*100, 3);
      html += `<div class="md-app-bar-row">`;
      html += `<span class="md-app-bar-name">${escHtml(app.name)}</span>`;
      html += `<div class="md-app-bar-track"><div class="md-app-bar-fill" style="width:${barPct}%;background:${col}"></div></div>`;
      html += `<span class="md-app-bar-val">${app.minutes}m</span>`;
      html += `</div>`;
    });
    html += `</div>`;
  });

  container.innerHTML = html;
  container.style.display = 'block';
}

/* ─── People Chart ─── */

function renderPeopleChart(d) {
  const container = document.getElementById('md-people-container');
  container.innerHTML = '';
  const people = (d.people||[]).slice(0,10);
  if(!people.length) {
    container.innerHTML = '<div class="md-empty"><div class="md-empty-icon">&#128101;</div>No interactions</div>';
    return;
  }
  const maxMin = Math.max(...people.map(p=>p.total_minutes),1);
  let html = '';
  people.forEach(p => {
    const pct = Math.max(p.total_minutes/maxMin*100, 4);
    html += `<div class="md-people-bar-row">
      <div class="md-people-name" title="${escHtml(p.name)}">${escHtml(p.name)}</div>
      <div class="md-people-bar-wrap"><div class="md-people-bar" style="width:${pct}%"></div></div>
      <div class="md-people-meta">${p.meetings} mtg · ${Math.round(p.total_minutes)}m</div>
    </div>`;
  });
  container.innerHTML = html;
}

/* ─── Work Status ─── */

function renderWorkStatus(d) {
  const el = document.getElementById('md-work-status');
  const completed = d.work_completed || [];
  const wip = d.work_in_progress || [];
  if(!completed.length && !wip.length) {
    el.innerHTML = '<div class="md-empty"><div class="md-empty-icon">&#128203;</div>No work items</div>';
    return;
  }
  let html = '';
  completed.forEach(w => {
    const priClass = (w.priority||'P1').toLowerCase();
    html += `<div class="md-work-item">
      <div class="md-work-icon done">&#10003;</div>
      <div>
        <div class="md-work-task done-text">${escHtml(w.task)}<span class="md-pri-pill ${priClass}">${w.priority||'P1'}</span></div>
        <div class="md-project-tag">${escHtml(w.project||'')}</div>
      </div>
    </div>`;
  });
  wip.forEach(w => {
    const priClass = (w.priority||'P1').toLowerCase();
    html += `<div class="md-work-item">
      <div class="md-work-icon wip">&#9654;</div>
      <div>
        <div class="md-work-task">${escHtml(w.task)}<span class="md-pri-pill ${priClass}">${w.priority||'P1'}</span></div>
        <div class="md-project-tag">${escHtml(w.project||'')}${w.notes?' · '+escHtml(w.notes):''}</div>
      </div>
    </div>`;
  });
  el.innerHTML = html;
}

/* ─── Task Timeline (recall blocks) ─── */

function renderTaskTimeline(d) {
  const el = document.getElementById('md-task-timeline');
  const tasks = d.tasks || [];
  if(!tasks.length) {
    el.innerHTML = '<div class="md-empty"><div class="md-empty-icon">&#128336;</div>No activity data</div>';
    return;
  }
  let html = '';
  tasks.forEach(t => {
    html += `<div class="md-tl-item">
      <div class="md-tl-time">${t.start}–${t.end}</div>
      <div class="md-tl-dot"></div>
      <div class="md-tl-body">
        <div class="md-tl-title">${escHtml(t.title)}</div>
        ${t.apps.length ? '<div class="md-tl-apps">'+t.apps.map(a=>'<span class="md-tl-app">'+escHtml(a)+'</span>').join('')+'</div>' : ''}
        ${t.details.length ? '<div class="md-tl-detail">'+t.details.slice(0,3).map(escHtml).join('<br>')+'</div>' : ''}
      </div>
    </div>`;
  });
  el.innerHTML = html;
}

/* ═══ My Work (table + raw editors) ═══ */
let _mwData = null;
let _mwView = 'table';
let _mwRawLoaded = false;

async function loadMyWork() {
  if (_mwView === 'table') {
    await loadTaskTable();
  } else {
    await loadRawEditors();
  }
}

function toggleMyWorkView(mode, btn) {
  _mwView = mode;
  document.querySelectorAll('.mw-toggle button').forEach(b=>b.classList.remove('active'));
  if(btn) btn.classList.add('active');
  document.getElementById('mw-table-view').style.display = mode==='table' ? '' : 'none';
  document.getElementById('mw-raw-view').style.display = mode==='raw' ? '' : 'none';
  if(mode==='table') { _mwData=null; loadTaskTable(); }
  else { _mwRawLoaded=false; loadRawEditors(); }
}

async function loadTaskTable() {
  const el = document.getElementById('mw-task-rows');
  el.innerHTML='<div style="padding:16px;color:var(--muted)">Loading...</div>';
  try {
    _mwData = await(await fetch('/api/tasks')).json();
    renderTaskTable();
    populateProjectFilter();
    updateTaskCounts();
  } catch(e) { el.innerHTML='<div style="padding:16px;color:var(--red)">Failed to load tasks</div>'; }
}

function renderTaskTable() {
  if(!_mwData) return;
  const el = document.getElementById('mw-task-rows');
  const fSt = document.getElementById('mw-flt-status').value;
  const fPr = document.getElementById('mw-flt-pri').value;
  const fPj = document.getElementById('mw-flt-proj').value;
  const hideDone = document.getElementById('mw-hide-done').checked;
  let h='';
  const statusOpts = '<option value="not_started">Not Started</option><option value="in_progress">In Progress</option><option value="complete">Complete</option><option value="waiting">Waiting</option><option value="blocked">Blocked</option>';
  const priOpts = '<option value="P0">P0</option><option value="P1">P1</option><option value="P2">P2</option>';
  const allProjects = (_mwData.project_names||[]);

  for(let pi=0;pi<_mwData.projects.length;pi++){
    const proj=_mwData.projects[pi];
    if(fPj && proj.name!==fPj) continue;
    const tasks = proj.tasks||[];
    const filtered = tasks.filter(t=>{
      if(hideDone && t.status==='complete') return false;
      if(fSt && t.status!==fSt) return false;
      if(fPr && t.priority!==fPr) return false;
      return true;
    });
    if(fSt || fPr || hideDone) { if(!filtered.length) continue; }
    let cnt = tasks.length;
    let done = tasks.filter(t=>t.status==='complete').length;
    for(const t of tasks){ const subs=t.subtasks||[]; cnt+=subs.length; done+=subs.filter(s=>s.status==='complete').length; }
    h+=`<div class="task-proj-hdr" onclick="toggleProject(${pi})">
      <span class="proj-toggle" id="proj-tog-${pi}">&#9660;</span>
      <span class="proj-name" id="proj-name-${pi}">${escHtml(proj.name)}</span>
      <button class="proj-edit-btn" onclick="event.stopPropagation();editProjectName(${pi})" title="Rename project">&#9998;</button>
      <span class="proj-meta"><span class="proj-owner-lbl" onclick="event.stopPropagation();editProjectOwner(${pi})" title="Click to edit owner">${proj.owner?escHtml(proj.owner):'(no owner)'}</span> &middot; ${done}/${cnt} done</span>
      <button class="btn btn-sm" style="margin-left:8px;font-size:.72rem" onclick="event.stopPropagation();addTask('${escHtml(proj.name)}')">+ Task</button>
    </div>`;
    h+=`<div id="proj-body-${pi}">`;
    const renderList = (fSt||fPr) ? filtered : tasks;
    for(let ti=0;ti<renderList.length;ti++){
      const t=renderList[ti];
      h+=taskRowHtml(pi,ti,t,allProjects,statusOpts,priOpts);
      const subs=t.subtasks||[];
      for(let si=0;si<subs.length;si++){
        if(hideDone && subs[si].status==='complete') continue;
        h+=subTaskRowHtml(pi,ti,si,subs[si],statusOpts,priOpts);
      }
    }
    h+=`</div>`;
  }
  if(_mwData.waiting && _mwData.waiting.length){
    h+=`<div class="task-proj-hdr"><span class="proj-name">Waiting On Others</span><span class="proj-meta">${_mwData.waiting.length} items</span></div>`;
    for(const w of _mwData.waiting){
      h+=`<div class="task-row" style="color:var(--orange)"><span>${escHtml(w.what)}</span><span>${escHtml(w.who)}</span><span></span><span>Waiting</span><span>${escHtml(w.since)}</span><span>${escHtml(w.followup)}</span><span></span></div>`;
    }
  }
  el.innerHTML=h||'<div style="padding:16px;color:var(--muted)">No tasks found. Click + Add Task to get started.</div>';
}

function taskRowHtml(pi,ti,t,allProjects,statusOpts,priOpts){
  const stCls='st-'+t.status;
  const prCls='pri-'+t.priority.toLowerCase();
  const isOverdue = t.due && t.status!=='complete' && isDatePast(t.due);
  return `<div class="task-row">
    <span data-label="Task"><input class="cfg-input" value="${escAttr(t.task)}" onchange="updTask(${pi},${ti},'task',this.value)"></span>
    <span data-label="Owner"><input class="cfg-input" value="${escAttr(t.owner||'')}" onchange="updTask(${pi},${ti},'owner',this.value)" style="font-size:.74rem"></span>
    <span data-label="Priority"><select class="cfg-input ${prCls}" onchange="updTask(${pi},${ti},'priority',this.value);this.className='cfg-input pri-'+this.value.toLowerCase()">${priOpts.replace('value="'+t.priority+'"','value="'+t.priority+'" selected')}</select></span>
    <span data-label="Status"><select class="cfg-input ${stCls}" onchange="updTask(${pi},${ti},'status',this.value);this.className='cfg-input st-'+this.value">${statusOpts.replace('value="'+t.status+'"','value="'+t.status+'" selected')}</select></span>
    <span data-label="Due"><input class="cfg-input${isOverdue?' overdue':''}" value="${escAttr(t.due)}" placeholder="e.g. Mar 1" onchange="updTask(${pi},${ti},'due',this.value)" style="font-size:.74rem"></span>
    <span data-label="Notes"><input class="cfg-input" value="${escAttr(t.notes)}" placeholder="// progress notes" onchange="updTask(${pi},${ti},'notes',this.value)" style="font-size:.74rem;color:var(--muted)"></span>
    <span class="task-actions"><button class="task-add-sub" onclick="addSubTask(${pi},${ti})" title="Add sub-task">+sub</button><button class="task-del" onclick="deleteTask(${pi},${ti})" title="Delete">&times;</button></span>
  </div>`;
}

function subTaskRowHtml(pi,ti,si,t,statusOpts,priOpts){
  const stCls='st-'+t.status;
  const prCls='pri-'+t.priority.toLowerCase();
  const isOverdue = t.due && t.status!=='complete' && isDatePast(t.due);
  return `<div class="task-row subtask">
    <span data-label="Task"><input class="cfg-input" value="${escAttr(t.task)}" onchange="updSubTask(${pi},${ti},${si},'task',this.value)"></span>
    <span data-label="Owner"><input class="cfg-input" value="${escAttr(t.owner||'')}" onchange="updSubTask(${pi},${ti},${si},'owner',this.value)" style="font-size:.74rem"></span>
    <span data-label="Priority"><select class="cfg-input ${prCls}" onchange="updSubTask(${pi},${ti},${si},'priority',this.value);this.className='cfg-input pri-'+this.value.toLowerCase()">${priOpts.replace('value="'+t.priority+'"','value="'+t.priority+'" selected')}</select></span>
    <span data-label="Status"><select class="cfg-input ${stCls}" onchange="updSubTask(${pi},${ti},${si},'status',this.value);this.className='cfg-input st-'+this.value">${statusOpts.replace('value="'+t.status+'"','value="'+t.status+'" selected')}</select></span>
    <span data-label="Due"><input class="cfg-input${isOverdue?' overdue':''}" value="${escAttr(t.due)}" placeholder="e.g. Mar 1" onchange="updSubTask(${pi},${ti},${si},'due',this.value)" style="font-size:.74rem"></span>
    <span data-label="Notes"><input class="cfg-input" value="${escAttr(t.notes)}" placeholder="// progress notes" onchange="updSubTask(${pi},${ti},${si},'notes',this.value)" style="font-size:.74rem;color:var(--muted)"></span>
    <span class="task-actions"><button class="task-del" onclick="deleteSubTask(${pi},${ti},${si})" title="Delete">&times;</button></span>
  </div>`;
}

function escAttr(s){return (s||'').replace(/&/g,'&amp;').replace(/"/g,'&quot;').replace(/</g,'&lt;');}

function isDatePast(d){
  if(!d) return false;
  try {
    const now=new Date(); now.setHours(0,0,0,0);
    const parts=d.match(/([A-Za-z]+)\s+(\d+)/);
    if(!parts) return false;
    const months={jan:0,feb:1,mar:2,apr:3,may:4,jun:5,jul:6,aug:7,sep:8,oct:9,nov:10,dec:11};
    const m=months[parts[1].toLowerCase().slice(0,3)];
    if(m===undefined) return false;
    const dt=new Date(now.getFullYear(),m,parseInt(parts[2]));
    return dt<now;
  } catch(e){return false;}
}

function updTask(pi,ti,field,val){
  if(!_mwData) return;
  const t=_mwData.projects[pi].tasks[ti];
  if(!t) return;
  t[field]=val;
  if(field==='status' && val==='complete' && !t.due){
    const now=new Date();
    const mon=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
    t.due=mon[now.getMonth()]+' '+now.getDate();
  }
  updateTaskCounts();
}

function addNewProject(){
  if(!_mwData) return;
  const name=prompt('New project/category name:');
  if(!name||!name.trim()) return;
  _mwData.projects.push({name:name.trim(),owner:'',goal:'',priority_status:'',tasks:[]});
  renderTaskTable();
  populateProjectFilter();
  updateTaskCounts();
}

function editProjectName(pi){
  if(!_mwData||!_mwData.projects[pi]) return;
  const cur=_mwData.projects[pi].name;
  const val=prompt('Rename project:',cur);
  if(val===null||!val.trim()||val.trim()===cur) return;
  _mwData.projects[pi].name=val.trim();
  renderTaskTable();
  populateProjectFilter();
}

function editProjectOwner(pi){
  if(!_mwData||!_mwData.projects[pi]) return;
  const cur=_mwData.projects[pi].owner||'';
  const val=prompt('Project owner:',cur);
  if(val===null) return;
  _mwData.projects[pi].owner=val.trim();
  renderTaskTable();
}

function addTask(projName){
  if(!_mwData) return;
  let pi=-1;
  if(projName){
    pi=_mwData.projects.findIndex(p=>p.name===projName);
  }
  if(pi<0){
    if(_mwData.projects.length) pi=0;
    else { _mwData.projects.push({name:'General',owner:'',goal:'',priority_status:'',tasks:[]}); pi=0; }
  }
  const newId='t'+(Date.now()%100000);
  _mwData.projects[pi].tasks.push({id:newId,task:'',status:'not_started',priority:'P1',due:'',owner:_mwData.projects[pi].owner||'',notes:'',section:'active',subtasks:[]});
  renderTaskTable();
  updateTaskCounts();
  const rows=document.querySelectorAll('#proj-body-'+pi+' .task-row:not(.subtask)');
  if(rows.length){const last=rows[rows.length-1];const inp=last.querySelector('input');if(inp)inp.focus();}
}

function addSubTask(pi,ti){
  if(!_mwData) return;
  const parent=_mwData.projects[pi].tasks[ti];
  if(!parent) return;
  if(!parent.subtasks) parent.subtasks=[];
  const newId='t'+(Date.now()%100000);
  parent.subtasks.push({id:newId,task:'',status:'not_started',priority:parent.priority||'P1',due:'',owner:parent.owner||'',notes:'',section:'active'});
  renderTaskTable();
  updateTaskCounts();
  const body=document.getElementById('proj-body-'+pi);
  if(body){const subs=body.querySelectorAll('.task-row.subtask');if(subs.length){const last=subs[subs.length-1];const inp=last.querySelector('input');if(inp)inp.focus();}}
}

function updSubTask(pi,ti,si,field,val){
  if(!_mwData) return;
  const parent=_mwData.projects[pi].tasks[ti];
  if(!parent||!parent.subtasks) return;
  const t=parent.subtasks[si];
  if(!t) return;
  t[field]=val;
  if(field==='status' && val==='complete' && !t.due){
    const now=new Date();
    const mon=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
    t.due=mon[now.getMonth()]+' '+now.getDate();
  }
  updateTaskCounts();
}

function deleteSubTask(pi,ti,si){
  if(!_mwData) return;
  const parent=_mwData.projects[pi].tasks[ti];
  if(!parent||!parent.subtasks) return;
  const t=parent.subtasks[si];
  if(t && t.task && !confirm('Delete sub-task "'+t.task+'"?')) return;
  parent.subtasks.splice(si,1);
  renderTaskTable();
  updateTaskCounts();
}

function deleteTask(pi,ti){
  if(!_mwData) return;
  const t=_mwData.projects[pi].tasks[ti];
  if(t && t.task && !confirm('Delete "'+t.task+'"?')) return;
  _mwData.projects[pi].tasks.splice(ti,1);
  renderTaskTable();
  updateTaskCounts();
}

function toggleProject(pi){
  const body=document.getElementById('proj-body-'+pi);
  const tog=document.getElementById('proj-tog-'+pi);
  if(!body) return;
  if(body.style.display==='none'){body.style.display='';if(tog)tog.classList.remove('collapsed');}
  else{body.style.display='none';if(tog)tog.classList.add('collapsed');}
}

function populateProjectFilter(){
  const sel=document.getElementById('mw-flt-proj');
  if(!sel||!_mwData) return;
  sel.innerHTML='<option value="">All Projects</option>';
  for(const p of _mwData.projects){
    sel.innerHTML+=`<option value="${escAttr(p.name)}">${escHtml(p.name)}</option>`;
  }
}

function filterTasks(){ renderTaskTable(); }

function updateTaskCounts(){
  if(!_mwData) return;
  let open=0,prog=0,done=0;
  function countTask(t){
    if(t.status==='not_started') open++;
    else if(t.status==='in_progress') prog++;
    else if(t.status==='complete') done++;
  }
  for(const p of _mwData.projects) for(const t of p.tasks||[]){
    countTask(t);
    for(const s of t.subtasks||[]) countTask(s);
  }
  const ce=document.getElementById('mw-cnt-open');
  const pe=document.getElementById('mw-cnt-prog');
  const de=document.getElementById('mw-cnt-done');
  if(ce) ce.textContent=open+' open';
  if(pe) pe.textContent=prog+' in progress';
  if(de) de.textContent=done+' complete';
}

async function saveTaskTable(){
  if(!_mwData) return;
  const msg=document.getElementById('mw-tbl-msg');
  if(msg){msg.textContent='Saving...';msg.style.color='var(--muted)';}
  try{
    const r=await fetch('/api/tasks',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(_mwData)});
    if(r.ok){if(msg){msg.textContent='Saved!';msg.style.color='var(--green)';} _mwRawLoaded=false;}
    else{const d=await r.json();if(msg){msg.textContent=d.error||'Failed';msg.style.color='var(--red)';}}
  }catch(e){if(msg){msg.textContent='Error';msg.style.color='var(--red)';}}
  setTimeout(()=>{if(msg) msg.textContent='';},3000);
}

async function loadRawEditors(){
  if(_mwRawLoaded) return;
  try{
    const [pri,tsk]=await Promise.all([
      fetch('/api/context/file?name=priorities.md').then(r=>r.json()),
      fetch('/api/context/file?name=tasks.md').then(r=>r.json()),
    ]);
    document.getElementById('mw-priorities').value=pri.content||'';
    document.getElementById('mw-tasks').value=tsk.content||'';
    _mwRawLoaded=true;
  }catch(e){
    document.getElementById('mw-priorities').value='# Failed to load';
    document.getElementById('mw-tasks').value='# Failed to load';
  }
}

async function saveContextFile(name){
  const isP=name==='priorities.md';
  const ta=document.getElementById(isP?'mw-priorities':'mw-tasks');
  const msg=document.getElementById(isP?'mw-pri-msg':'mw-task-msg');
  if(msg){msg.textContent='Saving...';msg.style.color='var(--muted)';}
  try{
    const r=await fetch('/api/context/file',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name,content:ta.value})});
    if(r.ok){if(msg){msg.textContent='Saved!';msg.style.color='var(--green)';} _mwData=null;}
    else{if(msg){msg.textContent='Failed';msg.style.color='var(--red)';}}
  }catch(e){if(msg){msg.textContent='Error';msg.style.color='var(--red)';}}
  setTimeout(()=>{if(msg) msg.textContent='';},3000);
}

/* ═══ Setup Wizard ═══ */
let setupData = null;

function wizGo(step) {
  document.querySelectorAll('.wiz-panel').forEach(p=>p.style.display='none');
  document.querySelectorAll('.wiz-step').forEach(s=>s.classList.remove('active'));
  const panel = document.getElementById('wiz-'+step);
  if(panel) panel.style.display='block';
  const steps = document.querySelectorAll('.wiz-step');
  if(steps[step-1]) steps[step-1].classList.add('active');
  if(step===1) loadSetup();
}

async function loadSetup() {
  try {
    setupData = await(await fetch('/api/setup/status')).json();
    renderSetupChecks(setupData);
    renderSuggestions(setupData.suggested_paths);
  } catch(e) { document.getElementById('setup-checks').textContent='Failed to load setup status'; }
}

function renderSetupChecks(data) {
  const el = document.getElementById('setup-checks');
  const checks = data.checks;
  let h='';
  const icons = {ok:'\u2705', missing:'\u274c', warning:'\u26a0\ufe0f'};
  const order = ['python','homebrew','screenpipe_installed','screenpipe_running','obsidian_vault','config_exists','pandoc','blackhole','launchd_installed','index_built'];
  for(const key of order) {
    const c = checks[key];
    if(!c) continue;
    h+=`<div class="setup-row">
      <span class="setup-icon">${icons[c.status]||'\u2753'}</span>
      <div class="setup-detail">
        <div>${escHtml(c.detail)}</div>
        ${c.fix?`<div class="setup-fix">${escHtml(c.fix)}</div>`:''}
      </div>
    </div>`;
  }
  el.innerHTML=h;
}

function renderSuggestions(paths) {
  const vaultEl = document.getElementById('cfg-vault-suggestions');
  const odEl = document.getElementById('cfg-od-suggestions');
  if(vaultEl && paths.obsidian_vaults) {
    vaultEl.innerHTML = paths.obsidian_vaults.map(p=>`<span class="chip" onclick="document.getElementById('cfg-vault').value='${escHtml(p)}'">${escHtml(p)}</span>`).join('');
  }
  if(odEl && paths.onedrive_dirs) {
    odEl.innerHTML = paths.onedrive_dirs.map(p=>`<span class="chip" onclick="document.getElementById('cfg-onedrive').value='${escHtml(p)}'">${escHtml(p)}</span>`).join('');
  }
}

async function saveSetupConfig() {
  const btn = document.getElementById('cfg-save-btn');
  const msg = document.getElementById('cfg-save-msg');
  btn.disabled=true; msg.textContent='Saving...'; msg.style.color='var(--muted)';
  const body = {
    obsidian_vault: document.getElementById('cfg-vault').value,
    email_source: document.querySelector('input[name="email-src"]:checked')?.value||'mail_app',
    calendar_source: document.querySelector('input[name="cal-src"]:checked')?.value||'calendar_app',
    graph_client_id: document.getElementById('cfg-graph-id').value,
    onedrive_dir: document.getElementById('cfg-onedrive').value,
  };
  try {
    const r = await fetch('/api/setup/config',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const d = await r.json();
    if(r.ok) { msg.textContent='Configuration saved!'; msg.style.color='var(--green)'; }
    else { msg.textContent=d.error||'Save failed'; msg.style.color='var(--red)'; }
  } catch(e) { msg.textContent='Network error'; msg.style.color='var(--red)'; }
  btn.disabled=false;
}

async function setupInstallAgents(btn) {
  btn.disabled=true;
  const msg = document.getElementById('act-agents');
  msg.textContent='Installing...'; msg.style.color='var(--muted)';
  try {
    const r = await fetch('/api/setup/install-agents',{method:'POST'});
    const d = await r.json();
    const count = Object.values(d.agents||{}).filter(v=>v==='installed').length;
    msg.textContent=`${count} agents installed`; msg.style.color='var(--green)';
  } catch(e) { msg.textContent='Failed'; msg.style.color='var(--red)'; }
  btn.disabled=false;
}

async function setupRunExtractors(btn) {
  btn.disabled=true;
  const msg = document.getElementById('act-extract');
  msg.textContent='Running extractors...'; msg.style.color='var(--muted)';
  const extractors = ['screenpipe','mail_app','calendar_app'];
  let ok=0;
  for(const ext of extractors) {
    try { await fetch('/api/run/'+ext,{method:'POST'}); ok++; } catch(e) {}
  }
  msg.textContent=`${ok} extractors started (running in background)`; msg.style.color='var(--green)';
  btn.disabled=false;
}

async function setupReindex(btn) {
  btn.disabled=true;
  const msg = document.getElementById('act-index');
  msg.textContent='Building index...'; msg.style.color='var(--muted)';
  try {
    await fetch('/api/setup/reindex',{method:'POST'});
    msg.textContent='Index build started (running in background)'; msg.style.color='var(--green)';
  } catch(e) { msg.textContent='Failed'; msg.style.color='var(--red)'; }
  btn.disabled=false;
}

/* ═══ Wizard: AI Agents (Step 3) ═══ */
const defaultModels = {openai:'gpt-5.2', anthropic:'claude-sonnet-4-20250514', google:'gemini/gemini-2.5-pro', azure:'azure/gpt-5.2', ollama:'ollama/llama4'};
function wizProviderChanged() {
  const p=document.getElementById('wiz-llm-provider').value;
  document.getElementById('wiz-llm-model').value=defaultModels[p]||'';
  document.getElementById('wiz-llm-apibase-row').style.display=(p==='ollama'||p==='azure')?'block':'none';
  if(p==='ollama') document.getElementById('wiz-llm-apibase').value='http://localhost:11434';
}
function setProviderChanged() {
  const p=document.getElementById('set-llm-provider').value;
  document.getElementById('set-llm-model').value=defaultModels[p]||'';
  document.getElementById('set-llm-apibase-row').style.display=(p==='ollama'||p==='azure')?'block':'none';
  if(p==='ollama') document.getElementById('set-llm-apibase').value='http://localhost:11434';
}
async function wizTestLLM(btn) {
  btn.disabled=true;
  const msg=document.getElementById('wiz-llm-msg');
  msg.textContent='Testing...'; msg.style.color='var(--muted)';
  try {
    const body={provider:document.getElementById('wiz-llm-provider').value, model:document.getElementById('wiz-llm-model').value};
    const ab=document.getElementById('wiz-llm-apibase').value;
    if(ab) body.api_base=ab;
    const r=await fetch('/api/agents/test-llm',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const d=await r.json();
    msg.textContent=d.ok?'Connected: '+d.detail:'Error: '+d.detail;
    msg.style.color=d.ok?'var(--green)':'var(--red)';
  } catch(e) { msg.textContent='Network error'; msg.style.color='var(--red)'; }
  btn.disabled=false;
}
async function wizSaveAgents(btn) {
  btn.disabled=true;
  const msg=document.getElementById('wiz-agents-msg');
  msg.textContent='Saving...'; msg.style.color='var(--muted)';
  const body={
    provider:document.getElementById('wiz-llm-provider').value,
    model:document.getElementById('wiz-llm-model').value,
    reasoning_effort:document.getElementById('wiz-llm-reasoning').value,
    api_base:document.getElementById('wiz-llm-apibase').value||null,
    email:{
      enabled:true,
      smtp_host:document.getElementById('wiz-smtp-host').value,
      smtp_port:parseInt(document.getElementById('wiz-smtp-port').value)||587,
      smtp_user:document.getElementById('wiz-smtp-user').value,
      from:document.getElementById('wiz-smtp-user').value,
      to:document.getElementById('wiz-smtp-to').value,
    }
  };
  try {
    const r=await fetch('/api/agents/config',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const d=await r.json();
    if(r.ok) { msg.textContent='Saved!'; msg.style.color='var(--green)'; }
    else { msg.textContent=d.error||'Failed'; msg.style.color='var(--red)'; }
  } catch(e) { msg.textContent='Network error'; msg.style.color='var(--red)'; }
  btn.disabled=false;
}

/* ═══ Settings: Shared Context ═══ */
let _ctxData={my_context:{},global_context:{}};
async function loadContext() {
  try {
    _ctxData=await(await fetch('/api/agents/context')).json();
    const mc=_ctxData.my_context||{};
    document.getElementById('ctx-name').value=mc.name||'';
    document.getElementById('ctx-title').value=mc.title||'';
    document.getElementById('ctx-company').value=mc.company||'';
    document.getElementById('ctx-reports-to').value=mc.reports_to||'';
    renderCtxRelationships();
    const gc=_ctxData.global_context||{};
    document.getElementById('gctx-company').value=gc.company||'';
    document.getElementById('gctx-industry').value=gc.industry||'';
    renderGctxTags('competitors',gc.competitors||[]);
    renderGctxPriorities(gc.strategic_priorities||[]);
    renderGctxTags('tools',(gc.tools||[]).concat(gc.platforms||[]));
  } catch(e) { console.error('Context load failed',e); }
}
function renderCtxRelationships() {
  const rels=(_ctxData.my_context||{}).relationships||[];
  const el=document.getElementById('ctx-rels');
  let h='';
  for(let i=0;i<rels.length;i++) {
    const r=rels[i];
    h+=`<div class="topic-card" style="padding:10px;margin-bottom:8px">
      <button class="topic-del" onclick="removeCtxRel(${i})">x</button>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:6px">
        <div><label class="cfg-label">Name</label><input class="cfg-input" value="${escHtml(r.name||'')}" onchange="updateCtxRel(${i},'name',this.value)"></div>
        <div><label class="cfg-label">Role</label><input class="cfg-input" value="${escHtml(r.role||'')}" onchange="updateCtxRel(${i},'role',this.value)"></div>
      </div>
      <div><label class="cfg-label">Relationship</label><input class="cfg-input" value="${escHtml(r.relationship||'')}" onchange="updateCtxRel(${i},'relationship',this.value)"></div>
    </div>`;
  }
  el.innerHTML=h;
}
function updateCtxRel(i,field,val) {
  const rels=(_ctxData.my_context||{}).relationships||[];
  if(rels[i]) rels[i][field]=val;
}
function removeCtxRel(i) {
  const rels=(_ctxData.my_context||{}).relationships||[];
  rels.splice(i,1);
  renderCtxRelationships();
}
function addCtxRelationship() {
  if(!_ctxData.my_context) _ctxData.my_context={};
  if(!_ctxData.my_context.relationships) _ctxData.my_context.relationships=[];
  _ctxData.my_context.relationships.push({name:'',role:'',relationship:''});
  renderCtxRelationships();
}
function renderGctxTags(field,items) {
  const el=document.getElementById('gctx-'+field);
  if(!el) return;
  el.innerHTML=items.map((t,i)=>`<span class="kw-pill">${escHtml(t)}<button onclick="removeGctxTag('${field}',${i})">x</button></span>`).join('');
}
function renderGctxPriorities(items) {
  const el=document.getElementById('gctx-priorities');
  el.innerHTML=items.map((p,i)=>`<div style="display:flex;gap:6px;align-items:center;margin-bottom:4px">
    <input class="cfg-input" value="${escHtml(p)}" onchange="_gctxPriorities[${i}]=this.value" style="flex:1">
    <button class="btn btn-sm" style="color:var(--red);padding:2px 8px" onclick="removeGctxPri(${i})">x</button>
  </div>`).join('');
  window._gctxPriorities=items;
}
function removeGctxTag(field,i) {
  const gc=_ctxData.global_context||{};
  const arr=field==='tools'?(gc.tools||[]).concat(gc.platforms||[]):gc[field]||[];
  arr.splice(i,1);
  if(field==='tools'){gc.tools=arr;gc.platforms=[];}else{gc[field]=arr;}
  renderGctxTags(field,arr);
}
function removeGctxPri(i) {
  const gc=_ctxData.global_context||{};
  const arr=gc.strategic_priorities||[];
  arr.splice(i,1);
  renderGctxPriorities(arr);
}
function addGctxTag(field) {
  const inputId={competitors:'gctx-comp-input',priorities:'gctx-pri-input',tools:'gctx-tool-input'}[field];
  const inp=document.getElementById(inputId);
  const val=(inp?inp.value:'').trim();
  if(!val) return;
  inp.value='';
  if(!_ctxData.global_context) _ctxData.global_context={};
  const gc=_ctxData.global_context;
  if(field==='priorities'){
    if(!gc.strategic_priorities) gc.strategic_priorities=[];
    gc.strategic_priorities.push(val);
    renderGctxPriorities(gc.strategic_priorities);
  } else if(field==='tools'){
    if(!gc.tools) gc.tools=[];
    gc.tools.push(val);
    renderGctxTags('tools',(gc.tools||[]).concat(gc.platforms||[]));
  } else {
    if(!gc[field]) gc[field]=[];
    gc[field].push(val);
    renderGctxTags(field,gc[field]);
  }
}
async function saveContext() {
  const msg1=document.getElementById('ctx-save-msg');
  const msg2=document.getElementById('gctx-save-msg');
  [msg1,msg2].forEach(m=>{if(m){m.textContent='Saving...';m.style.color='var(--muted)';}});
  _ctxData.my_context=_ctxData.my_context||{};
  _ctxData.my_context.name=document.getElementById('ctx-name').value;
  _ctxData.my_context.title=document.getElementById('ctx-title').value;
  _ctxData.my_context.company=document.getElementById('ctx-company').value;
  _ctxData.my_context.reports_to=document.getElementById('ctx-reports-to').value;
  _ctxData.global_context=_ctxData.global_context||{};
  _ctxData.global_context.company=document.getElementById('gctx-company').value;
  _ctxData.global_context.industry=document.getElementById('gctx-industry').value;
  if(window._gctxPriorities) _ctxData.global_context.strategic_priorities=window._gctxPriorities;
  try {
    const r=await fetch('/api/agents/context',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(_ctxData)});
    if(r.ok){[msg1,msg2].forEach(m=>{if(m){m.textContent='Saved!';m.style.color='var(--green)';}});}
    else {[msg1,msg2].forEach(m=>{if(m){m.textContent='Failed';m.style.color='var(--red)';}});}
  } catch(e) {[msg1,msg2].forEach(m=>{if(m){m.textContent='Error';m.style.color='var(--red)';}});}
}

/* ═══ Knowledge: Pinned Files ═══ */
let _kfData=[];
async function loadKnowledgeFiles() {
  try {
    _kfData=await(await fetch('/api/knowledge/files')).json();
    renderKnowledgeFiles();
    loadKnowledgeTemplates();
  } catch(e) { document.getElementById('kf-list').innerHTML='<div style="padding:16px;color:var(--red)">Failed to load</div>'; }
}
function renderKnowledgeFiles() {
  const el=document.getElementById('kf-list');
  if(!_kfData.length) { el.innerHTML='<div style="padding:16px;color:var(--muted)">No knowledge files found</div>'; return; }
  const cats={};
  for(const f of _kfData) {
    if(f.category==='Templates') continue;
    if(!cats[f.category]) cats[f.category]=[];
    cats[f.category].push(f);
  }
  let h='';
  for(const cat of ['Profile','Programs','Patterns','Preferences']) {
    const files=cats[cat]||[];
    if(!files.length) continue;
    h+=`<div style="padding:8px 14px;font-size:.76rem;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:.04em;border-bottom:1px solid var(--border);background:var(--surface2)">${cat} (${files.length})</div>`;
    for(const f of files) {
      const stale=f.age_days>30;
      const pinIcon=f.pinned?'&#9733;':'&#9734;';
      const pinColor=f.pinned?'var(--accent)':'var(--muted)';
      const staleTag=stale?`<span style="color:var(--yellow);font-size:.72rem;margin-left:6px">${f.age_days}d old</span>`:'';
      h+=`<div style="display:grid;grid-template-columns:28px 1fr auto 100px 40px;gap:0;padding:10px 14px;border-bottom:1px solid var(--border);font-size:.84rem;align-items:center;cursor:pointer" onclick="toggleKnowledgeExpand('${f.path}',this)">
        <span style="color:${pinColor};font-size:1rem;cursor:pointer" onclick="event.stopPropagation();toggleKnowledgePin('${f.path}',${f.pinned?'false':'true'})" title="${f.pinned?'Unpin':'Pin'}">${pinIcon}</span>
        <span style="font-weight:500">${escHtml(f.title)}${staleTag}</span>
        <div style="display:flex;gap:4px;flex-wrap:wrap">${(f.tags||[]).slice(0,3).map(t=>'<span class="kw-pill" style="font-size:.68rem">'+escHtml(t)+'</span>').join('')}</div>
        <span style="color:var(--muted);font-size:.78rem;text-align:right">${escHtml(f.updated_at||'')}</span>
        <span style="color:var(--muted);font-size:.78rem;text-align:center">&#9662;</span>
      </div>
      <div class="kf-expand" id="kfe-${f.path.replace(/[\/\.]/g,'_')}" style="display:none;padding:12px 14px;background:var(--bg);border-bottom:1px solid var(--border)"></div>`;
    }
  }
  el.innerHTML=h;
}
async function toggleKnowledgeExpand(path,row) {
  const id='kfe-'+path.replace(/[\/\.]/g,'_');
  const el=document.getElementById(id);
  if(!el) return;
  if(el.style.display==='block') { el.style.display='none'; return; }
  el.style.display='block';
  el.innerHTML='<div style="color:var(--muted);font-size:.82rem">Loading...</div>';
  try {
    const d=await(await fetch('/api/knowledge/file?path='+encodeURIComponent(path))).json();
    if(d.error) { el.innerHTML='<div style="color:var(--red)">'+escHtml(d.error)+'</div>'; return; }
    el.innerHTML=`<textarea class="cfg-textarea" style="min-height:200px;font-family:'SF Mono','Menlo',monospace;font-size:.8rem;line-height:1.6" id="kfed-${path.replace(/[\/\.]/g,'_')}">${escHtml(d.content)}</textarea>
    <div style="margin-top:8px;display:flex;gap:8px;align-items:center">
      <button class="btn btn-sm" onclick="saveKnowledgeFile('${path}')">Save</button>
      <span id="kfmsg-${path.replace(/[\/\.]/g,'_')}" style="font-size:.82rem"></span>
    </div>`;
  } catch(e) { el.innerHTML='<div style="color:var(--red)">Failed</div>'; }
}
async function saveKnowledgeFile(path) {
  const id=path.replace(/[\/\.]/g,'_');
  const content=document.getElementById('kfed-'+id)?.value||'';
  const msg=document.getElementById('kfmsg-'+id);
  if(msg){msg.textContent='Saving...';msg.style.color='var(--muted)';}
  try {
    const r=await fetch('/api/knowledge/file',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({path,content})});
    if(r.ok){if(msg){msg.textContent='Saved!';msg.style.color='var(--green)';}}
    else {if(msg){msg.textContent='Failed';msg.style.color='var(--red)';}}
  } catch(e){if(msg){msg.textContent='Error';msg.style.color='var(--red)';}}
}
async function toggleKnowledgePin(path,pinned) {
  try {
    await fetch('/api/knowledge/pin',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({path,pinned})});
    loadKnowledgeFiles();
  } catch(e) {}
}
async function loadKnowledgeTemplates() {
  const sel=document.getElementById('kf-new-template');
  if(!sel) return;
  const templates=_kfData.filter(f=>f.category==='Templates');
  sel.innerHTML='<option value="">New from template...</option>'+templates.map(t=>`<option value="${escHtml(t.filename)}">${escHtml(t.title)}</option>`).join('');
}
async function createKnowledgeFile() {
  const template=document.getElementById('kf-new-template')?.value;
  const category=document.getElementById('kf-new-category')?.value||'01_programs';
  const filename=document.getElementById('kf-new-name')?.value?.trim();
  if(!template||!filename) return;
  try {
    const r=await fetch('/api/knowledge/new',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({template,category,filename})});
    const d=await r.json();
    if(r.ok) { document.getElementById('kf-new-name').value=''; loadKnowledgeFiles(); }
    else { alert(d.error||'Failed'); }
  } catch(e) { alert('Network error'); }
}

/* ═══ Settings: Agent Config ═══ */
async function loadAgentConfig() {
  try {
    const d=await(await fetch('/api/agents/config')).json();
    document.getElementById('set-llm-provider').value=d.provider||'openai';
    document.getElementById('set-llm-model').value=d.model||'gpt-5.2';
    document.getElementById('set-llm-reasoning').value=d.reasoning_effort||'high';
    document.getElementById('set-llm-apibase').value=d.api_base||'';
    const showBase=d.provider==='ollama'||d.provider==='azure';
    document.getElementById('set-llm-apibase-row').style.display=showBase?'block':'none';
    if(d.email) {
      document.getElementById('set-smtp-host').value=d.email.smtp_host||'smtp.office365.com';
      document.getElementById('set-smtp-port').value=d.email.smtp_port||587;
      document.getElementById('set-smtp-user').value=d.email.smtp_user||d.email.from||'';
      document.getElementById('set-smtp-to').value=d.email.to||'';
    }
  } catch(e) {}
}
async function saveAgentSettings() {
  const msg=document.getElementById('set-llm-msg');
  msg.textContent='Saving...'; msg.style.color='var(--muted)';
  const body={
    provider:document.getElementById('set-llm-provider').value,
    model:document.getElementById('set-llm-model').value,
    reasoning_effort:document.getElementById('set-llm-reasoning').value,
    api_base:document.getElementById('set-llm-apibase').value||null,
    email:{
      smtp_host:document.getElementById('set-smtp-host').value,
      smtp_port:parseInt(document.getElementById('set-smtp-port').value)||587,
      smtp_user:document.getElementById('set-smtp-user').value,
      from:document.getElementById('set-smtp-user').value,
      to:document.getElementById('set-smtp-to').value,
    }
  };
  try {
    const r=await fetch('/api/agents/config',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    if(r.ok) { msg.textContent='Saved!'; msg.style.color='var(--green)'; }
    else { msg.textContent='Failed'; msg.style.color='var(--red)'; }
  } catch(e) { msg.textContent='Error'; msg.style.color='var(--red)'; }
}
async function testLLMFromSettings(btn) {
  btn.disabled=true;
  const msg=document.getElementById('set-llm-msg');
  msg.textContent='Testing...'; msg.style.color='var(--muted)';
  try {
    const r=await fetch('/api/agents/test-llm',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({})});
    const d=await r.json();
    msg.textContent=d.ok?'OK: '+d.detail:'Error: '+d.detail;
    msg.style.color=d.ok?'var(--green)':'var(--red)';
  } catch(e) { msg.textContent='Network error'; msg.style.color='var(--red)'; }
  btn.disabled=false;
}
async function testEmailFromSettings(btn) {
  btn.disabled=true;
  const msg=document.getElementById('set-email-msg');
  msg.textContent='Sending test...'; msg.style.color='var(--muted)';
  try {
    const r=await fetch('/api/agents/test-email',{method:'POST'});
    const d=await r.json();
    msg.textContent=d.ok?'Sent!':'Error: '+d.detail;
    msg.style.color=d.ok?'var(--green)':'var(--red)';
  } catch(e) { msg.textContent='Network error'; msg.style.color='var(--red)'; }
  btn.disabled=false;
}

/* ═══ Skills: Scheduled Agents ═══ */
async function loadAgentStatus() {
  try {
    const data=await(await fetch('/api/agents/status')).json();
    const card=document.getElementById('scheduled-agents-card');
    const rows=document.getElementById('agent-status-rows');
    if(!data.length) { card.style.display='none'; return; }
    card.style.display='block';
    let h='';
    for(const a of data) {
      const statusColor = a.status==='active'?'var(--green)':a.status==='installed'?'var(--yellow)':'var(--muted)';
      const lastRun = a.last_run ? new Date(a.last_run).toLocaleString([],{month:'short',day:'numeric',hour:'2-digit',minute:'2-digit'}) : 'Never';
      const reportDate = a.last_report ? a.last_report.replace('.md','') : '';
      const viewBtn = reportDate ? `<a class="btn btn-sm" href="/report/${encodeURIComponent(a.skill)}/${encodeURIComponent(reportDate)}" target="_blank" style="text-decoration:none;">View</a>` : '';
      h+=`<div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr auto;gap:0;padding:10px 12px;border-bottom:1px solid var(--border);font-size:.84rem;align-items:center">
        <span style="font-weight:600">${escHtml(a.skill)}</span>
        <span style="color:var(--muted)">${escHtml(a.schedule)}</span>
        <span style="color:var(--muted)">${lastRun}</span>
        <span style="color:${statusColor};font-weight:500">${a.status}</span>
        <div style="display:flex;gap:6px">
          ${viewBtn}
          <button class="btn btn-sm" onclick="runSkillNow('${a.skill}',this)">Run Now</button>
          <button class="btn btn-sm" onclick="toggleSkillConfig('${a.skill}')">Edit</button>
        </div>
      </div>
      <div class="skill-config-panel" id="cfg-panel-${a.skill}"></div>`;
    }
    rows.innerHTML=h;
  } catch(e) { console.error('Agent status failed',e); }
}
async function runSkillNow(name,btn) {
  btn.disabled=true; btn.textContent='Running...';
  try {
    const resp=await fetch('/api/agents/run-skill/'+name,{method:'POST'});
    const body=await resp.json();
    if(!resp.ok) {
      btn.textContent=body.error||'Error'; btn.disabled=false;
      setTimeout(()=>{btn.textContent='Run Now';btn.onclick=()=>runSkillNow(name,btn);},3000);
      return;
    }
    const startedAt=body.started_at;
    btn.textContent='Generating...';
    let attempts=0;
    const poll=setInterval(async()=>{
      attempts++;
      try {
        const sr=await fetch('/api/agents/run-status/'+name);
        const st=await sr.json();
        if(st.status==='failed') {
          clearInterval(poll);
          btn.textContent='Failed';btn.disabled=false;btn.style.color='var(--red)';
          console.error('Skill run failed:',st.error);
          setTimeout(()=>{btn.textContent='Run Now';btn.style.color='';btn.onclick=()=>runSkillNow(name,btn);},4000);
          return;
        }
        const today=new Date().toISOString().slice(0,10);
        const r=await fetch('/api/agents/reports/'+name+'/'+today+'?after='+startedAt);
        if(r.ok) {
          clearInterval(poll);
          btn.textContent='View Report';
          btn.disabled=false;
          btn.onclick=()=>window.open('/report/'+encodeURIComponent(name)+'/'+encodeURIComponent(today),'_blank');
          loadAgentStatus();
        } else if(attempts>120) {
          clearInterval(poll);
          btn.textContent='Run Now'; btn.disabled=false;
          btn.onclick=()=>runSkillNow(name,btn);
        }
      } catch(e) { console.error('Poll error:',e); }
    },3000);
  } catch(e) { btn.textContent='Error'; btn.disabled=false; }
}
function showSkillReport(name,date) {
  fetch('/api/agents/reports/'+name+'/'+date+'/json').then(r=>{
    if(r.ok) return r.json().then(d=>({json:true,data:d}));
    return fetch('/api/agents/reports/'+name+'/'+date).then(r2=>r2.json()).then(d=>({json:false,data:d}));
  }).then(result=>{
    if(result.json) {
      openRichView(name,date,result.data);
    } else {
      if(result.data.error) return;
      const backdrop=document.getElementById('skill-modal-backdrop');
      document.getElementById('skill-modal-title').textContent=name+' — '+date;
      document.getElementById('skill-modal-body').innerHTML=renderMarkdown(result.data.content);
      backdrop.classList.add('open');
    }
  });
}
function openRichView(name,date,data) {
  const overlay=document.getElementById('rich-view-overlay');
  const titleMap={'news-pulse':'News Pulse','weekly-status':'Weekly Status','plan-my-week':'Plan My Week','morning-brief':'Morning Brief','commitment-tracker':'Commitment Tracker','project-brief':'Project Brief','focus-audit':'Focus Audit','relationship-crm':'Relationship CRM','team-manager':'Team Manager','approvals-queue':'Approvals Queue'};
  document.getElementById('rich-view-title').textContent=titleMap[name]||name;
  document.getElementById('rich-view-date').textContent=date;
  const body=document.getElementById('rich-view-body');
  const renderers={'news-pulse':renderNewsPulse,'weekly-status':renderWeeklyStatus,'plan-my-week':renderPlanMyWeek,'morning-brief':renderMorningBrief,'commitment-tracker':renderCommitmentTracker,'project-brief':renderProjectBrief,'focus-audit':renderFocusAudit,'relationship-crm':renderRelationshipCrm,'team-manager':renderTeamManager,'approvals-queue':renderApprovalsQueue};
  const fn=renderers[name];
  if(fn) body.innerHTML=fn(data,name,date);
  else body.innerHTML='<p>Rich view not available for this skill.</p>';
  overlay.classList.add('open');
  document.body.style.overflow='hidden';
  if(name==='weekly-status') initWeeklyStatusCharts(data);
  setTimeout(()=>_initRichViewCharts(name,data),100);
}
function _initRichViewCharts(name,data) {
  if(name==='weekly-status') initWeeklyStatusCharts(data);
  if(name==='commitment-tracker') {
    createGaugeChart('ct-health-gauge',data.health_score||0,'Health');
    const bp=data.by_priority||{};
    createDoughnutChart('ct-priority-chart',['P0','P1','P2'],[bp.P0||0,bp.P1||0,bp.P2||0],['rgba(248,113,113,.7)','rgba(250,204,21,.7)','rgba(108,124,255,.7)']);
  }
  if(name==='project-brief') {
    (data.projects||[]).forEach((p,i)=>{createGaugeChart('pb-health-'+i,p.health_score||0,'Health');});
  }
  if(name==='plan-my-week') {
    createGaugeChart('pmw-score-gauge',data.week_score||0,'Week');
    const days=data.days||[];
    if(days.length) {
      const hasActuals=days.some(d=>d.actual_meeting_hours!=null);
      const datasets=[
        {label:'Planned Meetings',data:days.map(d=>d.meeting_hours||d.meetings?.hours||0),backgroundColor:'rgba(248,113,113,.35)',borderRadius:6,borderWidth:1,borderColor:'rgba(248,113,113,.6)'},
        {label:'Planned Focus',data:days.map(d=>d.focus_hours||0),backgroundColor:'rgba(74,222,128,.3)',borderRadius:6,borderWidth:1,borderColor:'rgba(74,222,128,.5)'}
      ];
      if(hasActuals) {
        datasets.push({label:'Actual Meetings',data:days.map(d=>d.actual_meeting_hours!=null?d.actual_meeting_hours:null),backgroundColor:'rgba(248,113,113,.8)',borderRadius:6});
        datasets.push({label:'Actual Focus',data:days.map(d=>d.actual_focus_hours!=null?d.actual_focus_hours:null),backgroundColor:'rgba(74,222,128,.8)',borderRadius:6});
      }
      createBarChart('pmw-capacity-chart',days.map(d=>d.day_name||d.date),datasets);
    }
  }
  if(name==='morning-brief') {
    createGaugeChart('mb-day-gauge',data.day_score||0,'Day');
    const dc=data.day_composition||{};
    if(dc.meeting_percent||dc.focus_percent) createDoughnutChart('mb-comp-chart',['Meetings','Focus','Admin'],[dc.meeting_percent||0,dc.focus_percent||0,dc.admin_percent||0],['rgba(248,113,113,.7)','rgba(74,222,128,.7)','rgba(128,128,128,.5)']);
  }
  if(name==='focus-audit') {
    createGaugeChart('fa-prod-gauge',data.productivity_score||0,'Score');
    const abRaw=data.app_breakdown||[];
    const abValid=abRaw.filter(a=>(a.minutes||0)>0);
    const taRaw=data.top_apps||[];
    const taValid=taRaw.filter(a=>(a.hours||0)>0);
    const chartData=abValid.length?abValid.map(a=>({name:a.name,val:a.minutes||0})):taValid.map(a=>({name:a.name,val:Math.round((a.hours||0)*60)}));
    if(chartData.length) createDoughnutChart('fa-app-chart',chartData.map(a=>a.name),chartData.map(a=>a.val),chartData.map((_,i)=>_avatarColors[i%_avatarColors.length]));
  }
  if(name==='relationship-crm') {
    const top=(data.top_contacts||[]).filter(c=>(c.interactions_30d||0)>0||(c.email_count_30d||0)>0||(c.meeting_count_30d||0)>0||(c.teams_count_30d||0)>0).slice(0,10);
    if(top.length) {
      const hasChannel=top.some(c=>(c.email_count_30d||0)>0||(c.meeting_count_30d||0)>0||(c.teams_count_30d||0)>0);
      const datasets=hasChannel?[
        {label:'Email',data:top.map(c=>c.email_count_30d||0),backgroundColor:'rgba(108,124,255,.7)',borderRadius:6},
        {label:'Meeting',data:top.map(c=>c.meeting_count_30d||0),backgroundColor:'rgba(74,222,128,.6)',borderRadius:6},
        {label:'Teams',data:top.map(c=>c.teams_count_30d||0),backgroundColor:'rgba(168,85,247,.7)',borderRadius:6}
      ]:[{label:'30d Interactions',data:top.map(c=>c.interactions_30d||0),backgroundColor:'rgba(108,124,255,.7)',borderRadius:6}];
      const el=document.getElementById('crm-freq-chart');
      if(el&&typeof Chart!=='undefined') new Chart(el,{type:'bar',data:{labels:top.map(c=>c.name.split(' ')[0]),datasets:datasets},options:{responsive:true,plugins:{legend:{labels:{color:'#9ca3af'}}},scales:{x:{stacked:hasChannel,ticks:{color:'#9ca3af'},grid:{color:'rgba(255,255,255,.05)'}},y:{stacked:hasChannel,ticks:{color:'#9ca3af'},grid:{color:'rgba(255,255,255,.05)'}}}}});
    }
  }
  if(name==='team-manager') {
    createGaugeChart('tm-health-gauge',data.team_health_score||0,'Health');
  }
}

/* ═══ Approvals Queue Rich View ═══ */
function renderApprovalsQueue(data) {
  const pc=data.pending_count||0, atc=data.approved_today_count||0, oc=data.overdue_count||0;
  const pcColor=pc>0?'var(--yellow)':'var(--green)';
  const ocColor=oc>0?'var(--red)':'var(--green)';
  let h=`<div class="grid" style="grid-template-columns:repeat(3,1fr);margin-bottom:24px">
    <div class="card" style="text-align:center"><div style="font-size:2rem;font-weight:700;color:${pcColor}">${pc}</div><div style="color:var(--muted);font-size:.84rem">Pending</div></div>
    <div class="card" style="text-align:center"><div style="font-size:2rem;font-weight:700;color:var(--green)">${atc}</div><div style="color:var(--muted);font-size:.84rem">Approved Today</div></div>
    <div class="card" style="text-align:center"><div style="font-size:2rem;font-weight:700;color:${ocColor}">${oc}</div><div style="color:var(--muted);font-size:.84rem">Overdue (&gt;3d)</div></div>
  </div>`;

  const bySys=data.by_system||{};
  if(Object.keys(bySys).length) {
    h+=`<div class="card" style="margin-bottom:16px"><div class="card-header"><span class="card-title">By System</span></div><div style="display:flex;gap:16px;flex-wrap:wrap">`;
    const sysColors={'Concur Expense':'var(--blue)','ServiceNow':'var(--orange)','DocuSign':'var(--accent)'};
    for(const [sys,cnt] of Object.entries(bySys)) {
      const c=sysColors[sys]||'var(--muted)';
      h+=`<div style="display:flex;align-items:center;gap:6px"><span style="width:10px;height:10px;border-radius:50%;background:${c};display:inline-block"></span><span style="font-size:.88rem">${escHtml(sys)}: <strong>${cnt}</strong></span></div>`;
    }
    h+=`</div></div>`;
  }

  const pending=data.pending||[];
  h+=`<div class="card"><div class="card-header"><span class="card-title">Pending Approvals (${pending.length})</span></div>`;
  if(pending.length===0) {
    h+=`<p style="color:var(--green);text-align:center;padding:24px">All clear — no pending approvals.</p>`;
  } else {
    h+=`<div style="overflow-x:auto"><table style="width:100%;border-collapse:collapse;font-size:.84rem">
      <thead><tr style="border-bottom:2px solid var(--border);text-align:left">
        <th style="padding:8px 12px">System</th><th style="padding:8px 12px">Subject</th><th style="padding:8px 12px">From</th><th style="padding:8px 12px">Age</th><th style="padding:8px 12px">Actions</th>
      </tr></thead><tbody>`;
    for(const item of pending) {
      const age=item.received_at?Math.floor((Date.now()-new Date(item.received_at).getTime())/(86400000))+'d':'?';
      const ageColor=parseInt(age)>3?'var(--red)':parseInt(age)>1?'var(--yellow)':'var(--text)';
      const sysLabel=item.display_name||item.system||'';
      const linkBtn=item.link?`<a href="${escHtml(item.link)}" target="_blank" class="btn btn-sm" style="text-decoration:none;margin-right:4px">Open</a>`:'';
      h+=`<tr style="border-bottom:1px solid var(--border)">
        <td style="padding:8px 12px"><span class="badge" style="background:rgba(108,124,255,.15);color:var(--accent)">${escHtml(sysLabel)}</span></td>
        <td style="padding:8px 12px">${escHtml(item.subject||'')}</td>
        <td style="padding:8px 12px;color:var(--muted)">${escHtml(item.from||'')}</td>
        <td style="padding:8px 12px;color:${ageColor};font-weight:600">${age}</td>
        <td style="padding:8px 12px;white-space:nowrap">${linkBtn}<button class="btn btn-sm btn-green" onclick="approvalAction('${escHtml(item.approval_id)}','approved',this)">Approve</button> <button class="btn btn-sm" onclick="approvalAction('${escHtml(item.approval_id)}','dismissed',this)">Dismiss</button></td>
      </tr>`;
    }
    h+=`</tbody></table></div>`;
  }
  h+=`</div>`;

  const closed=data.recently_closed||[];
  if(closed.length) {
    h+=`<div class="card" style="margin-top:16px"><div class="card-header"><span class="card-title">Recently Closed (7 days)</span></div>`;
    h+=`<div style="overflow-x:auto"><table style="width:100%;border-collapse:collapse;font-size:.84rem">
      <thead><tr style="border-bottom:2px solid var(--border);text-align:left">
        <th style="padding:8px 12px">Status</th><th style="padding:8px 12px">System</th><th style="padding:8px 12px">Subject</th><th style="padding:8px 12px">Closed</th><th style="padding:8px 12px">Evidence</th>
      </tr></thead><tbody>`;
    for(const item of closed) {
      const statusColor=item.status==='approved'?'var(--green)':'var(--muted)';
      const closedDate=(item.status_changed_at||'').slice(0,10);
      const evidence=item.evidence?escHtml(item.evidence.snippet||'').slice(0,60)+'...':'Manual';
      h+=`<tr style="border-bottom:1px solid var(--border)">
        <td style="padding:8px 12px"><span style="color:${statusColor};font-weight:600;text-transform:capitalize">${item.status}</span></td>
        <td style="padding:8px 12px">${escHtml(item.display_name||item.system||'')}</td>
        <td style="padding:8px 12px">${escHtml(item.subject||'')}</td>
        <td style="padding:8px 12px;color:var(--muted)">${closedDate}</td>
        <td style="padding:8px 12px;color:var(--muted);font-size:.78rem">${evidence}</td>
      </tr>`;
    }
    h+=`</tbody></table></div></div>`;
  }

  const analysis=data.analysis||{};
  if(analysis.oldest_pending_days>0||analysis.busiest_system!=='none') {
    h+=`<div class="card" style="margin-top:16px"><div class="card-header"><span class="card-title">Analysis</span></div>`;
    if(analysis.oldest_pending_days>0) h+=`<div class="stat-row"><span class="stat-label">Oldest pending</span><span class="stat-value">${analysis.oldest_pending_days} days</span></div>`;
    if(analysis.busiest_system&&analysis.busiest_system!=='none') h+=`<div class="stat-row"><span class="stat-label">Busiest system</span><span class="stat-value">${escHtml(analysis.busiest_system)}</span></div>`;
    h+=`</div>`;
  }
  return h;
}

async function approvalAction(approvalId, action, btn) {
  btn.disabled=true; btn.textContent='...';
  try {
    const r=await fetch('/api/approvals/action',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({approval_id:approvalId,action:action})});
    if(r.ok) {
      btn.textContent='Done';
      btn.style.color='var(--green)';
      setTimeout(()=>{
        const today=new Date().toISOString().slice(0,10);
        fetch('/api/agents/reports/approvals-queue/'+today+'/json').then(r=>r.ok?r.json():null).then(d=>{
          if(d) {
            const body=document.getElementById('rich-view-body')||document.getElementById('report-body');
            if(body) body.innerHTML=renderApprovalsQueue(d);
          }
        });
      },1500);
    } else {
      const d=await r.json();
      btn.textContent=d.error||'Error'; btn.style.color='var(--red)';
    }
  } catch(e) { btn.textContent='Error'; btn.style.color='var(--red)'; }
}

function closeRichView() {
  document.getElementById('rich-view-overlay').classList.remove('open');
  document.body.style.overflow='';
}
function downloadPDF() {
  const cards = document.querySelectorAll('.np-card');
  cards.forEach(c => c.classList.add('expanded'));
  const origTitle = document.title;
  const titleEl = document.getElementById('rich-view-title') || document.querySelector('.report-header h1');
  const dateEl = document.getElementById('rich-view-date') || document.querySelector('.report-header .date');
  if (titleEl && dateEl) document.title = titleEl.textContent + ' - ' + dateEl.textContent;

  document.querySelectorAll('canvas').forEach(canvas => {
    try {
      const img = document.createElement('img');
      img.src = canvas.toDataURL('image/png');
      img.style.cssText = 'max-width:100%;height:auto;border-radius:12px;';
      img.className = 'chart-print-img';
      canvas.parentNode.insertBefore(img, canvas);
      canvas.style.display = 'none';
      canvas.dataset.printHidden = '1';
    } catch(e) {}
  });

  const html = document.documentElement;
  const wasLight = html.classList.contains('exec-light');
  if (!wasLight) html.classList.add('exec-light');

  setTimeout(() => {
    window.print();
    document.title = origTitle;
    if (!wasLight) html.classList.remove('exec-light');
    document.querySelectorAll('canvas[data-print-hidden]').forEach(c => {
      c.style.display = '';
      delete c.dataset.printHidden;
      const img = c.parentNode.querySelector('.chart-print-img');
      if (img) img.remove();
    });
  }, 200);
}

/* ═══ Chart.js Helpers (Nuclear Overhaul) ═══ */
const _avatarColors=['#6c7cff','#4ade80','#f87171','#facc15','#a78bfa','#fb923c','#22d3ee','#e879f9'];

function _scoreColor(score) {
  if(score>=75) return 'var(--green)';
  if(score>=50) return '#facc15';
  if(score>=25) return 'var(--red)';
  return 'var(--red)';
}

function _scoreGradient(score) {
  if(score>=75) return ['rgba(74,222,128,.7)','rgba(74,222,128,.15)'];
  if(score>=50) return ['rgba(250,204,21,.7)','rgba(250,204,21,.15)'];
  return ['rgba(248,113,113,.7)','rgba(248,113,113,.15)'];
}

function createGaugeChart(canvasId, score, label) {
  const el=document.getElementById(canvasId);
  if(!el||typeof Chart==='undefined') return;
  const [fg,bg]=_scoreGradient(score||0);
  new Chart(el,{type:'doughnut',data:{datasets:[{data:[score||0,100-(score||0)],backgroundColor:[fg,bg],borderWidth:0,circumference:270,rotation:225}]},options:{responsive:true,cutout:'78%',plugins:{legend:{display:false},tooltip:{enabled:false}},animation:{animateRotate:true}}});
}

function createBarChart(canvasId, labels, datasets) {
  const el=document.getElementById(canvasId);
  if(!el||typeof Chart==='undefined') return;
  new Chart(el,{type:'bar',data:{labels:labels,datasets:datasets},options:{responsive:true,plugins:{legend:{labels:{color:'#9ca3af'}}},scales:{x:{ticks:{color:'#9ca3af'},grid:{color:'rgba(255,255,255,.05)'}},y:{ticks:{color:'#9ca3af'},grid:{color:'rgba(255,255,255,.05)'}}}}});
}

function createDoughnutChart(canvasId, labels, values, colors) {
  const el=document.getElementById(canvasId);
  if(!el||typeof Chart==='undefined') return;
  new Chart(el,{type:'doughnut',data:{labels:labels,datasets:[{data:values,backgroundColor:colors,borderWidth:0}]},options:{responsive:true,cutout:'55%',plugins:{legend:{labels:{color:'#9ca3af',font:{size:11}}}}}});
}

function _fmtAnalysis(text) {
  if(!text) return '';
  if(Array.isArray(text)) text=text.join('\n');
  if(typeof text!=='string') text=String(text);
  const lines=text.split(/\n/).map(l=>l.trim()).filter(Boolean);
  if(lines.length<=1&&text.length<200) return `<div class="rv-analysis-text">${escHtml(text)}</div>`;
  let out='<ul style="margin:4px 0 0 16px;font-size:.84rem;line-height:1.55">';
  lines.forEach(l=>{
    let clean=l.replace(/^[-–•*]\s*/,'').replace(/^\d+[.)]\s*/,'');
    if(!clean) return;
    out+=`<li style="margin-bottom:5px">${escHtml(clean)}</li>`;
  });
  out+='</ul>';
  return out;
}
function renderAnalysisCard(analysis) {
  if(!analysis) return '';
  let h='<div class="rv-analysis">';
  if(analysis.executive_insight) {
    const ei=analysis.executive_insight;
    const paras=ei.split(/\n\n|\n(?=[A-Z])/);
    const first=paras[0]||'';
    const rest=paras.slice(1).join('\n');
    h+=`<div class="rv-analysis-insight">${escHtml(first)}</div>`;
    if(rest.trim()) {
      const detailId='analysis-detail-'+Math.random().toString(36).slice(2,8);
      h+=`<div id="${detailId}" style="display:none;margin-top:8px">${_fmtAnalysis(rest)}</div>`;
      h+=`<button onclick="var d=document.getElementById('${detailId}');var s=d.style.display==='none';d.style.display=s?'block':'none';this.textContent=s?'Hide details':'Show details'" style="background:none;border:none;color:var(--accent);font-size:.78rem;cursor:pointer;padding:4px 0;font-weight:600">Show details</button>`;
    }
  }
  h+='<div class="rv-analysis-row">';
  if(analysis.biggest_risk) h+=`<div class="rv-analysis-item"><div class="rv-analysis-label">Biggest Risk</div>${_fmtAnalysis(analysis.biggest_risk)}</div>`;
  if(analysis.recommended_focus) h+=`<div class="rv-analysis-item"><div class="rv-analysis-label">Recommended Focus</div>${_fmtAnalysis(analysis.recommended_focus)}</div>`;
  h+='</div>';
  if(analysis.predictions&&analysis.predictions.length) {
    h+='<div style="margin-top:12px"><div class="rv-analysis-label">Predictions</div><ul style="margin:4px 0 0 16px;font-size:.84rem">';
    analysis.predictions.forEach(p=>{h+=`<li style="margin-bottom:5px">${escHtml(p)}</li>`;});
    h+='</ul></div>';
  }
  h+='</div>';
  return h;
}

function renderScoreBadge(score, label, id) {
  return `<div class="rv-gauge" style="display:inline-block"><canvas id="${id}" width="120" height="120"></canvas><div class="rv-gauge-label"><div class="rv-gauge-value">${score||0}</div><div class="rv-gauge-sub">${escHtml(label||'')}</div></div></div>`;
}

function renderProgressBar(percent, color) {
  const c=color||(percent>=75?'green':percent>=50?'yellow':'red');
  return `<div class="rv-progress"><div class="rv-progress-bar rv-progress-bar-${c}" style="width:${Math.min(100,percent||0)}%"></div></div>`;
}

/* ═══ Shared: Executive Header Renderer ═══ */
function renderExecutiveHeader(data, skillName) {
  const eh = data.executive_header;
  const heroImg = data.hero_image || (eh && eh.hero_image) || '';
  const bluf = (eh && eh.bluf) || data.day_summary || data.executive_summary || data.summary || '';
  const status = (eh && eh.status) || '';
  const metrics = (eh && eh.metrics) || [];

  let h = '<div class="exec-header">';
  if (heroImg) {
    h += `<img class="exec-header-hero" src="${escHtml(heroImg)}" alt="" onerror="this.style.display='none';this.nextElementSibling.classList.add('exec-header-no-hero')">`;
    h += '<div class="exec-header-overlay">';
  } else {
    h += '<div class="exec-header-no-hero">';
  }

  if (status) {
    const sc = status === 'green' ? 'exec-header-status-green' : status === 'red' ? 'exec-header-status-red' : 'exec-header-status-yellow';
    const labels = { green: 'On Track', yellow: 'Watch', red: 'Attention Needed' };
    h += `<div class="exec-header-status ${sc}">${labels[status] || status}</div>`;
  }

  if (bluf) h += `<div class="exec-header-bluf">${escHtml(bluf)}</div>`;

  if (metrics.length) {
    h += '<div class="exec-header-metrics">';
    metrics.forEach(raw => {
      let m = raw;
      if (typeof m === 'string') {
        const idx = m.indexOf(':');
        m = idx >= 0 ? { label: m.slice(0, idx).trim(), value: m.slice(idx + 1).trim() } : { label: m, value: '' };
      }
      const colorMap = { green: '#34D399', red: '#FCA5A5', yellow: '#FBBF24', accent: '#A5B4FC' };
      const valColor = m.color ? (colorMap[m.color] || '#fff') : '#fff';
      h += `<div class="exec-header-kpi">
        <div class="exec-header-kpi-value" style="color:${valColor}">${escHtml(String(m.value || '0'))}</div>
        <div class="exec-header-kpi-label">${escHtml(m.label || '')}</div>`;
      if (m.trend) {
        const tCls = m.trend > 0 ? 'exec-header-kpi-trend-up' : 'exec-header-kpi-trend-down';
        h += `<div class="exec-header-kpi-trend ${tCls}">${m.trend > 0 ? '&#9650;' : '&#9660;'} ${Math.abs(m.trend)}</div>`;
      }
      h += '</div>';
    });
    h += '</div>';
  }

  h += '</div></div>';
  return h;
}

function renderThreeMoves(moves) {
  if (!moves || !moves.length) return '';
  let h = '<div class="rv-section"><div class="rv-section-title">Your Three Moves</div><div class="exec-moves">';
  moves.slice(0, 3).forEach((m, i) => {
    h += `<div class="exec-move">
      <div class="exec-move-num">${i + 1}</div>
      <div class="exec-move-title">${escHtml(m.action || m.title || '')}</div>
      <div class="exec-move-detail">${escHtml(m.detail || m.impact || m.reason || '')}</div>
      ${m.time_needed ? `<div class="exec-move-time">${escHtml(m.time_needed)}</div>` : ''}
    </div>`;
  });
  h += '</div></div>';
  return h;
}

function renderTalkingPoints(points) {
  if (!points || !points.length) return '';
  let h = '<div class="exec-talking-points"><div class="exec-talking-points-title"><span style="font-size:1.1rem">&#128172;</span> Talking Points for Today</div>';
  points.forEach(p => {
    h += `<div class="exec-talking-point">${escHtml(typeof p === 'string' ? p : p.point || p.text || '')}</div>`;
  });
  h += '</div>';
  return h;
}

function renderStakeholderHeat(commitments) {
  if (!commitments || !commitments.length) return '';
  const byPerson = {};
  commitments.forEach(c => {
    const name = c.owed_to || c.requestor || 'Unknown';
    if (!byPerson[name]) byPerson[name] = { total: 0, overdue: 0 };
    byPerson[name].total++;
    if (c.status === 'overdue' || c.status === 'Overdue' || c._overdue) byPerson[name].overdue++;
  });
  const entries = Object.entries(byPerson).sort((a, b) => b[1].total - a[1].total);
  if (!entries.length) return '';
  let h = '<div class="rv-section"><div class="rv-section-title">Stakeholder Exposure</div><div class="exec-stakeholder-heat">';
  entries.forEach(([name, info]) => {
    const bg = info.overdue > 0 ? 'rgba(220,38,38,0.08)' : info.total >= 3 ? 'rgba(217,119,6,0.08)' : 'rgba(5,150,105,0.06)';
    const countColor = info.overdue > 0 ? 'var(--red)' : info.total >= 3 ? 'var(--yellow)' : 'var(--green)';
    h += `<div class="exec-stakeholder-card" style="background:${bg}">
      <div class="exec-stakeholder-name">${escHtml(name)}</div>
      <div class="exec-stakeholder-count" style="color:${countColor}">${info.total}</div>
      <div class="exec-stakeholder-label">${info.overdue ? info.overdue + ' overdue' : 'items'}</div>
    </div>`;
  });
  h += '</div></div>';
  return h;
}

/* ═══ Rich View: News Pulse (Single-Column Accordion) ═══ */

function _npThumb(item) { return item.generated_thumbnail||item.thumbnail_url||''; }
function _npHero(topic) { return topic?.hero_image||''; }

function _npCard(id, thumb, title, snippet, badges, heroImg, fullBody, soWhat, soWhatClass, sourceLine) {
  const thumbHtml = thumb
    ? `<img class="np-card-thumb${soWhatClass==='internal'?' internal':''}" src="${escHtml(thumb)}" alt="" onerror="this.style.background='var(--surface2)';this.src=''">`
    : `<div class="np-card-thumb" style="display:flex;align-items:center;justify-content:center;font-size:1.5rem;background:var(--surface2)">&#128240;</div>`;
  const heroHtml = heroImg ? `<img class="np-card-hero" src="${escHtml(heroImg)}" alt="" onerror="this.style.display='none'">` : '';
  const swClass = soWhatClass ? ` ${soWhatClass}` : '';
  const swHtml = soWhat ? `<div class="np-card-so-what${swClass}"><b>${soWhatClass==='internal'?'Implication:':'Why this matters:'}</b> ${escHtml(soWhat)}</div>` : '';
  return `<div class="np-card" id="${id}" onclick="npToggle('${id}')">
    <div class="np-card-header">
      ${thumbHtml}
      <div class="np-card-info">
        <div class="np-card-title">${escHtml(title)}</div>
        <div class="np-card-snippet">${escHtml(snippet)}</div>
        <div class="np-card-badges">${badges}</div>
      </div>
    </div>
    <div class="np-card-expand">
      ${heroHtml}
      <div class="np-card-body">${fullBody}</div>
      ${swHtml}
      ${sourceLine}
    </div>
  </div>`;
}

function _cleanEditorial(text) {
  if (!text) return '';
  return text.split('\n')
    .filter(l => !l.trim().match(/^(Hero image:|alt:|https?:\/\/)/i))
    .join('\n').trim();
}

function renderNewsPulse(data) {
  const topics=data.topics||[];
  const vault=data.vault_topics||[];
  const internal=data.internal_signals||[];
  const srcIcons={email:'&#9993;',meeting:'&#128197;',teams:'&#128172;',activity:'&#128187;'};
  let totalArticles=0;
  let h='';

  const ehData = data.executive_header || {
    bluf: data.bottom_line || '',
    hero_image: data.hero_image || '',
    metrics: [
      {label:'Stories',value:data.story_count||0},
      {label:'Topics',value:data.topic_count||0},
      {label:'Internal Signals',value:internal.length,color:internal.some(s=>s.urgency==='high')?'red':'accent'}
    ]
  };
  if(ehData.bluf || data.hero_image) h += renderExecutiveHeader({...data, executive_header:ehData}, 'news-pulse');

  if(internal.length) {
    h+=`<div class="np-section-label internal">Internal Signals (${internal.length})</div>`;
    internal.forEach((sig,si)=>{
      totalArticles++;
      const icon=srcIcons[sig.source_type]||'&#8226;';
      const urgPill=sig.urgency?`<span class="rv-pill ${sig.urgency==='high'?'rv-pill-red':sig.urgency==='medium'?'rv-pill-yellow':'rv-pill-blue'}">${sig.urgency}</span> `:'';
      const srcBadge=`<span style="font-size:.7rem;color:var(--muted)">${icon} ${escHtml(sig.source_detail||sig.source_type||'')}</span>`;
      const thumb=_npThumb(sig);
      h+=_npCard(
        'npi-'+si, thumb, sig.title||'', (sig.summary||'').slice(0,150),
        urgPill+srcBadge, thumb,
        escHtml(sig.summary||''), sig.so_what||'', 'internal',
        `<div class="np-card-source">${icon} ${escHtml(sig.source_detail||'')}</div>`
      );
    });
  }

  topics.forEach((t,ti)=>{
    h+=`<div class="np-section-label">${escHtml(t.name||t.category_label||'')}</div>`;
    if(t.editorial_frame) h+=`<div class="np-editorial">${escHtml(_cleanEditorial(t.editorial_frame))}</div>`;
    (t.articles||[]).forEach((a,ai)=>{
      totalArticles++;
      const thumb=_npThumb(a);
      const srcPill=a.source?`<span style="font-size:.68rem;color:var(--muted);background:var(--surface2);padding:2px 8px;border-radius:10px">${escHtml(a.source)}</span>`:'';
      const datePill=a.date?`<span style="font-size:.68rem;color:var(--muted)">${escHtml(a.date)}</span>`:'';
      const link=a.url?`<div class="np-card-source"><a href="${escHtml(a.url)}" target="_blank" onclick="event.stopPropagation()">Read full article &rarr;</a> <span>${escHtml(a.source||'')} &middot; ${escHtml(a.date||'')}</span></div>`:'';
      h+=_npCard(
        'npa-'+ti+'-'+ai, thumb, a.title||'', (a.summary||a.snippet||'').slice(0,150),
        srcPill+' '+datePill, thumb,
        escHtml(a.summary||''), a.so_what||'', '',
        link
      );
    });
  });

  vault.forEach((vt,vi)=>{
    h+=`<div class="np-section-label" style="color:var(--blue);border-color:var(--blue)">From Your Vault: ${escHtml(vt.name||'')}</div>`;
    if(vt.source_description) h+=`<div class="np-editorial">${escHtml(vt.source_description)}</div>`;
    (vt.articles||[]).forEach((a,ai)=>{
      totalArticles++;
      const thumb=_npThumb(a);
      const link=a.url?`<div class="np-card-source"><a href="${escHtml(a.url)}" target="_blank" onclick="event.stopPropagation()">Read full article &rarr;</a></div>`:'';
      h+=_npCard(
        'npv-'+vi+'-'+ai, thumb, a.title||'', (a.summary||a.snippet||'').slice(0,150),
        '', thumb,
        escHtml(a.summary||a.snippet||''), a.so_what||'', '',
        link
      );
    });
  });

  const tp = data.talking_points || [];
  if(tp.length) h += renderTalkingPoints(tp);

  const countText=`${data.story_count||totalArticles} stories across ${data.topic_count||topics.length} topics`;
  return `${h?'<div class="np-feed">'+h+'</div>':'<div style="margin-bottom:16px;font-size:.84rem;color:var(--muted);text-align:center">'+countText+'</div><div class="np-feed">'+h+'</div>'}`;
}

function npToggle(id) {
  const el=document.getElementById(id);
  if(!el) return;
  if(el.classList.contains('expanded')) {
    el.classList.remove('expanded');
  } else {
    el.classList.add('expanded');
    el.scrollIntoView({behavior:'smooth',block:'nearest'});
  }
}

/* ═══ Rich View: Weekly Status ═══ */
function renderWeeklyStatus(data) {
  const m=data.metrics||{};
  const cs=m.commitment_score||{};
  let h=``;

  const wow=data.week_over_week||data.trend_vs_prior_week||{};
  const ehData = data.executive_header || {
    bluf: data.summary || '',
    hero_image: data.hero_image || '',
    status: (cs.percent||0) >= 75 ? 'green' : (cs.percent||0) >= 50 ? 'yellow' : 'red',
    metrics: [
      {label:'Commitment Score',value:(cs.percent||0)+'%',color:(cs.percent||0)>=75?'green':(cs.percent||0)>=50?'yellow':'red'},
      {label:'Accomplishments',value:m.accomplishments_count||0,color:'green'},
      {label:'Meetings',value:m.meetings_total||0,trend:wow.meetings_delta},
      {label:'Emails Sent',value:m.emails_sent||0,trend:wow.emails_delta}
    ]
  };
  h += renderExecutiveHeader({...data, executive_header:ehData}, 'weekly-status');

  const vd=data.value_delivered||[];
  if(vd.length) {
    h+=`<div class="rv-section"><div class="rv-section-title">Impact Delivered</div>`;
    vd.forEach(v=>{
      h+=`<div class="rv-card" style="border-left:4px solid var(--green)"><div class="rv-card-title">${escHtml(v.statement||'')}</div>
        <div class="rv-card-body">${escHtml(v.detail||'')}</div>
        ${v.evidence_path?`<div class="rv-card-meta">${escHtml(v.evidence_path)}</div>`:''}</div>`;
    });
    h+=`</div>`;
  }

  h+=renderAnalysisCard(data.analysis);
  if(wow.meetings_delta!=null||wow.emails_delta!=null) {
    h+=`<div class="rv-metrics">`;
    if(wow.meetings_delta!=null) {const c=wow.meetings_delta<=0?'var(--green)':'var(--red)';h+=`<div class="rv-metric"><div class="rv-metric-value" style="color:${c}">${wow.meetings_delta>=0?'+':''}${wow.meetings_delta}</div><div class="rv-metric-label">Meetings WoW</div></div>`;}
    if(wow.emails_delta!=null) h+=`<div class="rv-metric"><div class="rv-metric-value">${wow.emails_delta>=0?'+':''}${wow.emails_delta}</div><div class="rv-metric-label">Emails WoW</div></div>`;
    if(wow.accomplishments_delta!=null) {const c=wow.accomplishments_delta>=0?'var(--green)':'var(--red)';h+=`<div class="rv-metric"><div class="rv-metric-value" style="color:${c}">${wow.accomplishments_delta>=0?'+':''}${wow.accomplishments_delta}</div><div class="rv-metric-label">Accomplishments WoW</div></div>`;}
    h+=`</div>`;
  }
  h+=`<div class="rv-cols">
    <div class="rv-chart-container"><div class="rv-section-title">Daily Meetings</div><canvas id="ws-meetings-chart"></canvas></div>
    <div class="rv-chart-container"><div class="rv-section-title">Time Allocation</div><canvas id="ws-time-chart"></canvas></div>
  </div>`;
  const accs=data.accomplishments||[];
  if(accs.length) {
    h+=`<div class="rv-section"><div class="rv-section-title">Key Accomplishments</div>`;
    accs.forEach(a=>{
      const catPill=a.category?`<span class="rv-pill rv-pill-blue">${escHtml(a.category)}</span>`:'';
      h+=`<div class="rv-card">${catPill}<div class="rv-card-title">${escHtml(a.title||'')}</div>
        <div class="rv-card-body">${escHtml(a.impact||'')}</div>
        ${a.evidence_path?`<div class="rv-card-meta">${escHtml(a.evidence_path)}</div>`:''}</div>`;
    });
    h+=`</div>`;
  }
  const pva=data.plan_vs_actual||[];
  if(pva.length) {
    h+=`<div class="rv-section"><div class="rv-section-title">Plan vs Actual</div>`;
    pva.forEach(p=>{
      const sc={'Delivered':'rv-pill-green','In Progress':'rv-pill-yellow','Missed':'rv-pill-red','Deferred':'rv-pill-gray'};
      h+=`<div class="rv-card" style="display:flex;align-items:center;gap:14px">
        <span class="rv-pill ${sc[p.status]||'rv-pill-gray'}">${escHtml(p.status||'')}</span>
        <div style="flex:1"><div class="rv-card-title">${escHtml(p.item||'')}</div>
        <div class="rv-card-meta">${escHtml(p.source||'')} ${p.evidence?'— '+escHtml(p.evidence):''}</div></div></div>`;
    });
    h+=`</div>`;
  }
  const blk=data.blockers||[];
  if(blk.length) {
    h+=`<div class="rv-section"><div class="rv-section-title">Blockers & Risks</div>`;
    blk.forEach(b=>{
      h+=`<div class="rv-card rv-severity-${b.severity||'medium'}"><div class="rv-card-title">${escHtml(b.title||'')}</div>
        <div class="rv-card-body">${escHtml(b.impact||'')}</div>
        ${b.mitigation?`<div class="rv-card-meta" style="color:var(--green)">Mitigation: ${escHtml(b.mitigation)}</div>`:''}</div>`;
    });
    h+=`</div>`;
  }
  const nw=data.next_week||[];
  if(nw.length) {
    h+=`<div class="rv-section"><div class="rv-section-title">Next Week Focus</div>`;
    nw.forEach(n=>{
      h+=`<div class="rv-card"><div class="rv-card-title">${escHtml(n.item||'')}</div>
        <div class="rv-card-meta">${escHtml(n.date||'')} ${n.evidence_path?'— '+escHtml(n.evidence_path):''}</div></div>`;
    });
    h+=`</div>`;
  }
  return h;
}
function initWeeklyStatusCharts(data) {
  const dm=data.daily_meetings||[];
  if(dm.length&&typeof Chart!=='undefined') {
    const ctx1=document.getElementById('ws-meetings-chart');
    if(ctx1) new Chart(ctx1,{type:'bar',data:{labels:dm.map(d=>d.day||d.date),datasets:[{label:'Meetings',data:dm.map(d=>d.count||0),backgroundColor:'rgba(108,124,255,.6)',borderRadius:6},{label:'Hours',data:dm.map(d=>d.hours||0),backgroundColor:'rgba(74,222,128,.5)',borderRadius:6}]},options:{responsive:true,plugins:{legend:{labels:{color:'#9ca3af'}}},scales:{x:{ticks:{color:'#9ca3af'},grid:{color:'rgba(255,255,255,.05)'}},y:{ticks:{color:'#9ca3af'},grid:{color:'rgba(255,255,255,.05)'}}}}});
    const m=data.metrics||{};
    const totalMeetingH=dm.reduce((s,d)=>s+(d.hours||0),0);
    const focusH=Math.max(0,40-totalMeetingH);
    const ctx2=document.getElementById('ws-time-chart');
    if(ctx2) new Chart(ctx2,{type:'doughnut',data:{labels:['Meetings','Focus Work','Email/Comms'],datasets:[{data:[totalMeetingH,focusH,Math.min(10,(m.emails_received||0)/10)],backgroundColor:['rgba(108,124,255,.7)','rgba(74,222,128,.7)','rgba(250,204,21,.7)'],borderWidth:0}]},options:{responsive:true,plugins:{legend:{labels:{color:'#9ca3af'}}}}});
  }
}

/* ═══ Rich View: Plan My Week ═══ */
function renderPlanMyWeek(data) {
  let h='';

  const dels=data.deliverables||[];
  const ehData = data.executive_header || {
    bluf: data.this_weeks_bet || data.executive_summary || '',
    hero_image: data.hero_image || '',
    status: (data.week_score||0) >= 70 ? 'green' : (data.week_score||0) >= 45 ? 'yellow' : 'red',
    metrics: [
      {label:'Week Score',value:data.week_score||0,color:(data.week_score||0)>=70?'green':(data.week_score||0)>=45?'yellow':'red'},
      {label:'Meeting Hours',value:(data.total_meeting_hours||0).toFixed(1)+'h'},
      {label:'Focus Hours',value:(data.total_focus_hours||0).toFixed(1)+'h',color:'green'},
      {label:'Deliverables',value:dels.length}
    ]
  };
  h += renderExecutiveHeader({...data, executive_header:ehData}, 'plan-my-week');

  const wp=data.week_progress;
  if(wp) {
    h+=`<div class="rv-week-progress">
      <div class="rv-week-progress-label">Day ${wp.current_day} of ${wp.total_days}</div>
      <div class="rv-week-progress-track"><div class="rv-week-progress-fill" style="width:${wp.percent}%"></div></div>
      <div class="rv-week-progress-pct">${wp.percent}% through week</div>
    </div>`;
  }

  h+=renderAnalysisCard(data.analysis);

  /* Metrics: Planned vs Actual */
  const tvp=data.trend_vs_prior_week||{};
  const hasActuals=data.actual_total_meeting_hours!=null;
  h+=`<div class="rv-metrics">`;
  if(tvp.delta_meetings_hours!=null) { const c=tvp.delta_meetings_hours<=0?'var(--green)':'var(--red)';h+=`<div class="rv-metric"><div class="rv-metric-value" style="color:${c}">${tvp.delta_meetings_hours>=0?'+':''}${tvp.delta_meetings_hours.toFixed(1)}h</div><div class="rv-metric-label">Meetings vs Last Week</div></div>`; }
  if(tvp.delta_focus_hours!=null) { const c=tvp.delta_focus_hours>=0?'var(--green)':'var(--red)';h+=`<div class="rv-metric"><div class="rv-metric-value" style="color:${c}">${tvp.delta_focus_hours>=0?'+':''}${tvp.delta_focus_hours.toFixed(1)}h</div><div class="rv-metric-label">Focus vs Last Week</div></div>`; }
  if(data.total_meeting_hours!=null) {
    const pm=data.total_meeting_hours, am=data.actual_total_meeting_hours;
    h+=`<div class="rv-metric"><div class="rv-metric-value">${pm.toFixed(1)}h</div><div class="rv-metric-label">Total Meetings</div>`;
    if(hasActuals) {
      const d=am-pm, cls=d<=0?'rv-metric-delta-good':'rv-metric-delta-bad';
      h+=`<div class="rv-metric-compare"><span class="rv-metric-planned">Plan ${pm.toFixed(1)}h</span><span class="rv-metric-actual">Actual <strong>${am.toFixed(1)}h</strong></span><span class="rv-metric-delta ${cls}">${d>=0?'+':''}${d.toFixed(1)}</span></div>`;
    }
    h+=`</div>`;
  }
  if(data.total_focus_hours!=null) {
    const pf=data.total_focus_hours, af=data.actual_total_focus_hours;
    h+=`<div class="rv-metric"><div class="rv-metric-value" style="color:var(--green)">${pf.toFixed(1)}h</div><div class="rv-metric-label">Total Focus</div>`;
    if(hasActuals) {
      const d=af-pf, cls=d>=0?'rv-metric-delta-good':'rv-metric-delta-bad';
      h+=`<div class="rv-metric-compare"><span class="rv-metric-planned">Plan ${pf.toFixed(1)}h</span><span class="rv-metric-actual">Actual <strong>${af.toFixed(1)}h</strong></span><span class="rv-metric-delta ${cls}">${d>=0?'+':''}${d.toFixed(1)}</span></div>`;
    }
    h+=`</div>`;
  }
  h+=`</div>`;

  /* Day Cards with status indicators and actuals */
  const days=data.days||[];
  if(days.length) {
    h+=`<div class="rv-day-strip">`;
    days.forEach((d,i)=>{
      const mc=d.meetings||{};
      const dens=d.density||'moderate';
      const cap=d.capacity_percent||0;
      const ds=d.day_status||'future';
      const statusCls=ds==='completed'?'rv-day-status-completed':ds==='in_progress'?'rv-day-status-today':'rv-day-status-future';
      h+=`<div class="rv-day-card rv-day-density-${dens}" onclick="showDayDetail(${i})">
        <div class="rv-day-card-name">${escHtml(d.day_name||'')}</div>
        <div class="rv-day-card-date">${escHtml(d.date||'')}</div>
        <div class="rv-day-status ${statusCls}"></div>
        <div class="rv-day-card-meetings" style="color:${dens==='heavy'?'var(--red)':dens==='moderate'?'#facc15':'var(--green)'}">${mc.count||0}</div>
        <div class="rv-day-card-hours">${mc.hours||0}h mtgs</div>
        ${renderProgressBar(cap, cap>75?'red':cap>50?'yellow':'green')}
        <div style="font-size:.66rem;color:var(--muted);margin-top:2px">${Math.round(cap)}% capacity</div>`;
      if(ds!=='future'&&d.actual_meeting_hours!=null) {
        const amh=d.actual_meeting_hours, pmh=mc.hours||0;
        const mCls=amh<=pmh?'rv-day-actual-good':'rv-day-actual-bad';
        const afh=d.actual_focus_hours||0, pfh=d.focus_hours||0;
        const fCls=afh>=pfh?'rv-day-actual-good':'rv-day-actual-bad';
        h+=`<div class="rv-day-actual">
          <div>Actual: <span class="rv-day-actual-val ${mCls}">${amh}h</span> mtg</div>
          <div>Focus: <span class="rv-day-actual-val ${fCls}">${afh}h</span></div>
        </div>`;
      }
      h+=`</div>`;
    });
    h+=`</div>`;
    h+=`<div class="rv-chart-container"><div class="rv-section-title">Daily Meeting vs Focus (Planned vs Actual)</div><canvas id="pmw-capacity-chart"></canvas></div>`;
  }

  /* Priorities */
  const pris=data.priorities||[];
  if(pris.length) {
    h+=`<div class="rv-section"><div class="rv-section-title">Priorities</div>`;
    pris.forEach(p=>{
      const sc={'On Track':'rv-pill-green','At Risk':'rv-pill-yellow','Overdue':'rv-pill-red','In Progress':'rv-pill-blue','Not Started':'rv-pill-gray'};
      h+=`<div class="rv-card" style="display:flex;align-items:center;gap:14px">
        <span class="rv-pill ${sc[p.status]||'rv-pill-gray'}">${escHtml(p.status||'')}</span>
        <div style="flex:1"><div class="rv-card-title">${escHtml(p.name||'')}</div>
        <div class="rv-card-meta">Owner: ${escHtml(p.owner||'--')} | Deadline: ${escHtml(p.deadline||'--')}</div></div></div>`;
    });
    h+=`</div>`;
  }

  /* Deliverables Timeline with today marker and status coloring */
  if(dels.length) {
    h+=`<div class="rv-section"><div class="rv-section-title">Deliverables Timeline</div><div class="rv-gantt">`;
    const dayNames=days.map(d=>d.day_name||d.date);
    const todayIdx=days.findIndex(d=>d.day_status==='in_progress');
    h+=`<div class="rv-gantt-row"><div class="rv-gantt-label"></div><div class="rv-gantt-track" style="display:flex;background:none;gap:2px;position:relative">`;
    dayNames.forEach((dn,di)=>{
      const ds=(days[di]||{}).day_status||'future';
      const bg=ds==='completed'?'rgba(74,222,128,.08)':ds==='in_progress'?'rgba(108,124,255,.08)':'transparent';
      h+=`<div style="flex:1;text-align:center;font-size:.7rem;color:${ds==='in_progress'?'var(--accent)':'var(--muted)'};font-weight:${ds==='in_progress'?'700':'400'};background:${bg};border-radius:4px;padding:2px 0">${escHtml(dn.slice(0,3))}</div>`;
    });
    h+=`</div></div>`;
    dels.forEach(dl=>{
      const dayIdx=days.findIndex(d=>d.day_name===dl.planned_day||d.date===dl.planned_day);
      const left=dayIdx>=0?(dayIdx/Math.max(days.length,1)*100)+'%':'0%';
      const isOverdue=dl.status==='Overdue'||dl.status==='overdue';
      const isDue=dl.status==='Due';
      const barColor=isOverdue?'var(--red)':isDue?'#facc15':'var(--accent)';
      const isPast=dayIdx>=0&&days[dayIdx]&&(days[dayIdx].day_status==='completed'||days[dayIdx].day_status==='in_progress');
      const statusLabel=isPast&&isOverdue?'OVD':isPast?'...':dl.status?(dl.status).slice(0,3):'';
      h+=`<div class="rv-gantt-row"><div class="rv-gantt-label" title="${escHtml(dl.name)}">${escHtml(dl.name||'')}</div>
        <div class="rv-gantt-track" style="position:relative">
          <div class="rv-gantt-bar" style="background:${barColor};width:${100/Math.max(days.length,1)}%;margin-left:${left};opacity:${isPast?'0.5':'1'}">${escHtml(statusLabel)}</div>`;
      if(todayIdx>=0) {
        const markerLeft=((todayIdx+0.5)/Math.max(days.length,1)*100)+'%';
        h+=`<div class="rv-gantt-today-marker" style="left:${markerLeft}"></div>`;
      }
      h+=`</div></div>`;
    });
    h+=`</div>`;
    h+=`<div class="rv-gantt-legend"><span><span class="rv-gantt-legend-dot" style="background:var(--accent)"></span>Planned</span><span><span class="rv-gantt-legend-dot" style="background:var(--red)"></span>Overdue</span><span><span class="rv-gantt-legend-dot" style="background:#facc15"></span>Due</span></div>`;
    h+=`</div>`;
  }

  h+=`<div id="pmw-day-detail"></div>`;
  window._pmwData=data;
  const cyo=data.commitments_you_owe||[];
  const coty=data.commitments_owed_to_you||[];
  if(cyo.length||coty.length) {
    h+=`<div class="rv-cols"><div class="rv-section"><div class="rv-section-title">You Owe Others</div>`;
    cyo.forEach(c=>{
      h+=`<div class="rv-card rv-severity-warning"><div class="rv-card-title">${escHtml(c.commitment||'')}</div>
        <div class="rv-card-meta">To: ${escHtml(c.owed_to||'')} | Deadline: ${escHtml(c.deadline||'')}</div></div>`;
    });
    h+=`</div><div class="rv-section"><div class="rv-section-title">Others Owe You</div>`;
    coty.forEach(c=>{
      h+=`<div class="rv-card"><div class="rv-card-title">${escHtml(c.person||'')}: ${escHtml(c.commitment||'')}</div>
        <div class="rv-card-meta">Deadline: ${escHtml(c.deadline||'')}</div></div>`;
    });
    h+=`</div></div>`;
  }
  const risks=data.risks||[];
  if(risks.length) {
    h+=`<div class="rv-section"><div class="rv-section-title">Risks & Watch Items</div>`;
    risks.forEach(r=>{
      h+=`<div class="rv-card rv-severity-${r.severity||'medium'}"><div class="rv-card-title">${escHtml(r.title||'')}</div>
        <div class="rv-card-body">${escHtml(r.detail||'')}</div></div>`;
    });
    h+=`</div>`;
  }
  const opts=data.optimization_suggestions||[];
  if(opts.length) {
    h+=`<div class="rv-section"><div class="rv-section-title">Optimization Suggestions</div>`;
    opts.forEach(o=>{h+=`<div class="rv-card" style="border-left:3px solid var(--green)"><div class="rv-card-body" style="font-size:.86rem">${escHtml(o)}</div></div>`;});
    h+=`</div>`;
  }
  return h;
}
function showDayDetail(idx) {
  document.querySelectorAll('.rv-day-card').forEach((e,i)=>e.classList.toggle('active',i===idx));
  const d=(window._pmwData?.days||[])[idx];
  if(!d) return;
  const el=document.getElementById('pmw-day-detail');
  const mc=d.meetings||{};
  const ds=d.day_status||'future';
  const statusLabel=ds==='completed'?'Completed':ds==='in_progress'?'In Progress':'Upcoming';
  let h=`<div class="rv-section" style="margin-top:16px"><div class="rv-section-title">${escHtml(d.day_name||'')} ${escHtml(d.date||'')} <span class="rv-pill ${ds==='completed'?'rv-pill-green':ds==='in_progress'?'rv-pill-blue':'rv-pill-gray'}" style="margin-left:8px;vertical-align:middle">${statusLabel}</span></div>`;

  if(ds!=='future'&&d.actual_meeting_hours!=null) {
    h+=`<div class="rv-metrics" style="margin-bottom:16px">
      <div class="rv-metric"><div class="rv-metric-value">${mc.hours||0}h</div><div class="rv-metric-label">Planned Meetings</div></div>
      <div class="rv-metric"><div class="rv-metric-value" style="color:${d.actual_meeting_hours<=(mc.hours||0)?'var(--green)':'var(--red)'}">${d.actual_meeting_hours}h</div><div class="rv-metric-label">Actual Meetings</div></div>
      <div class="rv-metric"><div class="rv-metric-value">${d.focus_hours||0}h</div><div class="rv-metric-label">Planned Focus</div></div>
      <div class="rv-metric"><div class="rv-metric-value" style="color:${(d.actual_focus_hours||0)>=(d.focus_hours||0)?'var(--green)':'var(--red)'}">${d.actual_focus_hours||0}h</div><div class="rv-metric-label">Actual Focus</div></div>
      ${d.actual_email_count?`<div class="rv-metric"><div class="rv-metric-value">${d.actual_email_count}</div><div class="rv-metric-label">Emails</div></div>`:''}
    </div>`;
  }

  if((mc.names||[]).length) {
    const attended=d.actual_meetings_attended||[];
    h+=`<div class="rv-card"><div class="rv-card-title">Meetings (${mc.count||0}, ${mc.hours||0}h)</div><ul style="margin:8px 0 0 16px;font-size:.86rem">`;
    (mc.names||[]).forEach(n=>{
      const wasAttended=attended.some(a=>a.toLowerCase().includes(n.toLowerCase())||n.toLowerCase().includes(a.toLowerCase()));
      const icon=ds!=='future'?(wasAttended?'<span style="color:var(--green);margin-right:4px" title="Attended">&#10003;</span>':'<span style="color:var(--muted);margin-right:4px" title="No transcript">&#8211;</span>'):'';
      h+=`<li>${icon}${escHtml(n)}</li>`;
    });
    h+=`</ul></div>`;
  }
  if((d.priority_work||[]).length) {
    h+=`<div class="rv-card"><div class="rv-card-title">Priority Work</div><ul style="margin:8px 0 0 16px;font-size:.86rem">`;
    (d.priority_work||[]).forEach(w=>{h+=`<li>${escHtml(w)}</li>`;});
    h+=`</ul></div>`;
  }
  if((d.follow_ups||[]).length) {
    h+=`<div class="rv-card"><div class="rv-card-title">Follow-ups</div><ul style="margin:8px 0 0 16px;font-size:.86rem">`;
    (d.follow_ups||[]).forEach(f=>{h+=`<li>${escHtml(f)}</li>`;});
    h+=`</ul></div>`;
  }
  if((d.constraints||[]).length) {
    h+=`<div class="rv-card rv-severity-warning"><div class="rv-card-title">Constraints</div><ul style="margin:8px 0 0 16px;font-size:.86rem">`;
    (d.constraints||[]).forEach(c=>{h+=`<li>${escHtml(c)}</li>`;});
    h+=`</ul></div>`;
  }
  h+=`</div>`;
  el.innerHTML=h;
}

/* ═══ Rich View: Morning Brief ═══ */
function renderMorningBrief(data) {
  let h='';

  const ehData = data.executive_header || {
    bluf: data.day_summary || '',
    status: (data.day_score||0) >= 70 ? 'green' : (data.day_score||0) >= 45 ? 'yellow' : 'red',
    hero_image: data.hero_image || '',
    metrics: [
      {label:'Day Score',value:data.day_score||0,color:(data.day_score||0)>=70?'green':(data.day_score||0)>=45?'yellow':'red'},
      {label:'Meetings',value:data.meeting_count||0},
      {label:'Need Reply',value:data.unread_reply_count||0,color:(data.unread_reply_count||0)>3?'red':'accent'},
      {label:'Commitments',value:data.commitment_count||0}
    ]
  };
  h += renderExecutiveHeader({...data, executive_header:ehData}, 'morning-brief');

  const tm = data.three_moves || data.quick_wins || [];
  if(tm.length) h += renderThreeMoves(tm.slice(0,3));

  h+=renderAnalysisCard(data.analysis);
  const em=data.energy_map||[];
  if(em.length) {
    h+=`<div class="rv-section"><div class="rv-section-title">Energy Map</div><div class="rv-energy-map">`;
    em.forEach(slot=>{
      const cls={'open':'rv-energy-open','meeting':'rv-energy-meeting','transition':'rv-energy-transition'}[slot.availability]||'rv-energy-transition';
      h+=`<div class="rv-energy-slot ${cls}" data-tip="${escHtml(slot.hour||'')}: ${escHtml(slot.suggested_use||slot.availability||'')}">${escHtml((slot.hour||'').replace(':00',''))}</div>`;
    });
    h+=`</div></div>`;
  }
  h+=`<div class="rv-cols">`;
  const dc=data.day_composition||{};
  if(dc.meeting_percent||dc.focus_percent) h+=`<div class="rv-chart-container"><div class="rv-section-title">Day Composition</div><canvas id="mb-comp-chart"></canvas></div>`;
  const qw=data.quick_wins||[];
  if(qw.length) {
    h+=`<div class="rv-quick-wins" style="align-content:start">`;
    qw.forEach(q=>{h+=`<div class="rv-quick-win"><div class="rv-quick-win-action">${escHtml(q.action||'')}</div><div class="rv-quick-win-impact">${escHtml(q.impact||'')}</div><div class="rv-quick-win-time">${escHtml(q.time_needed||'')}</div></div>`;});
    h+=`</div>`;
  }
  h+=`</div>`;
  /* Conflicts & Recommendations */
  const conflicts=data.conflicts||[];
  if(conflicts.length) {
    h+=`<div class="rv-section"><div class="rv-section-title" style="color:var(--red)">Scheduling Conflicts (${conflicts.length})</div>`;
    conflicts.forEach(c=>{
      h+=`<div class="rv-card rv-severity-high" style="margin-bottom:12px">`;
      h+=`<div class="rv-card-title" style="display:flex;align-items:center;gap:8px"><span style="font-size:1.1rem">&#9888;</span> ${escHtml(c.time_range||'')}</div>`;
      h+=`<div style="margin:8px 0"><b>Overlapping:</b></div><ul style="margin:0 0 0 16px;font-size:.86rem">`;
      (c.meetings||[]).forEach(m=>{h+=`<li>${escHtml(m)}</li>`;});
      h+=`</ul>`;
      if(c.recommendation) h+=`<div style="margin-top:10px;padding:10px 14px;background:rgba(108,124,255,.08);border-radius:8px;font-size:.86rem"><b style="color:var(--accent)">Recommendation:</b> ${escHtml(c.recommendation)}</div>`;
      if(c.delegate_action) h+=`<div style="margin-top:6px;font-size:.82rem;color:var(--muted)"><b>Action:</b> ${escHtml(c.delegate_action)}</div>`;
      h+=`</div>`;
    });
    h+=`</div>`;
  }
  /* Prep Tonight */
  const prep=data.prep_tonight||[];
  if(prep.length) {
    h+=`<div class="rv-section"><div class="rv-section-title">Prep Tonight (Top ${prep.length})</div>`;
    prep.sort((a,b)=>(a.priority||5)-(b.priority||5));
    prep.forEach((p,i)=>{
      const urgColor=p.priority<=2?'var(--red)':p.priority<=3?'#facc15':'var(--green)';
      h+=`<div class="rv-card" style="display:flex;gap:14px;align-items:start">
        <div style="width:32px;height:32px;border-radius:50%;background:${urgColor};display:flex;align-items:center;justify-content:center;color:#fff;font-weight:800;font-size:.9rem;flex-shrink:0">${i+1}</div>
        <div style="flex:1"><div class="rv-card-title">${escHtml(p.task||'')}</div>
        <div class="rv-card-meta">${p.related_meeting?'For: '+escHtml(p.related_meeting)+' &middot; ':''} ${escHtml(p.time_needed||'')}</div></div></div>`;
    });
    h+=`</div>`;
  }
  const signals=data.signals||[];
  if(signals.length) {
    h+=`<div class="rv-section">`;
    signals.forEach(s=>{
      const icon={'deadline':'&#9200;','escalation':'&#9888;','conflict':'&#9888;','risk':'&#9888;','opportunity':'&#10024;'}[s.type]||'&#8226;';
      h+=`<div class="rv-card rv-severity-${s.severity||'warning'}" style="display:flex;align-items:center;gap:10px">
        <span style="font-size:1.2rem">${icon}</span>
        <div><div class="rv-card-title" style="font-size:.88rem">${escHtml(s.message||'')}</div></div></div>`;
    });
    h+=`</div>`;
  }
  const pfa=data.priority_focus_areas||[];
  if(pfa.length) {
    h+=`<div class="rv-section"><div class="rv-section-title">Priority Focus Areas</div>`;
    pfa.forEach(p=>{
      const uc={'critical':'rv-pill-red','high':'rv-pill-yellow','medium':'rv-pill-blue'};
      h+=`<div class="rv-card" style="display:flex;align-items:start;gap:12px">
        <span class="rv-pill ${uc[p.urgency]||'rv-pill-blue'}">${escHtml(p.urgency||'')}</span>
        <div><div class="rv-card-title">${escHtml(p.name||'')}</div>
        <div class="rv-card-body">${escHtml(p.detail||'')}</div></div></div>`;
    });
    h+=`</div>`;
  }
  const allDayItems=(data.meetings||[]).filter(m=>m.time==='00:00-23:59'||m.time==='All Day'||m.all_day);
  const meetings=(data.meetings||[]).filter(m=>m.time!=='00:00-23:59'&&m.time!=='All Day'&&!m.all_day);
  const totalCalItems = meetings.length + allDayItems.length;
  if(totalCalItems) {
    h+=`<div class="rv-section"><div class="rv-section-title">Today's Schedule (${totalCalItems} items)</div>`;
    if(allDayItems.length) {
      h+=`<div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:16px">`;
      allDayItems.forEach(ad=>{
        const t = ad.title || ad.name || ad.meeting || ad.summary || '';
        h+=`<div class="rv-card" style="flex:1;min-width:180px;border-left:4px solid var(--blue);margin-bottom:0"><div class="rv-card-meta" style="margin-bottom:2px">All Day</div><div class="rv-card-title" style="font-size:.88rem">${escHtml(t)}</div></div>`;
      });
      h+=`</div>`;
    }
    const startH=7,endH=20;
    h+=`<div style="background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:12px 16px;margin-bottom:20px;overflow:hidden">`;
    for(let hr=startH;hr<endH;hr++){
      const hrLabel=hr<10?'0'+hr:hr;
      let mtgsInHr=0;
      meetings.forEach(mtg=>{
        const t=mtg.time||'';const parts=t.split('-');if(parts.length<2)return;
        const [sh]=(parts[0]||'').split(':').map(Number);const [eh,em]=(parts[1]||'').split(':').map(Number);
        if(!isNaN(sh)&&!isNaN(eh)&&!(sh>hr||eh<hr||(eh===hr&&em===0)))mtgsInHr++;
      });
      const isConflict=mtgsInHr>1;
      h+=`<div style="display:flex;min-height:38px;border-bottom:1px solid var(--border);${isConflict?'background:repeating-linear-gradient(45deg,transparent,transparent 4px,rgba(248,113,113,.08) 4px,rgba(248,113,113,.08) 8px);':''}">`;
      h+=`<div style="width:44px;flex-shrink:0;font-size:.72rem;color:var(--muted);padding-top:2px;text-align:right;padding-right:8px">${hrLabel}:00${isConflict?'<span style="color:var(--red);font-weight:700" title="Conflict"> !</span>':''}</div>`;
      h+=`<div style="flex:1;position:relative;display:flex;gap:3px;padding:2px 0">`;
      meetings.forEach(mtg=>{
        const t=mtg.time||'';
        const parts=t.split('-');if(parts.length<2) return;
        const [sh,sm]=(parts[0]||'').split(':').map(Number);
        const [eh,em]=(parts[1]||'').split(':').map(Number);
        if(isNaN(sh)||isNaN(eh)) return;
        if(sh>hr||eh<hr||(eh===hr&&em===0)) return;
        const pc={'high':'rgba(248,113,113,.65)','medium':'rgba(250,204,21,.5)','low':'rgba(108,124,255,.4)'};
        const bg=pc[mtg.priority]||'rgba(108,124,255,.35)';
        const fullTitle = mtg.title || mtg.name || mtg.meeting || mtg.summary || '';
        const shortTitle=(fullTitle||'').length>35?(fullTitle||'').slice(0,33)+'...':(fullTitle||'');
        h+=`<div style="flex:1;background:${bg};border-radius:5px;padding:2px 8px;font-size:.7rem;font-weight:600;color:#fff;display:flex;align-items:center;overflow:hidden;white-space:nowrap;cursor:pointer" onclick="document.getElementById('mb-mtg-${meetings.indexOf(mtg)}')?.scrollIntoView({behavior:'smooth',block:'nearest'})" title="${escHtml(fullTitle||'')} (${t})">${sh===hr?escHtml(shortTitle):''}</div>`;
      });
      h+=`</div></div>`;
    }
    h+=`</div>`;
    /* Expandable Meeting Prep Cards */
    h+=`<div class="rv-section-title" style="margin-top:20px">Meeting Prep Cards</div>`;
    meetings.forEach((mtg,mi)=>{
      const pc={'high':'var(--red)','medium':'#facc15','low':'var(--accent)'};
      const accentColor=pc[mtg.priority]||'var(--accent)';
      const attendees=mtg.attendees||[];
      const actions=mtg.action_items||[];
      const talkPts=mtg.talking_points||[];
      const detId='mb-mtg-'+mi;
      h+=`<div class="rv-card" id="${detId}" style="border-left:4px solid ${accentColor};cursor:pointer" onclick="var d=this.querySelector('.mb-mtg-detail');if(d){var o=d.style.display==='none';d.style.display=o?'block':'none';this.querySelector('.mb-mtg-arrow').textContent=o?'\\u25B2':'\\u25BC'}">`;
      h+=`<div style="display:flex;align-items:center;gap:12px">`;
      h+=`<div style="font-size:.82rem;font-weight:700;color:${accentColor};min-width:90px">${escHtml(mtg.time||'')}</div>`;
      const mtgTitle = mtg.title || mtg.name || mtg.meeting || mtg.summary || '';
      h+=`<div style="flex:1"><div class="rv-card-title" style="margin-bottom:0">${escHtml(mtgTitle)}</div></div>`;
      h+=`<span class="rv-pill" style="background:${accentColor};color:#fff">${mtg.priority||''}</span>`;
      h+=`<span style="font-size:.72rem;color:var(--muted)">${attendees.length?attendees.length+' attendees':'no attendees listed'}</span>`;
      h+=`<span class="mb-mtg-arrow" style="color:var(--muted);font-size:.8rem">&#9660;</span>`;
      h+=`</div>`;
      h+=`<div class="mb-mtg-detail" style="display:none;margin-top:14px;padding-top:14px;border-top:1px solid var(--border)" onclick="event.stopPropagation()">`;
      if(mtg.context) h+=`<div style="margin-bottom:12px"><div style="font-size:.72rem;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);margin-bottom:4px">Context</div><div style="font-size:.86rem;line-height:1.55">${escHtml(mtg.context)}</div></div>`;
      if(mtg.prior_notes) h+=`<div style="margin-bottom:12px"><div style="font-size:.72rem;text-transform:uppercase;letter-spacing:.06em;color:var(--accent);margin-bottom:4px">Prior Notes</div><div style="font-size:.86rem;line-height:1.55">${escHtml(mtg.prior_notes)}</div></div>`;
      if(talkPts.length) {
        h+=`<div style="margin-bottom:12px"><div style="font-size:.72rem;text-transform:uppercase;letter-spacing:.06em;color:var(--green);margin-bottom:4px">Talking Points</div><ul style="margin:0 0 0 16px;font-size:.84rem">`;
        talkPts.forEach(tp=>{h+=`<li style="margin-bottom:4px">${escHtml(tp)}</li>`;});
        h+=`</ul></div>`;
      }
      if(actions.length) {
        h+=`<div style="margin-bottom:12px"><div style="font-size:.72rem;text-transform:uppercase;letter-spacing:.06em;color:var(--red);margin-bottom:4px">Action Items to Drive</div><ul style="margin:0 0 0 16px;font-size:.84rem">`;
        actions.forEach(a=>{h+=`<li style="margin-bottom:4px">${escHtml(a)}</li>`;});
        h+=`</ul></div>`;
      }
      if(attendees.length) {
        h+=`<div><div style="font-size:.72rem;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);margin-bottom:6px">Attendees</div><div class="rv-attendees">`;
        attendees.forEach((att,ai)=>{
          const initials=(att.name||'?').split(' ').map(w=>w[0]).join('').toUpperCase().slice(0,2);
          const color=_avatarColors[ai%_avatarColors.length];
          const ooo=att.status&&att.status.toLowerCase().includes('ooo')?'rv-attendee-ooo':'';
          h+=`<div class="rv-attendee ${ooo}"><div class="rv-attendee-initial" style="background:${color}">${initials}</div><div><div style="font-weight:600">${escHtml(att.name||'')}</div>`;
          if(att.role) h+=`<div style="font-size:.7rem;color:var(--muted)">${escHtml(att.role)}</div>`;
          if(att.last_interaction) h+=`<div style="font-size:.68rem;color:var(--muted)">${escHtml(att.last_interaction)}</div>`;
          if(att.status) h+=`<div style="font-size:.68rem;color:var(--red)">${escHtml(att.status)}</div>`;
          h+=`</div></div>`;
        });
        h+=`</div></div>`;
      }
      h+=`</div></div>`;
    });
    if(allDayItems.length) {
      h+=`<div class="rv-section-title" style="margin-top:20px">All-Day Items</div>`;
      allDayItems.forEach((ad,ai)=>{
        const detId='mb-ad-'+ai;
        h+=`<div class="rv-card" id="${detId}" style="border-left:4px solid var(--blue);cursor:pointer" onclick="var d=this.querySelector('.mb-mtg-detail');if(d){var o=d.style.display==='none';d.style.display=o?'block':'none';this.querySelector('.mb-mtg-arrow').textContent=o?'\\u25B2':'\\u25BC'}">`;
        h+=`<div style="display:flex;align-items:center;gap:12px">`;
        h+=`<div style="font-size:.82rem;font-weight:700;color:var(--blue);min-width:90px">All Day</div>`;
        const adTitle = ad.title || ad.name || ad.meeting || ad.summary || '';
        h+=`<div style="flex:1"><div class="rv-card-title" style="margin-bottom:0">${escHtml(adTitle)}</div></div>`;
        h+=`<span class="rv-pill rv-pill-blue">all day</span>`;
        h+=`<span class="mb-mtg-arrow" style="color:var(--muted);font-size:.8rem">&#9660;</span>`;
        h+=`</div>`;
        h+=`<div class="mb-mtg-detail" style="display:none;margin-top:14px;padding-top:14px;border-top:1px solid var(--border)" onclick="event.stopPropagation()">`;
        if(ad.context) h+=`<div style="margin-bottom:12px"><div style="font-size:.72rem;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);margin-bottom:4px">Context</div><div style="font-size:.86rem;line-height:1.55">${escHtml(ad.context)}</div></div>`;
        const talkPts=ad.talking_points||[];
        if(talkPts.length) {
          h+=`<div style="margin-bottom:12px"><div style="font-size:.72rem;text-transform:uppercase;letter-spacing:.06em;color:var(--green);margin-bottom:4px">Talking Points</div><ul style="margin:0 0 0 16px;font-size:.84rem">`;
          talkPts.forEach(tp=>{h+=`<li style="margin-bottom:4px">${escHtml(tp)}</li>`;});
          h+=`</ul></div>`;
        }
        const actions=ad.action_items||[];
        if(actions.length) {
          h+=`<div><div style="font-size:.72rem;text-transform:uppercase;letter-spacing:.06em;color:var(--red);margin-bottom:4px">Action Items</div><ul style="margin:0 0 0 16px;font-size:.84rem">`;
          actions.forEach(a=>{h+=`<li style="margin-bottom:4px">${escHtml(a)}</li>`;});
          h+=`</ul></div>`;
        }
        h+=`</div></div>`;
      });
    }
    h+=`</div>`;
  }
  const pe=data.priority_emails||[];
  if(pe.length) {
    h+=`<div class="rv-section"><div class="rv-section-title">Priority Emails (${pe.length})</div>`;
    pe.forEach(e=>{
      const uc={'critical':'rv-severity-critical','high':'rv-severity-high','medium':'rv-severity-medium','low':'rv-severity-low'};
      h+=`<div class="rv-card ${uc[e.urgency]||''}">
        <div style="display:flex;justify-content:space-between;align-items:start">
          <div class="rv-card-title">${escHtml(e.subject||'')}</div>
          <span class="rv-pill ${e.urgency==='critical'?'rv-pill-red':e.urgency==='high'?'rv-pill-yellow':'rv-pill-blue'}">${escHtml(e.urgency||'')}</span>
        </div>
        <div class="rv-card-meta">From: ${escHtml(e.from||'')} &middot; ${escHtml(e.time_ago||'')}</div>
        ${e.action_needed?`<div class="rv-card-body" style="font-weight:600;color:var(--text)">Action: ${escHtml(e.action_needed)}</div>`:''}
      </div>`;
    });
    h+=`</div>`;
  }
  const cy=data.commitments_yours||[];
  const co=data.commitments_others||[];
  if(cy.length||co.length) {
    h+=`<div class="rv-cols"><div class="rv-section"><div class="rv-section-title">Your Commitments</div>`;
    cy.forEach(c=>{
      h+=`<div class="rv-card"><div class="rv-card-title">${escHtml(c.commitment||'')}</div>
        <div class="rv-card-meta">${escHtml(c.source||'')}</div></div>`;
    });
    h+=`</div><div class="rv-section"><div class="rv-section-title">Others Owe You</div>`;
    co.forEach(c=>{
      h+=`<div class="rv-card"><div class="rv-card-title">${escHtml(c.person||'')}</div>
        <div class="rv-card-body">${escHtml(c.commitment||'')}</div>
        <div class="rv-card-meta">${escHtml(c.source||'')}</div></div>`;
    });
    h+=`</div></div>`;
  }
  const nr=data.needs_reply||[];
  if(nr.length) {
    h+=`<div class="rv-section"><div class="rv-section-title">Needs Your Reply (${nr.length})</div>`;
    nr.forEach(n=>{
      h+=`<div class="rv-card rv-severity-warning"><div class="rv-card-title">${escHtml(n.subject||'')}</div>
        <div class="rv-card-meta">From: ${escHtml(n.from||'')} &middot; ${escHtml(n.time_ago||'')}</div></div>`;
    });
    h+=`</div>`;
  }
  const wc=data.weekend_catchup||[];
  if(wc.length) {
    h+=`<div class="rv-section"><div class="rv-section-title">Weekend Catch-up</div>`;
    wc.forEach(w=>{
      h+=`<div class="rv-card"><div class="rv-card-title">${escHtml(w.subject||'')}</div>
        <div class="rv-card-body">${escHtml(w.detail||'')}</div></div>`;
    });
    h+=`</div>`;
  }
  return h;
}

/* ═══ Rich View: Commitment Tracker ═══ */
function renderCommitmentTracker(data) {
  let h='';

  const biggestExposure = (data.analysis||{}).biggest_risk || '';
  const ehData = data.executive_header || {
    bluf: biggestExposure || `${data.total_open||0} open commitments, ${data.total_overdue||0} overdue`,
    hero_image: data.hero_image || '',
    status: (data.health_score||0) >= 70 ? 'green' : (data.health_score||0) >= 45 ? 'yellow' : 'red',
    metrics: [
      {label:'Health Score',value:data.health_score||0,color:(data.health_score||0)>=70?'green':(data.health_score||0)>=45?'yellow':'red'},
      {label:'Open',value:data.total_open||0},
      {label:'Overdue',value:data.total_overdue||0,color:(data.total_overdue||0)>0?'red':'green'},
      {label:'Owed to You',value:(data.others_owe_you||[]).length}
    ]
  };
  h += renderExecutiveHeader({...data, executive_header:ehData}, 'commitment-tracker');

  h += renderStakeholderHeat(data.your_commitments || []);

  h+=renderAnalysisCard(data.analysis);
  const statusMap={'not_started':'Not Started','in_progress':'In Progress','blocked':'Blocked','completed':'Done','overdue':'Overdue','open':'In Progress','unverified':'Not Started'};
  const kanbanCols={'Not Started':[],'In Progress':[],'Blocked':[],'Done':[]};
  (data.your_commitments||[]).forEach(c=>{const s=statusMap[c.status]||'Not Started';if(s==='Overdue'){kanbanCols['In Progress'].push({...c,_overdue:true});}else{(kanbanCols[s]||(kanbanCols[s]=[])).push(c);}});
  h+=`<div class="rv-section"><div class="rv-section-title">Commitment Board</div><div class="rv-kanban">`;
  Object.entries(kanbanCols).forEach(([col,items])=>{
    const colColor=col==='Done'?'var(--green)':col==='Blocked'?'var(--red)':col==='In Progress'?'#facc15':'var(--muted)';
    h+=`<div class="rv-kanban-col" style="border-top:3px solid ${colColor}"><div class="rv-kanban-col-title">${col} <span class="rv-kanban-count">${items.length}</span></div>`;
    items.forEach(c=>{
      const priColor={'P0':'var(--red)','P1':'#facc15','P2':'var(--accent)'}[c.priority]||'var(--muted)';
      h+=`<div class="rv-kanban-item${c._overdue?' rv-severity-high':''}"><div class="rv-kanban-item-title">${escHtml(c.commitment||'')}</div>`;
      h+=renderProgressBar(c.percent_complete||0);
      h+=`<div class="rv-kanban-item-meta" style="margin-top:6px">`;
      if(c.priority) h+=`<span style="color:${priColor};font-weight:700">${c.priority}</span> `;
      if(c.effort) h+=`<span class="rv-pill rv-pill-blue" style="font-size:.6rem">${c.effort}</span> `;
      if(c.owed_to) h+=`To: ${escHtml(c.owed_to)} `;
      if(c.deadline&&c.deadline!=='—') h+=`&middot; ${escHtml(c.deadline)}`;
      h+=`</div>`;
      if(c.project) h+=`<div class="rv-kanban-item-meta" style="color:var(--accent)">${escHtml(c.project)}</div>`;
      h+=`</div>`;
    });
    h+=`</div>`;
  });
  h+=`</div></div>`;
  h+=`<div class="rv-cols">`;
  h+=`<div class="rv-chart-container"><div class="rv-section-title">Priority Distribution</div><canvas id="ct-priority-chart"></canvas></div>`;
  const byProj=data.by_project||[];
  if(byProj.length){
    h+=`<div class="rv-section"><div class="rv-section-title">By Project</div>`;
    byProj.forEach(p=>{h+=`<div class="rv-card"><div class="rv-card-title">${escHtml(p.project||'Unassigned')}<span style="color:var(--muted);font-weight:400;margin-left:8px">${p.count||0} items</span></div>${renderProgressBar(p.avg_percent_complete||0)}</div>`;});
    h+=`</div>`;
  }
  h+=`</div>`;
  const others=data.others_owe_you||[];
  if(others.length){h+=`<div class="rv-section"><div class="rv-section-title">Others Owe You (${others.length})</div>`;others.forEach(c=>{h+=`<div class="rv-card"><div class="rv-card-title">${escHtml(c.person||'')}: ${escHtml(c.commitment||'')}</div><div class="rv-card-meta">Deadline: ${escHtml(c.deadline||'--')} &middot; ${escHtml(c.source||'')}</div></div>`;});h+=`</div>`;}
  const done=data.recently_completed||[];
  if(done.length){h+=`<div class="rv-section"><div class="rv-section-title">Recently Completed</div>`;done.forEach(c=>{h+=`<div class="rv-card" style="border-left:3px solid var(--green)"><div class="rv-card-title">${escHtml(c.commitment||'')}</div><div class="rv-card-meta">Completed: ${escHtml(c.completed_date||'')}</div></div>`;});h+=`</div>`;}
  return h;
}

/* ═══ Rich View: Project Brief ═══ */
function renderProjectBrief(data) {
  let h='';
  const ps=data.portfolio_summary||{};

  const ehData = data.executive_header || {
    bluf: (data.analysis||{}).executive_insight || '',
    hero_image: data.hero_image || '',
    status: (ps.blocked||0) > 0 ? 'red' : (ps.at_risk||0) > 0 ? 'yellow' : 'green',
    metrics: [
      {label:'Projects',value:ps.total_projects||0},
      {label:'On Track',value:ps.on_track||0,color:'green'},
      {label:'At Risk',value:ps.at_risk||0,color:(ps.at_risk||0)>0?'yellow':'green'},
      {label:'Blocked',value:ps.blocked||0,color:(ps.blocked||0)>0?'red':'green'},
      {label:'Avg Health',value:ps.avg_health||0}
    ]
  };
  h += renderExecutiveHeader({...data, executive_header:ehData}, 'project-brief');

  const decisions = [];
  (data.projects||[]).forEach(proj => {
    (proj.open_decisions||[]).forEach(d => {
      if(typeof d === 'string') {
        decisions.push({project: proj.name, decision: d, waiting_on: '', recommended_action: '', deadline: ''});
      } else {
        decisions.push({project: proj.name, decision: d.decision || d.title || '', waiting_on: d.waiting_on || '', recommended_action: d.recommended_action || '', deadline: d.deadline || ''});
      }
    });
  });
  if(decisions.length) {
    h += '<div class="rv-section"><div class="rv-section-title" style="color:var(--red)">Decisions Needed (' + decisions.length + ')</div>';
    decisions.forEach(d => {
      h += `<div class="exec-decision-card">
        <div class="exec-decision-title"><span style="font-size:1.1rem">&#9888;</span> ${escHtml(d.decision)}</div>
        ${d.waiting_on ? '<div class="exec-decision-detail"><b>Waiting on:</b> ' + escHtml(d.waiting_on) + '</div>' : ''}
        ${d.recommended_action ? '<div class="exec-decision-detail"><b>Action:</b> ' + escHtml(d.recommended_action) + '</div>' : ''}
        ${d.deadline ? '<div class="exec-decision-detail" style="color:var(--red)"><b>Deadline:</b> ' + escHtml(d.deadline) + '</div>' : ''}
        <div style="font-size:.72rem;color:var(--muted);margin-top:6px">Project: ${escHtml(d.project)}</div>
      </div>`;
    });
    h += '</div>';
  }

  h+=renderAnalysisCard(data.analysis);
  const projects=data.projects||[];
  h+=`<div class="rv-filter-bar">`;
  projects.forEach((p,i)=>{
    const sc={'On Track':'rv-pill-green','At Risk':'rv-pill-yellow','Blocked':'rv-pill-red','Completed':'rv-pill-green'};
    h+=`<span class="rv-filter-chip${i===0?' active':''}" onclick="document.querySelectorAll('.pb-proj').forEach((e,j)=>{e.style.display=j===${i}?'block':'none'});document.querySelectorAll('.rv-filter-bar .rv-filter-chip').forEach((e,j)=>e.classList.toggle('active',j===${i}))">${escHtml(p.name||'')} <span class="rv-pill ${sc[p.status]||''}" style="margin-left:4px">${p.status||''}</span></span>`;
  });
  h+=`</div>`;
  projects.forEach((proj,pi)=>{
    const sc={'On Track':'rv-pill-green','At Risk':'rv-pill-yellow','Blocked':'rv-pill-red','Completed':'rv-pill-green'};
    h+=`<div class="pb-proj" style="display:${pi===0?'block':'none'}">`;
    h+=`<div class="rv-gauge-wrap">`;
    h+=renderScoreBadge(proj.health_score,'Health','pb-health-'+pi);
    h+=`<div style="flex:1"><div class="rv-card" style="border-left:4px solid var(--accent);margin-bottom:0"><div class="rv-card-body">${escHtml(proj.summary||'')}</div>`;
    if(proj.executive_insight) h+=`<div style="margin-top:8px;font-style:italic;color:var(--accent);font-size:.84rem">${escHtml(proj.executive_insight)}</div>`;
    h+=`</div></div></div>`;
    if(proj.progress_percent!=null) h+=`<div style="margin-bottom:16px"><div style="display:flex;justify-content:space-between;font-size:.78rem;color:var(--muted)"><span>Progress</span><span>${proj.progress_percent}%</span></div>${renderProgressBar(proj.progress_percent)}</div>`;
    const ms=proj.milestones||[];
    if(ms.length) {
      h+=`<div class="rv-section"><div class="rv-section-title">Milestones</div><div class="rv-milestones">`;
      ms.forEach((m,mi)=>{
        if(mi>0) h+=`<div class="rv-milestone-line"></div>`;
        h+=`<div class="rv-milestone"><div class="rv-milestone-dot ${m.status||''}"></div><div class="rv-milestone-name">${escHtml(m.name||'')}</div><div class="rv-milestone-date">${escHtml(m.target_date||'')}</div></div>`;
      });
      h+=`</div></div>`;
    }
    const tm=proj.team_members||[];
    if(tm.length) {
      h+=`<div class="rv-section"><div class="rv-section-title">Team (${tm.length})</div><div class="rv-team-grid">`;
      tm.forEach((m,mi)=>{
        const color=_avatarColors[mi%_avatarColors.length];
        const initials=(m.name||'?').split(' ').map(w=>w[0]).join('').toUpperCase().slice(0,2);
        const engColor={'active':'var(--green)','passive':'#facc15','dormant':'var(--red)'}[m.engagement_level]||'var(--muted)';
        h+=`<div class="rv-team-member"><div class="rv-team-avatar" style="background:${color}">${initials}</div><div class="rv-team-name">${escHtml(m.name||'')}</div><div class="rv-team-role">${escHtml(m.role||'')}</div><div style="margin-top:4px"><span class="rv-team-badge" style="background:${engColor}"></span><span style="font-size:.68rem;color:var(--muted)">${escHtml(m.engagement_level||'')}</span></div></div>`;
      });
      h+=`</div></div>`;
    }
    const rm=proj.risk_matrix||[];
    if(rm.length) {
      h+=`<div class="rv-section"><div class="rv-section-title">Risk Matrix</div>`;
      rm.forEach(r=>{
        const sev=(r.probability==='high'&&r.impact==='high')?'high':(r.probability==='high'||r.impact==='high')?'medium':'low';
        h+=`<div class="rv-card rv-severity-${sev}"><div class="rv-card-title">${escHtml(r.risk||'')}</div><div class="rv-card-body"><span class="rv-pill rv-pill-${r.probability==='high'?'red':r.probability==='medium'?'yellow':'green'}">P:${r.probability}</span> <span class="rv-pill rv-pill-${r.impact==='high'?'red':r.impact==='medium'?'yellow':'green'}">I:${r.impact}</span></div><div class="rv-card-meta">${escHtml(r.mitigation||'')}</div></div>`;
      });
      h+=`</div>`;
    }
    const bl=proj.blockers||[];
    if(bl.length){h+=`<div class="rv-section"><div class="rv-section-title">Blockers</div>`;bl.forEach(b=>{h+=`<div class="rv-card rv-severity-${b.severity||'medium'}"><div class="rv-card-title">${escHtml(b.title||'')}</div><div class="rv-card-body">${escHtml(b.impact||'')}</div></div>`;});h+=`</div>`;}
    const ns=proj.next_steps||[];
    if(ns.length){h+=`<div class="rv-card"><div class="rv-card-title">Next Steps</div><ul style="margin:8px 0 0 16px;font-size:.86rem">`;ns.forEach(s=>{h+=`<li>${escHtml(s)}</li>`;});h+=`</ul></div>`;}
    const cm=proj.recent_communications||[];
    if(cm.length){h+=`<div class="rv-card"><div class="rv-card-title">Recent Communications</div>`;cm.forEach(c=>{h+=`<div style="font-size:.84rem;margin-top:8px;padding-top:8px;border-top:1px solid var(--border)"><b>${escHtml(c.subject||'')}</b><br><span style="color:var(--muted)">From: ${escHtml(c.from||'')} &middot; ${escHtml(c.date||'')}</span><br>${escHtml(c.summary||'')}</div>`;});h+=`</div>`;}
    h+=`</div>`;
  });
  return h;
}

/* ═══ Rich View: Focus Audit ═══ */
function renderFocusAudit(data) {
  const m=data.metrics||{};
  let h='';
  h+=`<div class="rv-gauge-wrap">`;
  h+=renderScoreBadge(data.productivity_score,'Productivity','fa-prod-gauge');
  h+=`<div style="flex:1"><div class="rv-metrics" style="margin-bottom:0">
    <div class="rv-metric"><div class="rv-metric-value">${m.total_active_hours!=null&&m.total_active_hours>=0?(m.total_active_hours).toFixed(1)+'h':'--'}</div><div class="rv-metric-label">Active Hours</div></div>
    <div class="rv-metric"><div class="rv-metric-value">${(m.deep_work_hours||0).toFixed(1)}h</div><div class="rv-metric-label">Deep Work</div></div>
    <div class="rv-metric"><div class="rv-metric-value">${(m.meeting_hours||0).toFixed(1)}h</div><div class="rv-metric-label">Meetings</div></div>
    <div class="rv-metric"><div class="rv-metric-value">${m.context_switches>=0?m.context_switches:'--'}</div><div class="rv-metric-label">Context Switches</div></div>
    <div class="rv-metric"><div class="rv-metric-value">${m.longest_focus_block_minutes||0}m</div><div class="rv-metric-label">Best Focus Block</div></div>
    <div class="rv-metric"><div class="rv-metric-value">${(m.meeting_load_percent||0).toFixed(0)}%</div><div class="rv-metric-label">Meeting Load</div></div>
  </div></div></div>`;
  if(data.focus_villain) h+=`<div class="rv-card rv-severity-high" style="margin-bottom:20px"><div class="rv-card-title" style="font-size:1rem">Focus Villain</div><div class="rv-card-body">${escHtml(data.focus_villain)}</div></div>`;
  h+=renderAnalysisCard(data.analysis);
  const comp=data.comparison||{};
  if(comp.vs_7d_avg&&comp.vs_7d_avg!=='same') {
    const deltaColor=comp.vs_7d_avg==='better'?'var(--green)':'var(--red)';
    const arrow=comp.vs_7d_avg==='better'?'&#9650;':'&#9660;';
    h+=`<div class="rv-metrics">`;
    h+=`<div class="rv-metric"><div class="rv-metric-value" style="color:${deltaColor}">${arrow} ${comp.vs_7d_avg}</div><div class="rv-metric-label">vs 7-Day Avg</div></div>`;
    if(comp.deep_work_delta&&comp.deep_work_delta!==0) h+=`<div class="rv-metric"><div class="rv-metric-value" style="color:${comp.deep_work_delta>=0?'var(--green)':'var(--red)'}">${comp.deep_work_delta>=0?'+':''}${comp.deep_work_delta.toFixed(1)}h</div><div class="rv-metric-label">Deep Work</div></div>`;
    if(comp.meeting_delta&&comp.meeting_delta!==0) h+=`<div class="rv-metric"><div class="rv-metric-value" style="color:${comp.meeting_delta<=0?'var(--green)':'var(--red)'}">${comp.meeting_delta>=0?'+':''}${comp.meeting_delta.toFixed(1)}h</div><div class="rv-metric-label">Meetings</div></div>`;
    h+=`</div>`;
  }
  /* Day Timeline - visual calendar showing how the day was spent */
  const hb=data.hourly_breakdown||[];
  if(hb.length) {
    const catColor={'deep_work':'rgba(74,222,128,.7)','meeting':'rgba(248,113,113,.7)','communication':'rgba(250,204,21,.7)','research':'rgba(108,124,255,.7)','admin':'rgba(128,128,128,.5)'};
    const catLabel={'deep_work':'Deep Work','meeting':'Meeting','communication':'Comms','research':'Research','admin':'Admin'};
    const hours=new Map();
    hb.forEach(b=>{const hr=b.hour||'';if(!hours.has(hr))hours.set(hr,[]);hours.get(hr).push(b);});
    h+=`<div class="rv-section"><div class="rv-section-title">Day Timeline</div>`;
    h+=`<div style="display:flex;gap:6px;margin-bottom:12px;flex-wrap:wrap">`;
    Object.entries(catColor).forEach(([k,c])=>{h+=`<span style="font-size:.72rem;color:var(--muted);display:flex;align-items:center;gap:4px"><span style="width:12px;height:12px;border-radius:3px;background:${c};display:inline-block"></span>${catLabel[k]||k}</span>`;});
    h+=`</div>`;
    [...hours.entries()].sort((a,b)=>a[0].localeCompare(b[0])).forEach(([hr,items])=>{
      h+=`<div style="display:flex;align-items:stretch;margin-bottom:4px;min-height:36px">`;
      h+=`<div style="width:50px;flex-shrink:0;font-size:.76rem;color:var(--muted);display:flex;align-items:center;justify-content:flex-end;padding-right:10px">${escHtml(hr)}</div>`;
      h+=`<div style="flex:1;display:flex;gap:3px">`;
      items.forEach(b=>{
        const cat=(b.category||'admin').toLowerCase().replace(/[^a-z_]/g,'');
        const bg=catColor[cat]||'rgba(128,128,128,.4)';
        const appShort=(b.app||'').split('—')[0].split('–')[0].trim().replace(/^(Microsoft |Calendar\/)/,'');
        h+=`<div style="flex:1;background:${bg};border-radius:6px;padding:4px 8px;font-size:.72rem;font-weight:600;color:#fff;display:flex;align-items:center;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${escHtml(b.app||'')}">${escHtml(appShort)}</div>`;
      });
      h+=`</div></div>`;
    });
    h+=`</div>`;
  }
  /* Heatmap */
  const hm=data.hourly_heatmap||[];
  if(hm.length) {
    h+=`<div class="rv-section"><div class="rv-section-title">Hourly Productivity Score</div><div class="rv-heatmap" style="grid-template-columns:repeat(${Math.min(hm.length,13)},1fr)">`;
    hm.forEach(cell=>{
      const s=cell.score||0;
      const bg=s>=75?'rgba(74,222,128,.6)':s>=50?'rgba(250,204,21,.5)':s>=25?'rgba(248,113,113,.4)':'rgba(128,128,128,.2)';
      h+=`<div class="rv-heatmap-cell" style="background:${bg}" title="${cell.hour}: ${cell.dominant_activity||''} (${s}/100)"><div style="font-size:.7rem">${escHtml(cell.hour||'').replace(':00','')}</div><div style="font-size:.62rem;opacity:.8">${s}</div></div>`;
    });
    h+=`</div></div>`;
  }
  /* App breakdown - filter out -1 values */
  const abRaw=data.app_breakdown||[];
  const ab=abRaw.filter(a=>(a.minutes||0)>0);
  const taRaw=data.top_apps||[];
  const ta=taRaw.filter(a=>(a.hours||0)>0);
  const appData=ab.length?ab:ta.map(a=>({name:a.name,minutes:Math.round((a.hours||0)*60),percent:0,category:a.category}));
  if(appData.length) {
    const total=appData.reduce((s,a)=>s+(a.minutes||0),0)||1;
    h+=`<div class="rv-cols">`;
    h+=`<div class="rv-chart-container"><div class="rv-section-title">App Usage</div><canvas id="fa-app-chart"></canvas></div>`;
    h+=`<div class="rv-section"><div class="rv-section-title">Time by App</div>`;
    appData.sort((a,b)=>(b.minutes||0)-(a.minutes||0));
    appData.forEach((a,i)=>{
      const pct=Math.round(((a.minutes||0)/total)*100);
      const color=_avatarColors[i%_avatarColors.length];
      h+=`<div style="margin-bottom:10px"><div style="display:flex;justify-content:space-between;font-size:.82rem;margin-bottom:3px"><span style="font-weight:600">${escHtml(a.name||'')}</span><span style="color:var(--muted)">${a.minutes||0}m (${pct}%)</span></div>`;
      h+=`<div class="rv-progress"><div class="rv-progress-bar" style="width:${pct}%;background:${color}"></div></div></div>`;
    });
    h+=`</div></div>`;
  }
  /* Focus windows */
  const fw=data.focus_windows||[];
  if(fw.length){h+=`<div class="rv-section"><div class="rv-section-title">Best Focus Windows</div>`;fw.forEach(f=>{h+=`<div class="rv-card" style="border-left:3px solid var(--green)"><div class="rv-card-title">${escHtml(f.start||'')} — ${f.duration_minutes||0} min</div><div class="rv-card-body">${escHtml(f.context||'')}</div><div class="rv-card-meta">${escHtml(f.app||'')}</div></div>`;});h+=`</div>`;}
  /* Fragmentation */
  const fh=data.fragmentation_hotspots||[];
  if(fh.length){h+=`<div class="rv-section"><div class="rv-section-title">Fragmentation Hotspots</div>`;fh.forEach(f=>{h+=`<div class="rv-card rv-severity-warning"><div class="rv-card-title">${escHtml(f.hour||'')}${f.switches>=0?' — '+f.switches+' switches':''}</div><div class="rv-card-body">${escHtml(f.detail||'')}</div></div>`;});h+=`</div>`;}
  /* Optimization plan with gain bars */
  const opt=data.optimization_plan||[];
  if(opt.length){
    opt.sort((a,b)=>(b.expected_gain_minutes||0)-(a.expected_gain_minutes||0));
    const maxGain=Math.max(...opt.map(o=>o.expected_gain_minutes||0),1);
    h+=`<div class="rv-section"><div class="rv-section-title">Optimization Plan</div>`;
    opt.forEach(o=>{
      const dc={'easy':'rv-pill-green','medium':'rv-pill-yellow','hard':'rv-pill-red'};
      const barPct=Math.round(((o.expected_gain_minutes||0)/maxGain)*100);
      h+=`<div class="rv-card"><div class="rv-card-title" style="display:flex;align-items:center;gap:8px"><span class="rv-pill ${dc[o.difficulty]||'rv-pill-gray'}">${o.difficulty||''}</span> +${o.expected_gain_minutes||0} min</div>`;
      h+=`<div class="rv-progress" style="margin:8px 0"><div class="rv-progress-bar rv-progress-bar-green" style="width:${barPct}%"></div></div>`;
      h+=`<div class="rv-card-body">${escHtml(o.suggestion||'')}</div></div>`;
    });
    h+=`</div>`;
  }
  /* Insights */
  const ins=[...(data.insights||[]),...(data.recommendations||[])];
  if(ins.length){h+=`<div class="rv-section"><div class="rv-section-title">Insights & Recommendations</div><ul style="margin:0 0 0 16px;font-size:.88rem">`;ins.forEach(i=>{h+=`<li style="margin-bottom:8px">${escHtml(i)}</li>`;});h+=`</ul></div>`;}
  return h;
}

/* ═══ Rich View: Relationship CRM ═══ */
function renderRelationshipCrm(data) {
  const ns=data.network_summary||{};
  let h=`<div class="rv-metrics">
    <div class="rv-metric"><div class="rv-metric-value">${data.active_contacts_count||0}</div><div class="rv-metric-label">Active Contacts</div></div>
    <div class="rv-metric"><div class="rv-metric-value">${ns.inner_circle_count||0}</div><div class="rv-metric-label">Inner Circle</div></div>
    <div class="rv-metric"><div class="rv-metric-value" style="color:var(--green)">${ns.growing_relationships||0}</div><div class="rv-metric-label">Growing</div></div>
    <div class="rv-metric"><div class="rv-metric-value" style="color:var(--red)">${ns.fading_relationships||0}</div><div class="rv-metric-label">Fading</div></div>
  </div>`;
  h+=renderAnalysisCard(data.analysis);
  h+=`<div class="rv-cols"><div class="rv-chart-container"><div class="rv-section-title">Interaction Frequency</div><canvas id="crm-freq-chart"></canvas></div>`;
  const edges=data.network_edges||[];
  if(edges.length) {
    const nodes=new Set();edges.forEach(e=>{nodes.add(e.from);nodes.add(e.to);});
    const nodeArr=[...nodes];const cx=460;const cy=150;const r=120;
    h+=`<div class="rv-network"><svg viewBox="0 0 920 300">`;
    nodeArr.forEach((n,i)=>{
      const angle=(2*Math.PI*i)/nodeArr.length-Math.PI/2;
      const x=cx+r*Math.cos(angle);const y=cy+r*Math.sin(angle);
      const col=_avatarColors[i%_avatarColors.length];
      h+=`<circle cx="${x}" cy="${y}" r="20" fill="${col}" opacity=".7"/><text x="${x}" y="${y+4}" text-anchor="middle" fill="#fff" font-size="8" font-weight="700">${escHtml(n.split(' ').map(w=>w[0]).join(''))}</text>`;
      h+=`<text x="${x}" y="${y+34}" text-anchor="middle" fill="#8b8fa3" font-size="7">${escHtml(n.split(' ')[0])}</text>`;
    });
    edges.forEach(e=>{
      const fi=nodeArr.indexOf(e.from);const ti=nodeArr.indexOf(e.to);
      if(fi<0||ti<0) return;
      const a1=(2*Math.PI*fi)/nodeArr.length-Math.PI/2;const a2=(2*Math.PI*ti)/nodeArr.length-Math.PI/2;
      const x1=cx+r*Math.cos(a1);const y1=cy+r*Math.sin(a1);
      const x2=cx+r*Math.cos(a2);const y2=cy+r*Math.sin(a2);
      h+=`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="rgba(108,124,255,.3)" stroke-width="${Math.min(e.weight||1,4)}"/>`;
    });
    h+=`</svg></div>`;
  }
  h+=`</div>`;
  const priority=(data.top_contacts||[]).filter(c=>c.is_priority);
  const regular=(data.top_contacts||[]).filter(c=>!c.is_priority);
  function renderContactCards(contacts,title){
    if(!contacts.length) return '';
    let ch=`<div class="rv-section"><div class="rv-section-title">${title} (${contacts.length})</div>`;
    contacts.forEach((c,ci)=>{
      const stars='★'.repeat(c.strength||0)+'☆'.repeat(5-(c.strength||0));
      const tc={'growing':'var(--green)','stable':'var(--blue)','fading':'var(--red)'}[c.trend]||'var(--muted)';
      const vc={'accelerating':'&#9650;','stable':'&#9644;','cooling':'&#9660;'}[c.velocity]||'';
      const vcol={'accelerating':'var(--green)','cooling':'var(--red)'}[c.velocity]||'var(--muted)';
      const color=_avatarColors[ci%_avatarColors.length];
      const initials=(c.name||'?').split(' ').map(w=>w[0]).join('').toUpperCase().slice(0,2);
      ch+=`<div class="rv-card" style="display:flex;gap:14px;align-items:start">
        <div class="rv-team-avatar" style="background:${color};width:42px;height:42px;font-size:.9rem;flex-shrink:0">${initials}</div>
        <div style="flex:1"><div class="rv-card-title">${escHtml(c.name||'')} <span style="color:var(--muted);font-weight:400;font-size:.78rem">${escHtml(c.organization||'')}</span></div>
        <div class="rv-card-body"><span style="color:var(--accent)">${stars}</span> <span class="rv-pill" style="background:${tc};color:#fff;font-size:.65rem">${c.trend||''}</span> <span style="color:${vcol};font-size:.8rem">${vc}</span>`;
      if(c.health_score!=null) ch+=` <span style="font-size:.72rem;color:var(--muted)">Health: ${c.health_score}/100</span>`;
      ch+=`<br><span style="font-size:.8rem;color:var(--muted)">Topics: ${escHtml((c.primary_topics||[]).join(', '))}</span></div>
        <div class="rv-card-meta">${(c.channels||[]).map(ch2=>{const cc={'Email':'#6c7cff','Meeting':'#4ade80','Teams Chat':'#a855f7','Teams Meeting':'#a855f7','Teams Call':'#a855f7','Audio':'#f59e0b'}[ch2]||'#6b7280';return `<span style="display:inline-block;padding:1px 6px;border-radius:8px;font-size:.65rem;background:${cc};color:#fff;margin-right:3px">${escHtml(ch2)}</span>`}).join('')} ${c.interactions_7d||0} this week &middot; ${c.interactions_30d||0} this month &middot; Last: ${escHtml(c.last_interaction||'')}</div></div></div>`;
    });
    ch+=`</div>`;
    return ch;
  }
  if(priority.length) h+=renderContactCards(priority,'Priority Contacts');
  if(regular.length) h+=renderContactCards(regular,'Other Active Contacts');
  const dorm=data.dormant_relationships||[];
  if(dorm.length){h+=`<div class="rv-section"><div class="rv-section-title">Reconnect</div>`;dorm.forEach(d=>{h+=`<div class="rv-card rv-severity-warning"><div class="rv-card-title">${escHtml(d.name||'')}</div><div class="rv-card-body">Last: ${escHtml(d.last_contact||'')} &middot; Was active on: ${escHtml(d.was_active||'')}</div><div class="rv-card-meta">Suggested: ${escHtml(d.suggested_action||'')}</div></div>`;});h+=`</div>`;}
  const newC=data.new_contacts||[];
  if(newC.length){h+=`<div class="rv-section"><div class="rv-section-title">New Contacts</div>`;newC.forEach(n=>{h+=`<div class="rv-card" style="border-left:3px solid var(--green)"><div class="rv-card-title">${escHtml(n.name||'')}</div><div class="rv-card-body">${escHtml(n.context||'')}</div><div class="rv-card-meta">${escHtml(n.channel||'')}</div></div>`;});h+=`</div>`;}
  return h;
}

/* ═══ Rich View: Team Manager ═══ */
function renderTeamManager(data) {
  const oi=data.open_items_summary||{};
  let h='';
  h+=`<div class="rv-gauge-wrap">`;
  h+=renderScoreBadge(data.team_health_score,'Team Health','tm-health-gauge');
  h+=`<div style="flex:1"><div class="rv-metrics" style="margin-bottom:0">
    <div class="rv-metric"><div class="rv-metric-value">${data.team_size||0}</div><div class="rv-metric-label">Team Size</div></div>
    <div class="rv-metric"><div class="rv-metric-value">${oi.yours||0}</div><div class="rv-metric-label">Your Items</div></div>
    <div class="rv-metric"><div class="rv-metric-value">${oi.theirs||0}</div><div class="rv-metric-label">Their Items</div></div>
    <div class="rv-metric"><div class="rv-metric-value" style="color:var(--red)">${(data.attention_needed||[]).length}</div><div class="rv-metric-label">Needs Attention</div></div>
  </div></div></div>`;
  h+=renderAnalysisCard(data.analysis);
  const att=data.attention_needed||[];
  if(att.length){h+=`<div class="rv-section"><div class="rv-section-title">Needs Attention</div>`;att.forEach(a=>{h+=`<div class="rv-card rv-severity-high"><div class="rv-card-title">${escHtml(a.name||'')}</div><div class="rv-card-body">${escHtml(a.reason||'')}</div><div class="rv-card-meta">Action: ${escHtml(a.suggested_action||'')}</div></div>`;});h+=`</div>`;}
  const members=data.team_members||[];
  if(members.length){
    h+=`<div class="rv-section"><div class="rv-section-title">Team Members</div>`;
    members.forEach((m,mi)=>{
      const sc={'green':'rv-pill-green','yellow':'rv-pill-yellow','red':'rv-pill-red'};
      const focus=(m.current_focus||[]).join(', ');const wins=(m.recent_wins||[]).join(', ');const topics=(m.suggested_1_1_topics||[]).join('; ');
      const color=_avatarColors[mi%_avatarColors.length];
      const initials=(m.name||'?').split(' ').map(w=>w[0]).join('').toUpperCase().slice(0,2);
      const gt={'accelerating':'&#9650;','steady':'&#9644;','stalling':'&#9660;'}[m.growth_trajectory]||'';
      const gtc={'accelerating':'var(--green)','stalling':'var(--red)'}[m.growth_trajectory]||'var(--muted)';
      h+=`<div class="rv-card" style="display:flex;gap:14px;align-items:start">
        <div class="rv-team-avatar" style="background:${color};width:42px;height:42px;font-size:.9rem;flex-shrink:0">${initials}</div>
        <div style="flex:1"><div class="rv-card-title"><span class="rv-pill ${sc[m.status]||'rv-pill-gray'}">${m.status||''}</span> ${escHtml(m.name||'')} <span style="color:var(--muted);font-weight:400;font-size:.78rem">${escHtml(m.role||'')}</span> <span style="color:${gtc}">${gt}</span></div>`;
      if(m.workload_score!=null) h+=`<div style="margin:6px 0"><span style="font-size:.72rem;color:var(--muted)">Workload</span>${renderProgressBar(m.workload_score,m.workload_score>80?'red':m.workload_score>60?'yellow':'green')}</div>`;
      h+=`<div class="rv-card-body">${focus?'<b>Focus:</b> '+escHtml(focus)+'<br>':''}${wins?'<b>Wins:</b> '+escHtml(wins)+'<br>':''}${topics?'<b>1:1 topics:</b> '+escHtml(topics):''}</div>
        <div class="rv-card-meta">Last 1:1: ${escHtml(m.last_one_on_one||'--')} &middot; ${m.interactions_7d||0} interactions${m.one_on_one_streak?` &middot; ${m.one_on_one_streak}wk 1:1 streak`:''}</div></div></div>`;
    });
    h+=`</div>`;
  }
  const wins=data.team_wins||[];
  if(wins.length){h+=`<div class="rv-section"><div class="rv-section-title">Team Wins</div><ul style="margin:0 0 0 16px;font-size:.88rem">`;wins.forEach(w=>{h+=`<li style="margin-bottom:6px">${escHtml(w)}</li>`;});h+=`</ul></div>`;}
  return h;
}

/* ═══ Skill Config Editor ═══ */
let _cfgCache={};
async function toggleSkillConfig(name) {
  const panel=document.getElementById('cfg-panel-'+name);
  if(panel.classList.contains('open')) { panel.classList.remove('open'); return; }
  panel.innerHTML='<div style="color:var(--muted);font-size:.82rem;padding:8px">Loading config...</div>';
  panel.classList.add('open');
  try {
    const d=await(await fetch('/api/agents/skill-config/'+name)).json();
    _cfgCache[name]=d;
    panel.innerHTML=renderSkillConfig(name,d);
  } catch(e) { panel.innerHTML='<div style="color:var(--red);font-size:.82rem;padding:8px">Failed to load config</div>'; }
}

function renderSkillConfig(name,d) {
  const m=d.manifest||{};
  let h=`<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;max-width:600px;margin-bottom:14px">
    <div><label class="cfg-label">Schedule</label><input class="cfg-input" id="cfg-sched-${name}" value="${escHtml(m.schedule||'')}"></div>
    <div><label class="cfg-label">Email Subject</label><input class="cfg-input" id="cfg-subj-${name}" value="${escHtml(m.email_subject||'')}"></div>
  </div>`;

  if(d.topics) {
    const t=d.topics;
    h+=`<div style="margin-bottom:14px"><label class="cfg-label">Personal Context</label>
      <textarea class="cfg-textarea" id="cfg-pctx-${name}">${escHtml(t.personal_context||'')}</textarea></div>`;

    h+=`<label class="cfg-label" style="margin-bottom:8px">Topics</label><div id="cfg-topics-${name}">`;
    const topics=t.topics||[];
    for(let i=0;i<topics.length;i++) {
      h+=renderTopicCard(name,i,topics[i]);
    }
    h+=`</div>`;
    h+=`<button class="btn btn-sm" style="margin-top:8px" onclick="addTopic('${name}')">+ Add Topic</button>`;

    const vs=t.vault_signals||{};
    h+=`<div style="margin-top:16px;display:flex;gap:16px;align-items:center">
      <label style="font-size:.82rem;color:var(--muted)"><input type="checkbox" id="cfg-vs-enabled-${name}" ${vs.enabled?'checked':''} style="margin-right:4px">Vault signal detection</label>
      <div><label class="cfg-label">Lookback hours</label><input type="number" class="cfg-input" style="width:80px" id="cfg-vs-hours-${name}" value="${vs.lookback_hours||72}"></div>
      <div><label class="cfg-label">Max auto topics</label><input type="number" class="cfg-input" style="width:80px" id="cfg-vs-max-${name}" value="${vs.max_auto_topics||3}"></div>
    </div>`;
  }

  h+=`<div style="margin-top:16px;display:flex;gap:8px;align-items:center">
    <button class="btn" onclick="saveSkillConfig('${name}')">Save</button>
    <span id="cfg-msg-${name}" style="font-size:.82rem"></span>
  </div>`;
  return h;
}

function renderTopicCard(name,idx,topic) {
  const kws=(topic.keywords||[]).map((k,ki)=>`<span class="kw-pill">${escHtml(k)}<button onclick="removeKeyword('${name}',${idx},${ki})">x</button></span>`).join('');
  return `<div class="topic-card" id="tc-${name}-${idx}">
    <button class="topic-del" onclick="removeTopic('${name}',${idx})" title="Delete topic">x</button>
    <div style="display:grid;grid-template-columns:1fr auto;gap:10px;margin-bottom:8px">
      <div><label class="cfg-label">Name</label><input class="cfg-input" value="${escHtml(topic.name||'')}" onchange="updateTopicField('${name}',${idx},'name',this.value)"></div>
      <div><label class="cfg-label">Depth</label><select class="cfg-input" style="width:100px" onchange="updateTopicField('${name}',${idx},'depth',this.value)">
        <option value="brief" ${topic.depth==='brief'?'selected':''}>Brief</option>
        <option value="detailed" ${topic.depth==='detailed'?'selected':''}>Detailed</option>
      </select></div>
    </div>
    <div style="margin-bottom:8px"><label class="cfg-label">Keywords</label>
      <div style="display:flex;flex-wrap:wrap;gap:2px;margin-bottom:6px">${kws}</div>
      <div style="display:flex;gap:6px"><input class="cfg-input" style="flex:1" id="kw-add-${name}-${idx}" placeholder="Add keyword..." onkeydown="if(event.key==='Enter')addKeyword('${name}',${idx})">
      <button class="btn btn-sm" onclick="addKeyword('${name}',${idx})">Add</button></div>
    </div>
    <div><label class="cfg-label">Context (why it matters)</label><input class="cfg-input" value="${escHtml(topic.context||'')}" onchange="updateTopicField('${name}',${idx},'context',this.value)"></div>
  </div>`;
}

function _getTopics(name) {
  if(!_cfgCache[name]||!_cfgCache[name].topics) return [];
  return _cfgCache[name].topics.topics||[];
}

function updateTopicField(name,idx,field,val) {
  const topics=_getTopics(name);
  if(topics[idx]) topics[idx][field]=val;
}

function removeKeyword(name,idx,kwIdx) {
  const topics=_getTopics(name);
  if(topics[idx]&&topics[idx].keywords) {
    topics[idx].keywords.splice(kwIdx,1);
    _refreshTopicCards(name);
  }
}

function addKeyword(name,idx) {
  const inp=document.getElementById('kw-add-'+name+'-'+idx);
  const val=(inp?inp.value:'').trim();
  if(!val) return;
  const topics=_getTopics(name);
  if(topics[idx]) {
    if(!topics[idx].keywords) topics[idx].keywords=[];
    topics[idx].keywords.push(val);
    _refreshTopicCards(name);
  }
}

function addTopic(name) {
  const topics=_getTopics(name);
  topics.push({name:'New Topic',keywords:[],depth:'brief',context:''});
  _refreshTopicCards(name);
}

function removeTopic(name,idx) {
  const topics=_getTopics(name);
  topics.splice(idx,1);
  _refreshTopicCards(name);
}

function _refreshTopicCards(name) {
  const container=document.getElementById('cfg-topics-'+name);
  if(!container) return;
  const topics=_getTopics(name);
  let h='';
  for(let i=0;i<topics.length;i++) h+=renderTopicCard(name,i,topics[i]);
  container.innerHTML=h;
}

async function saveSkillConfig(name) {
  const msg=document.getElementById('cfg-msg-'+name);
  msg.textContent='Saving...'; msg.style.color='var(--muted)';
  const body={manifest:{
    schedule:document.getElementById('cfg-sched-'+name)?.value||'',
    email_subject:document.getElementById('cfg-subj-'+name)?.value||'',
  }};

  if(_cfgCache[name]&&_cfgCache[name].topics) {
    const pctx=document.getElementById('cfg-pctx-'+name);
    body.topics={
      personal_context:pctx?pctx.value:'',
      topics:_getTopics(name),
      vault_signals:{
        enabled:document.getElementById('cfg-vs-enabled-'+name)?.checked||false,
        lookback_hours:parseInt(document.getElementById('cfg-vs-hours-'+name)?.value)||72,
        max_auto_topics:parseInt(document.getElementById('cfg-vs-max-'+name)?.value)||3,
      }
    };
  }

  try {
    const r=await fetch('/api/agents/skill-config/'+name,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const d=await r.json();
    if(d.status==='ok') { msg.textContent='Saved!'; msg.style.color='var(--green)'; loadAgentStatus(); }
    else { msg.textContent='Errors: '+(d.errors||[]).join(', '); msg.style.color='var(--red)'; }
  } catch(e) { msg.textContent='Network error'; msg.style.color='var(--red)'; }
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
  const olTitle = ol.new_outlook ? 'Outlook (New — use Mail.app)' : 'Outlook (Classic)';
  h += `<div class="card"><div class="card-header"><span class="card-title">${olTitle}</span>${badge(ol.health)}</div>`;
  if (ol.new_outlook) {
    h += `<div style="padding:8px 0;color:var(--muted);font-size:.82rem">New Outlook detected. Local DB not available.<br>Email extraction uses the <strong>Mail.app</strong> extractor instead.</div>`;
  } else {
    h += `<div class="stat-row"><span class="stat-label">Emails (ID)</span><span class="stat-value">${(ol.last_mail_id||0).toLocaleString()}</span></div>
    <div class="stat-row"><span class="stat-label">Events (ID)</span><span class="stat-value">${(ol.last_event_id||0).toLocaleString()}</span></div>
    <div class="stat-row"><span class="stat-label">Email files</span><span class="stat-value">${(ol.email_files||0).toLocaleString()}</span></div>
    <div class="stat-row"><span class="stat-label">Calendar</span><span class="stat-value">${ol.calendar_files||0}</span></div>
    <div class="stat-row"><span class="stat-label">Errors</span><span class="stat-value">${ol.errors||0}</span></div>
    <div style="margin-top:10px"><button class="btn btn-sm" onclick="runNow('outlook',this)">Run Now</button></div>`;
  }
  h += `</div>`;
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

/* ═══ Audio Device Picker ═══ */
async function loadAudioDevices() {
  const sel = document.getElementById('audio-device-select');
  sel.disabled = true;
  sel.innerHTML = '<option value="">Loading...</option>';
  try {
    const d = await(await fetch('/api/audio/devices')).json();
    if (d.error) {
      sel.innerHTML = '<option value="">Screenpipe unavailable</option>';
      return;
    }
    const devices = d.devices || [];
    if (!devices.length) {
      sel.innerHTML = '<option value="">No devices found</option>';
      return;
    }
    sel.innerHTML = '';
    devices.forEach(dev => {
      const opt = document.createElement('option');
      opt.value = dev.name;
      opt.textContent = dev.name;
      if (d.preferred && dev.name === d.preferred) opt.selected = true;
      sel.appendChild(opt);
    });
    sel.disabled = false;
  } catch(e) {
    sel.innerHTML = '<option value="">Error loading devices</option>';
  }
}
async function switchAudioDevice(name) {
  if (!name) return;
  const sel = document.getElementById('audio-device-select');
  const msg = document.getElementById('audio-device-msg');
  sel.disabled = true;
  msg.textContent = 'Switching...';
  msg.style.color = 'var(--muted)';
  try {
    const d = await(await fetch('/api/audio/device', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({device_name: name})
    })).json();
    if (d.error) {
      msg.textContent = d.error;
      msg.style.color = 'var(--red)';
    } else {
      msg.textContent = 'Switched!';
      msg.style.color = 'var(--green)';
    }
  } catch(e) {
    msg.textContent = 'Failed';
    msg.style.color = 'var(--red)';
  }
  sel.disabled = false;
  setTimeout(() => { msg.textContent = ''; }, 3000);
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
  const fmMatch=body.match(/^---\n([\s\S]*?)\n---\n?/);
  if(fmMatch){ fm=fmMatch[1]; body=body.slice(fmMatch[0].length); }

  const lines=body.split('\n');
  let html='', inUl=false, inOl=false, inTable=false, inPre=false;

  for(let i=0;i<lines.length;i++){
    let ln=lines[i];

    if(ln.startsWith('```')){ if(inPre){html+='</code></pre>';inPre=false;}else{html+='<pre><code>';inPre=true;}continue; }
    if(inPre){html+=escHtml(ln)+'\n';continue;}

    if(ln.trim().startsWith('|')&&ln.trim().endsWith('|')){
      const cells=ln.split('|').slice(1,-1).map(c=>c.trim());
      if(cells.every(c=>/^[-:]+$/.test(c))){continue;}
      if(!inTable){
        inTable=true;html+='<table><thead><tr>'+cells.map(c=>'<th>'+inlineMd(escHtml(c))+'</th>').join('')+'</tr></thead><tbody>';
        if(lines[i+1]&&/^\|[\s-:|]+\|$/.test(lines[i+1].trim())){i++;}
        continue;
      }
      html+='<tr>'+cells.map(c=>'<td>'+inlineMd(escHtml(c))+'</td>').join('')+'</tr>';
      continue;
    }
    if(inTable){html+='</tbody></table>';inTable=false;}

    if(/^#{1,4} /.test(ln)){
      if(inUl){html+='</ul>';inUl=false;}
      if(inOl){html+='</ol>';inOl=false;}
      const lvl=ln.match(/^(#+)/)[1].length;
      const txt=ln.replace(/^#+\s*/,'');
      html+=`<h${lvl}>${inlineMd(escHtml(txt))}</h${lvl}>`;continue;
    }
    if(/^[-*] /.test(ln.trim())){
      if(!inUl){html+='<ul>';inUl=true;}
      const indent=ln.match(/^(\s*)/)[1].length;
      html+=`<li>${inlineMd(escHtml(ln.replace(/^\s*[-*]\s*/,'')))}`;
      if(lines[i+1]&&/^\s+[-*] /.test(lines[i+1])&&lines[i+1].match(/^(\s*)/)[1].length>indent){
        html+='<ul>';
        while(i+1<lines.length&&/^\s+[-*] /.test(lines[i+1])&&lines[i+1].match(/^(\s*)/)[1].length>indent){
          i++;html+='<li>'+inlineMd(escHtml(lines[i].replace(/^\s*[-*]\s*/,'')))+'</li>';
        }
        html+='</ul>';
      }
      html+='</li>';continue;
    }
    if(inUl){html+='</ul>';inUl=false;}

    if(/^\d+\.\s/.test(ln.trim())){
      if(!inOl){html+='<ol>';inOl=true;}
      html+='<li>'+inlineMd(escHtml(ln.replace(/^\s*\d+\.\s*/,'')))+'</li>';continue;
    }
    if(inOl){html+='</ol>';inOl=false;}

    if(/^>/.test(ln)){html+='<blockquote>'+inlineMd(escHtml(ln.replace(/^>\s*/,'')))+'</blockquote>';continue;}
    if(ln.trim()==='---'||ln.trim()==='***'){html+='<hr>';continue;}
    if(ln.trim()===''){html+='';continue;}
    html+='<p>'+inlineMd(escHtml(ln))+'</p>';
  }
  if(inUl)html+='</ul>';if(inOl)html+='</ol>';if(inTable)html+='</tbody></table>';if(inPre)html+='</code></pre>';

  let out='';
  if(fm) out+=`<div class="frontmatter">${escHtml(fm)}</div>`;
  return out+html;
}
function inlineMd(s){
  s=s.replace(/`([^`]+)`/g,'<code>$1</code>');
  s=s.replace(/\*\*(.+?)\*\*/g,'<strong>$1</strong>');
  s=s.replace(/(?<!\*)\*([^*]+)\*(?!\*)/g,'<em>$1</em>');
  s=s.replace(/\[\[([^\]|]+)\|([^\]]+)\]\]/g,'<a>$2</a>');
  s=s.replace(/\[\[([^\]]+)\]\]/g,'<a>$1</a>');
  s=s.replace(/\[([^\]]+)\]\(([^)]+)\)/g,'<a href="$2" target="_blank">$1</a>');
  return s;
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

/* ═══ Skills ═══ */
let skillsData = [];
async function loadSkills() {
  try {
    skillsData = await(await fetch('/api/skills')).json();
    renderSkillsGrid();
  } catch(e) { document.getElementById('skills-grid').textContent='Failed to load skills'; }
}
function renderSkillsGrid() {
  const grid = document.getElementById('skills-grid');
  if(!skillsData.length) { grid.innerHTML='<div style="color:var(--muted)">No skills installed</div>'; return; }
  let h='';
  for(let i=0;i<skillsData.length;i++) {
    const s=skillsData[i];
    const tags=[`<span class="skill-tag">${s.file_count} file${s.file_count!==1?'s':''}</span>`];
    if(s.has_scripts) tags.push('<span class="skill-tag skill-tag-scripts">scripts</span>');
    if(s.sections.length) tags.push(`<span class="skill-tag">${s.sections.length} sections</span>`);
    if(s.has_manifest && s.schedule) tags.push(`<span class="skill-tag" style="background:rgba(74,222,128,.12);color:var(--green)">${escHtml(s.schedule)}</span>`);
    h+=`<div class="skill-card" onclick="openSkillModal(${i})">
      <div class="skill-card-name">${escHtml(s.name)}</div>
      <div class="skill-card-desc">${escHtml(s.description)}</div>
      <div class="skill-card-meta">${tags.join('')}</div>
    </div>`;
  }
  grid.innerHTML=h;
}
function renderSkillBody(md) {
  let body=md;
  body=escHtml(body);
  body=body.replace(/```(\w*)\n([\s\S]*?)```/g,'<pre><code>$2</code></pre>');
  body=body.replace(/`([^`]+)`/g,'<code>$1</code>');
  body=body.replace(/^### (.+)$/gm,'<h3>$1</h3>');
  body=body.replace(/^## (.+)$/gm,'<h2>$1</h2>');
  body=body.replace(/^# (.+)$/gm,'<h1>$1</h1>');
  body=body.replace(/\*\*(.+?)\*\*/g,'<strong>$1</strong>');
  body=body.replace(/\*(.+?)\*/g,'<em>$1</em>');
  body=body.replace(/\|([^\n]+)\|/g, function(match){
    const cells=match.split('|').filter(c=>c.trim());
    if(cells.every(c=>/^[\s-:]+$/.test(c))) return '';
    const isHeader=cells.every(c=>/\*\*/.test(c));
    const tag=isHeader?'th':'td';
    return '<tr>'+cells.map(c=>`<${tag}>${c.trim()}</${tag}>`).join('')+'</tr>';
  });
  body=body.replace(/(<tr>[\s\S]*?<\/tr>\s*)+/g,'<table>$&</table>');
  body=body.replace(/^&gt; (.+)$/gm,'<blockquote>$1</blockquote>');
  body=body.replace(/^- \*\*(.+?):\*\* (.+)$/gm,'<li><strong>$1:</strong> $2</li>');
  body=body.replace(/^- (.+)$/gm,'<li>$1</li>');
  body=body.replace(/(<li>[\s\S]*?<\/li>\s*)+/g,'<ul>$&</ul>');
  body=body.replace(/^(\d+)\. (.+)$/gm,'<li>$2</li>');
  body=body.replace(/\n\n/g,'</p><p>');
  body='<p>'+body+'</p>';
  body=body.replace(/<p>\s*<\/p>/g,'');
  body=body.replace(/<p>\s*(<h[123]>)/g,'$1');
  body=body.replace(/(<\/h[123]>)\s*<\/p>/g,'$1');
  body=body.replace(/<p>\s*(<ul>)/g,'$1');
  body=body.replace(/(<\/ul>)\s*<\/p>/g,'$1');
  body=body.replace(/<p>\s*(<table>)/g,'$1');
  body=body.replace(/(<\/table>)\s*<\/p>/g,'$1');
  body=body.replace(/<p>\s*(<pre>)/g,'$1');
  body=body.replace(/(<\/pre>)\s*<\/p>/g,'$1');
  return body;
}
function openSkillModal(idx) {
  const s=skillsData[idx];
  document.getElementById('skill-modal-title').textContent=s.name;
  document.getElementById('skill-modal-body').innerHTML=renderSkillBody(s.body);
  document.getElementById('skill-modal-backdrop').classList.add('open');
}
function closeSkillModal() {
  const el=document.getElementById('skill-modal-backdrop');
  el.classList.remove('open');
  el.style.display='';
}
document.addEventListener('keydown',e=>{ if(e.key==='Escape') closeSkillModal(); });

/* ═══ Chat ═══ */
let chatWs = null;
let chatConnected = false;

function chatConnect() {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  chatWs = new WebSocket(`${proto}//${location.host}/ws/chat`);
  const statusEl = document.getElementById('chat-status');

  chatWs.onopen = () => {
    chatConnected = true;
    statusEl.textContent = 'Connected';
    statusEl.className = 'chat-status connected';
    document.getElementById('chat-send').disabled = false;
  };

  chatWs.onclose = () => {
    chatConnected = false;
    statusEl.textContent = 'Disconnected — reconnecting...';
    statusEl.className = 'chat-status disconnected';
    document.getElementById('chat-send').disabled = true;
    setTimeout(chatConnect, 3000);
  };

  chatWs.onerror = () => {
    chatConnected = false;
    statusEl.textContent = 'Connection error';
    statusEl.className = 'chat-status disconnected';
  };

  chatWs.onmessage = (e) => {
    const data = JSON.parse(e.data);
    const msgs = document.getElementById('chat-messages');
    const typing = document.getElementById('chat-typing');
    if (typing) typing.remove();

    if (data.type === 'tool_call') {
      const el = document.createElement('div');
      el.className = 'chat-tool call';
      el.innerHTML = `<span>&#128295; <strong>${data.name}</strong>(${_chatArgsSummary(data.args)})</span>`;
      el.onclick = () => el.classList.toggle('expanded');
      msgs.appendChild(el);
    } else if (data.type === 'tool_result') {
      const el = document.createElement('div');
      el.className = 'chat-tool result';
      const preview = (data.output || '').slice(0, 120).replace(/</g,'&lt;');
      el.innerHTML = `<span>&#10004; ${data.name}: ${preview}${data.output && data.output.length>120?'...':''}</span><div class="tool-output">${(data.output||'').replace(/</g,'&lt;')}</div>`;
      el.onclick = () => el.classList.toggle('expanded');
      msgs.appendChild(el);
    } else if (data.type === 'done') {
      const el = document.createElement('div');
      el.className = 'chat-msg assistant';
      el.innerHTML = _chatMd(data.content || '');
      msgs.appendChild(el);
      document.getElementById('chat-send').disabled = false;
      document.getElementById('chat-input').disabled = false;
    } else if (data.type === 'error') {
      const el = document.createElement('div');
      el.className = 'chat-msg assistant';
      el.style.borderColor = 'var(--red)';
      el.textContent = 'Error: ' + (data.content || 'Unknown error');
      msgs.appendChild(el);
      document.getElementById('chat-send').disabled = false;
      document.getElementById('chat-input').disabled = false;
    }
    msgs.scrollTop = msgs.scrollHeight;
  };
}

function chatSend() {
  const input = document.getElementById('chat-input');
  const text = input.value.trim();
  if (!text || !chatConnected) return;

  const msgs = document.getElementById('chat-messages');
  const userEl = document.createElement('div');
  userEl.className = 'chat-msg user';
  userEl.textContent = text;
  msgs.appendChild(userEl);

  const typingEl = document.createElement('div');
  typingEl.className = 'chat-typing';
  typingEl.id = 'chat-typing';
  typingEl.innerHTML = '<span class="dots"></span> Thinking';
  msgs.appendChild(typingEl);

  msgs.scrollTop = msgs.scrollHeight;
  input.value = '';
  input.disabled = true;
  document.getElementById('chat-send').disabled = true;

  chatWs.send(JSON.stringify({type: 'message', content: text}));
}

if (document.getElementById('chat-input')) {
  document.getElementById('chat-input').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); chatSend(); }
  });
}

function _chatArgsSummary(args) {
  if (!args) return '';
  const entries = Object.entries(args);
  if (entries.length === 0) return '';
  return entries.map(([k,v]) => {
    const s = typeof v === 'string' ? v : JSON.stringify(v);
    return s.length > 50 ? s.slice(0,50)+'...' : s;
  }).join(', ');
}

function _chatMd(text) {
  return text
    .replace(/&/g, '&amp;').replace(/</g, '&lt;')
    .replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>')
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    .replace(/^### (.+)$/gm, '<h3>$1</h3>')
    .replace(/^## (.+)$/gm, '<h2>$1</h2>')
    .replace(/^# (.+)$/gm, '<h1>$1</h1>')
    .replace(/^\- (.+)$/gm, '<li>$1</li>')
    .replace(/(<li>.*<\/li>\n?)+/g, '<ul>$&</ul>')
    .replace(/^\d+\. (.+)$/gm, '<li>$1</li>')
    .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank">$1</a>')
    .replace(/\n\n/g, '<br><br>')
    .replace(/\n/g, '<br>');
}

/* ═══ Init ═══ */
if (document.getElementById('cards')) {
  refresh(); loadSync(); loadPipeline(); loadSkills(); loadAgentStatus(); loadAudioDevices();
  setInterval(refresh, R);
  setInterval(loadSync, 60000);
  browseTo('');
}
</script>
</body>
</html>"""


# ── WebSocket chat endpoint ──────────────────────────────────────────────────

_agent_sessions: dict[str, Any] = {}
_session_last_active: dict[str, float] = {}
_SESSION_IDLE_TIMEOUT = 600  # 10 minutes


async def _prune_idle_sessions() -> None:
    """Remove WebSocket sessions idle longer than timeout."""
    now = time.time()
    stale = [
        sid for sid, ts in _session_last_active.items()
        if (now - ts) > _SESSION_IDLE_TIMEOUT
    ]
    for sid in stale:
        _agent_sessions.pop(sid, None)
        _session_last_active.pop(sid, None)
        logger.info("Pruned idle chat session: %s", sid)


@app.on_event("startup")
async def _start_session_cleanup() -> None:
    async def _cleanup_loop() -> None:
        while True:
            await asyncio.sleep(120)
            try:
                await _prune_idle_sessions()
                await _prune_chat_sessions()
            except Exception:
                logger.exception("Session cleanup error")
    asyncio.create_task(_cleanup_loop())


@app.websocket("/ws/chat")
async def ws_chat(websocket: WebSocket) -> None:
    """WebSocket endpoint for the agent chat interface."""
    await websocket.accept()

    from src.agents.agent_loop import AgentSession

    session_id = str(id(websocket))
    if session_id not in _agent_sessions:
        _load_env_local()
        _agent_sessions[session_id] = AgentSession(_cfg())
    _session_last_active[session_id] = time.time()

    session = _agent_sessions[session_id]

    try:
        while True:
            data = await websocket.receive_json()
            _session_last_active[session_id] = time.time()
            msg_type = data.get("type", "")
            content = data.get("content", "").strip()

            if msg_type != "message" or not content:
                await websocket.send_json({"type": "error", "content": "Invalid message format"})
                continue

            try:
                async for event in session.run_turn(content):
                    await websocket.send_json(event)
            except Exception as exc:
                logger.exception("Agent loop error")
                await websocket.send_json({"type": "error", "content": str(exc)})
    except WebSocketDisconnect:
        _agent_sessions.pop(session_id, None)
        _session_last_active.pop(session_id, None)
        logger.info("Chat WebSocket disconnected: %s", session_id)
    except Exception:
        _agent_sessions.pop(session_id, None)
        _session_last_active.pop(session_id, None)
        logger.exception("Chat WebSocket unexpected error")


@app.get("/report/{skill_name}/{date}", response_class=HTMLResponse)
async def report_page(skill_name: str, date: str) -> HTMLResponse:
    """Standalone rich-view page for a skill report (opens in new tab).

    Embeds the JSON data directly in the page HTML so it renders
    instantly without a secondary fetch (which would block on
    WebSocket-saturated event loops).
    """
    title_map = {
        "news-pulse": "News Pulse",
        "weekly-status": "Weekly Status",
        "plan-my-week": "Plan My Week",
        "morning-brief": "Morning Brief",
        "commitment-tracker": "Commitment Tracker",
        "project-brief": "Project Brief",
        "focus-audit": "Focus & Time Audit",
        "relationship-crm": "Relationship Intelligence",
        "team-manager": "Team Manager",
        "approvals-queue": "Approvals Queue",
        "strategic-radar": "Strategic Radar",
        "decision-log": "Decision Log",
        "executive-package": "Executive Package",
    }
    display_title = title_map.get(skill_name, skill_name.replace("-", " ").title())

    cfg = _cfg()
    vault = Path(cfg.get("obsidian_vault", "")).expanduser()
    reports_dir_name = cfg.get("agents", {}).get("reports_dir", "90_reports")
    json_path = vault / reports_dir_name / skill_name / f"{date}.json"
    md_path = vault / reports_dir_name / skill_name / f"{date}.md"

    report_json = "null"
    md_content = ""
    if json_path.is_file():
        raw_json = json_path.read_text(encoding="utf-8", errors="replace")
        if skill_name in ("plan-my-week", "weekly-status"):
            try:
                enriched = json.loads(raw_json)
                if skill_name == "plan-my-week":
                    _enrich_plan_with_actuals(enriched, vault, cfg)
                else:
                    _enrich_weekly_status_with_actuals(enriched, vault, cfg)
                report_json = json.dumps(enriched)
            except Exception:
                report_json = raw_json
        else:
            report_json = raw_json
    elif md_path.is_file():
        md_content = md_path.read_text(encoding="utf-8", errors="replace")
        embedded = extract_embedded_json(md_content)
        if embedded:
            normalized = normalize_report(skill_name, embedded, markdown=md_content)
            if skill_name in ("plan-my-week", "weekly-status"):
                try:
                    if skill_name == "plan-my-week":
                        _enrich_plan_with_actuals(normalized, vault, cfg)
                    else:
                        _enrich_weekly_status_with_actuals(normalized, vault, cfg)
                except Exception:
                    pass
            report_json = json.dumps(normalized)
        elif skill_name == "project-brief":
            parsed = parse_project_brief_markdown(md_content)
            if parsed:
                report_json = json.dumps(parsed)
    css_match = re.search(r"(<style>.*?</style>)", DASHBOARD_HTML, re.DOTALL)
    css_block = css_match.group(1) if css_match else ""
    js_match = re.search(r"(<script>.*?</script>)", DASHBOARD_HTML, re.DOTALL)
    js_block = js_match.group(1) if js_match else ""

    escaped_md = json.dumps(md_content) if md_content else "null"

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{display_title} &mdash; {date}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
{css_block}
<style>
  body {{
    background: var(--bg); color: var(--text); margin: 0; padding: 0;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'SF Pro', system-ui, sans-serif;
  }}
  .report-header {{
    display: flex; align-items: center; gap: 16px;
    padding: 20px 32px; background: #1E1B4B; color: #fff;
  }}
  .report-header h1 {{ font-size: 1.4rem; font-weight: 700; margin: 0; letter-spacing: -0.3px; color: #fff; }}
  .report-header .date {{ font-size: .88rem; color: rgba(255,255,255,0.7); }}
  .report-body {{ padding: 32px; max-width: 1100px; margin: 0 auto; }}
  .report-actions {{
    display: flex; gap: 8px; margin-left: auto;
  }}
  .pdf-btn {{
    background: rgba(255,255,255,0.15); color: #fff; border: 1px solid rgba(255,255,255,0.2);
    border-radius: 8px; padding: 8px 18px; font-size: .82rem; font-weight: 600;
    cursor: pointer; transition: all .15s;
  }}
  .pdf-btn:hover {{ background: rgba(255,255,255,0.25); }}
  .theme-toggle {{
    background: rgba(255,255,255,0.1); color: rgba(255,255,255,0.7); border: 1px solid rgba(255,255,255,0.15);
    border-radius: 8px; padding: 8px 14px; font-size: .82rem; cursor: pointer; transition: all .15s;
  }}
  .theme-toggle:hover {{ background: rgba(255,255,255,0.2); color: #fff; }}
</style>
</head>
<body>
<div class="report-header">
  <h1>{display_title}</h1>
  <span class="date">{date}</span>
  <div class="report-actions">
    <button class="theme-toggle" onclick="toggleReportTheme()">Light Mode</button>
    <button class="pdf-btn" onclick="downloadPDF()">Download PDF</button>
  </div>
</div>
<div class="report-body" id="report-body"></div>
<div class="report-footer-print">
  <span>MemoryOS &middot; {display_title}</span>
  <span>{date}</span>
</div>
{js_block}
<script>
function toggleReportTheme() {{
  const html = document.documentElement;
  const btn = document.querySelector('.theme-toggle');
  if (html.classList.contains('exec-light')) {{
    html.classList.remove('exec-light');
    btn.textContent = 'Light Mode';
  }} else {{
    html.classList.add('exec-light');
    btn.textContent = 'Dark Mode';
  }}
}}
(function() {{
  const name = {json.dumps(skill_name)};
  const date = {json.dumps(date)};
  const body = document.getElementById('report-body');
  const data = {report_json};
  const mdContent = {escaped_md};
  if (data) {{
    const rendererMap = {{'news-pulse':renderNewsPulse,'weekly-status':renderWeeklyStatus,'plan-my-week':renderPlanMyWeek,'morning-brief':renderMorningBrief,'commitment-tracker':renderCommitmentTracker,'project-brief':renderProjectBrief,'focus-audit':renderFocusAudit,'relationship-crm':renderRelationshipCrm,'team-manager':renderTeamManager,'approvals-queue':renderApprovalsQueue}};
    const renderFn = rendererMap[name];
    if (renderFn) {{ body.innerHTML = renderFn(data, name, date); setTimeout(()=>_initRichViewCharts(name, data), 100); }}
    else body.innerHTML = '<p>Rich view not available.</p>';
  }} else if (mdContent) {{
    body.innerHTML = renderMarkdown(mdContent);
  }} else {{
    body.innerHTML = '<p style="text-align:center;padding:60px;color:var(--muted)">Report not found.</p>';
  }}
}})();
</script>
</body>
</html>"""
    return HTMLResponse(page)


@app.get("/", response_class=HTMLResponse)
async def dashboard() -> HTMLResponse:
    needs_setup = not CONFIG_PATH.is_file()
    if CONFIG_PATH.is_file():
        try:
            raw = yaml.safe_load(CONFIG_PATH.read_text())
            needs_setup = raw.get("obsidian_vault", "") == PLACEHOLDER_VAULT
        except Exception:
            needs_setup = True
    html = DASHBOARD_HTML
    if needs_setup:
        html = html.replace(
            "/* ═══ Init ═══ */",
            "/* ═══ Init ═══ */\n"
            "document.getElementById('tab-setup').click();\n",
        )
    return HTMLResponse(html)


# ── Entrypoint ───────────────────────────────────────────────────────────────

def _kill_stale_port(port: int = 8765) -> None:
    """Kill any existing process listening on the given port before startup."""
    killed = False
    try:
        result = subprocess.run(
            ["/usr/sbin/lsof", "-t", "-i", f":{port}", "-sTCP:LISTEN"],
            capture_output=True, text=True, timeout=5,
        )
        for pid_str in result.stdout.strip().split():
            pid = int(pid_str)
            if pid != os.getpid():
                logger.info("Killing stale process %d on port %d", pid, port)
                os.kill(pid, 9)
                killed = True
    except Exception:
        logger.warning("Failed to check/kill stale process on port %d", port, exc_info=True)
    if killed:
        time.sleep(0.5)


if __name__ == "__main__":
    import uvicorn
    _kill_stale_port(8765)
    uvicorn.run(
        "src.dashboard.app:app",
        host="0.0.0.0",
        port=8765,
        reload=False,
        log_level="info",
        ws_ping_interval=30,
        ws_ping_timeout=120,
    )
