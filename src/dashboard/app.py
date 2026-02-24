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
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

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
            result = subprocess.run(
                [str(VENV_PYTHON), "-m", "src.agents.skill_runner", "--skill", skill_name],
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


@app.get("/api/agents/reports/{skill_name}/{date}/json")
async def api_agents_report_json(skill_name: str, date: str, request: Request) -> JSONResponse:
    """Return structured JSON report for rich dashboard rendering."""
    cfg = _cfg()
    vault = Path(cfg.get("obsidian_vault", "")).expanduser()
    reports_dir_name = cfg.get("agents", {}).get("reports_dir", "90_reports")
    json_path = vault / reports_dir_name / skill_name / f"{date}.json"
    if not json_path.is_file():
        return JSONResponse({"error": "JSON report not found"}, status_code=404)
    after = request.query_params.get("after")
    if after:
        try:
            if json_path.stat().st_mtime <= float(after):
                return JSONResponse({"error": "Report not yet refreshed"}, status_code=404)
        except (ValueError, OSError):
            pass
    try:
        data = json.loads(json_path.read_text(encoding="utf-8", errors="replace"))
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
.rv-hero { width: 100%; border-radius: 14px; margin-bottom: 24px; object-fit: cover; max-height: 320px; object-position: center 30%; }

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
</style>
</head>
<body>

<!-- ══════ TOP TABS ══════ -->
<div class="top-tabs">
  <button class="top-tab" onclick="switchTab('setup',this)" id="tab-setup">Setup</button>
  <button class="top-tab active" onclick="switchTab('overview',this)">Overview</button>
  <button class="top-tab" onclick="switchTab('chat',this)" id="tab-chat" style="color:var(--accent);font-weight:600">Chat</button>
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
  </div>
  <div class="rich-view-body" id="rich-view-body"></div>
</div>

<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>

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
  if(name==='setup') loadSetup();
  if(name==='chat' && !chatConnected) chatConnect();
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
  const titleMap={'news-pulse':'News Pulse','weekly-status':'Weekly Status','plan-my-week':'Plan My Week','morning-brief':'Morning Brief','commitment-tracker':'Commitment Tracker','project-brief':'Project Brief','focus-audit':'Focus Audit','relationship-crm':'Relationship CRM','team-manager':'Team Manager'};
  document.getElementById('rich-view-title').textContent=titleMap[name]||name;
  document.getElementById('rich-view-date').textContent=date;
  const body=document.getElementById('rich-view-body');
  const renderers={'news-pulse':renderNewsPulse,'weekly-status':renderWeeklyStatus,'plan-my-week':renderPlanMyWeek,'morning-brief':renderMorningBrief,'commitment-tracker':renderCommitmentTracker,'project-brief':renderProjectBrief,'focus-audit':renderFocusAudit,'relationship-crm':renderRelationshipCrm,'team-manager':renderTeamManager};
  const fn=renderers[name];
  if(fn) body.innerHTML=fn(data,name,date);
  else body.innerHTML='<p>Rich view not available for this skill.</p>';
  overlay.classList.add('open');
  document.body.style.overflow='hidden';
  if(name==='weekly-status') initWeeklyStatusCharts(data);
  setTimeout(()=>_initRichViewCharts(name,data),100);
}
function _initRichViewCharts(name,data) {
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
    if(days.length) createBarChart('pmw-capacity-chart',days.map(d=>d.day_name||d.date),[{label:'Meeting',data:days.map(d=>d.meeting_hours||d.meetings?.hours||0),backgroundColor:'rgba(248,113,113,.6)',borderRadius:6},{label:'Focus',data:days.map(d=>d.focus_hours||0),backgroundColor:'rgba(74,222,128,.5)',borderRadius:6}]);
  }
  if(name==='morning-brief') {
    createGaugeChart('mb-day-gauge',data.day_score||0,'Day');
    const dc=data.day_composition||{};
    if(dc.meeting_percent||dc.focus_percent) createDoughnutChart('mb-comp-chart',['Meetings','Focus','Admin'],[dc.meeting_percent||0,dc.focus_percent||0,dc.admin_percent||0],['rgba(248,113,113,.7)','rgba(74,222,128,.7)','rgba(128,128,128,.5)']);
  }
  if(name==='focus-audit') {
    createGaugeChart('fa-prod-gauge',data.productivity_score||0,'Score');
    const ab=data.app_breakdown||data.top_apps||[];
    if(ab.length) createDoughnutChart('fa-app-chart',ab.map(a=>a.name),ab.map(a=>a.minutes||a.hours*60||0),ab.map((_,i)=>_avatarColors[i%_avatarColors.length]));
  }
  if(name==='relationship-crm') {
    const top=(data.top_contacts||[]).slice(0,8);
    if(top.length) createBarChart('crm-freq-chart',top.map(c=>c.name.split(' ')[0]),[{label:'7d',data:top.map(c=>c.interactions_7d||0),backgroundColor:'rgba(108,124,255,.7)',borderRadius:6},{label:'30d',data:top.map(c=>c.interactions_30d||0),backgroundColor:'rgba(74,222,128,.5)',borderRadius:6}]);
  }
  if(name==='team-manager') {
    createGaugeChart('tm-health-gauge',data.team_health_score||0,'Health');
  }
}
function closeRichView() {
  document.getElementById('rich-view-overlay').classList.remove('open');
  document.body.style.overflow='';
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

function renderAnalysisCard(analysis) {
  if(!analysis) return '';
  let h='<div class="rv-analysis">';
  if(analysis.executive_insight) h+=`<div class="rv-analysis-insight">${escHtml(analysis.executive_insight)}</div>`;
  h+='<div class="rv-analysis-row">';
  if(analysis.biggest_risk) h+=`<div class="rv-analysis-item"><div class="rv-analysis-label">Biggest Risk</div><div class="rv-analysis-text" style="color:var(--red)">${escHtml(analysis.biggest_risk)}</div></div>`;
  if(analysis.recommended_focus) h+=`<div class="rv-analysis-item"><div class="rv-analysis-label">Recommended Focus</div><div class="rv-analysis-text" style="color:var(--green)">${escHtml(analysis.recommended_focus)}</div></div>`;
  h+='</div>';
  if(analysis.predictions&&analysis.predictions.length) {
    h+='<div style="margin-top:12px"><div class="rv-analysis-label">Predictions</div>';
    analysis.predictions.forEach(p=>{h+=`<div class="rv-analysis-text" style="margin-bottom:4px">&#8226; ${escHtml(p)}</div>`;});
    h+='</div>';
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

  const countText=`${data.story_count||totalArticles} stories across ${data.topic_count||topics.length} topics`;
  return `<div style="margin-bottom:16px;font-size:.84rem;color:var(--muted);text-align:center">${countText}</div><div class="np-feed">${h}</div>`;
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
  if(data.hero_image) h+=`<img class="rv-hero" src="${escHtml(data.hero_image)}" alt="" onerror="this.style.display='none'">`;
  h+=`<div class="rv-metrics">
    <div class="rv-metric"><div class="rv-metric-value">${m.meetings_total||0}</div><div class="rv-metric-label">Meetings</div></div>
    <div class="rv-metric"><div class="rv-metric-value">${m.emails_received||0}</div><div class="rv-metric-label">Emails Received</div></div>
    <div class="rv-metric"><div class="rv-metric-value">${m.emails_sent||0}</div><div class="rv-metric-label">Emails Sent</div></div>
    <div class="rv-metric"><div class="rv-metric-value">${m.accomplishments_count||0}</div><div class="rv-metric-label">Accomplishments</div></div>
    <div class="rv-metric"><div class="rv-metric-value">${cs.percent||0}%</div><div class="rv-metric-label">Commitment Score (${cs.delivered||0}/${cs.total||0})</div></div>
  </div>`;
  if(data.summary) h+=`<div class="rv-card" style="margin-bottom:24px"><div class="rv-card-body">${escHtml(data.summary)}</div></div>`;
  h+=renderAnalysisCard(data.analysis);
  const wow=data.week_over_week||{};
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
  const vd=data.value_delivered||[];
  if(vd.length) {
    h+=`<div class="rv-section"><div class="rv-section-title">Value Delivered</div>`;
    vd.forEach(v=>{
      h+=`<div class="rv-card"><div class="rv-card-title">${escHtml(v.statement||'')}</div>
        <div class="rv-card-body">${escHtml(v.detail||'')}</div>
        ${v.evidence_path?`<div class="rv-card-meta">${escHtml(v.evidence_path)}</div>`:''}</div>`;
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
  if(data.hero_image) h+=`<img class="rv-hero" src="${escHtml(data.hero_image)}" alt="" onerror="this.style.display='none'">`;
  h+=`<div class="rv-gauge-wrap">`;
  h+=renderScoreBadge(data.week_score,'Week Score','pmw-score-gauge');
  h+=`<div style="flex:1">`;
  if(data.executive_summary) h+=`<div class="rv-card" style="margin-bottom:0;border-left:4px solid var(--accent)"><div class="rv-card-body" style="font-size:.95rem">${escHtml(data.executive_summary)}</div></div>`;
  h+=`</div></div>`;
  h+=renderAnalysisCard(data.analysis);
  const tvp=data.trend_vs_prior_week||{};
  if(tvp.delta_meetings_hours!=null||tvp.delta_focus_hours!=null) {
    h+=`<div class="rv-metrics">`;
    if(tvp.delta_meetings_hours!=null) { const c=tvp.delta_meetings_hours<=0?'var(--green)':'var(--red)';h+=`<div class="rv-metric"><div class="rv-metric-value" style="color:${c}">${tvp.delta_meetings_hours>=0?'+':''}${tvp.delta_meetings_hours.toFixed(1)}h</div><div class="rv-metric-label">Meetings vs Last Week</div></div>`; }
    if(tvp.delta_focus_hours!=null) { const c=tvp.delta_focus_hours>=0?'var(--green)':'var(--red)';h+=`<div class="rv-metric"><div class="rv-metric-value" style="color:${c}">${tvp.delta_focus_hours>=0?'+':''}${tvp.delta_focus_hours.toFixed(1)}h</div><div class="rv-metric-label">Focus vs Last Week</div></div>`; }
    if(data.total_meeting_hours!=null) h+=`<div class="rv-metric"><div class="rv-metric-value">${data.total_meeting_hours.toFixed(1)}h</div><div class="rv-metric-label">Total Meetings</div></div>`;
    if(data.total_focus_hours!=null) h+=`<div class="rv-metric"><div class="rv-metric-value" style="color:var(--green)">${data.total_focus_hours.toFixed(1)}h</div><div class="rv-metric-label">Total Focus</div></div>`;
    h+=`</div>`;
  }
  const days=data.days||[];
  if(days.length) {
    h+=`<div class="rv-day-strip">`;
    days.forEach((d,i)=>{
      const mc=d.meetings||{};
      const dens=d.density||'moderate';
      const cap=d.capacity_percent||0;
      h+=`<div class="rv-day-card rv-day-density-${dens}" onclick="showDayDetail(${i})">
        <div class="rv-day-card-name">${escHtml(d.day_name||'')}</div>
        <div class="rv-day-card-date">${escHtml(d.date||'')}</div>
        <div class="rv-day-card-meetings" style="color:${dens==='heavy'?'var(--red)':dens==='moderate'?'#facc15':'var(--green)'}">${mc.count||0}</div>
        <div class="rv-day-card-hours">${mc.hours||0}h mtgs</div>
        ${renderProgressBar(cap, cap>75?'red':cap>50?'yellow':'green')}
        <div style="font-size:.66rem;color:var(--muted);margin-top:2px">${Math.round(cap)}% capacity</div>
      </div>`;
    });
    h+=`</div>`;
    h+=`<div class="rv-chart-container"><div class="rv-section-title">Daily Meeting vs Focus</div><canvas id="pmw-capacity-chart"></canvas></div>`;
  }
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
  const dels=data.deliverables||[];
  if(dels.length) {
    h+=`<div class="rv-section"><div class="rv-section-title">Deliverables Timeline</div><div class="rv-gantt">`;
    const dayNames=days.map(d=>d.day_name||d.date);
    h+=`<div class="rv-gantt-row"><div class="rv-gantt-label"></div><div class="rv-gantt-track" style="display:flex;background:none;gap:2px">`;
    dayNames.forEach(dn=>{h+=`<div style="flex:1;text-align:center;font-size:.7rem;color:var(--muted)">${escHtml(dn.slice(0,3))}</div>`;});
    h+=`</div></div>`;
    dels.forEach(dl=>{
      const dayIdx=days.findIndex(d=>d.day_name===dl.planned_day||d.date===dl.planned_day);
      const left=dayIdx>=0?(dayIdx/Math.max(days.length,1)*100)+'%':'0%';
      const barColor=dl.status==='Overdue'||dl.status==='overdue'?'var(--red)':dl.status==='Due'?'#facc15':'var(--accent)';
      h+=`<div class="rv-gantt-row"><div class="rv-gantt-label" title="${escHtml(dl.name)}">${escHtml(dl.name||'')}</div>
        <div class="rv-gantt-track"><div class="rv-gantt-bar" style="background:${barColor};width:${100/Math.max(days.length,1)}%;margin-left:${left}">${escHtml((dl.status||'').slice(0,3))}</div></div></div>`;
    });
    h+=`</div></div>`;
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
  let h=`<div class="rv-section" style="margin-top:16px"><div class="rv-section-title">${escHtml(d.day_name||'')} ${escHtml(d.date||'')}</div>`;
  if((mc.names||[]).length) {
    h+=`<div class="rv-card"><div class="rv-card-title">Meetings (${mc.count||0}, ${mc.hours||0}h)</div><ul style="margin:8px 0 0 16px;font-size:.86rem">`;
    (mc.names||[]).forEach(n=>{h+=`<li>${escHtml(n)}</li>`;});
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
  if(data.hero_image) h+=`<img class="rv-hero" src="${escHtml(data.hero_image)}" alt="" onerror="this.style.display='none'">`;
  h+=`<div class="rv-gauge-wrap">`;
  h+=renderScoreBadge(data.day_score,'Day Score','mb-day-gauge');
  h+=`<div style="flex:1">`;
  if(data.day_summary) h+=`<div style="font-size:1.1rem;font-weight:600;color:var(--text);margin-bottom:12px">${escHtml(data.day_summary)}</div>`;
  h+=`<div class="rv-metrics" style="margin-bottom:0">
    <div class="rv-metric"><div class="rv-metric-value">${data.meeting_count||0}</div><div class="rv-metric-label">Meetings</div></div>
    <div class="rv-metric"><div class="rv-metric-value">${data.priority_email_count||0}</div><div class="rv-metric-label">Priority Emails</div></div>
    <div class="rv-metric"><div class="rv-metric-value">${data.unread_reply_count||0}</div><div class="rv-metric-label">Needs Reply</div></div>
    <div class="rv-metric"><div class="rv-metric-value">${data.commitment_count||0}</div><div class="rv-metric-label">Commitments</div></div>
  </div></div></div>`;
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
  const meetings=data.meetings||[];
  if(meetings.length) {
    h+=`<div class="rv-section"><div class="rv-section-title">Today's Calendar (${meetings.length} meetings)</div><div class="rv-timeline">`;
    meetings.forEach(mtg=>{
      const dotClass=mtg.priority||'medium';
      h+=`<div class="rv-timeline-item"><div class="rv-timeline-dot ${dotClass}"></div>
        <div class="rv-timeline-time">${escHtml(mtg.time||'')}</div>
        <div class="rv-card"><div class="rv-card-title">${escHtml(mtg.title||'')}</div>`;
      if(mtg.context) h+=`<div class="rv-card-body">${escHtml(mtg.context)}</div>`;
      if(mtg.prior_notes) h+=`<div class="rv-card-meta" style="margin-top:6px;color:var(--accent)">Prior: ${escHtml(mtg.prior_notes)}</div>`;
      const attendees=mtg.attendees||[];
      if(attendees.length) {
        h+=`<div class="rv-attendees">`;
        attendees.forEach((att,ai)=>{
          const initials=(att.name||'?').split(' ').map(w=>w[0]).join('').toUpperCase().slice(0,2);
          const color=_avatarColors[ai%_avatarColors.length];
          const ooo=att.status&&att.status.toLowerCase().includes('ooo')?'rv-attendee-ooo':'';
          h+=`<div class="rv-attendee ${ooo}">
            <div class="rv-attendee-initial" style="background:${color}">${initials}</div>
            <div><div style="font-weight:600">${escHtml(att.name||'')}</div>`;
          if(att.role) h+=`<div style="font-size:.7rem;color:var(--muted)">${escHtml(att.role)}</div>`;
          if(att.last_interaction) h+=`<div style="font-size:.68rem;color:var(--muted)">${escHtml(att.last_interaction)}</div>`;
          if(att.status) h+=`<div style="font-size:.68rem;color:var(--red)">${escHtml(att.status)}</div>`;
          h+=`</div></div>`;
        });
        h+=`</div>`;
      }
      h+=`</div></div>`;
    });
    h+=`</div></div>`;
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
  h+=`<div class="rv-gauge-wrap">`;
  h+=renderScoreBadge(data.health_score,'Health','ct-health-gauge');
  h+=`<div style="flex:1"><div class="rv-metrics" style="margin-bottom:0">
    <div class="rv-metric"><div class="rv-metric-value">${data.total_open||0}</div><div class="rv-metric-label">Open</div></div>
    <div class="rv-metric"><div class="rv-metric-value" style="color:var(--red)">${data.total_overdue||0}</div><div class="rv-metric-label">Overdue</div></div>
    <div class="rv-metric"><div class="rv-metric-value">${(data.others_owe_you||[]).length}</div><div class="rv-metric-label">Owed to You</div></div>
    <div class="rv-metric"><div class="rv-metric-value" style="color:var(--green)">${(data.recently_completed||[]).length}</div><div class="rv-metric-label">Completed</div></div>
  </div></div></div>`;
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
  if(ps.total_projects) {
    h+=`<div class="rv-metrics">
      <div class="rv-metric"><div class="rv-metric-value">${ps.total_projects||0}</div><div class="rv-metric-label">Projects</div></div>
      <div class="rv-metric"><div class="rv-metric-value" style="color:var(--green)">${ps.on_track||0}</div><div class="rv-metric-label">On Track</div></div>
      <div class="rv-metric"><div class="rv-metric-value" style="color:#facc15">${ps.at_risk||0}</div><div class="rv-metric-label">At Risk</div></div>
      <div class="rv-metric"><div class="rv-metric-value" style="color:var(--red)">${ps.blocked||0}</div><div class="rv-metric-label">Blocked</div></div>
      <div class="rv-metric"><div class="rv-metric-value">${ps.avg_health||0}</div><div class="rv-metric-label">Avg Health</div></div>
    </div>`;
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
    <div class="rv-metric"><div class="rv-metric-value">${(m.deep_work_hours||0).toFixed(1)}h</div><div class="rv-metric-label">Deep Work</div></div>
    <div class="rv-metric"><div class="rv-metric-value">${(m.meeting_hours||0).toFixed(1)}h</div><div class="rv-metric-label">Meetings</div></div>
    <div class="rv-metric"><div class="rv-metric-value">${m.context_switches||0}</div><div class="rv-metric-label">Context Switches</div></div>
    <div class="rv-metric"><div class="rv-metric-value">${m.longest_focus_block_minutes||0}m</div><div class="rv-metric-label">Best Focus Block</div></div>
    <div class="rv-metric"><div class="rv-metric-value">${(m.meeting_load_percent||0).toFixed(0)}%</div><div class="rv-metric-label">Meeting Load</div></div>
  </div></div></div>`;
  h+=renderAnalysisCard(data.analysis);
  if(data.focus_villain) h+=`<div class="rv-card rv-severity-high" style="margin-bottom:24px"><div class="rv-card-title" style="font-size:1rem">Focus Villain: ${escHtml(data.focus_villain)}</div><div class="rv-card-body">This app or pattern interrupts your focus the most. Consider batching or blocking it during deep work windows.</div></div>`;
  const comp=data.comparison||{};
  if(comp.vs_7d_avg) {
    const deltaColor=comp.vs_7d_avg==='better'?'var(--green)':comp.vs_7d_avg==='worse'?'var(--red)':'var(--muted)';
    const arrow=comp.vs_7d_avg==='better'?'&#9650;':comp.vs_7d_avg==='worse'?'&#9660;':'&#9644;';
    h+=`<div class="rv-metrics"><div class="rv-metric"><div class="rv-metric-value" style="color:${deltaColor}">${arrow}</div><div class="rv-metric-label">vs 7-day avg: ${comp.vs_7d_avg}</div></div>`;
    if(comp.deep_work_delta!=null) h+=`<div class="rv-metric"><div class="rv-metric-value" style="color:${comp.deep_work_delta>=0?'var(--green)':'var(--red)'}">${comp.deep_work_delta>=0?'+':''}${comp.deep_work_delta.toFixed(1)}h</div><div class="rv-metric-label">Deep Work Delta</div></div>`;
    if(comp.meeting_delta!=null) h+=`<div class="rv-metric"><div class="rv-metric-value" style="color:${comp.meeting_delta<=0?'var(--green)':'var(--red)'}">${comp.meeting_delta>=0?'+':''}${comp.meeting_delta.toFixed(1)}h</div><div class="rv-metric-label">Meeting Delta</div></div>`;
    h+=`</div>`;
  }
  const hm=data.hourly_heatmap||[];
  if(hm.length) {
    h+=`<div class="rv-section"><div class="rv-section-title">Hourly Productivity Heatmap</div><div class="rv-heatmap" style="grid-template-columns:repeat(${Math.min(hm.length,13)},1fr)">`;
    hm.forEach(cell=>{
      const s=cell.score||0;
      const bg=s>=75?'rgba(74,222,128,.6)':s>=50?'rgba(250,204,21,.5)':s>=25?'rgba(248,113,113,.4)':'rgba(128,128,128,.2)';
      h+=`<div class="rv-heatmap-cell" style="background:${bg}" title="${cell.hour}: ${cell.dominant_activity||''} (${s}/100)">${escHtml(cell.hour||'').replace(':00','')}</div>`;
    });
    h+=`</div></div>`;
  }
  h+=`<div class="rv-cols">`;
  h+=`<div class="rv-chart-container"><div class="rv-section-title">App Usage</div><canvas id="fa-app-chart"></canvas></div>`;
  const ab=data.app_breakdown||[];
  if(ab.length) {
    h+=`<div class="rv-section"><div class="rv-section-title">App Breakdown</div>`;
    const catColors={'deep_work':'rgba(74,222,128,.5)','meeting':'rgba(248,113,113,.5)','communication':'rgba(250,204,21,.5)','research':'rgba(108,124,255,.5)','admin':'rgba(128,128,128,.4)'};
    const maxMin=Math.max(...ab.map(a=>a.minutes||0),1);
    h+=`<div class="rv-treemap" style="grid-template-columns:${ab.map(a=>`${Math.max(1,Math.round((a.minutes||0)/maxMin*3))}fr`).join(' ')}">`;
    ab.forEach((a,i)=>{
      const bg=catColors[a.category]||_avatarColors[i%_avatarColors.length];
      h+=`<div class="rv-treemap-cell" style="background:${bg}"><div class="rv-treemap-name">${escHtml(a.name||'')}</div><div class="rv-treemap-val">${Math.round(a.minutes||0)}m (${(a.percent||0).toFixed(0)}%)</div></div>`;
    });
    h+=`</div></div>`;
  }
  h+=`</div>`;
  const fw=data.focus_windows||[];
  if(fw.length){h+=`<div class="rv-section"><div class="rv-section-title">Best Focus Windows</div>`;fw.forEach(f=>{h+=`<div class="rv-card" style="border-left:3px solid var(--green)"><div class="rv-card-title">${escHtml(f.start||'')} — ${f.duration_minutes||0} min</div><div class="rv-card-body">${escHtml(f.context||'')}</div><div class="rv-card-meta">${escHtml(f.app||'')}</div></div>`;});h+=`</div>`;}
  const fh=data.fragmentation_hotspots||[];
  if(fh.length){h+=`<div class="rv-section"><div class="rv-section-title">Fragmentation Hotspots</div>`;fh.forEach(f=>{h+=`<div class="rv-card rv-severity-warning"><div class="rv-card-title">${escHtml(f.hour||'')} — ${f.switches||0} switches</div><div class="rv-card-body">${escHtml(f.detail||'')}</div></div>`;});h+=`</div>`;}
  const opt=data.optimization_plan||[];
  if(opt.length){h+=`<div class="rv-section"><div class="rv-section-title">Optimization Plan</div>`;opt.forEach(o=>{const dc={'easy':'rv-pill-green','medium':'rv-pill-yellow','hard':'rv-pill-red'};h+=`<div class="rv-card"><div class="rv-card-title">${escHtml(o.suggestion||'')}</div><div class="rv-card-meta"><span class="rv-pill ${dc[o.difficulty]||'rv-pill-gray'}">${o.difficulty||''}</span> Expected gain: +${o.expected_gain_minutes||0} min</div></div>`;});h+=`</div>`;}
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
        <div class="rv-card-meta">${c.interactions_7d||0} this week &middot; ${c.interactions_30d||0} this month &middot; Last: ${escHtml(c.last_interaction||'')}</div></div></div>`;
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

document.getElementById('chat-input').addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); chatSend(); }
});

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
refresh(); loadSync(); loadPipeline(); loadSkills(); loadAgentStatus();
setInterval(refresh, R);
setInterval(loadSync, 60000);
browseTo('');
</script>
</body>
</html>"""


# ── WebSocket chat endpoint ──────────────────────────────────────────────────

_agent_sessions: dict[str, Any] = {}


@app.websocket("/ws/chat")
async def ws_chat(websocket: WebSocket) -> None:
    """WebSocket endpoint for the agent chat interface."""
    await websocket.accept()

    from src.agents.agent_loop import AgentSession

    session_id = str(id(websocket))
    if session_id not in _agent_sessions:
        _load_env_local()
        _agent_sessions[session_id] = AgentSession(_cfg())

    session = _agent_sessions[session_id]

    try:
        while True:
            data = await websocket.receive_json()
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
        logger.info("Chat WebSocket disconnected: %s", session_id)
    except Exception:
        _agent_sessions.pop(session_id, None)
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
    }
    display_title = title_map.get(skill_name, skill_name)

    cfg = _cfg()
    vault = Path(cfg.get("obsidian_vault", "")).expanduser()
    reports_dir_name = cfg.get("agents", {}).get("reports_dir", "90_reports")
    json_path = vault / reports_dir_name / skill_name / f"{date}.json"
    md_path = vault / reports_dir_name / skill_name / f"{date}.md"

    report_json = "null"
    md_content = ""
    if json_path.is_file():
        report_json = json_path.read_text(encoding="utf-8", errors="replace")
    elif md_path.is_file():
        md_content = md_path.read_text(encoding="utf-8", errors="replace")

    import re
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
{css_block}
<style>
  body {{ background: var(--bg); color: var(--text); margin: 0; padding: 0; }}
  .report-header {{
    display: flex; align-items: center; gap: 16px;
    padding: 18px 28px; border-bottom: 1px solid var(--border);
    background: var(--surface);
  }}
  .report-header h1 {{ font-size: 1.3rem; font-weight: 700; margin: 0; }}
  .report-header .date {{ font-size: .88rem; color: var(--muted); }}
  .report-body {{ padding: 28px; max-width: 1400px; margin: 0 auto; }}
</style>
</head>
<body>
<div class="report-header">
  <h1>{display_title}</h1>
  <span class="date">{date}</span>
</div>
<div class="report-body" id="report-body"></div>
{js_block}
<script>
(function() {{
  const name = {json.dumps(skill_name)};
  const date = {json.dumps(date)};
  const body = document.getElementById('report-body');
  const data = {report_json};
  const mdContent = {escaped_md};
  if (data) {{
    const rendererMap = {{'news-pulse':renderNewsPulse,'weekly-status':renderWeeklyStatus,'plan-my-week':renderPlanMyWeek,'morning-brief':renderMorningBrief,'commitment-tracker':renderCommitmentTracker,'project-brief':renderProjectBrief,'focus-audit':renderFocusAudit,'relationship-crm':renderRelationshipCrm,'team-manager':renderTeamManager}};
    const renderFn = rendererMap[name];
    if (renderFn) {{ body.innerHTML = renderFn(data, name, date); if (name === 'weekly-status') setTimeout(()=>initWeeklyStatusCharts(data), 100); }}
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
