#!/usr/bin/env python3
"""WiFi-based privacy mode for MemoryOS.

Checks the current WiFi SSID and automatically enables/disables privacy mode
based on whether the network is in the trusted list.

Designed to run every 60 seconds via launchd.

Trusted network matching (from config.yaml -> privacy.trusted_networks):
  - Exact match:  "JBBC"      -> matches only "JBBC"
  - Prefix match: "DLR*"      -> matches "DLR-Guest", "DLR-Corp", etc.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common.config import load_config

logger = logging.getLogger("memoryos.wifi_monitor")

SCREENPIPE_API = "http://localhost:3030"


def get_current_ssid() -> str | None:
    """Return the current WiFi SSID, or None if not connected.

    Uses ``scutil`` to read the AirPort state from the system configuration
    store.  This avoids ``system_profiler`` which triggers the macOS
    "would like to access data from other apps" permission dialog and
    still fails to return SSID data from launchd agents.
    """
    # scutil reads the dynamic store; SSID_STR is populated when WiFi is
    # connected and the process has adequate permissions.
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


def is_trusted_network(ssid: str | None, patterns: list[str]) -> bool:
    """Check if SSID matches any trusted network pattern."""
    if ssid is None:
        return False
    for pattern in patterns:
        if pattern.endswith("*"):
            if ssid.startswith(pattern[:-1]):
                return True
        else:
            if ssid == pattern:
                return True
    return False


def set_privacy_mode(flag_file: Path, enable: bool) -> None:
    """Enable or disable privacy mode by managing the flag file and Screenpipe API."""
    currently_on = flag_file.exists()

    if enable and not currently_on:
        flag_file.parent.mkdir(parents=True, exist_ok=True)
        flag_file.touch()
        _screenpipe_audio_control(stop=True)
        logger.info("Privacy mode ENABLED (untrusted/no network)")

    elif not enable and currently_on:
        flag_file.unlink(missing_ok=True)
        _screenpipe_audio_control(stop=False)
        logger.info("Privacy mode DISABLED (trusted network)")


def _screenpipe_audio_control(*, stop: bool) -> None:
    """Best-effort call to Screenpipe API to stop/start audio.

    The Screenpipe audio control endpoints may block or be unavailable.
    We run this in a thread with a short timeout so it never delays the monitor.
    """
    import threading

    def _call() -> None:
        import urllib.request
        import urllib.error
        endpoint = f"{SCREENPIPE_API}/audio/{'stop' if stop else 'start'}"
        try:
            req = urllib.request.Request(endpoint, method="POST")
            urllib.request.urlopen(req, timeout=3)
        except (urllib.error.URLError, OSError) as exc:
            logger.debug("Screenpipe API call failed (%s): %s", endpoint, exc)

    t = threading.Thread(target=_call, daemon=True)
    t.start()


def run(cfg: dict[str, Any]) -> None:
    """Main monitor logic: check WiFi and adjust privacy mode."""
    privacy_cfg = cfg.get("privacy", {})

    if not privacy_cfg.get("auto_privacy_enabled", True):
        logger.info("Auto-privacy disabled in config — skipping")
        return

    trusted = privacy_cfg.get("trusted_networks", [])
    flag_path = Path(privacy_cfg.get(
        "flag_file",
        str(Path(__file__).resolve().parent.parent / "config" / ".privacy_mode"),
    ))

    ssid = get_current_ssid()

    if ssid is None:
        logger.warning(
            "Could not detect WiFi SSID — preserving current privacy state"
        )
        return

    trusted_now = is_trusted_network(ssid, trusted)
    logger.info("WiFi SSID: %s (trusted=%s)", ssid, trusted_now)
    set_privacy_mode(flag_path, enable=not trusted_now)


def main() -> None:
    cfg = load_config()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    run(cfg)


if __name__ == "__main__":
    main()
