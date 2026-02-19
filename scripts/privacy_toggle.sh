#!/bin/bash
# Toggle Screenpipe privacy mode (pause/resume audio recording).
#
# Usage:
#   privacy_toggle.sh          # Toggle current state
#   privacy_toggle.sh on       # Enable privacy mode  (stop audio)
#   privacy_toggle.sh off      # Disable privacy mode (start audio)
#   privacy_toggle.sh status   # Print current state
#
# What it does:
#   1. Creates/removes a flag file so the extractor knows to skip audio.
#   2. Calls Screenpipe's API to actually stop/start the microphone.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
FLAG_FILE="$REPO_DIR/config/.privacy_mode"
SCREENPIPE_API="http://localhost:3030"

_is_on() { [ -f "$FLAG_FILE" ]; }

_enable() {
    touch "$FLAG_FILE"
    # Fire-and-forget: try Screenpipe API but don't wait (it can block)
    curl -sf --max-time 2 -X POST "$SCREENPIPE_API/audio/stop" >/dev/null 2>&1 &
    echo "Privacy mode ON — audio will be filtered from Obsidian notes."
}

_disable() {
    rm -f "$FLAG_FILE"
    # Fire-and-forget: try Screenpipe API but don't wait (it can block)
    curl -sf --max-time 2 -X POST "$SCREENPIPE_API/audio/start" >/dev/null 2>&1 &
    echo "Privacy mode OFF — audio recording will be included in notes."
}

_status() {
    if _is_on; then
        echo "Privacy mode is ON (audio paused)"
    else
        echo "Privacy mode is OFF (audio recording)"
    fi
}

case "${1:-toggle}" in
    on)      _enable  ;;
    off)     _disable ;;
    status)  _status  ;;
    toggle)
        if _is_on; then _disable; else _enable; fi
        ;;
    *)
        echo "Usage: $0 [on|off|status|toggle]"
        exit 1
        ;;
esac
