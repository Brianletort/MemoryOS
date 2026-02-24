#!/bin/bash
# MemoryOS Sentinel: nuclear fallback watchdog
# Pure bash -- no Python, no venv, no dependencies
# Runs every 60s via launchd. Ensures Screenpipe and all agents are alive.
set -uo pipefail

REPO_DIR="$(cd "$(dirname "$(dirname "$0")")" && pwd)"
AGENTS_DIR="$HOME/Library/LaunchAgents"
LOG="$REPO_DIR/logs/sentinel.log"
DB_PATH="$HOME/.screenpipe/db.sqlite"

mkdir -p "$REPO_DIR/logs"

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') [SENTINEL] $*" >> "$LOG"; }

# --- 1. Is Screenpipe app running? -------------------------------------------
if ! pgrep -f "screenpipe" > /dev/null 2>&1; then
    log "CRITICAL: Screenpipe not running. Launching..."
    open -a screenpipe 2>/dev/null || true
    sleep 5
fi

# --- 2. Are all launchd agents loaded? ---------------------------------------
AGENTS=(
    com.memoryos.watchdog
    com.memoryos.screenpipe
    com.memoryos.indexer
    com.memoryos.calendar-app
    com.memoryos.mail-app
    com.memoryos.outlook
    com.memoryos.onedrive
    com.memoryos.dashboard
)
LOADED=$(launchctl list 2>/dev/null || echo "")
for agent in "${AGENTS[@]}"; do
    if ! echo "$LOADED" | grep -q "$agent"; then
        plist="$AGENTS_DIR/${agent}.plist"
        if [ -f "$plist" ]; then
            log "Agent $agent missing. Reloading..."
            launchctl load "$plist" 2>/dev/null || true
        fi
    fi
done

# --- 3. During work hours: is Screenpipe DB fresh? ---------------------------
HOUR=$(date +%H)
if [ "$HOUR" -ge 7 ] && [ "$HOUR" -le 19 ] && [ -f "$DB_PATH" ]; then
    DB_AGE=$(( $(date +%s) - $(stat -f %m "$DB_PATH") ))
    if [ "$DB_AGE" -gt 300 ]; then
        log "WARNING: Screenpipe DB is ${DB_AGE}s stale. Restarting..."
        pkill -f "screenpipe" 2>/dev/null || true
        sleep 3
        open -a screenpipe 2>/dev/null || true
    fi
fi
