#!/bin/bash
# MemoryOS Sentinel: nuclear fallback watchdog
# Pure bash -- no Python, no venv, no dependencies
# Runs every 60s via launchd. Ensures Screenpipe and all agents are alive.
set -uo pipefail

REPO_DIR="$(cd "$(dirname "$(dirname "$0")")" && pwd)"
AGENTS_DIR="$HOME/Library/LaunchAgents"
LOG="$REPO_DIR/logs/sentinel.log"
DB_PATH="$HOME/.screenpipe/db.sqlite"
SP_COOLDOWN_FILE="/tmp/memoryos_sentinel_sp_cooldown"
SP_MAX_RETRIES=5
SP_COOLDOWN_SECS=300
SP_BACKOFF_SECS=1200
SP_API="http://localhost:3030"

mkdir -p "$REPO_DIR/logs"

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') [SENTINEL] $*" >> "$LOG"; }

activate_preferred_audio() {
    local preferred
    preferred=$(grep 'preferred_input_device:' "$REPO_DIR/config/config.yaml" 2>/dev/null \
        | head -1 | sed 's/.*preferred_input_device:\s*//' | sed 's/^"\(.*\)"$/\1/')
    if [ -z "$preferred" ] || [ "$preferred" = "null" ]; then
        return
    fi
    sleep 15
    local payload
    payload=$(printf '{"device_name":"%s"}' "$preferred")
    curl -sf -X POST "$SP_API/audio/device/start" \
        -H "Content-Type: application/json" -d "$payload" \
        > /dev/null 2>&1 && log "Activated audio device: $preferred" || true

    # Stop non-preferred input devices to prevent USB reconnect cascades
    local devices
    devices=$(curl -sf "$SP_API/audio/list" 2>/dev/null) || return
    echo "$devices" | python3 -c "
import sys, json
for d in json.load(sys.stdin):
    n = d.get('name', '')
    if n and '(output)' not in n and n != '$preferred':
        print(n)
" 2>/dev/null | while IFS= read -r dev; do
        curl -sf -X POST "$SP_API/audio/device/stop" \
            -H "Content-Type: application/json" \
            -d "{\"device_name\":\"$dev\"}" > /dev/null 2>&1 \
            && log "Stopped unwanted device: $dev"
    done
}

# --- 1. Is Screenpipe app running? -------------------------------------------
if ! pgrep -f "screenpipe" > /dev/null 2>&1; then
    log "CRITICAL: Screenpipe not running. Launching..."
    open -a screenpipe 2>/dev/null || true
    activate_preferred_audio &
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
    com.memoryos.resource-monitor
    com.memoryos.activity-summarizer
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

# --- 3. Is Screenpipe healthy? (24/7 monitoring) -----------------------------
SP_HEALTHY=false
SP_STATUS=$(curl -sf --max-time 5 "$SP_API/health" 2>/dev/null) || SP_STATUS=""

if [ -n "$SP_STATUS" ]; then
    if echo "$SP_STATUS" | grep -q '"status":\s*"healthy"'; then
        SP_HEALTHY=true
    fi
fi

if $SP_HEALTHY; then
    if [ -f "$SP_COOLDOWN_FILE" ]; then
        rm -f "$SP_COOLDOWN_FILE"
        log "Screenpipe healthy via API. Cooldown reset."
    fi
else
    NOW=$(date +%s)
    DB_MTIME=0
    [ -f "$DB_PATH" ] && DB_MTIME=$(stat -f %m "$DB_PATH")
    WAL_PATH="${DB_PATH}-wal"
    WAL_MTIME=0
    [ -f "$WAL_PATH" ] && WAL_MTIME=$(stat -f %m "$WAL_PATH")
    NEWEST_MTIME=$DB_MTIME
    [ "$WAL_MTIME" -gt "$NEWEST_MTIME" ] && NEWEST_MTIME=$WAL_MTIME

    if [ "$NEWEST_MTIME" -eq 0 ]; then
        log "WARNING: No Screenpipe DB files found and API unreachable."
    else
        DB_AGE=$((NOW - NEWEST_MTIME))
        if [ "$DB_AGE" -gt 300 ]; then
            RETRIES=0
            LAST_RESTART=0
            if [ -f "$SP_COOLDOWN_FILE" ]; then
                RETRIES=$(head -1 "$SP_COOLDOWN_FILE" 2>/dev/null || echo 0)
                LAST_RESTART=$(tail -1 "$SP_COOLDOWN_FILE" 2>/dev/null || echo 0)
            fi
            ELAPSED=$((NOW - LAST_RESTART))

            if [ "$RETRIES" -ge "$SP_MAX_RETRIES" ] && [ "$ELAPSED" -lt "$SP_BACKOFF_SECS" ]; then
                log "CRITICAL: Screenpipe unhealthy ${DB_AGE}s, exhausted $SP_MAX_RETRIES retries. Backing off until $((SP_BACKOFF_SECS - ELAPSED))s remain."
            elif [ "$ELAPSED" -ge "$SP_COOLDOWN_SECS" ]; then
                if [ "$ELAPSED" -ge "$SP_BACKOFF_SECS" ]; then RETRIES=0; fi
                log "WARNING: Screenpipe API down and DB ${DB_AGE}s stale. Restarting (attempt $((RETRIES+1))/$SP_MAX_RETRIES)..."
                pkill -f "screenpipe" 2>/dev/null || true
                sleep 3
                open -a screenpipe 2>/dev/null || true
                printf '%s\n%s\n' "$((RETRIES+1))" "$NOW" > "$SP_COOLDOWN_FILE"
                activate_preferred_audio &
            fi
        else
            if [ -f "$SP_COOLDOWN_FILE" ]; then
                rm -f "$SP_COOLDOWN_FILE"
            fi
        fi
    fi
fi
