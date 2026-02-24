#!/bin/bash
# MemoryOS -- Grant macOS permissions for autonomous operation
# Usage: ./scripts/grant_permissions.sh [--verify]
#
# --verify   Check whether permissions are granted and exit 0/1.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

VENV_PYTHON="$REPO_DIR/.venv/bin/python3"
if [[ ! -x "$VENV_PYTHON" ]]; then
    VENV_PYTHON="$(command -v python3)"
fi

PYTHON_REAL="$(readlink -f "$VENV_PYTHON" 2>/dev/null || python3 -c 'import sys; print(sys.executable)')"
FRAMEWORK_DIR="$(dirname "$(dirname "$PYTHON_REAL")")"
PYTHON_APP="$FRAMEWORK_DIR/Resources/Python.app"

VERIFY_ONLY=false
if [[ "${1:-}" == "--verify" ]]; then
    VERIFY_ONLY=true
fi

# ── Helpers ──────────────────────────────────────────────────────────────────

fda_granted() {
    python3 -c "import os; os.listdir('$HOME/Library/Mail')" 2>/dev/null
}

automation_hint() {
    if [[ -d "$PYTHON_APP" ]]; then
        echo "$PYTHON_APP"
    else
        echo "$PYTHON_REAL"
    fi
}

lsuielement_set() {
    if [[ -d "$PYTHON_APP" ]]; then
        local val
        val="$(defaults read "$PYTHON_APP/Contents/Info.plist" LSUIElement 2>/dev/null || echo "0")"
        [[ "$val" == "1" ]]
    else
        return 1
    fi
}

# ── Verify-only mode ────────────────────────────────────────────────────────

if $VERIFY_ONLY; then
    rc=0

    printf "%-30s" "Full Disk Access:"
    if fda_granted; then
        echo "GRANTED"
    else
        echo "NOT GRANTED"
        rc=1
    fi

    printf "%-30s" "LSUIElement:"
    if lsuielement_set; then
        echo "SET"
    else
        echo "NOT SET"
        rc=1
    fi

    printf "%-30s%s\n" "Python binary:" "$PYTHON_REAL"
    if [[ -d "$PYTHON_APP" ]]; then
        printf "%-30s%s\n" "Python.app:" "$PYTHON_APP"
    fi

    exit $rc
fi

# ── Interactive grant flow ──────────────────────────────────────────────────

echo "=== MemoryOS macOS Permissions ==="
echo ""
echo "MemoryOS needs macOS permissions to run autonomously without popups."
echo ""
echo "  Python binary:  $PYTHON_REAL"
if [[ -d "$PYTHON_APP" ]]; then
    echo "  Python.app:     $PYTHON_APP"
fi
echo ""

# ── 1. Full Disk Access ──
echo "--- 1. Full Disk Access ---"
echo ""
echo "  This stops the 'python3 would like to access data from other apps' popup."
echo ""
if fda_granted; then
    echo "  Status: GRANTED"
else
    echo "  Status: NOT GRANTED"
    echo ""
    echo "  Steps:"
    echo "    1. System Settings > Privacy & Security > Full Disk Access"
    echo "    2. Click '+', press Cmd+Shift+G, and paste this path:"
    echo ""
    if [[ -d "$PYTHON_APP" ]]; then
        echo "       $PYTHON_APP"
    else
        echo "       $PYTHON_REAL"
    fi
    echo ""
    echo "    3. Toggle it ON"
    echo ""
    echo "  Opening System Settings now..."
    open "x-apple.systempreferences:com.apple.preference.security?Privacy_AllFiles" 2>/dev/null || true
fi

echo ""

# ── 2. Automation (AppleScript control of Mail + Calendar) ──
echo "--- 2. Automation ---"
echo ""
echo "  This lets MemoryOS read email/calendar and send reports via Mail.app."
echo ""
echo "  Steps:"
echo "    1. System Settings > Privacy & Security > Automation"
echo "    2. Find 'python3' or 'Python'"
echo "    3. Toggle ON: Mail, Calendar"
echo ""
echo "  If you don't see Python listed, it will appear the next time a"
echo "  MemoryOS agent runs and you click 'Allow' on the popup."
echo ""

# ── 3. Suppress Python Dock icon ──
echo "--- 3. Suppress Python Dock Icon ---"
echo ""
if [[ -d "$PYTHON_APP" ]]; then
    PLIST="$PYTHON_APP/Contents/Info.plist"
    CURRENT="$(defaults read "$PLIST" LSUIElement 2>/dev/null || echo "0")"
    if [[ "$CURRENT" == "1" ]]; then
        echo "  Status: Already set (LSUIElement=1)"
    else
        defaults write "$PLIST" LSUIElement -bool true 2>/dev/null && \
            echo "  Set LSUIElement=1 on $PLIST" || \
            echo "  Could not set (may need to run with sudo)"
    fi
else
    echo "  Python.app bundle not found at $PYTHON_APP -- skipping"
fi

echo ""

# ── 4. Homebrew upgrade warning ──
echo "--- 4. Homebrew Python Upgrade Warning ---"
echo ""
echo "  IMPORTANT: Homebrew Python upgrades change the binary path, which"
echo "  invalidates Full Disk Access and Automation permissions."
echo ""
echo "  After running 'brew upgrade python', re-run this script:"
echo "    $SCRIPT_DIR/grant_permissions.sh"
echo ""
echo "  To check current permission status at any time:"
echo "    $SCRIPT_DIR/grant_permissions.sh --verify"
echo ""

echo "=== Done ==="
echo ""
echo "After granting permissions, restart MemoryOS agents:"
echo "  launchctl bootout gui/\$(id -u) ~/Library/LaunchAgents/com.memoryos.*.plist 2>/dev/null"
echo "  launchctl bootstrap gui/\$(id -u) ~/Library/LaunchAgents/com.memoryos.*.plist"
echo ""
