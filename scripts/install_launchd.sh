#!/bin/bash
# Install MemoryOS launchd agents from templates
# Generates .plist files with REPO_DIR and VENV_PYTHON substituted
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$(dirname "$SCRIPT_DIR")" && pwd)"
LAUNCHD_DIR="$REPO_DIR/launchd"
AGENTS_DIR="$HOME/Library/LaunchAgents"

# Detect venv python; fallback to system python3
VENV_PYTHON="$REPO_DIR/.venv/bin/python3"
if [[ ! -x "$VENV_PYTHON" ]]; then
    echo "Note: .venv not found, using system python3"
    VENV_PYTHON="$(command -v python3)"
fi

echo "=== Installing MemoryOS launchd agents ==="
echo "  REPO_DIR:     $REPO_DIR"
echo "  VENV_PYTHON:  $VENV_PYTHON"
echo "  Templates:    $LAUNCHD_DIR"
echo "  Target:       $AGENTS_DIR"
echo ""

mkdir -p "$AGENTS_DIR"
mkdir -p "$REPO_DIR/logs"

# Process each template
for template in "$LAUNCHD_DIR"/com.memoryos.*.plist.template; do
    [[ -f "$template" ]] || continue
    name=$(basename "$template" .plist.template)
    plist_name="${name}.plist"
    dest="$AGENTS_DIR/$plist_name"

    echo "Installing $plist_name..."
    if [[ -f "$dest" ]]; then
        echo "  Unloading existing agent..."
        launchctl unload "$dest" 2>/dev/null || true
    fi
    sed -e "s|{{REPO_DIR}}|$REPO_DIR|g" -e "s|{{VENV_PYTHON}}|$VENV_PYTHON|g" \
        "$template" > "$dest"
    launchctl load "$dest"
    echo "  Loaded successfully."
    echo ""
done

echo "=== All agents installed ==="
echo "Check status with: launchctl list | grep memoryos"
echo "View logs in: $REPO_DIR/logs/"
