#!/bin/bash
# Uninstall MemoryOS launchd agents
set -euo pipefail

AGENTS_DIR="$HOME/Library/LaunchAgents"

echo "=== Uninstalling MemoryOS launchd agents ==="

for plist in "$AGENTS_DIR"/com.memoryos.*.plist; do
    [ -f "$plist" ] || continue
    name=$(basename "$plist")
    echo "Unloading $name..."
    launchctl unload "$plist" 2>/dev/null || true
    rm -f "$plist"
    echo "  Removed."
done

echo ""
echo "=== All agents removed ==="
