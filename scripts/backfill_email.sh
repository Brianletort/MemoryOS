#!/bin/bash
# One-time backfill: import all existing emails from Outlook into Obsidian
# This processes ~65,000 emails. Expected time: ~9 minutes with 8 workers.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

VENV_PYTHON="$REPO_DIR/.venv/bin/python3"
export PYTHONPATH="$REPO_DIR"

echo "=== MemoryOS Email Backfill ==="
echo ""
echo "This will import ALL emails from your Outlook local database into"
echo "your Obsidian vault as individual markdown files."
echo ""
echo "Estimated time: depends on mailbox size"
echo "Output: configured Obsidian vault under 00_inbox/YYYY/MM/DD/"
echo ""
read -p "Continue? (y/N) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

echo ""
echo "Starting backfill..."
time "$VENV_PYTHON" "$REPO_DIR/src/extractors/outlook_extractor.py" --backfill

echo ""
echo "=== Backfill complete ==="
echo "Check output in your Obsidian vault under 00_inbox/"
