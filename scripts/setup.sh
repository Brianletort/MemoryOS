#!/bin/bash
# MemoryOS first-time setup
# Usage: ./scripts/setup.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== MemoryOS Setup ==="
echo ""

# ---- Python check ----
if ! command -v python3 &>/dev/null; then
    echo "ERROR: Python 3 is required. Install via: brew install python@3.12"
    exit 1
fi
PY_VER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "  Python:  $PY_VER"

# ---- Virtual environment ----
if [[ ! -d "$REPO_DIR/.venv" ]]; then
    echo ""
    echo "--- Creating virtual environment ---"
    python3 -m venv "$REPO_DIR/.venv"
fi
source "$REPO_DIR/.venv/bin/activate"
pip install --upgrade pip --quiet
echo "  Venv:    $REPO_DIR/.venv"

# ---- Python dependencies ----
echo ""
echo "--- Installing Python dependencies ---"
pip install -r "$REPO_DIR/requirements.txt" --quiet
echo "  Done."

# ---- Homebrew system dependencies ----
if command -v brew &>/dev/null; then
    echo ""
    echo "--- Checking system dependencies (Homebrew) ---"
    for pkg in pandoc blackhole-2ch; do
        if brew list "$pkg" &>/dev/null; then
            echo "  $pkg: installed"
        else
            echo "  Installing $pkg..."
            brew install "$pkg"
        fi
    done
else
    echo ""
    echo "WARNING: Homebrew not found. Install pandoc and blackhole-2ch manually."
fi

# ---- Directories ----
mkdir -p "$REPO_DIR/config"
mkdir -p "$REPO_DIR/logs"

# ---- Config file ----
if [[ ! -f "$REPO_DIR/config/config.yaml" ]]; then
    echo ""
    echo "--- Creating config/config.yaml from template ---"
    cp "$REPO_DIR/config/config.yaml.example" "$REPO_DIR/config/config.yaml"
    echo ""
    echo "  IMPORTANT: Edit config/config.yaml and set your paths:"
    echo "    - obsidian_vault  (required)"
    echo "    - screenpipe.db_path"
    echo "    - outlook.db_path  OR  leave Mail.app defaults"
    echo "    - onedrive.sync_dir  (if you use OneDrive)"
    echo ""
    echo "  Open in your editor:"
    echo "    \$EDITOR $REPO_DIR/config/config.yaml"
else
    echo ""
    echo "  config/config.yaml already exists -- skipping."
fi

# ---- Make scripts executable ----
chmod +x "$REPO_DIR/scripts"/*.sh 2>/dev/null || true

# ---- Quick validation ----
echo ""
echo "--- Validating ---"
"$REPO_DIR/.venv/bin/python3" -c "
from src.common.config import load_config
try:
    cfg = load_config('$REPO_DIR/config/config.yaml')
    print('  Config loaded successfully.')
    print(f'  Vault: {cfg[\"obsidian_vault\"]}')
except Exception as e:
    print(f'  Config validation failed: {e}')
    print('  Edit config/config.yaml and re-run this script.')
" 2>&1 || true

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "  1. Edit config/config.yaml with your paths"
echo "  2. Start Screenpipe (brew services start screenpipe, or open the app)"
echo "  3. Run the smoke test:   ./scripts/smoke_test.sh"
echo "  4. Install automation:   ./scripts/install_launchd.sh"
echo "  5. Open the dashboard:   http://localhost:8765"
echo ""
