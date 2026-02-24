#!/bin/bash
# MemoryOS developer setup
# Creates/updates the Python venv and installs dependencies.
# For full installation (including system deps), use ./install.sh instead.
#
# Usage: ./scripts/setup.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== MemoryOS Developer Setup ==="
echo ""

# ── Python check ─────────────────────────────────────────────────────────────

if ! command -v python3 &>/dev/null; then
    echo "ERROR: Python 3 is required. Install via: brew install python@3.12"
    echo "  Or run ./install.sh for a full guided installation."
    exit 1
fi
PY_VER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "  Python:  $PY_VER"

# ── Virtual environment ──────────────────────────────────────────────────────

if [[ ! -d "$REPO_DIR/.venv" ]]; then
    echo ""
    echo "--- Creating virtual environment ---"
    python3 -m venv "$REPO_DIR/.venv"
fi
source "$REPO_DIR/.venv/bin/activate"
pip install --upgrade pip --quiet
echo "  Venv:    $REPO_DIR/.venv"

# ── Python dependencies ──────────────────────────────────────────────────────

echo ""
echo "--- Installing Python dependencies ---"
pip install -r "$REPO_DIR/requirements.txt" --quiet
echo "  Done."

# ── Directories & config ─────────────────────────────────────────────────────

mkdir -p "$REPO_DIR/config" "$REPO_DIR/logs"

if [[ ! -f "$REPO_DIR/config/config.yaml" ]]; then
    cp "$REPO_DIR/config/config.yaml.example" "$REPO_DIR/config/config.yaml"
    echo ""
    echo "  Created config/config.yaml from template."
    echo "  Edit it or use the Setup Wizard: http://localhost:8765"
else
    echo ""
    echo "  config/config.yaml already exists."
fi

if [[ ! -f "$REPO_DIR/.env.local" ]] && [[ -f "$REPO_DIR/.env.example" ]]; then
    cp "$REPO_DIR/.env.example" "$REPO_DIR/.env.local"
    echo "  Created .env.local from template."
fi

chmod +x "$REPO_DIR/scripts"/*.sh 2>/dev/null || true
chmod +x "$REPO_DIR/scripts/memoryos" 2>/dev/null || true

# ── Validation ────────────────────────────────────────────────────────────────

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
    print('  Edit config/config.yaml or use the Setup Wizard.')
" 2>&1 || true

echo ""
echo "=== Setup Complete ==="
echo ""
echo "  Next steps:"
echo "    1. Start the dashboard:  python3 src/dashboard/app.py"
echo "    2. Open http://localhost:8765 and use the Setup Wizard"
echo "    3. Or install agents:    ./scripts/install_launchd.sh"
echo ""
