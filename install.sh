#!/bin/bash
# MemoryOS Installer
# Usage: ./install.sh
#
# Installs all system dependencies, creates a Python virtual environment,
# installs Python packages, then launches the Setup Wizard in your browser.
#
# Safe to re-run -- skips anything already installed.
set -euo pipefail

# ── Globals ──────────────────────────────────────────────────────────────────

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_FILE="$REPO_DIR/logs/install.log"
STEP=0
TOTAL_STEPS=10
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

# ── Helpers ──────────────────────────────────────────────────────────────────

step() {
    STEP=$((STEP + 1))
    echo ""
    echo -e "${BLUE}[${STEP}/${TOTAL_STEPS}]${NC} ${BOLD}$1${NC}"
}

ok() {
    echo -e "  ${GREEN}✓${NC} $1"
}

warn() {
    echo -e "  ${YELLOW}⚠${NC} $1"
}

fail() {
    echo -e "  ${RED}✗ $1${NC}"
    echo ""
    echo -e "  ${RED}Installation failed at step ${STEP}. Check logs:${NC}"
    echo "    $LOG_FILE"
    exit 1
}

log() {
    echo "[$(date '+%H:%M:%S')] $*" >> "$LOG_FILE"
}

# ── Pre-flight ───────────────────────────────────────────────────────────────

mkdir -p "$REPO_DIR/logs" "$REPO_DIR/config"
echo "MemoryOS install started at $(date)" > "$LOG_FILE"

echo ""
echo -e "${BOLD}═══════════════════════════════════════${NC}"
echo -e "${BOLD}  MemoryOS Installer${NC}"
echo -e "${BOLD}═══════════════════════════════════════${NC}"

# ── Step 1: macOS version ────────────────────────────────────────────────────

step "Checking macOS version"

MACOS_VER=$(sw_vers -productVersion 2>/dev/null || echo "0")
MACOS_MAJOR=$(echo "$MACOS_VER" | cut -d. -f1)
log "macOS version: $MACOS_VER (major: $MACOS_MAJOR)"

if [[ "$MACOS_MAJOR" -lt 13 ]]; then
    fail "macOS 13 (Ventura) or later is required. You have $MACOS_VER."
fi
ok "macOS $MACOS_VER"

# ── Step 2: Xcode CLI tools ─────────────────────────────────────────────────

step "Checking Xcode Command Line Tools"

if xcode-select -p &>/dev/null; then
    ok "Xcode CLI tools installed"
else
    echo "  Installing Xcode Command Line Tools (this may take a few minutes)..."
    xcode-select --install 2>/dev/null || true
    echo ""
    echo "  A system dialog should have appeared. Click 'Install' and wait."
    echo "  Once it finishes, re-run this script:"
    echo "    ./install.sh"
    exit 0
fi

# ── Step 3: Homebrew ─────────────────────────────────────────────────────────

step "Checking Homebrew"

if command -v brew &>/dev/null; then
    ok "Homebrew installed ($(brew --version | head -1))"
else
    echo "  Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)" >> "$LOG_FILE" 2>&1 || fail "Homebrew installation failed"

    # Add to PATH for Apple Silicon
    if [[ -f /opt/homebrew/bin/brew ]]; then
        eval "$(/opt/homebrew/bin/brew shellenv)"
    fi
    ok "Homebrew installed"
fi

# ── Step 4: Python ───────────────────────────────────────────────────────────

step "Checking Python"

NEED_PYTHON=false
if command -v python3 &>/dev/null; then
    PY_VER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "0.0")
    PY_MAJOR=$(echo "$PY_VER" | cut -d. -f1)
    PY_MINOR=$(echo "$PY_VER" | cut -d. -f2)
    if [[ "$PY_MAJOR" -ge 3 && "$PY_MINOR" -ge 9 ]]; then
        ok "Python $PY_VER"
    else
        NEED_PYTHON=true
    fi
else
    NEED_PYTHON=true
fi

if $NEED_PYTHON; then
    echo "  Installing Python 3.12 via Homebrew..."
    brew install python@3.12 >> "$LOG_FILE" 2>&1 || fail "Python installation failed"
    ok "Python 3.12 installed"
fi

# ── Step 5: pandoc ───────────────────────────────────────────────────────────

step "Checking pandoc"

if command -v pandoc &>/dev/null; then
    ok "pandoc installed ($(pandoc --version | head -1))"
else
    echo "  Installing pandoc..."
    brew install pandoc >> "$LOG_FILE" 2>&1 || fail "pandoc installation failed"
    ok "pandoc installed"
fi

# ── Step 6: BlackHole (optional) ─────────────────────────────────────────────

step "Checking BlackHole 2ch (optional, for meeting audio capture)"

if brew list blackhole-2ch &>/dev/null 2>&1; then
    ok "BlackHole 2ch installed"
else
    echo "  Installing BlackHole 2ch..."
    if brew install blackhole-2ch >> "$LOG_FILE" 2>&1; then
        ok "BlackHole 2ch installed"
    else
        warn "BlackHole 2ch install failed (non-critical). Meeting audio capture may be limited."
        warn "Install later: brew install blackhole-2ch"
    fi
fi

# ── Step 7: Python virtual environment ───────────────────────────────────────

step "Setting up Python virtual environment"

if [[ -d "$REPO_DIR/.venv" ]] && "$REPO_DIR/.venv/bin/python3" -c "import sys" &>/dev/null; then
    ok "Virtual environment exists"
else
    echo "  Creating virtual environment..."
    python3 -m venv "$REPO_DIR/.venv" >> "$LOG_FILE" 2>&1 || fail "Failed to create virtual environment"
    ok "Virtual environment created"
fi

source "$REPO_DIR/.venv/bin/activate"
log "Using Python: $(which python3)"

# ── Step 8: Python dependencies ──────────────────────────────────────────────

step "Installing Python dependencies"

echo "  This may take a minute on first install..."
pip install --upgrade pip --quiet >> "$LOG_FILE" 2>&1
pip install -r "$REPO_DIR/requirements.txt" --quiet >> "$LOG_FILE" 2>&1 || fail "pip install failed. Check $LOG_FILE"
ok "All Python packages installed"

# ── Step 9: Configuration files ──────────────────────────────────────────────

step "Preparing configuration"

if [[ ! -f "$REPO_DIR/config/config.yaml" ]]; then
    cp "$REPO_DIR/config/config.yaml.example" "$REPO_DIR/config/config.yaml"
    ok "Created config/config.yaml from template"
else
    ok "config/config.yaml already exists"
fi

if [[ ! -f "$REPO_DIR/.env.local" ]]; then
    if [[ -f "$REPO_DIR/.env.example" ]]; then
        cp "$REPO_DIR/.env.example" "$REPO_DIR/.env.local"
        ok "Created .env.local from template"
    else
        touch "$REPO_DIR/.env.local"
        ok "Created empty .env.local"
    fi
else
    ok ".env.local already exists"
fi

chmod +x "$REPO_DIR/scripts"/*.sh 2>/dev/null || true
chmod +x "$REPO_DIR/scripts/memoryos" 2>/dev/null || true

# Suppress Python Dock icon
PYTHON_REAL="$(readlink -f "$REPO_DIR/.venv/bin/python3" 2>/dev/null || "$REPO_DIR/.venv/bin/python3" -c 'import sys; print(sys.executable)')"
FRAMEWORK_DIR="$(dirname "$(dirname "$PYTHON_REAL")")"
PYTHON_APP_PLIST="$FRAMEWORK_DIR/Resources/Python.app/Contents/Info.plist"
if [[ -f "$PYTHON_APP_PLIST" ]]; then
    defaults write "$PYTHON_APP_PLIST" LSUIElement -bool true 2>/dev/null && \
        log "Set LSUIElement=1" || log "Could not set LSUIElement"
fi

# ── Step 10: Launch Setup Wizard ─────────────────────────────────────────────

step "Launching MemoryOS"

VENV_PYTHON="$REPO_DIR/.venv/bin/python3"

# Kill any existing dashboard
pkill -f "src/dashboard/app.py" 2>/dev/null || true
sleep 1

PYTHONPATH="$REPO_DIR" "$VENV_PYTHON" "$REPO_DIR/src/dashboard/app.py" >> "$REPO_DIR/logs/dashboard.log" 2>&1 &
DASH_PID=$!
sleep 2

if kill -0 "$DASH_PID" 2>/dev/null; then
    ok "Dashboard running at http://localhost:8765 (PID $DASH_PID)"
    open "http://localhost:8765" 2>/dev/null || true
else
    warn "Dashboard failed to start. Check logs/dashboard.log"
fi

# ── Summary ──────────────────────────────────────────────────────────────────

echo ""
echo -e "${BOLD}═══════════════════════════════════════${NC}"
echo -e "${BOLD}  Installation Complete${NC}"
echo -e "${BOLD}═══════════════════════════════════════${NC}"
echo ""
echo "  The Setup Wizard should have opened in your browser."
echo "  If not, go to: http://localhost:8765"
echo ""
echo "  The wizard will walk you through:"
echo "    1. Checking dependencies"
echo "    2. Setting your Obsidian vault path"
echo "    3. Configuring your LLM provider"
echo "    4. Activating background agents"
echo ""
echo -e "  ${YELLOW}macOS Permissions:${NC}"
echo "    After setup, grant Full Disk Access to Python for"
echo "    autonomous email/calendar extraction without popups:"
echo "      ./scripts/grant_permissions.sh"
echo ""
echo -e "  ${BOLD}Manage MemoryOS:${NC}"
echo "    ./scripts/memoryos start     Start all agents"
echo "    ./scripts/memoryos stop      Stop all agents"
echo "    ./scripts/memoryos status    Check what's running"
echo "    ./scripts/memoryos doctor    Full health check"
echo ""
echo "  Logs: $REPO_DIR/logs/"
echo ""
