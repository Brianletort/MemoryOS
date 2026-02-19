#!/bin/bash
# Smoke test: run all extractors once and verify output
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

VENV_PYTHON="$REPO_DIR/.venv/bin/python3"
export PYTHONPATH="$REPO_DIR"

echo "=== MemoryOS Smoke Test ==="
echo "Repo: $REPO_DIR"
echo "Python: $VENV_PYTHON"
echo ""

# Check dependencies
echo "--- Checking dependencies ---"
"$VENV_PYTHON" -c "import yaml; print('  pyyaml:', yaml.__version__)" 2>/dev/null || echo "  MISSING: pyyaml"
"$VENV_PYTHON" -c "import dateutil; print('  python-dateutil: OK')" 2>/dev/null || echo "  MISSING: python-dateutil"
"$VENV_PYTHON" -c "import markdownify; print('  markdownify: OK')" 2>/dev/null || echo "  MISSING: markdownify"
"$VENV_PYTHON" -c "import msal; print('  msal:', msal.__version__)" 2>/dev/null || echo "  MISSING: msal"
"$VENV_PYTHON" -c "import pdfplumber; print('  pdfplumber: OK')" 2>/dev/null || echo "  MISSING: pdfplumber (optional)"
"$VENV_PYTHON" -c "import pptx; print('  python-pptx: OK')" 2>/dev/null || echo "  MISSING: python-pptx (optional)"
"$VENV_PYTHON" -c "import openpyxl; print('  openpyxl: OK')" 2>/dev/null || echo "  MISSING: openpyxl (optional)"
which pandoc >/dev/null 2>&1 && echo "  pandoc: $(pandoc --version | head -1)" || echo "  MISSING: pandoc (optional, for docx)"
echo ""

# Test 1: Screenpipe extractor (dry run)
echo "--- Test 1: Screenpipe Extractor (dry run) ---"
"$VENV_PYTHON" "$REPO_DIR/src/extractors/screenpipe_extractor.py" --dry-run 2>&1 | tail -20
echo ""

# Test 2: Outlook extractor (dry run, small batch)
echo "--- Test 2: Outlook Extractor (dry run) ---"
"$VENV_PYTHON" "$REPO_DIR/src/extractors/outlook_extractor.py" --dry-run 2>&1 | tail -20
echo ""

# Test 3: OneDrive extractor (dry run)
echo "--- Test 3: OneDrive Extractor (dry run) ---"
"$VENV_PYTHON" "$REPO_DIR/src/extractors/onedrive_extractor.py" --dry-run 2>&1 | tail -20
echo ""

# Test 4: Mail.app Email extractor (dry run, requires Mail.app configured)
echo "--- Test 4: Mail.app Email Extractor (dry run) ---"
"$VENV_PYTHON" "$REPO_DIR/src/extractors/mail_app_extractor.py" --dry-run --days-back 1 2>&1 | tail -20
echo ""

# Test 5: Calendar.app extractor (dry run, requires Calendar.app configured)
echo "--- Test 5: Calendar.app Extractor (dry run) ---"
"$VENV_PYTHON" "$REPO_DIR/src/extractors/calendar_app_extractor.py" --dry-run --days-back 1 --days-forward 3 2>&1 | tail -20
echo ""

# Test 6: Memory index
echo "--- Test 6: Memory Index ---"
"$VENV_PYTHON" -m src.memory.cli stats 2>/dev/null || echo "  Index not built yet. Run: python3 -m src.memory.cli reindex"
echo ""

echo "=== Smoke test complete ==="
echo ""
echo "Next steps:"
echo "  1. Run extractors for real (writes markdown to vault):"
echo "     $VENV_PYTHON $REPO_DIR/src/extractors/screenpipe_extractor.py"
echo "     $VENV_PYTHON $REPO_DIR/src/extractors/mail_app_extractor.py"
echo "     $VENV_PYTHON $REPO_DIR/src/extractors/calendar_app_extractor.py"
echo "  2. Build the memory index:"
echo "     $VENV_PYTHON -m src.memory.cli reindex"
echo "  3. Install automation:"
echo "     ./scripts/install_launchd.sh"
