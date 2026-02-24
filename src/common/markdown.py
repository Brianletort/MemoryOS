"""Markdown helpers: YAML frontmatter, path safety, text deduplication."""

from __future__ import annotations

import logging
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

logger = logging.getLogger("memoryos.markdown")


def yaml_frontmatter(metadata: dict[str, Any]) -> str:
    """Render a YAML frontmatter block.

    Values are serialised simply: strings are quoted only if needed,
    lists become YAML arrays, booleans/numbers stay as-is.
    """
    lines = ["---"]
    for key, value in metadata.items():
        lines.append(f"{key}: {_yaml_value(value)}")
    lines.append("---")
    return "\n".join(lines)


def _yaml_value(v: Any) -> str:
    if v is None:
        return '""'
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, list):
        if not v:
            return "[]"
        items = "\n".join(f"  - {_yaml_value(i)}" for i in v)
        return f"\n{items}"
        # Note: this will result in key:\n  - item format
    s = str(v)
    # Quote if it contains YAML-special chars
    if any(c in s for c in ":#{}[]|>&*!%@`'\"\\,\n"):
        escaped = s.replace('"', '\\"')
        return f'"{escaped}"'
    return s


def sanitize_filename(name: str, max_length: int = 100) -> str:
    """Create a filesystem-safe filename from a string."""
    # Normalize unicode
    name = unicodedata.normalize("NFKD", name)
    # Replace problematic chars
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "-", name)
    # Collapse runs of dashes/spaces
    name = re.sub(r"[-\s]+", "-", name).strip("-")
    # Truncate
    if len(name) > max_length:
        name = name[:max_length].rstrip("-")
    return name or "untitled"


def safe_vault_path(vault_root: Path, subdir: str, filename: str) -> Path:
    """Resolve a path inside the vault, preventing directory traversal.

    Raises ValueError if the resolved path escapes the vault.
    """
    target = (vault_root / subdir / filename).resolve()
    vault_resolved = vault_root.resolve()
    if not str(target).startswith(str(vault_resolved)):
        raise ValueError(f"Path escapes vault: {target} is not under {vault_resolved}")
    return target


def text_similarity(a: str, b: str) -> float:
    """Quick similarity ratio between two strings (0.0-1.0)."""
    if not a or not b:
        return 0.0
    # Use first 500 chars for speed on large OCR blocks
    return SequenceMatcher(None, a[:500], b[:500]).ratio()


def write_markdown(path: Path, content: str) -> None:
    """Write markdown content to a file, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        logger.debug("Overwriting existing file (%d bytes): %s", path.stat().st_size, path)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)
    logger.debug("Wrote %d bytes to %s", len(content), path)


def append_markdown(path: Path, content: str) -> None:
    """Append markdown content to a file, creating it if it doesn't exist."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(content)
    logger.debug("Appended %d bytes to %s", len(content), path)


def clean_ocr_text(text: str, max_length: int = 2000) -> str:
    """Clean up raw OCR text: remove excessive whitespace, truncate if huge."""
    if not text:
        return ""
    # Collapse whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    if len(text) > max_length:
        text = text[:max_length] + "\n[...truncated]"
    return text
