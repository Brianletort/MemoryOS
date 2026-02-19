"""Incremental-cursor state management for MemoryOS extractors.

State is stored as a flat JSON file, keyed by extractor name.
Each extractor stores its own cursor dict (e.g. last processed ID or timestamp).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("memoryos.state")


def load_state(state_path: Path | str) -> dict[str, Any]:
    """Load the full state dict. Returns empty dict if file missing."""
    path = Path(state_path)
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Corrupt state file %s, resetting: %s", path, exc)
        return {}


def save_state(state_path: Path | str, state: dict[str, Any]) -> None:
    """Atomically write the full state dict."""
    path = Path(state_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2, default=str)
    tmp.replace(path)


def get_cursor(state: dict[str, Any], extractor: str, key: str, default: Any = 0) -> Any:
    """Get a single cursor value for an extractor."""
    return state.get(extractor, {}).get(key, default)


def set_cursor(state: dict[str, Any], extractor: str, key: str, value: Any) -> None:
    """Set a single cursor value for an extractor."""
    state.setdefault(extractor, {})[key] = value
