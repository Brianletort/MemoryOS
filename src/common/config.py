"""Load and validate MemoryOS configuration from config.yaml."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger("memoryos")

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "config.yaml"


def load_config(config_path: Path | str | None = None) -> dict[str, Any]:
    """Load configuration from YAML file, with env-var overrides.

    Environment variable overrides (if set):
        MEMORYOS_OBSIDIAN_VAULT  -> obsidian_vault
        MEMORYOS_SCREENPIPE_DB   -> screenpipe.db_path
        MEMORYOS_OUTLOOK_DB      -> outlook.db_path
        MEMORYOS_ONEDRIVE_DIR    -> onedrive.sync_dir
        MEMORYOS_STATE_FILE      -> state_file
        MEMORYOS_LOG_DIR         -> log_dir
    """
    path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, encoding="utf-8") as fh:
        cfg: dict[str, Any] = yaml.safe_load(fh)

    # Apply env-var overrides
    _env_override(cfg, "MEMORYOS_OBSIDIAN_VAULT", "obsidian_vault")
    _env_override(cfg, "MEMORYOS_SCREENPIPE_DB", "screenpipe", "db_path")
    _env_override(cfg, "MEMORYOS_OUTLOOK_DB", "outlook", "db_path")
    _env_override(cfg, "MEMORYOS_ONEDRIVE_DIR", "onedrive", "sync_dir")
    _env_override(cfg, "MEMORYOS_STATE_FILE", "state_file")
    _env_override(cfg, "MEMORYOS_LOG_DIR", "log_dir")

    _validate(cfg)
    return cfg


def _env_override(cfg: dict, env_key: str, *keys: str) -> None:
    """Override a nested config value from an environment variable."""
    val = os.environ.get(env_key)
    if val is None:
        return
    target = cfg
    for k in keys[:-1]:
        target = target.setdefault(k, {})
    target[keys[-1]] = val


def _validate(cfg: dict[str, Any]) -> None:
    """Validate that required paths exist or can be created."""
    vault = Path(cfg["obsidian_vault"])
    if not vault.is_dir():
        raise ValueError(f"Obsidian vault not found: {vault}")

    sp_db = Path(cfg["screenpipe"]["db_path"])
    if not sp_db.is_file():
        logger.warning("Screenpipe DB not found at %s — screenpipe extractor will skip", sp_db)

    ol_db = Path(cfg["outlook"]["db_path"])
    if not ol_db.is_file():
        logger.warning("Outlook DB not found at %s — outlook extractor will skip", ol_db)

    od_dir = Path(cfg["onedrive"]["sync_dir"])
    if not od_dir.is_dir():
        logger.warning("OneDrive sync dir not found at %s — onedrive extractor will skip", od_dir)


def resolve_output_dir(cfg: dict[str, Any], key: str) -> Path:
    """Return the absolute path for an output subdirectory, creating it if needed.

    ``key`` must be one of the keys under ``cfg['output']``.
    """
    vault = Path(cfg["obsidian_vault"])
    subdir = cfg["output"][key]
    out = vault / subdir
    out.mkdir(parents=True, exist_ok=True)
    return out


def setup_logging(cfg: dict[str, Any]) -> None:
    """Configure root logging: stderr + rotating file."""
    log_dir = Path(cfg["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)

    from logging.handlers import RotatingFileHandler

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    root = logging.getLogger("memoryos")
    root.setLevel(getattr(logging, cfg.get("log_level", "INFO")))

    # stderr
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    root.addHandler(sh)

    # rotating file
    fh = RotatingFileHandler(log_dir / "memoryos.log", maxBytes=5_000_000, backupCount=3)
    fh.setFormatter(fmt)
    root.addHandler(fh)
