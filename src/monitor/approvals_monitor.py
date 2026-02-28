#!/usr/bin/env python3
"""Approvals Queue Monitor -- scans emails and Screenpipe for pending approvals.

Deterministic (no LLM) monitor that:
  1. Queries the Memory Index for approval-request emails.
  2. Checks Screenpipe SQLite for evidence of completed approvals.
  3. Persists queue state to config/approvals_state.json.
  4. Writes daily reports to 90_reports/approvals-queue/.

Usage::

    python3 -m src.monitor.approvals_monitor
    python3 -m src.monitor.approvals_monitor --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.common.config import load_config, setup_logging
from src.common.markdown import yaml_frontmatter
from src.memory.index import MemoryIndex

logger = logging.getLogger("memoryos.approvals")

REPO_DIR = Path(__file__).resolve().parent.parent.parent
STATE_FILE = REPO_DIR / "config" / "approvals_state.json"
CONFIG_PATH = REPO_DIR / "config" / "config.yaml"
SKILLS_DIR = Path.home() / ".cursor" / "skills"

DEFAULT_EMAIL_LOOKBACK_DAYS = 7
DEFAULT_EVIDENCE_LOOKBACK_HOURS = 72
PRUNE_CLOSED_AFTER_DAYS = 30


# ── Default system rules (used when manifest is missing) ─────────────────────

DEFAULT_SYSTEMS: dict[str, dict[str, Any]] = {
    "concur": {
        "display_name": "Concur Expense",
        "email_rules": {
            "from_contains": ["concursolutions.com", "concur"],
            "subject_regex": r"(?i)(expense|report).*approv",
        },
        "external_id_regex": r"(?:Report|REP)[-#]?(\w+)",
        "evidence": {
            "browser_domains": ["concursolutions.com", "us2.concursolutions.com"],
            "success_text": ["approved", "approve this report"],
        },
    },
    "servicenow": {
        "display_name": "ServiceNow",
        "email_rules": {
            "from_contains": ["service-now.com", "servicenow"],
            "subject_regex": r"(?i)(approval|approve|requested)",
        },
        "external_id_regex": r"((?:INC|REQ|RITM|CHG|PRB)\d{5,})",
        "evidence": {
            "browser_domains": ["service-now.com"],
            "success_text": ["approved", "approval complete", "closed complete"],
        },
    },
    "docusign": {
        "display_name": "DocuSign",
        "email_rules": {
            "from_contains": ["docusign.net", "docusign.com"],
            "subject_regex": r"(?i)(review|sign|complete)",
        },
        "external_id_regex": None,
        "evidence": {
            "browser_domains": ["docusign.net", "docusign.com"],
            "success_text": ["completed", "you have signed", "signing complete"],
        },
    },
}


# ── State persistence ────────────────────────────────────────────────────────


def load_approvals_state() -> dict[str, Any]:
    if STATE_FILE.is_file():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {"items": {}, "last_email_scan": None, "last_evidence_scan_frame_id": 0}


def save_approvals_state(state: dict[str, Any]) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")
    tmp.rename(STATE_FILE)


# ── System rules loading ─────────────────────────────────────────────────────


def _load_system_rules() -> dict[str, dict[str, Any]]:
    """Load approval system rules from the skill manifest, falling back to defaults."""
    manifest_path = SKILLS_DIR / "approvals-queue" / "manifest.yaml"
    if manifest_path.is_file():
        try:
            manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
            systems = manifest.get("systems")
            if systems and isinstance(systems, dict):
                return systems
        except Exception as exc:
            logger.warning("Failed to load approvals manifest: %s", exc)
    return DEFAULT_SYSTEMS


# ── Email scanning ───────────────────────────────────────────────────────────


def _make_approval_id(system: str, email_path: str, external_id: str | None) -> str:
    if external_id:
        return f"{system}_{external_id}"
    digest = hashlib.sha256(email_path.encode()).hexdigest()[:12]
    return f"{system}_{digest}"


def _extract_external_id(text: str, pattern: str | None) -> str | None:
    if not pattern:
        return None
    m = re.search(pattern, text)
    return m.group(1) if m else None


def _email_matches_system(
    doc: dict[str, Any], rules: dict[str, Any],
) -> bool:
    """Check if an email document matches a system's email rules."""
    email_rules = rules.get("email_rules", {})
    content = doc.get("content", "")
    title = doc.get("title", "")

    from_contains = email_rules.get("from_contains", [])
    subject_regex = email_rules.get("subject_regex", "")

    from_match = False
    content_lower = content.lower()
    for pattern in from_contains:
        if pattern.lower() in content_lower:
            from_match = True
            break

    if not from_match:
        return False

    if subject_regex:
        if not re.search(subject_regex, title, re.IGNORECASE):
            return False

    return True


def _extract_from_field(content: str) -> str:
    """Extract the From field from email markdown content."""
    for line in content.splitlines()[:20]:
        stripped = line.strip()
        if stripped.lower().startswith("**from:**") or stripped.lower().startswith("from:"):
            return stripped.split(":", 1)[1].strip().strip("*").strip()
    return "unknown"


def scan_emails(
    systems: dict[str, dict[str, Any]],
    state: dict[str, Any],
    cfg: dict[str, Any],
) -> int:
    """Scan Memory Index for new approval-request emails. Returns count of new items."""
    mem_cfg = cfg.get("memory", {})
    db_path = mem_cfg.get("index_db", "config/memory.db")

    lookback = datetime.now() - timedelta(days=DEFAULT_EMAIL_LOOKBACK_DAYS)
    new_count = 0

    try:
        with MemoryIndex(db_path) as idx:
            emails = idx.get_recent(
                hours=DEFAULT_EMAIL_LOOKBACK_DAYS * 24,
                source_type="email",
                limit=500,
            )
    except Exception as exc:
        logger.error("Failed to query Memory Index: %s", exc)
        return 0

    existing_paths = {
        item["email_path"] for item in state.get("items", {}).values()
    }

    for doc in emails:
        path = doc.get("path", "")
        if path in existing_paths:
            continue

        for sys_name, rules in systems.items():
            if not _email_matches_system(doc, rules):
                continue

            content = doc.get("content", "")
            title = doc.get("title", "")
            ext_id_pattern = rules.get("external_id_regex")
            external_id = _extract_external_id(
                title + " " + content, ext_id_pattern,
            )

            approval_id = _make_approval_id(sys_name, path, external_id)

            if approval_id in state["items"]:
                continue

            created_at = doc.get("created_at", "")

            state["items"][approval_id] = {
                "approval_id": approval_id,
                "system": sys_name,
                "display_name": rules.get("display_name", sys_name),
                "subject": title,
                "from": _extract_from_field(content),
                "received_at": created_at[:19] if created_at else "",
                "external_id": external_id,
                "status": "pending",
                "status_changed_at": None,
                "evidence": None,
                "email_path": path,
                "link": None,
            }
            new_count += 1
            logger.info("New approval: %s -- %s", approval_id, title)
            break

    state["last_email_scan"] = datetime.now().isoformat(timespec="seconds")
    return new_count


# ── Screenpipe evidence scanning ─────────────────────────────────────────────


def _sanitize_url(url: str) -> str:
    """Strip query params from a URL for safe storage."""
    parsed = urlparse(url)
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", "", ""))


def _url_matches_domains(url: str, domains: list[str]) -> bool:
    if not url:
        return False
    try:
        host = urlparse(url).netloc.lower()
        return any(d.lower() in host for d in domains)
    except Exception:
        return False


SCREENPIPE_EVIDENCE_QUERY = """
SELECT f.id AS frame_id,
       f.timestamp,
       COALESCE(f.browser_url, '') AS browser_url,
       o.text
FROM frames f
JOIN ocr_text o ON o.frame_id = f.id
WHERE f.id > ?
  AND f.timestamp >= ?
ORDER BY f.id ASC
LIMIT 5000
"""


def scan_screenpipe_evidence(
    systems: dict[str, dict[str, Any]],
    state: dict[str, Any],
    cfg: dict[str, Any],
) -> int:
    """Check Screenpipe DB for evidence of completed approvals. Returns count closed."""
    sp_cfg = cfg.get("screenpipe", {})
    db_path_str = sp_cfg.get("db_path", "~/.screenpipe/db.sqlite")
    db_path = Path(db_path_str).expanduser()

    if not db_path.is_file():
        logger.warning("Screenpipe DB not found: %s", db_path)
        return 0

    pending = {
        aid: item for aid, item in state.get("items", {}).items()
        if item["status"] == "pending"
    }
    if not pending:
        return 0

    last_frame = state.get("last_evidence_scan_frame_id", 0)
    cutoff = (datetime.now() - timedelta(hours=DEFAULT_EVIDENCE_LOOKBACK_HOURS)).isoformat()

    closed_count = 0
    max_frame_id = last_frame

    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(SCREENPIPE_EVIDENCE_QUERY, (last_frame, cutoff)).fetchall()
        conn.close()
    except Exception as exc:
        logger.error("Failed to query Screenpipe DB: %s", exc)
        return 0

    for row in rows:
        frame_id = row["frame_id"]
        if frame_id > max_frame_id:
            max_frame_id = frame_id

        browser_url = row["browser_url"] or ""
        ocr_text = (row["text"] or "").lower()

        for aid, item in list(pending.items()):
            sys_rules = systems.get(item["system"], {})
            evidence_cfg = sys_rules.get("evidence", {})
            domains = evidence_cfg.get("browser_domains", [])
            success_texts = evidence_cfg.get("success_text", [])

            if not _url_matches_domains(browser_url, domains):
                continue

            text_match = any(st.lower() in ocr_text for st in success_texts)
            if not text_match:
                continue

            ext_id = item.get("external_id")
            if ext_id and ext_id.lower() not in ocr_text:
                continue

            now_str = datetime.now().isoformat(timespec="seconds")
            item["status"] = "approved"
            item["status_changed_at"] = now_str
            item["evidence"] = {
                "timestamp": row["timestamp"],
                "url": _sanitize_url(browser_url),
                "snippet": ocr_text[:200],
            }
            if browser_url:
                item["link"] = _sanitize_url(browser_url)

            state["items"][aid] = item
            del pending[aid]
            closed_count += 1
            logger.info("Auto-closed: %s (evidence at %s)", aid, row["timestamp"])

    state["last_evidence_scan_frame_id"] = max_frame_id
    return closed_count


# ── Report generation ────────────────────────────────────────────────────────


def _age_days(received_at: str) -> int:
    try:
        dt = datetime.fromisoformat(received_at)
        return (datetime.now() - dt).days
    except (ValueError, TypeError):
        return 0


def generate_report(state: dict[str, Any], vault: Path, cfg: dict[str, Any]) -> Path | None:
    """Write daily JSON + Markdown reports to the vault."""
    reports_dir_name = cfg.get("agents", {}).get("reports_dir", "90_reports")
    date_str = datetime.now().strftime("%Y-%m-%d")

    items = state.get("items", {})
    pending = [
        v for v in items.values() if v["status"] == "pending"
    ]
    pending.sort(key=lambda x: x.get("received_at", ""))

    today_str = datetime.now().strftime("%Y-%m-%d")
    approved_today = [
        v for v in items.values()
        if v["status"] == "approved"
        and (v.get("status_changed_at") or "")[:10] == today_str
    ]

    week_ago = (datetime.now() - timedelta(days=7)).isoformat()
    recently_closed = [
        v for v in items.values()
        if v["status"] in ("approved", "dismissed")
        and (v.get("status_changed_at") or "") >= week_ago
    ]
    recently_closed.sort(key=lambda x: x.get("status_changed_at", ""), reverse=True)

    by_system: dict[str, int] = {}
    for item in pending:
        sys = item.get("display_name") or item.get("system", "unknown")
        by_system[sys] = by_system.get(sys, 0) + 1

    oldest_days = max((_age_days(p.get("received_at", "")) for p in pending), default=0)
    overdue = [p for p in pending if _age_days(p.get("received_at", "")) > 3]

    report_data: dict[str, Any] = {
        "date": date_str,
        "pending_count": len(pending),
        "approved_today_count": len(approved_today),
        "overdue_count": len(overdue),
        "pending": pending,
        "approved_today": approved_today,
        "recently_closed": recently_closed,
        "by_system": by_system,
        "analysis": {
            "oldest_pending_days": oldest_days,
            "busiest_system": max(by_system, key=by_system.get, default="none") if by_system else "none",
        },
    }

    report_dir = vault / reports_dir_name / "approvals-queue"
    report_dir.mkdir(parents=True, exist_ok=True)

    json_path = report_dir / f"{date_str}.json"
    tmp = json_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(report_data, indent=2, default=str), encoding="utf-8")
    tmp.rename(json_path)

    md_path = report_dir / f"{date_str}.md"
    md_content = _render_markdown_report(report_data)
    md_path.write_text(md_content, encoding="utf-8")

    logger.info("Report written: %s (%d pending, %d closed today)", json_path, len(pending), len(approved_today))
    return json_path


def _render_markdown_report(data: dict[str, Any]) -> str:
    meta = {
        "date": data["date"],
        "source": "approvals-monitor",
        "type": "approvals-queue",
        "pending_count": data["pending_count"],
    }
    lines = [yaml_frontmatter(meta), ""]
    lines.append(f"# Approvals Queue -- {data['date']}")
    lines.append("")
    lines.append(f"**Pending:** {data['pending_count']} | "
                 f"**Approved today:** {data['approved_today_count']} | "
                 f"**Overdue (>3d):** {data['overdue_count']}")
    lines.append("")

    if data["pending"]:
        lines.append("## Pending Approvals")
        lines.append("")
        lines.append("| System | Subject | From | Received | Age |")
        lines.append("|--------|---------|------|----------|-----|")
        for item in data["pending"]:
            age = _age_days(item.get("received_at", ""))
            age_str = f"{age}d" if age > 0 else "today"
            system = item.get("display_name") or item.get("system", "")
            subject = item.get("subject", "")[:60]
            sender = item.get("from", "")[:30]
            received = (item.get("received_at") or "")[:10]
            lines.append(f"| {system} | {subject} | {sender} | {received} | {age_str} |")
        lines.append("")

    if data["approved_today"]:
        lines.append("## Approved Today")
        lines.append("")
        for item in data["approved_today"]:
            lines.append(f"- **{item.get('subject', '')}** ({item.get('display_name', '')})")
        lines.append("")

    if data["recently_closed"]:
        lines.append("## Recently Closed (7 days)")
        lines.append("")
        for item in data["recently_closed"]:
            status = item.get("status", "")
            changed = (item.get("status_changed_at") or "")[:10]
            lines.append(f"- [{status}] **{item.get('subject', '')}** -- {changed}")
        lines.append("")

    return "\n".join(lines)


# ── Pruning ──────────────────────────────────────────────────────────────────


def prune_old_items(state: dict[str, Any]) -> int:
    """Remove terminal items older than PRUNE_CLOSED_AFTER_DAYS."""
    cutoff = (datetime.now() - timedelta(days=PRUNE_CLOSED_AFTER_DAYS)).isoformat()
    to_remove = []
    for aid, item in state.get("items", {}).items():
        if item["status"] in ("approved", "dismissed"):
            changed = item.get("status_changed_at") or item.get("received_at", "")
            if changed and changed < cutoff:
                to_remove.append(aid)
    for aid in to_remove:
        del state["items"][aid]
    if to_remove:
        logger.info("Pruned %d old items", len(to_remove))
    return len(to_remove)


# ── Main ─────────────────────────────────────────────────────────────────────


def run(*, dry_run: bool = False) -> dict[str, Any]:
    """Execute one cycle of the approvals monitor."""
    cfg = load_config(CONFIG_PATH)
    vault = Path(cfg.get("obsidian_vault", "")).expanduser()

    systems = _load_system_rules()
    state = load_approvals_state()

    new_emails = scan_emails(systems, state, cfg)
    logger.info("Email scan: %d new approval requests", new_emails)

    closed = scan_screenpipe_evidence(systems, state, cfg)
    logger.info("Evidence scan: %d auto-closed", closed)

    pruned = prune_old_items(state)

    if dry_run:
        pending = [v for v in state["items"].values() if v["status"] == "pending"]
        print(f"DRY RUN: {len(pending)} pending, {new_emails} new, {closed} closed, {pruned} pruned")
        for item in pending:
            age = _age_days(item.get("received_at", ""))
            print(f"  [{item['system']}] {item['subject'][:60]} ({age}d)")
        return {"ok": True, "dry_run": True}

    save_approvals_state(state)
    report_path = generate_report(state, vault, cfg)

    return {
        "ok": True,
        "new_emails": new_emails,
        "auto_closed": closed,
        "pruned": pruned,
        "report_path": str(report_path) if report_path else None,
    }


def main() -> None:
    cfg = load_config(CONFIG_PATH)
    setup_logging(cfg)

    parser = argparse.ArgumentParser(description="MemoryOS Approvals Queue Monitor")
    parser.add_argument("--dry-run", action="store_true", help="Scan but don't persist")
    args = parser.parse_args()

    result = run(dry_run=args.dry_run)
    if not result.get("ok"):
        sys.exit(1)


if __name__ == "__main__":
    main()
