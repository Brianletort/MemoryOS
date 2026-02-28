"""Chat session and message persistence -- SQLite + Obsidian Markdown export.

Stores chat sessions and messages in config/chat.db and mirrors each
session to the Obsidian vault as a browsable Markdown file.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger("memoryos.chat.store")

REPO_DIR = Path(__file__).resolve().parent.parent.parent

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS chat_sessions (
    id         TEXT PRIMARY KEY,
    title      TEXT NOT NULL DEFAULT 'New Chat',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    pinned     INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS chat_messages (
    id         TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    role       TEXT NOT NULL,
    content    TEXT NOT NULL DEFAULT '',
    tool_calls TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_session
    ON chat_messages(session_id, created_at);
"""

VAULT_CHAT_DIR = "95_chat"


class ChatStore:
    """SQLite-backed chat session and message store with Obsidian export."""

    def __init__(
        self,
        db_path: str | Path | None = None,
        vault_path: str | Path | None = None,
    ) -> None:
        self._db_path = Path(db_path) if db_path else REPO_DIR / "config" / "chat.db"
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA)
        self._vault = Path(os.path.expanduser(vault_path)) if vault_path else None

    def close(self) -> None:
        self._conn.close()

    # ------------------------------------------------------------------
    # Sessions
    # ------------------------------------------------------------------

    def create_session(self, title: str = "New Chat") -> dict[str, Any]:
        sid = uuid.uuid4().hex[:12]
        now = datetime.now().isoformat()
        self._conn.execute(
            "INSERT INTO chat_sessions (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (sid, title, now, now),
        )
        self._conn.commit()
        return {"id": sid, "title": title, "created_at": now, "updated_at": now, "pinned": 0}

    def list_sessions(self) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM chat_sessions ORDER BY pinned DESC, updated_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM chat_sessions WHERE id = ?", (session_id,)
        ).fetchone()
        return dict(row) if row else None

    def update_session(self, session_id: str, **fields: Any) -> dict[str, Any] | None:
        allowed = {"title", "pinned"}
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates:
            return self.get_session(session_id)
        updates["updated_at"] = datetime.now().isoformat()
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        vals = list(updates.values()) + [session_id]
        self._conn.execute(
            f"UPDATE chat_sessions SET {set_clause} WHERE id = ?", vals
        )
        self._conn.commit()
        return self.get_session(session_id)

    def delete_session(self, session_id: str) -> bool:
        cur = self._conn.execute("DELETE FROM chat_sessions WHERE id = ?", (session_id,))
        self._conn.commit()
        if self._vault:
            md_path = self._vault / VAULT_CHAT_DIR / f"{session_id}.md"
            md_path.unlink(missing_ok=True)
        return cur.rowcount > 0

    def _touch_session(self, session_id: str) -> None:
        self._conn.execute(
            "UPDATE chat_sessions SET updated_at = ? WHERE id = ?",
            (datetime.now().isoformat(), session_id),
        )

    # ------------------------------------------------------------------
    # Messages
    # ------------------------------------------------------------------

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        mid = uuid.uuid4().hex[:16]
        now = datetime.now().isoformat()
        tc_json = json.dumps(tool_calls) if tool_calls else None
        self._conn.execute(
            "INSERT INTO chat_messages (id, session_id, role, content, tool_calls, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (mid, session_id, role, content, tc_json, now),
        )
        self._touch_session(session_id)
        self._conn.commit()
        return {
            "id": mid,
            "session_id": session_id,
            "role": role,
            "content": content,
            "tool_calls": tool_calls,
            "created_at": now,
        }

    def get_messages(self, session_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM chat_messages WHERE session_id = ? ORDER BY created_at",
            (session_id,),
        ).fetchall()
        results = []
        for r in rows:
            d = dict(r)
            if d.get("tool_calls"):
                try:
                    d["tool_calls"] = json.loads(d["tool_calls"])
                except (json.JSONDecodeError, TypeError):
                    pass
            results.append(d)
        return results

    def get_history_for_agent(self, session_id: str) -> list[dict[str, Any]]:
        """Return messages formatted for the agent loop's history list."""
        messages = self.get_messages(session_id)
        history: list[dict[str, Any]] = []
        for m in messages:
            entry: dict[str, Any] = {"role": m["role"], "content": m["content"]}
            if m.get("tool_calls"):
                entry["tool_calls"] = m["tool_calls"]
            history.append(entry)
        return history

    # ------------------------------------------------------------------
    # Obsidian Markdown export
    # ------------------------------------------------------------------

    def export_to_vault(self, session_id: str) -> Path | None:
        """Write/overwrite the session as a Markdown file in the vault."""
        if not self._vault:
            return None

        session = self.get_session(session_id)
        if not session:
            return None

        messages = self.get_messages(session_id)
        if not messages:
            return None

        chat_dir = self._vault / VAULT_CHAT_DIR
        chat_dir.mkdir(parents=True, exist_ok=True)
        md_path = chat_dir / f"{session_id}.md"

        lines: list[str] = [
            "---",
            f"title: \"{session['title']}\"",
            f"created: {session['created_at']}",
            f"updated: {session['updated_at']}",
            f"session_id: {session_id}",
            "type: chat-session",
            "---",
            "",
            f"# {session['title']}",
            "",
        ]

        for msg in messages:
            role_label = {"user": "You", "assistant": "MemoryOS"}.get(msg["role"], msg["role"])
            ts = msg["created_at"][:16].replace("T", " ")
            lines.append(f"## {role_label} — {ts}")
            lines.append("")
            lines.append(msg["content"])
            lines.append("")

        md_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Exported session %s to %s", session_id, md_path)
        return md_path
