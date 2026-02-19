"""SQLite FTS5-backed memory index for fast full-text search across the vault."""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger("memoryos.memory")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    path        TEXT UNIQUE NOT NULL,
    title       TEXT NOT NULL DEFAULT '',
    source_type TEXT NOT NULL DEFAULT 'unknown',
    content     TEXT NOT NULL DEFAULT '',
    created_at  TEXT NOT NULL,
    modified_at TEXT NOT NULL,
    tier        TEXT NOT NULL DEFAULT 'warm',
    mtime_ns    INTEGER NOT NULL DEFAULT 0
);

CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
    title,
    content,
    source_type,
    content='documents',
    content_rowid='id',
    tokenize='porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS documents_ai AFTER INSERT ON documents BEGIN
    INSERT INTO documents_fts(rowid, title, content, source_type)
    VALUES (new.id, new.title, new.content, new.source_type);
END;

CREATE TRIGGER IF NOT EXISTS documents_ad AFTER DELETE ON documents BEGIN
    INSERT INTO documents_fts(documents_fts, rowid, title, content, source_type)
    VALUES ('delete', old.id, old.title, old.content, old.source_type);
END;

CREATE TRIGGER IF NOT EXISTS documents_au AFTER UPDATE ON documents BEGIN
    INSERT INTO documents_fts(documents_fts, rowid, title, content, source_type)
    VALUES ('delete', old.id, old.title, old.content, old.source_type);
    INSERT INTO documents_fts(rowid, title, content, source_type)
    VALUES (new.id, new.title, new.content, new.source_type);
END;
"""


class MemoryIndex:
    """Persistent full-text index over vault Markdown files."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "MemoryIndex":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def upsert(
        self,
        path: str,
        title: str,
        source_type: str,
        content: str,
        created_at: datetime,
        modified_at: datetime,
        tier: str = "warm",
        mtime_ns: int = 0,
    ) -> None:
        """Insert or update a document in the index."""
        self._conn.execute(
            """
            INSERT INTO documents (path, title, source_type, content, created_at, modified_at, tier, mtime_ns)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
                title       = excluded.title,
                source_type = excluded.source_type,
                content     = excluded.content,
                created_at  = excluded.created_at,
                modified_at = excluded.modified_at,
                tier        = excluded.tier,
                mtime_ns    = excluded.mtime_ns
            """,
            (
                path,
                title,
                source_type,
                content,
                created_at.isoformat(),
                modified_at.isoformat(),
                tier,
                mtime_ns,
            ),
        )
        self._conn.commit()

    def remove(self, path: str) -> None:
        """Remove a document from the index."""
        self._conn.execute("DELETE FROM documents WHERE path = ?", (path,))
        self._conn.commit()

    def remove_missing(self, valid_paths: set[str]) -> int:
        """Remove index entries whose paths are no longer in *valid_paths*. Returns count removed."""
        cur = self._conn.execute("SELECT path FROM documents")
        stale = [row["path"] for row in cur if row["path"] not in valid_paths]
        if stale:
            self._conn.executemany("DELETE FROM documents WHERE path = ?", [(p,) for p in stale])
            self._conn.commit()
        return len(stale)

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        source_type: str | None = None,
        tier: str | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        limit: int = 25,
    ) -> list[dict[str, Any]]:
        """Full-text search with optional filters.

        Results are ranked by FTS5 relevance, with hot-tier documents boosted.
        """
        conditions = ["documents_fts MATCH ?"]
        params: list[Any] = [query]

        if source_type:
            conditions.append("d.source_type = ?")
            params.append(source_type)
        if tier:
            conditions.append("d.tier = ?")
            params.append(tier)
        if date_from:
            conditions.append("d.created_at >= ?")
            params.append(date_from.isoformat())
        if date_to:
            conditions.append("d.created_at <= ?")
            params.append(date_to.isoformat())

        where = " AND ".join(conditions)
        params.append(limit)

        sql = f"""
            SELECT d.*, rank,
                   CASE d.tier
                       WHEN 'hot'  THEN rank * 2.0
                       WHEN 'warm' THEN rank * 1.0
                       ELSE rank * 0.5
                   END AS boosted_rank
            FROM documents_fts
            JOIN documents d ON d.id = documents_fts.rowid
            WHERE {where}
            ORDER BY boosted_rank
            LIMIT ?
        """
        return [dict(row) for row in self._conn.execute(sql, params)]

    def get_by_date_range(
        self,
        date_from: datetime,
        date_to: datetime,
        *,
        source_type: str | None = None,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        """Retrieve documents within a date range, newest first."""
        conditions = ["created_at >= ?", "created_at <= ?"]
        params: list[Any] = [date_from.isoformat(), date_to.isoformat()]

        if source_type:
            conditions.append("source_type = ?")
            params.append(source_type)

        where = " AND ".join(conditions)
        params.append(limit)

        sql = f"""
            SELECT * FROM documents
            WHERE {where}
            ORDER BY created_at DESC
            LIMIT ?
        """
        return [dict(row) for row in self._conn.execute(sql, params)]

    def get_recent(
        self,
        hours: int = 24,
        *,
        source_type: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        """Shortcut: documents from the last N hours."""
        since = datetime.now() - timedelta(hours=hours)
        return self.get_by_date_range(since, datetime.now(), source_type=source_type, limit=limit)

    def get_by_path(self, path: str) -> dict[str, Any] | None:
        """Look up a single document by vault-relative path."""
        row = self._conn.execute("SELECT * FROM documents WHERE path = ?", (path,)).fetchone()
        return dict(row) if row else None

    def get_mtime(self, path: str) -> int | None:
        """Return stored mtime_ns for a path, or None if not indexed."""
        row = self._conn.execute("SELECT mtime_ns FROM documents WHERE path = ?", (path,)).fetchone()
        return row["mtime_ns"] if row else None

    def count(self, *, source_type: str | None = None, tier: str | None = None) -> int:
        """Count indexed documents, with optional filters."""
        conditions: list[str] = []
        params: list[Any] = []
        if source_type:
            conditions.append("source_type = ?")
            params.append(source_type)
        if tier:
            conditions.append("tier = ?")
            params.append(tier)
        where = (" WHERE " + " AND ".join(conditions)) if conditions else ""
        row = self._conn.execute(f"SELECT COUNT(*) AS n FROM documents{where}", params).fetchone()
        return row["n"]

    def stats(self) -> dict[str, Any]:
        """Summary statistics for the index."""
        total = self.count()
        by_type = {}
        for row in self._conn.execute(
            "SELECT source_type, COUNT(*) AS n FROM documents GROUP BY source_type"
        ):
            by_type[row["source_type"]] = row["n"]
        by_tier = {}
        for row in self._conn.execute(
            "SELECT tier, COUNT(*) AS n FROM documents GROUP BY tier"
        ):
            by_tier[row["tier"]] = row["n"]
        return {"total": total, "by_type": by_type, "by_tier": by_tier}
