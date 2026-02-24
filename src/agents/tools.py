"""Tool definitions and execution for the MemoryOS agent loop.

Each tool is defined as an OpenAI function-calling schema and backed by
a thin wrapper around an existing MemoryOS module (MemoryIndex, skill_runner,
emailer, web_search).
"""

from __future__ import annotations

import logging
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger("memoryos.agents.tools")

REPO_DIR = Path(__file__).resolve().parent.parent.parent

SHELL_ALLOWLIST = frozenset({
    "ls", "cat", "grep", "wc", "head", "tail", "date", "pwd", "du", "df",
    "git", "find", "sort", "uniq", "tr", "cut", "echo", "which", "file",
})

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "memory_search",
            "description": (
                "Full-text search across the entire memory vault (emails, meetings, "
                "transcripts, activity, documents). Use this to answer questions about "
                "what happened, who said what, or find specific content."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (FTS5 syntax: plain words, OR, phrases in quotes)",
                    },
                    "source_type": {
                        "type": "string",
                        "description": "Filter by source type: email, meeting, activity, teams, knowledge, slides",
                        "enum": ["email", "meeting", "activity", "teams", "knowledge", "slides"],
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results to return (default 25)",
                        "default": 25,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "vault_read",
            "description": (
                "Read the full contents of a specific file from the Obsidian vault. "
                "Use vault-relative paths like '10_meetings/2026/02/22/audio.md'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Vault-relative file path",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "vault_browse",
            "description": (
                "List files and subdirectories in a vault folder. Use to explore "
                "the vault structure before reading specific files."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "folder": {
                        "type": "string",
                        "description": "Vault-relative folder path (e.g. '10_meetings/2026/02')",
                    },
                },
                "required": ["folder"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "vault_recent",
            "description": (
                "Get recently created or modified documents from the vault. "
                "Useful for 'what happened today/this week' questions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "hours": {
                        "type": "integer",
                        "description": "Look back this many hours (default 24)",
                        "default": 24,
                    },
                    "source_type": {
                        "type": "string",
                        "description": "Filter by source type",
                        "enum": ["email", "meeting", "activity", "teams", "knowledge", "slides"],
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results (default 50)",
                        "default": 50,
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "context_summary",
            "description": (
                "Get today's auto-generated context summary including calendar, "
                "recent emails, upcoming events, and this week's activity. "
                "Call this first when answering questions about the current day or week."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_skill",
            "description": (
                "Execute a MemoryOS skill (morning-brief, meeting-prep, weekly-status, "
                "news-pulse, plan-my-week, etc.). Optionally email the result."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "skill_name": {
                        "type": "string",
                        "description": "Skill to run (directory name in skills folder)",
                    },
                    "send_email": {
                        "type": "boolean",
                        "description": "Email the report after generation (default false)",
                        "default": False,
                    },
                },
                "required": ["skill_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web for recent news or information using DuckDuckGo. "
                "Use for current events, competitor info, or anything not in the vault."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (plain text, will be split into keywords)",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Max results (default 5)",
                        "default": 5,
                    },
                    "time_range": {
                        "type": "string",
                        "description": "Time filter: 'd' (day), 'w' (week), 'm' (month)",
                        "enum": ["d", "w", "m"],
                        "default": "w",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": (
                "Send an email with the given subject and markdown body. Uses the "
                "configured email delivery (Mail.app or SMTP)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string", "description": "Email subject line"},
                    "body": {"type": "string", "description": "Email body in Markdown format"},
                    "to": {"type": "string", "description": "Recipient email (uses default if omitted)"},
                },
                "required": ["subject", "body"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "vault_write",
            "description": (
                "Create or update a file in the Obsidian vault. Use for saving notes, "
                "action items, drafts, or any content the user wants persisted."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Vault-relative file path to write (e.g. '00_inbox/2026/02/23/notes.md')",
                    },
                    "content": {
                        "type": "string",
                        "description": "Markdown content to write",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "meeting_overview",
            "description": (
                "Get a complete overview of all meetings for a specific date, "
                "including calendar events, attendees, and audio transcriptions. "
                "Returns structured data from all meeting files."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "Date in YYYY-MM-DD format. Defaults to today.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "shell_exec",
            "description": (
                "Run a shell command. Limited to safe, read-only commands: "
                "ls, cat, grep, wc, head, tail, date, pwd, git (read ops), find, sort, etc."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to execute",
                    },
                },
                "required": ["command"],
            },
        },
    },
]


def _get_vault_path(config: dict[str, Any]) -> Path:
    vault = config.get("obsidian_vault", "")
    return Path(os.path.expanduser(vault))


def _safe_vault_path(vault_root: Path, relative_path: str) -> Path | None:
    """Resolve a vault-relative path and reject traversal attempts."""
    resolved = (vault_root / relative_path).resolve()
    if not str(resolved).startswith(str(vault_root.resolve())):
        return None
    return resolved


def _get_memory_index(config: dict[str, Any]):
    """Lazy-import and create a MemoryIndex instance."""
    from src.memory.index import MemoryIndex
    db_path = config.get("memory", {}).get("index_db", "config/memory.db")
    full_path = REPO_DIR / db_path
    return MemoryIndex(full_path)


def _ensure_env_loaded() -> None:
    """Ensure .env.local is loaded into os.environ."""
    env_file = REPO_DIR / ".env.local"
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())


def execute_tool(name: str, arguments: dict[str, Any], config: dict[str, Any]) -> str:
    """Dispatch and execute a tool call. Returns the result as a string."""
    _ensure_env_loaded()
    try:
        if name == "memory_search":
            return _tool_memory_search(arguments, config)
        elif name == "vault_read":
            return _tool_vault_read(arguments, config)
        elif name == "vault_browse":
            return _tool_vault_browse(arguments, config)
        elif name == "vault_recent":
            return _tool_vault_recent(arguments, config)
        elif name == "context_summary":
            return _tool_context_summary(config)
        elif name == "run_skill":
            return _tool_run_skill(arguments)
        elif name == "web_search":
            return _tool_web_search(arguments)
        elif name == "send_email":
            return _tool_send_email(arguments)
        elif name == "vault_write":
            return _tool_vault_write(arguments, config)
        elif name == "meeting_overview":
            return _tool_meeting_overview(arguments, config)
        elif name == "shell_exec":
            return _tool_shell_exec(arguments)
        else:
            return f"Error: unknown tool '{name}'"
    except Exception as exc:
        logger.exception("Tool %s failed", name)
        return f"Error executing {name}: {exc}"


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def _sanitize_fts5_query(query: str) -> str:
    """Strip characters that break SQLite FTS5 MATCH syntax."""
    import re
    query = re.sub(r'[(){}[\]:;!@#$%^&*+=<>/?\\|~`]', ' ', query)
    query = re.sub(r'"[^"]*$', '', query)
    query = ' '.join(query.split())
    return query.strip()


def _tool_memory_search(args: dict[str, Any], config: dict[str, Any]) -> str:
    query = _sanitize_fts5_query(args.get("query", ""))
    if not query:
        return "Error: query is required"
    idx = _get_memory_index(config)
    try:
        results = idx.search(
            query,
            source_type=args.get("source_type"),
            limit=args.get("limit", 25),
        )
    except Exception as exc:
        logger.warning("memory_search failed for '%s': %s", query, exc)
        return f"Search error for '{query}': {exc}. Try simpler keywords."
    finally:
        idx.close()

    if not results:
        return f"No results found for '{query}'"

    lines = [f"Found {len(results)} results for '{query}':\n"]
    for r in results:
        title = r.get("title", "Untitled")
        path = r.get("path", "")
        snippet = (r.get("content", ""))[:2000]
        source = r.get("source_type", "")
        created = r.get("created_at", "")
        lines.append(f"- **{title}** ({source}, {created}) -- `{path}`\n  {snippet}\n")
    return "\n".join(lines)


def _tool_vault_read(args: dict[str, Any], config: dict[str, Any]) -> str:
    vault = _get_vault_path(config)
    rel_path = args.get("path", "")
    if not rel_path:
        return "Error: path is required"

    full = _safe_vault_path(vault, rel_path)
    if full is None:
        return "Error: path traversal not allowed"
    if not full.is_file():
        return f"Error: file not found: {rel_path}"

    content = full.read_text(encoding="utf-8", errors="replace")
    if len(content) > 32000:
        content = content[:32000] + "\n\n... [truncated, file is very large]"
    return content


def _tool_vault_browse(args: dict[str, Any], config: dict[str, Any]) -> str:
    vault = _get_vault_path(config)
    folder = args.get("folder", "")
    target = _safe_vault_path(vault, folder) if folder else vault

    if target is None:
        return "Error: path traversal not allowed"
    if not target.is_dir():
        return f"Error: directory not found: {folder}"

    entries: list[str] = []
    for item in sorted(target.iterdir()):
        if item.name.startswith("."):
            continue
        suffix = "/" if item.is_dir() else f" ({item.stat().st_size} bytes)"
        entries.append(f"  {item.name}{suffix}")

    if not entries:
        return f"Directory '{folder}' is empty"
    return f"Contents of '{folder or '/'}':\n" + "\n".join(entries)


def _tool_vault_recent(args: dict[str, Any], config: dict[str, Any]) -> str:
    hours = args.get("hours", 24)
    idx = _get_memory_index(config)
    try:
        results = idx.get_recent(
            hours=hours,
            source_type=args.get("source_type"),
            limit=args.get("limit", 50),
        )
    finally:
        idx.close()

    if not results:
        return f"No documents found in the last {hours} hours"

    lines = [f"Found {len(results)} documents from the last {hours} hours:\n"]
    for r in results:
        title = r.get("title", "Untitled")
        path = r.get("path", "")
        source = r.get("source_type", "")
        created = r.get("created_at", "")
        lines.append(f"- **{title}** ({source}, {created}) -- `{path}`")
    return "\n".join(lines)


def _tool_context_summary(config: dict[str, Any]) -> str:
    vault = _get_vault_path(config)
    context_dir_name = config.get("memory", {}).get("context_dir", "_context")
    context_dir = vault / context_dir_name

    if not context_dir.is_dir():
        return "Error: context directory not found"

    parts: list[str] = []
    for name in ("today.md", "upcoming.md", "recent_emails.md", "this_week.md", "priorities.md"):
        f = context_dir / name
        if f.is_file():
            content = f.read_text(encoding="utf-8", errors="replace")
            if len(content) > 8000:
                content = content[:8000] + "\n... [truncated]"
            parts.append(f"## {name}\n\n{content}")

    return "\n\n---\n\n".join(parts) if parts else "No context files found"


def _tool_run_skill(args: dict[str, Any]) -> str:
    skill_name = args.get("skill_name", "")
    if not skill_name:
        return "Error: skill_name is required"

    from src.agents.skill_runner import run_skill
    result = run_skill(
        skill_name,
        send_email=args.get("send_email", False),
    )

    if result.get("ok"):
        report_path = result.get("report_path", "unknown")
        msg = f"Skill '{skill_name}' completed successfully. Report saved to: {report_path}"
        if result.get("email_result", {}).get("ok"):
            msg += "\nEmail sent successfully."
        return msg
    else:
        return f"Skill '{skill_name}' failed: {result.get('error', 'unknown error')}"


def _tool_web_search(args: dict[str, Any]) -> str:
    query = args.get("query", "")
    if not query:
        return "Error: query is required"

    from src.agents.web_search import search_news
    results = search_news(
        topic=query,
        keywords=query.split(),
        max_results=args.get("max_results", 5),
        time_range=args.get("time_range", "w"),
    )

    if not results:
        return f"No web results found for '{query}'"

    lines = [f"Web search results for '{query}':\n"]
    for r in results:
        title = r.get("title", "")
        url = r.get("url", "")
        snippet = r.get("snippet", "")
        source = r.get("source", "")
        lines.append(f"- **{title}** ({source})\n  {url}\n  {snippet}\n")
    return "\n".join(lines)


def _tool_send_email(args: dict[str, Any]) -> str:
    subject = args.get("subject", "")
    body = args.get("body", "")
    if not subject or not body:
        return "Error: subject and body are required"

    from src.agents.emailer import send_report
    result = send_report(
        subject=subject,
        markdown_body=body,
        to=args.get("to"),
    )

    if result.get("ok"):
        return f"Email sent: {result.get('detail', 'success')}"
    else:
        return f"Email failed: {result.get('detail', 'unknown error')}"


def _tool_vault_write(args: dict[str, Any], config: dict[str, Any]) -> str:
    vault = _get_vault_path(config)
    rel_path = args.get("path", "")
    content = args.get("content", "")
    if not rel_path or not content:
        return "Error: path and content are required"

    full = _safe_vault_path(vault, rel_path)
    if full is None:
        return "Error: path traversal not allowed"

    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(content, encoding="utf-8")
    logger.info("Wrote vault file: %s (%d chars)", rel_path, len(content))
    return f"File written: {rel_path} ({len(content)} chars)"


def _tool_meeting_overview(args: dict[str, Any], config: dict[str, Any]) -> str:
    """Read all meeting files for a date and return a structured overview."""
    vault = _get_vault_path(config)
    date_str = args.get("date") or datetime.now().strftime("%Y-%m-%d")

    try:
        year, month, day = date_str.split("-")
    except ValueError:
        return f"Error: invalid date format '{date_str}', expected YYYY-MM-DD"

    meeting_dir = vault / "10_meetings" / year / month / day
    if not meeting_dir.is_dir():
        return f"No meeting directory found for {date_str} (looked in 10_meetings/{year}/{month}/{day})"

    parts: list[str] = [f"# Meeting Overview for {date_str}\n"]
    total_chars = 0
    max_total = 32000

    for md_file in sorted(meeting_dir.glob("*.md")):
        if total_chars >= max_total:
            parts.append("\n... [truncated — file limit reached]")
            break

        try:
            content = md_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        name_lower = md_file.stem.lower()
        if "calendar" in name_lower:
            header = f"## Calendar Events — `{md_file.name}`"
        elif "audio" in name_lower:
            header = f"## Meeting Transcripts — `{md_file.name}`"
        else:
            header = f"## {md_file.stem} — `{md_file.name}`"

        remaining = max_total - total_chars
        if len(content) > remaining:
            content = content[:remaining] + "\n\n... [truncated]"

        parts.append(f"{header}\n\n{content}")
        total_chars += len(content)

    if len(parts) == 1:
        return f"Meeting directory exists for {date_str} but contains no .md files"

    return "\n\n---\n\n".join(parts)


def _tool_shell_exec(args: dict[str, Any]) -> str:
    command = args.get("command", "").strip()
    if not command:
        return "Error: command is required"

    first_word = command.split()[0].split("/")[-1]
    if first_word not in SHELL_ALLOWLIST:
        return (
            f"Error: command '{first_word}' is not in the allowlist. "
            f"Allowed: {', '.join(sorted(SHELL_ALLOWLIST))}"
        )

    for dangerous in ("|", ">", ">>", ";", "&&", "||", "`", "$("):
        if dangerous in command:
            return f"Error: shell operator '{dangerous}' is not allowed for security reasons"

    try:
        result = subprocess.run(
            command.split(),
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(REPO_DIR),
        )
        output = result.stdout
        if result.stderr:
            output += f"\n[stderr]: {result.stderr}"
        if result.returncode != 0:
            output += f"\n[exit code: {result.returncode}]"
        if len(output) > 10000:
            output = output[:10000] + "\n... [truncated]"
        return output or "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: command timed out (30s limit)"
