"""MemoryOS Agent Loop v2 -- Cursor-quality agentic chat.

Three-phase turn architecture:
  1. Auto-RAG pre-load  -- gather rich context + keyword search (no LLM)
  2. Tool loop           -- gpt-5-mini drives tool calls for additional data
  3. Synthesis           -- gpt-5.2 produces thorough final response

Usage (CLI test)::

    python -m src.agents.agent_loop "What meetings do I have today?"
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.agents import llm_provider
from src.agents.tools import TOOL_SCHEMAS, execute_tool, _sanitize_fts5_query
from src.common.config import load_config

logger = logging.getLogger("memoryos.agents.loop")

REPO_DIR = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = REPO_DIR / "config" / "config.yaml"

MAX_TOOL_ITERATIONS = 5
MAX_HISTORY_MESSAGES = 40

TOOL_MODEL = "gpt-4o-mini"
SYNTHESIS_MODEL = "gpt-5.2"

TOOL_CONTEXT_FILES = (
    "today.md",
    "upcoming.md",
    "priorities.md",
)

SYNTHESIS_CONTEXT_FILES = (
    "today.md",
    "upcoming.md",
    "recent_emails.md",
    "priorities.md",
    "this_week.md",
    "core.md",
)

SYSTEM_PROMPT_TEMPLATE = """\
You are MemoryOS, a Jarvis-like personal AI assistant for a senior technology executive.
You have deep access to the user's work life: emails, calendar, meeting transcripts,
screen activity, Teams chats, documents, and a curated knowledge base -- all indexed
and searchable.

Today is {date}. Current time: {time}.

## Your capabilities
- Search and read the memory vault (emails, meetings, transcripts, activity, documents)
- Run skills (morning-brief, meeting-prep, weekly-status, news-pulse, plan-my-week, etc.)
- Search the web for current information
- Send emails via Mail.app or SMTP
- Write notes, action items, and documents to the vault
- Execute shell commands (read-only, scoped)

## How to respond
- Give thorough, executive-quality answers. Don't be brief unless asked to be.
- When reporting on meetings, emails, or activity: include names, times, key points, and action items.
- When multiple sources are relevant, cross-reference them (e.g., email thread + meeting transcript + Teams chat).
- Cite source file paths so the user can find the original.
- For multi-step tasks, chain tool calls without asking for permission at each step.
- Use Markdown formatting: headers, bullet points, bold for names/emphasis, code blocks where appropriate.
- NEVER use Markdown tables. Always use headers and bullet lists for structured data.
- For meeting summaries: always read the full calendar and audio files using vault_read to get complete data.
- For email queries: read at least the most important/recent emails in full, don't rely only on search snippets.

## Tool usage
- If the pre-loaded context already contains the answer, respond with NO tool calls. The system will use a stronger model for synthesis.
- Only call tools if the pre-loaded context is insufficient (e.g., need a specific file, a search for a person/topic not in context, or to execute an action).
- When you do use tools, be focused: 1-3 calls for most questions, up to 5 for complex multi-source queries.
- For questions about people, check emails, meetings, AND Teams -- don't stop at one source.
- If a search returns no results or an error, move on. Don't retry with similar terms.
- NEVER call context_summary as a tool -- the pre-loaded context already contains all context files.

## Pre-loaded context
The following context was gathered automatically before you received this message.
Use it as your primary knowledge base. Only call tools if you need additional detail.

{context}
"""

SYNTHESIS_PROMPT = """\
You are MemoryOS, a Jarvis-like personal AI assistant for a senior technology executive.
The tool-calling phase has gathered the data below. Now produce your final response.

Today is {date}. Current time: {time}.

## Response Quality Standards
- Executive-quality: insightful, actionable, comprehensive
- Use Markdown formatting: headers (##), bullet points, **bold** for names and emphasis
- NEVER use Markdown tables — they render poorly in the chat UI. Use headers and bullet lists instead.
- Include source file paths as inline references so the user can find originals

## Meeting Summary Format
When summarizing meetings, use this structure for EACH meeting:

### [Meeting Title] — [Time]
**Attendees:** [list of attendees if known]

**Summary:** [3-5 sentence executive summary of what was discussed]

**Key Discussion Points:**
- [Point 1]
- [Point 2]

**Decisions Made:**
- [Decision 1]

**Action Items:**
- [Owner] — [Action] — [Due date if mentioned]

## Email Overview Format
When summarizing emails, group by thread/topic:

### [Thread/Topic Name]
- **From:** [sender] | **Time:** [time]
- **Summary:** [1-2 sentences]
- **Action Required:** [Yes/No + what]

## General Guidelines
- Cross-reference sources when possible (e.g., connect email discussions to meeting decisions)
- Highlight urgent or time-sensitive items
- If data is missing or incomplete, say so explicitly rather than guessing
- Be thorough — don't summarize away important details
"""


def _load_env() -> None:
    env_file = REPO_DIR / ".env.local"
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())


def _load_context_files(
    config: dict[str, Any],
    file_list: tuple[str, ...],
) -> str:
    """Load the specified context files from the vault."""
    vault = Path(os.path.expanduser(config.get("obsidian_vault", "")))
    context_dir = vault / config.get("memory", {}).get("context_dir", "_context")

    parts: list[str] = []
    for name in file_list:
        f = context_dir / name
        if f.is_file():
            content = f.read_text(encoding="utf-8", errors="replace")
            parts.append(f"### {name}\n\n{content}")

    return "\n\n---\n\n".join(parts) if parts else "(No context files available)"


def _extract_keywords(text: str) -> list[str]:
    """Extract meaningful search keywords from a user query."""
    stop_words = {
        "i", "me", "my", "we", "our", "you", "your", "the", "a", "an", "is", "are",
        "was", "were", "be", "been", "being", "have", "has", "had", "do", "does",
        "did", "will", "would", "could", "should", "can", "may", "might", "shall",
        "to", "of", "in", "for", "on", "with", "at", "by", "from", "as", "into",
        "through", "about", "between", "after", "before", "during", "without",
        "what", "who", "when", "where", "which", "how", "why", "that", "this",
        "these", "those", "it", "its", "all", "any", "both", "each", "every",
        "some", "such", "no", "not", "only", "very", "just", "also", "than",
        "too", "so", "if", "or", "and", "but", "nor", "yet", "up", "out",
        "today", "yesterday", "tomorrow", "week", "month", "tell", "show",
        "give", "find", "get", "run", "make", "let", "know", "need", "want",
        "list", "many", "much",
    }
    words = re.findall(r"[a-zA-Z]+", text.lower())
    keywords = [w for w in words if w not in stop_words and len(w) > 2]
    return keywords[:8]


def _auto_rag(query: str, config: dict[str, Any]) -> str:
    """Run automatic memory search on query keywords before LLM sees the message."""
    keywords = _extract_keywords(query)
    if not keywords:
        return ""

    search_query = _sanitize_fts5_query(" ".join(keywords))
    if not search_query:
        return ""

    try:
        from src.memory.index import MemoryIndex
        db_path = config.get("memory", {}).get("index_db", "config/memory.db")
        idx = MemoryIndex(REPO_DIR / db_path)
        try:
            results = idx.search(search_query, limit=15)
        finally:
            idx.close()
    except Exception as exc:
        logger.warning("Auto-RAG search failed: %s", exc)
        return ""

    if not results:
        return ""

    lines = [f"### Auto-retrieved vault results for: {search_query}\n"]
    for r in results:
        title = r.get("title", "Untitled")
        path = r.get("path", "")
        snippet = (r.get("content", ""))[:1500]
        source = r.get("source_type", "")
        lines.append(f"- **{title}** ({source}) -- `{path}`\n  {snippet}\n")

    return "\n".join(lines)


class AgentSession:
    """Holds conversation state and drives the three-phase agentic loop."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or load_config(CONFIG_PATH)
        self.history: list[dict[str, Any]] = []

    def _trim_history(self) -> None:
        while len(self.history) > MAX_HISTORY_MESSAGES:
            self.history.pop(0)

    async def run_turn(self, user_message: str) -> AsyncGenerator[dict[str, Any], None]:
        """Run one user turn through the three-phase loop.

        Phase 1: Auto-RAG pre-load (no LLM)
        Phase 2: Tool loop with gpt-5-mini
        Phase 3: Synthesis with gpt-5.2
        """
        self.history.append({"role": "user", "content": user_message})
        self._trim_history()

        # ── Phase 1: Auto-RAG pre-load ──
        now = datetime.now()
        tool_context = await asyncio.to_thread(
            _load_context_files, self.config, TOOL_CONTEXT_FILES,
        )
        rag_results = await asyncio.to_thread(_auto_rag, user_message, self.config)

        tool_prompt_context = tool_context
        if rag_results:
            tool_prompt_context += "\n\n---\n\n" + rag_results

        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            date=now.strftime("%A, %B %d, %Y"),
            time=now.strftime("%I:%M %p"),
            context=tool_prompt_context,
        )

        messages = [{"role": "system", "content": system_prompt}] + self.history

        # ── Phase 2: Tool loop (gpt-5-mini) ──
        gathered_data: list[str] = []
        tool_call_count = 0

        for iteration in range(MAX_TOOL_ITERATIONS):
            try:
                response = llm_provider.complete_with_tools(
                    messages=messages,
                    tools=TOOL_SCHEMAS,
                    model_override=TOOL_MODEL,
                )
            except Exception as exc:
                logger.exception("Tool-loop LLM call failed")
                yield {"type": "error", "content": f"LLM error: {exc}"}
                return

            message = response.choices[0].message
            tool_calls = getattr(message, "tool_calls", None)

            if not tool_calls:
                break

            assistant_msg = message.model_dump(exclude_none=True)
            self.history.append(assistant_msg)
            messages.append(assistant_msg)

            for tc in tool_calls:
                fn_name = tc.function.name
                try:
                    fn_args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    fn_args = {}

                yield {"type": "tool_call", "name": fn_name, "args": fn_args}
                tool_call_count += 1

                result = await asyncio.to_thread(
                    execute_tool, fn_name, fn_args, self.config,
                )

                if len(result) > 32000:
                    result = result[:32000] + "\n... [truncated]"

                yield {"type": "tool_result", "name": fn_name, "output": result}
                gathered_data.append(f"[{fn_name}]: {result}")

                tool_msg = {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                }
                self.history.append(tool_msg)
                messages.append(tool_msg)

        # ── Phase 3: Synthesis (gpt-5.2) ──
        synthesis_system = SYNTHESIS_PROMPT.format(
            date=now.strftime("%A, %B %d, %Y"),
            time=now.strftime("%I:%M %p"),
        )

        synthesis_parts: list[str] = []
        if tool_prompt_context:
            synthesis_parts.append(tool_prompt_context)
        if rag_results:
            synthesis_parts.append(rag_results)
        if gathered_data:
            synthesis_parts.append("### Tool results gathered\n\n" + "\n\n".join(gathered_data))

        if not gathered_data and not rag_results:
            rich_context = await asyncio.to_thread(
                _load_context_files, self.config, SYNTHESIS_CONTEXT_FILES,
            )
            synthesis_parts = [rich_context]

        synthesis_messages = [
            {"role": "system", "content": synthesis_system},
            {"role": "user", "content": (
                f"Original question: {user_message}\n\n"
                f"Context and data:\n\n{''.join(synthesis_parts)}"
            )},
        ]

        try:
            synth_response = llm_provider.complete_with_tools(
                messages=synthesis_messages,
                tools=[],
                model_override=SYNTHESIS_MODEL,
            )
            text = synth_response.choices[0].message.content or ""
        except Exception as exc:
            logger.exception("Synthesis LLM call failed")
            yield {"type": "error", "content": f"Synthesis error: {exc}"}
            return

        self.history.append({"role": "assistant", "content": text})
        yield {"type": "done", "content": text}


# ---------------------------------------------------------------------------
# CLI test harness
# ---------------------------------------------------------------------------

async def _cli_main(query: str) -> None:
    _load_env()
    config = load_config(CONFIG_PATH)
    session = AgentSession(config)

    async for event in session.run_turn(query):
        if event["type"] == "tool_call":
            print(f"\n🔧 Tool: {event['name']}({json.dumps(event['args'], indent=2)})")
        elif event["type"] == "tool_result":
            output = event["output"]
            if len(output) > 500:
                output = output[:500] + "..."
            print(f"   → {output}")
        elif event["type"] == "done":
            print(f"\n💬 {event['content']}")
        elif event["type"] == "error":
            print(f"\n❌ {event['content']}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MemoryOS Agent CLI")
    parser.add_argument("query", nargs="?", default="What meetings do I have today?")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    asyncio.run(_cli_main(args.query))
