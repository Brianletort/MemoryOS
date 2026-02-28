"""Shared fixtures for chat API PAR loop tests."""

from __future__ import annotations

import base64
import io
import json
import os
import struct
import tempfile
import zlib
from pathlib import Path
from typing import Any, Callable
from unittest.mock import MagicMock, patch

import httpx
import pytest
import pytest_asyncio

os.environ.setdefault("OPENAI_API_KEY", "sk-test-fake-key")

REPO_DIR = Path(__file__).resolve().parent.parent


def _make_mock_litellm_response(content: str = "Test response", tool_calls: list | None = None):
    """Build a mock LiteLLM ModelResponse."""
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls
    msg.model_dump.return_value = {"role": "assistant", "content": content}

    choice = MagicMock()
    choice.message = msg

    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _make_mock_stream_response(chunks: list[str] | None = None):
    """Build a mock streaming response (iterable of chunks)."""
    if chunks is None:
        chunks = ["Hello ", "world, ", "this is ", "a test."]

    mock_chunks = []
    for text in chunks:
        delta = MagicMock()
        delta.content = text
        choice = MagicMock()
        choice.delta = delta
        chunk = MagicMock()
        chunk.choices = [choice]
        mock_chunks.append(chunk)
    return mock_chunks


def _mock_litellm_completion(**kwargs):
    """Side effect for litellm.completion that handles both streaming and non-streaming."""
    if kwargs.get("stream"):
        return iter(_make_mock_stream_response())
    return _make_mock_litellm_response()


@pytest.fixture()
def mock_llm():
    """Patch litellm.completion to return deterministic responses."""
    with patch("litellm.completion", side_effect=_mock_litellm_completion) as m:
        yield m


@pytest_asyncio.fixture()
async def client(mock_llm, tmp_path):
    """Async httpx client bound to the FastAPI app with mocked LLM and temp DB."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    vault_dir = tmp_path / "vault"
    vault_dir.mkdir()
    context_dir = vault_dir / "_context"
    context_dir.mkdir()
    (context_dir / "today.md").write_text("# Today\nNo data.")
    (context_dir / "upcoming.md").write_text("# Upcoming\nNo data.")
    (context_dir / "priorities.md").write_text("# Priorities\nNo data.")

    config_content = f"""
obsidian_vault: {vault_dir}
screenpipe:
  db_path: {tmp_path}/fake_screenpipe.sqlite
outlook:
  db_path: {tmp_path}/fake_outlook.sqlite
  messages_dir: {tmp_path}/messages
  events_dir: {tmp_path}/events
onedrive:
  sync_dir: {tmp_path}/onedrive
state_file: {tmp_path}/state.json
log_dir: {tmp_path}/logs
memory:
  index_db: {tmp_path}/memory.db
  context_dir: _context
agents:
  provider: openai
  model: gpt-5.2
  reasoning_effort: none
  temperature: 0.3
  skills_dir: {tmp_path}/skills
  reports_dir: 90_reports
"""
    config_file = config_dir / "config.yaml"
    config_file.write_text(config_content)

    with patch("src.chat.routes.CONFIG_PATH", config_file), \
         patch("src.agents.agent_loop.CONFIG_PATH", config_file), \
         patch("src.agents.llm_provider.CONFIG_PATH", config_file), \
         patch("src.common.config._validate"):

        from src.chat.routes import _agent_sessions, _session_last_active
        _agent_sessions.clear()
        _session_last_active.clear()

        from src.dashboard.app import app
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            yield c


def make_tiny_pdf() -> bytes:
    """Create a minimal valid PDF file."""
    return (
        b"%PDF-1.0\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
        b"3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R>>endobj\n"
        b"xref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n"
        b"0000000058 00000 n \n0000000115 00000 n \n"
        b"trailer<</Size 4/Root 1 0 R>>\nstartxref\n190\n%%EOF"
    )


def make_tiny_png() -> bytes:
    """Create a minimal valid 1x1 red PNG."""
    def _chunk(chunk_type: bytes, data: bytes) -> bytes:
        c = chunk_type + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = _chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
    raw = zlib.compress(b"\x00\xff\x00\x00")
    idat = _chunk(b"IDAT", raw)
    iend = _chunk(b"IEND", b"")
    return sig + ihdr + idat + iend


def make_tiny_csv() -> bytes:
    return b"name,age,city\nAlice,30,NYC\nBob,25,LA\n"


def make_tiny_txt() -> bytes:
    return b"Hello, this is a test text file.\nLine two.\n"


async def par_loop(
    act_fn: Callable,
    review_fn: Callable,
    max_retries: int = 3,
    label: str = "PAR",
) -> Any:
    """Plan-Act-Review loop with auto-retry.

    act_fn: async callable that returns a result
    review_fn: callable(result) -> (ok: bool, diagnosis: str)
    """
    last_diagnosis = ""
    for attempt in range(1, max_retries + 1):
        result = await act_fn()
        ok, diagnosis = review_fn(result)
        if ok:
            return result
        last_diagnosis = f"[{label} attempt {attempt}/{max_retries}] {diagnosis}"
    raise AssertionError(f"PAR loop failed: {last_diagnosis}")


def parse_sse_events(raw: str) -> list[dict[str, Any]]:
    """Parse SSE text into a list of {event, data} dicts."""
    events: list[dict[str, Any]] = []
    current_event = ""
    for line in raw.split("\n"):
        if line.startswith("event: "):
            current_event = line[7:].strip()
        elif line.startswith("data: "):
            try:
                data = json.loads(line[6:])
                events.append({"event": current_event, "data": data})
            except json.JSONDecodeError:
                pass
    return events
