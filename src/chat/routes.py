"""FastAPI router for the chat API -- SSE streaming, sessions, files, PPTX."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

logger = logging.getLogger("memoryos.chat.routes")

REPO_DIR = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = REPO_DIR / "config" / "config.yaml"

router = APIRouter(prefix="/api", tags=["chat"])

_agent_sessions: dict[str, Any] = {}
_session_last_active: dict[str, float] = {}
_SESSION_IDLE_TIMEOUT = 600


def _load_env_local() -> None:
    env_file = REPO_DIR / ".env.local"
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())


def _cfg() -> dict[str, Any]:
    from src.common.config import load_config
    return load_config(CONFIG_PATH)


def _get_store():
    from src.chat.store import ChatStore
    cfg = _cfg()
    vault = cfg.get("obsidian_vault", "")
    return ChatStore(vault_path=vault)


def _get_or_create_agent(session_id: str, config: dict[str, Any]):
    """Get an existing AgentSession or create a new one, restoring history."""
    from src.agents.agent_loop import AgentSession

    if session_id in _agent_sessions:
        _session_last_active[session_id] = time.time()
        return _agent_sessions[session_id]

    _load_env_local()
    agent = AgentSession(config)

    store = _get_store()
    try:
        history = store.get_history_for_agent(session_id)
        if history:
            agent.restore_history(history)
    finally:
        store.close()

    _agent_sessions[session_id] = agent
    _session_last_active[session_id] = time.time()
    return agent


# ------------------------------------------------------------------
# Session CRUD
# ------------------------------------------------------------------

@router.get("/sessions")
async def list_sessions() -> JSONResponse:
    store = _get_store()
    try:
        sessions = store.list_sessions()
    finally:
        store.close()
    return JSONResponse(sessions)


@router.post("/sessions")
async def create_session(request: Request) -> JSONResponse:
    body = await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}
    title = body.get("title", "New Chat")
    store = _get_store()
    try:
        session = store.create_session(title=title)
    finally:
        store.close()
    return JSONResponse(session, status_code=201)


@router.get("/sessions/{session_id}")
async def get_session(session_id: str) -> JSONResponse:
    store = _get_store()
    try:
        session = store.get_session(session_id)
        if not session:
            raise HTTPException(404, "Session not found")
        messages = store.get_messages(session_id)
    finally:
        store.close()
    session["messages"] = messages
    return JSONResponse(session)


@router.put("/sessions/{session_id}")
async def update_session(session_id: str, request: Request) -> JSONResponse:
    body = await request.json()
    store = _get_store()
    try:
        session = store.update_session(session_id, **body)
        if not session:
            raise HTTPException(404, "Session not found")
    finally:
        store.close()
    return JSONResponse(session)


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str) -> JSONResponse:
    store = _get_store()
    try:
        deleted = store.delete_session(session_id)
    finally:
        store.close()
    _agent_sessions.pop(session_id, None)
    _session_last_active.pop(session_id, None)
    if not deleted:
        raise HTTPException(404, "Session not found")
    return JSONResponse({"ok": True})


# ------------------------------------------------------------------
# SSE Chat endpoint
# ------------------------------------------------------------------

_MODEL_MAP: dict[str, tuple[str, str | None]] = {
    "default": ("gpt-5.2", "none"),
    "thinking": ("gpt-5.2", "high"),
    "pro": ("gpt-5.2-pro", "none"),
}


@router.post("/chat")
async def chat_sse(request: Request) -> StreamingResponse:
    """SSE streaming chat endpoint.

    Request body::

        {
            "session_id": "abc123",
            "message": "What meetings do I have today?",
            "model": "default",
            "web_enabled": true,
            "modes": ["image", "pptx"],
            "attachments": [{"stored_name": "xyz_report.pdf", "filename": "report.pdf"}]
        }
    """
    body = await request.json()
    message = body.get("message", "").strip()
    session_id = body.get("session_id")
    attachments = body.get("attachments", [])
    model_key = body.get("model", "default")
    web_enabled = body.get("web_enabled", True)
    modes = body.get("modes", [])

    if not message:
        raise HTTPException(400, "message is required")

    model_name, reasoning_effort = _MODEL_MAP.get(model_key, _MODEL_MAP["default"])

    config = _cfg()
    store = _get_store()

    try:
        if not session_id:
            title = message[:60].strip() or "New Chat"
            session = store.create_session(title=title)
            session_id = session["id"]
        else:
            existing = store.get_session(session_id)
            if not existing:
                raise HTTPException(404, "Session not found")

        store.add_message(session_id, "user", message)
    finally:
        store.close()

    agent = _get_or_create_agent(session_id, config)

    if attachments:
        from src.chat.file_handler import get_upload_path, extract_text
        attachment_parts: list[str] = []
        for att in attachments:
            stored = att.get("stored_name", "")
            path = get_upload_path(stored)
            if path:
                text = extract_text(path)
                if len(text) > 15000:
                    text = text[:15000] + "\n... [truncated]"
                attachment_parts.append(f"**{att.get('filename', stored)}:**\n{text}")
        if attachment_parts:
            agent.set_attachment_context("\n\n".join(attachment_parts))

    async def event_stream():
        yield _sse_event("session", {"session_id": session_id})

        tool_calls_log: list[dict[str, Any]] = []
        final_content = ""

        try:
            async for event in agent.run_turn(
                message,
                model=model_name,
                reasoning_effort=reasoning_effort,
                web_enabled=web_enabled,
                modes=modes,
            ):
                etype = event.get("type", "")

                if etype == "tool_call":
                    tc_data = {"name": event["name"], "args": event.get("args", {})}
                    tool_calls_log.append(tc_data)
                    yield _sse_event("tool_call", tc_data)

                elif etype == "tool_result":
                    output = event.get("output", "")
                    if len(output) > 2000:
                        output = output[:2000] + "..."
                    yield _sse_event("tool_result", {
                        "name": event["name"],
                        "output": output,
                    })

                elif etype == "status":
                    yield _sse_event("status", {"text": event.get("text", "")})

                elif etype == "content":
                    chunk = event.get("text", "")
                    final_content += chunk
                    yield _sse_event("content", {"text": chunk})

                elif etype == "done":
                    final_content = event.get("content", final_content)

                elif etype == "error":
                    yield _sse_event("error", {"message": event.get("content", "Unknown error")})
                    return

        except Exception as exc:
            logger.exception("Chat stream error")
            yield _sse_event("error", {"message": str(exc)})
            return

        download_urls = _extract_download_urls(final_content)
        for url in download_urls:
            yield _sse_event("file_ready", {"url": url, "filename": url.split("/")[-1]})

        store2 = _get_store()
        try:
            store2.add_message(session_id, "assistant", final_content, tool_calls_log or None)
            store2.export_to_vault(session_id)
        finally:
            store2.close()

        yield _sse_event("done", {"session_id": session_id, "message_count": len(agent.history)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _sse_event(event_type: str, data: dict[str, Any]) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


def _chunk_text(text: str, size: int) -> list[str]:
    """Split text into chunks for streaming."""
    if len(text) <= size:
        return [text]
    chunks = []
    for i in range(0, len(text), size):
        chunks.append(text[i:i + size])
    return chunks


def _extract_download_urls(text: str) -> list[str]:
    """Find /api/files/download/ URLs in the response text."""
    import re
    return re.findall(r'/api/files/download/[^\s\)\"\']+', text)


# ------------------------------------------------------------------
# File upload & download
# ------------------------------------------------------------------

@router.post("/files/upload")
async def upload_file(file: UploadFile = File(...)) -> JSONResponse:
    if not file.filename:
        raise HTTPException(400, "No filename provided")

    content = await file.read()

    from src.chat.file_handler import save_upload
    try:
        result = save_upload(file.filename, content)
    except ValueError as exc:
        raise HTTPException(400, str(exc))

    return JSONResponse({
        "file_id": result["file_id"],
        "filename": result["filename"],
        "stored_name": result["stored_as"],
        "size": result["size"],
        "text_preview": result["text"][:500] if result["text"] else "",
        "text_length": result["text_length"],
    })


@router.get("/files/download/{filename}")
async def download_file(filename: str) -> FileResponse:
    from src.chat.file_handler import get_generated_path, get_upload_path

    path = get_generated_path(filename) or get_upload_path(filename)
    if not path:
        raise HTTPException(404, "File not found")

    return FileResponse(
        path=str(path),
        filename=filename,
        media_type="application/octet-stream",
    )


# ------------------------------------------------------------------
# PPTX build endpoint
# ------------------------------------------------------------------

@router.post("/pptx/build")
async def build_pptx(request: Request) -> JSONResponse:
    """Build a PPTX from a slide spec and return download info."""
    spec = await request.json()

    from src.chat.pptx_builder import build_slides
    result = await asyncio.to_thread(build_slides, spec)

    if not result.get("ok"):
        raise HTTPException(500, result.get("error", "PPTX build failed"))

    return JSONResponse({
        "ok": True,
        "filename": result["filename"],
        "download_url": f"/api/files/download/{result['filename']}",
        "slide_count": result["slide_count"],
    })


# ------------------------------------------------------------------
# Session cleanup
# ------------------------------------------------------------------

async def prune_idle_sessions() -> None:
    """Remove agent sessions idle longer than timeout."""
    now = time.time()
    stale = [
        sid for sid, ts in _session_last_active.items()
        if (now - ts) > _SESSION_IDLE_TIMEOUT
    ]
    for sid in stale:
        _agent_sessions.pop(sid, None)
        _session_last_active.pop(sid, None)
        logger.info("Pruned idle agent session: %s", sid)
