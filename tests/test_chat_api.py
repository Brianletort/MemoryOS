"""PAR (Plan-Act-Review) loop tests for the MemoryOS Chat API.

Each test follows: Plan expected outcome -> Act (API call) -> Review (validate).
Failed reviews auto-retry up to 3 times with diagnostics.
"""

from __future__ import annotations

import json
from unittest.mock import patch, MagicMock

import httpx
import pytest

from tests.conftest import (
    make_tiny_csv,
    make_tiny_pdf,
    make_tiny_png,
    make_tiny_txt,
    par_loop,
    parse_sse_events,
)

# ---------------------------------------------------------------------------
# 1. Session CRUD
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestSessionCRUD:
    """PAR loops for session create, list, get, update, pin, delete."""

    async def test_create_session(self, client: httpx.AsyncClient) -> None:
        async def act():
            return await client.post("/api/sessions", json={"title": "Test Session"})

        def review(r: httpx.Response):
            if r.status_code != 201:
                return False, f"Expected 201, got {r.status_code}: {r.text}"
            data = r.json()
            if "id" not in data:
                return False, f"Missing 'id' in response: {data}"
            if data.get("title") != "Test Session":
                return False, f"Title mismatch: {data.get('title')}"
            return True, ""

        await par_loop(act, review, label="create_session")

    async def test_list_sessions(self, client: httpx.AsyncClient) -> None:
        await client.post("/api/sessions", json={"title": "Session A"})
        await client.post("/api/sessions", json={"title": "Session B"})

        async def act():
            return await client.get("/api/sessions")

        def review(r: httpx.Response):
            if r.status_code != 200:
                return False, f"Expected 200, got {r.status_code}"
            data = r.json()
            if not isinstance(data, list) or len(data) < 2:
                return False, f"Expected list with >=2 sessions, got {len(data) if isinstance(data, list) else type(data)}"
            return True, ""

        await par_loop(act, review, label="list_sessions")

    async def test_get_session_with_messages(self, client: httpx.AsyncClient) -> None:
        create_resp = await client.post("/api/sessions", json={"title": "Msg Test"})
        sid = create_resp.json()["id"]

        async def act():
            return await client.get(f"/api/sessions/{sid}")

        def review(r: httpx.Response):
            if r.status_code != 200:
                return False, f"Expected 200, got {r.status_code}"
            data = r.json()
            if "messages" not in data:
                return False, f"Missing 'messages' key: {list(data.keys())}"
            return True, ""

        await par_loop(act, review, label="get_session")

    async def test_update_session_title(self, client: httpx.AsyncClient) -> None:
        create_resp = await client.post("/api/sessions", json={"title": "Old Title"})
        sid = create_resp.json()["id"]

        async def act():
            return await client.put(f"/api/sessions/{sid}", json={"title": "New Title"})

        def review(r: httpx.Response):
            if r.status_code != 200:
                return False, f"Expected 200, got {r.status_code}"
            if r.json().get("title") != "New Title":
                return False, f"Title not updated: {r.json().get('title')}"
            return True, ""

        await par_loop(act, review, label="update_session")

    async def test_pin_unpin_session(self, client: httpx.AsyncClient) -> None:
        create_resp = await client.post("/api/sessions", json={"title": "Pin Test"})
        sid = create_resp.json()["id"]

        r = await client.put(f"/api/sessions/{sid}", json={"pinned": 1})
        assert r.json().get("pinned") == 1

        r = await client.put(f"/api/sessions/{sid}", json={"pinned": 0})
        assert r.json().get("pinned") == 0

    async def test_delete_session(self, client: httpx.AsyncClient) -> None:
        create_resp = await client.post("/api/sessions", json={"title": "Delete Me"})
        sid = create_resp.json()["id"]

        r = await client.delete(f"/api/sessions/{sid}")
        assert r.status_code == 200

        r = await client.get(f"/api/sessions/{sid}")
        assert r.status_code == 404

    async def test_jump_between_sessions(self, client: httpx.AsyncClient) -> None:
        """Create 3 sessions, verify each returns independently."""
        sids = []
        for i in range(3):
            r = await client.post("/api/sessions", json={"title": f"Session {i}"})
            sids.append(r.json()["id"])

        for i, sid in enumerate(sids):
            async def act(s=sid):
                return await client.get(f"/api/sessions/{s}")

            def review(r: httpx.Response, idx=i):
                if r.status_code != 200:
                    return False, f"Session {idx} returned {r.status_code}"
                data = r.json()
                if data.get("title") != f"Session {idx}":
                    return False, f"Session {idx} title mismatch: {data.get('title')}"
                return True, ""

            await par_loop(act, review, label=f"jump_session_{i}")


# ---------------------------------------------------------------------------
# 2. SSE Chat Streaming
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestChatStreaming:
    """PAR loops for chat message streaming with model/web/mode options."""

    async def test_basic_chat_creates_session(self, client: httpx.AsyncClient) -> None:
        async def act():
            return await client.post("/api/chat", json={
                "message": "Hello MemoryOS",
                "model": "default",
            })

        def review(r: httpx.Response):
            if r.status_code != 200:
                return False, f"Expected 200, got {r.status_code}: {r.text}"
            events = parse_sse_events(r.text)
            event_types = [e["event"] for e in events]
            if "session" not in event_types:
                return False, f"Missing 'session' event. Got: {event_types}"
            if "content" not in event_types:
                return False, f"Missing 'content' event. Got: {event_types}"
            if "done" not in event_types:
                return False, f"Missing 'done' event. Got: {event_types}"
            return True, ""

        await par_loop(act, review, label="basic_chat")

    async def test_chat_with_existing_session(self, client: httpx.AsyncClient) -> None:
        create_resp = await client.post("/api/sessions", json={"title": "Chat Test"})
        sid = create_resp.json()["id"]

        r = await client.post("/api/chat", json={
            "session_id": sid,
            "message": "What happened today?",
        })
        assert r.status_code == 200

        session_resp = await client.get(f"/api/sessions/{sid}")
        messages = session_resp.json().get("messages", [])
        roles = [m["role"] for m in messages]
        assert "user" in roles
        assert "assistant" in roles

    async def test_model_default(self, client: httpx.AsyncClient, mock_llm) -> None:
        r = await client.post("/api/chat", json={
            "message": "Test default model",
            "model": "default",
        })
        assert r.status_code == 200
        events = parse_sse_events(r.text)
        assert any(e["event"] == "content" for e in events)

    async def test_model_thinking(self, client: httpx.AsyncClient, mock_llm) -> None:
        r = await client.post("/api/chat", json={
            "message": "Test thinking model",
            "model": "thinking",
        })
        assert r.status_code == 200
        events = parse_sse_events(r.text)
        assert any(e["event"] == "done" for e in events)

    async def test_model_pro(self, client: httpx.AsyncClient, mock_llm) -> None:
        r = await client.post("/api/chat", json={
            "message": "Test pro model",
            "model": "pro",
        })
        assert r.status_code == 200
        events = parse_sse_events(r.text)
        assert any(e["event"] == "done" for e in events)

    async def test_web_disabled(self, client: httpx.AsyncClient, mock_llm) -> None:
        r = await client.post("/api/chat", json={
            "message": "Search for something",
            "web_enabled": False,
        })
        assert r.status_code == 200

    async def test_web_enabled(self, client: httpx.AsyncClient, mock_llm) -> None:
        r = await client.post("/api/chat", json={
            "message": "Search for something",
            "web_enabled": True,
        })
        assert r.status_code == 200

    async def test_image_mode(self, client: httpx.AsyncClient, mock_llm) -> None:
        r = await client.post("/api/chat", json={
            "message": "Create an image of a sunset",
            "modes": ["image"],
        })
        assert r.status_code == 200
        events = parse_sse_events(r.text)
        assert any(e["event"] == "done" for e in events)

    async def test_pptx_mode(self, client: httpx.AsyncClient, mock_llm) -> None:
        r = await client.post("/api/chat", json={
            "message": "Build a deck about AI strategy",
            "modes": ["pptx"],
        })
        assert r.status_code == 200
        events = parse_sse_events(r.text)
        assert any(e["event"] == "done" for e in events)

    async def test_empty_message_rejected(self, client: httpx.AsyncClient) -> None:
        r = await client.post("/api/chat", json={"message": ""})
        assert r.status_code == 400

    async def test_nonexistent_session_rejected(self, client: httpx.AsyncClient) -> None:
        r = await client.post("/api/chat", json={
            "session_id": "nonexistent-id",
            "message": "Hello",
        })
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# 3. File Upload
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestFileUpload:
    """PAR loops for file upload, text extraction, and download."""

    async def _upload(self, client: httpx.AsyncClient, filename: str, content: bytes) -> httpx.Response:
        return await client.post(
            "/api/files/upload",
            files={"file": (filename, content)},
        )

    async def test_upload_txt(self, client: httpx.AsyncClient) -> None:
        async def act():
            return await self._upload(client, "test.txt", make_tiny_txt())

        def review(r: httpx.Response):
            if r.status_code != 200:
                return False, f"Expected 200, got {r.status_code}: {r.text}"
            data = r.json()
            if not data.get("stored_name"):
                return False, f"Missing stored_name: {data}"
            if not data.get("text_preview"):
                return False, f"No text extracted: {data}"
            return True, ""

        await par_loop(act, review, label="upload_txt")

    async def test_upload_csv(self, client: httpx.AsyncClient) -> None:
        async def act():
            return await self._upload(client, "data.csv", make_tiny_csv())

        def review(r: httpx.Response):
            if r.status_code != 200:
                return False, f"Expected 200, got {r.status_code}"
            data = r.json()
            if "name" not in (data.get("text_preview") or ""):
                return False, f"CSV header not in preview: {data.get('text_preview', '')[:100]}"
            return True, ""

        await par_loop(act, review, label="upload_csv")

    async def test_upload_png(self, client: httpx.AsyncClient) -> None:
        async def act():
            return await self._upload(client, "photo.png", make_tiny_png())

        def review(r: httpx.Response):
            if r.status_code != 200:
                return False, f"Expected 200, got {r.status_code}"
            data = r.json()
            if not data.get("stored_name"):
                return False, "Missing stored_name"
            return True, ""

        await par_loop(act, review, label="upload_png")

    async def test_upload_unsupported_type(self, client: httpx.AsyncClient) -> None:
        r = await self._upload(client, "malware.exe", b"MZ\x00\x00")
        assert r.status_code == 400

    async def test_download_uploaded_file(self, client: httpx.AsyncClient) -> None:
        upload_resp = await self._upload(client, "hello.txt", b"Hello world")
        stored = upload_resp.json()["stored_name"]

        async def act():
            return await client.get(f"/api/files/download/{stored}")

        def review(r: httpx.Response):
            if r.status_code != 200:
                return False, f"Download returned {r.status_code}"
            if b"Hello world" not in r.content:
                return False, f"Content mismatch: {r.content[:50]}"
            return True, ""

        await par_loop(act, review, label="download_file")

    async def test_download_nonexistent(self, client: httpx.AsyncClient) -> None:
        r = await client.get("/api/files/download/nonexistent_file.txt")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# 4. PPTX Build
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestPPTXBuild:
    """PAR loops for PowerPoint generation via the API endpoint."""

    async def test_build_basic_deck(self, client: httpx.AsyncClient) -> None:
        mock_result = {"ok": True, "filename": "test_deck_abc123.pptx", "path": "/tmp/test.pptx", "slide_count": 2}

        with patch("src.chat.pptx_builder.build_slides", return_value=mock_result):
            spec = {
                "title": "Test Deck",
                "theme": "dark",
                "slides": [
                    {"type": "title", "title": "Welcome", "subtitle": "A test deck"},
                    {"type": "content", "title": "Slide 1", "bullets": ["Point A", "Point B"]},
                ],
            }

            async def act():
                return await client.post("/api/pptx/build", json=spec)

            def review(r: httpx.Response):
                if r.status_code != 200:
                    return False, f"Expected 200, got {r.status_code}: {r.text}"
                data = r.json()
                if not data.get("ok"):
                    return False, f"Build not ok: {data}"
                if data.get("slide_count") != 2:
                    return False, f"Expected 2 slides, got {data.get('slide_count')}"
                if not data.get("filename", "").endswith(".pptx"):
                    return False, f"Filename not .pptx: {data.get('filename')}"
                return True, ""

            await par_loop(act, review, label="build_pptx")

    async def test_build_all_slide_types(self, client: httpx.AsyncClient) -> None:
        mock_result = {"ok": True, "filename": "full_deck_xyz.pptx", "path": "/tmp/full.pptx", "slide_count": 6}

        with patch("src.chat.pptx_builder.build_slides", return_value=mock_result):
            spec = {
                "title": "Full Deck",
                "slides": [
                    {"type": "title", "title": "Title Slide", "subtitle": "Sub"},
                    {"type": "header", "title": "Section Header"},
                    {"type": "content", "title": "Content", "bullets": ["A", "B"]},
                    {"type": "metrics", "title": "Metrics", "metrics": [["$1M", "Revenue"], ["42%", "Growth"]]},
                    {"type": "comparison", "title": "Compare", "headers": ["Feature", "A", "B"], "rows": [["Speed", "Fast", "Slow"]]},
                    {"type": "takeaways", "title": "Takeaways", "items": ["Key point 1", "Key point 2"]},
                ],
            }

            r = await client.post("/api/pptx/build", json=spec)
            assert r.status_code == 200
            assert r.json()["slide_count"] == 6

    async def test_build_failure(self, client: httpx.AsyncClient) -> None:
        mock_result = {"ok": False, "error": "Missing title"}

        with patch("src.chat.pptx_builder.build_slides", return_value=mock_result):
            r = await client.post("/api/pptx/build", json={"slides": []})
            assert r.status_code == 500


# ---------------------------------------------------------------------------
# 5. Image Generation Tool (unit-level)
# ---------------------------------------------------------------------------

class TestImageGenTool:
    """PAR loops for the generate_image tool."""

    def test_generate_image_missing_prompt(self) -> None:
        from src.agents.tools import execute_tool
        result = execute_tool("generate_image", {"prompt": ""}, {})
        assert "Error" in result or "error" in result.lower()

    def test_generate_image_no_api_key(self) -> None:
        saved = os.environ.pop("OPENAI_API_KEY", None)
        try:
            with patch("src.agents.image_gen.generate_image", return_value=None):
                from src.agents.tools import execute_tool
                result = execute_tool("generate_image", {"prompt": "a sunset"}, {})
                assert "error" in result.lower() or "failed" in result.lower()
        finally:
            if saved:
                os.environ["OPENAI_API_KEY"] = saved

    def test_generate_image_success(self) -> None:
        fake_png = make_tiny_png()
        fake_b64 = __import__("base64").b64encode(fake_png).decode()

        mock_result = MagicMock()
        mock_result.data = [MagicMock(b64_json=fake_b64)]

        mock_client = MagicMock()
        mock_client.images.generate.return_value = mock_result

        with patch("src.agents.image_gen._get_client", return_value=mock_client):
            os.environ["OPENAI_API_KEY"] = "sk-test"
            from src.agents.tools import execute_tool
            result = execute_tool("generate_image", {"prompt": "a sunset over mountains"}, {})
            assert "/api/files/download/" in result
            assert "image" in result.lower() or "Generated" in result


# ---------------------------------------------------------------------------
# 6. Web Search Tool (unit-level)
# ---------------------------------------------------------------------------

class TestWebSearchTool:
    """PAR loops for the web_search tool."""

    def test_web_search_empty_query(self) -> None:
        from src.agents.tools import execute_tool
        result = execute_tool("web_search", {"query": ""}, {})
        assert "Error" in result or "error" in result.lower()

    def test_web_search_with_brave(self) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {"title": "Test Article", "url": "https://example.com", "description": "A test", "age": "1h", "meta_url": {"hostname": "example.com"}, "thumbnail": {}},
            ]
        }
        mock_response.raise_for_status = MagicMock()

        os.environ["BRAVE_SEARCH_API_KEY"] = "test-key"
        with patch("requests.get", return_value=mock_response):
            from src.agents.tools import execute_tool
            result = execute_tool("web_search", {"query": "test news"}, {})
            assert "Test Article" in result or "example.com" in result

    def test_web_search_ddg_fallback(self) -> None:
        saved_keys = {}
        for k in ("BRAVE_SEARCH_API_KEY", "BRAVE_API_KEY", "BRAVE_AI_API_KEY"):
            saved_keys[k] = os.environ.pop(k, None)

        try:
            mock_ddgs_instance = MagicMock()
            mock_ddgs_instance.__enter__ = MagicMock(return_value=mock_ddgs_instance)
            mock_ddgs_instance.__exit__ = MagicMock(return_value=False)
            mock_ddgs_instance.news.return_value = [
                {"title": "DDG Result", "url": "https://ddg.example.com", "body": "Fallback result", "date": "", "source": "DDG"},
            ]

            from src.agents import web_search as ws_mod
            with patch.object(ws_mod, "_brave_api_key", return_value=None), \
                 patch("duckduckgo_search.DDGS", return_value=mock_ddgs_instance):
                from src.agents.tools import execute_tool
                result = execute_tool("web_search", {"query": "test fallback"}, {})
                assert "DDG Result" in result or "ddg.example.com" in result
        finally:
            for k, v in saved_keys.items():
                if v is not None:
                    os.environ[k] = v


# ---------------------------------------------------------------------------
# 7. End-to-End Session Flow
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestE2ESessionFlow:
    """Full PAR loop integration: create session, chat, switch models, upload, jump back."""

    async def test_full_user_journey(self, client: httpx.AsyncClient) -> None:
        # Step 1: Create session
        r = await client.post("/api/sessions", json={"title": "E2E Test"})
        assert r.status_code == 201
        sid1 = r.json()["id"]

        # Step 2: Send chat with default model
        async def chat_default():
            return await client.post("/api/chat", json={
                "session_id": sid1,
                "message": "What is on my calendar today?",
                "model": "default",
                "web_enabled": True,
            })

        def review_chat(r: httpx.Response):
            if r.status_code != 200:
                return False, f"Chat returned {r.status_code}"
            events = parse_sse_events(r.text)
            if not any(e["event"] == "done" for e in events):
                return False, f"No done event. Events: {[e['event'] for e in events]}"
            return True, ""

        await par_loop(chat_default, review_chat, label="e2e_chat_default")

        # Step 3: Send chat with pro model
        async def chat_pro():
            return await client.post("/api/chat", json={
                "session_id": sid1,
                "message": "Give me a detailed analysis",
                "model": "pro",
            })

        await par_loop(chat_pro, review_chat, label="e2e_chat_pro")

        # Step 4: Upload a file
        upload_r = await client.post(
            "/api/files/upload",
            files={"file": ("notes.txt", make_tiny_txt())},
        )
        assert upload_r.status_code == 200
        stored_name = upload_r.json()["stored_name"]

        # Step 5: Chat with image mode
        async def chat_image():
            return await client.post("/api/chat", json={
                "session_id": sid1,
                "message": "Generate a diagram",
                "modes": ["image"],
            })

        await par_loop(chat_image, review_chat, label="e2e_chat_image")

        # Step 6: Create second session and chat
        r2 = await client.post("/api/sessions", json={"title": "E2E Session 2"})
        sid2 = r2.json()["id"]

        async def chat_s2():
            return await client.post("/api/chat", json={
                "session_id": sid2,
                "message": "Different conversation",
            })

        await par_loop(chat_s2, review_chat, label="e2e_chat_session2")

        # Step 7: Jump back to first session, verify messages preserved
        async def verify_s1():
            return await client.get(f"/api/sessions/{sid1}")

        def review_s1(r: httpx.Response):
            if r.status_code != 200:
                return False, f"Session 1 returned {r.status_code}"
            msgs = r.json().get("messages", [])
            user_msgs = [m for m in msgs if m["role"] == "user"]
            if len(user_msgs) < 3:
                return False, f"Expected >=3 user messages in session 1, got {len(user_msgs)}"
            return True, ""

        await par_loop(verify_s1, review_s1, label="e2e_verify_session1")

        # Step 8: Verify session 2 is separate
        async def verify_s2():
            return await client.get(f"/api/sessions/{sid2}")

        def review_s2(r: httpx.Response):
            if r.status_code != 200:
                return False, f"Session 2 returned {r.status_code}"
            msgs = r.json().get("messages", [])
            user_msgs = [m for m in msgs if m["role"] == "user"]
            if len(user_msgs) != 1:
                return False, f"Expected 1 user message in session 2, got {len(user_msgs)}"
            return True, ""

        await par_loop(verify_s2, review_s2, label="e2e_verify_session2")

        # Step 9: Delete session 1
        r = await client.delete(f"/api/sessions/{sid1}")
        assert r.status_code == 200

        r = await client.get(f"/api/sessions/{sid1}")
        assert r.status_code == 404


import os
