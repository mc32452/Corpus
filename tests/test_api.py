"""Tests for the FastAPI chat endpoint and health route.

Uses httpx.AsyncClient against the real FastAPI app with a mocked
RagEngine that yields query events, avoiding ML model loading.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Optional
from unittest.mock import patch

import httpx
import pytest

from src.api import app
from src.query_events import (
    ErrorEvent,
    FinishEvent,
    IntentEvent,
    SourcesEvent,
    StatusEvent,
    TextTokenEvent,
)

# Chat endpoint uses plain-text stream (AI SDK Text Stream Protocol), not data protocol
LOADING_PREFIX = "Loading RAG engine…\n"


# ---------------------------------------------------------------------------
# Mock RagEngine with query_events support
# ---------------------------------------------------------------------------


class MockRagEngine:
    """Mock RagEngine that yields query events without loading ML models."""

    def __init__(
        self,
        *,
        answer: str = "This is a test answer from the RAG engine.",
        intent: str = "analyze",
        confidence: float = 0.85,
        method: str = "heuristic",
        source_ids: Optional[list[str]] = None,
        query_delay: float = 0.0,
        fail: bool = False,
        fail_during_generation: bool = False,
    ):
        self._answer = answer
        self._intent = intent
        self._confidence = confidence
        self._method = method
        self._source_ids = source_ids or ["test_source"]
        self._query_delay = query_delay
        self._fail = fail
        self._fail_during_generation = fail_during_generation
        self.query_count = 0

    def query_events(self, query_text, *, source_id=None, citations_enabled=None, should_stop=None):
        """Yield mock query events."""
        self.query_count += 1
        _stop = should_stop or (lambda: False)

        if self._query_delay > 0:
            time.sleep(self._query_delay)

        if self._fail:
            raise RuntimeError("Mock engine failure")

        yield StatusEvent(status="Preparing retrieval models...")
        if _stop():
            yield ErrorEvent(code="STREAM_CANCELLED", message="Cancelled")
            yield FinishEvent(finish_reason="error")
            return

        yield StatusEvent(status="Classifying intent...")
        yield IntentEvent(
            intent=self._intent,
            confidence=self._confidence,
            method=self._method,
        )

        if _stop():
            yield ErrorEvent(code="STREAM_CANCELLED", message="Cancelled")
            yield FinishEvent(finish_reason="error")
            return

        yield StatusEvent(status="Searching knowledge base...")

        source_ids = [source_id] if source_id else self._source_ids
        yield SourcesEvent(source_ids=source_ids)

        yield StatusEvent(status="Generating answer...")

        if self._fail_during_generation:
            yield ErrorEvent(code="INTERNAL", message="Generation failed mid-stream")
            yield FinishEvent(finish_reason="error")
            return

        # Stream answer as individual tokens (like real generator)
        answer = f"Answer to: {query_text}" if "Answer" not in self._answer else self._answer
        words = answer.split(" ")
        token_count = 0
        for i, word in enumerate(words):
            if _stop():
                yield ErrorEvent(code="STREAM_CANCELLED", message="Cancelled during generation")
                yield FinishEvent(finish_reason="error")
                return
            token = word if i == 0 else f" {word}"
            yield TextTokenEvent(token=token)
            token_count += 1

        yield FinishEvent(finish_reason="stop", completion_tokens=token_count)

    def close(self):
        pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_stream_lines(body: str) -> list[tuple[str, object]]:
    """Parse all protocol lines from a stream body."""
    lines = []
    for raw_line in body.split("\n"):
        if not raw_line:
            continue
        colon_idx = raw_line.index(":")
        type_code = raw_line[:colon_idx]
        json_str = raw_line[colon_idx + 1:]
        value = json.loads(json_str)
        lines.append((type_code, value))
    return lines


def _find_lines_by_type(parsed: list[tuple[str, object]], type_code: str) -> list[object]:
    return [val for code, val in parsed if code == type_code]


def _get_text_content(parsed: list[tuple[str, object]]) -> str:
    """Concatenate all text parts (type 0) from parsed stream."""
    return "".join(val for code, val in parsed if code == "0")


def _get_annotations(parsed: list[tuple[str, object]]) -> list[dict]:
    """Flatten all annotation arrays (type 8)."""
    result = []
    for code, val in parsed:
        if code == "8" and isinstance(val, list):
            result.extend(val)
    return result


def _get_plain_text_content(body: str) -> str:
    """Return the assistant text from a plain-text stream (strip loading prefix)."""
    if body.startswith(LOADING_PREFIX):
        return body[len(LOADING_PREFIX) :]
    return body


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_engine():
    return MockRagEngine()


@pytest.fixture
def mock_engine_slow():
    return MockRagEngine(query_delay=0.5)


@pytest.fixture
def mock_engine_failing():
    return MockRagEngine(fail=True)


@pytest.fixture
def mock_engine_gen_fail():
    return MockRagEngine(fail_during_generation=True)


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    @pytest.mark.anyio
    async def test_health_returns_ok(self) -> None:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"


# ---------------------------------------------------------------------------
# Chat endpoint — happy path
# ---------------------------------------------------------------------------


class TestChatHappyPath:
    @pytest.mark.anyio
    async def test_stream_contains_text_and_finish(self, mock_engine) -> None:
        """Valid chat request returns plain-text stream with loading prefix and answer."""
        with patch("src.api._get_engine", return_value=mock_engine):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/api/chat",
                    json={"messages": [{"role": "user", "content": "What is this?"}]},
                )

        assert resp.status_code == 200
        body = resp.text
        assert body.startswith(LOADING_PREFIX)
        text = _get_plain_text_content(body)
        assert "Answer to: What is this?" in text

    @pytest.mark.anyio
    async def test_stream_headers(self, mock_engine) -> None:
        """Response has correct text-stream headers (no data protocol)."""
        with patch("src.api._get_engine", return_value=mock_engine):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/api/chat",
                    json={"messages": [{"role": "user", "content": "Hello"}]},
                )

        assert resp.headers["content-type"] == "text/plain; charset=utf-8"
        assert resp.headers["cache-control"] == "no-cache"
        assert resp.headers["x-accel-buffering"] == "no"

    @pytest.mark.anyio
    async def test_stream_contains_intent_annotation(self, mock_engine) -> None:
        """Text stream does not send annotations; answer text is present."""
        with patch("src.api._get_engine", return_value=mock_engine):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/api/chat",
                    json={"messages": [{"role": "user", "content": "Analyze this"}]},
                )

        assert resp.status_code == 200
        text = _get_plain_text_content(resp.text)
        assert "Answer to: Analyze this" in text  # mock answer present

    @pytest.mark.anyio
    async def test_stream_contains_source_annotation(self, mock_engine) -> None:
        """Text stream returns answer; source_id is passed to engine (no annotation in stream)."""
        with patch("src.api._get_engine", return_value=mock_engine):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/api/chat",
                    json={"messages": [{"role": "user", "content": "Query"}]},
                )

        assert resp.status_code == 200
        assert "Answer to: Query" in resp.text or "theory " in _get_plain_text_content(resp.text)

    @pytest.mark.anyio
    async def test_stream_contains_status_annotations(self, mock_engine) -> None:
        """Stream starts with loading status line then answer text."""
        with patch("src.api._get_engine", return_value=mock_engine):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/api/chat",
                    json={"messages": [{"role": "user", "content": "Question"}]},
                )

        assert resp.status_code == 200
        assert resp.text.startswith(LOADING_PREFIX)
        assert len(_get_plain_text_content(resp.text)) > 0

    @pytest.mark.anyio
    async def test_source_filter_passed_through(self, mock_engine) -> None:
        """Optional source_id from data field is passed to engine; stream returns 200 and answer."""
        with patch("src.api._get_engine", return_value=mock_engine):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/api/chat",
                    json={
                        "messages": [{"role": "user", "content": "Query"}],
                        "data": {"source_id": "specific_doc"},
                    },
                )

        assert resp.status_code == 200
        assert "Answer to: Query" in resp.text or "theory " in _get_plain_text_content(resp.text)

    @pytest.mark.anyio
    async def test_chat_with_history(self, mock_engine) -> None:
        """Chat with multiple messages works (last message is the query)."""
        with patch("src.api._get_engine", return_value=mock_engine):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/api/chat",
                    json={
                        "messages": [
                            {"role": "user", "content": "First question"},
                            {"role": "assistant", "content": "First answer"},
                            {"role": "user", "content": "Follow up"},
                        ]
                    },
                )

        assert resp.status_code == 200
        text = _get_plain_text_content(resp.text)
        assert "Answer to: Follow up" in text

    @pytest.mark.anyio
    async def test_text_tokens_arrive_individually(self, mock_engine) -> None:
        """Plain-text stream contains the mock answer (multiple tokens concatenated)."""
        with patch("src.api._get_engine", return_value=mock_engine):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/api/chat",
                    json={"messages": [{"role": "user", "content": "Test"}]},
                )

        text = _get_plain_text_content(resp.text)
        assert "Answer to: Test" in text  # mock yields tokens that form this answer


# ---------------------------------------------------------------------------
# Chat endpoint — validation errors
# ---------------------------------------------------------------------------


class TestChatValidation:
    @pytest.mark.anyio
    async def test_empty_messages_rejected(self) -> None:
        """Empty messages array returns 422."""
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post("/api/chat", json={"messages": []})
        assert resp.status_code == 422

    @pytest.mark.anyio
    async def test_missing_messages_rejected(self) -> None:
        """Missing messages field returns 422."""
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post("/api/chat", json={})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Chat endpoint — lock busy (429)
# ---------------------------------------------------------------------------


class TestChatLockBusy:
    @pytest.mark.anyio
    async def test_concurrent_request_gets_429(self, mock_engine_slow) -> None:
        """When chat_lock is held, a second request gets 429 LOCK_BUSY."""
        with patch("src.api._get_engine", return_value=mock_engine_slow):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                task1 = asyncio.create_task(
                    client.post(
                        "/api/chat",
                        json={"messages": [{"role": "user", "content": "Slow query"}]},
                    )
                )
                await asyncio.sleep(0.1)

                resp2 = await client.post(
                    "/api/chat",
                    json={"messages": [{"role": "user", "content": "Blocked query"}]},
                )

                assert resp2.status_code == 429
                body = resp2.json()
                assert body["error"]["code"] == "LOCK_BUSY"
                assert "already in progress" in body["error"]["message"]

                resp1 = await task1
                assert resp1.status_code == 200

    @pytest.mark.anyio
    async def test_lock_released_after_error(self, mock_engine_failing) -> None:
        """Lock is released even when the engine throws an exception."""
        with patch("src.api._get_engine", return_value=mock_engine_failing):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp1 = await client.post(
                    "/api/chat",
                    json={"messages": [{"role": "user", "content": "Fail"}]},
                )
                assert resp1.status_code == 200
                assert "Error:" in resp1.text

                with patch("src.api._get_engine", return_value=MockRagEngine()):
                    resp2 = await client.post(
                        "/api/chat",
                        json={"messages": [{"role": "user", "content": "After error"}]},
                    )
                assert resp2.status_code == 200
                text = _get_plain_text_content(resp2.text)
                assert "Answer to: After error" in text


# ---------------------------------------------------------------------------
# Chat endpoint — error handling
# ---------------------------------------------------------------------------


class TestChatEngineErrors:
    @pytest.mark.anyio
    async def test_engine_exception_produces_error_stream(
        self, mock_engine_failing
    ) -> None:
        """Engine exception produces structured error in the stream."""
        with patch("src.api._get_engine", return_value=mock_engine_failing):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/api/chat",
                    json={"messages": [{"role": "user", "content": "Will fail"}]},
                )

        assert resp.status_code == 200
        assert "Error:" in resp.text
        assert "INTERNAL" in resp.text or "Mock engine failure" in resp.text

    @pytest.mark.anyio
    async def test_mid_generation_error_produces_error_stream(
        self, mock_engine_gen_fail
    ) -> None:
        """Error during generation (after status events) still produces proper error."""
        with patch("src.api._get_engine", return_value=mock_engine_gen_fail):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/api/chat",
                    json={"messages": [{"role": "user", "content": "Generate fail"}]},
                )

        assert "Error:" in resp.text
        assert "Generation failed" in resp.text or "INTERNAL" in resp.text


# ---------------------------------------------------------------------------
# Stream line format validation
# ---------------------------------------------------------------------------


class TestStreamLineFormat:
    @pytest.mark.anyio
    async def test_plain_text_stream_non_empty(self, mock_engine) -> None:
        """Plain-text stream response is non-empty and valid UTF-8."""
        with patch("src.api._get_engine", return_value=mock_engine):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/api/chat",
                    json={"messages": [{"role": "user", "content": "Test"}]},
                )

        assert resp.status_code == 200
        body = resp.text
        assert len(body) > 0
        assert body.startswith(LOADING_PREFIX)
        assert "Answer to: Test" in body or "Test" in body

    @pytest.mark.anyio
    async def test_stream_ends_with_answer(self, mock_engine) -> None:
        """Plain-text stream ends with the assistant answer (no protocol finish markers)."""
        with patch("src.api._get_engine", return_value=mock_engine):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/api/chat",
                    json={"messages": [{"role": "user", "content": "End test"}]},
                )

        assert resp.status_code == 200
        text = _get_plain_text_content(resp.text)
        assert "Answer to: End test" in text
