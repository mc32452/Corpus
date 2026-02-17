"""AI SDK UI message stream (SSE) encoder.

Encodes backend query events into Server-Sent Events consumed by AI SDK UI
message streams.

Line format::

    data: {json_payload}\\n\\n

References
----------
- https://sdk.vercel.ai/docs/ai-sdk-ui/stream-protocol
"""

from __future__ import annotations

import json
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Response headers required by the protocol
# ---------------------------------------------------------------------------

STREAM_HEADERS: dict[str, str] = {
    "Content-Type": "text/event-stream; charset=utf-8",
    "X-Vercel-AI-UI-Message-Stream": "v1",
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",
}

# ---------------------------------------------------------------------------
# Low-level line encoders
# ---------------------------------------------------------------------------


def _encode_sse_payload(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def encode_text(token: str) -> str:
    """Encode a text chunk.

    >>> encode_text("Hello")
    'data: {"type": "text", "value": "Hello"}\\n\\n'
    """
    return _encode_sse_payload({"type": "text", "value": token})


def encode_data(payload: list[dict[str, Any]]) -> str:
    """Encode one or more custom data parts as SSE payload lines."""
    lines: list[str] = []
    for item in payload:
        custom_type = item.get("type")
        if isinstance(custom_type, str):
            data_type = custom_type if custom_type.startswith("data-") else f"data-{custom_type}"
            lines.append(_encode_sse_payload({"type": data_type, "data": item}))
        else:
            lines.append(_encode_sse_payload({"type": "data", "data": item}))
    return "".join(lines)


def encode_error(message: str) -> str:
    """Encode an error payload that terminates the stream.

    >>> encode_error("Something went wrong")
    'data: {"type": "error", "error": "Something went wrong"}\\n\\n'
    """
    return _encode_sse_payload({"type": "error", "error": message})


def encode_annotation(annotations: list[dict[str, Any]]) -> str:
    """Encode annotations as custom data-* parts."""
    return encode_data(annotations)


def encode_finish_message(
    finish_reason: str = "stop",
    *,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> str:
    """Encode stream finish with optional usage stats."""
    payload = {
        "type": "finish",
        "finishReason": finish_reason,
        "usage": {
            "promptTokens": prompt_tokens,
            "completionTokens": completion_tokens,
        },
    }
    return _encode_sse_payload(payload)


def encode_finish_step(
    finish_reason: str = "stop",
    *,
    is_continued: bool = False,
) -> str:
    """Encode a finish-step marker as a custom data part."""
    payload = {
        "finishReason": finish_reason,
        "isContinued": is_continued,
    }
    return _encode_sse_payload({"type": "data-finish-step", "data": payload})


# ---------------------------------------------------------------------------
# High-level annotation helpers (typed constructors)
# ---------------------------------------------------------------------------


def annotation_status(status: str) -> str:
    """Encode a status annotation line.

    >>> annotation_status("Classifying intent...")
    'data: {"type": "data-status", "data": {"type": "status", "status": "Classifying intent..."}}\\n\\n'
    """
    return encode_annotation([{"type": "status", "status": status}])


def annotation_sources(source_ids: list[str]) -> str:
    """Encode a sources annotation line.

    >>> annotation_sources(["doc_a", "doc_b"])
    'data: {"type": "data-sources", "data": {"type": "sources", "sourceIds": ["doc_a", "doc_b"]}}\\n\\n'
    """
    return encode_annotation([{"type": "sources", "sourceIds": source_ids}])


def annotation_intent(
    intent: str,
    confidence: float,
    method: str,
) -> str:
    """Encode an intent classification annotation line.

    >>> annotation_intent("analyze", 0.85, "heuristic")
    'data: {"type": "data-intent", "data": {"type": "intent", "intent": "analyze", "confidence": 0.85, "method": "heuristic"}}\\n\\n'
    """
    return encode_annotation([{
        "type": "intent",
        "intent": intent,
        "confidence": confidence,
        "method": method,
    }])


def annotation_error(code: str, message: str) -> str:
    """Encode a structured error annotation (sent before ``encode_error``).

    This allows frontends to read the structured error code from annotations
    while still terminating the stream with the standard error line.

    >>> annotation_error("INTERNAL", "Generation failed")
    'data: {"type": "data-error", "data": {"type": "error", "error": {"code": "INTERNAL", "message": "Generation failed"}}}\\n\\n'
    """
    return encode_annotation([{
        "type": "error",
        "error": {"code": code, "message": message},
    }])


def annotation_error_with_metadata(
    code: str,
    message: str,
    metadata: Optional[dict[str, Any]] = None,
) -> str:
    payload: dict[str, Any] = {
        "type": "error",
        "error": {"code": code, "message": message},
    }
    if metadata:
        payload["error"]["metadata"] = metadata
    return encode_annotation([payload])


def annotation_citations(citations: list[dict[str, Any]]) -> str:
    """Encode citation list as a custom annotation part."""
    return encode_annotation([{"type": "citations", "citations": citations}])


# ---------------------------------------------------------------------------
# HTTP error body helper (for non-streaming error responses)
# ---------------------------------------------------------------------------


def http_error_body(code: str, message: str) -> dict[str, Any]:
    """Build the standard JSON error body for non-streaming HTTP responses.

    >>> http_error_body("LOCK_BUSY", "Another query is in progress")
    {'error': {'code': 'LOCK_BUSY', 'message': 'Another query is in progress'}}
    """
    return {"error": {"code": code, "message": message}}
