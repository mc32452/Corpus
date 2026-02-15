"""AI SDK Data Stream Protocol v1 encoder.

Encodes structured events into the line format consumed by Vercel AI SDK's
``useChat`` hook with ``streamProtocol: 'data'`` (the default).

Line format::

    {type_code}:{json_value}\\n

Type codes (subset we use):

    0  text part          — ``0:"token text"\\n``
    2  data part          — ``2:[{...}]\\n``
    3  error part         — ``3:"error message"\\n``
    8  message annotation — ``8:[{...}]\\n``
    d  finish message     — ``d:{"finishReason":"stop"}\\n``
    e  finish step        — ``e:{"finishReason":"stop","isContinued":false}\\n``

References
----------
- https://sdk.vercel.ai/docs/ai-sdk-ui/stream-protocol#data-stream-protocol
"""

from __future__ import annotations

import json
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Response headers required by the protocol
# ---------------------------------------------------------------------------

STREAM_HEADERS: dict[str, str] = {
    "Content-Type": "text/plain; charset=utf-8",
    "X-Vercel-AI-Data-Stream": "v1",
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",
}

# ---------------------------------------------------------------------------
# Type-code constants
# ---------------------------------------------------------------------------

_TEXT = "0"
_DATA = "2"
_ERROR = "3"
_ANNOTATION = "8"
_FINISH_MESSAGE = "d"
_FINISH_STEP = "e"


# ---------------------------------------------------------------------------
# Low-level line encoders
# ---------------------------------------------------------------------------


def encode_text(token: str) -> str:
    """Encode a text token.

    >>> encode_text("Hello")
    '0:"Hello"\\n'
    """
    return f"{_TEXT}:{json.dumps(token)}\n"


def encode_data(payload: list[dict[str, Any]]) -> str:
    """Encode a data array.

    >>> encode_data([{"key": "value"}])
    '2:[{"key": "value"}]\\n'
    """
    return f"{_DATA}:{json.dumps(payload)}\n"


def encode_error(message: str) -> str:
    """Encode an error string that terminates the stream.

    >>> encode_error("Something went wrong")
    '3:"Something went wrong"\\n'
    """
    return f"{_ERROR}:{json.dumps(message)}\n"


def encode_annotation(annotations: list[dict[str, Any]]) -> str:
    """Encode message annotations (status, sources, intent metadata, etc.).

    >>> encode_annotation([{"type": "status", "status": "Loading..."}])
    '8:[{"type": "status", "status": "Loading..."}]\\n'
    """
    return f"{_ANNOTATION}:{json.dumps(annotations)}\n"


def encode_finish_message(
    finish_reason: str = "stop",
    *,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> str:
    """Encode stream finish with optional usage stats.

    >>> encode_finish_message("stop")
    'd:{"finishReason":"stop","usage":{"promptTokens":0,"completionTokens":0}}\\n'
    """
    payload = {
        "finishReason": finish_reason,
        "usage": {
            "promptTokens": prompt_tokens,
            "completionTokens": completion_tokens,
        },
    }
    return f"{_FINISH_MESSAGE}:{json.dumps(payload)}\n"


def encode_finish_step(
    finish_reason: str = "stop",
    *,
    is_continued: bool = False,
) -> str:
    """Encode a step finish marker.

    >>> encode_finish_step("stop")
    'e:{"finishReason":"stop","isContinued":false}\\n'
    """
    payload = {
        "finishReason": finish_reason,
        "isContinued": is_continued,
    }
    return f"{_FINISH_STEP}:{json.dumps(payload)}\n"


# ---------------------------------------------------------------------------
# High-level annotation helpers (typed constructors)
# ---------------------------------------------------------------------------


def annotation_status(status: str) -> str:
    """Encode a status annotation line.

    >>> annotation_status("Classifying intent...")
    '8:[{"type": "status", "status": "Classifying intent..."}]\\n'
    """
    return encode_annotation([{"type": "status", "status": status}])


def annotation_sources(source_ids: list[str]) -> str:
    """Encode a sources annotation line.

    >>> annotation_sources(["doc_a", "doc_b"])
    '8:[{"type": "sources", "sourceIds": ["doc_a", "doc_b"]}]\\n'
    """
    return encode_annotation([{"type": "sources", "sourceIds": source_ids}])


def annotation_intent(
    intent: str,
    confidence: float,
    method: str,
) -> str:
    """Encode an intent classification annotation line.

    >>> annotation_intent("analyze", 0.85, "heuristic")
    '8:[{"type": "intent", "intent": "analyze", "confidence": 0.85, "method": "heuristic"}]\\n'
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
    '8:[{"type": "error", "error": {"code": "INTERNAL", "message": "Generation failed"}}]\\n'
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


# ---------------------------------------------------------------------------
# HTTP error body helper (for non-streaming error responses)
# ---------------------------------------------------------------------------


def http_error_body(code: str, message: str) -> dict[str, Any]:
    """Build the standard JSON error body for non-streaming HTTP responses.

    >>> http_error_body("LOCK_BUSY", "Another query is in progress")
    {'error': {'code': 'LOCK_BUSY', 'message': 'Another query is in progress'}}
    """
    return {"error": {"code": code, "message": message}}
