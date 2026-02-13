"""Tests for AI SDK Data Stream Protocol v1 encoder.

Validates exact line format, JSON validity, newline termination,
and edge cases (empty strings, unicode, special characters).
"""

from __future__ import annotations

import json

import pytest

from src.stream_protocol import (
    STREAM_HEADERS,
    annotation_error,
    annotation_intent,
    annotation_sources,
    annotation_status,
    encode_annotation,
    encode_data,
    encode_error,
    encode_finish_message,
    encode_finish_step,
    encode_text,
    http_error_body,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_line(line: str) -> tuple[str, object]:
    """Parse a protocol line into (type_code, decoded_json_value).

    Every valid line is ``{type_code}:{json}\n``.
    """
    assert line.endswith("\n"), f"Line must end with newline: {line!r}"
    body = line[:-1]  # strip trailing \n
    colon_idx = body.index(":")
    type_code = body[:colon_idx]
    json_str = body[colon_idx + 1 :]
    value = json.loads(json_str)
    return type_code, value


# ---------------------------------------------------------------------------
# encode_text
# ---------------------------------------------------------------------------


class TestEncodeText:
    def test_simple_token(self) -> None:
        line = encode_text("Hello")
        assert line == '0:"Hello"\n'
        code, val = _parse_line(line)
        assert code == "0"
        assert val == "Hello"

    def test_empty_string(self) -> None:
        line = encode_text("")
        assert line == '0:""\n'
        code, val = _parse_line(line)
        assert code == "0"
        assert val == ""

    def test_unicode_characters(self) -> None:
        line = encode_text("Héllo wörld 你好 🌍")
        code, val = _parse_line(line)
        assert code == "0"
        assert val == "Héllo wörld 你好 🌍"

    def test_special_json_characters(self) -> None:
        """Quotes, backslashes, and newlines must be JSON-escaped."""
        text = 'He said "hello"\nand\\walked away'
        line = encode_text(text)
        code, val = _parse_line(line)
        assert code == "0"
        assert val == text

    def test_single_space(self) -> None:
        line = encode_text(" ")
        code, val = _parse_line(line)
        assert val == " "

    def test_multiline_content(self) -> None:
        text = "line1\nline2\nline3"
        line = encode_text(text)
        code, val = _parse_line(line)
        assert val == text
        # The encoded line itself should be a single line (embedded newlines are escaped)
        assert line.count("\n") == 1


# ---------------------------------------------------------------------------
# encode_data
# ---------------------------------------------------------------------------


class TestEncodeData:
    def test_single_object(self) -> None:
        line = encode_data([{"key": "value"}])
        code, val = _parse_line(line)
        assert code == "2"
        assert val == [{"key": "value"}]

    def test_empty_array(self) -> None:
        line = encode_data([])
        code, val = _parse_line(line)
        assert code == "2"
        assert val == []

    def test_multiple_objects(self) -> None:
        data = [{"a": 1}, {"b": 2}]
        line = encode_data(data)
        code, val = _parse_line(line)
        assert code == "2"
        assert val == data

    def test_nested_objects(self) -> None:
        data = [{"outer": {"inner": [1, 2, 3]}}]
        line = encode_data(data)
        code, val = _parse_line(line)
        assert val == data


# ---------------------------------------------------------------------------
# encode_error
# ---------------------------------------------------------------------------


class TestEncodeError:
    def test_simple_error(self) -> None:
        line = encode_error("Something went wrong")
        assert line == '3:"Something went wrong"\n'
        code, val = _parse_line(line)
        assert code == "3"
        assert val == "Something went wrong"

    def test_error_with_quotes(self) -> None:
        msg = 'File "test.pdf" not found'
        line = encode_error(msg)
        code, val = _parse_line(line)
        assert code == "3"
        assert val == msg

    def test_empty_error(self) -> None:
        line = encode_error("")
        code, val = _parse_line(line)
        assert code == "3"
        assert val == ""


# ---------------------------------------------------------------------------
# encode_annotation
# ---------------------------------------------------------------------------


class TestEncodeAnnotation:
    def test_single_annotation(self) -> None:
        line = encode_annotation([{"type": "status", "status": "Loading..."}])
        code, val = _parse_line(line)
        assert code == "8"
        assert val == [{"type": "status", "status": "Loading..."}]

    def test_empty_annotations(self) -> None:
        line = encode_annotation([])
        code, val = _parse_line(line)
        assert code == "8"
        assert val == []

    def test_multiple_annotations(self) -> None:
        annotations = [
            {"type": "status", "status": "step1"},
            {"type": "sources", "sourceIds": ["a"]},
        ]
        line = encode_annotation(annotations)
        code, val = _parse_line(line)
        assert val == annotations


# ---------------------------------------------------------------------------
# encode_finish_message
# ---------------------------------------------------------------------------


class TestEncodeFinishMessage:
    def test_default_finish(self) -> None:
        line = encode_finish_message()
        code, val = _parse_line(line)
        assert code == "d"
        assert val["finishReason"] == "stop"
        assert val["usage"]["promptTokens"] == 0
        assert val["usage"]["completionTokens"] == 0

    def test_custom_reason_and_usage(self) -> None:
        line = encode_finish_message(
            "length", prompt_tokens=100, completion_tokens=50
        )
        code, val = _parse_line(line)
        assert code == "d"
        assert val["finishReason"] == "length"
        assert val["usage"]["promptTokens"] == 100
        assert val["usage"]["completionTokens"] == 50

    def test_error_finish_reason(self) -> None:
        line = encode_finish_message("error")
        code, val = _parse_line(line)
        assert val["finishReason"] == "error"


# ---------------------------------------------------------------------------
# encode_finish_step
# ---------------------------------------------------------------------------


class TestEncodeFinishStep:
    def test_default_step(self) -> None:
        line = encode_finish_step()
        code, val = _parse_line(line)
        assert code == "e"
        assert val["finishReason"] == "stop"
        assert val["isContinued"] is False

    def test_continued_step(self) -> None:
        line = encode_finish_step("stop", is_continued=True)
        code, val = _parse_line(line)
        assert val["isContinued"] is True


# ---------------------------------------------------------------------------
# High-level annotation helpers
# ---------------------------------------------------------------------------


class TestAnnotationHelpers:
    def test_status(self) -> None:
        line = annotation_status("Classifying intent...")
        code, val = _parse_line(line)
        assert code == "8"
        assert len(val) == 1
        assert val[0]["type"] == "status"
        assert val[0]["status"] == "Classifying intent..."

    def test_sources(self) -> None:
        line = annotation_sources(["doc_a", "doc_b"])
        code, val = _parse_line(line)
        assert val[0]["type"] == "sources"
        assert val[0]["sourceIds"] == ["doc_a", "doc_b"]

    def test_sources_empty(self) -> None:
        line = annotation_sources([])
        code, val = _parse_line(line)
        assert val[0]["sourceIds"] == []

    def test_intent(self) -> None:
        line = annotation_intent("analyze", 0.85, "heuristic")
        code, val = _parse_line(line)
        assert val[0]["type"] == "intent"
        assert val[0]["intent"] == "analyze"
        assert val[0]["confidence"] == 0.85
        assert val[0]["method"] == "heuristic"

    def test_intent_low_confidence(self) -> None:
        line = annotation_intent("overview", 0.40, "fallback")
        code, val = _parse_line(line)
        assert val[0]["confidence"] == 0.40

    def test_error_annotation(self) -> None:
        line = annotation_error("INTERNAL", "Generation failed")
        code, val = _parse_line(line)
        assert val[0]["type"] == "error"
        assert val[0]["error"]["code"] == "INTERNAL"
        assert val[0]["error"]["message"] == "Generation failed"

    def test_error_annotation_all_codes(self) -> None:
        """Verify all error codes from the contract can be encoded."""
        for error_code in [
            "LOCK_BUSY",
            "SOURCE_NOT_FOUND",
            "INGEST_FAILED",
            "STREAM_CANCELLED",
            "INTERNAL",
        ]:
            line = annotation_error(error_code, f"Test {error_code}")
            code, val = _parse_line(line)
            assert val[0]["error"]["code"] == error_code


# ---------------------------------------------------------------------------
# HTTP error body
# ---------------------------------------------------------------------------


class TestHttpErrorBody:
    def test_structure(self) -> None:
        body = http_error_body("LOCK_BUSY", "Another query is in progress")
        assert body == {
            "error": {
                "code": "LOCK_BUSY",
                "message": "Another query is in progress",
            }
        }

    def test_serializable(self) -> None:
        body = http_error_body("INTERNAL", "Unexpected error")
        serialized = json.dumps(body)
        deserialized = json.loads(serialized)
        assert deserialized == body


# ---------------------------------------------------------------------------
# STREAM_HEADERS
# ---------------------------------------------------------------------------


class TestStreamHeaders:
    def test_content_type(self) -> None:
        assert STREAM_HEADERS["Content-Type"] == "text/plain; charset=utf-8"

    def test_data_stream_header(self) -> None:
        assert STREAM_HEADERS["X-Vercel-AI-Data-Stream"] == "v1"

    def test_no_cache(self) -> None:
        assert STREAM_HEADERS["Cache-Control"] == "no-cache"

    def test_no_buffering(self) -> None:
        assert STREAM_HEADERS["X-Accel-Buffering"] == "no"


# ---------------------------------------------------------------------------
# Integration: full stream sequence
# ---------------------------------------------------------------------------


class TestFullStreamSequence:
    """Simulate a complete chat stream and verify the full byte sequence."""

    def test_happy_path_sequence(self) -> None:
        """A normal chat response: status → intent → text chunks → sources → finish."""
        lines: list[str] = []

        # 1. Status updates
        lines.append(annotation_status("Classifying intent..."))
        lines.append(annotation_intent("analyze", 0.85, "heuristic"))
        lines.append(annotation_status("Searching knowledge base..."))
        lines.append(annotation_status("Generating answer..."))

        # 2. Text tokens
        lines.append(encode_text("The "))
        lines.append(encode_text("theory "))
        lines.append(encode_text("of "))
        lines.append(encode_text("generative "))
        lines.append(encode_text("grammar..."))

        # 3. Sources
        lines.append(annotation_sources(["linguistics_doc"]))

        # 4. Finish
        lines.append(encode_finish_step("stop"))
        lines.append(encode_finish_message("stop", prompt_tokens=500, completion_tokens=25))

        # Verify: all lines are valid protocol lines
        for line in lines:
            assert line.endswith("\n")
            _parse_line(line)  # must not raise

        # Verify: concatenation produces valid stream body
        body = "".join(lines)
        assert body.count("\n") == len(lines)

    def test_error_mid_stream_sequence(self) -> None:
        """Error occurs after some text has been sent."""
        lines: list[str] = []

        lines.append(annotation_status("Generating answer..."))
        lines.append(encode_text("Partial "))
        lines.append(encode_text("response"))

        # Error occurs
        lines.append(annotation_error("INTERNAL", "Model crashed"))
        lines.append(encode_error("Model crashed"))

        # Finish with error reason
        lines.append(encode_finish_step("error"))
        lines.append(encode_finish_message("error"))

        for line in lines:
            _parse_line(line)

        # The error annotation contains structured info
        _, error_ann = _parse_line(lines[3])
        assert error_ann[0]["error"]["code"] == "INTERNAL"

        # The error line contains the human-readable message
        _, error_msg = _parse_line(lines[4])
        assert error_msg == "Model crashed"

    def test_cancellation_sequence(self) -> None:
        """Stream cancelled by client disconnect."""
        lines: list[str] = []

        lines.append(encode_text("Partial"))
        lines.append(annotation_error("STREAM_CANCELLED", "Client disconnected"))
        lines.append(encode_error("Client disconnected"))
        lines.append(encode_finish_step("error"))
        lines.append(encode_finish_message("error"))

        for line in lines:
            _parse_line(line)
