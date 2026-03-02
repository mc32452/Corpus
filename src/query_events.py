"""Structured event types for streaming RAG query execution.

These events are yielded by ``RagEngine.query_events()`` and consumed by
the API layer to produce AI SDK Data Stream Protocol lines.  They are
simple dataclasses — no framework dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union


@dataclass(frozen=True)
class StatusEvent:
    """Pipeline status update (e.g. "Classifying intent...")."""

    status: str


@dataclass(frozen=True)
class IntentEvent:
    """Intent classification result."""

    intent: str  # e.g. "analyze", "summarize"
    confidence: float
    method: str  # e.g. "heuristic", "llm-fallback"


@dataclass(frozen=True)
class SourcesEvent:
    """Source IDs matched by retrieval."""

    source_ids: list[str]


@dataclass(frozen=True)
class TextTokenEvent:
    """A single text token from LLM generation."""

    token: str


@dataclass(frozen=True)
class ThinkingTokenEvent:
    """A single thinking/reasoning token from LLM generation.

    Emitted only when thinking mode is active.  These tokens appear before
    any ``TextTokenEvent`` and represent the model\'s internal reasoning chain.
    The API layer streams them as AI SDK ``reasoning-*`` parts.
    """

    token: str


@dataclass(frozen=True)
class ErrorEvent:
    """An error that occurred during pipeline execution."""

    code: str  # e.g. "INTERNAL", "STREAM_CANCELLED"
    message: str
    metadata: Optional[dict[str, object]] = None


@dataclass(frozen=True)
class CitationListEvent:
    """Ordered list of citation entries built from packed retrieval results.

    Emitted once after budget packing when citations are enabled, before
    "Generating answer...".  Each entry maps a numbered citation index to
    a chunk with its metadata so the frontend can resolve [1], [2], etc.
    """

    citations: list[dict[str, object]] = field(default_factory=list)


@dataclass(frozen=True)
class FinishEvent:
    """Pipeline execution complete."""

    finish_reason: str = "stop"  # "stop" or "error"
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass(frozen=True)
class ToolCallBeginEvent:
    """Start of a tool call — maps to AI SDK v6 ``tool-call-begin`` frame.

    Emitted when the model invokes a tool.  The frontend runtime creates
    a tool-call message part which is rendered by ``makeAssistantToolUI``
    or the ``ToolFallback`` component.
    """

    tool_call_id: str
    tool_name: str
    arguments: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ToolResultEvent:
    """Result of a tool execution — maps to AI SDK v6 ``tool-result`` frame.

    Emitted after the tool has been executed so the frontend can display
    the result inside the tool-call card.
    """

    tool_call_id: str
    result: object = None


# Union type for type checking (Python 3.9 compatible)
QueryEvent = Union[
    StatusEvent,
    IntentEvent,
    SourcesEvent,
    TextTokenEvent,
    ThinkingTokenEvent,
    CitationListEvent,
    ErrorEvent,
    FinishEvent,
    ToolCallBeginEvent,
    ToolResultEvent,
]
