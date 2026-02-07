from __future__ import annotations

from typing import Optional

from .intent import Intent


INTENT_INSTRUCTIONS: dict[Intent, dict[str, str]] = {
    Intent.OVERVIEW: {
        "task": "Provide a brief, high-level description of this document.",
        "format": (
            "Your first sentence MUST state what type of document this is and its primary purpose. "
            "Keep your response to 1 short paragraph OR a maximum of 3 concise bullet points. "
            "Do NOT include page numbers, document headers, or citation markers."
        ),
        "tone": "Neutral and concise.",
    },
    Intent.SUMMARIZE: {
        "task": "Provide a structured summary of the key points from the context.",
        "format": (
            "Start with one sentence stating what the document is. "
            "Then provide 3-5 distinct key points using bullet points. "
            "Do NOT include page numbers or citation markers."
        ),
        "tone": "Academic but accessible.",
    },
    Intent.EXPLAIN: {
        "task": "Explain the content in simple, non-technical language.",
        "format": (
            "Use short paragraphs and avoid jargon. Use at least one analogy. "
            "Stop immediately after your final clarifying sentence."
        ),
        "tone": "Conversational.",
    },
    Intent.ANALYZE: {
        "task": "Analyze and synthesize the specific aspect the user is asking about.",
        "format": (
            "Present a synthesized analysis with evidence - do NOT use bullet points. "
            "Write in flowing paragraphs that connect ideas."
        ),
        "tone": "Thoughtful and balanced.",
    },
}

_SYSTEM_MESSAGE = """You are a helpful research assistant. Follow these rules strictly:

1. Base your answer ONLY on the provided context.
2. Answer the user's SPECIFIC question - do not default to a generic summary.
3. If the context doesn't contain relevant information, say so.
4. Stop generating after completing your answer.

Do NOT include meta-commentary, self-evaluations, or phrases like "Answer ends here" or "This response reflects..."."""

_CITATION_RULES = """
CITATION REQUIREMENTS:
- Context chunks are marked with [CHUNK START | SOURCE: SourceID | PAGE: X]
- Extract the SourceID and PAGE from these markers
- Cite every factual claim using [SourceID, p. X] format
- Example: [Chomsky_Skinner_Review, p. 1]
- REQUIRED: Every claim must have a citation"""


def build_prompt(
    context: str,
    question: str,
    intent: Optional[Intent] = None,
    extra_instructions: Optional[str] = None,
    citations_enabled: bool = False,
    source_legend: Optional[str] = None,
) -> str:
    """Build an intent-aware prompt for the LLM."""
    if intent is None:
        intent = Intent.OVERVIEW

    cfg = INTENT_INSTRUCTIONS.get(intent, INTENT_INSTRUCTIONS[Intent.OVERVIEW])
    extra_block = f"\nAdditional constraints: {extra_instructions.strip()}" if extra_instructions and extra_instructions.strip() else ""

    citation_block = ""
    if citations_enabled:
        citation_block = f"\n{_CITATION_RULES}"
        format_instructions = cfg['format']
        for pattern in ["Do NOT include page numbers, document headers, or citation markers. ", "Do NOT include page numbers or citation markers.", "Do NOT include page numbers."]:
            format_instructions = format_instructions.replace(pattern, "")
        format_instructions += " Include inline citations [SourceID, p. X] for factual claims."
        cfg = {**cfg, 'format': format_instructions}

    system_block = (
        f"{_SYSTEM_MESSAGE}{citation_block}\n\n"
        f"Task: {cfg['task']}\nFormat: {cfg['format']}\nTone: {cfg['tone']}{extra_block}"
    )
    legend_block = f"\n\n{source_legend}" if citations_enabled and source_legend else ""

    return f"System: {system_block}\n\nContext:\n{context}{legend_block}\n\nQuestion: {question}\n\nAnswer:"


def _build_system_block(cfg: dict[str, str], citation_block: str, extra_block: str) -> str:
    return f"{_SYSTEM_MESSAGE}{citation_block}\n\nTask: {cfg['task']}\nFormat: {cfg['format']}\nTone: {cfg['tone']}{extra_block}"


def _prepare_config(intent: Optional[Intent], citations_enabled: bool, extra_instructions: Optional[str]) -> tuple[dict[str, str], str, str]:
    intent = intent or Intent.OVERVIEW
    cfg = INTENT_INSTRUCTIONS.get(intent, INTENT_INSTRUCTIONS[Intent.OVERVIEW]).copy()
    extra_block = f"\nAdditional constraints: {extra_instructions.strip()}" if extra_instructions and extra_instructions.strip() else ""
    citation_block = ""
    if citations_enabled:
        citation_block = f"\n{_CITATION_RULES}"
        for pattern in ["Do NOT include page numbers, document headers, or citation markers. ", "Do NOT include page numbers or citation markers.", "Do NOT include page numbers."]:
            cfg['format'] = cfg['format'].replace(pattern, "")
        cfg['format'] += " Include inline citations [SourceID, p. X] for factual claims."
    return cfg, citation_block, extra_block


def build_messages(
    context: str,
    question: str,
    intent: Optional[Intent] = None,
    extra_instructions: Optional[str] = None,
    citations_enabled: bool = False,
    source_legend: Optional[str] = None,
) -> list[dict[str, str]]:
    """Build intent-aware chat messages for the LLM."""
    cfg, citation_block, extra_block = _prepare_config(intent, citations_enabled, extra_instructions)
    system_block = _build_system_block(cfg, citation_block, extra_block)
    legend_block = f"\n\n{source_legend}" if citations_enabled and source_legend else ""
    user_block = f"Context:\n{context}{legend_block}\n\nQuestion: {question}\n\nAnswer:"
    return [{"role": "system", "content": system_block}, {"role": "user", "content": user_block}]
