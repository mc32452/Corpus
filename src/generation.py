from __future__ import annotations

from typing import Optional

from .intent import Intent


# Intent-specific instructions (v1.1 - with overview for high-level queries)
# NOTE: Bullet points are NOT enforced by default. Format is flexible and intent-driven.
INTENT_INSTRUCTIONS: dict[Intent, dict[str, str]] = {
    # OVERVIEW: Bird's-eye view - concise, purpose-driven
    Intent.OVERVIEW: {
        "task": "Provide a brief, high-level description of this document.",
        "format": (
            "Your first sentence MUST state what type of document this is and its primary purpose "
            "(e.g., 'This is a critical review of...', 'This is a research paper that examines...'). "
            "Keep your response to 1 short paragraph OR a maximum of 3 concise bullet points. "
            "Do NOT go into detailed arguments or methodology. "
            "Do NOT include page numbers, document headers, or citation markers in the text."
        ),
        "tone": "Neutral and concise. Think 'back-of-book blurb' level.",
    },
    # SUMMARIZE: Detailed summary with key points
    Intent.SUMMARIZE: {
        "task": "Provide a structured summary of the key points from the context.",
        "format": (
            "Start with one sentence stating what the document is. "
            "Then provide 3-5 distinct key points. Use bullet points if listing multiple items. "
            "Each point should be unique - do not repeat the same idea in different words. "
            "Do NOT include page numbers or citation markers."
        ),
        "tone": "Academic but accessible.",
    },
    # EXPLAIN: Simple language for non-experts
    Intent.EXPLAIN: {
        "task": "Explain the content in simple, non-technical language.",
        "format": "Use short paragraphs. Avoid jargon. Use analogies if helpful. Do NOT include page numbers.",
        "tone": "Conversational, as if explaining to a curious friend.",
    },
    # ANALYZE: Critique, controversy, evaluation
    Intent.ANALYZE: {
        "task": "Analyze the specific aspect the user is asking about.",
        "format": "Present evidence and reasoning. Acknowledge multiple perspectives if relevant.",
        "tone": "Thoughtful and balanced.",
    },
}

# System message with grounding rules - placed BEFORE context to prevent continuation
_SYSTEM_MESSAGE = """You are a helpful research assistant. Follow these rules strictly:

1. Base your answer ONLY on the provided context below.
2. Answer the user's SPECIFIC question - do not default to a generic summary.
3. If the context doesn't contain relevant information, say so.
4. Stop generating after completing your answer. Do not continue with follow-up text.
5. Never repeat instructions or add meta-commentary about your response."""


def build_prompt(
    context: str,
    question: str,
    intent: Optional[Intent] = None,
) -> str:
    """
    Build an intent-aware prompt for the LLM.
    
    Structure: System instructions -> Task/Format/Tone -> Context -> Question
    All instructions are placed BEFORE the context to prevent model continuation.
    
    Args:
        context: Retrieved document context
        question: User's original question
        intent: Classified intent (defaults to OVERVIEW if None)
    
    Returns:
        Formatted prompt string
    """
    if intent is None:
        intent = Intent.OVERVIEW
    
    cfg = INTENT_INSTRUCTIONS.get(intent, INTENT_INSTRUCTIONS[Intent.OVERVIEW])
    
    # Build system block with all instructions BEFORE context
    system_block = (
        f"{_SYSTEM_MESSAGE}\n\n"
        f"Task: {cfg['task']}\n"
        f"Format: {cfg['format']}\n"
        f"Tone: {cfg['tone']}"
    )

    # User block contains only context and question - no trailing instructions
    return (
        f"System: {system_block}\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )


# Legacy function for backward compatibility
def build_prompt_legacy(context: str, question: str) -> str:
    """Original prompt builder (deprecated, kept for reference)."""
    instruction = (
        "You are a helpful research assistant.\n"
        "Task: Summarize the retrieved context to answer the user's question.\n"
        "Constraints:\n"
        "1. Start directly with bullet points. Do NOT write an introduction paragraph.\n"
        "2. Provide exactly 3-5 distinct bullet points.\n"
        "3. Stop writing immediately after the last bullet point.\n"
        "4. Do not repeat the same point in different words.\n"
        "Terminate response immediately after the last point."
    )

    return (
        f"System: {instruction}\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\nAnswer:"
    )
