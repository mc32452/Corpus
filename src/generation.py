from __future__ import annotations

from typing import Optional

from .intent import Intent


# ---------------------------------------------------------------------------
# System message with paragraph formatting rules
# ---------------------------------------------------------------------------

_SYSTEM_MESSAGE = """You are a research assistant. Follow these rules strictly:

1. Use ONLY the provided context. Do not rely on outside knowledge.
2. Answer the user's SPECIFIC question — do not provide unrelated information.
3. Write in SHORT, focused paragraphs (60-100 words each). Each paragraph should develop ONE idea.
4. Start each body paragraph with a clear topic sentence.
5. If a paragraph exceeds 100 words, split it into two paragraphs.
6. You may synthesize ideas across multiple context chunks, even if sources don't reference each other.
7. Only state "The provided context does not contain sufficient information" if you genuinely cannot construct ANY relevant answer. NEVER append this disclaimer after a substantive answer.
8. Stop generating immediately after completing your answer.
9. Do NOT include meta-commentary, self-evaluations, filler phrases, or sign-offs.
10. Do NOT end with speculative hedges or "future possibilities" not grounded in the context."""

_CITATION_RULES = """
CITATION REQUIREMENTS (MANDATORY):
- Context chunks are numbered [CHUNK 1 | SOURCE: ...], [CHUNK 2 | SOURCE: ...], etc.
- You MUST cite every factual claim using the chunk number: [1], [2], [3], etc.
- Place the citation immediately after the sentence it supports, before the period when possible.
- Example: "The unemployment rate rose to 5.2% [1]. Housing starts declined by 12% [2]."
- If multiple chunks support one claim, cite all of them: [1][3].
- Do NOT omit citations. Every statement derived from the context MUST have at least one [N] marker."""


# ---------------------------------------------------------------------------
# Intent-specific instructions
# ---------------------------------------------------------------------------

INTENT_INSTRUCTIONS_REGULAR: dict[Intent, dict[str, str]] = {
    Intent.OVERVIEW: {
        "task": (
            "Provide a brief, high-level description of this document's type and purpose. "
            "Do NOT describe detailed findings, specific arguments, or examples."
        ),
        "format": (
            "Your first sentence MUST state what type of document this is and its primary purpose. "
            "Write 2-3 sentences total (maximum 60 words). "
            "Do NOT use bullet points."
        ),
        "tone": "Neutral and concise.",
    },
    
    Intent.SUMMARIZE: {
        "task": (
            "Extract the main claims and findings from the context. "
            "Merge overlapping points. Report only what the document states."
        ),
        "format": (
            "Start with one sentence (20-30 words) identifying the document. "
            "Then list 3-5 key points as bullet points. "
            "Each bullet should be one direct sentence (15-25 words) capturing one core idea. "
            "Do NOT add commentary or interpretation."
        ),
        "tone": "Academic but accessible.",
    },
    
    Intent.EXPLAIN: {
        "task": (
            "Explain the content as if teaching someone curious but with no background in this field. "
            "Use everyday language. Avoid all jargon and technical terms. "
            "Do NOT introduce facts, definitions, or topics not present in the context."
        ),
        "format": (
            "Write 3-5 short paragraphs (3-5 sentences each, maximum 80 words per paragraph). "
            "Each paragraph should explain ONE concept or idea. "
            "\n\n"
            "Structure:\n"
            "• Opening paragraph: Introduce the core concept in simple terms.\n"
            "• Middle paragraphs (2-3): Develop the explanation. Each paragraph = one sub-idea.\n"
            "• Final paragraph: Provide a concrete takeaway or practical implication.\n"
            "\n"
            "CRITICAL: Include at least one analogy or everyday comparison to make the key idea concrete. "
            "If a sentence feels too complex, break it into two shorter sentences. "
            "Stop after your final clarifying sentence — no wrap-up or meta-commentary."
        ),
        "tone": "Conversational and clear.",
    },
    
    Intent.ANALYZE: {
        "task": (
            "Explain the core themes, patterns, or conflicts in the context. "
            "Go beyond description — explain WHY things are the way they are. "
            "Highlight where ideas converge and where they diverge. "
            "Cover a range of DISTINCT points — do not revisit the same example or argument more than once."
        ),
        "format": (
            "Write 6-8 short, focused paragraphs (60-100 words each). Use blank lines between paragraphs. "
            "\n\n"
            "Structure:\n"
            "• Opening paragraph: Frame the topic and preview 2-3 key themes you'll analyze.\n"
            "• Body paragraphs (4-6): Each paragraph develops ONE distinct theme or pattern. "
            "Start each with a topic sentence that signals what this paragraph explores. "
            "Use specific evidence from the context. "
            "Use transitions between paragraphs (\"Building on this,\" \"A second theme,\" \"This tension reveals\").\n"
            "• Closing paragraph: Synthesize what the analysis reveals, grounded in the context.\n"
            "\n"
            "CRITICAL RULES:\n"
            "• Each paragraph = 60-100 words maximum. If a point needs more, split into two paragraphs.\n"
            "• Start body paragraphs with topic sentences.\n"
            "• Do NOT repeat points, examples, or evidence across paragraphs.\n"
            "• Do NOT use bullet points — write in prose."
        ),
        "tone": "Analytical, objective, and scholarly.",
    },
    
    Intent.COMPARE: {
        "task": (
            "Identify the two or more ideas, positions, theories, or documents being compared. "
            "For each, state the core claim or position clearly. "
            "Then systematically map: (1) where they agree or overlap, "
            "(2) where they diverge or conflict, (3) what drives the differences "
            "(e.g., differing assumptions, methods, or goals)."
        ),
        "format": (
            "Write 6-8 short, focused paragraphs (60-100 words each). Use blank lines between paragraphs. "
            "\n\n"
            "Structure:\n"
            "• Opening paragraph: State what is being compared and the axis of comparison. "
            "Name the items clearly in your first sentence.\n"
            "• Shared ground (2-3 paragraphs): Explain where they converge. "
            "Each paragraph = ONE point of convergence. "
            "Start each with a topic sentence. Use evidence from the context.\n"
            "• Key differences (2-3 paragraphs): Explain where they diverge. "
            "Each paragraph = ONE point of divergence. "
            "Use transitions (\"In contrast,\" \"However,\" \"The divergence stems from\"). "
            "Explain what drives each difference.\n"
            "• Closing paragraph: What the comparison reveals, grounded in the context.\n"
            "\n"
            "CRITICAL RULES:\n"
            "• Each paragraph = 60-100 words maximum. If explaining a complex point, split it.\n"
            "• Start body paragraphs with topic sentences.\n"
            "• Do NOT repeat points or examples across paragraphs.\n"
            "• Do NOT use bullet points — write in prose."
        ),
        "tone": "Balanced, precise, and scholarly.",
    },
    
    Intent.CRITIQUE: {
        "task": (
            "Identify the central argument or claim the user is asking about. "
            "Report how the text itself evaluates, defends, or challenges that argument. "
            "Surface any critiques, objections, limitations, or counterpoints the text raises. "
            "If the text is one-sided, say so — do NOT invent counterarguments not present in the context. "
            "You may note what the text leaves unaddressed, but label it as an omission, not a flaw. "
            "Present a range of DISTINCT arguments — do not revisit the same point more than once."
        ),
        "format": (
            "Write 5-7 short, focused paragraphs (60-100 words each). Use blank lines between paragraphs. "
            "\n\n"
            "Structure:\n"
            "• Opening paragraph: State the argument being examined.\n"
            "• Strengths (2-3 paragraphs): How the text supports or defends the argument. "
            "Each paragraph = one distinct line of support. "
            "Start each with a topic sentence (\"The text's first defense is...\").\n"
            "• Limitations (1-2 paragraphs): Critiques or counterpoints the text itself raises (if any). "
            "Use transitions (\"However,\" \"A noted limitation,\" \"The text acknowledges\").\n"
            "• Closing paragraph: Summarize the text's own position on the argument.\n"
            "\n"
            "CRITICAL RULES:\n"
            "• Each paragraph = 60-100 words maximum. One point per paragraph.\n"
            "• Start body paragraphs with topic sentences.\n"
            "• Do NOT invent critiques not in the context.\n"
            "• Do NOT repeat points or examples.\n"
            "• Do NOT use bullet points — write in prose."
        ),
        "tone": "Evaluative, text-grounded, and intellectually rigorous.",
    },
    
    Intent.FACTUAL: {
        "task": (
            "Answer the user's question directly and concisely using ONLY the provided context. "
            "Extract the specific fact, name, date, or detail the question asks for. "
            "If the answer is explicitly stated, quote or paraphrase the relevant passage. "
            "Do NOT provide analysis, background, or tangential information."
        ),
        "format": (
            "Give the direct answer in 1-3 sentences (maximum 60 words). "
            "If helpful, include a brief quote from the context. "
            "Do NOT use bullet points. Do NOT provide additional context beyond what is asked."
        ),
        "tone": "Direct, precise, and factual.",
    },
    
    Intent.COLLECTION: {
        "task": (
            "Describe the documents available in this collection based on the provided summaries. "
            "Identify the topics, themes, and scope of the corpus as a whole. "
            "Highlight how the documents relate to each other, if applicable."
        ),
        "format": (
            "Write 3-5 short paragraphs (60-80 words each). "
            "\n\n"
            "Structure:\n"
            "• Opening paragraph: Describe the overall scope and focus of the collection.\n"
            "• Middle paragraphs (1-3): Briefly describe each major document or topic cluster. "
            "One paragraph per document or theme.\n"
            "• Closing paragraph: Note common themes or connections between documents.\n"
            "\n"
            "Do NOT use bullet points. Write in prose."
        ),
        "tone": "Informative and concise.",
    },
}

# ---------------------------------------------------------------------------
# Deep Research mode (currently mirrors regular)
# ---------------------------------------------------------------------------

INTENT_INSTRUCTIONS_DEEP_RESEARCH: dict[Intent, dict[str, str]] = {
    intent: {k: v for k, v in cfg.items()}
    for intent, cfg in INTENT_INSTRUCTIONS_REGULAR.items()
}

# Backward-compatible alias
INTENT_INSTRUCTIONS = INTENT_INSTRUCTIONS_REGULAR


def _get_intent_instructions(mode: Optional[str] = None) -> dict[Intent, dict[str, str]]:
    """Return the intent instruction set for the given operating mode."""
    if mode == "power-deep-research":
        return INTENT_INSTRUCTIONS_DEEP_RESEARCH
    return INTENT_INSTRUCTIONS_REGULAR


def _build_system_block(cfg: dict[str, str], citation_block: str, extra_block: str) -> str:
    return f"{_SYSTEM_MESSAGE}{citation_block}\n\nTask: {cfg['task']}\nFormat: {cfg['format']}\nTone: {cfg['tone']}{extra_block}"


def _prepare_config(
    intent: Optional[Intent],
    citations_enabled: bool,
    extra_instructions: Optional[str],
    mode: Optional[str] = None
) -> tuple[dict[str, str], str, str]:
    intent = intent or Intent.OVERVIEW
    instructions = _get_intent_instructions(mode)
    cfg = instructions.get(intent, instructions[Intent.OVERVIEW]).copy()
    extra_block = (
        f"\nAdditional constraints: {extra_instructions.strip()}"
        if extra_instructions and extra_instructions.strip()
        else ""
    )
    citation_block = ""
    if citations_enabled:
        citation_block = f"\n{_CITATION_RULES}"
        cfg["format"] += " Include numbered inline citations [1], [2], etc. after every factual claim."
    return cfg, citation_block, extra_block


def build_messages(
    context: str,
    question: str,
    intent: Optional[Intent] = None,
    extra_instructions: Optional[str] = None,
    citations_enabled: bool = False,
    source_legend: Optional[str] = None,
    mode: Optional[str] = None,
) -> list[dict[str, str]]:
    """Build intent-aware chat messages for the LLM."""
    cfg, citation_block, extra_block = _prepare_config(
        intent, citations_enabled, extra_instructions, mode=mode
    )
    system_block = _build_system_block(cfg, citation_block, extra_block)
    legend_block = f"\n\n{source_legend}" if citations_enabled and source_legend else ""
    user_block = f"Context:\n{context}{legend_block}\n\nQuestion: {question}\n\nAnswer:"
    return [
        {"role": "system", "content": system_block},
        {"role": "user", "content": user_block}
    ]
