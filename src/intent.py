"""Intent classification for RAG query processing."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class Intent(Enum):
    OVERVIEW = "overview"
    SUMMARIZE = "summarize"
    EXPLAIN = "explain"
    ANALYZE = "analyze"


@dataclass(frozen=True)
class IntentResult:
    intent: Intent
    confidence: float
    method: str

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")


_INTENT_PATTERNS: dict[Intent, list[re.Pattern]] = {
    Intent.OVERVIEW: [
        re.compile(r"^what\s+is\s+this\s*\??$", re.IGNORECASE),  # Exact "what is this?"
        re.compile(r"\bwhat\s+is\s+(this|the)\s+(paper|text|document|article)\s+(about\s*)?\??", re.IGNORECASE),
        re.compile(r"\bwhat\s+is\s+this\s+about\b", re.IGNORECASE),
        re.compile(r"\btell\s+me\s+about\s+(this|the)\s*(document|paper|text|article)?\b", re.IGNORECASE),
        re.compile(r"\bwhat('s| is)\s+the\s+point\s+of\s+this\b", re.IGNORECASE),
        re.compile(r"\bwhat\s+am\s+i\s+(reading|looking at)\b", re.IGNORECASE),
        re.compile(r"\bgive\s+me\s+(a|the)\s+(gist|overview)\b", re.IGNORECASE),
        re.compile(r"\bquick\s+(overview|summary)\b", re.IGNORECASE),
        re.compile(r"\bin\s+a\s+nutshell\b", re.IGNORECASE),
        re.compile(r"\bwhat\s+is\s+this\s+paper\b", re.IGNORECASE),
    ],
    Intent.EXPLAIN: [
        re.compile(r"\bexplain\b", re.IGNORECASE),
        re.compile(r"\bsimple\s+terms\b", re.IGNORECASE),
        re.compile(r"\bsimplif", re.IGNORECASE),
        re.compile(r"\blayman", re.IGNORECASE),
        re.compile(r"\bnon-expert", re.IGNORECASE),
        re.compile(r"\beasy\s+to\s+understand\b", re.IGNORECASE),
        re.compile(r"\bbreak\s*(it\s*)?down\b", re.IGNORECASE),
        re.compile(r"\bELI5\b", re.IGNORECASE),
        re.compile(r"\bwhat\s+does\s+(this|that|it)\s+mean\b", re.IGNORECASE),
    ],
    Intent.ANALYZE: [
        re.compile(r"\bhow\s+does\b", re.IGNORECASE),
        re.compile(r"\bin\s+what\s+way\b", re.IGNORECASE),
        re.compile(r"\bcompare\b", re.IGNORECASE),
        re.compile(r"\bcritique\b", re.IGNORECASE),
        re.compile(r"\bwhy\b.*\b(controversial|debate|disagree|critic)", re.IGNORECASE),
        re.compile(r"\bcontrovers", re.IGNORECASE),
        re.compile(r"\bcritici[sz]", re.IGNORECASE),
        re.compile(r"\bwhat\s+(are|were)\s+the\s+(criticism|objection|argument)", re.IGNORECASE),
        re.compile(r"\bhow\s+did\s+(people|scholars|critics)\s+react\b", re.IGNORECASE),
        re.compile(r"\bwhat\s+(is|was)\s+the\s+debate\b", re.IGNORECASE),
        re.compile(r"\banalyze\b", re.IGNORECASE),
        re.compile(r"\bevaluate\b", re.IGNORECASE),
        re.compile(r"\bstrengths?\s+and\s+weaknesses?\b", re.IGNORECASE),
        re.compile(r"\bpros?\s+and\s+cons?\b", re.IGNORECASE),
    ],
    Intent.SUMMARIZE: [
        re.compile(r"\bsummari[sz]e\b", re.IGNORECASE),
        re.compile(r"\bdetailed\s+(summary|overview)\b", re.IGNORECASE),
        re.compile(r"\bmain\s+(point|idea|argument|theme)s?\b", re.IGNORECASE),
        re.compile(r"\bkey\s+(point|takeaway|finding)s?\b", re.IGNORECASE),
        re.compile(r"\btl;?dr\b", re.IGNORECASE),
        re.compile(r"\bbullet\s*points?\b", re.IGNORECASE),
        re.compile(r"\blist\s+(the\s+)?(main|key)\b", re.IGNORECASE),
    ],
}

_HEURISTIC_CONFIDENCE = {"strong_match": 0.85, "single_match": 0.70, "weak_match": 0.50}
_TECHNICAL_TERM_HINTS = {"stimulus", "reinforcement", "skinner"}


def _has_technical_terms(query: str) -> bool:
    if re.search(r"['\"`].+?['\"`]", query) or re.search(r"\b[A-Z][a-z]{2,}\b", query):
        return True
    return any(term in query.lower() for term in _TECHNICAL_TERM_HINTS)


def _is_technical_how_why(query: str) -> bool:
    return bool(re.match(r"^\s*(how|why)\b", query, re.IGNORECASE)) and _has_technical_terms(query)


def _classify_heuristic(query: str) -> IntentResult:
    """Classify intent using regex pattern matching."""
    scores: dict[Intent, int] = {intent: 0 for intent in Intent}
    for intent, patterns in _INTENT_PATTERNS.items():
        for pattern in patterns:
            if pattern.search(query):
                scores[intent] += 1

    analyze_bias = _is_technical_how_why(query)
    if analyze_bias:
        scores[Intent.ANALYZE] += 2

    best_intent = max(scores, key=lambda k: scores[k])
    best_score = scores[best_intent]

    if best_score == 0:
        return IntentResult(intent=Intent.OVERVIEW, confidence=0.40, method="fallback")

    matching_intents = [i for i, s in scores.items() if s > 0]
    if len(matching_intents) > 1 and best_score == 1:
        confidence = _HEURISTIC_CONFIDENCE["weak_match"]
    elif best_score >= 2:
        confidence = _HEURISTIC_CONFIDENCE["strong_match"]
    else:
        confidence = _HEURISTIC_CONFIDENCE["single_match"]

    if best_intent == Intent.ANALYZE and analyze_bias:
        confidence = min(0.95, confidence + 0.15)

    return IntentResult(intent=best_intent, confidence=confidence, method="heuristic")


def _build_classification_prompt(query: str) -> str:
    """Build a minimal prompt for LLM-based intent classification."""
    return f"""You are a strict JSON generator.
Return ONLY a single JSON object and nothing else.
No markdown, no code fences, no explanations.

Classify the user's intent into exactly one category.

Categories:
- overview: User wants a brief, high-level description of what the document is and its purpose
- summarize: User wants a detailed summary with key points and bullet points
- explain: User wants the content explained simply, for non-experts
- analyze: User wants analysis, critique, controversy, or evaluation

User query: "{query}"

Respond with ONLY a JSON object in this exact format:
{{"intent": "<overview|summarize|explain|analyze>", "confidence": <0.0-1.0>}}"""


def _parse_llm_response(response: str) -> Optional[Tuple[Intent, float]]:
    """Parse JSON intent classification from LLM response."""
    import json
    response = response.strip()
    if "```" in response:
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
        if match:
            response = match.group(1)

    start, end = response.find("{"), response.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    try:
        data = json.loads(response[start:end + 1])
        intent_map = {"overview": Intent.OVERVIEW, "summarize": Intent.SUMMARIZE, "explain": Intent.EXPLAIN, "analyze": Intent.ANALYZE}
        intent = intent_map.get(data.get("intent", "").lower().strip())
        if intent is None:
            return None
        return (intent, max(0.0, min(1.0, float(data.get("confidence", 0.5)))))
    except (json.JSONDecodeError, ValueError, TypeError):
        return None


class IntentClassifier:
    """Classifies user queries into intent types with LLM or heuristic fallback."""

    def __init__(self, generator: Optional[object] = None, confidence_threshold: float = 0.6, use_llm: bool = True) -> None:
        self._generator = generator
        self._confidence_threshold = confidence_threshold
        self._use_llm = use_llm and generator is not None

    def classify(self, query: str) -> IntentResult:
        """Classify query intent. Falls back to OVERVIEW if confidence < threshold."""
        if not query.strip():
            return IntentResult(intent=Intent.OVERVIEW, confidence=1.0, method="fallback")

        result: Optional[IntentResult] = None
        if self._use_llm and self._generator is not None:
            try:
                result = self._classify_with_llm(query)
            except Exception as e:
                logger.warning(f"LLM classification failed: {e}, falling back to heuristic")

        if result is None:
            result = _classify_heuristic(query)

        if result.confidence < self._confidence_threshold:
            logger.info(f"Intent '{result.intent.value}' confidence {result.confidence:.2f} below threshold, falling back to overview")
            return IntentResult(intent=Intent.OVERVIEW, confidence=result.confidence, method=f"{result.method}+overview_fallback")

        return result
    
    def _classify_with_llm(self, query: str) -> Optional[IntentResult]:
        """Classify using LLM generator."""
        if self._generator is None:
            return None

        from .generator import GenerationConfig
        config = GenerationConfig(max_tokens=50, temperature=0.1, top_p=0.9)
        response = self._generator.generate(_build_classification_prompt(query), config=config)
        parsed = _parse_llm_response(response)

        if parsed is None:
            logger.warning(f"Failed to parse LLM response: {response}")
            return None

        return IntentResult(intent=parsed[0], confidence=parsed[1], method="llm")


def classify_intent(
    query: str,
    generator: Optional[object] = None,
    confidence_threshold: float = 0.6,
    use_llm: bool = True,
) -> IntentResult:
    """Convenience function to classify query intent."""
    return IntentClassifier(generator=generator, confidence_threshold=confidence_threshold, use_llm=use_llm).classify(query)
