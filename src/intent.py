"""Intent classification for RAG query processing."""
from __future__ import annotations

import logging
import math
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
    COMPARE = "compare"
    CRITIQUE = "critique"
    FACTUAL = "factual"
    COLLECTION = "collection"


@dataclass(frozen=True)
class IntentResult:
    intent: Intent
    confidence: float
    method: str

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")


_INTENT_PATTERNS: dict[Intent, list[re.Pattern]] = {
    Intent.COLLECTION: [
        re.compile(r"\bwhat\s+(documents?|docs?|files?|sources?)\s+(do\s+)?(we|i|you)\s+have\b", re.IGNORECASE),
        re.compile(r"\bwhat('s| is)\s+(in\s+)?(here|this\s+(collection|corpus|workspace|library))\b", re.IGNORECASE),
        re.compile(r"\blist\s+(all\s+)?(the\s+)?(documents?|docs?|files?|sources?)\b", re.IGNORECASE),
        re.compile(r"\bwhat\s+are\s+(we|these)\s+(looking\s+at|working\s+with)\b", re.IGNORECASE),
        re.compile(r"\bwhat\s+are\s+(the|all\s+the)\s+(documents?|docs?|files?|sources?)\b", re.IGNORECASE),
        re.compile(r"\bsummari[sz]e\s+(all|every(thing)?|the\s+(whole|entire))\b", re.IGNORECASE),
        re.compile(r"\boverview\s+of\s+(all|every|the\s+entire)\b", re.IGNORECASE),
        re.compile(r"\bwhat('s| is)\s+the\s+(corpus|collection)\s+(about|contain)", re.IGNORECASE),
        re.compile(r"\bshow\s+(me\s+)?(all|the)\s+(documents?|docs?|files?|sources?)\b", re.IGNORECASE),
        re.compile(r"\bgive\s+(me\s+)?(an?\s+)?overview\s+of\s+(all|everything)\b", re.IGNORECASE),
        re.compile(r"\bwhat\s+topics?\s+(are|do)\s+(these|the)\s+(documents?|docs?)\s+cover\b", re.IGNORECASE),
        re.compile(r"\bdescribe\s+(all|the)\s+(documents?|docs?|sources?)\b", re.IGNORECASE),
        re.compile(r"\bwhat\s+are\s+the\s+docs\s+in\s+here\b", re.IGNORECASE),
        re.compile(r"\bshow\s+(me\s+)?all\s+(the\s+)?sources\b", re.IGNORECASE),
        re.compile(r"\bdescribe\s+all\s+(the\s+)?sources\b", re.IGNORECASE),
    ],
    Intent.FACTUAL: [
        re.compile(r"\bwhat\s+particular\b", re.IGNORECASE),
        re.compile(r"\bwhat\s+(specific|exact)\b", re.IGNORECASE),
        re.compile(r"\bwho\s+(is|was|are|were|did|does|wrote|said)\b", re.IGNORECASE),
        re.compile(r"\bwhen\s+(did|was|were|is)\b", re.IGNORECASE),
        re.compile(r"\bwhere\s+(did|was|were|is|does)\b", re.IGNORECASE),
        re.compile(r"\bwhich\s+(specific|particular)?\s*\w+\s+(is|was|are|were|did|does)\b", re.IGNORECASE),
        re.compile(r"\bname\s+the\b", re.IGNORECASE),
        re.compile(r"\bidentify\s+the\b", re.IGNORECASE),
        re.compile(r"\bwhat\s+(year|date|time|number|name|title|author)\b", re.IGNORECASE),
        re.compile(r"\bwhat\s+(is|are|was|were)\s+the\s+(name|title|author|year|date)\b", re.IGNORECASE),
        re.compile(r"\bhow\s+many\b", re.IGNORECASE),
        re.compile(r"\bwhat\s+(does|did)\s+\w+\s+(say|write|argue|claim|state|mention)\s+about\b", re.IGNORECASE),
        re.compile(r"\baccording\s+to\b", re.IGNORECASE),
        re.compile(r"\bwhat\s+(llm|model|algorithm|method|technique|tool|framework)\b", re.IGNORECASE),
        re.compile(r"\bwhat\s+is\s+the\s+(author|writer)\s+(talking|writing|referring)\b", re.IGNORECASE),
        re.compile(r"\bwhat\s+does\s+the\s+(author|writer|text|document|paper)\s+(say|state|claim|argue|mention)\b", re.IGNORECASE),
    ],
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
    Intent.COMPARE: [
        re.compile(r"\bcompare\b", re.IGNORECASE),
        re.compile(r"\bcontrast\b", re.IGNORECASE),
        re.compile(r"\bdiffer(ence|s|ent)?\b", re.IGNORECASE),
        re.compile(r"\bsimilarit(y|ies)\b", re.IGNORECASE),
        re.compile(r"\b(how|in\s+what\s+way)\s+does\b.*\b(relate|compare|differ)\b", re.IGNORECASE),
        re.compile(r"\bversus\b|\bvs\.?\b", re.IGNORECASE),
        re.compile(r"\b(both|each|two|these)\b.*\b(approach|view|theory|position|argument)s?\b", re.IGNORECASE),
        re.compile(r"\bside.by.side\b", re.IGNORECASE),
        re.compile(r"\bhow\s+(is|are|does|do)\b.*\b(like|unlike)\b", re.IGNORECASE),
    ],
    Intent.CRITIQUE: [
        re.compile(r"\bcritique\b", re.IGNORECASE),
        re.compile(r"\bcritici[sz]", re.IGNORECASE),
        re.compile(r"\bevaluate\s+(the|this|that|whether|if|an?)\b", re.IGNORECASE),
        re.compile(r"\bassess\s+(the|this|that|whether|if|an?)\b", re.IGNORECASE),
        re.compile(r"\bwhy\b.*\b(controversial|debate|disagree|critic)", re.IGNORECASE),
        re.compile(r"\bcontrovers", re.IGNORECASE),
        re.compile(r"\bwhat\s+(are|were)\s+the\s+(criticism|objection)s?\b", re.IGNORECASE),
        re.compile(r"\bhow\s+did\s+(people|scholars|critics)\s+react\b", re.IGNORECASE),
        re.compile(r"\bwhat\s+(is|was)\s+the\s+debate\b", re.IGNORECASE),
        re.compile(r"\bstrengths?\s+and\s+weaknesses?\b", re.IGNORECASE),
        re.compile(r"\bpros?\s+and\s+cons?\b", re.IGNORECASE),
        re.compile(r"\bto\s+what\s+extent\b", re.IGNORECASE),
        re.compile(r"\bhow\s+(valid|sound|strong|weak|convincing)\b", re.IGNORECASE),
    ],
    Intent.ANALYZE: [
        re.compile(r"\bhow\s+does\b", re.IGNORECASE),
        re.compile(r"\bin\s+what\s+way\b", re.IGNORECASE),
        re.compile(r"\banalyze\b", re.IGNORECASE),
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

_COMMAND_VERB_INTENTS: dict[re.Pattern, Intent] = {
    re.compile(r"^\s*compare\b", re.IGNORECASE): Intent.COMPARE,
    re.compile(r"^\s*contrast\b", re.IGNORECASE): Intent.COMPARE,
    re.compile(r"^\s*summari[sz]e\b", re.IGNORECASE): Intent.SUMMARIZE,
    re.compile(r"^\s*explain\b", re.IGNORECASE): Intent.EXPLAIN,
    re.compile(r"^\s*analy[sz]e\b", re.IGNORECASE): Intent.ANALYZE,
    re.compile(r"^\s*trace\b", re.IGNORECASE): Intent.ANALYZE,
    re.compile(r"^\s*extract\b", re.IGNORECASE): Intent.FACTUAL,
    re.compile(r"^\s*identify\b", re.IGNORECASE): Intent.FACTUAL,
}

_COMPARATIVE_STRUCTURES: list[re.Pattern] = [
    re.compile(r"\bvs\.?\b", re.IGNORECASE),
    re.compile(r"\bversus\b", re.IGNORECASE),
    re.compile(r"\bdifferences?\s+between\b", re.IGNORECASE),
    re.compile(r"\bsimilarit(y|ies)\s+between\b", re.IGNORECASE),
    re.compile(r"\bin\s+light\s+of\b", re.IGNORECASE),
    re.compile(r"\bbetween\b.+\band\b", re.IGNORECASE),
]

_EXTRACTION_STRUCTURES: list[re.Pattern] = [
    re.compile(r"\bextract\b", re.IGNORECASE),
    re.compile(r"\blist\s+all\s+(names?|dates?|years?|titles?|authors?|citations?)\b", re.IGNORECASE),
    re.compile(r"\bgive\s+me\s+the\s+(dates?|years?|names?|citations?)\b", re.IGNORECASE),
    re.compile(r"\bformat\b.*\bas\s+(a\s+)?table\b", re.IGNORECASE),
    re.compile(r"\btabular\b", re.IGNORECASE),
]

_SUMMARIZATION_STRUCTURES: list[re.Pattern] = [
    re.compile(r"\btl;?dr\b", re.IGNORECASE),
    re.compile(r"\bmain\s+points?\b", re.IGNORECASE),
    re.compile(r"\bkey\s+points?\b", re.IGNORECASE),
    re.compile(r"\boverview\s+of\b", re.IGNORECASE),
]


def _apply_structural_intent_signals(query: str, scores: dict[Intent, int]) -> None:
    """Apply structure-aware intent boosts beyond plain keyword presence."""
    normalized = query.strip()

    corpus_scope = bool(
        re.search(r"\b(all|every|entire|whole)\b", normalized, re.IGNORECASE)
        and re.search(r"\b(documents?|docs?|sources?|collection|corpus)\b", normalized, re.IGNORECASE)
    )
    overview_everything_scope = bool(
        re.search(r"\boverview\s+of\s+everything\b", normalized, re.IGNORECASE)
        or re.search(r"\bsummari[sz]e\s+everything\b", normalized, re.IGNORECASE)
    )

    for pattern, intent in _COMMAND_VERB_INTENTS.items():
        if pattern.search(normalized):
            if intent in (Intent.SUMMARIZE, Intent.FACTUAL) and corpus_scope:
                continue
            scores[intent] += 3

    if any(pattern.search(normalized) for pattern in _COMPARATIVE_STRUCTURES):
        scores[Intent.COMPARE] += 2

    if any(pattern.search(normalized) for pattern in _EXTRACTION_STRUCTURES):
        scores[Intent.FACTUAL] += 3

    if any(pattern.search(normalized) for pattern in _SUMMARIZATION_STRUCTURES):
        scores[Intent.SUMMARIZE] += 2

    if corpus_scope or overview_everything_scope:
        scores[Intent.COLLECTION] += 4

    # Noun-phrase analytical framing: "X's critique of Y" usually asks for analysis
    # of arguments, not an imperative to "critique".
    if re.search(r"\b\w+(?:'s|s)\s+critique\s+of\b", normalized, re.IGNORECASE):
        scores[Intent.ANALYZE] += 2

    # Causal chain phrasing is analytical/multi-hop.
    if re.search(r"\btrace\b.*\b(chain|path|link)\b", normalized, re.IGNORECASE):
        scores[Intent.ANALYZE] += 3
    if "->" in normalized:
        scores[Intent.ANALYZE] += 2


def _has_technical_terms(query: str) -> bool:
    if re.search(r"['\"`].+?['\"`]", query) or re.search(r"\b[A-Z][a-z]{2,}\b", query):
        return True
    return any(term in query.lower() for term in _TECHNICAL_TERM_HINTS)


def _is_technical_how_why(query: str) -> bool:
    # "how many" is a factual question, not analytical
    if re.match(r"^\s*how\s+many\b", query, re.IGNORECASE):
        return False
    return bool(re.match(r"^\s*(how|why)\b", query, re.IGNORECASE)) and _has_technical_terms(query)


def _classify_heuristic(query: str) -> IntentResult:
    """Classify intent using regex pattern matching."""
    scores: dict[Intent, int] = {intent: 0 for intent in Intent}
    for intent, patterns in _INTENT_PATTERNS.items():
        for pattern in patterns:
            if pattern.search(query):
                scores[intent] += 1

    _apply_structural_intent_signals(query, scores)

    # ---- noun-phrase de-boost ----
    # "Chomsky's critique", "chomskys critique", "the critique of X" → the
    # word "critique" is a noun, NOT an instruction to critique.  De-boost
    # CRITIQUE so it doesn't tie with COMPARE / ANALYZE on queries that
    # merely *mention* a critique.
    if scores[Intent.CRITIQUE] > 0:
        noun_critique = re.search(
            r"(?:\b\w+(?:'s|s)\s+|(?:the|a|an|this|that|his|her|their|its)\s+)critique\b",
            query, re.IGNORECASE,
        )
        if noun_critique:
            scores[Intent.CRITIQUE] = max(0, scores[Intent.CRITIQUE] - 1)
            logger.debug(
                "De-boosted CRITIQUE: 'critique' appears as noun phrase (%s)",
                noun_critique.group(),
            )

    analyze_bias = _is_technical_how_why(query)
    if analyze_bias:
        # Boost whichever analytical intent scored highest, or ANALYZE as default
        analytical = [Intent.COMPARE, Intent.CRITIQUE, Intent.ANALYZE]
        best_analytical = max(analytical, key=lambda k: scores[k])
        if any(scores[i] > 0 for i in analytical):
            scores[best_analytical] += 2
        else:
            scores[Intent.ANALYZE] += 2

    best_intent = max(scores, key=lambda k: scores[k])
    best_score = scores[best_intent]

    if best_score == 0:
        return IntentResult(intent=Intent.OVERVIEW, confidence=0.40, method="fallback")

    matching_intents = [i for i, s in scores.items() if s > 0]

    # COMPARE/CRITIQUE should win over generic ANALYZE only when ANALYZE
    # evidence is weak (avoid overriding strong structural analyze signals).
    if best_intent == Intent.ANALYZE and (
        scores[Intent.ANALYZE] <= 1
        and (scores[Intent.COMPARE] > 0 or scores[Intent.CRITIQUE] > 0)
    ):
        if scores[Intent.COMPARE] >= scores[Intent.CRITIQUE]:
            best_intent = Intent.COMPARE
            best_score = scores[Intent.COMPARE]
        else:
            best_intent = Intent.CRITIQUE
            best_score = scores[Intent.CRITIQUE]

    # COLLECTION should win ties with SUMMARIZE when both match
    # (e.g., "Summarize all documents" matches both)
    if Intent.COLLECTION in matching_intents and Intent.SUMMARIZE in matching_intents:
        if scores[Intent.COLLECTION] >= scores[Intent.SUMMARIZE]:
            best_intent = Intent.COLLECTION
            best_score = scores[Intent.COLLECTION]

    # COMPARE should win ties with CRITIQUE when both match at score 1
    # ("What is a similarity in Chomsky's critique" → COMPARE, not CRITIQUE)
    if (
        scores[Intent.COMPARE] > 0
        and scores[Intent.CRITIQUE] > 0
        and scores[Intent.COMPARE] >= scores[Intent.CRITIQUE]
    ):
        best_intent = Intent.COMPARE
        best_score = scores[Intent.COMPARE]
        # Zero out CRITIQUE so the confidence calc doesn't penalise the
        # false tie — we've resolved the ambiguity.
        scores[Intent.CRITIQUE] = 0

    # Recalculate matching intents after all tiebreaks / de-boosts.
    matching_intents = [i for i, s in scores.items() if s > 0]

    if len(matching_intents) > 1 and best_score == 1:
        confidence = _HEURISTIC_CONFIDENCE["weak_match"]
    elif best_score >= 2:
        confidence = _HEURISTIC_CONFIDENCE["strong_match"]
    else:
        confidence = _HEURISTIC_CONFIDENCE["single_match"]

    if best_intent in (Intent.ANALYZE, Intent.COMPARE, Intent.CRITIQUE) and analyze_bias:
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
- compare: User wants a side-by-side comparison or contrast between two or more named ideas, theories, positions, or documents
- critique: User explicitly asks to evaluate, critique, or assess the merits of an argument — uses words like "evaluate", "critique", "strengths", "weaknesses", "how convincing"
- analyze: User wants to understand how or why something works, or wants analysis of themes, patterns, causes, or mechanisms (default for "how does X" and "why does X" questions)
- factual: User wants a direct factual answer extracted from the text (who, what, when, where, which, how many)
- collection: User wants to know what documents are available, or wants an overview of all documents in the corpus

User query: "{query}"

Respond with ONLY a JSON object in this exact format:
{{"intent": "<overview|summarize|explain|compare|critique|analyze|factual|collection>", "confidence": <0.0-1.0>}}"""


def _build_logit_classification_prompt(query: str) -> str:
    """Prompt for logits-based single-label intent scoring."""
    return f"""Classify the user query into exactly one intent label.
Choose one label from: overview, summarize, explain, analyze, compare, critique, factual, collection.

User query: {query}

Intent label:"""


def _build_mc_logit_prompt(query: str) -> str:
    """Prompt for option-token intent classification (A-H)."""
    return f"""Choose exactly one best intent for the query.
Reply with one letter only: A, B, C, D, E, F, G, or H.

A = overview
B = summarize
C = explain
D = analyze
E = compare
F = critique
G = factual
H = collection

Query: {query}
Answer:"""


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
        intent_map = {"overview": Intent.OVERVIEW, "summarize": Intent.SUMMARIZE, "explain": Intent.EXPLAIN, "analyze": Intent.ANALYZE, "compare": Intent.COMPARE, "critique": Intent.CRITIQUE, "factual": Intent.FACTUAL, "collection": Intent.COLLECTION}
        intent = intent_map.get(data.get("intent", "").lower().strip())
        if intent is None:
            return None
        return (intent, max(0.0, min(1.0, float(data.get("confidence", 0.5)))))
    except (json.JSONDecodeError, ValueError, TypeError):
        return None


class IntentClassifier:
    """Classifies user queries into intent types with LLM or heuristic fallback.

    Supports two kinds of LLM backend:

    * **lightweight_generator** – an object with a ``generate_text(prompt, max_tokens)``
      method (e.g. the reranker's 0.6B backbone).  Preferred because it is
      already loaded and adds only ~100-200 ms.
    * **generator** – a full ``MlxGenerator`` instance.  Only used when no
      lightweight generator is supplied and ``use_llm=True``.
    """

    def __init__(
        self,
        generator: Optional[object] = None,
        confidence_threshold: float = 0.6,
        use_llm: bool = True,
        lightweight_generator: Optional[object] = None,
        mode: Optional[str] = None,   # kept for backward compat; unused
    ) -> None:
        self._generator = generator
        self._lightweight_generator = lightweight_generator
        self._confidence_threshold = confidence_threshold
        # Prefer lightweight generator (already loaded, fast).
        if lightweight_generator is not None:
            self._use_llm = True
        else:
            self._use_llm = use_llm and generator is not None

    def classify(self, query: str) -> IntentResult:
        """Classify query intent. Falls back to OVERVIEW if confidence < threshold."""
        if not query.strip():
            return IntentResult(intent=Intent.OVERVIEW, confidence=1.0, method="fallback")

        result: Optional[IntentResult] = None
        if self._use_llm and (self._lightweight_generator is not None or self._generator is not None):
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
        """Classify using LLM generator (lightweight or full)."""
        intent_map = {
            "overview": Intent.OVERVIEW,
            "summarize": Intent.SUMMARIZE,
            "explain": Intent.EXPLAIN,
            "analyze": Intent.ANALYZE,
            "compare": Intent.COMPARE,
            "critique": Intent.CRITIQUE,
            "factual": Intent.FACTUAL,
            "collection": Intent.COLLECTION,
        }

        logits_model = None
        logits_backend = ""
        if self._lightweight_generator is not None and hasattr(self._lightweight_generator, "score_intent_labels"):
            logits_model = self._lightweight_generator
            logits_backend = "llm-light-logits"
        elif self._generator is not None and hasattr(self._generator, "score_intent_labels"):
            logits_model = self._generator
            logits_backend = "llm-dedicated-logits"

        if logits_model is not None:
            try:
                option_to_intent = {
                    "A": Intent.OVERVIEW,
                    "B": Intent.SUMMARIZE,
                    "C": Intent.EXPLAIN,
                    "D": Intent.ANALYZE,
                    "E": Intent.COMPARE,
                    "F": Intent.CRITIQUE,
                    "G": Intent.FACTUAL,
                    "H": Intent.COLLECTION,
                }
                options = list(option_to_intent.keys())

                prompt = _build_mc_logit_prompt(query)
                raw_scores = logits_model.score_intent_labels(prompt, options)
                prior_prompt = _build_mc_logit_prompt("N/A")
                prior_scores = logits_model.score_intent_labels(prior_prompt, options)

                if raw_scores:
                    score_values = {
                        opt: float(metrics.get("avg_logprob", metrics.get("logprob_sum", float("-inf"))))
                        for opt, metrics in raw_scores.items()
                        if opt in option_to_intent
                    }
                    prior_values = {
                        opt: float(metrics.get("avg_logprob", metrics.get("logprob_sum", 0.0)))
                        for opt, metrics in prior_scores.items()
                        if opt in score_values
                    }
                    calibrated_values = {
                        opt: score_values[opt] - prior_values.get(opt, 0.0)
                        for opt in score_values
                    }

                    heuristic = _classify_heuristic(query)
                    heuristic_option = next((opt for opt, intent in option_to_intent.items() if intent == heuristic.intent), None)
                    if heuristic_option in calibrated_values:
                        heuristic_conf = min(max(heuristic.confidence, 1e-4), 1 - 1e-4)
                        heuristic_log_odds = math.log(heuristic_conf / (1.0 - heuristic_conf))
                        calibrated_values[heuristic_option] += 0.25 * heuristic_log_odds

                    if score_values:
                        ordered = sorted(calibrated_values.items(), key=lambda item: item[1], reverse=True)
                        best_option, best_score = ordered[0]
                        second_score = ordered[1][1] if len(ordered) > 1 else float("-inf")

                        max_score = max(calibrated_values.values())
                        exp_scores = {k: math.exp(v - max_score) for k, v in calibrated_values.items()}
                        norm = sum(exp_scores.values())
                        probs = {k: (v / norm if norm > 0 else 0.0) for k, v in exp_scores.items()}

                        top_prob = probs.get(best_option, 0.0)
                        second_prob = 0.0
                        if len(ordered) > 1:
                            second_option = ordered[1][0]
                            second_prob = probs.get(second_option, 0.0)

                        margin = best_score - second_score if second_score != float("-inf") else best_score
                        confidence = 1.0 / (1.0 + math.exp(-4.0 * margin))

                        raw_logit_debug = {
                            opt: round(float(metrics.get("raw_logit_sum", 0.0)), 4)
                            for opt, metrics in raw_scores.items()
                            if opt in score_values
                        }
                        prob_debug = {opt: round(prob, 4) for opt, prob in probs.items()}
                        logger.info(
                            "Intent logits | backend=%s best=%s margin=%.4f raw_logits=%s calibrated=%s probs=%s",
                            logits_backend,
                            best_option,
                            margin,
                            raw_logit_debug,
                            {opt: round(value, 4) for opt, value in calibrated_values.items()},
                            prob_debug,
                        )

                        return IntentResult(
                            intent=option_to_intent[best_option],
                            confidence=max(0.0, min(1.0, confidence)),
                            method=logits_backend,
                        )
            except Exception as e:
                logger.warning("Logits intent classification failed (%s): %s", logits_backend, e)

        prompt = _build_classification_prompt(query)
        response: Optional[str] = None

        if self._lightweight_generator is not None:
            # Fast path: reranker's 0.6B backbone (~100-200 ms)
            try:
                response = self._lightweight_generator.generate_text(
                    prompt, max_tokens=50, temperature=0.1,
                )
            except Exception as e:
                logger.warning("Lightweight LLM intent classification failed: %s", e)
                return None
        elif self._generator is not None:
            # Slow path: full LLM
            from .generator import GenerationConfig
            config = GenerationConfig(max_tokens=50, temperature=0.1, top_p=0.9)
            response = self._generator.generate(prompt, config=config)
        else:
            return None

        if response is None:
            return None

        parsed = _parse_llm_response(response)

        if parsed is None:
            logger.warning("Failed to parse LLM intent response: %s", response[:200])
            return None

        method = "llm-light" if self._lightweight_generator is not None else "llm"
        return IntentResult(intent=parsed[0], confidence=parsed[1], method=method)


def classify_intent(
    query: str,
    generator: Optional[object] = None,
    confidence_threshold: float = 0.6,
    use_llm: bool = True,
    lightweight_generator: Optional[object] = None,
    mode: Optional[str] = None,
) -> IntentResult:
    """Convenience function to classify query intent."""
    return IntentClassifier(
        generator=generator,
        confidence_threshold=confidence_threshold,
        use_llm=use_llm,
        lightweight_generator=lightweight_generator,
    ).classify(query)
