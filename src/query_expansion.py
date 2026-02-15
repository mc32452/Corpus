from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from .generator import GenerationConfig, MlxGenerator
from .intent import Intent

logger = logging.getLogger(__name__)

_GENERIC_NOISE_TERMS = {
    "information",
    "data",
    "details",
    "thing",
    "things",
    "stuff",
    "general",
    "overview",
    "content",
}

_STATIC_TERMS: dict[Intent, list[str]] = {
    Intent.OVERVIEW: ["main theme", "core idea"],
    Intent.SUMMARIZE: ["main argument", "thesis", "conclusion", "key points"],
    Intent.EXPLAIN: ["definition", "mechanism", "example"],
    Intent.ANALYZE: [
        "argument structure",
        "assumptions",
        "evidence",
        "counterargument",
        "implications",
    ],
    Intent.COMPARE: ["compare", "contrast", "difference", "similarity"],
    Intent.CRITIQUE: ["criticism", "weakness", "strength", "objection", "evidence"],
    Intent.FACTUAL: ["specific claim", "named entity", "date", "author"],
    Intent.COLLECTION: [],
}


@dataclass(frozen=True)
class ExpansionPayload:
    keywords: list[str] = field(default_factory=list)
    synonyms: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    exclude: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ExpansionResult:
    expanded_query: str
    static_terms: list[str]
    llm_terms: list[str]
    dropped_terms: list[str]
    exclude_terms: list[str]
    cache_key: str
    cache_hit: bool
    applied_llm: bool
    strategy: str
    latency_ms: float


class QueryExpander:
    """Hybrid static + LLM query expansion with caching and guardrails."""

    def __init__(
        self,
        *,
        llm_model_id: str,
        embedding_model: Any,
        similarity_threshold: float = 0.4,
        max_terms: int = 8,
        llm_terms_cap: int = 5,
        confidence_gate: float = 0.7,
    ) -> None:
        self._llm_model_id = llm_model_id
        self._embedding_model = embedding_model
        self._similarity_threshold = similarity_threshold
        self._max_terms = max_terms
        self._llm_terms_cap = llm_terms_cap
        self._confidence_gate = confidence_gate

        self._generator: Optional[MlxGenerator] = None
        self._cache: dict[str, ExpansionPayload] = {}
        self._query_vectors: dict[str, tuple[Intent, list[float]]] = {}
        self._prefetch: dict[tuple[str, str], ExpansionPayload] = {}

    @staticmethod
    def static_terms(intent: Intent) -> list[str]:
        return list(_STATIC_TERMS.get(intent, []))

    def prefetch(self, query: str, intent: Intent) -> None:
        key = (query.strip(), intent.value)
        if key in self._prefetch:
            return
        payload = self._generate_payload(query, intent)
        self._prefetch[key] = payload

    def expand(
        self,
        *,
        query: str,
        intent: Intent,
        intent_confidence: float,
        enable_llm: bool,
        apply_llm: bool,
    ) -> ExpansionResult:
        start = time.perf_counter()
        base_query = query.strip()
        static_terms = self.static_terms(intent)
        query_vector = self._embed(base_query)
        query_hash = self._hash_vector(query_vector)
        similar_hashes = self._top_similar_hashes(query_vector, intent)
        cache_key = f"{query_hash}:{intent.value}:{'|'.join(similar_hashes)}"

        payload = ExpansionPayload()
        cache_hit = False
        if enable_llm and intent != Intent.COLLECTION:
            if cache_key in self._cache:
                payload = self._cache[cache_key]
                cache_hit = True
            else:
                prefetch_key = (base_query, intent.value)
                if prefetch_key in self._prefetch:
                    payload = self._prefetch.pop(prefetch_key)
                else:
                    payload = self._generate_payload(base_query, intent)
                self._cache[cache_key] = payload

        llm_raw_terms = self._dedupe(payload.keywords + payload.synonyms + payload.entities)
        llm_filtered_terms, dropped = self._semantic_filter(
            query=base_query,
            query_vector=query_vector,
            terms=llm_raw_terms,
            excluded_terms=payload.exclude,
        )

        llm_cap = self._llm_terms_cap
        if intent_confidence < self._confidence_gate:
            llm_cap = min(2, llm_cap)

        remaining_slots = max(0, self._max_terms - len(static_terms))
        llm_cap = min(llm_cap, remaining_slots)
        llm_terms = llm_filtered_terms[:llm_cap]

        final_terms = static_terms + (llm_terms if apply_llm else [])
        final_query = self._append_terms(base_query, final_terms)
        self._query_vectors[query_hash] = (intent, query_vector)

        return ExpansionResult(
            expanded_query=final_query,
            static_terms=static_terms,
            llm_terms=llm_terms,
            dropped_terms=dropped,
            exclude_terms=payload.exclude,
            cache_key=cache_key,
            cache_hit=cache_hit,
            applied_llm=apply_llm and bool(llm_terms),
            strategy="hybrid" if apply_llm else "static+shadow-llm",
            latency_ms=(time.perf_counter() - start) * 1000,
        )

    def _generate_payload(self, query: str, intent: Intent) -> ExpansionPayload:
        try:
            raw = self._generate_llm_payload(query, intent)
            data = self._parse_json(raw)
            return ExpansionPayload(
                keywords=self._clean_terms(data.get("keywords", [])),
                synonyms=self._clean_terms(data.get("synonyms", [])),
                entities=self._clean_terms(data.get("entities", [])),
                exclude=self._clean_terms(data.get("exclude", [])),
            )
        except Exception as exc:
            logger.warning("Query expansion LLM failed: %s", exc)
            return ExpansionPayload()

    def _generate_llm_payload(self, query: str, intent: Intent) -> str:
        generator = self._ensure_generator()
        prompt = (
            f'Given this query: "{query}"\n'
            f"Detected intent: {intent.value}\n\n"
            "Generate 3-5 retrieval enhancement terms as JSON:\n"
            "{\n"
            '  "keywords": ["term1", "term2"],\n'
            '  "synonyms": ["alt1", "alt2"],\n'
            '  "entities": ["Entity1"],\n'
            '  "exclude": ["noise_term"]\n'
            "}\n\n"
            "Only output the JSON, no explanation."
        )
        return generator.generate(
            prompt,
            config=GenerationConfig(
                max_tokens=180,
                temperature=0.0,
                top_p=0.9,
                repetition_penalty=1.05,
            ),
        )

    def _ensure_generator(self) -> MlxGenerator:
        if self._generator is None:
            self._generator = MlxGenerator(self._llm_model_id)
        return self._generator

    def _embed(self, text: str) -> list[float]:
        embeddings = self._embedding_model.encode([text], normalize_embeddings=True)
        first = embeddings[0]
        return first.tolist() if hasattr(first, "tolist") else list(first)

    @staticmethod
    def _hash_vector(vector: list[float]) -> str:
        rounded = ",".join(f"{x:.4f}" for x in vector[:64])
        return hashlib.sha1(rounded.encode("utf-8")).hexdigest()[:16]

    def _top_similar_hashes(
        self,
        query_vector: list[float],
        intent: Intent,
        top_k: int = 3,
    ) -> list[str]:
        scored: list[tuple[str, float]] = []
        for key, (cached_intent, cached_vec) in self._query_vectors.items():
            if cached_intent != intent:
                continue
            scored.append((key, self._cosine(query_vector, cached_vec)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [k for k, _ in scored[:top_k]]

    def _semantic_filter(
        self,
        *,
        query: str,
        query_vector: list[float],
        terms: list[str],
        excluded_terms: list[str],
    ) -> tuple[list[str], list[str]]:
        if not terms:
            return [], []

        excluded = {t.lower() for t in excluded_terms}
        candidates = [
            term for term in terms
            if term.lower() not in excluded and term.lower() not in _GENERIC_NOISE_TERMS
        ]
        if not candidates:
            return [], terms

        term_vectors_raw = self._embedding_model.encode(candidates, normalize_embeddings=True)
        term_vectors = [
            v.tolist() if hasattr(v, "tolist") else list(v)
            for v in term_vectors_raw
        ]

        kept: list[str] = []
        dropped: list[str] = []
        for term, vec in zip(candidates, term_vectors):
            sim = self._cosine(query_vector, vec)
            if sim >= self._similarity_threshold:
                kept.append(term)
            else:
                dropped.append(term)

        dropped.extend([t for t in terms if t not in candidates])
        kept = self._dedupe(kept)
        return kept, self._dedupe(dropped)

    @staticmethod
    def _append_terms(query: str, terms: list[str]) -> str:
        if not terms:
            return query
        return f"{query} {' '.join(terms)}"

    @staticmethod
    def _dedupe(values: list[str]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for value in values:
            key = value.strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(value.strip())
        return out

    @staticmethod
    def _clean_terms(values: Any) -> list[str]:
        if not isinstance(values, list):
            return []
        out: list[str] = []
        for item in values:
            if not isinstance(item, str):
                continue
            text = re.sub(r"\s+", " ", item).strip()
            if text:
                out.append(text)
        return out

    @staticmethod
    def _parse_json(raw: str) -> dict[str, Any]:
        content = raw.strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*", "", content, flags=re.IGNORECASE)
            content = re.sub(r"\s*```$", "", content)
        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if match:
            content = match.group(0)
        data = json.loads(content)
        return data if isinstance(data, dict) else {}

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(y * y for y in b) ** 0.5
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)
