from __future__ import annotations

from src.intent import Intent
from src.query_expansion import QueryExpander
from tests.conftest import MockEmbeddingModel


class _StubQueryExpander(QueryExpander):
    def __init__(self, embedding_model, payload_text: str):
        super().__init__(
            llm_model_id="stub",
            embedding_model=embedding_model,
            similarity_threshold=0.4,
            max_terms=8,
            llm_terms_cap=5,
            confidence_gate=0.7,
        )
        self._payload_text = payload_text

    def _generate_llm_payload(self, query: str, intent: Intent) -> str:
        return self._payload_text


class TestQueryExpansion:
    def test_hybrid_expansion_includes_static_and_llm_terms(self):
        expander = _StubQueryExpander(
            MockEmbeddingModel(),
            '{"keywords": ["poverty of the stimulus"], "synonyms": ["innateness hypothesis"], "entities": ["Chomsky"], "exclude": []}',
        )
        result = expander.expand(
            query="Explain Chomsky's argument",
            intent=Intent.ANALYZE,
            intent_confidence=0.9,
            enable_llm=True,
            apply_llm=True,
        )
        assert result.static_terms
        assert result.applied_llm is True
        assert "argument structure" in result.expanded_query
        assert "Chomsky" in result.expanded_query

    def test_low_confidence_limits_llm_terms(self):
        expander = _StubQueryExpander(
            MockEmbeddingModel(),
            '{"keywords": ["k1", "k2", "k3"], "synonyms": ["s1", "s2"], "entities": [], "exclude": []}',
        )
        result = expander.expand(
            query="Analyze this",
            intent=Intent.ANALYZE,
            intent_confidence=0.4,
            enable_llm=True,
            apply_llm=True,
        )
        assert len(result.llm_terms) <= 2

    def test_shadow_mode_keeps_static_query(self):
        expander = _StubQueryExpander(
            MockEmbeddingModel(),
            '{"keywords": ["semantic term"], "synonyms": [], "entities": [], "exclude": []}',
        )
        result = expander.expand(
            query="Summarize the paper",
            intent=Intent.SUMMARIZE,
            intent_confidence=0.9,
            enable_llm=True,
            apply_llm=False,
        )
        assert result.applied_llm is False
        assert "semantic term" not in result.expanded_query
        assert "main argument" in result.expanded_query

    def test_cache_reuse_for_same_signature(self):
        expander = _StubQueryExpander(
            MockEmbeddingModel(),
            '{"keywords": ["termA"], "synonyms": [], "entities": [], "exclude": []}',
        )
        first = expander.expand(
            query="Compare Skinner and Chomsky",
            intent=Intent.COMPARE,
            intent_confidence=0.9,
            enable_llm=True,
            apply_llm=True,
        )
        second = expander.expand(
            query="Compare Skinner and Chomsky",
            intent=Intent.COMPARE,
            intent_confidence=0.9,
            enable_llm=True,
            apply_llm=True,
        )
        assert first.cache_hit is False
        assert second.cache_hit is True
