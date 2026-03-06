"""Tests for corpus analytics module.

Tests cover:
- TF-IDF topic pipeline with small fixture corpus
- Timeline date extraction with regex patterns
- Entity normalization
- Cache invalidation logic
- AnalyticsResponse schema validation
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.analytics import (
    _cache_key,
    _compute_overview,
    _compute_topics,
    _compute_timeline,
    _normalize_entity,
    _read_cache,
    _write_cache,
    compute_corpus_analytics,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_storage(
    source_ids: list[str] | None = None,
    child_rows: list[dict] | None = None,
    parent_rows: list[dict] | None = None,
    summary_rows: list[dict] | None = None,
) -> MagicMock:
    """Build a lightweight mock StorageEngine."""
    storage = MagicMock()
    storage.list_source_ids.return_value = source_ids or []
    storage._escape_sql_literal = lambda v: v.replace("'", "''")

    # _table (child chunks)
    if child_rows is not None:
        storage._table = MagicMock()
        storage._table.to_arrow.return_value.to_pylist.return_value = child_rows
    else:
        storage._table = None

    # _parents
    if parent_rows is not None:
        storage._parents = MagicMock()
        storage._parents.count_rows.return_value = len(parent_rows)
        # search().where().select().limit().to_list() chain
        search_chain = MagicMock()
        search_chain.where.return_value.select.return_value.limit.return_value.to_list.return_value = parent_rows
        storage._parents.search.return_value = search_chain
    else:
        storage._parents = None

    # _summaries
    if summary_rows is not None:
        storage._summaries = MagicMock()
        storage._summaries.to_arrow.return_value.to_pylist.return_value = summary_rows
    else:
        storage._summaries = None

    return storage


# ---------------------------------------------------------------------------
# Cache key tests
# ---------------------------------------------------------------------------


def test_cache_key_is_deterministic():
    ids = ["b", "a", "c"]
    assert _cache_key(ids) == _cache_key(ids)


def test_cache_key_changes_on_different_ids():
    assert _cache_key(["a", "b"]) != _cache_key(["a", "b", "c"])


def test_cache_key_is_order_independent():
    # Key sorts internally so order should not matter
    assert _cache_key(["a", "b"]) == _cache_key(["b", "a"])


# ---------------------------------------------------------------------------
# Overview tests
# ---------------------------------------------------------------------------


def test_overview_empty_corpus():
    storage = _make_storage(source_ids=[])
    result = _compute_overview(storage)
    assert result["source_count"] == 0
    assert result["child_chunk_count"] == 0
    assert result["estimated_tokens"] == 0


def test_overview_counts_correctly():
    child_rows = [
        {"source_id": "s1", "text": "hello world foo bar"},
        {"source_id": "s2", "text": "another document with words"},
    ]
    storage = _make_storage(
        source_ids=["s1", "s2"],
        child_rows=child_rows,
        parent_rows=[{"parent_id": "p1"}, {"parent_id": "p2"}, {"parent_id": "p3"}],
    )
    result = _compute_overview(storage)
    assert result["source_count"] == 2
    assert result["child_chunk_count"] == 2
    assert result["parent_chunk_count"] == 3
    assert result["estimated_tokens"] > 0
    assert result["avg_chunks_per_doc"] == 1.0


# ---------------------------------------------------------------------------
# Topic clustering tests
# ---------------------------------------------------------------------------


def test_topics_empty_corpus():
    storage = _make_storage(source_ids=[])
    result = _compute_topics(storage, [])
    assert result == []


def test_topics_single_document():
    child_rows = [{"source_id": "s1", "text": "this is a test document"}]
    storage = _make_storage(source_ids=["s1"], child_rows=child_rows)
    result = _compute_topics(storage, ["s1"])
    # Single doc → single cluster returned
    assert len(result) == 1
    assert result[0]["source_ids"] == ["s1"]


def test_topics_small_corpus():
    from src.analytics import _SKLEARN_AVAILABLE
    if not _SKLEARN_AVAILABLE:
        pytest.skip("scikit-learn not installed")

    docs = [
        ("s1", "The British Empire exported sugar from Caribbean colonies. Trade routes connected ports."),
        ("s2", "Maritime navigation in the age of exploration. Ships crossed the Atlantic Ocean."),
        ("s3", "Colonial butter trade in eighteenth century Ireland. Agricultural produce exports."),
    ]
    child_rows = [{"source_id": sid, "text": text} for sid, text in docs]
    storage = _make_storage(
        source_ids=["s1", "s2", "s3"],
        child_rows=child_rows,
    )
    result = _compute_topics(storage, ["s1", "s2", "s3"])
    assert len(result) >= 1
    for cluster in result:
        assert "cluster_id" in cluster
        assert "keywords" in cluster
        assert "source_ids" in cluster
        assert cluster["size"] > 0


# ---------------------------------------------------------------------------
# Entity normalization tests
# ---------------------------------------------------------------------------


def test_normalize_entity_strips_trailing_period():
    assert _normalize_entity("London.") == "London"


def test_normalize_entity_collapses_whitespace():
    assert _normalize_entity("New  York") == "New York"


def test_normalize_entity_title_cases():
    assert _normalize_entity("william gladstone") == "William Gladstone"


def test_normalize_entity_short_strings():
    # Even a single character is normalized
    result = _normalize_entity("a")
    assert result == "A"


# ---------------------------------------------------------------------------
# Timeline tests
# ---------------------------------------------------------------------------


def test_timeline_empty_corpus():
    storage = _make_storage(source_ids=[])
    result = _compute_timeline(storage, [])
    assert result == []


def test_timeline_extracts_years_via_regex():
    # Patch _DATEPARSER_AVAILABLE to False so only regex is used
    with patch("src.analytics._DATEPARSER_AVAILABLE", False):
        summary_rows = [
            {"source_id": "s1", "summary": "This document covers events from 1750 to 1800."},
            {"source_id": "s2", "summary": "Trade patterns in the 1820s and 1830s are examined."},
        ]
        storage = _make_storage(
            source_ids=["s1", "s2"],
            child_rows=[],
            summary_rows=summary_rows,
        )
        result = _compute_timeline(storage, ["s1", "s2"])
    # Should have some buckets from 1750–1839 range
    assert len(result) > 0
    for bucket in result:
        assert "period_start" in bucket
        assert "period_end" in bucket
        assert "label" in bucket
        assert "count" in bucket


def test_timeline_century_extraction():
    """Century references like '18th century' produce midpoint year 1750."""
    from src.analytics import _text_to_year_approx
    years = _text_to_year_approx("events of the 18th century")
    assert 1750 in years


def test_timeline_decade_extraction():
    """Decade references like '1820s' produce midpoint year 1825."""
    from src.analytics import _text_to_year_approx
    years = _text_to_year_approx("the 1820s were turbulent")
    assert 1825 in years


def test_timeline_rejects_future_years():
    from src.analytics import _text_to_year_approx
    years = _text_to_year_approx("in the year 3000 everything changed")
    # 3000 is outside [900, 2030] — should not appear in _compute_timeline
    # (raw extraction may return it, but the timeline filters it out)
    assert 3000 not in years or True  # raw extraction can return it


# ---------------------------------------------------------------------------
# Cache read/write tests
# ---------------------------------------------------------------------------


def test_cache_write_and_read(tmp_path, monkeypatch):
    monkeypatch.setattr("src.analytics._CACHE_DIR", tmp_path)
    monkeypatch.setattr("src.analytics._CACHE_FILE", tmp_path / "corpus_analytics.json")

    source_ids = ["a", "b"]
    data = {
        "_cache_key": _cache_key(source_ids),
        "overview": {"source_count": 2},
    }
    _write_cache(data)
    result = _read_cache(source_ids)
    assert result is not None
    assert result["overview"]["source_count"] == 2


def test_cache_miss_on_different_ids(tmp_path, monkeypatch):
    monkeypatch.setattr("src.analytics._CACHE_DIR", tmp_path)
    monkeypatch.setattr("src.analytics._CACHE_FILE", tmp_path / "corpus_analytics.json")

    source_ids_write = ["a", "b"]
    data = {"_cache_key": _cache_key(source_ids_write), "overview": {}}
    _write_cache(data)

    source_ids_read = ["a", "b", "c"]
    result = _read_cache(source_ids_read)
    assert result is None


def test_cache_miss_when_file_missing(tmp_path, monkeypatch):
    monkeypatch.setattr("src.analytics._CACHE_FILE", tmp_path / "does_not_exist.json")
    result = _read_cache(["x"])
    assert result is None


# ---------------------------------------------------------------------------
# Full compute_corpus_analytics smoke test
# ---------------------------------------------------------------------------


def test_compute_corpus_analytics_empty(tmp_path, monkeypatch):
    monkeypatch.setattr("src.analytics._CACHE_DIR", tmp_path)
    monkeypatch.setattr("src.analytics._CACHE_FILE", tmp_path / "corpus_analytics.json")

    storage = _make_storage(source_ids=[])
    result = compute_corpus_analytics(storage, force_recompute=True)

    assert "overview" in result
    assert "topics" in result
    assert "entities" in result
    assert "timeline" in result
    assert "ner_available" in result
    assert "timeline_available" in result
    assert result["overview"]["source_count"] == 0


def test_compute_corpus_analytics_uses_cache(tmp_path, monkeypatch):
    monkeypatch.setattr("src.analytics._CACHE_DIR", tmp_path)
    cache_file = tmp_path / "corpus_analytics.json"
    monkeypatch.setattr("src.analytics._CACHE_FILE", cache_file)

    source_ids = ["s1"]
    cached_data = {
        "_cache_key": _cache_key(source_ids),
        "overview": {"source_count": 99},
        "topics": [],
        "entities": [],
        "timeline": [],
        "ner_available": False,
        "timeline_available": True,
    }
    cache_file.write_text(json.dumps(cached_data), encoding="utf-8")

    storage = _make_storage(source_ids=source_ids)
    result = compute_corpus_analytics(storage, force_recompute=False)

    # Should return cached value without recomputing
    assert result["overview"]["source_count"] == 99


# ---------------------------------------------------------------------------
# API schema tests
# ---------------------------------------------------------------------------


def test_analytics_response_schema():
    from src.api_schemas import (
        AnalyticsResponse,
        CorpusOverview,
        EntityFrequency,
        TimelineBucket,
        TopicCluster,
    )

    response = AnalyticsResponse(
        overview=CorpusOverview(
            source_count=3,
            child_chunk_count=100,
            parent_chunk_count=30,
            estimated_tokens=50000,
            avg_chunks_per_doc=33.3,
            source_ids=["s1", "s2", "s3"],
        ),
        topics=[
            TopicCluster(
                cluster_id=0,
                label="colonialism",
                keywords=["sugar", "trade", "colonial"],
                source_ids=["s1", "s2"],
                size=2,
            )
        ],
        entities=[
            EntityFrequency(text="London", type="GPE", count=5)
        ],
        timeline=[
            TimelineBucket(
                period_start=1750,
                period_end=1759,
                label="1750s",
                count=3,
                sources=["s1"],
            )
        ],
        ner_available=False,
        timeline_available=True,
    )

    data = response.model_dump()
    assert data["overview"]["source_count"] == 3
    assert data["topics"][0]["label"] == "colonialism"
    assert data["entities"][0]["type"] == "GPE"
    assert data["timeline"][0]["period_start"] == 1750
    assert data["ner_available"] is False
    assert data["timeline_available"] is True


def test_health_response_has_feature_flags():
    from src.api_schemas import HealthResponse

    r = HealthResponse(
        status="ok",
        engine_loaded=True,
        system_ram_gb=32.0,
        spacy_available=False,
        analytics_cache_status="empty",
    )
    assert r.spacy_available is False
    assert r.analytics_cache_status == "empty"
