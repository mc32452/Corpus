"""Tests for unified LanceDB storage engine."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import lancedb
import pytest

from src.models import ChildChunk, Metadata, ParentChunk
from src.storage import StorageConfig, StorageEngine
from tests.conftest import (
    MockEmbeddingModel,
    Timer,
    generate_parent_child_corpus,
    get_test_logger,
)

logger = get_test_logger("storage")


# ===========================================================================
# Parent store (LanceDB)
# ===========================================================================

class TestParentStore:
    def test_add_and_retrieve_parent(self, tmp_storage: StorageEngine):
        """Stored parent text should be retrievable by parent_id."""
        sources = tmp_storage.list_source_ids()
        assert len(sources) > 0, "Should have source_ids after ingest"
        texts = tmp_storage.get_parent_texts_by_source(source_id=sources[0])
        assert len(texts) > 0, "Should have parent texts for first source"

    def test_missing_parent_returns_none(self, tmp_storage: StorageEngine):
        assert tmp_storage.get_parent_text("nonexistent-id") is None

    def test_list_source_ids(self, tmp_storage: StorageEngine):
        sources = tmp_storage.list_source_ids()
        assert "test_doc_linguistics" in sources
        assert "test_doc_philosophy" in sources

    def test_get_parent_texts_by_source(self, tmp_storage: StorageEngine):
        texts = tmp_storage.get_parent_texts_by_source(source_id="test_doc_linguistics")
        assert len(texts) > 0

    def test_get_parent_text_handles_quotes_in_id(self, tmp_path: Path):
        config = StorageConfig(lance_dir=tmp_path / "lance")
        engine = StorageEngine(config)
        quoted_parent = ParentChunk(
            id="parent-'quoted'",
            text="Quoted parent text",
            metadata=Metadata(
                source_id="doc-1",
                page_number=1,
                page_label="1",
                display_page="1",
                header_path="Document",
                parent_id=None,
            ),
        )
        engine.add_parents([quoted_parent])
        assert engine.get_parent_text("parent-'quoted'") == "Quoted parent text"

    def test_get_parent_texts_batch(self, tmp_storage: StorageEngine, mock_embedder: MockEmbeddingModel):
        q_vec = mock_embedder.encode(["Chomsky theory epistemology"], normalize_embeddings=True)[0]
        hits = tmp_storage.hybrid_search(query_text="Chomsky theory epistemology", query_vector=q_vec, top_k=5)
        parent_ids = [r["metadata"].get("parent_id") for r in hits if r.get("metadata")]
        parent_ids = [pid for pid in parent_ids if isinstance(pid, str)]
        assert parent_ids

        found = tmp_storage.get_parent_texts(parent_ids + ["nonexistent-id"])
        for parent_id in parent_ids:
            assert parent_id in found
        assert "nonexistent-id" not in found


# ===========================================================================
# Child chunk store (LanceDB vectors + FTS)
# ===========================================================================

class TestChildStore:
    def test_hybrid_search_returns_results(self, tmp_storage: StorageEngine, mock_embedder: MockEmbeddingModel):
        """Hybrid search should return child chunks."""
        q_vec = mock_embedder.encode(["Chomsky grammar"], normalize_embeddings=True)[0]
        results = tmp_storage.hybrid_search(
            query_text="Chomsky grammar", query_vector=q_vec, top_k=5,
        )
        assert len(results) > 0

    def test_hybrid_search_top_k_respected(self, tmp_storage: StorageEngine, mock_embedder: MockEmbeddingModel):
        """Returned results should not exceed top_k."""
        q_vec = mock_embedder.encode(["test query"], normalize_embeddings=True)[0]
        results = tmp_storage.hybrid_search(
            query_text="test query", query_vector=q_vec, top_k=2,
        )
        assert len(results) <= 2

    def test_hybrid_search_returns_metadata(self, tmp_storage: StorageEngine, mock_embedder: MockEmbeddingModel):
        """Each result should include metadata with source_id."""
        q_vec = mock_embedder.encode(["Chomsky grammar"], normalize_embeddings=True)[0]
        results = tmp_storage.hybrid_search(
            query_text="Chomsky grammar", query_vector=q_vec, top_k=3,
        )
        for r in results:
            assert "source_id" in r["metadata"]

    def test_get_children_by_ids(self, tmp_storage: StorageEngine, mock_embedder: MockEmbeddingModel):
        """get_children_by_ids should return text and metadata."""
        q_vec = mock_embedder.encode(["test"], normalize_embeddings=True)[0]
        search_results = tmp_storage.hybrid_search(
            query_text="test", query_vector=q_vec, top_k=3,
        )
        ids = [r["id"] for r in search_results]
        fetched = tmp_storage.get_children_by_ids(ids)
        assert len(fetched) == len(ids)
        for cid, data in fetched.items():
            assert "text" in data
            assert "metadata" in data

    def test_empty_ids_returns_empty(self, tmp_storage: StorageEngine):
        assert tmp_storage.get_children_by_ids([]) == {}

    def test_hybrid_search_deterministic(self, tmp_storage: StorageEngine, mock_embedder: MockEmbeddingModel):
        """Same query should produce same results on repeated calls."""
        q_vec = mock_embedder.encode(["Chomsky language"], normalize_embeddings=True)[0]
        r1 = tmp_storage.hybrid_search(
            query_text="Chomsky language", query_vector=q_vec, top_k=5,
        )
        r2 = tmp_storage.hybrid_search(
            query_text="Chomsky language", query_vector=q_vec, top_k=5,
        )
        assert [r["id"] for r in r1] == [r["id"] for r in r2]

    def test_hybrid_search_source_filter(self, tmp_storage: StorageEngine, mock_embedder: MockEmbeddingModel):
        """Source filter should restrict results to one source."""
        q_vec = mock_embedder.encode(["knowledge epistemology"], normalize_embeddings=True)[0]
        results = tmp_storage.hybrid_search(
            query_text="knowledge epistemology", query_vector=q_vec,
            top_k=10, source_id="test_doc_philosophy",
        )
        for r in results:
            assert r["metadata"].get("source_id") == "test_doc_philosophy"

    def test_hybrid_search_source_filter_handles_quotes(self, tmp_path: Path, mock_embedder: MockEmbeddingModel):
        config = StorageConfig(lance_dir=tmp_path / "lance")
        engine = StorageEngine(config)
        quoted_source = "doc-'quoted'"
        parent = ParentChunk(
            id="p-quoted",
            text="Parent text",
            metadata=Metadata(
                source_id=quoted_source,
                page_number=1,
                page_label="1",
                display_page="1",
                header_path="Document",
                parent_id=None,
            ),
        )
        child = ChildChunk(
            text="quoted source child text",
            metadata=Metadata(
                source_id=quoted_source,
                page_number=1,
                page_label="1",
                display_page="1",
                header_path="Document",
                parent_id=parent.id,
            ),
        )
        engine.add_parents([parent])
        embedding = mock_embedder.encode([child.text], normalize_embeddings=True)
        engine.add_children([child], embeddings=embedding)

        q_vec = mock_embedder.encode(["quoted source child"], normalize_embeddings=True)[0]
        results = engine.hybrid_search(
            query_text="quoted source child",
            query_vector=q_vec,
            top_k=5,
            source_id=quoted_source,
        )
        assert results
        assert all(r["metadata"].get("source_id") == quoted_source for r in results)

    def test_persist_reopen(self, tmp_path: Path, mock_embedder: MockEmbeddingModel):
        """Data should survive close + reopen."""
        config = StorageConfig(lance_dir=tmp_path / "lance")
        engine = StorageEngine(config)
        parents, children = generate_parent_child_corpus()
        engine.add_parents(parents)
        texts = [c.text for c in children]
        embeddings = mock_embedder.encode(texts, normalize_embeddings=True)
        engine.add_children(children, embeddings=embeddings)
        engine.close()

        engine2 = StorageEngine(config)
        sources = engine2.list_source_ids()
        assert len(sources) > 0, "Data should persist across sessions"
        engine2.close()

    def test_range_metadata_round_trip(self, tmp_path: Path, mock_embedder: MockEmbeddingModel):
        config = StorageConfig(lance_dir=tmp_path / "lance")
        engine = StorageEngine(config)

        parent = ParentChunk(
            id="p-range",
            text="[Page 10] Parent context",
            metadata=Metadata(
                source_id="doc-range",
                page_number=10,
                start_page=10,
                end_page=12,
                page_label="10",
                display_page="10",
                header_path="Document",
                parent_id=None,
            ),
        )
        child = ChildChunk(
            id="c-range",
            text="[Page 10] child chunk",
            metadata=Metadata(
                source_id="doc-range",
                page_number=10,
                start_page=10,
                end_page=11,
                page_label="10",
                display_page="10",
                header_path="Document",
                parent_id=parent.id,
            ),
        )

        engine.add_parents([parent])
        embedding = mock_embedder.encode([child.text], normalize_embeddings=True)
        engine.add_children([child], embeddings=embedding)

        fetched = engine.get_children_by_ids([child.id])
        assert child.id in fetched
        meta = fetched[child.id]["metadata"]
        assert meta["start_page"] == 10
        assert meta["end_page"] == 11

        engine.close()

    def test_existing_tables_migrate_range_columns(self, tmp_path: Path):
        lance_dir = tmp_path / "lance"
        db = lancedb.connect(str(lance_dir))

        db.create_table(
            "child_chunks",
            [
                {
                    "id": "c1",
                    "text": "legacy child",
                    "source_id": "doc-legacy",
                    "page_number": 1,
                    "page_label": "1",
                    "display_page": "1",
                    "header_path": "Document",
                    "parent_id": "p1",
                    "vector": [0.1, 0.2, 0.3],
                }
            ],
        )
        db.create_table(
            "parent_chunks",
            [
                {
                    "parent_id": "p1",
                    "source_id": "doc-legacy",
                    "page_number": 1,
                    "page_label": "1",
                    "display_page": "1",
                    "header_path": "Document",
                    "text": "legacy parent",
                }
            ],
        )

        engine = StorageEngine(StorageConfig(lance_dir=lance_dir))
        assert engine._table is not None
        assert engine._parents is not None
        assert "start_page" in engine._table.schema.names
        assert "end_page" in engine._table.schema.names
        assert "start_page" in engine._parents.schema.names
        assert "end_page" in engine._parents.schema.names
        engine.close()


# ===========================================================================
# Source summaries (LanceDB)
# ===========================================================================

class TestSourceSummaries:
    def test_upsert_and_get_summary(self, tmp_storage: StorageEngine):
        tmp_storage.upsert_source_summary(source_id="test_doc", summary="A test summary.")
        summaries = tmp_storage.get_source_summaries()
        assert "test_doc" in summaries
        assert summaries["test_doc"] == "A test summary."

    def test_upsert_overwrites(self, tmp_storage: StorageEngine):
        tmp_storage.upsert_source_summary(source_id="doc_x", summary="v1")
        tmp_storage.upsert_source_summary(source_id="doc_x", summary="v2")
        summaries = tmp_storage.get_source_summaries()
        assert summaries["doc_x"] == "v2"

    def test_empty_source_id_rejected(self, tmp_storage: StorageEngine):
        with pytest.raises(ValueError):
            tmp_storage.upsert_source_summary(source_id="", summary="text")

    def test_empty_summary_rejected(self, tmp_storage: StorageEngine):
        with pytest.raises(ValueError):
            tmp_storage.upsert_source_summary(source_id="doc", summary="")

    def test_upsert_summary_handles_quotes_in_source_id(self, tmp_storage: StorageEngine):
        quoted_source = "doc-'quoted'"
        tmp_storage.upsert_source_summary(source_id=quoted_source, summary="v1")
        tmp_storage.upsert_source_summary(source_id=quoted_source, summary="v2")
        summaries = tmp_storage.get_source_summaries()
        assert summaries[quoted_source] == "v2"


# ===========================================================================
# Latency: index operations
# ===========================================================================

class TestStorageLatency:
    def test_hybrid_search_latency(self, tmp_storage: StorageEngine, mock_embedder: MockEmbeddingModel):
        """Measure hybrid search latency."""
        q_vec = mock_embedder.encode(["Chomsky theory"], normalize_embeddings=True)[0]
        top_k_values = [5, 10, 20]
        for k in top_k_values:
            with Timer("hybrid_search", top_k=k) as t:
                tmp_storage.hybrid_search(
                    query_text="Chomsky theory", query_vector=q_vec, top_k=k,
                )
            logger.info(f"hybrid_search top_k={k}: {t.result.elapsed_ms:.2f}ms")

    def test_parent_lookup_latency(self, tmp_storage: StorageEngine):
        """Measure parent text lookup latency."""
        parents, _ = generate_parent_child_corpus()
        for parent in parents[:3]:
            with Timer("parent_lookup", parent_id=parent.id) as t:
                tmp_storage.get_parent_text(parent.id)
            logger.info(f"parent_lookup: {t.result.elapsed_ms:.2f}ms")
