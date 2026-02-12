from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import lancedb
import pyarrow as pa

from .models import ChildChunk, ParentChunk

logger = logging.getLogger(__name__)


@dataclass
class StorageConfig:
    sqlite_path: Path
    lance_dir: Path
    lance_table: str = "child_chunks"


class StorageEngine:
    """Storage engine backed by LanceDB (vectors + FTS) and SQLite (parents + summaries)."""

    def __init__(self, config: StorageConfig) -> None:
        self._config = config
        config.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        config.lance_dir.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(str(config.sqlite_path))
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS parent_chunks (
                parent_id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                page_number INTEGER,
                page_label TEXT,
                display_page TEXT,
                header_path TEXT NOT NULL,
                text TEXT NOT NULL
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS source_summaries (
                source_id TEXT PRIMARY KEY,
                summary TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

        # LanceDB for child chunk vectors + full-text search
        self._db = lancedb.connect(str(config.lance_dir))
        self._table_name = config.lance_table
        self._table: Optional[lancedb.table.Table] = None
        try:
            self._table = self._db.open_table(self._table_name)
            logger.info("Opened existing LanceDB table '%s'", self._table_name)
        except Exception:
            logger.info("LanceDB table '%s' does not exist yet; will be created on first ingest", self._table_name)

    def close(self) -> None:
        self._conn.close()

    def add_parents(self, parents: Iterable[ParentChunk]) -> None:
        rows = [
            (
                parent.id,
                parent.metadata.source_id,
                parent.metadata.page_number,
                parent.metadata.page_label,
                parent.metadata.display_page,
                parent.metadata.header_path,
                parent.text,
            )
            for parent in parents
        ]
        if not rows:
            return
        with self._conn:
            self._conn.executemany(
                """
                INSERT OR REPLACE INTO parent_chunks
                    (parent_id, source_id, page_number, page_label, display_page, header_path, text)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

    def add_children(
        self,
        children: Iterable[ChildChunk],
        *,
        embeddings: Optional[list[list[float]]] = None,
    ) -> None:
        child_list = list(children)
        if not child_list:
            return

        if embeddings is not None and len(embeddings) != len(child_list):
            raise ValueError("Embeddings length must match children length.")

        records: list[dict[str, Any]] = []
        for i, child in enumerate(child_list):
            record: dict[str, Any] = {
                "id": child.id,
                "text": child.text,
                "source_id": child.metadata.source_id,
                "page_number": child.metadata.page_number or 0,
                "page_label": child.metadata.page_label or "",
                "display_page": child.metadata.display_page or "",
                "header_path": child.metadata.header_path,
                "parent_id": child.metadata.parent_id or "",
            }
            if embeddings is not None:
                record["vector"] = embeddings[i]
            records.append(record)

        if self._table is None:
            self._table = self._db.create_table(self._table_name, records)
            # Create FTS index on the text column for hybrid search
            self._table.create_fts_index("text", replace=True)
            logger.info(
                "Created LanceDB table '%s' with %d rows + FTS index",
                self._table_name, len(records),
            )
        else:
            self._table.add(records)
            # Rebuild FTS index to include new data
            self._table.create_fts_index("text", replace=True)
            logger.info("Added %d rows to LanceDB table '%s' + rebuilt FTS index", len(records), self._table_name)

    def get_parent_text(self, parent_id: str) -> Optional[str]:
        cursor = self._conn.execute(
            "SELECT text FROM parent_chunks WHERE parent_id = ?",
            (parent_id,),
        )
        row = cursor.fetchone()
        return row[0] if row else None

    def get_children_by_ids(self, ids: list[str]) -> dict[str, dict[str, object]]:
        """Fetch child chunks by their IDs from LanceDB."""
        if not ids or self._table is None:
            return {}
        # Use SQL filter to fetch by ID list
        id_list = ", ".join(f"'{cid}'" for cid in ids)
        try:
            rows = self._table.search().where(f"id IN ({id_list})").limit(len(ids)).to_list()
        except Exception:
            return {}
        result: dict[str, dict[str, object]] = {}
        for row in rows:
            child_id = row.get("id")
            if child_id:
                meta = {
                    "source_id": row.get("source_id", ""),
                    "page_number": row.get("page_number"),
                    "page_label": row.get("page_label"),
                    "display_page": row.get("display_page"),
                    "header_path": row.get("header_path", ""),
                    "parent_id": row.get("parent_id", ""),
                }
                # Clean up empty/zero values to match Chroma-era behavior
                meta = {k: v for k, v in meta.items() if v is not None and v != "" and v != 0}
                result[child_id] = {"text": row.get("text", ""), "metadata": meta}
        return result

    def hybrid_search(
        self,
        *,
        query_text: str,
        query_vector: list[float],
        top_k: int,
        source_id: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """LanceDB native hybrid search (vector ANN + full-text BM25 with RRF fusion)."""
        if self._table is None:
            raise RuntimeError("LanceDB table is not initialized. Run ingest first.")

        builder = (
            self._table
            .search(query_type="hybrid")
            .vector(query_vector)
            .text(query_text)
            .limit(top_k)
        )
        if source_id:
            builder = builder.where(f"source_id = '{source_id}'", prefilter=True)

        rows = builder.to_list()

        results: list[dict[str, Any]] = []
        for rank, row in enumerate(rows, start=1):
            meta = {
                "source_id": row.get("source_id", ""),
                "page_number": row.get("page_number"),
                "page_label": row.get("page_label"),
                "display_page": row.get("display_page"),
                "header_path": row.get("header_path", ""),
                "parent_id": row.get("parent_id", ""),
            }
            meta = {k: v for k, v in meta.items() if v is not None and v != "" and v != 0}
            results.append({
                "id": row.get("id", ""),
                "text": row.get("text", ""),
                "metadata": meta,
                "score": float(row.get("_relevance_score", 0.0)),
                "rank": rank,
            })
        return results

    def list_source_ids(self) -> list[str]:
        cursor = self._conn.execute(
            "SELECT DISTINCT source_id FROM parent_chunks ORDER BY source_id"
        )
        rows = cursor.fetchall()
        return [row[0] for row in rows if row and row[0]]

    def upsert_source_summary(self, *, source_id: str, summary: str) -> None:
        if not source_id.strip():
            raise ValueError("source_id must be non-empty.")
        if not summary.strip():
            raise ValueError("summary must be non-empty.")
        with self._conn:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO source_summaries (source_id, summary)
                VALUES (?, ?)
                """,
                (source_id.strip(), summary.strip()),
            )

    def get_source_summaries(self) -> dict[str, str]:
        cursor = self._conn.execute(
            "SELECT source_id, summary FROM source_summaries ORDER BY source_id"
        )
        rows = cursor.fetchall()
        return {
            source_id: summary
            for source_id, summary in rows
            if source_id and summary
        }

    def get_parent_texts_by_source(self, *, source_id: str) -> list[str]:
        cursor = self._conn.execute(
            "SELECT text FROM parent_chunks WHERE source_id = ? ORDER BY page_number",
            (source_id,),
        )
        rows = cursor.fetchall()
        return [row[0] for row in rows if row and row[0]]
