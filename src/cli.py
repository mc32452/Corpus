from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from .config import select_model_config
from .generator import MlxGenerator
from .retrieval import RetrievalEngine
from .storage import StorageConfig, StorageEngine


def _dedupe_context(texts: Iterable[str]) -> str:
    seen = set()
    ordered = []
    for text in texts:
        cleaned = text.strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        ordered.append(cleaned)
    return "\n\n".join(ordered)


def run() -> None:
    parser = argparse.ArgumentParser(description="Offline RAG CLI")
    parser.add_argument("query", help="User query")
    parser.add_argument("--sqlite", default="data/context.sqlite", help="SQLite DB path")
    parser.add_argument("--chroma", default="data/chroma", help="Chroma persistence dir")
    parser.add_argument("--bm25", default="data/bm25.json", help="BM25 JSON path")
    parser.add_argument("--collection", default="child_chunks", help="Chroma collection name")
    parser.add_argument("--tier", default=None, help="Override hardware tier")
    parser.add_argument("--model", default="models/llm", help="Path to mlx-lm model")
    args = parser.parse_args()

    config = select_model_config(manual_tier=args.tier)

    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:  # pragma: no cover - dependency runtime
        raise RuntimeError("sentence-transformers is required for embeddings.") from exc

    try:
        from FlagEmbedding import FlagReranker
    except Exception as exc:  # pragma: no cover - dependency runtime
        raise RuntimeError("FlagEmbedding is required for reranking.") from exc

    embedding_model = SentenceTransformer(config.embedding_model, device=config.embedding_device)
    reranker = FlagReranker(config.reranker_model, use_fp16=True)

    storage = StorageEngine(
        StorageConfig(
            sqlite_path=Path(args.sqlite),
            chroma_dir=Path(args.chroma),
            chroma_collection=args.collection,
        )
    )

    bm25_path = Path(args.bm25)
    if bm25_path.exists():
        storage.load_bm25(bm25_path)

    retrieval = RetrievalEngine(storage=storage, embedding_model=embedding_model, reranker=reranker)

    results = retrieval.search(args.query)
    context = _dedupe_context(
        [result.parent_text for result in results if result.parent_text]
    )

    prompt = f"Context:\n{context}\n\nQuestion: {args.query}\nAnswer:"

    generator = MlxGenerator(args.model)
    answer = generator.generate(prompt)
    print(answer)


if __name__ == "__main__":
    run()
