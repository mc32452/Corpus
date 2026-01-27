## Project: Offline Advanced RAG (Apple Silicon Optimized)
**Role:** Senior AI Engineer / Systems Architect
**Objective:** Build a high-performance, offline-first RAG (Retrieval-Augmented Generation) system optimized for Apple Silicon (M-series chips).
**Constraints:** strict offline functionality, highly optimized VRAM usage, Python-based.

### 1. Technology Stack
*   **LLM Inference:** `mlx-lm` (Apple Machine Learning/Metal optimized).
*   **Embeddings:** `sentence-transformers` (Run on CPU to save VRAM).
*   **Re-ranking:** `FlagEmbedding` (BGE-Reranker).
*   **Sparse Search:** `rank_bm25`.
*   **Vector Store:** `chromadb` (Local) or `faiss-cpu`.
*   **Key-Value Store:** `sqlite3` (for Parent Chunks).

### 2. Core Architecture Modules

#### Module A: Data Layer (The Parent-Child Splitter)
Do not use simple character splitting. Implement a hierarchical class structure:
*   **`ParentChunk`**: 1000–1500 tokens. Contains the full "meat" of the text. Stored in SQLite.
*   **`ChildChunk`**: 200–300 tokens. Overlapping slides derived from the Parent. Stored in Vector Index.
*   **`Metadata`**: Every chunk must track: `source_id`, `page_number`, `header_path` (e.g., "Intro > Methodology"), and `parent_id`.
*   **Markdown Awareness**: Use a markdown header splitter to ensure `ChildChunks` inherit the semantic header path of their parent section.

#### Module B: The Storage Engine
*   **Vector Index**: Stores embeddings of `ChildChunks`. Use `int8` quantization if possible for the index.
*   **Lexical Index**: Maintain a BM25 index of the raw text tokens from `ChildChunks` for keyword matching.
*   **Context Store (SQLite)**: A persistent DB mapping `parent_id` -> `full_text_content`.

#### Module C: The Hybrid Retrieval Pipeline (Two-Stage)
Implement a class `RetrievalEngine` with this exact flow:
1.  **Parallel Search**:
    *   *Dense*: Cosine similarity on Vector Index (Top 30).
    *   *Sparse*: BM25 score on Lexical Index (Top 30).
2.  **Fusion (RRF)**: Combine results using Reciprocal Rank Fusion:
    $$Score = \sum (1 / (60 + rank))$$
3.  **Re-Ranking (The Filter)**:
    *   Take top 20 fused results.
    *   Pass query + text pairs through `BGE-Reranker-v2-m3`.
    *   Select Top 5 absolute best matches.
4.  **Context Expansion**:
    *   Map the final Top 5 `ChildChunks` back to their `ParentChunks`.
    *   Return the *Parent* text to the LLM to ensure full context.

#### Module D: Hardware Configuration & Model Loader
Implement a configuration manager `config.py` that detects system RAM or allows manual overrides based on these tiers:

**Tier 1: High Performance (M4 Max / 64GB+)**
*   **LLM**: Qwen 2.5/3 - 32B (Instruct) - 4bit/8bit
*   **Embedding**: BAAI BGE-M3 (Full precision)

**Tier 2: Efficiency (M1/M2/M3 Pro / 32GB)**
*   **LLM**: Mistral Small 3.1 (24B) or Gemma 3 (12B) - 4bit
*   **Embedding**: BAAI BGE-M3 (Quantized)

*Note: All embedding and BM25 operations must be forced to CPU (`device='cpu'`) to reserve Metal/GPU VRAM strictly for the LLM.*

### 3. Implementation Steps for the Agent

1.  **Setup**: Create a clean Python project structure (`/src`, `/data`, `/models`).
2.  **Data Structures**: Define the Pydantic models for `ParentChunk` and `ChildChunk`.
3.  **Ingestion**: Write the Markdown/PDF ingestion script that populates the Vector Store and SQLite.
4.  **Retrieval Logic**: Implement the Hybrid Search + RRF + Reranker pipeline.
5.  **Generation**: Implement the `mlx-lm` generator class that takes the retrieved Parent Context and prompts the specific LLM.
6.  **CLI/Interface**: A simple script to query the system.

***

>