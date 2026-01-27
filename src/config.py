from __future__ import annotations

from dataclasses import dataclass
import os
import platform


@dataclass(frozen=True)
class ModelConfig:
    tier: str
    llm_model: str
    embedding_model: str
    reranker_model: str
    embedding_device: str = "cpu"


def _detect_ram_gb() -> float:
    system = platform.system().lower()
    if system == "darwin":
        try:
            import subprocess

            output = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
            return int(output.strip()) / (1024**3)
        except Exception:
            return 0.0
    if hasattr(os, "sysconf"):
        try:
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            return (pages * page_size) / (1024**3)
        except (ValueError, OSError, AttributeError):
            return 0.0
    return 0.0


def select_model_config(*, manual_tier: str | None = None) -> ModelConfig:
    tier = (manual_tier or os.getenv("RAG_TIER", "")).strip().lower()
    if not tier:
        ram_gb = _detect_ram_gb()
        if ram_gb >= 64:
            tier = "high"
        elif ram_gb >= 32:
            tier = "efficiency"
        else:
            tier = "efficiency"

    if tier in {"high", "high-performance", "tier1"}:
        return ModelConfig(
            tier="high",
            llm_model="mlx-community/Llama-3.3-70B-Instruct-4bit",
            embedding_model="BAAI/bge-m3",
            reranker_model="BAAI/bge-reranker-v2-m3",
            embedding_device="cpu",
        )

    if tier in {"efficiency", "tier2"}:
        return ModelConfig(
            tier="efficiency",
            llm_model="mlx-community/Llama-3.3-70B-Instruct-4bit",
            embedding_model="BAAI/bge-m3",
            reranker_model="BAAI/bge-reranker-v2-m3",
            embedding_device="cpu",
        )

    raise ValueError(
        "Unknown tier. Use 'high' or 'efficiency' or set RAG_TIER accordingly."
    )
