from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Optional


@dataclass(frozen=True)
class GenerationConfig:
    max_tokens: Optional[int] = None
    temperature: float = 0.2
    top_p: float = 0.9
    repetition_penalty: Optional[float] = None


class MlxGenerator:
    def __init__(self, model_path: str) -> None:
        self._model_id = model_path
        try:
            from mlx_lm import load
        except Exception as exc:  # pragma: no cover - dependency runtime
            raise RuntimeError("mlx-lm is not available. Install mlx-lm to continue.") from exc

        try:
            self._model, self._tokenizer = load(model_path)
        except Exception as exc:  # pragma: no cover - dependency runtime
            raise RuntimeError(f"Failed to load mlx-lm model at {model_path}.") from exc

    @staticmethod
    def _infer_model_size_b(model_id: str) -> Optional[float]:
        match = re.search(r"(\d+(?:\.\d+)?)\s*[bB]", model_id)
        if not match:
            return None
        try:
            return float(match.group(1))
        except ValueError:
            return None

    @staticmethod
    def _build_repetition_penalty_processor(penalty: float):
        if penalty <= 1.0:
            return None

        def _processor(tokens, logits):
            try:
                import mlx.core as mx
            except Exception:
                return logits

            token_ids = tokens.tolist()
            if token_ids and isinstance(token_ids[0], list):
                token_ids = token_ids[0]
            if not token_ids:
                return logits

            logits_np = mx.array(logits)
            for token_id in set(int(t) for t in token_ids):
                token_logit = logits_np[..., token_id]
                adjusted = mx.where(
                    token_logit > 0,
                    token_logit / penalty,
                    token_logit * penalty,
                )
                logits_np = logits_np.at[..., token_id].set(adjusted)
            return logits_np

        return _processor

    def generate(self, prompt: str, *, config: Optional[GenerationConfig] = None) -> str:
        if not prompt.strip():
            raise ValueError("prompt must be a non-empty string.")

        cfg = config or GenerationConfig()
        model_size = self._infer_model_size_b(self._model_id)
        max_tokens = cfg.max_tokens
        repetition_penalty = cfg.repetition_penalty

        if model_size is not None:
            if model_size < 30:
                max_tokens = max_tokens or 500
                repetition_penalty = repetition_penalty or 1.15
            elif model_size >= 70:
                repetition_penalty = repetition_penalty or 1.05
        try:
            from mlx_lm import generate
            from mlx_lm.generate import make_sampler
        except Exception as exc:  # pragma: no cover - dependency runtime
            raise RuntimeError("mlx-lm generate is not available.") from exc

        try:
            sampler = make_sampler(temp=cfg.temperature, top_p=cfg.top_p)
            logits_processors = []
            if repetition_penalty is not None:
                processor = self._build_repetition_penalty_processor(repetition_penalty)
                if processor is not None:
                    logits_processors.append(processor)

            return generate(
                self._model,
                self._tokenizer,
                prompt,
                max_tokens=max_tokens or 512,
                sampler=sampler,
                logits_processors=logits_processors or None,
            )
        except Exception as exc:  # pragma: no cover - dependency runtime
            raise RuntimeError("mlx-lm generation failed.") from exc
