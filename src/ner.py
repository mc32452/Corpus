"""
Offline NER for place name extraction at ingest time.
Uses GLiNER medium v2.1 directly via the gliner package.
Falls back to regex (extract_places_from_query) if GLiNER is unavailable.

GLiNER achieves F1 ~0.85-0.89 on general NER vs ~0.67 for regex heuristics,
meaningfully reducing false positives on person names and titles.
"""
from __future__ import annotations

import logging
import threading

log = logging.getLogger(__name__)

_GLINER_MODEL = "urchade/gliner_medium-v2.1"
_NER_LABELS = ["location", "historical place", "city", "region", "country"]
_NER_THRESHOLD = 0.4
_BATCH_SIZE = 16

_model = None
_model_lock = threading.Lock()
_model_ready = False


def _get_model():
    global _model, _model_ready
    if _model_ready:
        return _model
    with _model_lock:
        if _model_ready:
            return _model
        try:
            from gliner import GLiNER

            _model = GLiNER.from_pretrained(_GLINER_MODEL)
            _model_ready = True
            log.info("GLiNER model loaded (%s).", _GLINER_MODEL)
        except Exception as exc:
            log.warning("GLiNER unavailable — NER will fall back to regex: %s", exc)
            _model = None
            _model_ready = True
    return _model


def extract_places_ner(texts: list[str]) -> list[list[str]]:
    """Extract place candidates for each text; fallback to regex on failures."""
    model = _get_model()
    if model is not None:
        try:
            results: list[list[str]] = []
            for start in range(0, len(texts), _BATCH_SIZE):
                batch = texts[start : start + _BATCH_SIZE]
                for text in batch:
                    entities = model.predict_entities(
                        text,
                        _NER_LABELS,
                        threshold=_NER_THRESHOLD,
                    )
                    # Preserve order while deduping exact repeats.
                    seen: set[str] = set()
                    items: list[str] = []
                    for ent in entities:
                        candidate = str(ent.get("text", "")).strip()
                        if not candidate or candidate in seen:
                            continue
                        seen.add(candidate)
                        items.append(candidate)
                    results.append(items)
            return results
        except Exception as exc:
            log.warning("GLiNER inference failed, falling back to regex: %s", exc)

    from .geocoder import extract_places_from_query

    return [extract_places_from_query(text) for text in texts]
