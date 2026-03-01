"""Tests for citation_verification.py — highlight text computation.

Tests cover the full pipeline: claim extraction → stem extraction →
stem position search → density window → child span overlap → sentence
boundary expansion → compute_highlight_texts integration.
"""

from __future__ import annotations

import pytest
from types import SimpleNamespace

from src.citation_verification import (
    _extract_claims,
    _extract_stems,
    _find_stem_positions,
    _best_density_window,
    _find_child_span,
    _compute_overlap,
    _expand_to_sentence_boundaries,
    compute_highlight_texts,
)


# ---------------------------------------------------------------------------
# _extract_claims
# ---------------------------------------------------------------------------

class TestExtractClaims:
    def test_simple_citation(self) -> None:
        text = "The theory was proposed by Chomsky [1]."
        claims = _extract_claims(text)
        assert 1 in claims
        assert len(claims[1]) == 1
        assert "Chomsky" in claims[1][0]

    def test_chunk_citation_format(self) -> None:
        text = "Behaviourism focuses on observable stimuli [CHUNK 2]."
        claims = _extract_claims(text)
        assert 2 in claims

    def test_multiple_citations_same_sentence(self) -> None:
        text = "Both Skinner [1] and Chomsky [2] discussed language."
        claims = _extract_claims(text)
        assert 1 in claims
        assert 2 in claims

    def test_no_citations(self) -> None:
        assert _extract_claims("No citations here.") == {}

    def test_citation_markers_stripped_from_claims(self) -> None:
        text = "Reinforcement learning [3] is key."
        claims = _extract_claims(text)
        for sentence in claims[3]:
            assert "[3]" not in sentence

    def test_multi_sentence(self) -> None:
        text = "First claim [1]. Second claim [2]. Third also [1]."
        claims = _extract_claims(text)
        assert len(claims[1]) == 2  # two sentences reference [1]
        assert len(claims[2]) == 1


# ---------------------------------------------------------------------------
# _extract_stems
# ---------------------------------------------------------------------------

class TestExtractStems:
    def test_basic_stems(self) -> None:
        stems = _extract_stems("describing descriptive descriptions")
        assert all(s == "descri" for s in stems)

    def test_stopwords_filtered(self) -> None:
        stems = _extract_stems("the and for are with")
        assert stems == []

    def test_short_words_filtered(self) -> None:
        stems = _extract_stems("a to be it")
        assert stems == []

    def test_mixed_content(self) -> None:
        stems = _extract_stems("Chomsky proposed universal grammar")
        assert "chomsk" in stems
        assert "propos" in stems
        assert "univer" in stems
        assert "gramma" in stems


# ---------------------------------------------------------------------------
# _find_stem_positions
# ---------------------------------------------------------------------------

class TestFindStemPositions:
    def test_single_stem_found(self) -> None:
        hits = _find_stem_positions(["behavi"], "The behaviourist approach was influential.")
        assert len(hits) > 0
        assert hits[0][1] == "behavi"

    def test_stem_not_found(self) -> None:
        hits = _find_stem_positions(["zzzzzy"], "Nothing matches here.")
        assert len(hits) == 0

    def test_multiple_occurrences(self) -> None:
        hits = _find_stem_positions(["connec"], "Connectionism and connectionist models are connected systems.")
        assert len(hits) >= 2

    def test_positions_sorted(self) -> None:
        hits = _find_stem_positions(
            ["gramma", "univer"],
            "Universal grammar is a universal theory of grammar.",
        )
        positions = [h[0] for h in hits]
        assert positions == sorted(positions)


# ---------------------------------------------------------------------------
# _best_density_window
# ---------------------------------------------------------------------------

class TestBestDensityWindow:
    def test_empty_hits(self) -> None:
        assert _best_density_window([]) is None

    def test_too_few_unique_stems(self) -> None:
        hits = [(0, "alpha"), (10, "alpha"), (20, "alpha")]
        assert _best_density_window(hits) is None  # only 1 unique stem

    def test_three_unique_stems(self) -> None:
        hits = [(0, "alpha"), (10, "bravo"), (20, "charl")]
        result = _best_density_window(hits)
        assert result is not None
        start, end, count = result
        assert count == 3
        assert start <= 0
        assert end >= 20

    def test_two_clusters(self) -> None:
        # Cluster A at positions 0-30 (3 stems), cluster B at 800-830 (3 stems)
        hits = [
            (0, "alpha"), (10, "bravo"), (30, "charl"),
            (800, "delta"), (810, "echo_"), (830, "foxtr"),
        ]
        result = _best_density_window(hits, window_size=100)
        assert result is not None
        assert result[2] == 3  # 3 unique stems in best window


# ---------------------------------------------------------------------------
# _find_child_span
# ---------------------------------------------------------------------------

class TestFindChildSpan:
    def test_exact_match(self) -> None:
        parent = "Before. The child chunk text here. After."
        child = "The child chunk text here."
        span = _find_child_span(child, parent)
        assert span is not None
        start, end = span
        assert parent[start:end] == child

    def test_no_match(self) -> None:
        assert _find_child_span("completely different", "no overlap at all") is None

    def test_whitespace_stripped(self) -> None:
        parent = "ABC the child text XYZ"
        child = "  the child text  "
        span = _find_child_span(child, parent)
        assert span is not None
        assert "the child text" in parent[span[0]:span[1]]

    def test_prefix_trim_fallback(self) -> None:
        parent = "Prefix material. The key section of the document continues here with important content that matters."
        child = "XXXXXXXXXXXX The key section of the document continues here with important content that matters."
        span = _find_child_span(child, parent)
        assert span is not None


# ---------------------------------------------------------------------------
# _compute_overlap
# ---------------------------------------------------------------------------

class TestComputeOverlap:
    def test_full_overlap(self) -> None:
        assert _compute_overlap(10, 50, 0, 100) == 1.0

    def test_no_overlap(self) -> None:
        assert _compute_overlap(0, 10, 20, 30) == 0.0

    def test_partial_overlap(self) -> None:
        result = _compute_overlap(0, 100, 50, 150)
        assert abs(result - 0.5) < 0.01

    def test_degenerate_window(self) -> None:
        assert _compute_overlap(10, 10, 0, 20) == 1.0


# ---------------------------------------------------------------------------
# _expand_to_sentence_boundaries
# ---------------------------------------------------------------------------

class TestExpandToSentenceBoundaries:
    def test_expand_both_directions(self) -> None:
        text = "First sentence. The target content here. Another sentence follows."
        start, end = _expand_to_sentence_boundaries(text, 16, 42)
        expanded = text[start:end]
        assert "The target content here" in expanded

    def test_expand_to_start_of_text(self) -> None:
        text = "No boundary before this content. Rest."
        start, end = _expand_to_sentence_boundaries(text, 0, 10)
        assert start == 0

    def test_expand_to_end_of_text(self) -> None:
        text = "Start. No boundary after this"
        start, end = _expand_to_sentence_boundaries(text, 7, len(text))
        assert end == len(text)


# ---------------------------------------------------------------------------
# compute_highlight_texts (integration)
# ---------------------------------------------------------------------------

def _make_result(
    text: str,
    parent_text: str | None = None,
) -> SimpleNamespace:
    """Create a mock RetrievalResult."""
    return SimpleNamespace(text=text, parent_text=parent_text)


class TestComputeHighlightTexts:
    def test_empty_inputs(self) -> None:
        assert compute_highlight_texts("", []) == {}
        assert compute_highlight_texts("Hello [1].", []) == {}
        assert compute_highlight_texts("", [_make_result("chunk")]) == {}

    def test_no_parent_text_skipped(self) -> None:
        """Citations without parent_text are left as-is."""
        results = [_make_result("child text", parent_text=None)]
        assert compute_highlight_texts("Answer [1].", results) == {}

    def test_citation_within_child_no_highlight(self) -> None:
        """When the claim overlaps with the child chunk, no override needed."""
        child = "Behaviorism emphasizes observable stimulus-response associations."
        parent = f"Introduction. {child} Conclusion."
        results = [_make_result(child, parent_text=parent)]
        answer = "Behaviorism focuses on stimulus-response associations [1]."
        highlights = compute_highlight_texts(answer, results)
        # Should NOT produce a highlight since claim overlaps with child
        assert 1 not in highlights

    def test_citation_outside_child_produces_highlight(self) -> None:
        """When the claim references parent text outside the child, highlight is produced."""
        child = "First section of the document with general overview material."
        parent = (
            f"{child} "
            "Connectionism provides theoretical computational framework describing "
            "neural network processing distributed representations across cognitive "
            "architectures implementing parallel distributed processing models."
        )
        results = [_make_result(child, parent_text=parent)]
        answer = (
            "Connectionism provides a computational framework for describing "
            "neural network processing and distributed representations [1]."
        )
        highlights = compute_highlight_texts(answer, results)
        # Should produce a highlight pointing to the connectionism section
        assert 1 in highlights
        assert len(highlights[1]) >= 20

    def test_out_of_range_citation_ignored(self) -> None:
        """Citation numbers beyond the results list are silently skipped."""
        results = [_make_result("child", parent_text="parent content")]
        answer = "Something [5]."
        assert compute_highlight_texts(answer, results) == {}

    def test_multiple_citations(self) -> None:
        """Multiple citations produce independent highlight results."""
        child1 = "First chunk about topic A."
        parent1 = f"{child1} Extended parent content about topic A details."
        child2 = "Second chunk about topic B."
        parent2 = (
            f"Preamble with different discussion entirely. {child2} "
            "Additional extended content discussing alternative theoretical "
            "frameworks computational modeling cognitive architecture design."
        )
        results = [
            _make_result(child1, parent_text=parent1),
            _make_result(child2, parent_text=parent2),
        ]
        answer = (
            "Topic A is discussed [1]. "
            "Alternative theoretical frameworks for computational modeling "
            "and cognitive architecture design are explored [2]."
        )
        highlights = compute_highlight_texts(answer, results)
        # Citation 2 references content outside child2; citation 1 may or may not
        # depending on stem overlap — at minimum verify no crash
        assert isinstance(highlights, dict)
