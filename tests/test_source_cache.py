"""Tests for text snapshot storage and content resolution.

Validates save/read/delete of snapshots, original file reading,
and the full resolution chain (original → snapshot → None).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.source_cache import (
    _sanitize_filename,
    delete_snapshot,
    read_original_file,
    read_snapshot,
    resolve_content,
    save_snapshot,
)


# ---------------------------------------------------------------------------
# _sanitize_filename
# ---------------------------------------------------------------------------


class TestSanitizeFilename:
    def test_simple_id(self) -> None:
        assert _sanitize_filename("my_document") == "my_document.txt"

    def test_id_with_special_chars(self) -> None:
        """IDs with spaces/slashes get hashed."""
        name = _sanitize_filename("my document/v2")
        assert name.endswith(".txt")
        assert len(name) > 4  # hash + .txt

    def test_long_id_gets_hashed(self) -> None:
        name = _sanitize_filename("a" * 300)
        assert name.endswith(".txt")
        assert len(name) < 100  # hash is short


# ---------------------------------------------------------------------------
# save_snapshot / read_snapshot
# ---------------------------------------------------------------------------


class TestSnapshotReadWrite:
    def test_save_and_read(self, tmp_path: Path) -> None:
        """Save a snapshot and read it back."""
        content = "This is the full text of the document."
        path = save_snapshot("test_doc", content, cache_dir=tmp_path)
        assert Path(path).exists()

        result = read_snapshot(path)
        assert result == content

    def test_save_creates_directory(self, tmp_path: Path) -> None:
        """save_snapshot creates the cache directory if it doesn't exist."""
        nested_dir = tmp_path / "nested" / "cache"
        path = save_snapshot("doc", "text", cache_dir=nested_dir)
        assert Path(path).exists()

    def test_read_nonexistent_returns_none(self) -> None:
        assert read_snapshot("/nonexistent/path/file.txt") is None

    def test_save_overwrite(self, tmp_path: Path) -> None:
        """Saving twice overwrites the previous content."""
        save_snapshot("doc", "original text", cache_dir=tmp_path)
        path = save_snapshot("doc", "updated text", cache_dir=tmp_path)
        assert read_snapshot(path) == "updated text"

    def test_unicode_content(self, tmp_path: Path) -> None:
        content = "Héllo wörld 你好 🌍"
        path = save_snapshot("unicode_doc", content, cache_dir=tmp_path)
        assert read_snapshot(path) == content

    def test_empty_content(self, tmp_path: Path) -> None:
        path = save_snapshot("empty_doc", "", cache_dir=tmp_path)
        assert read_snapshot(path) == ""

    def test_large_content(self, tmp_path: Path) -> None:
        content = "word " * 100_000  # ~500KB
        path = save_snapshot("large_doc", content, cache_dir=tmp_path)
        assert read_snapshot(path) == content


# ---------------------------------------------------------------------------
# delete_snapshot
# ---------------------------------------------------------------------------


class TestDeleteSnapshot:
    def test_delete_existing(self, tmp_path: Path) -> None:
        path = save_snapshot("doc", "text", cache_dir=tmp_path)
        assert delete_snapshot(path) is True
        assert not Path(path).exists()

    def test_delete_nonexistent(self) -> None:
        assert delete_snapshot("/nonexistent/file.txt") is False

    def test_read_after_delete_returns_none(self, tmp_path: Path) -> None:
        path = save_snapshot("doc", "text", cache_dir=tmp_path)
        delete_snapshot(path)
        assert read_snapshot(path) is None


# ---------------------------------------------------------------------------
# read_original_file
# ---------------------------------------------------------------------------


class TestReadOriginalFile:
    def test_read_text_file(self, tmp_path: Path) -> None:
        f = tmp_path / "doc.md"
        f.write_text("# Hello\n\nThis is markdown.", encoding="utf-8")
        content = read_original_file(str(f))
        assert content == "# Hello\n\nThis is markdown."

    def test_nonexistent_returns_none(self) -> None:
        assert read_original_file("/nonexistent/file.txt") is None

    def test_unicode_file(self, tmp_path: Path) -> None:
        f = tmp_path / "unicode.txt"
        f.write_text("Ünïcödé content 中文", encoding="utf-8")
        assert read_original_file(str(f)) == "Ünïcödé content 中文"


# ---------------------------------------------------------------------------
# resolve_content (fallback chain)
# ---------------------------------------------------------------------------


class TestResolveContent:
    def test_original_preferred_over_snapshot(self, tmp_path: Path) -> None:
        """When both exist, original file takes precedence."""
        original = tmp_path / "original.txt"
        original.write_text("Original content", encoding="utf-8")

        snapshot_path = save_snapshot("doc", "Snapshot content", cache_dir=tmp_path / "cache")

        result = resolve_content(str(original), snapshot_path)
        assert result is not None
        content, source = result
        assert content == "Original content"
        assert source == "original"

    def test_snapshot_fallback_when_original_missing(self, tmp_path: Path) -> None:
        """When original is gone, snapshot is used."""
        snapshot_path = save_snapshot("doc", "Snapshot content", cache_dir=tmp_path / "cache")

        result = resolve_content("/nonexistent/file.txt", snapshot_path)
        assert result is not None
        content, source = result
        assert content == "Snapshot content"
        assert source == "snapshot"

    def test_snapshot_fallback_when_no_source_path(self, tmp_path: Path) -> None:
        """When source_path is None, snapshot is used."""
        snapshot_path = save_snapshot("doc", "Cached text", cache_dir=tmp_path / "cache")

        result = resolve_content(None, snapshot_path)
        assert result is not None
        content, source = result
        assert content == "Cached text"
        assert source == "snapshot"

    def test_none_when_both_missing(self) -> None:
        """When neither exists, returns None."""
        result = resolve_content("/gone/file.txt", "/gone/snapshot.txt")
        assert result is None

    def test_none_when_both_none(self) -> None:
        result = resolve_content(None, None)
        assert result is None

    def test_original_file_moved_then_snapshot(self, tmp_path: Path) -> None:
        """Simulate: ingest with original → move original → content from snapshot."""
        # Step 1: Create original file
        original = tmp_path / "paper.md"
        original.write_text("Original paper text", encoding="utf-8")

        # Step 2: Save snapshot during ingest
        snapshot_path = save_snapshot("paper", "Original paper text", cache_dir=tmp_path / "cache")

        # Step 3: Move original file (simulating user action)
        original.unlink()

        # Step 4: Resolve — should get snapshot
        result = resolve_content(str(original), snapshot_path)
        assert result is not None
        assert result[0] == "Original paper text"
        assert result[1] == "snapshot"

    def test_delete_source_then_404(self, tmp_path: Path) -> None:
        """After deleting both original and snapshot, resolve returns None."""
        original = tmp_path / "paper.md"
        original.write_text("text", encoding="utf-8")
        snapshot_path = save_snapshot("paper", "text", cache_dir=tmp_path / "cache")

        # Delete both
        original.unlink()
        delete_snapshot(snapshot_path)

        result = resolve_content(str(original), snapshot_path)
        assert result is None
