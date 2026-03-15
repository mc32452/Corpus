"""Pydantic domain models for document storage.

These models represent the LanceDB schema and are intentionally separate
from the HTTP boundary models in ``api_schemas.py`` — domain models describe
stored entities; API schemas describe what clients send and receive.

Architecture
~~~~~~~~~~~~
- ``Metadata`` is shared by both chunk types and carries source provenance
  (source_id, page numbers, header path, parent linkage).
- ``ParentChunk`` holds the wider ~1 200-token passage used for LLM context;
  ``ChildChunk`` holds the ~250-token passage used for dense/sparse retrieval
  and reranking.  Each child's ``metadata.parent_id`` links back to its
  parent, enabling context expansion after retrieval.
- All models are ``frozen=True`` (immutable after construction) to prevent
  accidental mutation inside the pipeline.
"""
from __future__ import annotations

import uuid
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Metadata(BaseModel):
    """Metadata for document chunks."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: str = Field(..., min_length=1)
    page_number: Optional[int] = Field(default=None, ge=1)
    start_page: Optional[int] = Field(default=None, ge=1)
    end_page: Optional[int] = Field(default=None, ge=1)
    page_label: Optional[str] = Field(default=None, description="Logical page label from PDF")
    display_page: Optional[str] = Field(default=None, description="Human-readable page for citations")
    header_path: str = Field(..., min_length=1)
    parent_id: Optional[str] = None

    @model_validator(mode="after")
    def _validate_page_range(self) -> "Metadata":
        if (
            self.start_page is not None
            and self.end_page is not None
            and self.start_page > self.end_page
        ):
            raise ValueError("start_page must be <= end_page")
        return self


class ParentChunk(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str = Field(..., min_length=1)
    metadata: Metadata


class ChildChunk(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str = Field(..., min_length=1)
    metadata: Metadata
