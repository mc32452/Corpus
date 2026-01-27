from __future__ import annotations

import uuid
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class Metadata(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: str = Field(..., min_length=1)
    page_number: Optional[int] = Field(default=None, ge=1)
    header_path: str = Field(..., min_length=1)
    parent_id: Optional[str] = None


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
