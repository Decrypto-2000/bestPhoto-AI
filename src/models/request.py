"""Pydantic request models."""
from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field, model_validator


class RankRequest(BaseModel):
    """
    Submit images for ranking.

    Provide either `paths` (list of individual image file paths)
    or `folder` (directory to scan), or both.

    Paths may be:
    - Absolute Linux paths:   /mnt/nas/photos/img.jpg
    - Absolute Windows paths: C:\\Photos\\img.jpg  or  \\\\server\\share\\img.jpg
    """

    paths: Optional[List[str]] = Field(
        default=None,
        description="List of absolute image file paths (local or NAS).",
        examples=[["/mnt/nas/event/photo1.jpg", "C:\\Photos\\photo2.jpg"]],
    )

    folder: Optional[str] = Field(
        default=None,
        description="Absolute path to a folder. All images inside will be ranked.",
        examples=["/mnt/nas/wedding_2024"],
    )

    recursive: bool = Field(
        default=False,
        description="When folder is provided, scan sub-folders recursively.",
    )

    extensions: Optional[List[str]] = Field(
        default=None,
        description="Restrict to these file extensions (without dot). Defaults to all supported formats.",
        examples=[["jpg", "jpeg", "png"]],
    )

    mode: str = Field(
        default="general",
        description="Scoring mode: 'wedding', 'portrait', 'event', or 'general'.",
        pattern="^(wedding|portrait|event|general)$",
    )

    top_n: Optional[int] = Field(
        default=None,
        ge=1,
        description="Return only the top N ranked images. Omit to return all.",
    )

    async_mode: bool = Field(
        default=False,
        alias="async",
        description="If true and batch is large, return a job_id immediately.",
    )

    model_config = {"populate_by_name": True}

    @model_validator(mode="after")
    def at_least_one_source(self) -> "RankRequest":
        if not self.paths and not self.folder:
            raise ValueError("Provide at least one of: 'paths' or 'folder'.")
        return self
