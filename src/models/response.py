"""Pydantic response models."""
from __future__ import annotations
from typing import Dict, List, Optional, Any
from pydantic import BaseModel


class ScoreBreakdown(BaseModel):
    face_clarity: Optional[float] = None       # 0-10, None if no face
    eyes_open: Optional[float] = None          # 0-10, None if no face
    smile: Optional[float] = None             # 0-10, None if no face
    emotion: float                             # 0-10
    technical_quality: float                   # 0-10
    composition: float                         # 0-10
    cinematic: float                           # 0-10
    invitation_suit: float                     # 0-10
    storytelling: float                        # 0-10


class ImageMeta(BaseModel):
    has_faces: bool
    face_count: int
    is_duplicate: bool
    duplicate_of: Optional[str] = None        # path of the "original" image


class RankedImage(BaseModel):
    path: str
    rank: int
    total_score: float                         # weighted total 0-10
    scores: ScoreBreakdown
    meta: ImageMeta


class RankResponse(BaseModel):
    status: str = "success"
    mode: str
    total_images: int
    processing_time_ms: float
    ranked: List[RankedImage]


class AsyncJobResponse(BaseModel):
    status: str = "accepted"
    job_id: str
    poll_url: str
    message: str = "Large batch submitted. Poll poll_url for results."


class JobStatusResponse(BaseModel):
    job_id: str
    status: str                                # pending | processing | completed | failed
    progress: int                              # 0-100
    total_images: Optional[int] = None
    processed_images: Optional[int] = None
    result: Optional[RankResponse] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    models_loaded: Dict[str, bool]
    version: str = "1.0.0"
