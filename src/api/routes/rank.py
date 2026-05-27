"""
Image ranking endpoints.

POST /api/v1/rank
    Rank images from file paths or a folder.
    Returns ranked list (sync) or a job_id (async for large batches).

GET /api/v1/jobs/{job_id}
    Poll status of an async ranking job.
"""
from __future__ import annotations

import logging
from typing import Union

from fastapi import APIRouter, BackgroundTasks, HTTPException

from src.config.settings import settings
from src.core.image_loader import collect_image_paths
from src.core.job_store import job_store
from src.core.pipeline import ScoringPipeline
from src.models.request import RankRequest
from src.models.response import AsyncJobResponse, JobStatusResponse, RankResponse
from src.scorers.duplicate_scorer import DuplicateScorer
from src.scorers.face_scorer import FaceScorer
from src.scorers.semantic_scorer import SemanticScorer
from src.scorers.technical_scorer import TechnicalScorer

router = APIRouter()
logger = logging.getLogger(__name__)

# Scorer singletons — stateless after init, shared across requests
_face_scorer      = FaceScorer()
_technical_scorer = TechnicalScorer()
_duplicate_scorer = DuplicateScorer()
_semantic_scorer  = SemanticScorer()


def _pipeline() -> ScoringPipeline:
    return ScoringPipeline(
        face_scorer=_face_scorer,
        technical_scorer=_technical_scorer,
        duplicate_scorer=_duplicate_scorer,
        semantic_scorer=_semantic_scorer,
    )


async def _run_async_job(job_id: str, request: RankRequest, image_paths):
    """Background task for large-batch async processing."""
    try:
        await job_store.update(
            job_id, status="processing", total_images=len(image_paths)
        )
        result = await _pipeline().run(
            image_paths, request.mode, request.top_n, job_id=job_id
        )
        await job_store.update(job_id, status="completed", result=result.model_dump())
    except Exception as e:
        logger.exception(f"Async job {job_id} failed: {e}")
        await job_store.update(job_id, status="failed", error=str(e))


@router.post(
    "/rank",
    response_model=Union[RankResponse, AsyncJobResponse],
    summary="Rank images by photographic quality",
    description=(
        "Submit a list of image paths and/or a folder path. "
        "Returns images ranked by composite quality score using face analysis, "
        "technical quality assessment, duplicate detection, and AI semantic scoring via CLIP."
    ),
)
async def rank_images(
    request: RankRequest,
    background_tasks: BackgroundTasks,
) -> Union[RankResponse, AsyncJobResponse]:

    # 1. Resolve all image paths
    try:
        image_paths = collect_image_paths(
            paths=request.paths,
            folder=request.folder,
            recursive=request.recursive,
            extensions=request.extensions,
        )
    except (FileNotFoundError, NotADirectoryError, ValueError) as e:
        raise HTTPException(status_code=422, detail=str(e))

    if not image_paths:
        raise HTTPException(
            status_code=422,
            detail="No supported image files found at the provided paths/folder.",
        )

    # 2. Route: async for large batches or when explicitly requested
    use_async = request.async_mode or len(image_paths) > settings.SYNC_BATCH_LIMIT

    if use_async:
        job = await job_store.create()
        background_tasks.add_task(_run_async_job, job.job_id, request, image_paths)
        return AsyncJobResponse(
            job_id=job.job_id,
            poll_url=f"/api/v1/jobs/{job.job_id}",
            message=(
                f"Batch of {len(image_paths)} images submitted for async processing. "
                "Poll poll_url for progress and results."
            ),
        )

    # 3. Synchronous processing
    return await _pipeline().run(image_paths, request.mode, request.top_n)


@router.get(
    "/jobs/{job_id}",
    response_model=JobStatusResponse,
    summary="Poll status of an async ranking job",
)
async def get_job(job_id: str) -> JobStatusResponse:
    job = await job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        progress=job.progress,
        total_images=job.total_images,
        processed_images=job.processed_images,
        result=job.result,
        error=job.error,
    )
