"""
Scoring pipeline orchestrator.

Runs all scorers concurrently via a thread pool, then combines
results with mode-specific weights and applies duplicate penalties.
"""
from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config.weights import (
    DUPLICATE_PENALTY_FACTOR,
    NO_FACE_REDISTRIBUTE,
    SCORING_WEIGHTS,
)
from src.models.response import ImageMeta, RankedImage, RankResponse, ScoreBreakdown
from src.scorers.duplicate_scorer import DuplicateScorer
from src.scorers.face_scorer import FaceScorer
from src.scorers.semantic_scorer import SemanticScorer
from src.scorers.technical_scorer import TechnicalScorer

logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=4)


def _redistribute_weights(
    weights: Dict[str, float],
    keys_to_remove: List[str],
) -> Dict[str, float]:
    """Remove keys and proportionally redistribute their weight to remaining keys."""
    removed_weight = sum(weights.get(k, 0.0) for k in keys_to_remove)
    remaining = {k: v for k, v in weights.items() if k not in keys_to_remove}
    total_remaining = sum(remaining.values())
    if total_remaining == 0:
        return remaining
    factor = (total_remaining + removed_weight) / total_remaining
    return {k: v * factor for k, v in remaining.items()}


def _compute_weighted_score(
    scores: Dict[str, Optional[float]],
    weights: Dict[str, float],
    has_faces: bool,
) -> float:
    """Compute 0-10 weighted total, redistributing weights for absent face scores."""
    effective = dict(weights)
    if not has_faces:
        effective = _redistribute_weights(effective, NO_FACE_REDISTRIBUTE)

    total = 0.0
    total_weight = 0.0
    for dim, w in effective.items():
        val = scores.get(dim)
        if val is not None:
            total += val * w
            total_weight += w

    if total_weight == 0:
        return 0.0
    return round(total / total_weight * 10, 2)


def _round_opt(val: Optional[float]) -> Optional[float]:
    return round(val, 2) if val is not None else None


class ScoringPipeline:
    """
    Orchestrates all scorers for a batch of images.
    Thread-safe after construction.
    """

    def __init__(
        self,
        face_scorer: FaceScorer,
        technical_scorer: TechnicalScorer,
        duplicate_scorer: DuplicateScorer,
        semantic_scorer: SemanticScorer,
    ):
        self.face_scorer      = face_scorer
        self.technical_scorer = technical_scorer
        self.duplicate_scorer = duplicate_scorer
        self.semantic_scorer  = semantic_scorer

    def _score_single(self, path: Path) -> Dict[str, Any]:
        """Score one image synchronously (runs in thread pool)."""
        result: Dict[str, Any] = {"path": str(path), "error": None}
        try:
            result["face"]     = self.face_scorer.score(path)
            result["tech"]     = self.technical_scorer.score(path)
            result["semantic"] = self.semantic_scorer.score(path)
        except Exception as e:
            logger.warning(f"Error scoring {path}: {e}")
            result["error"] = str(e)
        return result

    async def run(
        self,
        image_paths: List[Path],
        mode: str,
        top_n: Optional[int] = None,
        job_id: Optional[str] = None,
    ) -> RankResponse:
        """
        Full async pipeline: score → deduplicate → weight → rank.

        Args:
            image_paths: validated image Paths
            mode:        scoring mode (wedding/portrait/event/general)
            top_n:       return only top N; None = return all
            job_id:      if provided, update job_store progress

        Returns:
            RankResponse with sorted ranked images.
        """
        from src.core.job_store import job_store

        start = time.monotonic()
        weights = SCORING_WEIGHTS.get(mode, SCORING_WEIGHTS["general"])
        logger.info(f"Pipeline start: {len(image_paths)} images, mode={mode}")

        # ── Step 1: Score each image in parallel ──────────────────────────
        loop = asyncio.get_event_loop()
        futures = [
            loop.run_in_executor(_executor, self._score_single, p)
            for p in image_paths
        ]

        raw_results: List[Dict] = []
        for i, fut in enumerate(asyncio.as_completed(futures)):
            raw_results.append(await fut)
            if job_id:
                progress = int((i + 1) / len(image_paths) * 80)
                await job_store.update(
                    job_id, progress=progress, processed_images=i + 1
                )

        # ── Step 2: Duplicate detection (batch operation) ─────────────────
        valid_paths = [Path(r["path"]) for r in raw_results if not r.get("error")]
        dup_map = self.duplicate_scorer.find_duplicates(valid_paths)

        # ── Step 3: Build ranked image objects ────────────────────────────
        ranked: List[RankedImage] = []

        for r in raw_results:
            if r.get("error"):
                continue

            path_str = r["path"]
            face     = r.get("face", {})
            tech     = r.get("tech", {})
            semantic = r.get("semantic", {})

            has_faces  = face.get("has_faces", False)
            face_count = face.get("face_count", 0)

            scores_raw: Dict[str, Optional[float]] = {
                "face_clarity":     face.get("face_clarity"),
                "eyes_open":        face.get("eyes_open"),
                "smile":            semantic.get("smile"),
                "emotion":          semantic.get("emotion", 5.0),
                "technical_quality":tech.get("technical_quality", 5.0),
                "composition":      face.get("composition", tech.get("composition", 5.0)),
                "cinematic":        semantic.get("cinematic", 5.0),
                "invitation_suit":  semantic.get("invitation_suit", 5.0),
                "storytelling":     semantic.get("storytelling", 5.0),
            }

            if not has_faces:
                scores_raw["face_clarity"] = None
                scores_raw["eyes_open"]    = None
                scores_raw["smile"]        = None

            total_score = _compute_weighted_score(scores_raw, weights, has_faces)

            dup_original = dup_map.get(path_str)
            is_duplicate = dup_original is not None
            if is_duplicate:
                total_score = round(total_score * DUPLICATE_PENALTY_FACTOR, 2)

            ranked.append(
                RankedImage(
                    path=path_str,
                    rank=0,
                    total_score=total_score,
                    scores=ScoreBreakdown(
                        face_clarity=_round_opt(scores_raw["face_clarity"]),
                        eyes_open=_round_opt(scores_raw["eyes_open"]),
                        smile=_round_opt(scores_raw["smile"]),
                        emotion=round(scores_raw["emotion"] or 5.0, 2),
                        technical_quality=round(scores_raw["technical_quality"] or 5.0, 2),
                        composition=round(scores_raw["composition"] or 5.0, 2),
                        cinematic=round(scores_raw["cinematic"] or 5.0, 2),
                        invitation_suit=round(scores_raw["invitation_suit"] or 5.0, 2),
                        storytelling=round(scores_raw["storytelling"] or 5.0, 2),
                    ),
                    meta=ImageMeta(
                        has_faces=has_faces,
                        face_count=face_count,
                        is_duplicate=is_duplicate,
                        duplicate_of=dup_original,
                    ),
                )
            )

        # ── Step 4: Sort and assign ranks ─────────────────────────────────
        ranked.sort(key=lambda x: x.total_score, reverse=True)
        for i, img in enumerate(ranked, start=1):
            img.rank = i

        if top_n:
            ranked = ranked[:top_n]

        elapsed_ms = round((time.monotonic() - start) * 1000, 1)
        if job_id:
            await job_store.update(job_id, progress=100)

        logger.info(
            f"Pipeline done: {len(ranked)} ranked in {elapsed_ms:.0f}ms"
        )
        return RankResponse(
            status="success",
            mode=mode,
            total_images=len(image_paths),
            processing_time_ms=elapsed_ms,
            ranked=ranked,
        )
