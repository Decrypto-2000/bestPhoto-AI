"""Health check endpoint."""
from fastapi import APIRouter

from src.models.response import HealthResponse
from src.scorers.semantic_scorer import SemanticScorer

router = APIRouter()


@router.get("/health", response_model=HealthResponse, summary="Service health check")
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        models_loaded={
            "clip":      SemanticScorer._model is not None,
            "mediapipe": True,   # stateless, no persistent load needed
            "opencv":    True,
        },
    )
