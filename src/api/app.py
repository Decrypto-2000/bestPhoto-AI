"""FastAPI application factory with lifespan model loading."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes.health import router as health_router
from src.api.routes.rank import router as rank_router
from src.scorers.semantic_scorer import SemanticScorer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML models at startup; release on shutdown."""
    logger.info("BestPhoto AI starting up...")
    SemanticScorer.load_models()
    logger.info("All models loaded. Ready to rank images.")
    yield
    logger.info("BestPhoto AI shutting down.")


def create_app() -> FastAPI:
    app = FastAPI(
        title="BestPhoto AI",
        description=(
            "Ranks images by photographic quality across 9 dimensions. "
            "Supports solo/group/non-portrait photos, NAS and local paths (Linux + Windows), "
            "and mode-aware scoring for wedding, portrait, event, and general photography."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health_router, prefix="/api/v1", tags=["Health"])
    app.include_router(rank_router,   prefix="/api/v1", tags=["Ranking"])

    return app
