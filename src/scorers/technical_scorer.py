"""
Technical Quality Scorer using OpenCV.

Computes:
  - blur_score       : image sharpness via Laplacian variance (0-10)
  - exposure_score   : exposure quality via histogram analysis (0-10)
  - noise_score      : estimated noise level (0-10; higher = cleaner)
  - technical_quality: weighted combination (0-10)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np

from src.scorers.base import BaseScorer

logger = logging.getLogger(__name__)


def _blur_score(gray: np.ndarray) -> float:
    """Laplacian variance → sharpness 0-10. lap_var ~50 = borderline, ~500 = crisp."""
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return min(10.0, float(lap_var) / 50.0)


def _exposure_score(gray: np.ndarray) -> float:
    """Histogram-based exposure quality 0-10. Ideal mean ~90-160; penalises clipping."""
    mean_b = float(gray.mean())
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    total_pixels = gray.size

    shadow_ratio    = hist[:30].sum()  / total_pixels
    highlight_ratio = hist[225:].sum() / total_pixels
    clipping_penalty = min(1.0, (shadow_ratio + highlight_ratio) * 5)

    brightness_score = max(0.0, 1.0 - abs(mean_b - 125) / 125.0)
    raw = (brightness_score * 0.7 + (1 - clipping_penalty) * 0.3) * 10
    return round(max(0.0, min(10.0, raw)), 2)


def _noise_score(gray: np.ndarray) -> float:
    """High-frequency residual after Gaussian blur → noise estimate. 0-10 (higher = cleaner)."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    diff = gray.astype(np.float32) - blurred.astype(np.float32)
    noise_std = float(np.std(diff))
    return round(max(0.0, min(10.0, (1.0 - noise_std / 20.0) * 10.0)), 2)


class TechnicalScorer(BaseScorer):
    """Scores technical image quality using OpenCV."""

    def score(self, image_path: Path) -> Dict[str, Any]:
        img = cv2.imread(str(image_path))
        if img is None:
            logger.warning(f"Could not read image for technical scoring: {image_path}")
            return {
                "technical_quality": 5.0,
                "blur_score": 5.0,
                "exposure_score": 5.0,
                "noise_score": 5.0,
            }

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        blur     = _blur_score(gray)
        exposure = _exposure_score(gray)
        noise    = _noise_score(gray)

        # Blur matters most for photographic quality
        technical_quality = round(blur * 0.5 + exposure * 0.3 + noise * 0.2, 2)

        return {
            "technical_quality": technical_quality,
            "blur_score":        round(blur, 2),
            "exposure_score":    exposure,
            "noise_score":       noise,
        }
