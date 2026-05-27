"""
Face Scorer using MediaPipe.

Detects faces and computes:
  - face_clarity  : sharpness of the face region (0-10)
  - eyes_open     : fraction of detected faces with open eyes × 10 (0-10)
  - face_count    : number of faces
  - has_faces     : bool
  - composition   : rule-of-thirds alignment of main face (0-10)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import mediapipe as mp
import numpy as np

from src.config.settings import settings
from src.scorers.base import BaseScorer

logger = logging.getLogger(__name__)

# MediaPipe face mesh eye landmark indices
_LEFT_EYE  = [362, 385, 387, 263, 373, 380]
_RIGHT_EYE = [33,  160, 158, 133, 153, 144]


def _ear(landmarks, eye_indices: List[int], w: int, h: int) -> float:
    """Eye Aspect Ratio — values < EAR_THRESHOLD indicate a closed eye."""
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_indices]
    v1 = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    v2 = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    hd = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
    if hd < 1e-6:
        return 0.0
    return (v1 + v2) / (2.0 * hd)


def _sharpness(gray: np.ndarray) -> float:
    """Laplacian variance as a sharpness measure."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _thirds_score(cx: float, cy: float) -> float:
    """
    Rule-of-thirds score for a face centred at (cx, cy) in [0,1] space.
    Returns 0-10; peaks at the four rule-of-thirds intersections.
    """
    thirds_x = [1 / 3, 2 / 3]
    thirds_y = [1 / 3, 2 / 3]
    min_dist = min(
        ((cx - tx) ** 2 + (cy - ty) ** 2) ** 0.5
        for tx in thirds_x
        for ty in thirds_y
    )
    score = max(0.0, 1.0 - min_dist / 0.47)
    return round(score * 10, 2)


class FaceScorer(BaseScorer):
    """
    MediaPipe-powered face analysis scorer.
    Stateless — MediaPipe contexts are opened per call (thread-safe).
    """

    def score(self, image_path: Path) -> Dict[str, Any]:
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            logger.warning(f"Could not read image: {image_path}")
            return self._no_face_result()

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # --- Face Detection ---
        mp_fd = mp.solutions.face_detection
        detections = []
        with mp_fd.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.4,
        ) as detector:
            fd_result = detector.process(img_rgb)
            if fd_result.detections:
                detections = fd_result.detections

        if not detections:
            return self._no_face_result()

        face_count = len(detections)

        # --- Face Mesh for EAR and per-face sharpness ---
        mp_fm = mp.solutions.face_mesh
        eyes_open_scores: List[float] = []
        clarity_scores: List[float] = []
        composition_scores: List[float] = []

        with mp_fm.FaceMesh(
            static_image_mode=True,
            max_num_faces=min(face_count + 2, 10),
            refine_landmarks=True,
            min_detection_confidence=0.4,
        ) as mesh:
            fm_result = mesh.process(img_rgb)

            if fm_result.multi_face_landmarks:
                for face_lm in fm_result.multi_face_landmarks:
                    lm = face_lm.landmark

                    # Eye openness via EAR
                    left_ear  = _ear(lm, _LEFT_EYE,  w, h)
                    right_ear = _ear(lm, _RIGHT_EYE, w, h)
                    avg_ear = (left_ear + right_ear) / 2.0
                    open_score = min(10.0, max(0.0, (avg_ear - settings.EAR_THRESHOLD) / 0.12 * 10))
                    eyes_open_scores.append(open_score)

                    # Face sharpness from bounding box of landmarks
                    xs = [int(l.x * w) for l in lm]
                    ys = [int(l.y * h) for l in lm]
                    x1, x2 = max(0, min(xs)), min(w, max(xs))
                    y1, y2 = max(0, min(ys)), min(h, max(ys))
                    if x2 > x1 and y2 > y1:
                        face_region = gray[y1:y2, x1:x2]
                        lap_var = _sharpness(face_region)
                        clarity = min(10.0, lap_var / 50.0)
                        clarity_scores.append(clarity)

                    # Composition: rule-of-thirds from face centroid
                    cx = float(np.mean([l.x for l in lm]))
                    cy = float(np.mean([l.y for l in lm]))
                    composition_scores.append(_thirds_score(cx, cy))

        face_clarity = round(float(np.mean(clarity_scores)), 2) if clarity_scores else 5.0
        eyes_open    = round(float(np.mean(eyes_open_scores)), 2) if eyes_open_scores else 5.0
        composition  = round(float(np.mean(composition_scores)), 2) if composition_scores else 5.0

        return {
            "has_faces":    True,
            "face_count":   face_count,
            "face_clarity": face_clarity,
            "eyes_open":    eyes_open,
            "composition":  composition,
        }

    @staticmethod
    def _no_face_result() -> Dict[str, Any]:
        return {
            "has_faces":    False,
            "face_count":   0,
            "face_clarity": None,
            "eyes_open":    None,
            "composition":  5.0,
        }
