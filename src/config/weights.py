"""
Mode-specific scoring weights.

Each mode defines weights for 9 scoring dimensions.
Weights must sum to 1.0 per mode.

Dimensions:
  - face_clarity        : sharpness of detected faces
  - eyes_open           : fraction of faces with open eyes
  - smile               : smile / happiness presence (CLIP)
  - emotion             : emotional moment quality (CLIP)
  - technical_quality   : overall blur, exposure, noise score
  - composition         : rule-of-thirds & subject placement
  - cinematic           : cinematic framing quality (CLIP)
  - invitation_suit     : suitability for formal invitations (CLIP)
  - storytelling        : narrative/storytelling quality (CLIP)
"""

SCORING_WEIGHTS: dict[str, dict[str, float]] = {
    "wedding": {
        "face_clarity":     0.12,
        "eyes_open":        0.10,
        "smile":            0.12,
        "emotion":          0.12,
        "technical_quality":0.08,
        "composition":      0.10,
        "cinematic":        0.10,
        "invitation_suit":  0.14,
        "storytelling":     0.12,
    },
    "portrait": {
        "face_clarity":     0.22,
        "eyes_open":        0.18,
        "smile":            0.10,
        "emotion":          0.10,
        "technical_quality":0.13,
        "composition":      0.12,
        "cinematic":        0.07,
        "invitation_suit":  0.05,
        "storytelling":     0.03,
    },
    "event": {
        "face_clarity":     0.10,
        "eyes_open":        0.08,
        "smile":            0.12,
        "emotion":          0.14,
        "technical_quality":0.10,
        "composition":      0.10,
        "cinematic":        0.10,
        "invitation_suit":  0.06,
        "storytelling":     0.20,
    },
    "general": {
        "face_clarity":     0.12,
        "eyes_open":        0.08,
        "smile":            0.08,
        "emotion":          0.10,
        "technical_quality":0.18,
        "composition":      0.16,
        "cinematic":        0.12,
        "invitation_suit":  0.06,
        "storytelling":     0.10,
    },
}

# Duplicate penalty: multiply total score by this factor when image is a duplicate
DUPLICATE_PENALTY_FACTOR: float = 0.35

# No-face fallback: when no faces detected, face_clarity / eyes_open / smile
# weights are redistributed proportionally to remaining dimensions.
NO_FACE_REDISTRIBUTE: list[str] = ["face_clarity", "eyes_open", "smile"]
