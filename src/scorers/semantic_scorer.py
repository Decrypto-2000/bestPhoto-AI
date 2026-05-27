"""
Semantic Scorer using OpenAI CLIP (via HuggingFace Transformers).

Scores each image against curated text prompt pairs for:
  - smile           : happiness / smiling presence
  - emotion         : emotional moment quality
  - cinematic       : cinematic framing & lighting
  - composition     : photographic composition elegance
  - invitation_suit : suitability for formal invitations / print
  - storytelling    : narrative / storytelling quality

CLIP model is loaded ONCE at startup via load_models().
Text features are precomputed at load time for efficiency.
The score() method is thread-safe after loading.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from src.config.settings import settings
from src.scorers.base import BaseScorer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Curated prompt pairs  (positive vs negative)
# Positive-class probability is mapped to 0-10.
# ---------------------------------------------------------------------------
CRITERION_PROMPTS: Dict[str, Dict[str, List[str]]] = {
    "smile": {
        "positive": [
            "a person smiling warmly and joyfully",
            "people laughing and smiling together",
            "a genuine happy smile on a person's face",
        ],
        "negative": [
            "a person with a serious or neutral expression",
            "people with blank or sad faces",
            "a formal portrait with no smile",
        ],
    },
    "emotion": {
        "positive": [
            "an emotionally touching and heartfelt moment",
            "a candid photo full of genuine emotion",
            "people sharing an emotional joyful experience",
            "a tender intimate human moment captured on camera",
        ],
        "negative": [
            "a stiff formal posed photograph with no emotion",
            "people looking bored or disengaged",
            "an empty scene with no human emotion",
        ],
    },
    "cinematic": {
        "positive": [
            "cinematic professional photography with dramatic beautiful lighting",
            "award-winning editorial photography with cinematic quality",
            "high-end photojournalism with cinematic composition and depth",
        ],
        "negative": [
            "a flat poorly lit amateur snapshot",
            "a boring casual phone photo with no cinematic quality",
            "an overexposed or underexposed blurry photo",
        ],
    },
    "composition": {
        "positive": [
            "a photograph with elegant professional composition and framing",
            "a beautifully composed photo with perfect rule of thirds balance",
            "a well-balanced artistic photograph with strong visual flow",
        ],
        "negative": [
            "a poorly composed cluttered photograph with bad framing",
            "a snapshot with distracting background and unbalanced composition",
            "a photo with the subject awkwardly placed at the edge",
        ],
    },
    "invitation_suit": {
        "positive": [
            "an elegant formal portrait perfect for a wedding invitation card",
            "a clean professional photograph suitable for formal print and stationery",
            "a beautiful polished portrait ideal for event invitations",
        ],
        "negative": [
            "a casual unflattering snapshot not suitable for invitations",
            "a blurry dark photo unsuitable for formal printing",
            "a silly or casual photo inappropriate for formal event use",
        ],
    },
    "storytelling": {
        "positive": [
            "a photograph that tells a compelling human story",
            "a decisive moment captured with narrative depth and context",
            "a powerful documentary photo that conveys a rich story",
            "a candid image that captures a memorable life moment",
        ],
        "negative": [
            "a generic posed photo with no story to tell",
            "a meaningless snapshot of nothing in particular",
            "a random photo with no narrative or emotional context",
        ],
    },
}


class SemanticScorer(BaseScorer):
    """
    CLIP-based semantic image scorer.

    Usage:
        SemanticScorer.load_models()   # once at startup
        scorer = SemanticScorer()
        result = scorer.score(Path("photo.jpg"))
    """

    _model: Optional[CLIPModel] = None
    _processor: Optional[CLIPProcessor] = None
    _device: str = "cpu"
    _text_features: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

    @classmethod
    def load_models(cls):
        """Load CLIP and precompute text features. Call once at app startup."""
        if cls._model is not None:
            return

        logger.info(f"Loading CLIP model: {settings.CLIP_MODEL_NAME} on {settings.DEVICE}")
        cls._device = settings.DEVICE
        cls._processor = CLIPProcessor.from_pretrained(settings.CLIP_MODEL_NAME)
        cls._model = CLIPModel.from_pretrained(settings.CLIP_MODEL_NAME).to(cls._device)
        cls._model.eval()

        logger.info("Precomputing CLIP text features for all scoring criteria...")
        cls._precompute_text_features()
        logger.info("SemanticScorer ready ✓")

    @classmethod
    def _precompute_text_features(cls):
        for criterion, prompts in CRITERION_PROMPTS.items():
            pos_feats = cls._encode_texts(prompts["positive"])
            neg_feats = cls._encode_texts(prompts["negative"])
            cls._text_features[criterion] = (pos_feats, neg_feats)

    @classmethod
    def _encode_texts(cls, texts: List[str]) -> torch.Tensor:
        inputs = cls._processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        ).to(cls._device)
        with torch.no_grad():
            feats = cls._model.get_text_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats  # (n_texts, embed_dim)

    @classmethod
    def _encode_image(cls, pil_image: Image.Image) -> torch.Tensor:
        inputs = cls._processor(images=pil_image, return_tensors="pt").to(cls._device)
        with torch.no_grad():
            feats = cls._model.get_image_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats  # (1, embed_dim)

    @classmethod
    def _clip_score(
        cls,
        img_feats: torch.Tensor,
        pos_feats: torch.Tensor,
        neg_feats: torch.Tensor,
    ) -> float:
        """Positive probability via softmax over mean positive and negative similarities. → 0-10."""
        pos_sims = (img_feats @ pos_feats.T).squeeze()
        neg_sims = (img_feats @ neg_feats.T).squeeze()

        pos_mean = pos_sims.mean() if pos_sims.dim() > 0 else pos_sims
        neg_mean = neg_sims.mean() if neg_sims.dim() > 0 else neg_sims

        probs = torch.stack([pos_mean, neg_mean]).softmax(dim=0)
        return round(float(probs[0].item()) * 10, 2)

    def score(self, image_path: Path) -> Dict[str, Any]:
        if self._model is None:
            logger.error("SemanticScorer.load_models() was never called.")
            return {c: 5.0 for c in CRITERION_PROMPTS}

        try:
            pil_img = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.warning(f"Could not open {image_path} for semantic scoring: {e}")
            return {c: 5.0 for c in CRITERION_PROMPTS}

        img_feats = self._encode_image(pil_img)

        return {
            criterion: self._clip_score(img_feats, pos_feats, neg_feats)
            for criterion, (pos_feats, neg_feats) in self._text_features.items()
        }
