"""Base scorer interface."""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict


class BaseScorer(ABC):
    """All scorers implement this interface."""

    @abstractmethod
    def score(self, image_path: Path) -> Dict[str, Any]:
        """
        Score a single image.

        Returns:
            Dict of dimension -> float (0-10 scale) or metadata values.
        """
        ...
