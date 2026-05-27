"""
Duplicate Scorer using perceptual hashing (pHash).

Groups near-identical images via Hamming distance on their perceptual hash.
Within each group the first-encountered image is the "representative" (original);
all others are flagged as duplicates with a reference to it.

Returns:
    dup_map: Dict[path_str, Optional[str]]
        None  → image is unique or the group representative
        str   → path string of the representative (= this image is a duplicate)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import imagehash
from PIL import Image

from src.config.settings import settings

logger = logging.getLogger(__name__)


class DuplicateScorer:
    """Perceptual-hash based near-duplicate detector."""

    def find_duplicates(
        self,
        image_paths: List[Path],
        threshold: Optional[int] = None,
    ) -> Dict[str, Optional[str]]:
        """
        Detect near-duplicate images across the entire batch.

        Args:
            image_paths: all image paths to compare
            threshold:   max pHash Hamming distance to call two images duplicates
                         (default: settings.DUPLICATE_HASH_THRESHOLD)

        Returns:
            Mapping of path_str → None (original/unique) or representative path_str.
        """
        if threshold is None:
            threshold = settings.DUPLICATE_HASH_THRESHOLD

        # --- Compute pHash for each image ---
        hashes: Dict[str, imagehash.ImageHash] = {}
        for p in image_paths:
            try:
                img = Image.open(p).convert("RGB")
                hashes[str(p)] = imagehash.phash(img)
            except Exception as e:
                logger.warning(f"Could not hash {p}: {e}")

        path_strs = list(hashes.keys())
        n = len(path_strs)

        # --- Union-Find grouping ---
        parent = list(range(n))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i: int, j: int):
            pi, pj = find(i), find(j)
            if pi != pj:
                parent[pj] = pi

        for i in range(n):
            for j in range(i + 1, n):
                if (hashes[path_strs[i]] - hashes[path_strs[j]]) <= threshold:
                    union(i, j)

        # --- Build result map ---
        groups: Dict[int, List[int]] = {}
        for i in range(n):
            root = find(i)
            groups.setdefault(root, []).append(i)

        dup_map: Dict[str, Optional[str]] = {}
        for root, members in groups.items():
            representative = path_strs[root]
            for idx in members:
                p_str = path_strs[idx]
                dup_map[p_str] = None if idx == root else representative

        # Images that failed hashing are treated as unique
        for p in image_paths:
            ps = str(p)
            if ps not in dup_map:
                dup_map[ps] = None

        return dup_map
