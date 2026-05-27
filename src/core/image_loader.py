"""
Cross-platform image loading.

Resolves paths from:
  - Individual file paths (list)
  - Folder paths (with optional recursive scan)

Works on both Linux (/mnt/nas/...) and Windows (C:\\... or \\\\server\\share\\...).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import List, Optional, Set

from src.config.settings import settings


def _normalize_path(raw: str) -> Path:
    """
    Normalise a path string to a pathlib.Path regardless of OS origin.
    Handles:
      - POSIX paths: /mnt/nas/photo.jpg
      - Windows drive paths: C:\\Photos\\img.jpg
      - UNC paths: \\\\server\\share\\path\\img.jpg
    """
    # Replace forward slashes with OS sep for Windows UNC paths starting with //
    p = raw.strip()
    try:
        return Path(p)
    except Exception:
        # Fallback: strip and try again
        return Path(p.replace("\\", os.sep).replace("/", os.sep))


def _allowed_extension(path: Path, allowed: Set[str]) -> bool:
    return path.suffix.lower() in allowed


def collect_image_paths(
    paths: Optional[List[str]] = None,
    folder: Optional[str] = None,
    recursive: bool = False,
    extensions: Optional[List[str]] = None,
) -> List[Path]:
    """
    Return a deduplicated, sorted list of image Paths from:
    - `paths`: explicit file path list
    - `folder`: directory to scan

    Args:
        paths: absolute file paths (strings, any OS format)
        folder: absolute folder path to scan for images
        recursive: scan sub-folders when folder is given
        extensions: restrict to these extensions (without dot, e.g. ['jpg','png'])

    Returns:
        List of resolved absolute Path objects.

    Raises:
        FileNotFoundError: if a provided path does not exist.
        NotADirectoryError: if folder is not a directory.
    """
    if extensions:
        allowed = {"." + e.lower().lstrip(".") for e in extensions}
    else:
        allowed = settings.SUPPORTED_EXTENSIONS

    collected: Set[Path] = set()

    # --- Explicit file paths ---
    if paths:
        for raw in paths:
            p = _normalize_path(raw)
            if not p.exists():
                raise FileNotFoundError(f"Image path not found: {p}")
            if not p.is_file():
                raise ValueError(f"Path is not a file: {p}")
            if _allowed_extension(p, allowed):
                collected.add(p.resolve())

    # --- Folder scan ---
    if folder:
        folder_path = _normalize_path(folder)
        if not folder_path.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        if not folder_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {folder_path}")

        if recursive:
            glob_pattern = "**/*"
        else:
            glob_pattern = "*"

        for f in folder_path.glob(glob_pattern):
            if f.is_file() and _allowed_extension(f, allowed):
                collected.add(f.resolve())

    return sorted(collected)
