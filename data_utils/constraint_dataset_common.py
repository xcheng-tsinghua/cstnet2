from __future__ import annotations

from pathlib import Path

import numpy as np


CONSTRAINT_POINT_COLUMNS = 12


def discover_txt_files(root: str | Path) -> list[Path]:
    """Return every TXT file below root in deterministic path order."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"dataset directory not found: {root}")
    files = sorted(
        (path for path in root.rglob("*") if path.is_file() and path.suffix.lower() == ".txt"),
        key=lambda path: str(path).lower(),
    )
    if not files:
        raise FileNotFoundError(f"no .txt point-cloud files found below: {root}")
    return files


def load_constraint_point_file(path: str | Path, *, task_name: str) -> np.ndarray:
    """Load and validate the shared normal-free 12-column constraint layout."""
    path = Path(path)
    point_set = np.loadtxt(path, dtype=np.float32)
    if point_set.ndim == 1:
        point_set = point_set.reshape(1, -1)
    if point_set.ndim != 2 or point_set.shape[1] != CONSTRAINT_POINT_COLUMNS:
        raise ValueError(
            f"expected {CONSTRAINT_POINT_COLUMNS} columns in {task_name} sample "
            f"{path}, got shape {point_set.shape}"
        )
    if not np.isfinite(point_set).all():
        raise ValueError(f"non-finite value found in {task_name} sample: {path}")
    return point_set


def sample_without_replacement(
    point_set: np.ndarray,
    n_points: int,
    *,
    path: str | Path,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    if n_points <= 0:
        raise ValueError("n_points must be positive")
    if point_set.shape[0] < n_points:
        raise ValueError(
            f"insufficient points in sample {path}: "
            f"current={point_set.shape[0]}, required={n_points}"
        )
    chooser = np.random.choice if rng is None else rng.choice
    indices = chooser(point_set.shape[0], n_points, replace=False)
    return point_set[indices]


def split_constraint_columns(point_set: np.ndarray, is_contain_normal=False):
    """Split xyz, primitive attributes, and primitive-instance affiliation."""
    if is_contain_normal:
        xyz = point_set[:, 0:3]
        pmt = point_set[:, 3].astype(np.int32)
        direction = point_set[:, 4:7]
        dimension = point_set[:, 7]
        normal = point_set[:, 8:11]
        location = point_set[:, 11:14]
        affiliate_idx = point_set[:, 14].astype(np.int32)
    else:
        xyz = point_set[:, 0:3]
        pmt = point_set[:, 3].astype(np.int32)
        direction = point_set[:, 4:7]
        dimension = point_set[:, 7]
        location = point_set[:, 8:11]
        affiliate_idx = point_set[:, 11].astype(np.int32)
    return xyz, pmt, direction, dimension, location, affiliate_idx
