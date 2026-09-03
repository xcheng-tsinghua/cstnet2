from __future__ import annotations

from pathlib import Path

import numpy as np

CONSTRAINT_POINT_COLUMNS = 12
VALID_DIRECTION_PRIMITIVES = (0, 1, 2)
VALID_DIMENSION_PRIMITIVES = (1, 2, 3)


def zero_invalid_constraint_components(
    pmt: np.ndarray,
    direction: np.ndarray,
    dimension: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return attributes with invalid direction and dimension set to zero.

    Validity comes only from the primitive type, never from a numeric sentinel.
    Consequently, legacy ``(0, 0, -1)``/``-1`` files and zero-filled files
    produce the same in-memory representation without rewriting the dataset.
    """
    pmt = np.asarray(pmt)
    direction = np.array(direction, copy=True)
    dimension = np.array(dimension, copy=True)
    if direction.shape != (*pmt.shape, 3):
        raise ValueError(
            "direction shape must equal primitive shape followed by 3; "
            f"got pmt={pmt.shape}, direction={direction.shape}"
        )
    if dimension.shape not in (pmt.shape, (*pmt.shape, 1)):
        raise ValueError(
            "dimension shape must equal primitive shape with an optional "
            f"trailing singleton; got pmt={pmt.shape}, dimension={dimension.shape}"
        )

    direction_valid = np.isin(pmt, VALID_DIRECTION_PRIMITIVES)
    dimension_valid = np.isin(pmt, VALID_DIMENSION_PRIMITIVES)
    direction[~direction_valid] = 0.0
    if dimension.shape == pmt.shape:
        dimension[~dimension_valid] = 0.0
    else:
        dimension[~dimension_valid, :] = 0.0
    return direction, dimension


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


def load_constraint_point_file(
    path: str | Path,
    *,
    task_name: str,
    allow_extra_columns: bool = False,
) -> np.ndarray:
    """Load the 12-column constraint core, optionally discarding later columns."""
    path = Path(path)
    point_set = np.loadtxt(path, dtype=np.float32)
    if point_set.ndim == 1:
        point_set = point_set.reshape(1, -1)
    if point_set.ndim != 2:
        raise ValueError(
            f"expected a 2D array in {task_name} sample "
            f"{path}, got shape {point_set.shape}"
        )
    column_count = point_set.shape[1]
    valid_column_count = (
        column_count >= CONSTRAINT_POINT_COLUMNS
        if allow_extra_columns
        else column_count == CONSTRAINT_POINT_COLUMNS
    )
    if not valid_column_count:
        expectation = (
            f"at least {CONSTRAINT_POINT_COLUMNS}"
            if allow_extra_columns
            else str(CONSTRAINT_POINT_COLUMNS)
        )
        raise ValueError(
            f"expected {expectation} columns in {task_name} sample "
            f"{path}, got shape {point_set.shape}"
        )
    if allow_extra_columns and column_count > CONSTRAINT_POINT_COLUMNS:
        point_set = point_set[:, :CONSTRAINT_POINT_COLUMNS]
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
    # if n_points <= 0:
    #     raise ValueError("n_points must be positive")
    # if point_set.shape[0] < n_points:
    #     raise ValueError(
    #         f"insufficient points in sample {path}: "
    #         f"current={point_set.shape[0]}, required={n_points}"
    #     )
    # chooser = np.random.choice if rng is None else rng.choice
    # indices = chooser(point_set.shape[0], n_points, replace=False)
    # return point_set[indices]
    if n_points <= 0:
        raise ValueError("n_points must be positive")

    current_points = point_set.shape[0]

    if current_points == 0:
        raise ValueError(f"empty point set in sample {path}")

    chooser = np.random.choice if rng is None else rng.choice

    if current_points >= n_points:
        indices = chooser(current_points, n_points, replace=False)
        return point_set[indices]

    # warnings.warn(
    #     f"insufficient points in sample {path}: "
    #     f"current={current_points}, required={n_points}; "
    #     f"duplicate points will be sampled",
    #     category=RuntimeWarning,
    #     stacklevel=2,
    # )

    # 先保留所有原始点，再通过有放回采样补足缺少的点。
    additional_count = n_points - current_points
    additional_indices = chooser(
        current_points,
        additional_count,
        replace=True,
    )

    indices = np.concatenate(
        [
            np.arange(current_points),
            additional_indices,
        ]
    )

    # 打乱顺序，避免重复点集中在数组末尾。
    if rng is None:
        np.random.shuffle(indices)
    else:
        rng.shuffle(indices)

    return point_set[indices]


def split_constraint_columns(
    point_set: np.ndarray,
    *,
    canonicalize_invalid: bool = True,
):
    """Split the normal-free 12-column constraint layout."""
    xyz = point_set[:, 0:3]
    pmt = point_set[:, 3].astype(np.int32)
    direction = point_set[:, 4:7]
    dimension = point_set[:, 7]
    location = point_set[:, 8:11]
    affiliate_idx = point_set[:, 11].astype(np.int32)
    if canonicalize_invalid:
        direction, dimension = zero_invalid_constraint_components(
            pmt, direction, dimension
        )
    return xyz, pmt, direction, dimension, location, affiliate_idx
