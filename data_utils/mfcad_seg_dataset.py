from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler

DEFAULT_LABEL_MAP = Path(__file__).with_name("mfcad_label_map.json")
COMPONENT_NAMES = ("primitive_type", "direction", "dimension", "continuity", "location")
EXPECTED_COLUMNS = 17


def load_label_map(path: str | os.PathLike[str] = DEFAULT_LABEL_MAP) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"MFCAD++ label map not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    labels = metadata.get("labels")
    if not isinstance(labels, list) or not labels:
        raise ValueError(f"label map must contain a non-empty 'labels' list: {path}")
    ids = [int(label["id"]) for label in labels]
    if ids != list(range(len(labels))):
        raise ValueError("MFCAD++ label ids must be contiguous and ordered from zero")
    if len({str(label["name"]) for label in labels}) != len(labels):
        raise ValueError("MFCAD++ label names must be unique")
    return metadata


def _numeric_path_key(path: Path) -> tuple[int, int | str]:
    try:
        return 0, int(path.stem)
    except ValueError:
        return 1, path.stem


def _resolve_split_dir(root: Path, split: str) -> tuple[str, Path]:
    normalized = split.lower()
    candidates = ("val", "validation") if normalized in {"val", "validation"} else (normalized,)
    for candidate in candidates:
        path = root / candidate
        if path.is_dir():
            return ("val" if normalized in {"val", "validation"} else normalized), path
    raise FileNotFoundError(
        f"MFCAD++ split {split!r} not found below {root}; tried {', '.join(candidates)}"
    )


class DistributedEvalSampler(Sampler[int]):
    """Shard evaluation data across ranks without padding or duplicate samples."""

    def __init__(self, dataset: Dataset):
        if not (dist.is_available() and dist.is_initialized()):
            raise RuntimeError("DistributedEvalSampler requires an initialized process group")
        self.dataset = dataset
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.world_size))

    def __len__(self) -> int:
        if self.rank >= len(self.dataset):
            return 0
        return (len(self.dataset) - 1 - self.rank) // self.world_size + 1


class MFCADSegmentationDataset(Dataset):
    """Read normalized 17-column MFCAD++ point clouds with Stage 1 constraints."""

    def __init__(
        self,
        root: str | os.PathLike[str],
        split: str,
        n_points: int | None = 2048,
        label_map_path: str | os.PathLike[str] = DEFAULT_LABEL_MAP,
        use_npy_cache: bool = False,
        validate_face_labels: bool = True,
    ):
        self.root = Path(root)
        self.split, self.split_dir = _resolve_split_dir(self.root, split)
        self.n_points = n_points
        if n_points is not None and n_points <= 0:
            raise ValueError("n_points must be positive or None")
        self.use_npy_cache = bool(use_npy_cache)
        self.validate_face_labels = bool(validate_face_labels)
        self.label_map_path = str(Path(label_map_path).resolve())
        self.label_map = load_label_map(label_map_path)
        self.num_classes = len(self.label_map["labels"])
        self.files = sorted(self.split_dir.glob("*.txt"), key=_numeric_path_key)
        if not self.files:
            raise FileNotFoundError(f"no point cloud .txt files found in {self.split_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def _load_array(self, path: Path) -> np.ndarray:
        if self.use_npy_cache:
            npy_path = Path(str(path) + ".npy")
            if npy_path.is_file():
                array = np.asarray(np.load(npy_path, mmap_mode="r"), dtype=np.float32)
            else:
                array = np.loadtxt(path, dtype=np.float32)
                temporary = Path(str(npy_path) + f".tmp.{os.getpid()}.npy")
                try:
                    np.save(temporary, array)
                    os.replace(temporary, npy_path)
                finally:
                    if temporary.exists():
                        temporary.unlink()
        else:
            array = np.loadtxt(path, dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.ndim != 2 or array.shape[1] != EXPECTED_COLUMNS:
            raise ValueError(
                f"expected {EXPECTED_COLUMNS} columns in {path}, got shape {array.shape}"
            )
        if not np.isfinite(array).all():
            raise ValueError(f"non-finite value found in MFCAD++ sample: {path}")
        return array

    def _point_indices(self, count: int) -> np.ndarray:
        if self.n_points is None or self.n_points == count:
            return np.arange(count, dtype=np.int64)
        if self.split == "train":
            return np.random.choice(count, self.n_points, replace=count < self.n_points)
        if self.n_points < count:
            return np.linspace(0, count - 1, self.n_points, dtype=np.int64)
        repeats = int(np.ceil(self.n_points / count))
        return np.tile(np.arange(count, dtype=np.int64), repeats)[: self.n_points]

    def __getitem__(self, index: int) -> dict[str, Any]:
        path = self.files[index]
        point_set = self._load_array(path)
        point_set = point_set[self._point_indices(point_set.shape[0])]

        xyz = point_set[:, 0:3].copy()
        raw_pmt = point_set[:, 3]
        pmt = raw_pmt.astype(np.int64)
        if not np.array_equal(raw_pmt, pmt.astype(raw_pmt.dtype)) or np.any((pmt < 0) | (pmt > 4)):
            raise ValueError(f"invalid primitive type in {path}; expected integer ids in [0, 4]")

        direction = point_set[:, 4:7].copy()
        dimension = point_set[:, 7:8].copy()
        continuity = point_set[:, 8:11].copy()
        location = point_set[:, 11:14].copy()
        raw_face_ids = point_set[:, 15]
        raw_labels = point_set[:, 16]
        face_ids = raw_face_ids.astype(np.int64)
        labels = raw_labels.astype(np.int64)
        if not np.array_equal(raw_face_ids, face_ids.astype(raw_face_ids.dtype)):
            raise ValueError(f"non-integer Face id found in {path}")
        if not np.array_equal(raw_labels, labels.astype(raw_labels.dtype)):
            raise ValueError(f"non-integer segmentation label found in {path}")
        if np.any((labels < 0) | (labels >= self.num_classes)):
            observed = np.unique(labels).tolist()
            raise ValueError(
                f"segmentation labels {observed} in {path} are outside label-map range "
                f"[0, {self.num_classes - 1}]"
            )

        primitive_valid = np.ones_like(pmt, dtype=bool)
        direction_valid = np.isin(pmt, (0, 1, 2))
        dimension_valid = np.isin(pmt, (1, 2, 3))
        continuity_valid = np.ones_like(pmt, dtype=bool)
        location_valid = np.isin(pmt, (0, 1, 2, 3))
        constraint_masks = np.stack(
            [primitive_valid, direction_valid, dimension_valid, continuity_valid, location_valid],
            axis=-1,
        )

        # Remove all invalid sentinels before the network sees the values. The
        # separate masks are propagated through shared FPS and used by cross-attention.
        direction[~direction_valid] = 0.0
        dimension[~dimension_valid] = 0.0
        location[~location_valid] = 0.0
        primitive_one_hot = np.eye(5, dtype=np.float32)[pmt]
        constraints = np.concatenate(
            [primitive_one_hot, direction, dimension, continuity, location],
            axis=-1,
        ).astype(np.float32, copy=False)

        if self.validate_face_labels:
            for face_id in np.unique(face_ids):
                face_labels = np.unique(labels[face_ids == face_id])
                if face_labels.size != 1:
                    raise ValueError(
                        f"face {face_id} in {path} has inconsistent labels {face_labels.tolist()}"
                    )

        return {
            "xyz": torch.from_numpy(xyz.astype(np.float32, copy=False)),
            "constraints": torch.from_numpy(constraints),
            "constraint_masks": torch.from_numpy(constraint_masks.astype(np.bool_)),
            "labels": torch.from_numpy(labels),
            "face_ids": torch.from_numpy(face_ids),
            "sample_id": path.stem,
            "path": str(path),
        }

    @staticmethod
    def create_dataloaders(
        root: str | os.PathLike[str],
        batch_size: int = 4,
        n_points: int | None = 2048,
        num_workers: int = 4,
        label_map_path: str | os.PathLike[str] = DEFAULT_LABEL_MAP,
        use_npy_cache: bool = False,
        distributed: bool = False,
        pin_memory: bool = True,
        drop_last: bool = True,
    ) -> tuple[DataLoader, DataLoader, DataLoader | None]:
        common = {
            "root": root,
            "n_points": n_points,
            "label_map_path": label_map_path,
            "use_npy_cache": use_npy_cache,
        }
        train_dataset = MFCADSegmentationDataset(split="train", **common)
        val_dataset = MFCADSegmentationDataset(split="val", **common)
        try:
            test_dataset = MFCADSegmentationDataset(split="test", **common)
        except FileNotFoundError:
            test_dataset = None

        def make_loader(dataset: MFCADSegmentationDataset, train: bool) -> DataLoader:
            if distributed and train:
                sampler = DistributedSampler(dataset, shuffle=True)
            elif distributed:
                sampler = DistributedEvalSampler(dataset)
            else:
                sampler = None
            return DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=train and sampler is None,
                sampler=sampler,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=train and drop_last,
                persistent_workers=num_workers > 0,
            )

        train_loader = make_loader(train_dataset, train=True)
        val_loader = make_loader(val_dataset, train=False)
        test_loader = make_loader(test_dataset, train=False) if test_dataset is not None else None
        return train_loader, val_loader, test_loader
