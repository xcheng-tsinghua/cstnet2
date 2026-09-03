from __future__ import annotations

from bisect import bisect_right
from pathlib import Path

import numpy as np
from colorama import Fore
from torch.utils.data import DataLoader, Dataset, RandomSampler

from data_utils.constraint_dataset_common import (
    discover_txt_files,
    load_constraint_point_file,
    sample_without_replacement,
    split_constraint_columns,
    zero_invalid_constraint_components,
)
from data_utils.stage1_h5 import (
    STAGE1_H5_FIELDS,
    STAGE1_H5_FORMAT,
    STAGE1_H5_VERSION,
    discover_stage1_h5_files,
    import_h5py,
)


def _has_at_least_n_points(path: Path, n_points: int) -> bool:
    """Return whether a TXT sample contains at least ``n_points`` data rows."""
    point_count = 0
    with path.open("r", encoding="utf-8", errors="replace") as point_file:
        for line in point_file:
            # Match numpy.loadtxt's handling of blank lines and ``#`` comments.
            if not line.split("#", maxsplit=1)[0].strip():
                continue
            point_count += 1
            if point_count >= n_points:
                return True
    return False


class Stage1ConstraintDataset(Dataset):
    """Read Stage 1 TXT samples or typed HDF5 shards."""

    def __init__(
        self,
        root: str | Path,
        n_points: int = 2048,
        data_augmentation: bool = False,
        sample_seed: int | None = None,
        storage_format: str = "auto",
    ):
        self.root = Path(root)
        self.n_points = int(n_points)
        if self.n_points <= 0:
            raise ValueError("n_points must be positive")
        self.data_augmentation = bool(data_augmentation)
        self.sample_seed = sample_seed
        if storage_format not in {"auto", "txt", "h5"}:
            raise ValueError("storage_format must be one of: auto, txt, h5")
        self.storage_format = self._resolve_storage_format(storage_format)
        self.files: list[Path] = []
        self.h5_files: list[Path] = []
        self._h5_valid_indices: list[np.ndarray] = []
        self._h5_cumulative_samples: list[int] = []
        self._h5_handles = {}

        if self.storage_format == "txt":
            discovered_files = discover_txt_files(self.root)
            self.files = [
                path
                for path in discovered_files
                if _has_at_least_n_points(path, self.n_points)
            ]
            skipped_count = len(discovered_files) - len(self.files)
            sample_count = len(self.files)
        else:
            skipped_count, sample_count = self._index_h5_shards()

        if sample_count == 0:
            raise ValueError(
                f"no Stage 1 samples below {self.root} contain at least "
                f"{self.n_points} points; skipped {skipped_count} sample(s)"
            )
        print(f"Stage 1 constraint dataset: {self.root}")
        print(f"storage format: {self.storage_format}")
        print(f"instance all: {sample_count}")
        if skipped_count:
            print(
                f"skipped {skipped_count} sample(s) with fewer than "
                f"{self.n_points} points"
            )

    def _resolve_storage_format(self, requested_format: str) -> str:
        if requested_format != "auto":
            return requested_format
        if discover_stage1_h5_files(self.root):
            return "h5"
        return "txt"

    def _index_h5_shards(self) -> tuple[int, int]:
        h5py = import_h5py()
        self.h5_files = discover_stage1_h5_files(self.root)
        if not self.h5_files:
            raise FileNotFoundError(f"no Stage 1 .h5 files found below: {self.root}")
        skipped_count = 0
        total_valid = 0
        for path in self.h5_files:
            with h5py.File(path, "r") as h5_file:
                if h5_file.attrs.get("format") != STAGE1_H5_FORMAT:
                    raise ValueError(f"not a cstnet2 Stage 1 HDF5 file: {path}")
                if h5_file.attrs.get("format_version") != STAGE1_H5_VERSION:
                    raise ValueError(f"unsupported Stage 1 HDF5 version in: {path}")
                missing = [name for name in STAGE1_H5_FIELDS if name not in h5_file]
                if missing:
                    raise ValueError(f"missing HDF5 fields {missing} in {path}")
                offsets = np.asarray(h5_file["offsets"], dtype=np.int64)
                if (
                    offsets.ndim != 1
                    or len(offsets) == 0
                    or offsets[0] != 0
                    or np.any(np.diff(offsets) < 0)
                ):
                    raise ValueError(f"invalid offsets in Stage 1 HDF5 file: {path}")
                total_points = int(offsets[-1])
                point_fields = STAGE1_H5_FIELDS[1:-1]
                if any(len(h5_file[name]) != total_points for name in point_fields):
                    raise ValueError(f"HDF5 field lengths do not match offsets in: {path}")
                point_counts = np.diff(offsets)
                valid_indices = np.flatnonzero(point_counts >= self.n_points).astype(
                    np.int64, copy=False
                )
                skipped_count += int(len(point_counts) - len(valid_indices))
                total_valid += int(len(valid_indices))
                self._h5_valid_indices.append(valid_indices)
                self._h5_cumulative_samples.append(total_valid)
        return skipped_count, total_valid

    def __len__(self):
        if self.storage_format == "txt":
            return len(self.files)
        return self._h5_cumulative_samples[-1]

    def _h5_handle(self, shard_index: int):
        handle = self._h5_handles.get(shard_index)
        if handle is None:
            handle = import_h5py().File(self.h5_files[shard_index], "r")
            self._h5_handles[shard_index] = handle
        return handle

    def _load_h5_sample(self, index: int) -> tuple[np.ndarray, ...]:
        shard_index = bisect_right(self._h5_cumulative_samples, index)
        previous_total = (
            0 if shard_index == 0 else self._h5_cumulative_samples[shard_index - 1]
        )
        local_rank = index - previous_total
        local_index = int(self._h5_valid_indices[shard_index][local_rank])
        h5_file = self._h5_handle(shard_index)
        start = int(h5_file["offsets"][local_index])
        stop = int(h5_file["offsets"][local_index + 1])
        fields = tuple(
            np.asarray(h5_file[field_name][start:stop])
            for field_name in (
                "xyz",
                "pmt",
                "direction",
                "dimension",
                "location",
                "affiliate_idx",
            )
        )
        return (
            fields[0],
            fields[1].astype(np.int32, copy=False),
            fields[2],
            fields[3],
            fields[4],
            fields[5].astype(np.int32, copy=False),
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_h5_handles"] = {}
        return state

    def close(self) -> None:
        """Close HDF5 handles owned by the current process."""
        handles = getattr(self, "_h5_handles", {})
        for handle in handles.values():
            try:
                handle.close()
            except Exception:
                pass
        handles.clear()

    def __del__(self):
        self.close()

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        if self.storage_format == "txt":
            path = self.files[index]
            point_set = load_constraint_point_file(path, task_name="Stage 1")
            fields = split_constraint_columns(point_set)
        else:
            fields = self._load_h5_sample(index)
            path = f"HDF5 sample {index} below {self.root}"
        rng = None
        if self.sample_seed is not None:
            rng = np.random.default_rng(int(self.sample_seed) + int(index))
        row_indices = sample_without_replacement(
            np.arange(fields[0].shape[0]),
            self.n_points,
            path=path,
            rng=rng,
        )
        xyz, pmt, direction, dimension, location, affiliate_idx = (
            field[row_indices] for field in fields
        )
        # HDF5 shards can contain either legacy or zero sentinels. TXT samples
        # have already passed through this same idempotent canonicalization.
        direction, dimension = zero_invalid_constraint_components(
            pmt, direction, dimension
        )
        if self.data_augmentation:
            xyz = xyz + np.random.normal(0.0, 0.02, size=xyz.shape)
        return xyz, pmt, direction, dimension, location, affiliate_idx

    @staticmethod
    def create_dataloader(
        root,
        bs,
        n_points,
        num_workers,
        shuffle,
        is_sample=False,
        sample_seed=None,
        storage_format="auto",
    ):
        dataset = Stage1ConstraintDataset(
            root=root,
            n_points=n_points,
            sample_seed=sample_seed,
            storage_format=storage_format,
        )
        loader_kwargs = {
            "batch_size": bs,
            "num_workers": num_workers,
            "pin_memory": True,
            "drop_last": False,
        }
        if num_workers > 0:
            loader_kwargs.update({
                "persistent_workers": True,
                "prefetch_factor": 4,
            })

        if is_sample:
            sample_batches = 4 if shuffle else 2
            sample_count = min(len(dataset), bs * sample_batches)
            print(
                Fore.RED
                + f"-> sample {sample_count}/{len(dataset)} files for Stage 1 debug"
            )
            sampler = RandomSampler(
                dataset,
                num_samples=sample_count,
                replacement=False,
            )
            return DataLoader(dataset, sampler=sampler, **loader_kwargs)

        print(
            Fore.GREEN
            + f"-> create full Stage 1 dataloader: files={len(dataset)}, "
            f"shuffle={shuffle}"
        )
        return DataLoader(dataset, shuffle=shuffle, **loader_kwargs)
