from __future__ import annotations

from pathlib import Path

import numpy as np
from colorama import Fore
from torch.utils.data import DataLoader, Dataset, RandomSampler

from data_utils.constraint_dataset_common import (
    discover_txt_files,
    load_constraint_point_file,
    sample_without_replacement,
    split_constraint_columns,
)


class Stage1ConstraintDataset(Dataset):
    """Read all normal-free 12-column Stage 1 samples below one directory."""

    def __init__(
        self,
        root: str | Path,
        n_points: int = 2000,
        data_augmentation: bool = False,
        sample_seed: int | None = None,
    ):
        self.root = Path(root)
        self.n_points = int(n_points)
        self.data_augmentation = bool(data_augmentation)
        self.sample_seed = sample_seed
        self.files = discover_txt_files(self.root)
        print(f"Stage 1 constraint dataset: {self.root}")
        print(f"instance all: {len(self.files)}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        point_set = load_constraint_point_file(path, task_name="Stage 1")
        rng = None
        if self.sample_seed is not None:
            rng = np.random.default_rng(int(self.sample_seed) + int(index))
        point_set = sample_without_replacement(
            point_set,
            self.n_points,
            path=path,
            rng=rng,
        )
        xyz, pmt, direction, dimension, location, affiliate_idx = (
            split_constraint_columns(point_set, True)
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
    ):
        dataset = Stage1ConstraintDataset(
            root=root,
            n_points=n_points,
            sample_seed=sample_seed,
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
