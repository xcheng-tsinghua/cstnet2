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


class Stage2ClassificationDataset(Dataset):
    """Read a class-directory split for Stage 2 part classification."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        n_points: int = 2000,
        classes: dict[str, int] | None = None,
        data_augmentation: bool = False,
    ):
        self.root = Path(root)
        self.split = str(split).lower()
        self.split_dir = self.root / self.split
        if not self.split_dir.is_dir():
            raise FileNotFoundError(
                f"Stage 2 classification split not found: {self.split_dir}"
            )
        self.n_points = int(n_points)
        self.data_augmentation = bool(data_augmentation)

        category_dirs = sorted(
            (path for path in self.split_dir.iterdir() if path.is_dir()),
            key=lambda path: path.name.lower(),
        )
        if classes is None:
            if not category_dirs:
                raise FileNotFoundError(
                    f"no class directories found in {self.split_dir}"
                )
            self.classes = {
                path.name: index for index, path in enumerate(category_dirs)
            }
        else:
            self.classes = dict(classes)
            unknown = sorted(
                path.name for path in category_dirs if path.name not in self.classes
            )
            if unknown:
                raise ValueError(
                    f"unknown classes in {self.split} split: {unknown}"
                )

        self.datapath = []
        for class_name, class_index in sorted(
            self.classes.items(), key=lambda item: item[1]
        ):
            class_dir = self.split_dir / class_name
            if not class_dir.is_dir():
                continue
            for path in discover_txt_files(class_dir):
                self.datapath.append((class_index, path))
        if not self.datapath:
            raise FileNotFoundError(
                f"no Stage 2 classification samples found in {self.split_dir}"
            )
        print(f"Stage 2 classification dataset [{self.split}]: {self.split_dir}")
        print(self.classes)
        print(f"instance all: {len(self.datapath)}")

    def __len__(self):
        return len(self.datapath)

    def __getitem__(self, index):
        class_index, path = self.datapath[index]
        point_set = load_constraint_point_file(
            path,
            task_name="Stage 2 classification",
        )
        point_set = sample_without_replacement(
            point_set,
            self.n_points,
            path=path,
        )
        xyz, pmt, direction, dimension, location, _ = (
            split_constraint_columns(point_set)
        )
        if self.data_augmentation:
            xyz = xyz + np.random.normal(0.0, 0.02, size=xyz.shape)
        return (
            xyz,
            class_index,
            pmt,
            direction,
            dimension,
            location,
        )

    def n_classes(self):
        return len(self.classes)

    @staticmethod
    def create_dataloaders(root, bs, n_points, num_workers, is_sample=False):
        train_set = Stage2ClassificationDataset(
            root=root,
            split="train",
            n_points=n_points,
        )
        test_set = Stage2ClassificationDataset(
            root=root,
            split="test",
            n_points=n_points,
            classes=train_set.classes,
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
            print(Fore.RED + "-> sample the Stage 2 classification dataset")
            train_sampler = RandomSampler(
                train_set,
                num_samples=min(len(train_set), bs * 4),
                replacement=False,
            )
            test_sampler = RandomSampler(
                test_set,
                num_samples=min(len(test_set), bs * 2),
                replacement=False,
            )
            return (
                DataLoader(train_set, sampler=train_sampler, **loader_kwargs),
                DataLoader(test_set, sampler=test_sampler, **loader_kwargs),
            )

        print(Fore.GREEN + "-> create full Stage 2 classification dataloaders")
        return (
            DataLoader(train_set, shuffle=True, **loader_kwargs),
            DataLoader(test_set, shuffle=False, **loader_kwargs),
        )
