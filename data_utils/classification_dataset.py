from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
from colorama import Fore
from torch.utils.data import DataLoader, Dataset, RandomSampler

from data_utils.constraint_dataset_common import (
    discover_txt_files,
    load_constraint_point_file,
    sample_without_replacement,
    split_constraint_columns,
)


SPLIT_FILE_NAME = "split_file.json"
DEFAULT_TEST_RATIO = 0.2
DEFAULT_SPLIT_SEED = 42


def _category_directories(root: Path) -> list[Path]:
    return sorted(
        (path for path in root.iterdir() if path.is_dir()),
        key=lambda path: path.name.lower(),
    )


def _existing_split_file(root: Path) -> Path | None:
    """Find the canonical manifest or the extensionless legacy spelling."""
    for name in (SPLIT_FILE_NAME, "split_file"):
        path = root / name
        if path.is_file():
            return path
    return None


def _relative_sample_path(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _create_split_file(
    root: Path,
    category_dirs: list[Path],
    *,
    test_ratio: float,
    split_seed: int,
) -> Path:
    if not 0.0 < test_ratio < 1.0:
        raise ValueError(f"test_ratio must be between 0 and 1, got {test_ratio}")

    rng = random.Random(split_seed)
    train_paths: list[str] = []
    test_paths: list[str] = []
    for category_dir in category_dirs:
        paths = discover_txt_files(category_dir)
        rng.shuffle(paths)

        # A class must remain represented in training. When possible, also put
        # at least one sample from it in the test set.
        if len(paths) == 1:
            test_count = 0
        else:
            test_count = min(
                len(paths) - 1,
                max(1, math.ceil(len(paths) * test_ratio)),
            )
        test_paths.extend(
            _relative_sample_path(root, path) for path in paths[:test_count]
        )
        train_paths.extend(
            _relative_sample_path(root, path) for path in paths[test_count:]
        )

    if not test_paths:
        raise ValueError(
            "cannot create a classification test split: every class has only "
            "one sample; add at least one more sample to any class"
        )

    manifest = {
        "version": 1,
        "test_ratio": float(test_ratio),
        "seed": int(split_seed),
        "train": sorted(train_paths, key=str.lower),
        "test": sorted(test_paths, key=str.lower),
    }
    split_file = root / SPLIT_FILE_NAME
    split_file.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Created classification split file: {split_file} "
        f"(train={len(train_paths)}, test={len(test_paths)}, "
        f"requested_test_ratio={test_ratio:.1%}, seed={split_seed})"
    )
    return split_file


def _read_split_file(root: Path, split_file: Path) -> dict[str, list[Path]]:
    try:
        manifest: Any = json.loads(split_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"invalid classification split file {split_file}: {error}"
        ) from error

    if not isinstance(manifest, dict):
        raise ValueError(
            "classification split file must contain a JSON object: "
            f"{split_file}"
        )

    result: dict[str, list[Path]] = {}
    seen: set[str] = set()
    resolved_root = root.resolve()
    for split in ("train", "test"):
        entries = manifest.get(split)
        if not isinstance(entries, list) or not entries:
            raise ValueError(
                f"classification split file field {split!r} must be a non-empty list: "
                f"{split_file}"
            )
        result[split] = []
        for entry in entries:
            if not isinstance(entry, str) or not entry.strip():
                raise ValueError(
                    "invalid path in classification split file field "
                    f"{split!r}: {entry!r}"
                )
            relative = Path(entry)
            path = (root / relative).resolve()
            try:
                path.relative_to(resolved_root)
            except ValueError as error:
                raise ValueError(
                    f"classification split path escapes the dataset root: {entry!r}"
                ) from error
            key = str(path).lower()
            if key in seen:
                raise ValueError(
                    f"duplicate classification sample across split_file entries: {entry!r}"
                )
            seen.add(key)
            if not path.is_file() or path.suffix.lower() != ".txt":
                raise FileNotFoundError(
                    f"classification sample from {split_file.name} not found: {path}"
                )
            result[split].append(path)
    return result


def _flat_layout(
    root: Path,
    *,
    test_ratio: float,
    split_seed: int,
) -> tuple[dict[str, int], dict[str, list[Path]], Path]:
    split_file = _existing_split_file(root)
    if split_file is None:
        category_dirs = _category_directories(root)
        if not category_dirs:
            raise FileNotFoundError(f"no class directories found in {root}")
        split_file = _create_split_file(
            root,
            category_dirs,
            test_ratio=test_ratio,
            split_seed=split_seed,
        )
    split_paths = _read_split_file(root, split_file)

    resolved_root = root.resolve()
    split_classes: dict[str, set[str]] = {"train": set(), "test": set()}
    for split, paths in split_paths.items():
        for path in paths:
            relative_parts = path.relative_to(resolved_root).parts
            if len(relative_parts) < 2:
                raise ValueError(
                    "classification split samples must be below a top-level "
                    f"class directory: {path}"
                )
            split_classes[split].add(relative_parts[0])

    class_names = sorted(
        split_classes["train"] | split_classes["test"], key=str.lower
    )
    missing_from_train = sorted(
        split_classes["test"] - split_classes["train"], key=str.lower
    )
    if missing_from_train:
        raise ValueError(
            "every classification class must have a training sample; classes "
            f"found only in test: {missing_from_train}"
        )
    classes = {name: index for index, name in enumerate(class_names)}
    return classes, split_paths, split_file


class Stage2ClassificationDataset(Dataset):
    """Read Stage 2 classification data from split or class-root layouts.

    Supported layouts are ``root/{train,test}/class/**/*.txt`` and
    ``root/class/**/*.txt``. The latter is governed by ``split_file.json``.
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        n_points: int = 2000,
        classes: dict[str, int] | None = None,
        data_augmentation: bool = False,
        test_ratio: float = DEFAULT_TEST_RATIO,
        split_seed: int = DEFAULT_SPLIT_SEED,
    ):
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"dataset directory not found: {self.root}")
        self.split = str(split).lower()
        if self.split not in {"train", "test"}:
            raise ValueError(
                "classification split must be 'train' or 'test', "
                f"got {split!r}"
            )
        self.n_points = int(n_points)
        self.data_augmentation = bool(data_augmentation)

        train_dir = self.root / "train"
        test_dir = self.root / "test"
        uses_split_directories = train_dir.is_dir() and test_dir.is_dir()
        if uses_split_directories:
            self.split_dir = self.root / self.split
            category_dirs = _category_directories(self.split_dir)
            discovered_classes = {
                path.name: index for index, path in enumerate(category_dirs)
            }
            paths_by_class = {
                path.name: discover_txt_files(path) for path in category_dirs
            }
            source_description = str(self.split_dir)
        else:
            discovered_classes, split_paths, split_file = _flat_layout(
                self.root,
                test_ratio=float(test_ratio),
                split_seed=int(split_seed),
            )
            paths_by_class = {name: [] for name in discovered_classes}
            resolved_root = self.root.resolve()
            for path in split_paths[self.split]:
                class_name = path.relative_to(resolved_root).parts[0]
                paths_by_class[class_name].append(path)
            source_description = f"{self.root} ({split_file.name})"

        if classes is None:
            if not discovered_classes:
                raise FileNotFoundError(f"no class directories found in {self.root}")
            self.classes = discovered_classes
        else:
            self.classes = dict(classes)
            unknown = sorted(
                name for name in discovered_classes if name not in self.classes
            )
            if unknown:
                raise ValueError(f"unknown classes in {self.split} split: {unknown}")

        self.datapath: list[tuple[int, Path]] = []
        for class_name, class_index in sorted(
            self.classes.items(), key=lambda item: item[1]
        ):
            for path in paths_by_class.get(class_name, []):
                self.datapath.append((class_index, path))
        if not self.datapath:
            raise FileNotFoundError(
                f"no Stage 2 classification samples found in {source_description} "
                f"for split {self.split!r}"
            )
        print(f"Stage 2 classification dataset [{self.split}]: {source_description}")
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
    def create_dataloaders(
        root,
        bs,
        n_points,
        num_workers,
        is_sample=False,
        test_ratio=DEFAULT_TEST_RATIO,
        split_seed=DEFAULT_SPLIT_SEED,
    ):
        train_set = Stage2ClassificationDataset(
            root=root,
            split="train",
            n_points=n_points,
            test_ratio=test_ratio,
            split_seed=split_seed,
        )
        test_set = Stage2ClassificationDataset(
            root=root,
            split="test",
            n_points=n_points,
            classes=train_set.classes,
            test_ratio=test_ratio,
            split_seed=split_seed,
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
