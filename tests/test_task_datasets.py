from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

import train_cls
from data_utils.classification_dataset import Stage2ClassificationDataset
from data_utils.mfcad_seg_dataset import Stage2SegmentationDataset
from data_utils.stage1_dataset import Stage1ConstraintDataset


def _constraint_sample(point_count=8):
    sample = np.zeros((point_count, 12), dtype=np.float32)
    sample[:, 0:3] = np.arange(point_count * 3).reshape(point_count, 3)
    sample[:, 3] = np.arange(point_count) % 5
    sample[:, 4] = 1.0
    sample[:, 7] = 0.5
    sample[:, 11] = np.arange(point_count) // 2
    return sample


class TaskDatasetSeparationTest(unittest.TestCase):
    def test_three_tasks_expose_distinct_dataset_classes(self):
        self.assertIsNot(Stage1ConstraintDataset, Stage2ClassificationDataset)
        self.assertIsNot(Stage1ConstraintDataset, Stage2SegmentationDataset)
        self.assertIsNot(Stage2ClassificationDataset, Stage2SegmentationDataset)

    def test_classification_uses_stable_train_class_mapping_for_test(self):
        with tempfile.TemporaryDirectory(dir=".") as temporary:
            root = Path(temporary)
            for split in ("train", "test"):
                for class_name in ("Bolts", "Clamps"):
                    class_dir = root / split / class_name
                    class_dir.mkdir(parents=True)
                    np.savetxt(class_dir / "sample.txt", _constraint_sample())

            train_loader, test_loader = (
                Stage2ClassificationDataset.create_dataloaders(
                    root=root,
                    bs=2,
                    n_points=4,
                    num_workers=0,
                )
            )
            train_batch = next(iter(train_loader))
            test_batch = next(iter(test_loader))

        self.assertIsInstance(train_loader.dataset, Stage2ClassificationDataset)
        self.assertIsInstance(test_loader.dataset, Stage2ClassificationDataset)
        self.assertEqual(train_loader.dataset.classes, test_loader.dataset.classes)
        self.assertEqual(len(train_batch), 6)
        self.assertEqual(len(test_batch), 6)
        constraints = train_cls.constraints_from_dataset_batch(
            train_batch,
            torch.device("cpu"),
        )
        self.assertEqual(tuple(constraints.shape), (2, 4, 12))

    def test_classification_class_root_layout_creates_and_reuses_split_file(self):
        with tempfile.TemporaryDirectory(dir=".") as temporary:
            root = Path(temporary)
            for class_name in ("Bolts", "Clamps"):
                for index in range(5):
                    nested = root / class_name / "family" / f"variant_{index % 2}"
                    nested.mkdir(parents=True, exist_ok=True)
                    np.savetxt(nested / f"sample_{index}.txt", _constraint_sample())

            train_loader, test_loader = Stage2ClassificationDataset.create_dataloaders(
                root=root,
                bs=2,
                n_points=4,
                num_workers=0,
            )
            split_file = root / "split_file.json"
            self.assertTrue(split_file.is_file())
            manifest = json.loads(split_file.read_text(encoding="utf-8"))
            self.assertEqual(len(manifest["train"]), 8)
            self.assertEqual(len(manifest["test"]), 2)
            self.assertEqual(len(train_loader.dataset), 8)
            self.assertEqual(len(test_loader.dataset), 2)
            self.assertEqual(
                train_loader.dataset.classes,
                {"Bolts": 0, "Clamps": 1},
            )
            self.assertTrue(
                all(
                    len(path.relative_to(root.resolve()).parts) > 2
                    for _, path in train_loader.dataset.datapath
                )
            )

            # A persisted manifest is authoritative: newly discovered files do
            # not silently change a reproducible split.
            new_file = root / "Bolts" / "new_sample.txt"
            np.savetxt(new_file, _constraint_sample())
            reused_train = Stage2ClassificationDataset(root, "train", n_points=4)
            reused_test = Stage2ClassificationDataset(root, "test", n_points=4)
            self.assertEqual(len(reused_train), 8)
            self.assertEqual(len(reused_test), 2)
            self.assertNotIn(
                new_file.resolve(), [path for _, path in reused_train.datapath]
            )

            new_class = root / "Washers"
            new_class.mkdir()
            np.savetxt(new_class / "sample.txt", _constraint_sample())
            stable_train = Stage2ClassificationDataset(root, "train", n_points=4)
            self.assertEqual(stable_train.classes, {"Bolts": 0, "Clamps": 1})

    def test_classification_split_file_is_authoritative(self):
        with tempfile.TemporaryDirectory(dir=".") as temporary:
            root = Path(temporary)
            for class_name in ("Bolts", "Clamps"):
                class_dir = root / class_name
                class_dir.mkdir(parents=True)
                for index in range(2):
                    np.savetxt(class_dir / f"sample_{index}.txt", _constraint_sample())

            manifest = {
                "version": 1,
                "test_ratio": 0.2,
                "seed": 42,
                "train": ["Bolts/sample_0.txt", "Clamps/sample_1.txt"],
                "test": ["Bolts/sample_1.txt", "Clamps/sample_0.txt"],
            }
            (root / "split_file.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )

            train_set = Stage2ClassificationDataset(root, "train", n_points=4)
            test_set = Stage2ClassificationDataset(
                root, "test", n_points=4, classes=train_set.classes
            )
            self.assertEqual(
                {
                    path.relative_to(root.resolve()).as_posix()
                    for _, path in train_set.datapath
                },
                set(manifest["train"]),
            )
            self.assertEqual(
                {
                    path.relative_to(root.resolve()).as_posix()
                    for _, path in test_set.datapath
                },
                set(manifest["test"]),
            )


if __name__ == "__main__":
    unittest.main()
