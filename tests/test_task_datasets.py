from __future__ import annotations

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


if __name__ == "__main__":
    unittest.main()
