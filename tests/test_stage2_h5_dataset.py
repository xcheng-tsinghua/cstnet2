from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from data_utils.classification_dataset import Stage2ClassificationDataset
from data_utils.mfcad_seg_dataset import Stage2SegmentationDataset
from data_utils.stage2_h5 import convert_classification_txt_to_h5, convert_segmentation_txt_to_h5
from functional.segmentation_loss import compute_training_class_statistics


def sample(count, segmentation=False):
    array = np.zeros((count, 14 if segmentation else 12), dtype=np.float32)
    array[:, :3] = np.arange(count * 3).reshape(count, 3) / 7
    array[:, 3] = np.arange(count) % 5
    array[:, 4:7] = (0, 0, -1)
    array[:, 7] = -1
    array[:, 8:11] = (0.1, 0.2, 0.3)
    array[:, 11] = np.arange(count) // 2
    if segmentation:
        array[:, 12] = np.arange(count) // 2
        array[:, 13] = array[:, 12] % 3
    return array


def write_classification(root, flat=False):
    for split in (("",) if flat else ("train", "test")):
        for name in ("Bolts", "Clamps"):
            for index, count in enumerate((3, 7, 9)):
                path = root / split / name / "nested" / f"{index}.txt"
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savetxt(path, sample(count))


def write_segmentation(root, with_test=True):
    for split in (("train", "validation", "test") if with_test else ("train", "validation")):
        for index, count in ((10, 3), (2, 9)):
            path = root / split / f"{index}.txt"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savetxt(path, sample(count, True))


class Stage2H5Test(unittest.TestCase):
    def assert_items_equal(self, txt, h5):
        self.assertEqual(len(txt), len(h5))
        try:
            for index in range(len(txt)):
                np.random.seed(101 + index)
                expected = txt[index]
                np.random.seed(101 + index)
                actual = h5[index]
                if isinstance(expected, dict):
                    for key in expected:
                        if isinstance(expected[key], torch.Tensor):
                            torch.testing.assert_close(expected[key], actual[key], rtol=0, atol=0)
                        else:
                            self.assertEqual(expected[key], actual[key])
                else:
                    for left, right in zip(expected, actual):
                        np.testing.assert_array_equal(left, right)
        finally:
            h5.close()

    def test_classification_roundtrip_layouts_and_original_split(self):
        for flat in (False, True):
            with self.subTest(flat=flat), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                source, output = root / "txt", root / "h5"
                write_classification(source, flat)
                # Establish the split before conversion: conversion must reuse it.
                txt_train = Stage2ClassificationDataset(source, "train", n_points=5)
                shards = convert_classification_txt_to_h5(source, output, samples_per_shard=2)
                self.assertGreater(len(shards), 2)
                h5_train = Stage2ClassificationDataset(output, "train", n_points=5)
                self.assertEqual(h5_train.storage_format, "h5")
                self.assertEqual(txt_train.classes, h5_train.classes)
                self.assert_items_equal(txt_train, h5_train)
                self.assert_items_equal(
                    Stage2ClassificationDataset(source, "test", n_points=5),
                    Stage2ClassificationDataset(output, "test", n_points=5),
                )
                single = Stage2ClassificationDataset(shards[0], "train", n_points=5)
                self.assertEqual(len(single), 2)
                single.close()

    def test_segmentation_roundtrip_and_full_point_class_statistics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, output = root / "txt", root / "h5"
            write_segmentation(source)
            convert_segmentation_txt_to_h5(source, output, samples_per_shard=1, compression="gzip")
            for split in ("train", "val", "test"):
                for n_points in (None, 2, 5, 12):
                    self.assert_items_equal(
                        Stage2SegmentationDataset(source, split, n_points=n_points),
                        Stage2SegmentationDataset(output, split, n_points=n_points),
                    )
            txt = Stage2SegmentationDataset(source, "train", n_points=2, use_npy_cache=True)
            h5 = Stage2SegmentationDataset(output, "train", n_points=2)
            try:
                weights, stats = compute_training_class_statistics(txt, root / "txt_stats.json")
                # HDF5 must work even when TXT loading is unavailable.
                with patch("numpy.loadtxt", side_effect=AssertionError("unexpected TXT read")):
                    actual, actual_stats = compute_training_class_statistics(h5, root / "h5_stats.json")
                torch.testing.assert_close(weights, actual)
                self.assertEqual(stats, actual_stats)
                self.assertEqual(stats["total_points"], 12)
                self.assertEqual(stats["class_counts"][:3], [6, 4, 2])
            finally:
                h5.close()

    def test_spawn_workers_after_parent_has_opened_h5_and_txt_removed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_classification(root / "cls")
            write_segmentation(root / "seg", with_test=False)
            convert_classification_txt_to_h5(root / "cls", root / "cls_h5", samples_per_shard=2)
            convert_segmentation_txt_to_h5(root / "seg", root / "seg_h5", samples_per_shard=1)
            datasets = [
                Stage2ClassificationDataset(root / "cls_h5", "train", n_points=5),
                Stage2SegmentationDataset(root / "seg_h5", "train", n_points=5),
            ]
            for source in (root / "cls", root / "seg"):
                for path in source.rglob("*.txt"):
                    path.unlink()
            for dataset in datasets:
                try:
                    dataset[0]  # Open in parent before the dataset is pickled for workers.
                    loader = DataLoader(dataset, batch_size=2, num_workers=2,
                                        multiprocessing_context="spawn")
                    batches = list(loader)
                    count = sum((b[0] if isinstance(b, list) else b["xyz"]).shape[0] for b in batches)
                    self.assertEqual(count, len(dataset))
                finally:
                    dataset.close()
            loaders = Stage2SegmentationDataset.create_dataloaders(
                root / "seg_h5", num_workers=0, n_points=5, drop_last=False,
            )
            self.assertIsNone(loaders[2])
            for loader in loaders[:2]:
                self.assertGreater(len(list(loader)), 0)
                loader.dataset.close()

    def test_format_selection_overwrite_and_schema_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_classification(root)
            output = root / "converted"
            shards = convert_classification_txt_to_h5(root, output, samples_per_shard=1)
            auto = Stage2ClassificationDataset(root, "train")
            self.assertEqual(auto.storage_format, "h5")
            auto.close()
            self.assertEqual(Stage2ClassificationDataset(root, "train", storage_format="txt").storage_format, "txt")
            with self.assertRaises(FileExistsError):
                convert_classification_txt_to_h5(root, output)
            shards = convert_classification_txt_to_h5(root, output, overwrite=True, samples_per_shard=20)
            self.assertEqual(len(list(output.glob("*.h5"))), 2)
            with h5py.File(shards[0], "r+") as handle:
                handle["offsets"][1] = 0
            with self.assertRaisesRegex(ValueError, "offsets"):
                Stage2ClassificationDataset(output, "train")
            with self.assertRaisesRegex(ValueError, "incompatible"):
                Stage2SegmentationDataset(output, "train")

    def test_class_mapping_with_missing_test_class_and_custom_label_map(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_classification(root / "cls")
            for path in (root / "cls" / "test" / "Bolts").rglob("*.txt"):
                path.unlink()
            # TXT classification discovers class directories, so remove the empty ones.
            (root / "cls" / "test" / "Bolts" / "nested").rmdir()
            (root / "cls" / "test" / "Bolts").rmdir()
            convert_classification_txt_to_h5(root / "cls", root / "cls_h5", compression="none")
            test = Stage2ClassificationDataset(root / "cls_h5", "test", n_points=3)
            try:
                self.assertEqual(test.classes, {"Bolts": 0, "Clamps": 1})
                self.assertTrue(all(test[index][1] == 1 for index in range(len(test))))
            finally:
                test.close()
            write_segmentation(root / "seg", with_test=False)
            label_map = root / "labels.json"
            label_map.write_text(
                '{"labels": [{"id": 0, "name": "a"}, {"id": 1, "name": "b"}, {"id": 2, "name": "c"}]}',
                encoding="utf-8",
            )
            convert_segmentation_txt_to_h5(root / "seg", root / "seg_h5", label_map_path=label_map)
            with self.assertRaisesRegex(ValueError, "label map"):
                Stage2SegmentationDataset(root / "seg_h5", "train")
            dataset = Stage2SegmentationDataset(root / "seg_h5", "train", label_map_path=label_map)
            try:
                self.assertEqual(dataset.num_classes, 3)
                self.assertLess(int(dataset[0]["labels"].max()), 3)
            finally:
                dataset.close()

    def test_training_arguments(self):
        import train_cls
        import train_seg
        for module in (train_cls, train_seg):
            args = module.parse_args(["--data_root", "converted", "--data_format", "h5"])
            self.assertEqual(args.data_root, "converted")
            self.assertEqual(args.data_format, "h5")
            self.assertEqual(module.parse_args([]).data_format, "auto")


if __name__ == "__main__":
    unittest.main()
