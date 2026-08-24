from __future__ import annotations

import importlib.util
import gc
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from data_utils.stage1_dataset import Stage1ConstraintDataset
from data_utils.stage1_h5 import convert_stage1_txt_to_h5, parse_args


H5PY_AVAILABLE = importlib.util.find_spec("h5py") is not None


def _sample(point_count: int, marker: float) -> np.ndarray:
    sample = np.zeros((point_count, 12), dtype=np.float32)
    sample[:, 0] = marker
    sample[:, 1:3] = np.arange(point_count * 2, dtype=np.float32).reshape(-1, 2)
    sample[:, 3] = np.arange(point_count) % 5
    sample[:, 4:7] = (1.0, 0.0, 0.0)
    sample[:, 7] = 0.5
    sample[:, 8:11] = marker + 10.0
    sample[:, 11] = np.arange(point_count) // 2
    return sample


@unittest.skipUnless(H5PY_AVAILABLE, "h5py is required for HDF5 tests")
class Stage1H5DatasetTest(unittest.TestCase):
    def test_conversion_accepts_multiple_input_directories(self):
        import h5py

        with tempfile.TemporaryDirectory(dir=".") as temporary:
            root = Path(temporary)
            first_root = root / "first"
            second_root = root / "second"
            (first_root / "nested").mkdir(parents=True)
            second_root.mkdir()
            np.savetxt(first_root / "nested" / "one.txt", _sample(6, 1.0))
            np.savetxt(second_root / "two.txt", _sample(7, 2.0))
            output_root = root / "h5"

            shards = convert_stage1_txt_to_h5(
                [first_root, second_root, first_root],
                output_root,
                samples_per_shard=10,
                compression="none",
            )
            dataset = Stage1ConstraintDataset(output_root, n_points=4)
            manifest = json.loads(
                (output_root / "stage1_manifest.json").read_text(encoding="utf-8")
            )
            with h5py.File(shards[0], "r") as h5_file:
                source_names = h5_file["source_path"].asstr()[:].tolist()

            self.assertEqual(len(dataset), 2)
            self.assertEqual(
                manifest["input_roots"],
                [str(first_root.resolve()), str(second_root.resolve())],
            )
            self.assertEqual(manifest["sample_count"], 2)
            self.assertEqual(
                source_names,
                ["root_000_first/nested/one.txt", "root_001_second/two.txt"],
            )

    def test_cli_accepts_one_or_more_input_directories(self):
        args = parse_args(
            ["--input_dir", "first", "second", "--output_dir", "output"]
        )

        self.assertEqual(args.input_dir, [Path("first"), Path("second")])

    def test_recursive_conversion_and_txt_h5_reads_match(self):
        with tempfile.TemporaryDirectory(dir=".") as temporary:
            root = Path(temporary)
            txt_root = root / "txt"
            h5_root = root / "h5"
            paths = [txt_root / "one.txt", txt_root / "nested" / "two.txt"]
            for path, sample in zip(paths, (_sample(8, 1.0), _sample(9, 2.0))):
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savetxt(path, sample)

            shards = convert_stage1_txt_to_h5(
                txt_root, h5_root, samples_per_shard=1, compression="none"
            )
            txt_dataset = Stage1ConstraintDataset(
                txt_root, n_points=5, sample_seed=17, storage_format="txt"
            )
            h5_dataset = Stage1ConstraintDataset(
                h5_root, n_points=5, sample_seed=17, storage_format="h5"
            )

            self.assertEqual(len(shards), 2)
            self.assertEqual(len(txt_dataset), len(h5_dataset))
            for index in range(len(txt_dataset)):
                for txt_field, h5_field in zip(txt_dataset[index], h5_dataset[index]):
                    np.testing.assert_array_equal(txt_field, h5_field)
            h5_dataset.close()

    def test_conversion_discards_columns_after_the_constraint_core(self):
        import h5py

        with tempfile.TemporaryDirectory(dir=".") as temporary:
            root = Path(temporary)
            txt_root = root / "txt"
            txt_root.mkdir()
            constraint_core = _sample(8, 3.0)
            extra_columns = np.full((8, 5), 99.0, dtype=np.float32)
            np.savetxt(
                txt_root / "with_extra_columns.txt",
                np.concatenate((constraint_core, extra_columns), axis=1),
            )

            with self.assertRaisesRegex(ValueError, "expected 12 columns"):
                Stage1ConstraintDataset(
                    txt_root, n_points=4, storage_format="txt"
                )[0]
            shard = convert_stage1_txt_to_h5(
                txt_root, root / "h5", compression="none"
            )[0]
            with h5py.File(shard, "r") as h5_file:
                np.testing.assert_array_equal(h5_file["xyz"][:], constraint_core[:, 0:3])
                np.testing.assert_array_equal(h5_file["pmt"][:], constraint_core[:, 3])
                np.testing.assert_array_equal(
                    h5_file["direction"][:], constraint_core[:, 4:7]
                )
                np.testing.assert_array_equal(
                    h5_file["dimension"][:], constraint_core[:, 7]
                )
                np.testing.assert_array_equal(
                    h5_file["location"][:], constraint_core[:, 8:11]
                )
                np.testing.assert_array_equal(
                    h5_file["affiliate_idx"][:], constraint_core[:, 11]
                )

    def test_conversion_rejects_samples_with_fewer_than_12_columns(self):
        with tempfile.TemporaryDirectory(dir=".") as temporary:
            root = Path(temporary)
            txt_root = root / "txt"
            txt_root.mkdir()
            np.savetxt(txt_root / "too_few_columns.txt", np.zeros((8, 11)))

            with self.assertRaisesRegex(ValueError, "expected at least 12 columns"):
                convert_stage1_txt_to_h5(
                    txt_root, root / "h5", compression="none"
                )

    def test_auto_prefers_h5_and_h5_filter_uses_offsets(self):
        with tempfile.TemporaryDirectory(dir=".") as temporary:
            root = Path(temporary)
            nested = root / "nested"
            nested.mkdir()
            np.savetxt(root / "enough.txt", _sample(6, 1.0))
            np.savetxt(nested / "too_few.txt", _sample(3, 2.0))
            convert_stage1_txt_to_h5(
                root, root / "converted", samples_per_shard=10, compression="none"
            )

            dataset = Stage1ConstraintDataset(root, n_points=4)

            self.assertEqual(dataset.storage_format, "h5")
            self.assertEqual(len(dataset), 1)

    def test_dataset_accepts_a_single_h5_file(self):
        with tempfile.TemporaryDirectory(dir=".") as temporary:
            root = Path(temporary)
            txt_root = root / "txt"
            txt_root.mkdir()
            np.savetxt(txt_root / "sample.txt", _sample(6, 3.0))
            shard = convert_stage1_txt_to_h5(
                txt_root, root / "h5", compression="none"
            )[0]

            dataset = Stage1ConstraintDataset(shard, n_points=4)

            self.assertEqual(dataset.storage_format, "h5")
            self.assertEqual(len(dataset), 1)

    def test_h5_dataloader_supports_worker_processes(self):
        with tempfile.TemporaryDirectory(dir=".") as temporary:
            root = Path(temporary)
            txt_root = root / "txt"
            txt_root.mkdir()
            for index in range(4):
                np.savetxt(txt_root / f"{index}.txt", _sample(6, float(index)))
            h5_root = root / "h5"
            convert_stage1_txt_to_h5(
                txt_root, h5_root, samples_per_shard=2, compression="none"
            )

            loader = Stage1ConstraintDataset.create_dataloader(
                root=h5_root,
                bs=2,
                n_points=4,
                num_workers=2,
                shuffle=False,
                storage_format="h5",
            )
            batches = list(loader)

            self.assertEqual(sum(batch[0].shape[0] for batch in batches), 4)
            del batches
            del loader
            gc.collect()
