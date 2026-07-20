from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

import gen_cst_pred
from functional.constraints import assemble_constraints_from_stage1
from networks.cst_pred_wrapper import CstPredWrapper


def prediction(count: int) -> dict[str, np.ndarray]:
    return {
        "pmt": np.arange(count) % 5,
        "mad": np.full((count, 3), 0.25, dtype=np.float32),
        "dim": np.arange(count, dtype=np.float32),
        "nor": np.full((count, 3), 0.5, dtype=np.float32),
        "loc": np.full((count, 3), -0.5, dtype=np.float32),
        "affiliate_idx": np.arange(count) // 2,
    }


class GenerateConstraintPredictionsTest(unittest.TestCase):
    def test_raw_layout_preserves_opaque_suffix_after_constraint_core(self):
        source = np.arange(20, dtype=np.float64).reshape(4, 5)
        output = gen_cst_pred.build_output_array(
            source, prediction(4), input_layout="raw"
        )

        self.assertEqual(output.shape, (4, 17))
        np.testing.assert_array_equal(output[:, :3], source[:, :3])
        np.testing.assert_array_equal(output[:, 15:], source[:, 3:])
        np.testing.assert_array_equal(output[:, 3], prediction(4)["pmt"])
        np.testing.assert_array_equal(output[:, 14], prediction(4)["affiliate_idx"])

    def test_auto_layout_replaces_gt_constraints_and_keeps_task_columns(self):
        source = np.zeros((4, 17), dtype=np.float64)
        source[:, :3] = np.arange(12).reshape(4, 3)
        source[:, 3] = (np.arange(4) + 1) % 5
        source[:, 14] = np.arange(4) // 2
        source[:, 15:] = np.array([[10, 20], [10, 20], [11, 21], [11, 21]])

        output = gen_cst_pred.build_output_array(source, prediction(4))

        self.assertEqual(output.shape, (4, 17))
        np.testing.assert_array_equal(output[:, 15:], source[:, 15:])
        np.testing.assert_array_equal(output[:, 3], prediction(4)["pmt"])

    def test_text_and_npy_round_trip(self):
        array = gen_cst_pred.build_output_array(
            np.arange(15, dtype=np.float64).reshape(3, 5),
            prediction(3),
            input_layout="raw",
        )
        with tempfile.TemporaryDirectory(dir=".") as temporary:
            root = Path(temporary)
            for name, delimiter in (("cloud.txt", " "), ("cloud.csv", ","), ("cloud.npy", " ")):
                with self.subTest(name=name):
                    path = root / "nested" / name
                    gen_cst_pred.save_point_file(path, array, delimiter)
                    loaded, loaded_delimiter = gen_cst_pred.load_point_file(path)
                    np.testing.assert_allclose(loaded, array, rtol=1e-6, atol=1e-6)
                    if path.suffix != ".npy":
                        self.assertEqual(loaded_delimiter, delimiter)

    def test_constraint_assembly_returns_cluster_affiliations(self):
        xyz = torch.tensor(
            [[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [2.0, 0.0, 0.0], [2.1, 0.0, 0.0]]]
        )
        embedding = torch.tensor(
            [[[1.0, 0.0], [1.0, 0.01], [0.0, 1.0], [0.01, 1.0]]]
        )
        log_pmt = torch.zeros(1, 4, 5)
        log_pmt[..., 0] = 1.0

        constraints = assemble_constraints_from_stage1(
            xyz, embedding, log_pmt, cluster_bandwidth=0.1, normal_k=2
        )

        self.assertEqual(tuple(constraints["affiliate_idx"].shape), (1, 4))
        self.assertEqual(constraints["affiliate_idx"].unique().numel(), 2)

    def test_cli_defaults_to_checkpoint_metadata_and_xyz_text_files(self):
        args = gen_cst_pred.parse_args(
            ["--input_dir", "input", "--output_dir", "output", "--checkpoint", "weights.pth"]
        )
        self.assertEqual(args.model, "auto")
        self.assertFalse(hasattr(args, "stage1_mode"))
        self.assertEqual(args.extensions, ".txt")
        self.assertEqual(args.input_layout, "auto")
        self.assertFalse(args.overwrite)

    def test_real_stage1_checkpoint_xyz_inference_smoke(self):
        with tempfile.TemporaryDirectory(dir=".") as temporary:
            checkpoint_path = Path(temporary) / "last.pth"
            model = CstPredWrapper("pointnet")
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": {
                        "model": "pointnet",
                        "stage1_mode": "multitask",
                        "use_extra_features": False,
                        "normal_source": "none",
                        "feature_k": 16,
                        "cluster_bandwidth": 0.35,
                    },
                },
                checkpoint_path,
            )
            predictor = gen_cst_pred.Stage1Predictor(
                checkpoint_path, torch.device("cpu")
            )
            predicted = predictor.predict(
                np.random.default_rng(7).normal(size=(32, 3)).astype(np.float32)
            )

        self.assertEqual(predicted["pmt"].shape, (32,))
        self.assertEqual(predicted["mad"].shape, (32, 3))
        self.assertEqual(predicted["affiliate_idx"].shape, (32,))
        self.assertTrue(np.isfinite(predicted["loc"]).all())


if __name__ == "__main__":
    unittest.main()
