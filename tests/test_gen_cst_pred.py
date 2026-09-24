from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

import gen_cst_pred
from data_utils.stage1_dataset import Stage1ConstraintDataset
from functional.constraints import (
    assemble_constraints_from_stage1,
    constraints_to_tensor,
)
from networks.cst_pred_wrapper import CstPredWrapper
from networks.stage1_extractor import FrozenStage1ConstraintExtractor


def prediction(count: int) -> dict[str, np.ndarray]:
    return {
        "pmt": np.arange(count) % 5,
        "mad": np.full((count, 3), 0.25, dtype=np.float32),
        "dim": np.arange(count, dtype=np.float32),
        "loc": np.full((count, 3), -0.5, dtype=np.float32),
    }


class GenerateConstraintPredictionsTest(unittest.TestCase):
    def test_replaces_constraints_and_keeps_xyz_and_task_columns(self):
        source = np.zeros((4, 14), dtype=np.float64)
        source[:, :3] = np.arange(12).reshape(4, 3)
        source[:, 3] = (np.arange(4) + 1) % 5
        source[:, 11] = np.arange(4) + 0.123456
        source[:, 12:] = np.array([[10, 20], [10, 20], [11, 21], [11, 21]])

        output = gen_cst_pred.build_output_array(source, prediction(4))

        self.assertEqual(output.shape, (4, 14))
        np.testing.assert_array_equal(output[:, :3], source[:, :3])
        np.testing.assert_array_equal(output[:, 11:], source[:, 11:])
        np.testing.assert_array_equal(output[:, 3], prediction(4)["pmt"])

    def test_exactly_11_columns_and_unknown_old_types_are_replaced(self):
        source = np.full((4, 11), -100.25)
        source[:, :3] = np.arange(12).reshape(4, 3)
        predicted = prediction(4)
        output = gen_cst_pred.build_output_array(source, predicted)
        self.assertEqual(output.shape, source.shape)
        np.testing.assert_array_equal(output[:, :3], source[:, :3])
        np.testing.assert_array_equal(output[:, 3], predicted["pmt"])

    def test_default_layout_does_not_silently_insert_columns(self):
        with self.assertRaisesRegex(ValueError, "at least 11 columns"):
            gen_cst_pred.build_output_array(np.zeros((4, 10)), prediction(4))

    def test_text_uses_six_decimal_places_without_truncating_extra_attribute(self):
        source = np.full((4, 12), 1.23456789)
        output = gen_cst_pred.build_output_array(source, prediction(4))
        with tempfile.TemporaryDirectory(dir=".") as temporary:
            path = Path(temporary) / "cloud.txt"
            gen_cst_pred.save_point_file(path, output, " ")
            tokens = path.read_text().splitlines()[0].split()
            self.assertEqual(tokens[3], "0")
            self.assertEqual(tokens[0], "1.234568")
            self.assertEqual(tokens[11], "1.234568")
            for index, token in enumerate(tokens):
                if index != 3:
                    self.assertRegex(token, r"^-?\d+\.\d{6}$")

    def test_output_zeroes_invalid_direction_and_dimension(self):
        source = np.zeros((5, 11), dtype=np.float32)
        output = gen_cst_pred.build_output_array(source, prediction(5))

        np.testing.assert_array_equal(output[3:5, 4:7], np.zeros((2, 3)))
        self.assertEqual(float(output[0, 7]), 0.0)
        self.assertEqual(float(output[4, 7]), 0.0)
        np.testing.assert_array_equal(output[0:3, 4:7], np.full((3, 3), 0.25))

    def test_text_round_trip(self):
        array = gen_cst_pred.build_output_array(
            np.arange(39, dtype=np.float64).reshape(3, 13),
            prediction(3),
        )
        with tempfile.TemporaryDirectory(dir=".") as temporary:
            root = Path(temporary)
            for name, delimiter in (("space.txt", " "), ("comma.txt", ",")):
                with self.subTest(name=name):
                    path = root / "nested" / name
                    gen_cst_pred.save_point_file(path, array, delimiter)
                    loaded, loaded_delimiter = gen_cst_pred.load_point_file(path)
                    np.testing.assert_allclose(loaded, array, rtol=1e-6, atol=1e-6)
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

        for use_pca_normals in (False, True):
            with self.subTest(use_pca_normals=use_pca_normals):
                constraints = assemble_constraints_from_stage1(
                    xyz,
                    embedding,
                    log_pmt,
                    cluster_bandwidth=0.1,
                    cluster_method="radius",
                    normal_k=2,
                    use_pca_normals_for_fitting=use_pca_normals,
                )
                self.assertEqual(
                    set(constraints),
                    {
                        "primitive_type",
                        "direction",
                        "dimension",
                        "location",
                        "affiliate_idx",
                    },
                )
                self.assertEqual(tuple(constraints_to_tensor(constraints).shape), (1, 4, 12))
                self.assertEqual(tuple(constraints["affiliate_idx"].shape), (1, 4))
                self.assertEqual(constraints["affiliate_idx"].unique().numel(), 2)

    def test_mean_shift_joint_initialization_is_used_by_xyz_fitting(self):
        axis = torch.nn.functional.normalize(
            torch.tensor([0.31, -0.42, 0.85], dtype=torch.float64), dim=0
        )
        foot_seed = torch.tensor([0.27, -0.31, 0.13], dtype=torch.float64)
        foot = foot_seed - torch.dot(foot_seed, axis) * axis
        radius = torch.tensor(0.43, dtype=torch.float64)
        theta = torch.linspace(-0.2, 0.2, 21, dtype=torch.float64)
        axial = torch.linspace(-0.01, 0.01, 7, dtype=torch.float64)
        theta_grid, axial_grid = torch.meshgrid(theta, axial, indexing="ij")
        helper = torch.zeros_like(axis)
        helper[int(axis.abs().argmin())] = 1.0
        radial_x = torch.nn.functional.normalize(
            torch.cross(axis, helper, dim=0), dim=0
        )
        radial_y = torch.nn.functional.normalize(
            torch.cross(axis, radial_x, dim=0), dim=0
        )
        radial = (
            torch.cos(theta_grid).reshape(-1, 1) * radial_x
            + torch.sin(theta_grid).reshape(-1, 1) * radial_y
        )
        xyz = (
            foot
            + axial_grid.reshape(-1, 1) * axis
            + radius * radial
        ).unsqueeze(0)
        point_count = xyz.shape[1]
        embedding = torch.tensor([1.0, 0.0], dtype=torch.float64).view(1, 1, 2)
        embedding = embedding.expand(1, point_count, 2).clone()
        log_pmt = torch.zeros(1, point_count, 5, dtype=torch.float64)
        log_pmt[..., 1] = 1.0
        mad = axis.view(1, 1, 3).expand(1, point_count, 3).clone()
        mad[:, ::2] *= -1.0
        dim = radius.view(1, 1).expand(1, point_count).clone()
        loc = foot.view(1, 1, 3).expand(1, point_count, 3).clone()

        constraints = assemble_constraints_from_stage1(
            xyz,
            embedding,
            log_pmt,
            cluster_method="meanshift",
            mean_shift_bandwidth=0.2,
            mean_shift_iterations=5,
            use_pca_normals_for_fitting=False,
            mad_prediction=mad,
            dim_prediction=dim,
            loc_prediction=loc,
        )

        fitted_axis = constraints["direction"][0, 0]
        axis_error = torch.acos(
            torch.dot(fitted_axis, axis).abs().clamp(-1.0, 1.0)
        )
        self.assertLess(float(axis_error), 1e-6)
        self.assertTrue(
            torch.allclose(
                constraints["dimension"][0],
                radius.expand_as(constraints["dimension"][0]),
                atol=1e-7,
                rtol=0.0,
            )
        )
        self.assertTrue(
            torch.allclose(
                constraints["location"][0],
                foot.expand_as(constraints["location"][0]),
                atol=1e-7,
                rtol=0.0,
            )
        )
        self.assertEqual(constraints["affiliate_idx"].unique().numel(), 1)

    def test_stage1_dataset_reads_12_column_txt_and_ignores_stale_npy(self):
        with tempfile.TemporaryDirectory(dir=".") as temporary:
            sample_dir = Path(temporary) / "train" / "part"
            sample_dir.mkdir(parents=True)
            sample = np.zeros((8, 12), dtype=np.float64)
            sample[:, 0:3] = np.arange(24, dtype=np.float64).reshape(8, 3)
            sample[:, 3] = np.arange(8) % 5
            sample[:, 4] = 1.0
            sample[:, 7] = 0.5
            sample[:, 8:11] = 7.0
            sample[:, 11] = np.arange(8) // 2
            txt_path = sample_dir / "sample.txt"
            np.savetxt(txt_path, sample)
            np.save(str(txt_path) + ".npy", np.zeros((8, 15), dtype=np.float64))

            dataset = Stage1ConstraintDataset(temporary, n_points=4)
            fields = dataset[0]

        self.assertEqual(len(fields), 6)
        self.assertEqual(fields[0].shape, (4, 3))
        np.testing.assert_array_equal(fields[4], np.full((4, 3), 7.0))

    def test_stage1_directory_loader_recursively_reads_every_txt(self):
        with tempfile.TemporaryDirectory(dir=".") as temporary:
            root = Path(temporary)
            sample = np.zeros((8, 12), dtype=np.float64)
            sample[:, 0:3] = np.arange(24, dtype=np.float64).reshape(8, 3)
            sample[:, 3] = np.arange(8) % 5
            sample[:, 4] = 1.0
            sample[:, 7] = 0.5
            sample[:, 11] = np.arange(8) // 2
            paths = [
                root / "one.txt",
                root / "category" / "two.txt",
                root / "any" / "depth" / "three.txt",
            ]
            for index, path in enumerate(paths):
                path.parent.mkdir(parents=True, exist_ok=True)
                current = sample.copy()
                current[:, 0] = index
                np.savetxt(path, current)
            np.save(root / "ignored.npy", np.zeros((8, 12), dtype=np.float64))

            loader = Stage1ConstraintDataset.create_dataloader(
                root=temporary,
                bs=2,
                n_points=4,
                num_workers=0,
                shuffle=False,
                sample_seed=7,
            )
            batches = list(loader)
            first_read = loader.dataset[0][0]
            second_read = loader.dataset[0][0]

        self.assertEqual(len(loader.dataset), 3)
        self.assertEqual(sum(batch[0].shape[0] for batch in batches), 3)
        np.testing.assert_array_equal(first_read, second_read)

    def test_stage1_dataset_skips_txt_files_with_too_few_points(self):
        with tempfile.TemporaryDirectory(dir=".") as temporary:
            root = Path(temporary)
            enough_points = np.zeros((6, 12), dtype=np.float64)
            too_few_points = np.zeros((3, 12), dtype=np.float64)
            enough_points[:, 3] = np.arange(6) % 5
            enough_points[:, 11] = np.arange(6)
            np.savetxt(root / "enough.txt", enough_points)
            np.savetxt(root / "too_few.txt", too_few_points)

            dataset = Stage1ConstraintDataset(root, n_points=4)

            self.assertEqual(dataset.files, [root / "enough.txt"])
            self.assertEqual(len(dataset), 1)

    def test_stage1_dataset_reports_when_all_txt_files_are_too_small(self):
        with tempfile.TemporaryDirectory(dir=".") as temporary:
            root = Path(temporary)
            np.savetxt(root / "too_few.txt", np.zeros((3, 12)))

            with self.assertRaisesRegex(
                ValueError,
                "no Stage 1 samples.*at least 4 points",
            ):
                Stage1ConstraintDataset(root, n_points=4)

    def test_cli_defaults_to_joint_checkpoint(self):
        args = gen_cst_pred.parse_args(
            ["--input_dir", "input", "--output_dir", "output"]
        )
        self.assertEqual(args.model, "auto")
        self.assertFalse(hasattr(args, "stage1_mode"))
        self.assertEqual(Path(args.checkpoint), gen_cst_pred.DEFAULT_CHECKPOINT)
        self.assertFalse(hasattr(args, "cluster_method"))
        self.assertFalse(args.overwrite)

    def test_real_stage1_checkpoint_xyz_inference_smoke(self):
        with tempfile.TemporaryDirectory(dir=".") as temporary:
            checkpoint_path = Path(temporary) / "last.pth"
            model = CstPredWrapper("attn_3dgcn")
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": {
                        "model": "attn_3dgcn",
                        "constraint_route": "direct_mlp_v1",
                        "train_phase": "joint",
                        "use_extra_features": False,
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
        self.assertEqual(set(predicted), {"pmt", "mad", "dim", "loc"})
        self.assertTrue(np.isfinite(predicted["loc"]).all())

    def test_generate_mixed_directory_preserves_paths_and_skips_existing_files(self):
        with tempfile.TemporaryDirectory(dir=".") as temporary:
            root = Path(temporary)
            input_dir, output_dir = root / "input", root / "output"
            (input_dir / "nested" / "empty").mkdir(parents=True)
            source = np.arange(4 * 13, dtype=np.float64).reshape(4, 13) + 0.123456
            np.savetxt(input_dir / "nested" / "part.txt", source)
            np.savetxt(input_dir / "part.TXT", source)
            for name in ("ignored.h5", "ignored.hdf5", "ignored.npy", "ignored.csv"):
                (input_dir / name).write_bytes(b"not a TXT point cloud")
            args = gen_cst_pred.parse_args(["--input_dir", str(input_dir), "--output_dir", str(output_dir)])
            with mock.patch.object(gen_cst_pred, "Stage1Predictor") as factory:
                predictor = factory.return_value
                predictor.predict.side_effect = lambda xyz: prediction(len(xyz))
                gen_cst_pred.generate_dataset(args)
                self.assertEqual(predictor.predict.call_count, 2)
                gen_cst_pred.generate_dataset(args)
                self.assertEqual(predictor.predict.call_count, 2)
                args.overwrite = True
                gen_cst_pred.generate_dataset(args)
                self.assertEqual(predictor.predict.call_count, 4)
            self.assertTrue((output_dir / "nested" / "empty").is_dir())
            txt = np.loadtxt(output_dir / "nested" / "part.txt")
            np.testing.assert_array_equal(txt[:, 11:], source[:, 11:])
            self.assertEqual(
                {path.relative_to(output_dir).as_posix() for path in output_dir.rglob("*") if path.is_file()},
                {"part.TXT", "nested/part.txt"},
            )

    def test_frozen_extractor_rejects_semantic_only_geometry_initialization(self):
        with tempfile.TemporaryDirectory(dir=".") as temporary:
            checkpoint_path = Path(temporary) / "semantic.pth"
            model = CstPredWrapper("pointnet")
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": {"train_phase": "semantic", "constraint_route": "direct_mlp_v1"},
                },
                checkpoint_path,
            )
            with self.assertRaisesRegex(ValueError, "geometry or joint"):
                FrozenStage1ConstraintExtractor(model_name="pointnet", checkpoint=str(checkpoint_path))


if __name__ == "__main__":
    unittest.main()
