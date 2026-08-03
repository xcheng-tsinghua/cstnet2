from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from functional.wandb_utils import (
    flatten_wandb_metrics,
    flatten_wandb_summary_metrics,
    initialize_wandb_run,
    read_wandb_run_id_from_checkpoint,
    read_env_file,
    require_wandb_api_key,
    wandb_confusion_matrix,
)
from networks.utils import all_metric_cls


class WandBLoggingTest(unittest.TestCase):
    def test_all_training_clis_require_wandb_without_disable_flag(self):
        import train_cls
        import train_cst_pred
        import train_seg
        import eval_seg

        for args in (
            train_cst_pred.parse_args(["--data_root", "dataset"]),
            train_cls.parse_args([]),
            train_seg.parse_args([]),
            eval_seg.parse_args(["checkpoint.pth"]),
        ):
            self.assertFalse(hasattr(args, "use_wandb"))
            self.assertEqual(args.wandb_project, "cstnet2")

    def test_stage1_boolean_options_are_flag_based(self):
        import train_cst_pred

        defaults = train_cst_pred.parse_args(["--data_root", "dataset"])
        self.assertFalse(hasattr(defaults, "stage1_mode"))
        self.assertFalse(defaults.is_sample)
        self.assertEqual(defaults.data_root, "dataset")
        self.assertFalse(hasattr(defaults, "local"))
        self.assertFalse(hasattr(defaults, "root_local"))
        self.assertFalse(hasattr(defaults, "root_sever"))
        self.assertFalse(defaults.disable_extra_features)
        self.assertFalse(defaults.overfit_one_batch)
        self.assertFalse(defaults.use_amp)
        self.assertTrue(defaults.enable_mad_loss)
        self.assertTrue(defaults.enable_grad_diagnostics)
        self.assertEqual(defaults.checkpoint_policy, "auto")
        self.assertFalse(hasattr(defaults, "resume_checkpoint"))
        self.assertFalse(hasattr(defaults, "init_from_checkpoint"))

        enabled = train_cst_pred.parse_args([
            "--is_sample",
            "--data_root",
            "dataset",
            "--disable_extra_features",
            "--overfit_one_batch",
            "--use_amp",
            "--disable_mad_loss",
            "--disable_grad_diagnostics",
        ])
        self.assertTrue(enabled.is_sample)
        self.assertEqual(enabled.data_root, "dataset")
        self.assertTrue(enabled.disable_extra_features)
        self.assertTrue(enabled.overfit_one_batch)
        self.assertTrue(enabled.use_amp)
        self.assertFalse(enabled.enable_mad_loss)
        self.assertFalse(enabled.enable_grad_diagnostics)

    def test_env_reader_and_required_key(self):
        previous = os.environ.get("WANDB_API_KEY")
        try:
            with tempfile.TemporaryDirectory() as directory:
                env_path = Path(directory) / ".env"
                env_path.write_text(
                    "# comment\nexport WANDB_API_KEY='test-secret-key'\n",
                    encoding="utf-8",
                )
                self.assertEqual(
                    read_env_file(env_path),
                    {"WANDB_API_KEY": "test-secret-key"},
                )
                self.assertEqual(require_wandb_api_key(env_path), "test-secret-key")
                self.assertEqual(os.environ["WANDB_API_KEY"], "test-secret-key")

                env_path.write_text(
                    "WANDB_API_KEY=your_wandb_api_key_here\n", encoding="utf-8"
                )
                with self.assertRaisesRegex(ValueError, "placeholder"):
                    require_wandb_api_key(env_path)
        finally:
            if previous is None:
                os.environ.pop("WANDB_API_KEY", None)
            else:
                os.environ["WANDB_API_KEY"] = previous

    def test_nested_metrics_are_fully_flattened(self):
        flattened = flatten_wandb_metrics(
            "val/metric",
            {
                "mean_iou": 0.5,
                "per_class_iou": [0.25, 0.75],
                "confusion_matrix": [[3, 1], [2, 4]],
            },
        )
        self.assertEqual(flattened["val/metric/mean_iou"], 0.5)
        self.assertEqual(flattened["val/metric/per_class_iou/1"], 0.75)
        self.assertEqual(flattened["val/metric/confusion_matrix/1/0"], 2.0)

    def test_summary_metrics_skip_vectors_and_matrices(self):
        summaries = flatten_wandb_summary_metrics(
            "val/metric",
            {
                "mean_iou": 0.5,
                "optimization": {"gradient_norm": 1.25},
                "per_class_iou": [0.25, 0.75],
                "confusion_matrix": [[3, 1], [2, 4]],
                "tensor_vector": np.asarray([1.0, 2.0]),
            },
        )
        self.assertEqual(
            summaries,
            {
                "val/metric/mean_iou": 0.5,
                "val/metric/optimization/gradient_norm": 1.25,
            },
        )

    def test_confusion_matrix_is_logged_as_one_native_panel(self):
        table = object()
        chart = object()
        fake_wandb = SimpleNamespace(
            Table=Mock(return_value=table),
            plot_table=Mock(return_value=chart),
        )
        with patch.dict(sys.modules, {"wandb": fake_wandb}):
            result = wandb_confusion_matrix(
                [[3, 1], [2, 4]],
                ["plane", "cylinder"],
                title="Primitive Confusion Matrix",
            )

        self.assertIs(result, chart)
        fake_wandb.Table.assert_called_once_with(
            columns=["Actual", "Predicted", "nPredictions"],
            data=[
                ["plane", "plane", 3],
                ["plane", "cylinder", 1],
                ["cylinder", "plane", 2],
                ["cylinder", "cylinder", 4],
            ],
        )
        fake_wandb.plot_table.assert_called_once_with(
            "wandb/confusion_matrix/v1",
            table,
            {
                "Actual": "Actual",
                "Predicted": "Predicted",
                "nPredictions": "nPredictions",
            },
            {"title": "Primitive Confusion Matrix"},
        )

    def test_initializer_authenticates_from_env_and_forces_online_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            env_path = Path(directory) / ".env"
            env_path.write_text("WANDB_API_KEY=test-key\n", encoding="utf-8")
            fake_run = object()
            fake_wandb = SimpleNamespace(
                login=Mock(),
                init=Mock(return_value=fake_run),
            )
            with (
                patch.dict(sys.modules, {"wandb": fake_wandb}),
                patch.dict(os.environ, {}, clear=False),
            ):
                result = initialize_wandb_run(
                    project="project",
                    entity="team",
                    name="run",
                    config={"epochs": 2},
                    env_path=env_path,
                )

        self.assertIs(result, fake_run)
        fake_wandb.login.assert_called_once_with(key="test-key", relogin=True)
        fake_wandb.init.assert_called_once_with(
            project="project",
            entity="team",
            name="run",
            config={"epochs": 2},
            mode="online",
        )

    def test_initializer_resumes_exact_wandb_run_id(self):
        with tempfile.TemporaryDirectory() as directory:
            env_path = Path(directory) / ".env"
            env_path.write_text("WANDB_API_KEY=test-key\n", encoding="utf-8")
            fake_wandb = SimpleNamespace(
                login=Mock(),
                init=Mock(return_value=SimpleNamespace(id="run-123")),
            )
            with patch.dict(sys.modules, {"wandb": fake_wandb}):
                initialize_wandb_run(
                    project="project",
                    entity="team",
                    name="pointnet2",
                    config={"model": "pointnet2"},
                    run_id="run-123",
                    env_path=env_path,
                )

        fake_wandb.init.assert_called_once_with(
            project="project",
            entity="team",
            name="pointnet2",
            config={"model": "pointnet2"},
            mode="online",
            id="run-123",
            resume="must",
        )

    def test_checkpoint_wandb_run_id_reader_supports_new_and_legacy_files(self):
        import torch

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "last.pth"
            torch.save({"model": {}, "wandb_run_id": "stable-id"}, checkpoint)
            self.assertEqual(
                read_wandb_run_id_from_checkpoint(checkpoint), "stable-id"
            )
            torch.save({"model": {}}, checkpoint)
            self.assertEqual(read_wandb_run_id_from_checkpoint(checkpoint), "")

    def test_classification_metrics_include_per_class_and_confusion_data(self):
        scores = [
            np.asarray(
                [
                    [0.8, 0.1, 0.1],
                    [0.2, 0.7, 0.1],
                    [0.1, 0.2, 0.7],
                    [0.2, 0.6, 0.2],
                ],
                dtype=np.float32,
            )
        ]
        labels = [np.asarray([0, 1, 2, 2], dtype=np.int64)]
        metrics = all_metric_cls(scores, labels)

        self.assertIn("instance_accuracy", metrics)
        self.assertAlmostEqual(metrics["top_1_accuracy"], 0.75)
        self.assertAlmostEqual(metrics["top_5_accuracy"], 1.0)
        self.assertEqual(metrics["top_5_effective_k"], 3)
        self.assertEqual(len(metrics["per_class_precision"]), 3)
        self.assertEqual(len(metrics["per_class_average_precision"]), 3)
        self.assertEqual(np.asarray(metrics["confusion_matrix"]).shape, (3, 3))


if __name__ == "__main__":
    unittest.main()
