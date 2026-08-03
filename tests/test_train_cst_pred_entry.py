from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

import train_cst_pred
from data_utils.stage1_dataset import Stage1ConstraintDataset


class TrainStage1EntryTest(unittest.TestCase):
    def test_manual_checkpoint_path_arguments_are_removed(self):
        args = train_cst_pred.parse_args([])
        self.assertFalse(hasattr(args, "resume_checkpoint"))
        self.assertFalse(hasattr(args, "init_from_checkpoint"))
        self.assertEqual(args.checkpoint_policy, "auto")

    def test_main_builds_one_recursive_train_loader(self):
        with tempfile.TemporaryDirectory(dir=".") as temporary:
            data_root = Path(temporary) / "data"
            (data_root / "nested").mkdir(parents=True)
            sample = np.zeros((12, 12), dtype=np.float32)
            sample[:, 3] = np.arange(12) % 5
            sample[:, 4] = 1.0
            sample[:, 7] = 0.5
            sample[:, 11] = np.arange(12) // 3
            np.savetxt(data_root / "first.txt", sample)
            np.savetxt(data_root / "nested" / "second.txt", sample)

            args = train_cst_pred.parse_args([
                "--data_root",
                str(data_root),
                "--model",
                "pointnet",
                "--epoch",
                "1",
                "--bs",
                "2",
                "--n_points",
                "8",
                "--workers",
                "0",
                "--checkpoint_root",
                str(Path(temporary) / "checkpoints"),
            ])
            run = Mock(id="test-run")
            trainer = Mock()
            with (
                patch(
                    "train_cst_pred.initialize_wandb_run",
                    return_value=run,
                ),
                patch(
                    "train_cst_pred.CstPredTrainer",
                    return_value=trainer,
                ) as trainer_class,
            ):
                train_cst_pred.main(args)

        trainer.start.assert_called_once_with()
        run.finish.assert_called_once_with()
        trainer_kwargs = trainer_class.call_args.kwargs
        self.assertNotIn("test_loader", trainer_kwargs)
        self.assertEqual(trainer_kwargs["checkpoint_action"], "scratch")
        self.assertEqual(trainer_kwargs["checkpoint_source"], "")
        self.assertEqual(
            Path(trainer_kwargs["checkpoint_dir"]),
            Path(temporary) / "checkpoints" / "pointnet" / "semantic",
        )
        self.assertEqual(len(trainer_kwargs["train_loader"].dataset), 2)
        self.assertIsInstance(
            trainer_kwargs["train_loader"].dataset,
            Stage1ConstraintDataset,
        )

    def test_geometry_auto_initializes_from_semantic_best_checkpoint(self):
        with tempfile.TemporaryDirectory(dir=".") as temporary:
            temporary = Path(temporary)
            data_root = temporary / "data"
            data_root.mkdir()
            sample = np.zeros((8, 12), dtype=np.float32)
            sample[:, 3] = np.arange(8) % 5
            sample[:, 4] = 1.0
            sample[:, 7] = 0.5
            sample[:, 11] = np.arange(8) // 2
            np.savetxt(data_root / "sample.txt", sample)

            checkpoint_root = temporary / "checkpoints"
            semantic_best = (
                checkpoint_root
                / "pointnet"
                / "semantic"
                / "best_constraint_score.pth"
            )
            semantic_best.parent.mkdir(parents=True)
            semantic_best.touch()

            args = train_cst_pred.parse_args([
                "--data_root",
                str(data_root),
                "--model",
                "pointnet",
                "--train_phase",
                "geometry",
                "--epoch",
                "1",
                "--bs",
                "1",
                "--n_points",
                "8",
                "--workers",
                "0",
                "--checkpoint_root",
                str(checkpoint_root),
            ])
            run = Mock(id="geometry-run")
            trainer = Mock()
            with (
                patch("train_cst_pred.initialize_wandb_run", return_value=run),
                patch(
                    "train_cst_pred.CstPredTrainer",
                    return_value=trainer,
                ) as trainer_class,
            ):
                train_cst_pred.main(args)

        trainer_kwargs = trainer_class.call_args.kwargs
        self.assertEqual(trainer_kwargs["checkpoint_action"], "init")
        self.assertEqual(
            Path(trainer_kwargs["checkpoint_source"]),
            semantic_best,
        )
        self.assertEqual(
            Path(trainer_kwargs["checkpoint_dir"]),
            checkpoint_root / "pointnet" / "geometry",
        )
        trainer.start.assert_called_once_with()
        run.finish.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
