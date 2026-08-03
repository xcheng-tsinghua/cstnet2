from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

import train_cst_pred
from data_utils.stage1_dataset import Stage1ConstraintDataset


class TrainStage1EntryTest(unittest.TestCase):
    def test_data_root_is_required(self):
        with self.assertRaises(SystemExit):
            train_cst_pred.parse_args([])

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
        self.assertEqual(len(trainer_kwargs["train_loader"].dataset), 2)
        self.assertIsInstance(
            trainer_kwargs["train_loader"].dataset,
            Stage1ConstraintDataset,
        )


if __name__ == "__main__":
    unittest.main()
