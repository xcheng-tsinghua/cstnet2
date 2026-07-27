from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

import eval_cst_pred
from networks.cst_pred_wrapper import CstPredWrapper


def _sample(point_count, offset):
    rng = np.random.default_rng(100 + offset)
    sample = np.zeros((point_count, 12), dtype=np.float32)
    sample[:, 0:3] = rng.normal(size=(point_count, 3))
    sample[:, 3] = np.arange(point_count) % 5
    sample[:, 4:7] = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    sample[:, 7] = 0.5
    sample[:, 8:11] = rng.normal(size=(point_count, 3)) * 0.1
    sample[:, 11] = np.arange(point_count) // 4
    return sample


class EvaluateStage1Test(unittest.TestCase):
    def test_checkpoint_evaluates_every_txt_and_writes_report(self):
        with tempfile.TemporaryDirectory(dir=".") as temporary:
            root = Path(temporary)
            data_root = root / "evaluation_data"
            (data_root / "nested").mkdir(parents=True)
            np.savetxt(data_root / "first.txt", _sample(12, 0))
            np.savetxt(data_root / "nested" / "second.txt", _sample(12, 1))

            checkpoint_path = root / "last.pth"
            model = CstPredWrapper("pointnet")
            torch.save({
                "epoch": 3,
                "model": model.state_dict(),
                "args": {
                    "model": "pointnet",
                    "n_points": 8,
                    "train_phase": "joint",
                    "use_extra_features": False,
                    "feature_k": 16,
                    "cluster_bandwidth": 0.35,
                    "geom_start_epoch": 0,
                    "geom_ramp_epochs": 1,
                },
            }, checkpoint_path)
            report_path = root / "report.json"
            args = eval_cst_pred.parse_args([
                "--data_root",
                str(data_root),
                "--checkpoint",
                str(checkpoint_path),
                "--bs",
                "2",
                "--workers",
                "0",
                "--device",
                "cpu",
                "--output_json",
                str(report_path),
            ])
            report = eval_cst_pred.main(args)

            with report_path.open(encoding="utf-8") as file:
                saved_report = json.load(file)

        self.assertEqual(report["dataset_file_count"], 2)
        self.assertEqual(saved_report["checkpoint_epoch"], 3)
        self.assertIn("loss_all", saved_report["loss"])
        self.assertIn("pmt_miou", saved_report["metrics"])
        self.assertIn("cluster_ari_real", saved_report["metrics"])


if __name__ == "__main__":
    unittest.main()
