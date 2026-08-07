import os
import sys
import tempfile
import unittest

import torch


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from functional.stage1_direct_loss import direct_constraint_loss
from functional.stage1_direct_trainer import (
    DIRECT_CHECKPOINT_TASK,
    Stage1DirectTrainer,
    load_direct_checkpoint,
)
from networks.stage1_direct_baselines import (
    DIRECT_BASELINE_MODEL_NAMES,
    build_stage1_direct_baseline,
    finalize_direct_constraints,
)
from train_stage1_direct_baseline import parse_args


def small_config(model):
    return {
        "model": model,
        "feature_dim": 16,
        "head_hidden_dim": 8,
        "head_dropout": 0.0,
        "dgcnn_k": 4,
        "attn_neighbors": 4,
        "attn_k": 4,
        "pointtransformer_k": 4,
        "pointtransformer_width": 8,
        "pointtransformer_depth": 1,
        "pointmamba_tokens": 8,
        "pointmamba_group_size": 4,
        "pointmamba_width": 8,
        "pointmamba_depth": 1,
        "pointnext_k": 4,
        "pointmlp_group_size": 4,
    }


def synthetic_batch(batch_size=2, n_points=16):
    torch.manual_seed(9)
    xyz = torch.randn(batch_size, n_points, 3)
    primitive = torch.arange(n_points).view(1, -1).repeat(batch_size, 1) % 5
    direction = torch.nn.functional.normalize(
        torch.randn(batch_size, n_points, 3), dim=-1
    )
    dimension = torch.rand(batch_size, n_points) + 0.1
    location = torch.randn(batch_size, n_points, 3) * 0.2
    affiliate = (torch.arange(n_points) // 4).view(1, -1).repeat(batch_size, 1)
    return xyz, primitive, direction, dimension, location, affiliate


class Stage1DirectBaselineTest(unittest.TestCase):
    def test_all_eight_backbones_have_common_four_head_output(self):
        xyz = torch.randn(1, 32, 3)
        for model_name in DIRECT_BASELINE_MODEL_NAMES:
            with self.subTest(model=model_name):
                model = build_stage1_direct_baseline(small_config(model_name)).eval()
                with torch.no_grad():
                    output = model(xyz)
                    constraints = model.predict_constraints(xyz)
                self.assertEqual(tuple(output["log_pmt"].shape), (1, 32, 5))
                self.assertEqual(tuple(output["mad"].shape), (1, 32, 3))
                self.assertEqual(tuple(output["dim"].shape), (1, 32))
                self.assertEqual(tuple(output["loc"].shape), (1, 32, 3))
                self.assertEqual(tuple(constraints.shape), (1, 32, 12))
                self.assertNotIn("embedding", output)
                self.assertTrue(all(torch.isfinite(value).all() for value in output.values()))

    def test_finalization_applies_direction_and_invalid_sentinels(self):
        primitive = torch.tensor([[0, 1, 2, 3, 4]])
        logits = torch.full((1, 5, 5), -20.0)
        logits.scatter_(-1, primitive.unsqueeze(-1), 20.0)
        prediction = {
            "log_pmt": torch.log_softmax(logits, dim=-1),
            "mad": torch.tensor(
                [[[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0],
                  [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]
            ),
            "dim": torch.tensor([[9.0, 2.0, 0.3, 4.0, 9.0]]),
            "loc": torch.ones(1, 5, 3),
        }
        finalized = finalize_direct_constraints(prediction)
        self.assertTrue(torch.equal(finalized["primitive_type"].argmax(-1), primitive))
        self.assertTrue(torch.allclose(finalized["direction"][0, 0], torch.tensor([1.0, 0.0, 0.0])))
        self.assertTrue(torch.allclose(finalized["direction"][0, 1], torch.tensor([0.0, 1.0, 0.0])))
        self.assertTrue(torch.allclose(finalized["direction"][0, 2], torch.tensor([0.0, 0.0, 1.0])))
        self.assertTrue(torch.allclose(finalized["direction"][0, 3], torch.tensor([0.0, 0.0, -1.0])))
        self.assertTrue(torch.allclose(finalized["direction"][0, 4], torch.tensor([0.0, 0.0, -1.0])))
        self.assertEqual(float(finalized["dimension"][0, 0]), -1.0)
        self.assertEqual(float(finalized["dimension"][0, 4]), -1.0)
        self.assertTrue(torch.equal(finalized["location"][0, 4], torch.zeros(3)))

    def test_loss_ignores_undefined_component_targets(self):
        primitive = torch.tensor([[0, 1, 2, 3, 4]])
        logits = torch.full((1, 5, 5), -30.0)
        logits.scatter_(-1, primitive.unsqueeze(-1), 30.0)
        direction_gt = torch.tensor([[[1.0, 0.0, 0.0]] * 5])
        direction_pred = direction_gt.clone()
        direction_pred[0, 0] *= -1.0
        direction_pred[0, 3:] = 100.0
        dimension_gt = torch.tensor([[-1.0, 2.0, 0.3, 4.0, -1.0]])
        dimension_pred = dimension_gt.clone()
        dimension_pred[0, [0, 4]] = 100.0
        location_gt = torch.zeros(1, 5, 3)
        location_pred = location_gt.clone()
        location_pred[0, 4] = 100.0
        predictions = {
            "log_pmt": torch.log_softmax(logits, dim=-1),
            "mad": direction_pred,
            "dim": dimension_pred,
            "loc": location_pred,
        }
        loss, losses = direct_constraint_loss(
            predictions,
            primitive,
            direction_gt,
            dimension_gt,
            location_gt,
        )
        self.assertLess(float(loss), 1e-6)
        self.assertEqual(float(losses["mad_loss"]), 0.0)
        self.assertEqual(float(losses["dim_loss"]), 0.0)
        self.assertEqual(float(losses["loc_loss"]), 0.0)

    def test_independent_training_checkpoint_and_resume(self):
        config = small_config("pointtransformer")
        batch = synthetic_batch()
        with tempfile.TemporaryDirectory(dir=".") as temporary:
            model = build_stage1_direct_baseline(config)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, 0.9)
            trainer = Stage1DirectTrainer(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                train_loader=[batch],
                val_loader=[batch],
                output_dir=temporary,
                device=torch.device("cpu"),
                epochs=1,
                loss_weights={"w_pmt": 1.0, "w_mad": 0.02, "w_dim": 0.05, "w_loc": 0.02},
                checkpoint_args=config,
                wandb_run=None,
            )
            summary = trainer.fit()
            self.assertIn("final/direction_mean_angular_error_deg", summary["val"])
            for filename in ("last.pth", "best_loss.pth", "best_pmt_miou.pth", "history.json"):
                self.assertTrue(os.path.isfile(os.path.join(temporary, filename)))
            checkpoint = load_direct_checkpoint(os.path.join(temporary, "last.pth"))
            self.assertEqual(checkpoint["task"], DIRECT_CHECKPOINT_TASK)
            self.assertNotIn("embedding", checkpoint["model_config"])

            resumed_model = build_stage1_direct_baseline(config)
            resumed_optimizer = torch.optim.Adam(resumed_model.parameters(), lr=1e-3)
            resumed_scheduler = torch.optim.lr_scheduler.StepLR(resumed_optimizer, 1, 0.9)
            resumed = Stage1DirectTrainer(
                model=resumed_model,
                optimizer=resumed_optimizer,
                scheduler=resumed_scheduler,
                train_loader=[batch],
                val_loader=[batch],
                output_dir=temporary,
                device=torch.device("cpu"),
                epochs=2,
                loss_weights={"w_pmt": 1.0, "w_mad": 0.02, "w_dim": 0.05, "w_loc": 0.02},
                checkpoint_args=config,
                wandb_run=None,
            )
            resumed.load_checkpoint(os.path.join(temporary, "last.pth"))
            self.assertEqual(resumed.start_epoch, 1)
            self.assertEqual(resumed.global_step, 1)

    def test_training_entry_lists_only_direct_baselines(self):
        for model_name in DIRECT_BASELINE_MODEL_NAMES:
            args = parse_args(["--model", model_name])
            self.assertEqual(args.model, model_name)
        defaults = parse_args([])
        self.assertEqual(defaults.model, "pointnet2")


if __name__ == "__main__":
    unittest.main()
