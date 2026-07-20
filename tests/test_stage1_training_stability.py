import os
import sys
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from functional.cst_pred_trainer import (
    CstPredTrainer,
    _aggregate_metric_dicts,
    warn_if_primitive_collapsed,
)
from functional.stage1_metrics import evaluate_constraint_attribute_metrics
from networks.cst_pred_wrapper import CstPredWrapper


LOSS_NAMES = ("mad", "dim", "nor", "loc", "geom", "inst")


def synthetic_batch(batch_size=2, n_points=20):
    torch.manual_seed(17)
    xyz = torch.randn(batch_size, n_points, 3)
    primitive = torch.arange(n_points).view(1, -1).repeat(batch_size, 1) % 5
    direction = torch.nn.functional.normalize(
        torch.randn(batch_size, n_points, 3), dim=-1
    )
    dimension = torch.rand(batch_size, n_points) + 0.2
    normal = torch.nn.functional.normalize(
        torch.randn(batch_size, n_points, 3), dim=-1
    )
    location = torch.randn(batch_size, n_points, 3) * 0.1
    affiliate = (torch.arange(n_points) // 4).view(1, -1).repeat(batch_size, 1)
    category = torch.zeros(batch_size, dtype=torch.long)
    return xyz, category, primitive, direction, dimension, normal, location, affiliate


def checkpoint_args(phase, n_points=20):
    return {
        "model": "pointnet",
        "train_phase": phase,
        "use_extra_features": False,
        "normal_source": "none",
        "feature_k": 16,
        "n_points": n_points,
        "w_pmt": 1.0,
        "w_cluster": 0.5,
        "w_mad": 0.02,
        "w_dim": 0.05,
        "w_nor": 0.1,
        "w_loc": 0.02,
        "w_geom": 0.02,
        "w_inst": 0.005,
        "enable_mad_loss": True,
        "enable_dim_loss": True,
        "enable_nor_loss": True,
        "enable_loc_loss": True,
        "enable_geom_loss": True,
        "enable_inst_loss": True,
        "geom_start_epoch": 0,
        "geom_ramp_epochs": 4,
        "joint_backbone_lr_scale": 0.1,
        "use_amp": False,
    }


def make_trainer(
    root, phase, max_epoch=1, resume="", init_from="", args_override=None
):
    batch = synthetic_batch()
    return CstPredTrainer(
        model=CstPredWrapper("pointnet"),
        train_loader=[batch],
        test_loader=[batch],
        checkpoint_dir=os.path.join(root, "checkpoints"),
        log_savepth=os.path.join(root, "log.json"),
        max_epoch=max_epoch,
        lr=1e-4,
        save_str="stage1_stability_smoke",
        train_phase=phase,
        enabled_losses={name: True for name in LOSS_NAMES},
        checkpoint_args=(
            checkpoint_args(phase) if args_override is None else args_override
        ),
        geom_start_epoch=0,
        geom_ramp_epochs=4,
        joint_backbone_lr_scale=0.1,
        resume_checkpoint=resume,
        init_from_checkpoint=init_from,
        enable_grad_diagnostics=False,
    )


class Stage1TrainingStabilityTest(unittest.TestCase):
    def test_constraint_attribute_errors_use_valid_primitive_masks(self):
        primitive = torch.tensor([[0, 1, 2, 3, 4]])
        direction_gt = torch.tensor(
            [[[1.0, 0.0, 0.0]] * 5],
        )
        direction_pred = direction_gt.clone()
        direction_pred[0, 1] = torch.tensor([0.0, 1.0, 0.0])
        direction_pred[0, 2] = torch.tensor([-1.0, 0.0, 0.0])

        continuity_gt = torch.tensor(
            [[[0.0, 0.0, 1.0]] * 5],
        )
        continuity_pred = continuity_gt.clone()
        continuity_pred[0, 4] = torch.tensor([0.0, 0.0, -1.0])

        dimension_gt = torch.zeros(1, 5)
        dimension_pred = torch.tensor([[100.0, 1.0, 2.0, 3.0, 100.0]])
        location_gt = torch.zeros(1, 5, 3)
        location_pred = torch.zeros(1, 5, 3)
        location_pred[0, :4, 0] = torch.tensor([1.0, 2.0, 3.0, 4.0])
        location_pred[0, 4, 0] = 100.0

        batch_metrics = evaluate_constraint_attribute_metrics(
            direction_pred,
            dimension_pred,
            continuity_pred,
            location_pred,
            primitive,
            direction_gt,
            dimension_gt,
            continuity_gt,
            location_gt,
        )
        metrics = _aggregate_metric_dicts([batch_metrics])

        self.assertAlmostEqual(
            metrics["direction_mean_angular_error_deg"], 30.0, places=4
        )
        self.assertAlmostEqual(
            metrics["continuity_mean_angular_error_deg"], 36.0, places=4
        )
        self.assertAlmostEqual(
            metrics["dimension_mean_absolute_error"], 2.0, places=6
        )
        self.assertAlmostEqual(
            metrics["location_mean_distance_error"], 2.5, places=6
        )
        self.assertEqual(metrics["direction_valid_points"], 3.0)
        self.assertEqual(metrics["continuity_valid_points"], 5.0)
        self.assertEqual(metrics["dimension_valid_points"], 3.0)
        self.assertEqual(metrics["location_valid_points"], 4.0)

    def test_stage1_checkpoint_persists_wandb_run_id(self):
        with tempfile.TemporaryDirectory(dir=".") as root:
            trainer = make_trainer(root, "semantic")
            trainer.wandb_run = SimpleNamespace(id="stage1-run-id")
            self.assertEqual(
                trainer._checkpoint_payload(0)["wandb_run_id"],
                "stage1-run-id",
            )

    def test_checkpoint_io_failure_is_reported_without_raising(self):
        with tempfile.TemporaryDirectory(dir=".") as root:
            trainer = make_trainer(root, "semantic")
            with mock.patch(
                "functional.cst_pred_trainer.safe_torch_save",
                return_value=False,
            ) as safe_save:
                status = trainer.save(0, improved=["pmt_miou"])

            self.assertEqual(status, {"last": False, "pmt_miou": False})
            self.assertEqual(safe_save.call_count, 2)

    def test_all_multitask_phase_smokes(self):
        for phase in ("semantic", "geometry", "joint"):
            with self.subTest(phase=phase):
                with tempfile.TemporaryDirectory(dir=".") as root:
                    trainer = make_trainer(root, phase)
                    loss, metrics = trainer.process_batch(
                        synthetic_batch(),
                        global_epoch=1,
                        is_train=True,
                        diagnose_gradients=phase == "semantic",
                    )
                    self.assertTrue(torch.isfinite(loss["loss_all"]))
                    self.assertIn("pmt_confusion_matrix", metrics)
                    self.assertIn(
                        "_constraint_attribute_sum/direction_angular_error_deg",
                        metrics,
                    )
                    if phase == "semantic":
                        self.assertIn("grad_norm/pmt", metrics)
                        self.assertIn("grad_cosine/pmt_vs_geom", metrics)

    def test_geometry_freezes_semantic_path(self):
        with tempfile.TemporaryDirectory(dir=".") as root:
            trainer = make_trainer(root, "geometry")
            trainable = {
                name for name, parameter in trainer.model.named_parameters()
                if parameter.requires_grad
            }
            self.assertTrue(trainable)
            self.assertTrue(all(not name.startswith("embedding.") for name in trainable))
            self.assertTrue(all(not name.startswith("cls_head.") for name in trainable))
            self.assertTrue(all(not name.startswith("emb_head.") for name in trainable))
            self.assertTrue(any(name.startswith("geometry_decoder.") for name in trainable))

    def test_joint_uses_lower_backbone_lr(self):
        with tempfile.TemporaryDirectory(dir=".") as root:
            trainer = make_trainer(root, "joint")
            learning_rates = trainer.current_lrs()
            self.assertAlmostEqual(
                learning_rates["backbone_high"], learning_rates["heads"] * 0.1
            )
            frozen_backbone = [
                parameter
                for name, parameter in trainer.model.named_parameters()
                if name.startswith("embedding.") and not parameter.requires_grad
            ]
            self.assertTrue(frozen_backbone)

    def test_checkpoint_resume_and_model_only_init(self):
        with tempfile.TemporaryDirectory(dir=".") as root:
            first = make_trainer(root, "joint", max_epoch=1)
            initial_lrs = [group["lr"] for group in first.optimizer.param_groups]
            first.scheduler.step_size = 1
            first.start()
            last_path = os.path.join(root, "checkpoints", "last.pth")
            first_state = torch.load(last_path, map_location="cpu")
            first_lrs = [group["lr"] for group in first_state["optimizer"]["param_groups"]]
            for initial_lr, saved_lr in zip(initial_lrs, first_lrs):
                self.assertAlmostEqual(saved_lr, initial_lr * 0.9)
            self.assertEqual(first_state["epoch"], 0)
            self.assertEqual(first_state["global_step"], 1)
            self.assertEqual(first_state["scheduler"]["last_epoch"], 1)
            for filename in (
                "last.pth",
                "best_pmt_miou.pth",
                "best_cluster_ari.pth",
                "best_constraint_score.pth",
            ):
                self.assertTrue(os.path.isfile(os.path.join(root, "checkpoints", filename)))

            legacy_multitask_path = os.path.join(root, "legacy_multitask.pth")
            first_state["args"]["stage1_mode"] = "multitask"
            first_state["checkpoint_config"]["stage1_mode"] = "multitask"
            torch.save(first_state, legacy_multitask_path)
            legacy_resumed = make_trainer(
                root, "joint", max_epoch=2, resume=legacy_multitask_path
            )
            self.assertEqual(legacy_resumed.start_epoch, 1)

            resumed = make_trainer(
                root, "joint", max_epoch=2, resume=last_path
            )
            self.assertEqual(resumed.start_epoch, 1)
            self.assertEqual(resumed.global_step, 1)
            self.assertEqual(
                [group["lr"] for group in resumed.optimizer.param_groups], first_lrs
            )
            self.assertEqual(
                resumed.scheduler.last_epoch, first_state["scheduler"]["last_epoch"]
            )
            resumed.start()
            resumed_state = torch.load(last_path, map_location="cpu")
            self.assertEqual(resumed_state["epoch"], 1)
            self.assertEqual(resumed_state["global_step"], 2)
            self.assertAlmostEqual(resumed_state["loss_schedule"]["aux_progress"], 0.25)

            mismatched_args = checkpoint_args("joint")
            mismatched_args["n_points"] = 21
            with self.assertRaisesRegex(ValueError, "configuration mismatch"):
                make_trainer(
                    root,
                    "joint",
                    max_epoch=2,
                    resume=last_path,
                    args_override=mismatched_args,
                )

            initialized = make_trainer(
                root, "joint", max_epoch=1, init_from=last_path
            )
            self.assertEqual(initialized.start_epoch, 0)
            self.assertEqual(initialized.global_step, 0)
            self.assertEqual(len(initialized.optimizer.state), 0)

    def test_collapse_warning(self):
        self.assertTrue(warn_if_primitive_collapsed({
            "pmt_pred_histogram": [96, 1, 1, 1, 1]
        }, split="synthetic", epoch=0))


if __name__ == "__main__":
    unittest.main()
