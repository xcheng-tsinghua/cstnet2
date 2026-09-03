import os
import sys
import tempfile
import unittest
import json
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
from functional.loss import constraint_loss, instance_consistency_loss
from functional.stage1_metrics import evaluate_constraint_attribute_metrics
from networks.cst_pred_wrapper import CstPredWrapper


LOSS_NAMES = ("mad", "dim", "loc", "geom", "inst")


def synthetic_batch(batch_size=2, n_points=20):
    torch.manual_seed(17)
    xyz = torch.randn(batch_size, n_points, 3)
    primitive = torch.arange(n_points).view(1, -1).repeat(batch_size, 1) % 5
    direction = torch.nn.functional.normalize(
        torch.randn(batch_size, n_points, 3), dim=-1
    )
    dimension = torch.rand(batch_size, n_points) + 0.2
    location = torch.randn(batch_size, n_points, 3) * 0.1
    affiliate = (torch.arange(n_points) // 4).view(1, -1).repeat(batch_size, 1)
    return xyz, primitive, direction, dimension, location, affiliate


def checkpoint_args(phase, n_points=20):
    return {
        "model": "pointnet",
        "train_phase": phase,
        "use_extra_features": False,
        "feature_k": 16,
        "n_points": n_points,
        "w_pmt": 1.0,
        "w_cluster": 0.5,
        "w_mad": 0.02,
        "w_dim": 0.05,
        "w_loc": 0.02,
        "w_geom": 0.02,
        "w_inst": 0.005,
        "enable_mad_loss": True,
        "enable_dim_loss": True,
        "enable_loc_loss": True,
        "enable_geom_loss": True,
        "enable_inst_loss": True,
        "geom_start_epoch": 0,
        "geom_ramp_epochs": 4,
        "joint_backbone_lr_scale": 0.1,
        "use_amp": False,
    }


def make_trainer(
    root,
    phase,
    max_epoch=1,
    checkpoint_action="scratch",
    checkpoint_source="",
    args_override=None,
):
    batch = synthetic_batch()
    return CstPredTrainer(
        model=CstPredWrapper("pointnet"),
        train_loader=[batch],
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
        checkpoint_action=checkpoint_action,
        checkpoint_source=checkpoint_source,
    )


class Stage1TrainingStabilityTest(unittest.TestCase):
    def test_constraint_attribute_errors_use_valid_primitive_masks(self):
        primitive = torch.tensor([[0, 1, 2, 3, 4]])
        direction_gt = torch.tensor(
            [[[1.0, 0.0, 0.0]] * 5],
        )
        direction_pred = direction_gt.clone()
        direction_pred[0, 1] = torch.tensor([0.0, 1.0, 0.0])
        direction_gt[0, 2] = torch.nn.functional.normalize(
            torch.tensor([1.0, 0.0, 1e-7]), dim=0
        )
        direction_pred[0, 2] = torch.nn.functional.normalize(
            torch.tensor([-1.0, 0.0, 1e-7]), dim=0
        )

        dimension_gt = torch.zeros(1, 5)
        dimension_pred = torch.tensor([[100.0, 1.0, 2.0, 3.0, 100.0]])
        location_gt = torch.zeros(1, 5, 3)
        location_pred = torch.zeros(1, 5, 3)
        location_pred[0, :4, 0] = torch.tensor([1.0, 2.0, 3.0, 4.0])
        location_pred[0, 4, 0] = 100.0

        batch_metrics = evaluate_constraint_attribute_metrics(
            direction_pred,
            dimension_pred,
            location_pred,
            primitive,
            direction_gt,
            dimension_gt,
            location_gt,
        )
        metrics = _aggregate_metric_dicts([batch_metrics])

        self.assertAlmostEqual(
            metrics["direction_mean_angular_error_deg"], 30.0, places=4
        )
        self.assertAlmostEqual(
            metrics["dimension_mean_absolute_error"], 2.0, places=6
        )
        self.assertAlmostEqual(
            metrics["location_mean_distance_error"], 2.5, places=6
        )
        self.assertEqual(metrics["direction_valid_points"], 3.0)
        self.assertEqual(metrics["dimension_valid_points"], 3.0)
        self.assertEqual(metrics["location_valid_points"], 4.0)

    def test_direction_loss_is_sign_invariant_at_dir_unify_boundary(self):
        primitive = torch.tensor([[1]])
        logits = torch.full((1, 1, 5), -20.0)
        logits[..., 1] = 20.0
        direction_gt = torch.nn.functional.normalize(
            torch.tensor([[[1.0, 0.0, 1e-7]]]), dim=-1
        )
        direction_pred = torch.nn.functional.normalize(
            torch.tensor([[[-1.0, 0.0, 1e-7]]]), dim=-1
        )
        xyz = torch.zeros(1, 1, 3)
        dimension = torch.ones(1, 1)
        location = torch.zeros(1, 1, 3)
        affiliate = torch.zeros(1, 1, dtype=torch.long)

        total, losses = constraint_loss(
            xyz=xyz,
            log_pmt_pred=torch.log_softmax(logits, dim=-1),
            mad_pred=direction_pred,
            dim_pred=dimension,
            loc_pred=location,
            pmt_gt=primitive,
            mad_gt=direction_gt,
            dim_gt=dimension,
            loc_gt=location,
            affil_idx=affiliate,
            point_emb=None,
            weights={
                "w_pmt": 0.0,
                "w_cluster": 0.0,
                "w_mad": 1.0,
                "w_dim": 0.0,
                "w_loc": 0.0,
                "w_geom": 0.0,
                "w_inst": 0.0,
            },
            global_epoch=0,
        )
        batch_metrics = evaluate_constraint_attribute_metrics(
            direction_pred,
            dimension,
            location,
            primitive,
            direction_gt,
            dimension,
            location,
        )
        metrics = _aggregate_metric_dicts([batch_metrics])

        self.assertLess(float(losses["raw/mad"]), 1e-10)
        self.assertLess(float(total), 1e-10)
        self.assertLess(metrics["direction_mean_angular_error_deg"], 1e-4)

    def test_instance_direction_consistency_is_sign_invariant(self):
        direction = torch.nn.functional.normalize(
            torch.tensor(
                [[[1.0, 0.0, 1e-7], [-1.0, 0.0, 1e-7]]]
            ),
            dim=-1,
        )
        logits = torch.zeros(1, 2, 5).log_softmax(dim=-1)
        loss = instance_consistency_loss(
            logits,
            direction,
            torch.ones(1, 2),
            torch.zeros(1, 2, 3),
            torch.zeros(1, 2, dtype=torch.long),
            torch.ones(1, 2, dtype=torch.long),
        )
        self.assertLess(float(loss), 1e-10)

    def test_remote_location_and_large_radius_have_bounded_gradients(self):
        point_count = 4
        xyz = torch.tensor(
            [[[-0.1, 0.0, 0.0], [0.1, 0.0, 0.0],
              [0.0, -0.1, 0.0], [0.0, 0.1, 0.0]]]
        )
        primitive = torch.ones(1, point_count, dtype=torch.long)
        logits = torch.zeros(1, point_count, 5).log_softmax(dim=-1)
        direction = torch.tensor([[[0.0, 0.0, 1.0]] * point_count])
        dimension_gt = torch.full((1, point_count), 1000.0)
        dimension_pred = torch.ones(1, point_count, requires_grad=True)
        location_gt = torch.tensor([[[1000.0, 0.0, 0.0]] * point_count])
        location_pred = torch.zeros(1, point_count, 3, requires_grad=True)
        affiliate = torch.zeros(1, point_count, dtype=torch.long)

        total, losses = constraint_loss(
            xyz,
            logits,
            direction,
            dimension_pred,
            location_pred,
            primitive,
            direction,
            dimension_gt,
            location_gt,
            affiliate,
            weights={
                "w_pmt": 0.0,
                "w_cluster": 0.0,
                "w_mad": 0.0,
                "w_dim": 1.0,
                "w_loc": 1.0,
                "w_geom": 0.0,
                "w_inst": 0.0,
            },
            global_epoch=0,
        )
        total.backward()

        self.assertTrue(torch.isfinite(total))
        self.assertLess(float(losses["observability/dim_mean"]), 1.0)
        self.assertLess(float(losses["observability/loc_mean"]), 1.0)
        self.assertLess(float(dimension_pred.grad.abs().max()), 0.01)
        self.assertLess(float(location_pred.grad.abs().max()), 0.01)

    def test_parameter_losses_average_primitive_instances_not_points(self):
        primitive = torch.full((1, 4), 2, dtype=torch.long)
        logits = torch.zeros(1, 4, 5).log_softmax(dim=-1)
        direction = torch.tensor([[[0.0, 0.0, 1.0]] * 4])
        dimension_pred = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        dimension_gt = torch.zeros(1, 4)
        location = torch.zeros(1, 4, 3)
        affiliate = torch.tensor([[0, 1, 1, 1]])

        _, losses = constraint_loss(
            torch.zeros(1, 4, 3),
            logits,
            direction,
            dimension_pred,
            location,
            primitive,
            direction,
            dimension_gt,
            location,
            affiliate,
            weights={
                "w_pmt": 0.0,
                "w_cluster": 0.0,
                "w_mad": 0.0,
                "w_dim": 1.0,
                "w_loc": 0.0,
                "w_geom": 0.0,
                "w_inst": 0.0,
            },
            global_epoch=0,
        )

        # Smooth L1(1, beta=0.1) is 0.95. One erroneous instance and one
        # exact instance therefore average to 0.475 regardless of point count.
        self.assertAlmostEqual(float(losses["raw/dim"]), 0.475, places=6)

    def test_remote_cone_geometry_uses_bounded_angular_residual(self):
        point_count = 4
        xyz = torch.tensor(
            [[[-0.1, 0.0, 0.0], [0.1, 0.0, 0.0],
              [0.0, -0.1, 0.0], [0.0, 0.1, 0.0]]]
        )
        primitive = torch.full((1, point_count), 2, dtype=torch.long)
        logits = torch.zeros(1, point_count, 5).log_softmax(dim=-1)
        direction = torch.tensor([[[0.0, 0.0, 1.0]] * point_count])
        dimension = torch.full((1, point_count), 0.5, requires_grad=True)
        location = torch.tensor(
            [[[0.0, 0.0, 1000.0]] * point_count], requires_grad=True
        )
        affiliate = torch.zeros(1, point_count, dtype=torch.long)

        total, losses = constraint_loss(
            xyz,
            logits,
            direction,
            dimension,
            location,
            primitive,
            direction,
            dimension.detach(),
            location.detach(),
            affiliate,
            weights={
                "w_pmt": 0.0,
                "w_cluster": 0.0,
                "w_mad": 0.0,
                "w_dim": 0.0,
                "w_loc": 0.0,
                "w_geom": 1.0,
                "w_inst": 0.0,
            },
            global_epoch=1,
            geom_start_epoch=0,
            geom_ramp_epochs=0,
        )
        total.backward()

        self.assertLess(float(losses["raw/geom"]), 0.1)
        self.assertLess(float(dimension.grad.abs().max()), 0.02)
        self.assertLess(float(location.grad.abs().max()), 0.02)

    def test_real_clustering_metrics_can_be_sampled(self):
        with tempfile.TemporaryDirectory(dir=".") as root:
            trainer = make_trainer(root, "semantic")
            trainer.cluster_metric_interval = 3
            self.assertTrue(trainer._should_compute_cluster_metrics(0))
            self.assertFalse(trainer._should_compute_cluster_metrics(1))
            self.assertTrue(trainer._should_compute_cluster_metrics(3))
            trainer.cluster_metric_interval = 0
            self.assertTrue(trainer._should_compute_cluster_metrics(0))
            self.assertFalse(trainer._should_compute_cluster_metrics(1))

            with mock.patch(
                "functional.cst_pred_trainer.evaluate_predicted_clustering"
            ) as real_metric, mock.patch(
                "functional.cst_pred_trainer.evaluate_clustering"
            ) as oracle_metric:
                _, metrics = trainer.process_batch(
                    synthetic_batch(),
                    global_epoch=0,
                    is_train=False,
                    compute_cluster_metrics=False,
                )
            real_metric.assert_not_called()
            oracle_metric.assert_not_called()
            self.assertNotIn("cluster_ari_real", metrics)

    def test_stage1_checkpoint_persists_wandb_run_id(self):
        with tempfile.TemporaryDirectory(dir=".") as root:
            trainer = make_trainer(root, "semantic")
            trainer.wandb_run = SimpleNamespace(id="stage1-run-id")
            self.assertEqual(
                trainer._checkpoint_payload(0)["wandb_run_id"],
                "stage1-run-id",
            )

    def test_epoch_fitted_metrics_use_complete_route_outputs(self):
        with tempfile.TemporaryDirectory(dir=".") as root:
            trainer = make_trainer(root, "joint")
            batch = synthetic_batch()
            fitted_constraints = {
                "direction": batch[2].clone(),
                "dimension": batch[3].clone(),
                "location": batch[4].clone(),
            }
            with mock.patch(
                "functional.cst_pred_trainer.assemble_constraints_from_stage1",
                return_value=fitted_constraints,
            ) as assemble:
                metrics = trainer.evaluate_fitted_epoch(global_epoch=0)

            self.assertLess(metrics["direction_mean_angular_error_deg"], 0.01)
            self.assertEqual(metrics["dimension_mean_absolute_error"], 0.0)
            self.assertEqual(metrics["location_mean_distance_error"], 0.0)
            fitter_args = assemble.call_args.kwargs
            self.assertIsNotNone(fitter_args["mad_prediction"])
            self.assertIsNotNone(fitter_args["dim_prediction"])
            self.assertIsNotNone(fitter_args["loc_prediction"])
            self.assertTrue(fitter_args["use_prediction_initialization"])
            self.assertTrue(fitter_args["use_pca_normals_for_fitting"])

    def test_stage1_wandb_metrics_do_not_use_redundant_train_prefix(self):
        with tempfile.TemporaryDirectory(dir=".") as root:
            trainer = make_trainer(root, "semantic")
            run = SimpleNamespace(id="stage1-run-id", log=mock.Mock())
            trainer.wandb_run = run
            trainer.process_epoch = mock.Mock(return_value=(
                {"loss_all": 1.0, "raw/pmt": 0.5},
                {
                    "pmt_miou": 0.4,
                    "cluster_ari_real": 0.3,
                    "pmt_confusion_matrix": torch.eye(5),
                },
            ))
            trainer.evaluate_fitted_epoch = mock.Mock(return_value={
                "direction_mean_angular_error_deg": 12.0,
                "dimension_mean_absolute_error": 0.2,
                "location_mean_distance_error": 0.3,
            })
            trainer.append_save_dict = mock.Mock()
            trainer._update_best_metrics = mock.Mock(return_value=[])
            trainer.save = mock.Mock(return_value={"last": True})

            with mock.patch(
                "functional.cst_pred_trainer.wandb_confusion_matrix",
                return_value="confusion-chart",
            ):
                trainer.start()

            payload = run.log.call_args.args[0]
            self.assertIn("loss/loss_all", payload)
            self.assertIn("metric/pmt_miou", payload)
            self.assertEqual(
                payload["metric/fitted/direction_mean_angular_error_deg"],
                12.0,
            )
            self.assertEqual(
                payload["metric/fitted/dimension_mean_absolute_error"],
                0.2,
            )
            self.assertEqual(
                payload["metric/fitted/location_mean_distance_error"],
                0.3,
            )
            self.assertIn("confusion_matrix/primitive", payload)
            self.assertFalse(
                any(
                    key in {"epoch", "global_step"}
                    or key.startswith("train/")
                    or key.startswith("best")
                    or key.startswith("lr/")
                    or key.startswith("time/")
                    or key.startswith("checkpoint/")
                    for key in payload
                )
            )
            self.assertEqual(run.log.call_args.kwargs["step"], 0)

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
                    )
                    self.assertTrue(torch.isfinite(loss["loss_all"]))
                    self.assertIn("pmt_confusion_matrix", metrics)
                    self.assertIn(
                        "_constraint_attribute_sum/direction_angular_error_deg",
                        metrics,
                    )
                    self.assertNotIn("raw/nor", loss)
                    for legacy_key in (
                        "pmt_loss", "cluster_loss", "mad_loss", "dim_loss",
                        "loc_loss", "geom_loss", "inst_loss", "aux_factor",
                        "schedule/global_epoch",
                    ):
                        self.assertNotIn(legacy_key, loss)
                    self.assertFalse(
                        any(
                            key.startswith("grad_")
                            or key.startswith("optimization/gradient_")
                            for key in metrics
                        )
                    )
                    if phase == "geometry":
                        self.assertGreater(float(loss["weighted/mad"]), 0.0)
                        self.assertAlmostEqual(
                            float(loss["effective_weight/mad"]), 0.02
                        )

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

    def test_checkpoint_resume_and_previous_phase_init(self):
        with tempfile.TemporaryDirectory(dir=".") as root:
            first = make_trainer(root, "joint", max_epoch=1)
            initial_lrs = [group["lr"] for group in first.optimizer.param_groups]
            first.scheduler.step_size = 1
            first.start()
            last_path = os.path.join(root, "checkpoints", "last.pth")
            first_state = torch.load(last_path, map_location="cpu")
            with open(os.path.join(root, "log.json"), encoding="utf-8") as file:
                training_log = json.load(file)
            self.assertIn("train", training_log)
            self.assertNotIn("test", training_log)
            self.assertEqual(
                training_log["run"]["best_metrics_source"],
                "train",
            )
            self.assertEqual(first_state["best_metrics_source"], "train")
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
                root,
                "joint",
                max_epoch=2,
                checkpoint_action="resume",
                checkpoint_source=legacy_multitask_path,
            )
            self.assertEqual(legacy_resumed.start_epoch, 1)

            resumed = make_trainer(
                root,
                "joint",
                max_epoch=2,
                checkpoint_action="resume",
                checkpoint_source=last_path,
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
            with self.assertWarnsRegex(RuntimeWarning, "point_count differs"):
                point_count_resumed = make_trainer(
                    root,
                    "joint",
                    max_epoch=2,
                    checkpoint_action="resume",
                    checkpoint_source=last_path,
                    args_override=mismatched_args,
                )
            self.assertEqual(
                point_count_resumed.start_epoch,
                int(resumed_state["epoch"]) + 1,
            )

            incompatible_args = checkpoint_args("joint")
            incompatible_args["model"] = "dgcnn"
            with self.assertRaisesRegex(ValueError, "configuration mismatch"):
                make_trainer(
                    root,
                    "joint",
                    max_epoch=2,
                    checkpoint_action="resume",
                    checkpoint_source=last_path,
                    args_override=incompatible_args,
                )

            initialized = make_trainer(
                root,
                "joint",
                max_epoch=1,
                checkpoint_action="init",
                checkpoint_source=last_path,
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
