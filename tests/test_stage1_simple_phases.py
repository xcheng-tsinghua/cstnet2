import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
import torch.nn.functional as F

from functional.cst_pred_trainer import CstPredTrainer
from functional.stage1_phase_loss import stage1_phase_loss, TRAINING_RECIPE
from networks.cst_pred_wrapper import CstPredWrapper
from train_cst_pred import parse_args


class SimplePhasesTest(unittest.TestCase):
    def test_loss_terms_and_gradients_follow_phase(self):
        pmt = torch.tensor([[0, 1, 2, 3, 4, 0]])
        targets = (torch.randn(1, 6, 3), torch.rand(1, 6), torch.randn(1, 6, 3))
        affiliate = torch.tensor([[0, 1, 2, 3, 4, 0]])
        expected = {
            "semantic": {"pmt", "cluster"},
            "geometry": {"mad", "dim", "loc"},
            "joint": {"pmt", "cluster", "mad", "dim", "loc"},
        }
        field = {"pmt": "log_pmt", "cluster": "embedding", "mad": "mad", "dim": "dim", "loc": "loc"}
        for phase, names in expected.items():
            predictions = {
                "log_pmt": torch.randn(1, 6, 5, requires_grad=True),
                "embedding": torch.randn(1, 6, 8, requires_grad=True),
                "mad": torch.randn(1, 6, 3, requires_grad=True),
                "dim": torch.rand(1, 6, requires_grad=True),
                "loc": torch.randn(1, 6, 3, requires_grad=True),
            }
            # Old geometric regularizers and observability are never part of this route.
            with (
                mock.patch("functional.loss._parameter_observability_weights", side_effect=AssertionError("unexpected")),
                mock.patch("functional.loss._stage1_geometry_losses", side_effect=AssertionError("unexpected")),
                mock.patch("functional.loss.instance_consistency_loss", side_effect=AssertionError("unexpected")),
            ):
                total, logs = stage1_phase_loss(predictions, pmt, *targets, affiliate, train_phase=phase)
            self.assertEqual({k[4:] for k in logs if k.startswith("raw/")}, names)
            torch.testing.assert_close(total, sum(logs[f"raw/{name}"] for name in names))
            total.backward()
            for name, key in field.items():
                self.assertEqual(predictions[key].grad is not None, name in names)

    def test_semantic_skips_all_attribute_computations(self):
        predictions = {
            "log_pmt": F.log_softmax(torch.randn(1, 6, 5), -1),
            "embedding": torch.randn(1, 6, 8),
        }
        with mock.patch("functional.stage1_phase_loss._masked_mse", side_effect=AssertionError("unexpected")):
            loss, _ = stage1_phase_loss(
                predictions, torch.zeros(1, 6, dtype=torch.long), None, None, None,
                torch.zeros(1, 6, dtype=torch.long), train_phase="semantic",
            )
        self.assertTrue(torch.isfinite(loss))

    def test_geometry_is_plain_valid_point_mse(self):
        pmt = torch.tensor([[0, 1, 2, 3, 4]])
        target_dir = torch.tensor([[[0., 0., 1.]] * 5])
        predictions = {"mad": -target_dir, "dim": torch.tensor([[99., 1., 2., 3., 99.]]),
                       "loc": torch.tensor([[[1., 0., 0.], [2., 0., 0.], [3., 0., 0.], [4., 0., 0.], [99., 99., 99.]]])}
        with mock.patch("functional.stage1_phase_loss.discriminative_loss", side_effect=AssertionError("unexpected")):
            _, logs = stage1_phase_loss(
                predictions, pmt, target_dir, torch.zeros(1, 5), torch.zeros(1, 5, 3),
                None, train_phase="geometry",
            )
        self.assertEqual(float(logs["raw/mad"]), 0.)
        self.assertAlmostEqual(float(logs["raw/dim"]), 14 / 3, places=6)
        self.assertAlmostEqual(float(logs["raw/loc"]), 30 / 12, places=6)

    def test_joint_reenables_frozen_batchnorm_and_updates_low_backbone(self):
        model = CstPredWrapper("pointnet")
        model.set_train_phase("geometry")
        model.train()
        model.apply_train_phase_mode()
        self.assertFalse(model.embedding.training)
        model.set_train_phase("joint")
        model.train()
        model.apply_train_phase_mode()
        self.assertTrue(all(m.training for m in model.modules()))
        before = model.embedding.conv1.weight.detach().clone()
        with tempfile.TemporaryDirectory(dir=".") as root:
            trainer = CstPredTrainer(model, [], root, str(Path(root) / "log.json"), 1, 1e-5, "test", train_phase="joint")
            xyz = torch.randn(2, 24, 3)
            batch = (xyz, torch.ones(2, 24, dtype=torch.long), F.normalize(torch.randn_like(xyz), dim=-1),
                     torch.ones(2, 24), torch.randn_like(xyz), torch.zeros(2, 24, dtype=torch.long))
            trainer.process_batch(batch, 0, True)
        self.assertFalse(torch.equal(before, model.embedding.conv1.weight))

    def test_old_recipe_cannot_silently_resume(self):
        with tempfile.TemporaryDirectory(dir=".") as root:
            trainer = CstPredTrainer(CstPredWrapper("pointnet"), [], root, str(Path(root) / "log.json"), 1, 1e-4, "test")
            state = trainer._checkpoint_payload(0)
            self.assertEqual(state["args"]["training_recipe"], TRAINING_RECIPE)
            state["checkpoint_config"].pop("training_recipe")
            with self.assertRaisesRegex(ValueError, "checkpoint_policy restart"):
                trainer._validate_resume_config(state)

    def test_geometry_can_initialize_from_existing_semantic_weights(self):
        with tempfile.TemporaryDirectory(dir=".") as root:
            source_model = CstPredWrapper("pointnet")
            path = Path(root) / "semantic.pth"
            torch.save({"model": source_model.state_dict(), "args": {
                "constraint_route": "direct_mlp_v1", "train_phase": "semantic",
            }}, path)
            trainer = CstPredTrainer(
                CstPredWrapper("pointnet"), [], root, str(Path(root) / "log.json"),
                1, 1e-4, "test", train_phase="geometry",
                checkpoint_action="init", checkpoint_source=str(path),
            )
            self.assertEqual(trainer.start_epoch, 0)
            self.assertEqual(len(trainer.optimizer.state), 0)
            for name, tensor in source_model.state_dict().items():
                torch.testing.assert_close(trainer.model.state_dict()[name], tensor)

    def test_semantic_does_not_update_attribute_heads_or_buffers(self):
        with tempfile.TemporaryDirectory(dir=".") as root:
            model = CstPredWrapper("attn_3dgcn")
            xyz = torch.randn(2, 24, 3)
            batch = (xyz, torch.ones(2, 24, dtype=torch.long), F.normalize(torch.randn_like(xyz), dim=-1),
                     torch.ones(2, 24), torch.randn_like(xyz), torch.zeros(2, 24, dtype=torch.long))
            trainer = CstPredTrainer(model, [batch], root, str(Path(root) / "log.json"), 1, 1e-4, "test")
            before = {name: value.clone() for name, value in model.state_dict().items()}
            trainer.process_epoch(0)
            for name, tensor in model.state_dict().items():
                if name.startswith(("mad_head.", "dim_head.", "loc_head.")):
                    torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)
            for prefix in ("embedding.", "cls_head.", "emb_head."):
                self.assertTrue(any(not torch.equal(tensor, before[name])
                                    for name, tensor in model.state_dict().items() if name.startswith(prefix)))

    def test_cli_has_only_five_loss_weights_and_small_joint_lr(self):
        args = parse_args(["--train_phase", "joint"])
        self.assertEqual(args.lr, 1e-5)
        self.assertEqual({k for k in vars(args) if k.startswith("w_")}, {"w_pmt", "w_cluster", "w_mad", "w_dim", "w_loc"})
        self.assertTrue(all(getattr(args, k) == 1.0 for k in vars(args) if k.startswith("w_")))
        for key in ("geom_start_epoch", "geom_ramp_epochs", "joint_backbone_lr_scale", "enable_geom_loss", "enable_inst_loss"):
            self.assertFalse(hasattr(args, key))


if __name__ == "__main__":
    unittest.main()
