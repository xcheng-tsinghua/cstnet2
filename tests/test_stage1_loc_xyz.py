import tempfile
import unittest
from pathlib import Path

import torch

from functional.cst_pred_trainer import CstPredTrainer
from functional.point_features import stage1_forward
from networks.cst_pred_wrapper import CstPredWrapper, LOC_INPUT


class LocXYZTest(unittest.TestCase):
    def test_only_loc_head_input_is_expanded_for_all_backbones(self):
        for backbone in ("pointnet", "pointnet2", "attn_3dgcn"):
            model = CstPredWrapper(backbone)
            self.assertEqual(model.loc_head.linear_layers[0].in_channels, 131)
            self.assertEqual(model.loc_head.linear_layers[0].out_channels, 20)
            self.assertEqual(model.mad_head.linear_layers[0].in_channels, 128)
            self.assertEqual(model.dim_head.linear_layers[0].in_channels, 128)

    def test_xyz_reaches_loc_and_has_gradients_with_frozen_backbone(self):
        model = CstPredWrapper("attn_3dgcn")
        model.set_train_phase("geometry")
        model.eval()
        xyz = torch.randn(2, 64, 3)
        captured = []
        handle = model.loc_head.register_forward_pre_hook(lambda module, args: captured.append(args[0]))
        outputs = model(xyz)
        handle.remove()
        self.assertEqual(captured[0].shape, (2, 131, 64))
        torch.testing.assert_close(captured[0][:, -3:], xyz.transpose(1, 2))
        outputs["loc"].square().mean().backward()
        self.assertGreater(model.loc_head.linear_layers[0].weight.grad[:, -3:].abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in model.embedding.parameters()))

    def test_translated_cloud_can_change_loc_with_same_relative_features(self):
        torch.manual_seed(15)
        xyz = torch.randint(-128, 128, (2, 64, 3)).float() / 256
        for extra in (False, True):
            model = CstPredWrapper("attn_3dgcn", channel_fea=2 if extra else 0).eval()
            # Deterministically route x through the real loc MLP, removing reliance
            # on random weights to demonstrate the newly available information.
            with torch.no_grad():
                first = model.loc_head.linear_layers[0]
                first.weight.zero_()
                first.bias.zero_()
                first.weight[0, -3, 0] = 1
                model.loc_head.outlayer.weight.zero_()
                model.loc_head.outlayer.bias.zero_()
                model.loc_head.outlayer.weight[0, 0, 0] = 1
                a = stage1_forward(model, xyz, use_extra_features=extra)
                b = stage1_forward(model, xyz + torch.tensor([0.5, 0., 0.]), use_extra_features=extra)
            for key in ("embedding", "log_pmt", "mad", "dim"):
                torch.testing.assert_close(a[key], b[key], atol=1e-5, rtol=1e-5)
            self.assertGreater((a["loc"] - b["loc"]).abs().max().item(), 0.1)

    def legacy_checkpoint(self, model, phase):
        state = {key: value.clone() for key, value in model.state_dict().items()}
        key = "loc_head.linear_layers.0.weight"
        state[key] = state[key][:, :-3, :].clone()
        return {"model": state, "args": {"constraint_route": "direct_mlp_v1", "train_phase": phase}}

    def test_legacy_semantic_migration_preserves_every_existing_weight(self):
        with tempfile.TemporaryDirectory() as directory:
            model = CstPredWrapper("attn_3dgcn")
            checkpoint = self.legacy_checkpoint(model, "semantic")
            source = Path(directory) / "old_semantic.pth"
            torch.save(checkpoint, source)
            target = CstPredWrapper("attn_3dgcn")
            key = "loc_head.linear_layers.0.weight"
            fresh_xyz = target.state_dict()[key][:, -3:, :].clone()
            trainer = CstPredTrainer(target, [], directory, str(Path(directory) / "log.json"),
                                     1, 1e-4, "test", train_phase="geometry", checkpoint_action="init",
                                     checkpoint_source=str(source))
            for name, tensor in checkpoint["model"].items():
                actual = target.state_dict()[name]
                torch.testing.assert_close(actual[:, :-3, :] if name == key else actual, tensor)
            torch.testing.assert_close(target.state_dict()[key][:, -3:, :], fresh_xyz)
            self.assertEqual(len(trainer.optimizer.state), 0)
            self.assertEqual(trainer.start_epoch, 0)
            self.assertEqual(trainer._checkpoint_payload(0)["args"]["loc_input"], LOC_INPUT)
            checkpoint["model"].pop("cls_head.outlayer.weight")
            torch.save(checkpoint, source)
            with self.assertRaisesRegex(RuntimeError, "exact model state"):
                CstPredTrainer(CstPredWrapper("attn_3dgcn"), [], directory,
                               str(Path(directory) / "log.json"), 1, 1e-4, "test",
                               train_phase="geometry", checkpoint_action="init", checkpoint_source=str(source))

    def test_legacy_geometry_cannot_initialize_new_joint(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "old_geometry.pth"
            torch.save(self.legacy_checkpoint(CstPredWrapper("attn_3dgcn"), "geometry"), source)
            with self.assertRaisesRegex(RuntimeError, "restart geometry"):
                CstPredTrainer(CstPredWrapper("attn_3dgcn"), [], directory,
                               str(Path(directory) / "log.json"), 1, 1e-5, "test",
                               train_phase="joint", checkpoint_action="init", checkpoint_source=str(source))


if __name__ == "__main__":
    unittest.main()
