import tempfile
import unittest
from pathlib import Path

import torch

from functional.cst_pred_trainer import CstPredTrainer
from functional.point_features import stage1_forward
from networks.cst_pred_wrapper import CstPredWrapper, LOC_INPUT


class LocXYZTest(unittest.TestCase):
    def test_all_attribute_inputs_are_expanded_for_all_backbones(self):
        for backbone in ("pointnet", "pointnet2", "attn_3dgcn"):
            model = CstPredWrapper(backbone)
            self.assertEqual(model.loc_head.linear_layers[0].in_channels, 131)
            self.assertEqual(model.loc_head.linear_layers[0].out_channels, 20)
            self.assertEqual(model.mad_head.linear_layers[0].in_channels, 131)
            self.assertEqual(model.dim_head.linear_layers[0].in_channels, 131)

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
            for key in ("embedding", "log_pmt"):
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

    def test_all_attribute_heads_receive_xyz_and_learn_coordinate_weights(self):
        torch.manual_seed(19)
        model = CstPredWrapper("attn_3dgcn")
        model.set_train_phase("geometry")
        model.eval()
        captured = {}
        handles = []
        for name in ("mad", "dim", "loc"):
            head = getattr(model, f"{name}_head")
            handles.append(head.register_forward_pre_hook(
                lambda module, args, name=name: captured.__setitem__(name, args[0])))
        xyz = torch.randn(2, 64, 3)
        result = model(xyz)
        for handle in handles:
            handle.remove()
        loss = sum((result[name] - torch.randn_like(result[name])).square().mean()
                   for name in ("mad", "dim", "loc"))
        loss.backward()
        for name in ("mad", "dim", "loc"):
            self.assertEqual(captured[name].shape, (2, 131, 64))
            torch.testing.assert_close(captured[name][:, -3:], xyz.transpose(1, 2))
            head = getattr(model, f"{name}_head")
            self.assertGreater(head.linear_layers[0].weight.grad[:, -3:].abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in model.embedding.parameters()))
        self.assertTrue(all(p.grad is None for p in model.emb_head.parameters()))
        self.assertTrue(all(p.grad is None for p in model.cls_head.parameters()))

    def test_both_legacy_semantic_layouts_initialize_new_geometry(self):
        for old_heads in (("mad", "dim", "loc"), ("mad", "dim")):
            with self.subTest(old_heads=old_heads), tempfile.TemporaryDirectory() as directory:
                target = CstPredWrapper("attn_3dgcn")
                old = {key: tensor.clone() for key, tensor in target.state_dict().items()}
                expanded_keys = {f"{head}_head.linear_layers.0.weight" for head in old_heads}
                fresh = {key: target.state_dict()[key][:, -3:].clone() for key in expanded_keys}
                for key in expanded_keys:
                    old[key] = torch.randn_like(old[key][:, :-3])
                source = Path(directory) / "semantic.pth"
                torch.save({"model": old, "args": {
                    "constraint_route": "direct_mlp_v1", "train_phase": "semantic",
                }}, source)
                trainer = CstPredTrainer(target, [], directory, str(Path(directory) / "log.json"),
                                         1, 1e-4, "test", train_phase="geometry", checkpoint_action="init",
                                         checkpoint_source=str(source))
                for key, tensor in old.items():
                    actual = target.state_dict()[key]
                    torch.testing.assert_close(actual[:, :-3] if key in expanded_keys else actual, tensor)
                for key in expanded_keys:
                    torch.testing.assert_close(target.state_dict()[key][:, -3:], fresh[key])
                state = trainer._checkpoint_payload(0)
                self.assertEqual(state["args"]["mad_input"], "backbone_xyz_v1")
                self.assertEqual(state["args"]["dim_input"], "backbone_xyz_v1")
                # A former loc-only run must not resume under the new layout.
                state["checkpoint_config"].pop("mad_input")
                state["checkpoint_config"].pop("dim_input")
                with self.assertRaisesRegex(ValueError, "configuration mismatch"):
                    trainer._validate_resume_config(state)

    def test_loc_only_geometry_checkpoint_is_rejected_for_new_joint(self):
        with tempfile.TemporaryDirectory() as directory:
            old = CstPredWrapper("attn_3dgcn").state_dict()
            for head in ("mad", "dim"):
                key = f"{head}_head.linear_layers.0.weight"
                old[key] = old[key][:, :-3].clone()
            source = Path(directory) / "old_geometry.pth"
            torch.save({"model": old, "args": {
                "constraint_route": "direct_mlp_v1", "train_phase": "geometry",
            }}, source)
            with self.assertRaisesRegex(RuntimeError, "restart geometry"):
                CstPredTrainer(CstPredWrapper("attn_3dgcn"), [], directory,
                               str(Path(directory) / "log.json"), 1, 1e-5, "test",
                               train_phase="joint", checkpoint_action="init", checkpoint_source=str(source))


if __name__ == "__main__":
    unittest.main()
