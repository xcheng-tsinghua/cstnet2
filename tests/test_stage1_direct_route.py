import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from functional.direct_constraints import direct_constraints, validate_direct_checkpoint
from functional.cst_pred_trainer import CstPredTrainer
from networks.cst_pred_wrapper import CstPredWrapper
from networks.stage1_extractor import FrozenStage1ConstraintExtractor
from train_cst_pred import parse_args


class DirectRouteTest(unittest.TestCase):
    def test_canonical_components_and_no_embedding_dependency(self):
        output = {
            "log_pmt": torch.eye(5).unsqueeze(0),
            "mad": torch.tensor([[[0., 0., -2.]] * 5]),
            "dim": torch.tensor([[2., 2., 9., 3., 8.]]),
            "loc": torch.tensor([[[1., 2., 3.]] * 5]),
        }
        result = direct_constraints(output)
        self.assertEqual(set(result), {"primitive_type", "direction", "dimension", "location"})
        torch.testing.assert_close(result["primitive_type"], torch.eye(5).unsqueeze(0))
        torch.testing.assert_close(result["direction"][0, :3], torch.tensor([[0., 0., 1.]] * 3))
        torch.testing.assert_close(result["direction"][0, 3:], torch.zeros(2, 3))
        torch.testing.assert_close(result["location"][0, 0], torch.tensor([0., 0., 3.]))
        torch.testing.assert_close(result["location"][0, 1], torch.tensor([1., 2., 0.]))
        torch.testing.assert_close(result["location"][0, 2:4], output["loc"][0, 2:4])
        torch.testing.assert_close(result["location"][0, 4], torch.zeros(3))
        self.assertTrue(0 < result["dimension"][0, 2] < torch.pi / 2)
        self.assertEqual(result["dimension"][0, 0], 0)
        self.assertEqual(result["dimension"][0, 4], 0)

    def test_geometry_updates_only_three_heads_and_frozen_buffers(self):
        with tempfile.TemporaryDirectory(dir=".") as root:
            model = CstPredWrapper("attn_3dgcn")
            xyz = torch.randn(2, 24, 3)
            batch = (xyz, torch.ones(2, 24, dtype=torch.long),
                     torch.nn.functional.normalize(torch.randn_like(xyz), dim=-1),
                     torch.ones(2, 24), torch.randn_like(xyz),
                     torch.zeros(2, 24, dtype=torch.long))
            trainer = CstPredTrainer(model, [batch], root, str(Path(root) / "log.json"),
                                     1, 1e-4, "test", train_phase="geometry")
            model.train()
            model.apply_train_phase_mode()
            before = {k: v.clone() for k, v in model.state_dict().items()}
            with mock.patch("functional.loss.discriminative_loss", side_effect=AssertionError("disabled")):
                trainer.process_batch(batch, 1, True)
            after = model.state_dict()
            for key in before:
                if key.startswith(("embedding.", "emb_head.", "cls_head.")):
                    torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)
            for head in ("mad_head.", "dim_head.", "loc_head."):
                self.assertTrue(any(not torch.equal(before[k], after[k]) for k in before if k.startswith(head)))

    def test_all_backbones_keep_partial_joint_freeze(self):
        for backbone in ("pointnet", "pointnet2", "attn_3dgcn"):
            model = CstPredWrapper(backbone)
            model.set_train_phase("joint")
            parameters = list(model.embedding.parameters())
            self.assertTrue(any(p.requires_grad for p in parameters))
            self.assertTrue(any(not p.requires_grad for p in parameters))

    def test_frozen_inference_has_no_fitting_and_returns_12_channels(self):
        with tempfile.TemporaryDirectory(dir=".") as root:
            path = Path(root) / "joint.pth"
            model = CstPredWrapper("pointnet")
            torch.save({"model": model.state_dict(), "args": {
                "constraint_route": "direct_mlp_v1", "train_phase": "joint"
            }}, path)
            extractor = FrozenStage1ConstraintExtractor("pointnet", str(path))
            with mock.patch("functional.constraints.assemble_constraints_from_stage1", side_effect=AssertionError("removed")):
                result = extractor(torch.randn(2, 24, 3, requires_grad=True))
            self.assertEqual(result.shape, (2, 24, 12))
            self.assertFalse(result.requires_grad)
            self.assertTrue(all(not p.requires_grad for p in extractor.parameters()))

    def test_checkpoint_guard_and_cli(self):
        with self.assertRaisesRegex(ValueError, "direct_mlp_v1"):
            validate_direct_checkpoint({"train_phase": "joint"})
        with self.assertRaisesRegex(ValueError, "geometry or joint"):
            validate_direct_checkpoint({"constraint_route": "direct_mlp_v1", "train_phase": "semantic"})
        args = parse_args([])
        self.assertEqual(args.lr, 1e-4)
        self.assertEqual(parse_args(["--train_phase", "joint"]).lr, 1e-5)
        for name in ("use_amp", "cluster_method", "normal_k", "cluster_metric_interval"):
            self.assertFalse(hasattr(args, name))


if __name__ == "__main__":
    unittest.main()
