from __future__ import annotations

import sys
import unittest
from unittest import mock
from pathlib import Path

try:
    import torch
except ImportError:  # pragma: no cover - depends on the active training env
    torch = None


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class Stage2ClassifierShapeTest(unittest.TestCase):
    def test_classification_checkpoint_failures_do_not_raise(self):
        import train_cls

        with mock.patch(
            "train_cls.safe_torch_save", side_effect=[False, False]
        ) as safe_save:
            status = train_cls.save_classification_checkpoints(
                {"weight": torch.ones(1)},
                "last.pth",
                "best.pth",
                save_best=True,
            )

        self.assertEqual(status, {"last": False, "best": False})
        self.assertEqual(safe_save.call_count, 2)

    def setUp(self):
        if torch is None:
            self.skipTest("PyTorch is required for Stage2 classifier shape tests")

        from functional.constraints import CONSTRAINT_DIM

        self.n_classes = 12
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.xyz = torch.randn(2, 1024, 3, device=self.device)
        self.constraints = torch.randn(2, 1024, CONSTRAINT_DIM, device=self.device)

    def _check_log_probs(self, model):
        model = model.to(self.device)
        model.eval()
        with torch.no_grad():
            out = model(self.xyz, self.constraints)
        self.assertEqual(tuple(out.shape), (2, self.n_classes))

    def test_training_cli_selects_models_and_constraints(self):
        import train_cls

        defaults = train_cls.parse_args([])
        self.assertEqual(defaults.model, "constraint_aware")
        self.assertFalse(defaults.baseline_use_constraints)
        self.assertFalse(defaults.resume)
        self.assertFalse(hasattr(defaults, "dgcnn_k"))
        self.assertFalse(hasattr(defaults, "pointmamba_tokens"))
        self.assertFalse(hasattr(defaults, "use_wandb"))
        self.assertFalse(hasattr(defaults, "constraint_source"))
        self.assertFalse(hasattr(defaults, "stage1_ckpt"))
        self.assertFalse(hasattr(defaults, "stage1_model"))
        self.assertFalse(hasattr(defaults, "stage2_variant"))
        self.assertEqual(defaults.wandb_project, "cstnet2")

        args = train_cls.parse_args([
            "--model", "pointnet2",
            "--baseline_use_constraints",
            "--use_stats_token",
            "--is_sample",
            "--local",
            "--resume",
        ])
        self.assertEqual(args.model, "pointnet2")
        self.assertTrue(args.baseline_use_constraints)
        self.assertTrue(args.use_stats_token)
        self.assertTrue(args.is_sample)
        self.assertTrue(args.local)
        self.assertTrue(args.resume)

    def test_classification_constraints_are_read_from_dataset_fields(self):
        import train_cls

        from functional.constraints import ground_truth_constraints_to_tensor

        batch_size, n_points = 2, 8
        data = (
            torch.randn(batch_size, n_points, 3),
            torch.zeros(batch_size, dtype=torch.long),
            torch.randint(0, 5, (batch_size, n_points)),
            torch.randn(batch_size, n_points, 3),
            torch.randn(batch_size, n_points),
            torch.randn(batch_size, n_points, 3),
            torch.randn(batch_size, n_points, 3),
            torch.zeros(batch_size, n_points, dtype=torch.long),
        )
        expected = ground_truth_constraints_to_tensor(
            data[2], data[3], data[4], data[5], data[6]
        )
        actual = train_cls.constraints_from_dataset_batch(
            data, torch.device("cpu")
        )

        self.assertTrue(torch.equal(actual, expected))

    def test_classification_model_config_and_run_names(self):
        from networks.classification_models import (
            classification_model_config,
            classification_model_uses_constraints,
            classification_run_name,
        )

        config = classification_model_config(
            {
                "model": "dgcnn",
                "baseline_use_constraints": True,
                "dgcnn_k": 12,
                "lr": 999,
            }
        )
        self.assertEqual(
            config,
            {
                "model": "dgcnn",
                "baseline_use_constraints": True,
            },
        )
        self.assertTrue(classification_model_uses_constraints(config))
        self.assertEqual(classification_run_name(config), "dgcnn_constraints")

        pointmamba = classification_model_config(
            {
                "model": "pointmamba",
                "pointmamba_tokens": 64,
                "pointmamba_group_size": 16,
            }
        )
        self.assertEqual(
            pointmamba,
            {
                "model": "pointmamba",
                "baseline_use_constraints": False,
            },
        )

        xyz_only = classification_model_config({"model": "pointnet"})
        self.assertFalse(classification_model_uses_constraints(xyz_only))
        self.assertEqual(classification_run_name(xyz_only), "pointnet")

        with self.assertRaisesRegex(ValueError, "only valid for baseline"):
            classification_model_config(
                {
                    "model": "constraint_aware",
                    "baseline_use_constraints": True,
                }
            )

        constraint_aware = classification_model_config(
            {"model": "constraint_aware"}
        )
        self.assertEqual(
            constraint_aware,
            {
                "model": "constraint_aware",
                "stage2_norm": "ln",
                "token_dim": 256,
                "transformer_layers": 3,
                "transformer_heads": 8,
                "token_dropout": 0.1,
                "stream_dropout": 0.1,
                "use_stats_token": False,
            },
        )
        self.assertNotIn("stage2_variant", constraint_aware)
        self.assertEqual(
            classification_run_name(constraint_aware),
            "constraint_aware_token_fusion",
        )

    def test_registered_baseline_classifier_shapes(self):
        from networks.classification_models import build_classification_model

        cases = (
            ({"model": "pointnet"}, 32),
            ({"model": "pointnet2"}, 64),
            ({"model": "dgcnn"}, 24),
            ({"model": "attn3dgcn"}, 24),
            ({"model": "pointtransformer"}, 16),
            ({"model": "pointmamba"}, 16),
            ({"model": "pointnext"}, 16),
            ({"model": "pointmlp"}, 16),
        )
        for base_config, n_points in cases:
            for use_constraints in (False, True):
                config = {
                    **base_config,
                    "baseline_use_constraints": use_constraints,
                }
                with self.subTest(
                    model=config["model"], use_constraints=use_constraints
                ):
                    model = build_classification_model(
                        self.n_classes, config
                    ).to(self.device).eval()
                    xyz = torch.randn(2, n_points, 3, device=self.device)
                    constraints = torch.randn(
                        2, n_points, 15, device=self.device
                    )
                    with torch.no_grad():
                        output = model(xyz, constraints)
                    self.assertEqual(tuple(output.shape), (2, self.n_classes))
                    self.assertTrue(torch.isfinite(output).all())

    def test_new_baselines_support_training_backward(self):
        from networks.classification_models import build_classification_model
        from networks.segmentation_models import build_segmentation_model

        configs = (
            {"model": "pointtransformer"},
            {"model": "pointmamba"},
            {"model": "pointnext"},
            {"model": "pointmlp"},
        )
        xyz = torch.randn(2, 16, 3, device=self.device)
        constraints = torch.randn(2, 16, 15, device=self.device)
        for base_config in configs:
            config = {**base_config, "baseline_use_constraints": True}
            with self.subTest(model=config["model"]):
                classifier = build_classification_model(5, config).to(self.device)
                classifier(xyz, constraints).square().mean().backward()
                segmenter = build_segmentation_model(5, config).to(self.device)
                segmenter(xyz, constraints, None).square().mean().backward()
                for model in (classifier, segmenter):
                    self.assertTrue(
                        all(
                            parameter.grad is None
                            or torch.isfinite(parameter.grad).all()
                            for parameter in model.parameters()
                        )
                    )

    def test_constraint_enabled_baseline_requires_constraints(self):
        from networks.classification_models import build_classification_model

        model = build_classification_model(
            self.n_classes,
            {"model": "pointnet", "baseline_use_constraints": True},
        ).to(self.device).eval()
        with self.assertRaisesRegex(ValueError, "expects constraints"):
            with torch.no_grad():
                model(self.xyz, None)

    def test_constraint_aware_token_fusion_classifier_shapes(self):
        from networks.stage2 import CstNetStage2Classifier

        model = CstNetStage2Classifier(n_classes=self.n_classes)
        self._check_log_probs(model)
        model.eval()
        with torch.no_grad():
            aux = model(self.xyz, self.constraints, return_aux=True)

        self.assertEqual(tuple(aux["log_probs"].shape), (2, self.n_classes))
        self.assertEqual(tuple(aux["main_logits"].shape), (2, self.n_classes))
        self.assertEqual(tuple(aux["aux_final_constraint_logits"].shape), (2, self.n_classes))
        self.assertEqual(tuple(aux["aux_component_token_logits"].shape), (2, self.n_classes))


if __name__ == "__main__":
    unittest.main()
