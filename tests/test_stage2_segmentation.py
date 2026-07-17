from __future__ import annotations

import unittest

try:
    import torch
except ImportError:  # pragma: no cover - depends on the active training environment
    torch = None


@unittest.skipIf(torch is None, "PyTorch is required for Stage 2 segmentation tests")
class Stage2SegmentationTest(unittest.TestCase):
    def test_training_paths_and_boolean_resume(self):
        import tempfile
        from pathlib import Path

        import train_stage2_seg

        default_args = train_stage2_seg.parse_args([])
        self.assertFalse(default_args.resume)
        self.assertFalse(hasattr(default_args, "output_dir"))
        self.assertTrue(train_stage2_seg.parse_args(["--resume"]).resume)
        self.assertTrue(train_stage2_seg.parse_args(["--resume", "true"]).resume)
        self.assertFalse(train_stage2_seg.parse_args(["--resume", "false"]).resume)

        original_root = train_stage2_seg.MODEL_OUTPUT_ROOT
        try:
            with tempfile.TemporaryDirectory() as directory:
                train_stage2_seg.MODEL_OUTPUT_ROOT = Path(directory)
                config = {
                    "model": "pointnet2",
                    "baseline_use_constraints": True,
                }
                output_dir, checkpoint = train_stage2_seg.resolve_training_paths(
                    config, resume=False
                )
                self.assertEqual(output_dir.name, "pointnet2_constraints")
                self.assertIsNone(checkpoint)

                with self.assertRaises(FileNotFoundError):
                    train_stage2_seg.resolve_training_paths(config, resume=True)
                output_dir.mkdir(parents=True)
                expected_checkpoint = output_dir / "last.pth"
                expected_checkpoint.touch()
                _, checkpoint = train_stage2_seg.resolve_training_paths(
                    config, resume=True
                )
                self.assertEqual(checkpoint, expected_checkpoint)
        finally:
            train_stage2_seg.MODEL_OUTPUT_ROOT = original_root

    def test_registered_baseline_models_use_common_output_layout(self):
        from networks.segmentation_models import build_segmentation_model

        cases = (
            ({"model": "pointnet"}, 32),
            ({"model": "pointnet2"}, 64),
            ({"model": "dgcnn", "dgcnn_k": 8}, 24),
            (
                {"model": "attn3dgcn", "attn_neighbors": 8, "attn_k": 4},
                24,
            ),
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
                    model = build_segmentation_model(num_classes=5, config=config).eval()
                    xyz = torch.randn(1, n_points, 3)
                    constraints = torch.randn(1, n_points, 15)
                    masks = torch.ones(1, n_points, 5, dtype=torch.bool)
                    with torch.no_grad():
                        logits = model(xyz, constraints, masks)
                    self.assertEqual(tuple(logits.shape), (1, n_points, 5))
                    self.assertTrue(torch.isfinite(logits).all())
                    restored = build_segmentation_model(num_classes=5, config=config)
                    restored.load_state_dict(model.state_dict(), strict=True)

    def test_constraint_enabled_baseline_requires_constraint_tensor(self):
        from networks.segmentation_models import build_segmentation_model

        model = build_segmentation_model(
            num_classes=5,
            config={"model": "pointnet", "baseline_use_constraints": True},
        ).eval()
        xyz = torch.randn(1, 24, 3)
        with self.assertRaisesRegex(ValueError, "expects constraints"):
            with torch.no_grad():
                model(xyz, None, None)

    def test_model_config_contains_only_architecture_parameters(self):
        from networks.segmentation_models import segmentation_model_config

        self.assertEqual(
            segmentation_model_config(
                {
                    "model": "constraint_aware",
                    "feature_dim": 48,
                    "norm_type": "bn",
                    "dgcnn_k": 99,
                }
            ),
            {"model": "constraint_aware", "feature_dim": 48, "norm_type": "bn"},
        )
        self.assertEqual(
            segmentation_model_config({"model": "dgcnn", "dgcnn_k": 12}),
            {
                "model": "dgcnn",
                "baseline_use_constraints": False,
                "dgcnn_k": 12,
            },
        )
        self.assertEqual(
            segmentation_model_config(
                {
                    "model": "pointnet2",
                    "baseline_use_constraints": True,
                }
            ),
            {"model": "pointnet2", "baseline_use_constraints": True},
        )
        self.assertEqual(
            segmentation_model_config({"feature_dim": 32, "norm_type": "ln"}),
            {"model": "constraint_aware", "feature_dim": 32, "norm_type": "ln"},
        )
        with self.assertRaises(ValueError):
            segmentation_model_config({"model": "missing"})
        with self.assertRaises(ValueError):
            segmentation_model_config(
                {"model": "constraint_aware", "baseline_use_constraints": True}
            )

    def test_resume_rejects_a_different_segmentation_model(self):
        import tempfile
        from pathlib import Path

        from functional.stage2_seg_trainer import Stage2SegmentationTrainer

        label_map = {"labels": [{"id": 0}, {"id": 1}]}
        checkpoint = {
            "epoch": 0,
            "global_step": 1,
            "model": {},
            "optimizer": {},
            "scheduler": {},
            "scaler": None,
            "best_metric": 0.0,
            "args": {"model": "pointnet"},
            "model_config": {"model": "pointnet"},
            "label_map": label_map,
            "class_weights": torch.ones(2),
            "rng_state": {},
        }
        trainer = Stage2SegmentationTrainer.__new__(Stage2SegmentationTrainer)
        trainer.label_map = label_map
        trainer.checkpoint_args = {
            "model": "pointnet",
            "baseline_use_constraints": True,
        }
        trainer.class_weights = torch.ones(2)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "last.pth"
            torch.save(checkpoint, checkpoint_path)
            with self.assertRaisesRegex(ValueError, "model configuration"):
                trainer.load_checkpoint(checkpoint_path)

    def test_nonfinite_gradient_handling_respects_amp(self):
        from functional.stage2_seg_trainer import Stage2SegmentationTrainer

        class FiniteForwardInfiniteBackward(torch.autograd.Function):
            @staticmethod
            def forward(ctx, value):
                return value.new_ones(())

            @staticmethod
            def backward(ctx, grad_output):
                return torch.full_like(grad_output, float("inf"))

        class OverflowScaler:
            def __init__(self):
                self.scale_value = 65536.0
                self.step_called = False

            def scale(self, loss):
                return loss

            def unscale_(self, optimizer):
                return None

            def get_scale(self):
                return self.scale_value

            def step(self, optimizer):
                self.step_called = True

            def update(self):
                self.scale_value *= 0.5

        parameter = torch.nn.Parameter(torch.tensor(1.0))
        trainer = Stage2SegmentationTrainer.__new__(Stage2SegmentationTrainer)
        trainer.model = torch.nn.ParameterList([parameter])
        trainer.optimizer = torch.optim.SGD([parameter], lr=0.1)
        trainer.gradient_clip_norm = 1.0
        trainer.use_amp = True
        trainer.scaler = OverflowScaler()

        loss = FiniteForwardInfiniteBackward.apply(parameter)
        gradient_norm, step_skipped = trainer._backward_and_step(loss)

        self.assertFalse(torch.isfinite(gradient_norm))
        self.assertTrue(step_skipped)
        self.assertTrue(trainer.scaler.step_called)
        self.assertEqual(trainer.scaler.get_scale(), 32768.0)

        parameter.grad = None
        trainer.use_amp = False
        trainer.scaler = OverflowScaler()
        loss = FiniteForwardInfiniteBackward.apply(parameter)
        with self.assertRaises(FloatingPointError):
            trainer._backward_and_step(loss)

    def test_feature_propagation_single_coarse_point(self):
        from networks.feature_propagation import FeaturePropagation

        module = FeaturePropagation(in_channels=6, out_channels=5)
        xyz_fine = torch.randn(2, 7, 3)
        xyz_coarse = torch.randn(2, 1, 3)
        feat_fine = torch.randn(2, 7, 2)
        feat_coarse = torch.randn(2, 1, 4)
        output = module(xyz_fine, xyz_coarse, feat_fine, feat_coarse)
        self.assertEqual(tuple(output.shape), (2, 7, 5))
        self.assertTrue(torch.isfinite(output).all())

    def test_face_metric_uses_mean_logits(self):
        from functional.segmentation_metrics import SegmentationMetrics

        logits = torch.tensor(
            [[[0.6, 0.4], [0.6, 0.4], [-10.0, 10.0]]], dtype=torch.float32
        )
        labels = torch.ones(1, 3, dtype=torch.long)
        face_ids = torch.zeros(1, 3, dtype=torch.long)
        metrics = SegmentationMetrics(num_classes=2)
        metrics.update(logits, labels, face_ids)
        result = metrics.compute()
        self.assertAlmostEqual(result["point_overall_accuracy"], 1.0 / 3.0)
        self.assertAlmostEqual(result["face_overall_accuracy"], 1.0)

    def test_small_segmentation_forward(self):
        from networks.stage2_segmentation import Stage2SegmentationModel

        model = Stage2SegmentationModel(
            num_classes=5,
            feature_dim=8,
            encoder_channels=(16, 32, 64),
            decoder_channels=(32, 24, 16),
            global_context_dim=16,
            n_centers=(16, 8, 4),
            n_neighbors=(8, 4, 4),
        ).eval()
        xyz = torch.randn(1, 24, 3)
        constraints = torch.randn(1, 24, 15)
        masks = torch.ones(1, 24, 5, dtype=torch.bool)
        with torch.no_grad():
            logits = model(xyz, constraints, masks)
        self.assertEqual(tuple(logits.shape), (1, 24, 5))
        self.assertTrue(torch.isfinite(logits).all())


if __name__ == "__main__":
    unittest.main()
