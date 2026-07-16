from __future__ import annotations

import unittest

try:
    import torch
except ImportError:  # pragma: no cover - depends on the active training environment
    torch = None


@unittest.skipIf(torch is None, "PyTorch is required for Stage 2 segmentation tests")
class Stage2SegmentationTest(unittest.TestCase):
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

