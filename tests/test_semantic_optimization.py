import copy
import unittest
from unittest import mock

import torch

from functional.finite_checks import assert_finite_tensors
from functional.neighborhood import PointNeighborhood, build_neighborhood
from functional.point_features import build_stage1_input_features, stage1_forward
from networks.attn_3dgcn import square_distance, get_neighbor_index
from networks.cst_pred_wrapper import CstPredWrapper


class SemanticOptimizationTest(unittest.TestCase):
    def test_one_search_shared_by_features_gcn_and_attention(self):
        model = CstPredWrapper("attn_3dgcn", channel_fea=2).eval()
        xyz = torch.randn(2, 64, 3)
        with mock.patch("functional.neighborhood.torch.cdist", wraps=torch.cdist) as distance:
            result = stage1_forward(model, xyz, use_extra_features=True, feature_k=28)
        self.assertEqual(distance.call_count, 1)
        self.assertEqual(result["log_pmt"].shape, (2, 64, 5))
        self.assertTrue(all(torch.isfinite(value).all() for value in result.values()))

    def compare_legacy(self, device):
        torch.manual_seed(142)
        xyz = torch.randn(2, 64, 3, device=device)
        model = CstPredWrapper("attn_3dgcn", channel_fea=2).to(device).eval()
        old = copy.deepcopy(model)
        # Independent old searches: PCA k=16, GCN k=20 excluding rank zero,
        # attention full argsort including rank zero.
        with torch.no_grad():
            distances = torch.cdist(xyz, xyz)
            values, indices = distances.topk(17, largest=False, sorted=True)
            neighbors = xyz[torch.arange(2, device=device)[:, None, None], indices[..., 1:]]
            centered = neighbors - neighbors.mean(2, keepdim=True)
            cov = centered.transpose(-1, -2) @ centered / 16
            eigenvalues = torch.linalg.eigh(cov)[0].clamp_min(0)
            features = torch.cat((eigenvalues[..., :1] / (eigenvalues.sum(-1, keepdim=True) + 1e-6), values[..., 1:].mean(-1, keepdim=True)), -1)
            gcn_index = get_neighbor_index(xyz, 20)
            self_index = torch.arange(64, device=device).view(1, 64, 1).expand(2, -1, -1)
            old_gcn = PointNeighborhood(torch.cat((self_index, gcn_index), -1), torch.zeros(2, 64, 21, device=device))
            old_attention = PointNeighborhood(square_distance(xyz, xyz).argsort(dim=-1)[..., :16], None)
        original_forward = old.embedding.attention.forward
        with mock.patch.object(old.embedding.attention, "forward", side_effect=lambda x, f, neighborhood=None: original_forward(x, f, old_attention)):
            expected = old(xyz, features, neighborhood=old_gcn)
        actual = stage1_forward(model, xyz, use_extra_features=True)
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], atol=2e-5, rtol=2e-4)
        sum(value.square().mean() for value in expected.values()).backward()
        sum(value.square().mean() for value in actual.values()).backward()
        for (name, a), (_, b) in zip(old.named_parameters(), model.named_parameters()):
            torch.testing.assert_close(a.grad, b.grad, atol=2e-5, rtol=3e-4, msg=name)

    def test_outputs_and_gradients_match_old_search_on_distinct_points(self):
        self.compare_legacy("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_outputs_and_gradients(self):
        self.compare_legacy("cuda")

    def test_duplicate_points_and_small_clouds(self):
        for n in (1, 2, 8):
            xyz = torch.zeros(2, n, 3)
            neighborhood = build_neighborhood(xyz, 29)
            self.assertEqual(neighborhood.indices.shape, (2, n, n))
            torch.testing.assert_close(neighborhood.distances, torch.zeros(2, n, n))
            features = build_stage1_input_features(xyz, neighborhood=neighborhood)
            self.assertTrue(torch.isfinite(features).all())
        xyz = torch.randn(2, 64, 3)
        neighborhood = build_neighborhood(xyz, 29)
        torch.testing.assert_close(neighborhood.indices[..., 0], torch.arange(64).expand(2, -1))
        nearest, _ = neighborhood.excluding_first(20)
        self.assertFalse((nearest == torch.arange(64).view(1, 64, 1)).any())

    def test_finite_checks_retain_field_names(self):
        for device in (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]):
            assert_finite_tensors({}, "empty")
            assert_finite_tensors({"valid": torch.ones(4, device=device), "metadata": 3}, "outputs")
            with self.assertRaisesRegex(FloatingPointError, "non-finite outputs.*bad_nan.*bad_inf"):
                assert_finite_tensors({"valid": torch.ones(1, device=device),
                                      "bad_nan": torch.tensor(float("nan"), device=device),
                                      "bad_inf": torch.tensor(float("inf"), device=device)}, "outputs")


if __name__ == "__main__":
    unittest.main()
