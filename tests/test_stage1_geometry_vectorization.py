"""Value and gradient parity against the original Stage 1 geometry objective."""
import unittest
from unittest import mock

import torch
import torch.nn.functional as F

from functional import loss as optimized
from functional.instance_groups import InstanceGroups
import reference_stage1_geometry as reference


def fixture(mode="mixed", device="cpu"):
    torch.manual_seed(91)
    b, n = 3, 40
    labels = (torch.arange(n, device=device) // 4 * 17 - 51).expand(b, -1).clone()
    # Noncontiguous labels, singleton, and the same ids in different clouds.
    labels[:, -1] = 99999
    primitive = (torch.arange(n, device=device) // 4 % 5).expand(b, -1).clone()
    if mode == "mixed":
        primitive[:, :4] = torch.tensor([1, 0, 1, 0], device=device)  # majority tie
    elif mode == "singletons":
        labels = torch.arange(n, device=device).expand(b, -1)
    elif mode == "other":
        primitive.fill_(4)
    else:
        primitive.fill_(int(mode))
    xyz = torch.randn(b, n, 3, device=device, dtype=torch.float64)
    targets = (F.normalize(torch.randn_like(xyz), dim=-1),
               torch.rand(b, n, device=device, dtype=torch.float64) * 2,
               torch.randn_like(xyz))
    predictions = [torch.randn(b, n, 5, device=device, dtype=torch.float64).log_softmax(-1),
                   torch.randn_like(xyz), torch.randn_like(targets[1]), torch.randn_like(xyz)]
    return xyz, labels, primitive, targets, predictions


def objective(module, xyz, labels, primitive, target, predictions):
    log_pmt, mad, dim, loc = predictions
    weights = module._parameter_observability_weights(xyz, primitive, target[0], target[1], target[2], labels)
    values = [module._robust_dimension_loss(dim, target[1], primitive, labels, weights[0]),
              module._robust_location_loss(loc, target[2], primitive, labels, weights[1]),
              module._masked_vector_mse(mad, target[0], primitive < 3, sign_invariant=True)]
    values.extend(module._stage1_geometry_losses(xyz, mad, dim, loc, primitive, labels, *weights).values())
    values.append(module.instance_consistency_loss(log_pmt, mad, dim, loc, labels, primitive))
    return torch.stack(values)


class GeometryVectorizationTest(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_float32_values_and_gradients(self):
        xyz, labels, primitive, target, initial = fixture(device="cuda")
        xyz = xyz.float()
        target = [x.float() for x in target]
        old = [x.float().requires_grad_() for x in initial]
        new = [x.detach().clone().requires_grad_() for x in old]
        a = objective(reference, xyz, labels, primitive, target, old)
        b = objective(optimized, xyz, labels, primitive, target, new)
        torch.testing.assert_close(a, b, atol=3e-6, rtol=3e-5)
        a.sum().backward()
        b.sum().backward()
        for x, y in zip(old, new):
            torch.testing.assert_close(x.grad, y.grad, atol=3e-6, rtol=3e-5)

    def test_complete_geometry_values_and_gradients(self):
        for mode in ("mixed", "singletons", "other", "0", "1", "2", "3"):
            with self.subTest(mode=mode):
                xyz, labels, primitive, target, initial = fixture(mode)
                old = [x.clone().requires_grad_() for x in initial]
                new = [x.clone().requires_grad_() for x in initial]
                expected = objective(reference, xyz, labels, primitive, target, old)
                actual = objective(optimized, xyz, labels, primitive, target, new)
                torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-6)
                expected.sum().backward()
                actual.sum().backward()
                for a, b in zip(old, new):
                    # Inactive components may now have an explicit zero gradient.
                    ga = torch.zeros_like(a) if a.grad is None else a.grad
                    gb = torch.zeros_like(b) if b.grad is None else b.grad
                    torch.testing.assert_close(ga, gb, atol=2e-7, rtol=2e-6)

    def test_consistency_without_primitive_labels(self):
        _, labels, _, _, initial = fixture()
        old = [x.clone().requires_grad_() for x in initial]
        new = [x.clone().requires_grad_() for x in initial]
        a = reference.instance_consistency_loss(*old, labels)
        b = optimized.instance_consistency_loss(*new, labels)
        torch.testing.assert_close(a, b)
        a.backward()
        b.backward()
        for x, y in zip(old, new):
            torch.testing.assert_close(x.grad, y.grad)

    def test_balanced_reduction_mask_and_weights(self):
        _, labels, _, _, _ = fixture()
        for empty in (False, True):
            values = torch.randn_like(labels, dtype=torch.float64, requires_grad=True)
            other = values.detach().clone().requires_grad_()
            mask = torch.zeros_like(labels, dtype=torch.bool) if empty else torch.rand_like(values) > 0.4
            weights = torch.rand_like(values)
            a = reference._instance_balanced_mean(values, mask, labels, values, weights)
            b = optimized._instance_balanced_mean(other, mask, InstanceGroups(labels), other, weights)
            torch.testing.assert_close(a, b)
            a.backward()
            b.backward()
            torch.testing.assert_close(values.grad, other.grad)

    def test_geometry_training_builds_only_one_grouping(self):
        xyz, labels, primitive, target, initial = fixture()
        predictions = [x.clone().requires_grad_() for x in initial]
        with mock.patch("functional.instance_groups.torch.unique", wraps=torch.unique) as unique:
            value, _ = optimized.constraint_loss(
                xyz, *predictions, primitive, *target, labels,
                global_epoch=10, geom_start_epoch=0, geom_ramp_epochs=1,
                enabled_losses={"pmt": False, "cluster": False},
            )
            value.backward()
        self.assertEqual(unique.call_count, 1)


if __name__ == "__main__":
    unittest.main()
