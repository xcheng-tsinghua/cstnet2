import math
import unittest

import torch
import torch.nn.functional as F

from functional.stage1_direct_loss import direct_constraint_loss
from functional.stage1_direct_metrics import Stage1DirectMetricAccumulator
from functional.stage1_direct_trainer import Stage1DirectTrainer
from functional.stage1_metrics import (
    CONSTRAINT_ATTRIBUTE_METRIC_SPECS,
    evaluate_constraint_attribute_metrics,
    primitive_metrics_from_confusion,
)
from networks.stage1_direct_baselines import finalize_direct_constraints


def devices():
    return ['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']


def old_losses(pred, pmt, mad, dim, loc):
    losses = [F.nll_loss(pred['log_pmt'].reshape(-1, 5), pmt.reshape(-1))]
    for name, target, types in [('mad', mad, (0, 1, 2)), ('dim', dim, (1, 2, 3)), ('loc', loc, (0, 1, 2, 3))]:
        mask = sum((pmt == t).int() for t in types).bool()
        if not mask.any():
            losses.append(pred[name].sum() * 0)
        elif name == 'mad':
            a, b = [F.normalize(x[mask], dim=-1, eps=1e-6) for x in (pred[name], target)]
            losses.append(torch.minimum((a-b).square().mean(-1), (a+b).square().mean(-1)).mean())
        else:
            losses.append(F.mse_loss(pred[name][mask], target[mask]))
    return sum(losses)


def reference_attributes(pred, pmt, targets, trim=0.0):
    output = {}
    for name, types, spec in zip(('mad', 'dim', 'loc'), ((0, 1, 2), (1, 2, 3), (0, 1, 2, 3)), CONSTRAINT_ATTRIBUTE_METRIC_SPECS.values()):
        total, count = 0.0, 0
        for a, b, labels in zip(pred[name], targets[name], pmt):
            valid = sum((labels == t).int() for t in types).bool()
            valid &= (torch.isfinite(a) & torch.isfinite(b)) if name == 'dim' else (torch.isfinite(a).all(-1) & torch.isfinite(b).all(-1))
            if name == 'mad':
                valid &= (a.norm(dim=-1) > 1e-6) & (b.norm(dim=-1) > 1e-6)
                a, b = [F.normalize(x[valid].float(), dim=-1, eps=1e-6) for x in (a, b)]
                errors = torch.acos((a*b).sum(-1).abs().clamp(0, 1)) * (180 / math.pi)
            else:
                diff = a[valid].float() - b[valid].float()
                errors = diff.abs() if name == 'dim' else diff.norm(dim=-1)
            kept = errors.numel() - math.floor(errors.numel() * trim)
            total += errors.sort().values[:kept].sum().item()
            count += kept
        output[spec[0]], output[spec[1]] = total, count
    return output


class DirectBaselineOptimizationTest(unittest.TestCase):
    def test_loss_values_and_gradients(self):
        for device in devices():
            for labels in (torch.arange(35).reshape(5, 7) % 5, torch.full((2, 7), 4), torch.zeros(2, 7, dtype=torch.long)):
                with self.subTest(device=device, labels=labels.unique().tolist()):
                    torch.manual_seed(42)
                    pmt = labels.to(device)
                    shape = pmt.shape
                    pred = {name: torch.randn((*shape, width) if width else shape, device=device, dtype=torch.float64, requires_grad=True) for name, width in [('log_pmt', 5), ('mad', 3), ('dim', 0), ('loc', 3)]}
                    targets = [torch.randn_like(pred[name]) for name in ('mad', 'dim', 'loc')]
                    # Undefined labels may contain NaN and must not affect loss or gradients.
                    for target, types in zip(targets, ((0, 1, 2), (1, 2, 3), (0, 1, 2, 3))):
                        mask = sum((pmt == t).int() for t in types).bool()
                        target[~mask] = float('nan')
                    old = old_losses(pred, pmt, *targets)
                    new, _ = direct_constraint_loss(pred, pmt, *targets)
                    torch.testing.assert_close(new, old)
                    old_grad = torch.autograd.grad(old, tuple(pred.values()), retain_graph=True)
                    new_grad = torch.autograd.grad(new, tuple(pred.values()))
                    for actual, expected in zip(new_grad, old_grad):
                        torch.testing.assert_close(actual, expected)

    def test_metrics_match_reference_with_uneven_batches(self):
        for device in devices():
            torch.manual_seed(12)
            accumulator = Stage1DirectMetricAccumulator()
            confusion = torch.zeros(5, 5, device=device)
            sums = [{}, {}]
            for batch_size in (3, 1):
                pmt = torch.randint(5, (batch_size, 23), device=device)
                pred = {name: torch.randn((batch_size, 23, width) if width else (batch_size, 23), device=device) for name, width in [('log_pmt', 5), ('mad', 3), ('dim', 0), ('loc', 3)]}
                targets = {name: torch.randn_like(pred[name]) for name in ('mad', 'dim', 'loc')}
                targets['mad'][:, 0] = 0
                targets['dim'][:, 1] = float('nan')
                targets['loc'][:, 2] = float('inf')
                accumulator.update(pred, pmt, *targets.values())
                final = finalize_direct_constraints(pred)
                for index, values in enumerate((pred, dict(mad=final['direction'], dim=final['dimension'], loc=final['location']))):
                    for trim in (0.0, 0.2):
                        reference = reference_attributes(values, pmt, targets, trim)
                        actual = evaluate_constraint_attribute_metrics(values['mad'], values['dim'], values['loc'], pmt, *targets.values(), trim_ratio=trim)
                        for key in reference:
                            torch.testing.assert_close(actual[key].cpu(), torch.tensor(reference[key], dtype=actual[key].dtype), rtol=2e-5, atol=1e-4)
                        if trim == 0:
                            for key, value in reference.items():
                                sums[index][key] = sums[index].get(key, 0) + value
                confusion += torch.bincount((pmt * 5 + pred['log_pmt'].argmax(-1)).reshape(-1), minlength=25).reshape(5, 5)
                self.assertEqual(accumulator.confusion.device.type, device)
                self.assertFalse(any(value.requires_grad for value in accumulator.raw_attributes.values()))
                accumulator.compute()  # Must not reset or move accumulated tensors.
            summary = accumulator.compute()
            for key, value in primitive_metrics_from_confusion(confusion).items():
                torch.testing.assert_close(torch.tensor(summary[key]), value.cpu())
            for index, prefix in enumerate(('', 'final/')):
                for key, (sum_key, count_key, _) in CONSTRAINT_ATTRIBUTE_METRIC_SPECS.items():
                    expected = sums[index][sum_key] / max(sums[index][count_key], 1)
                    self.assertAlmostEqual(summary[prefix + key], expected, places=4)

    def test_finite_checks_keep_field_names(self):
        for device in devices():
            pred = {name: torch.zeros(1, device=device) for name in ('log_pmt', 'mad', 'dim', 'loc')}
            pred['dim'].fill_(float('inf'))
            pred['loc'].fill_(float('nan'))
            with self.assertRaisesRegex(FloatingPointError, "dim.*loc"):
                Stage1DirectTrainer._assert_finite_predictions(pred)
        self.assertEqual(Stage1DirectMetricAccumulator().compute()['pmt_acc'], 0)


if __name__ == '__main__':
    unittest.main()
