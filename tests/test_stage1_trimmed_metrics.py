import unittest
from types import SimpleNamespace

import torch

from functional.cst_pred_trainer import _aggregate_metric_dicts
from functional.stage1_direct_metrics import Stage1DirectMetricAccumulator
from functional.stage1_direct_loss import geometry_loss_masks
from functional.stage1_metrics import (
    ATTRIBUTE_TRIM_RATIOS,
    _trimmed_error_sum_and_count,
    evaluate_constraint_attribute_metrics,
)


class TrimmedMetricsTest(unittest.TestCase):
    def test_inrange_uses_joint_gt_masks_and_retained_point_counts(self):
        loader = SimpleNamespace(dataset=SimpleNamespace(loc_abs_limit=3., dim_max=6.))
        accumulator = Stage1DirectMetricAccumulator()
        batches = []
        for size in (1, 2):
            pmt = torch.ones(size, 4, dtype=torch.long)
            direction = torch.zeros(size, 4, 3)
            direction[..., 2] = 1
            dim_gt = torch.tensor([[100., 1., 2., 6.]]) if size == 1 else torch.tensor([[100., 0., 0., 0.]] * 2)
            loc_gt = torch.tensor([[[100., 0., 0.], [3., -3., 3.], [0., 0., 0.], [-4., 0., 0.]]]) if size == 1 else torch.full((2, 4, 3), 100.)
            affiliate = torch.arange(4).expand(size, 4)
            masks = geometry_loss_masks(loader, dim_gt, loc_gt, affiliate)
            pred = {
                "log_pmt": torch.nn.functional.one_hot(pmt, 5).float(),
                "mad": direction, "dim": torch.zeros_like(dim_gt), "loc": torch.zeros_like(loc_gt),
            }
            accumulator.update(pred, pmt, direction, dim_gt, loc_gt, range_masks=masks)
            batches.append(evaluate_constraint_attribute_metrics(
                direction, pred["dim"], pred["loc"], pmt, direction, dim_gt, loc_gt,
                include_trimmed=True, range_masks=masks,
            ))
        own = _aggregate_metric_dicts(batches)
        baseline = accumulator.compute()
        for summary in (own, baseline):
            self.assertAlmostEqual(summary["inrange/location_mean_distance_error"], 27 ** .5 / 2, places=6)
            self.assertAlmostEqual(summary["inrange/dimension_mean_absolute_error"], 1.5)
            self.assertEqual(summary["inrange/direction_mean_angular_error_deg"], 0.)
            self.assertGreater(summary["location_mean_distance_error"], 100.)
            self.assertGreater(summary["trim10p/location_mean_distance_error"], 100.)
            self.assertAlmostEqual(summary["dimension_mean_absolute_error"], 309 / 12)
        self.assertEqual(own["inrange/location_valid_points"], 2)
        self.assertEqual(own["inrange/dimension_valid_points"], 2)
        self.assertEqual(own["inrange/direction_valid_points"], 2)
        self.assertFalse(any(key.startswith("_constraint") for key in own))

    def test_per_cloud_trimming_and_weighted_epoch_means(self):
        # Two clouds have 100 and 20 valid points, plus an entirely invalid cloud.
        # Their error ranges differ, distinguishing per-cloud from global trimming.
        pmt = torch.ones(3, 100, dtype=torch.long)
        pmt[1, 20:] = 4
        pmt[2] = 4
        errors = torch.stack((torch.arange(100.), torch.arange(100.) + 200, torch.zeros(100)))
        mad = torch.zeros(3, 100, 3)
        mad[..., 2] = 1
        loc = torch.zeros_like(mad)
        loc[..., 0] = errors
        predictions = {
            "log_pmt": torch.nn.functional.one_hot(pmt, 5).float(),
            "mad": mad, "dim": errors + 1, "loc": loc,
        }
        dim_gt = torch.ones_like(errors)
        loc_gt = torch.zeros_like(loc)
        accumulator = Stage1DirectMetricAccumulator()
        batches = []
        for index in (slice(0, 2), slice(2, 3)):
            accumulator.update(
                {key: value[index] for key, value in predictions.items()},
                pmt[index], mad[index], dim_gt[index], loc_gt[index],
            )
            batches.append(evaluate_constraint_attribute_metrics(
                mad[index], predictions["dim"][index], loc[index], pmt[index],
                mad[index], dim_gt[index], loc_gt[index], include_trimmed=True,
            ))
        own = _aggregate_metric_dicts(batches)
        baseline = accumulator.compute()
        for section, ratio in ATTRIBUTE_TRIM_RATIOS.items():
            kept = [values[:len(values) - int(len(values) * ratio)]
                    for values in (list(range(100)), list(range(200, 220))) ]
            expected = sum(map(sum, kept)) / sum(map(len, kept))
            for summary in (own, baseline):
                self.assertAlmostEqual(summary[f"{section}/dimension_mean_absolute_error"], expected)
                self.assertAlmostEqual(summary[f"{section}/location_mean_distance_error"], expected)
                self.assertEqual(summary[f"{section}/direction_mean_angular_error_deg"], 0)
            self.assertEqual(own[f"{section}/location_valid_points"], sum(map(len, kept)))
        self.assertFalse(any(key.startswith("_constraint") for key in own))

    def test_extreme_outlier_does_not_erase_retained_errors(self):
        errors = torch.ones(1, 100)
        errors[0, -1] = 1e20
        total, count = _trimmed_error_sum_and_count(errors, torch.ones_like(errors, dtype=torch.bool), 0.01)
        self.assertEqual(total.item(), 99)
        self.assertEqual(count.item(), 99)

    def test_direction_trimming_and_empty_masks(self):
        target = torch.zeros(1, 100, 3)
        target[..., 2] = 1
        prediction = target.clone()
        prediction[0, -10:] = torch.tensor([1., 0., 0.])
        dim = torch.zeros(1, 100)
        pmt = torch.zeros(1, 100, dtype=torch.long)
        summary = _aggregate_metric_dicts([evaluate_constraint_attribute_metrics(
            prediction, dim, target, pmt, target, dim, target, include_trimmed=True,
        )])
        for section, count, remaining in (("trim1p", 99, 9), ("trim5p", 95, 5), ("trim10p", 90, 0)):
            self.assertAlmostEqual(summary[f"{section}/direction_mean_angular_error_deg"], 90 * remaining / count)
            self.assertEqual(summary[f"{section}/dimension_valid_points"], 0)
            self.assertEqual(summary[f"{section}/dimension_mean_absolute_error"], 0)


if __name__ == "__main__":
    unittest.main()
