import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from data_utils.stage1_dataset import Stage1ConstraintDataset
from functional.stage1_direct_loss import direct_constraint_loss, geometry_loss_masks
from functional.stage1_phase_loss import stage1_phase_loss
from functional.stage1_direct_trainer import Stage1DirectTrainer
from functional.cst_pred_trainer import CstPredTrainer
from networks.cst_pred_wrapper import CstPredWrapper


def targets():
    pmt = torch.ones(1, 4, dtype=torch.long)
    mad = torch.tensor([[[0., 0., 1.]] * 4])
    dim = torch.tensor([[100., 2., 0., 6.]])
    loc = torch.tensor([[[0., 0., 0.], [100., 0., 0.], [3., -3., 3.], [-3.1, 0., 0.]]])
    return pmt, mad, dim, loc


def predictions():
    return {
        "log_pmt": torch.zeros(1, 4, 5, requires_grad=True).log_softmax(-1),
        "mad": torch.tensor([[[1., 0., 1.]] * 4], requires_grad=True),
        "dim": torch.ones(1, 4, requires_grad=True),
        "loc": torch.ones(1, 4, 3, requires_grad=True),
        "embedding": torch.randn(1, 4, 3, requires_grad=True),
    }


class RangeMaskTest(unittest.TestCase):
    def test_masked_losses_and_gradients_for_both_routes(self):
        pmt, mad, dim, loc = targets()
        loader = SimpleNamespace(dataset=SimpleNamespace(loc_abs_limit=3., dim_max=6.))
        affiliate = torch.arange(4).view(1, 4)
        masks = geometry_loss_masks(loader, dim, loc, affiliate)
        for mask in masks.values():
            self.assertEqual(mask.tolist(), [[False, False, True, False]])
        for route in ("baseline", "geometry", "joint"):
            with self.subTest(route=route):
                pred = predictions()
                pred["log_pmt"].retain_grad()
                if route == "baseline":
                    loss, logs = direct_constraint_loss(pred, pmt, mad, dim, loc, **masks)
                    dim_loss, loc_loss = logs["dim_loss"], logs["loc_loss"]
                else:
                    loss, logs = stage1_phase_loss(
                        pred, pmt, mad, dim, loc, affiliate,
                        train_phase=route, **masks,
                    )
                    dim_loss, loc_loss = logs["raw/dim"], logs["raw/loc"]
                self.assertAlmostEqual(dim_loss.item(), 1.)
                self.assertAlmostEqual(loc_loss.item(), 8.)
                if route == "joint":
                    _, unmasked = stage1_phase_loss(pred, pmt, mad, dim, loc, affiliate, train_phase=route)
                    torch.testing.assert_close(logs["raw/cluster"], unmasked["raw/cluster"])
                loss.backward()
                for name in ("dim", "loc", "mad"):
                    self.assertTrue(torch.all(pred[name].grad[0, [0, 1, 3]] == 0))
                    self.assertGreater(pred[name].grad[0, 2].abs().sum().item(), 0)
                if route in ("baseline", "joint"):
                    self.assertTrue(torch.all(pred["log_pmt"].grad.abs().sum(-1) > 0))

    def test_instance_mask_propagates_within_cloud_but_not_across_clouds(self):
        loader = SimpleNamespace(dataset=SimpleNamespace(loc_abs_limit=3., dim_max=6.))
        devices = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
        for device in devices:
            dim = torch.ones(2, 6, device=device)
            loc = torch.zeros(2, 6, 3, device=device)
            affiliate = torch.tensor([[7, 7, 8, 8, 2**40, 2**40]] * 2, device=device)
            dim[0, 0] = 7
            loc[0, 3, 0] = 3.1
            dim[1] = 6
            loc[1] = torch.tensor([3., -3., 3.], device=device)
            masks = geometry_loss_masks(loader, dim, loc, affiliate)
            for mask in masks.values():
                self.assertEqual(mask.tolist(), [
                    [False, False, False, False, True, True],
                    [True, True, True, True, True, True],
                ])

    def test_all_excluded_returns_finite_zero_loss_and_zero_gradients(self):
        pmt, mad, dim, loc = targets()
        masks = {name: torch.zeros_like(pmt, dtype=torch.bool)
                 for name in ("mad_valid_mask", "dim_valid_mask", "loc_valid_mask")}
        for route in ("baseline", "geometry"):
            pred = predictions()
            if route == "baseline":
                total, logs = direct_constraint_loss(pred, pmt, mad, dim, loc, **masks)
                self.assertEqual(logs["dim_loss"].item(), 0)
                self.assertEqual(logs["loc_loss"].item(), 0)
            else:
                total, logs = stage1_phase_loss(pred, pmt, mad, dim, loc, None, train_phase=route, **masks)
                self.assertEqual(logs["raw/dim"].item(), 0)
                self.assertEqual(logs["raw/loc"].item(), 0)
            total.backward()
            self.assertTrue(torch.isfinite(total))
            self.assertEqual(pred["dim"].grad.abs().sum().item(), 0)
            self.assertEqual(pred["loc"].grad.abs().sum().item(), 0)
            self.assertEqual(pred["mad"].grad.abs().sum().item(), 0)

    def test_custom_dataset_limits_reach_both_trainers_and_metrics_keep_raw_targets(self):
        with tempfile.TemporaryDirectory(dir=".") as root:
            root = Path(root)
            txt = root / "data"
            txt.mkdir()
            rows = np.zeros((4, 12), dtype=np.float32)
            rows[:, 3] = 1
            rows[:, 6] = 1
            rows[:, 7] = [1, 2, 3, 4]
            rows[:, 8] = [0, 1, 2, 3]
            rows[:, 11] = [0, 1, 1, 2]
            np.savetxt(txt / "sample.txt", rows)
            loader = Stage1ConstraintDataset.create_dataloader(
                txt, 1, 4, 0, False, sample_seed=1, loc_abs_limit=1., dim_max=2.,
            )
            self.assertEqual(loader.dataset.loc_abs_limit, 1.)
            self.assertEqual(loader.dataset.dim_max, 2.)
            # A Subset must use the same dataset policy.
            loader = DataLoader(Subset(loader.dataset, [0]), batch_size=1)
            batch = next(iter(loader))
            self.assertEqual(batch[3].max().item(), 4.)
            self.assertEqual(batch[4].max().item(), 3.)
            pred = predictions()
            pred["dim"] = torch.zeros(1, 4, requires_grad=True)
            pred["loc"] = torch.zeros(1, 4, 3, requires_grad=True)

            model = CstPredWrapper("pointnet")
            own = CstPredTrainer(model, loader, str(root / "own"), str(root / "log.json"),
                                 1, 1e-5, "range_test", train_phase="geometry")
            with patch("functional.cst_pred_trainer.stage1_forward", return_value=pred):
                logs, metrics = own.process_batch(batch, 0, False)
            self.assertAlmostEqual(logs["raw/dim"].item(), 1.)
            self.assertAlmostEqual(logs["raw/loc"].item(), 0.)
            self.assertEqual(metrics["_constraint_attribute_count/location"].item(), 4)
            self.assertEqual(metrics["_constraint_attribute_count/inrange/location"].item(), 1)
            self.assertEqual(metrics["_constraint_attribute_count/inrange/direction"].item(), 1)
            self.assertEqual(metrics["_constraint_attribute_sum/inrange/location_distance_error"].item(), 0.)

            class FixedModel(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.anchor = torch.nn.Parameter(torch.zeros(()))

                def forward(self, xyz):
                    return pred

            fixed = FixedModel()
            baseline = Stage1DirectTrainer(
                model=fixed, optimizer=torch.optim.SGD(fixed.parameters(), lr=.01),
                scheduler=None, train_loader=loader, output_dir=root / "baseline",
                device=torch.device("cpu"), epochs=1, checkpoint_args={},
            )
            with patch.object(baseline, "_backward_and_step"):
                losses, metrics = baseline._run_epoch(0)
            self.assertAlmostEqual(losses["dim_loss"], 1.)
            self.assertAlmostEqual(losses["loc_loss"], 0.)
            self.assertAlmostEqual(metrics["location_mean_distance_error"], 1.5)
            self.assertAlmostEqual(metrics["dimension_mean_absolute_error"], 2.5)
            self.assertAlmostEqual(metrics["inrange/location_mean_distance_error"], 0.)
            self.assertAlmostEqual(metrics["inrange/dimension_mean_absolute_error"], 1.)

    def test_invalid_thresholds_are_rejected(self):
        for kwargs in ({"loc_abs_limit": -1}, {"dim_max": float("nan")}, {"dim_max": float("inf")}):
            with self.assertRaisesRegex(ValueError, "finite and nonnegative"):
                Stage1ConstraintDataset("unused", **kwargs)


if __name__ == "__main__":
    unittest.main()
