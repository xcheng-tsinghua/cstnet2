from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from functional.stage2_ablation_config import COMPONENTS, COMPONENT_SLICES, EXPERIMENTS, expand_experiments
from functional import stage2_ablation_runner as runner
from networks.stage2 import CstNetStage2Classifier
from networks.stage2_ablation import Stage2AblationModel, intervene_inputs
from train_cls_ablation import parse_args


def small_model(task, count, experiment, model_name="constraint_aware", config=None):
    assert task == "cls"
    model = CstNetStage2Classifier(count, feature_dim=8, latent_dim=16, token_dim=16,
        transformer_heads=4, transformer_layers=1, dropout=0, stream_dropout=0,
        use_stats_token=True)
    return Stage2AblationModel(model, experiment)


def write_data(root, flat=False):
    rng = np.random.default_rng(12)
    manifest = {"train": [], "test": []}
    for split in ("train", "test"):
        for index in range(6):
            path = root if flat else root / split
            path /= f"class_{index % 2}"
            path.mkdir(parents=True, exist_ok=True)
            array = np.zeros((10, 12), dtype=np.float32)
            array[:, :3] = rng.normal(size=(10, 3))
            array[:, 3] = np.arange(10) % 5
            array[:, 4:7] = (0, 0, 1)
            array[:, 7] = 0.2
            array[:, 8:11] = (0.1, 0.2, 0.3)
            array[:, 11] = np.arange(10) // 2
            file = path / f"{split}_{index}.txt"
            np.savetxt(file, array)
            manifest[split].append(file.relative_to(root).as_posix())
    return manifest


class Stage2AblationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        self.wandb_runs = []
        def create_run(**kwargs):
            run = mock.Mock()
            run.id = kwargs.get("run_id") or f"test-run-{len(self.wandb_runs)}"
            self.wandb_runs.append(run)
            return run
        patcher = mock.patch.object(runner, "initialize_wandb_run", side_effect=create_run)
        self.wandb_init = patcher.start()
        self.addCleanup(patcher.stop)
        patcher = mock.patch.object(runner, "wandb_confusion_matrix", return_value="confusion-chart")
        self.confusion = patcher.start()
        self.addCleanup(patcher.stop)

    def test_training_defaults_match_train_cls(self):
        import train_cls
        reference = train_cls.parse_args([])
        args = parse_args([])
        for here, there in (("batch_size", "bs"), ("epochs", "epoch"),
                            ("learning_rate", "lr"), ("weight_decay", "decay_rate")):
            self.assertEqual(getattr(args, here), getattr(reference, there))
        for key in ("wandb_entity", "stage2_norm", "token_dim",
                    "transformer_layers", "transformer_heads", "token_dropout",
                    "stream_dropout", "use_stats_token", "root_local", "root_sever"):
            self.assertEqual(getattr(args, key), getattr(reference, key))
        aliases = parse_args(["--bs", "3", "--epoch", "2", "--lr", "0.002", "--decay_rate", "0.003"])
        self.assertEqual((aliases.batch_size, aliases.epochs, aliases.learning_rate, aliases.weight_decay),
                         (3, 2, 0.002, 0.003))

    def test_cli_accepts_exactly_nine_experiments_and_one_seed(self):
        expected = {"xyz_only", *(f"no_{c}" for c in COMPONENTS),
                    *(f"only_{c}" for c in COMPONENTS)}
        self.assertEqual(set(EXPERIMENTS), expected)
        self.assertEqual(len(expand_experiments(["all", "xyz_only"])), 9)
        self.assertEqual(parse_args(["--dry_run", "--seed", "17"]).seed, 17)
        self.assertEqual(parse_args(["--dry_run"]).seed, 42)
        self.assertFalse(hasattr(parse_args(["--dry_run"]), "seeds"))
        for options in (["--experiments", "full"], ["--experiments", "mean_fusion"],
                        ["--seeds", "1", "2"], ["--seed", "-1"],
                        ["--task", "seg"], ["--val_ratio", "0.2"], ["--test_ratio", "0.2"]):
            with self.subTest(options=options), self.assertRaises(SystemExit):
                parse_args(["--dry_run", *options])

    def test_entry_runs_each_experiment_once_with_requested_seed(self):
        from train_cls_ablation import main
        with mock.patch.object(runner, "run_experiment") as run, mock.patch.object(runner, "summarize") as summary:
            main(["--data_root", "unused", "--experiments", "all", "--seed", "17"])
        self.assertEqual(run.call_count, 9)
        self.assertEqual({call.args[1] for call in run.call_args_list}, set(EXPERIMENTS))
        self.assertTrue(all(call.args[2] == 17 for call in run.call_args_list))
        summary.assert_called_once_with("model_trained/stage2_ablation", 17)

    def test_removed_values_and_gradients_are_zero(self):
        constraints = torch.randn(2, 8, 12, requires_grad=True)
        original = constraints.detach().clone()
        for component, (start, stop) in zip(COMPONENTS, COMPONENT_SLICES):
            keep = tuple(c for c in COMPONENTS if c != component)
            out = intervene_inputs(constraints, keep)
            self.assertEqual(out[..., start:stop].abs().sum(), 0)
            grad = torch.autograd.grad(out.sum(), constraints)[0]
            self.assertEqual(grad[..., start:stop].abs().sum(), 0)
        torch.testing.assert_close(constraints, original)

    def test_all_nine_forward_backward_and_no_input_leaks(self):
        xyz, constraints = torch.randn(2, 8, 3), torch.randn(2, 8, 12)
        for experiment, spec in EXPERIMENTS.items():
            with self.subTest(experiment=experiment):
                model = small_model("cls", 3, experiment).eval()
                changed = constraints.clone()
                for name, (start, stop) in zip(COMPONENTS, COMPONENT_SLICES):
                    if name not in spec.components:
                        changed[..., start:stop] += 100
                torch.testing.assert_close(model(xyz, constraints), model(xyz, changed), rtol=0, atol=0)
                model.train()
                out = model(xyz, constraints)
                self.assertEqual(tuple(out.shape), (2, 3))
                out.square().mean().backward()
                self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_existing_directories_use_all_train_and_test_samples(self):
        from data_utils.classification_dataset import Stage2ClassificationDataset
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_data(root / "data")
            args = self._args(root)
            factory = Stage2ClassificationDataset.create_dataloaders
            with mock.patch.object(Stage2ClassificationDataset, "create_dataloaders", wraps=factory) as create:
                train, test, count, metadata = runner.build_datasets(args, 0)
            create.assert_called_once()
            self.assertEqual((len(train), len(test), count), (6, 6, 2))
            self.assertEqual(train.indices, list(range(6)))
            self.assertEqual(test.indices, list(range(6)))
            self.assertNotIn("val_indices", metadata)
            self.assertFalse((root / "data/split_file.json").exists())
            before = test[0][0].copy()
            np.random.rand(100)
            np.testing.assert_array_equal(before, test[0][0])
            self.assertEqual(sum(len(batch[1]) for batch in runner.make_loader(train, args, True)), 6)

    def test_existing_manifest_is_preserved_and_missing_split_is_rejected(self):
        from data_utils.classification_dataset import Stage2ClassificationDataset
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = write_data(root / "data", flat=True)
            args = self._args(root)
            with self.assertRaisesRegex(FileNotFoundError, "do not create splits"):
                runner.build_datasets(args, 0)
            split = root / "data/split_file.json"
            self.assertFalse(split.exists())
            # Deliberately asymmetric split; the ablation must not rebalance it.
            manifest["test"] += manifest["train"][2:]
            manifest["train"] = manifest["train"][:2]
            runner.write_json(split, manifest)
            original = split.read_bytes()
            train, test, _, _ = runner.build_datasets(args, 0)
            reference_train, reference_test = Stage2ClassificationDataset.create_dataloaders(
                args.data_root, args.batch_size, args.n_points, 0)
            self.assertEqual(train.dataset.datapath, reference_train.dataset.datapath)
            self.assertEqual(test.dataset.datapath, reference_test.dataset.datapath)
            self.assertEqual((len(train), len(test)), (2, 10))
            self.assertEqual(split.read_bytes(), original)

    def test_hdf5_existing_split_preserved(self):
        from data_utils.stage2_h5 import convert_classification_txt_to_h5
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_data(root / "data")
            convert_classification_txt_to_h5(root / "data", root / "h5")
            args = self._args(root)
            args.data_root = str(root / "h5")
            train, test, _, metadata = runner.build_datasets(args, 0)
            self.assertEqual((len(train), len(test)), (6, 6))
            self.assertTrue(metadata["train_source"]["storage_sha256"])
            next(iter(runner.make_loader(train, args, True)))
            train.dataset.close()
            test.dataset.close()

    def test_summary_lists_single_seed_results_without_averaging(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for index, (seed, score, lr) in enumerate(((0, 0.7, 0.001), (0, 0.9, 0.01), (1, 0.5, 0.001))):
                path = root / str(index)
                path.mkdir()
                runner.write_json(path / "result.json", {
                    "task": "cls", "model": "constraint_aware", "constraint_source": "predicted",
                    "experiment": "xyz_only", "seed": seed, "best_epoch": 1,
                    "test": {"instance_accuracy": score}, "parameters": {"total": 10},
                    "protocol": {"args": {"learning_rate": lr}, "seed": seed,
                                 "experiment": "xyz_only", "intervention": {}}})
            rows = runner.summarize(root, 0)
            self.assertEqual(len(rows), 2)
            self.assertEqual({row["instance_accuracy"] for row in rows}, {0.7, 0.9})
            self.assertEqual(len({row["protocol_id"] for row in rows}), 2)
            self.assertTrue(all(row["seed"] == 0 for row in rows))
            self.assertTrue(all("instance_accuracy_mean" not in row for row in rows))

    def _args(self, root, output="out", epochs=1):
        return parse_args(["--data_root", str(root / "data"),
            "--output_dir", str(root / output), "--device", "cpu", "--workers", "0",
            "--epochs", str(epochs), "--seed", "0", "--batch_size", "2", "--n_points", "8"])

    def test_end_to_end_classification_and_test_checkpoint_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_data(root / "data")
            args = self._args(root)
            with mock.patch.object(runner, "build_ablation_model", side_effect=small_model):
                result = runner.run_experiment(args, "xyz_only", 0)
                self.assertEqual(result["best_epoch"], 1)
                self.assertNotIn("val", result)
                best = runner.load_checkpoint(runner.run_directory(args, "xyz_only", 0) / "best.pth")
                self.assertEqual(best["best"], result["test"]["instance_accuracy"])
                self.assertEqual(best["wandb_run_id"], self.wandb_runs[0].id)
                self.wandb_runs[0].finish.assert_called_once()
                log = self.wandb_runs[0].log.call_args.args[0]
                for key in ("loss/train", "loss/test", "learning_rate", "best/test_instance_accuracy",
                            "train/metric/instance_accuracy", "test/metric/class_accuracy",
                            "train/optimization/gradient_norm_mean", "train/confusion_matrix", "test/confusion_matrix"):
                    self.assertIn(key, log)
                self.assertEqual(self.wandb_runs[0].log.call_args.kwargs["step"], 0)
                args.mode = "evaluate"
                args.data_root = None
                evaluated = runner.run_experiment(args, "xyz_only", 0)
                self.assertEqual(result["test"], evaluated)
                self.assertEqual(self.wandb_init.call_count, 1)
            self.assertEqual(runner.summarize(args.output_dir, 0)[0]["seed"], 0)

    def test_non_resume_overwrites_results_and_starts_a_new_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_data(root / "data")
            args = self._args(root)
            directory = runner.run_directory(args, "xyz_only", 0)
            with mock.patch.object(runner, "build_ablation_model", side_effect=small_model):
                runner.run_experiment(args, "xyz_only", 0)
                old_id = self.wandb_runs[-1].id
                (directory / "last.pth").write_bytes(b"old checkpoint must not be loaded")
                (directory / "epoch_0999.json").write_text("old epoch")
                (directory / "evaluation_test.json").write_text("old evaluation")
                (directory / "notes.txt").write_text("keep")
                args.learning_rate *= 2
                runner.run_experiment(args, "xyz_only", 0)
            checkpoint = runner.load_checkpoint(directory / "last.pth")
            self.assertEqual(checkpoint["epoch"], 0)
            self.assertEqual(checkpoint["protocol"]["args"]["learning_rate"], args.learning_rate)
            self.assertNotEqual(checkpoint["wandb_run_id"], old_id)
            self.assertEqual(self.wandb_init.call_args.kwargs["run_id"], "")
            self.assertEqual(self.wandb_runs[-1].log.call_args.kwargs["step"], 0)
            self.assertFalse((directory / "epoch_0999.json").exists())
            self.assertFalse((directory / "evaluation_test.json").exists())
            self.assertEqual((directory / "notes.txt").read_text(), "keep")
            self.assertTrue((directory / "result.json").is_file())

    def test_experiments_have_independent_wandb_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_data(root / "data")
            args = self._args(root)
            with mock.patch.object(runner, "build_ablation_model", side_effect=small_model):
                for experiment in ("xyz_only", "only_direction"):
                    runner.run_experiment(args, experiment, 0)
            names = [call.kwargs["name"] for call in self.wandb_init.call_args_list]
            self.assertEqual(len(set(names)), 2)
            self.assertTrue(all(call.kwargs["run_id"] == "" for call in self.wandb_init.call_args_list))
            for run in self.wandb_runs:
                run.finish.assert_called_once()

    def test_interrupted_resume_matches_uninterrupted_and_rejects_different_protocol(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_data(root / "data")
            args = self._args(root, "complete", epochs=2)
            with mock.patch.object(runner, "build_ablation_model", side_effect=small_model):
                runner.run_experiment(args, "no_location", 0)
                args.output_dir = str(root / "resumed")
                real_epoch = runner.run_epoch
                calls = 0

                def interrupted(*pos, **kw):
                    nonlocal calls
                    calls += 1
                    if calls == 3:
                        raise RuntimeError("simulated interruption")
                    return real_epoch(*pos, **kw)

                with mock.patch.object(runner, "run_epoch", side_effect=interrupted):
                    with self.assertRaisesRegex(RuntimeError, "simulated"):
                        runner.run_experiment(args, "no_location", 0)
                interrupted_id = self.wandb_runs[-1].id
                self.wandb_runs[-1].finish.assert_called_once()
                args.resume = True
                runner.run_experiment(args, "no_location", 0)
                self.assertEqual(self.wandb_init.call_args.kwargs["run_id"], interrupted_id)
                self.assertEqual(self.wandb_runs[-1].log.call_args.kwargs["step"], 1)
                self.wandb_runs[-1].finish.assert_called_once()
                state1 = runner.load_checkpoint(root / "complete/cls/constraint_aware/predicted/no_location/seed_0/last.pth")
                state2 = runner.load_checkpoint(root / "resumed/cls/constraint_aware/predicted/no_location/seed_0/last.pth")
                for key in state1["model"]:
                    torch.testing.assert_close(state1["model"][key], state2["model"][key], rtol=0, atol=0)
                args.learning_rate *= 2
                with self.assertRaisesRegex(ValueError, "protocol mismatch"):
                    runner.run_experiment(args, "no_location", 0)


if __name__ == "__main__":
    unittest.main()
