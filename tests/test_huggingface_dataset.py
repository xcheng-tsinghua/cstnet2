import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from data_utils.huggingface_dataset import parse_dataset_url, resolve_stage1_data_root


URL = "https://huggingface.co/datasets/ZXCCHENGXI/cstnet2_s1_small_v2/tree/main"


class HuggingFaceDatasetTest(unittest.TestCase):
    def test_repository_revision_and_subdirectory(self):
        expected = ("ZXCCHENGXI/cstnet2_s1_small_v2", "main", "")
        self.assertEqual(parse_dataset_url(URL), expected)
        self.assertEqual(parse_dataset_url(URL.split("/tree/")[0]), expected)
        self.assertEqual(parse_dataset_url(URL + "/?x=1#files"), expected)
        self.assertEqual(parse_dataset_url(URL.replace("main", "v2") + "/train/h5"),
                         (expected[0], "v2", "train/h5"))
        self.assertEqual(parse_dataset_url(URL.replace("main", "refs%2Fpr%2F1"))[1], "refs/pr/1")

    def test_local_paths_need_no_hub_dependency(self):
        with patch.dict("sys.modules", {"huggingface_hub": None}):
            for path in ("dataset", "D:\\data\\train", "/data/train", "shard.h5"):
                self.assertEqual(resolve_stage1_data_root(path), path)
            with self.assertRaisesRegex(ImportError, "pip install huggingface_hub"):
                resolve_stage1_data_root(URL)

    def test_invalid_urls_fail_before_download(self):
        for url in ("https://example.com/datasets/a/b", "https://huggingface.co/a/b",
                    URL.replace("/tree/", "/blob/"), URL + "/%2e%2e",
                    URL + "/C%3A", URL + "/%2Ftmp", URL.replace("/main", "")):
            with self.subTest(url=url), self.assertRaises(ValueError):
                parse_dataset_url(url)

    def test_three_phases_use_same_snapshot_cache_arguments(self):
        with tempfile.TemporaryDirectory(dir=".") as temporary:
            snapshot = Path(temporary) / "snapshot"
            snapshot.mkdir()
            (snapshot / "data.h5").touch()
            download = Mock(return_value=str(snapshot))
            with patch.dict("sys.modules", {"huggingface_hub": SimpleNamespace(snapshot_download=download)}):
                roots = [resolve_stage1_data_root(URL, cache_dir=temporary) for _ in range(3)]
            self.assertEqual(roots, [str(snapshot)] * 3)
            self.assertEqual(download.call_args_list[0], download.call_args_list[2])
            kwargs = download.call_args.kwargs
            self.assertEqual(kwargs["repo_type"], "dataset")
            self.assertEqual(kwargs["revision"], "main")
            self.assertEqual(kwargs["repo_id"], "ZXCCHENGXI/cstnet2_s1_small_v2")
            self.assertEqual(Path(kwargs["cache_dir"]), Path(temporary))

    def test_subfolder_and_format_filters_and_empty_download(self):
        with tempfile.TemporaryDirectory(dir=".") as temporary:
            root = Path(temporary)
            (root / "train").mkdir()
            (root / "train" / "sample.txt").touch()
            download = Mock(return_value=str(root))
            with patch.dict("sys.modules", {"huggingface_hub": SimpleNamespace(snapshot_download=download)}):
                resolved = resolve_stage1_data_root(URL + "/train", storage_format="txt")
                self.assertEqual(Path(resolved), root / "train")
                self.assertEqual(download.call_args.kwargs["allow_patterns"], ["train/*.[tT][xX][tT]"])
                with self.assertRaisesRegex(FileNotFoundError, "No Stage 1"):
                    resolve_stage1_data_root(URL, storage_format="h5")

    def test_download_failure_does_not_use_partial_snapshot(self):
        download = Mock(side_effect=ConnectionError("interrupted download"))
        with patch.dict("sys.modules", {"huggingface_hub": SimpleNamespace(snapshot_download=download)}):
            with self.assertRaises(ConnectionError):
                resolve_stage1_data_root(URL)

    def test_training_entry_downloads_before_constructing_local_dataset(self):
        import train_cst_pred
        with tempfile.TemporaryDirectory(dir=".") as temporary:
            root = Path(temporary)
            sample = np.zeros((24, 12), dtype=np.float32)
            sample[:, 3] = np.arange(24) % 5
            sample[:, 11] = np.arange(24) // 4
            np.savetxt(root / "sample.txt", sample)
            args = train_cst_pred.parse_args([
                "--data_root", URL, "--model", "pointnet", "--n_points", "24",
                "--workers", "0", "--checkpoint_root", str(root / "weights"),
            ])
            download = Mock(return_value=str(root))
            run = Mock(id="test")
            with patch.dict("sys.modules", {"huggingface_hub": SimpleNamespace(snapshot_download=download)}), \
                    patch("train_cst_pred.initialize_wandb_run", return_value=run), \
                    patch("train_cst_pred.CstPredTrainer") as trainer:
                train_cst_pred.main(args)
            kwargs = trainer.call_args.kwargs
            self.assertEqual(len(kwargs["train_loader"].dataset), 1)
            self.assertEqual(kwargs["checkpoint_args"]["data_root"], URL)
            self.assertEqual(kwargs["checkpoint_args"]["resolved_data_root"], str(root))
            trainer.return_value.start.assert_called_once()

    def test_baseline_entry_accepts_url_and_local_path(self):
        import train_stage1_direct_baseline as baseline
        # Initialize PyTorch's lazy optimizer imports before sys.modules is mocked.
        baseline.torch.optim.Adam(baseline.torch.nn.Linear(1, 1).parameters())
        with tempfile.TemporaryDirectory(dir=".") as temporary:
            root = Path(temporary)
            sample = np.zeros((24, 12), dtype=np.float32)
            sample[:, 3] = np.arange(24) % 5
            sample[:, 11] = np.arange(24) // 4
            np.savetxt(root / "sample.txt", sample)
            for source in (URL, str(root)):
                with self.subTest(source=source):
                    args = baseline.parse_args([
                        "--data_root", source, "--model", "pointnet", "--n_points", "24",
                        "--workers", "0", "--device", "cpu", "--data_format", "txt",
                        "--hf_cache_dir", str(root / "hf_cache"),
                        "--output_root", str(root / "weights"),
                    ])
                    download = Mock(return_value=str(root))
                    run = Mock(id="baseline-test")
                    with patch.dict("sys.modules", {"huggingface_hub": SimpleNamespace(snapshot_download=download)}), \
                            patch.object(baseline, "initialize_wandb_run", return_value=run), \
                            patch.object(baseline, "Stage1DirectTrainer") as trainer:
                        baseline.main(args)
                    kwargs = trainer.call_args.kwargs
                    self.assertEqual(len(kwargs["train_loader"].dataset), 1)
                    self.assertEqual(kwargs["checkpoint_args"]["data_root"], source)
                    self.assertEqual(kwargs["checkpoint_args"]["resolved_data_root"], str(root))
                    trainer.return_value.fit.assert_called_once_with(resume_checkpoint=None)
                    run.finish.assert_called_once()
                    if source == URL:
                        self.assertEqual(download.call_args.kwargs["cache_dir"], str(root / "hf_cache"))
                        self.assertEqual(download.call_args.kwargs["allow_patterns"], ["*.[tT][xX][tT]"])
                    else:
                        download.assert_not_called()


if __name__ == "__main__":
    unittest.main()
