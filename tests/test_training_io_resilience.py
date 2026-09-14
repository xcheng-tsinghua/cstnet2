import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from functional.console_io import ResilientTextStream, resilient_console
from functional.checkpoint_io import safe_json_save, safe_torch_save
from functional.cst_pred_trainer import CstPredTrainer
from networks.cst_pred_wrapper import CstPredWrapper


class BrokenConsole(io.StringIO):
    broken = True

    def write(self, text):
        # Reproduce Colorama's write -> flush -> OSError stack.
        self.flush()
        return super().write(text)

    def flush(self):
        if self.broken:
            raise OSError(5, "Input/output error")
        return super().flush()


class TrainingIOResilienceTest(unittest.TestCase):
    def test_console_failure_and_recovery(self):
        output = BrokenConsole()
        stream = ResilientTextStream(output)
        self.assertEqual(stream.write("lost output"), len("lost output"))
        stream.flush()
        self.assertEqual(stream.failed_operations, 2)
        output.broken = False
        stream.write("next epoch")
        self.assertIn("console recovered after 2", output.getvalue())
        self.assertIn("next epoch", output.getvalue())
        self.assertEqual(stream.pending_failures, 0)

    def test_console_context_does_not_swallow_training_io_errors(self):
        with mock.patch("sys.stdout", BrokenConsole()), mock.patch("sys.stderr", BrokenConsole()):
            with self.assertRaisesRegex(OSError, "dataset read"):
                with resilient_console():
                    print("output failure is harmless")
                    raise OSError(5, "dataset read failed")

    def test_checkpoint_retries_even_when_warning_logger_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "last.pth"
            destination.write_bytes(b"old checkpoint")
            logger = mock.Mock(side_effect=OSError(5, "console failed"))
            with mock.patch("functional.checkpoint_io.os.replace", side_effect=OSError(5, "disk failed")) as replace:
                self.assertFalse(safe_torch_save({"epoch": 24}, destination, retry_seconds=0, logger=logger))
            self.assertEqual(replace.call_count, 3)
            self.assertEqual(destination.read_bytes(), b"old checkpoint")

    def test_json_failure_preserves_previous_file_and_can_recover(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "log.json"
            destination.write_text('{"epoch": 23}', encoding="utf-8")
            with mock.patch("functional.checkpoint_io.os.replace", side_effect=OSError(5, "disk failed")) as replace:
                self.assertFalse(safe_json_save({"epoch": 24}, destination, retry_seconds=0))
            self.assertEqual(replace.call_count, 3)
            self.assertEqual(json.loads(destination.read_text())["epoch"], 23)
            self.assertTrue(safe_json_save({"epoch": 25}, destination))
            self.assertEqual(json.loads(destination.read_text())["epoch"], 25)
            self.assertEqual(list(Path(directory).glob("*.tmp.*")), [])

    def test_json_programming_error_is_not_suppressed(self):
        with self.assertRaises(TypeError):
            safe_json_save({"invalid": object()}, "unused.json")

    def test_two_epochs_finish_and_resume_despite_output_failures(self):
        xyz = torch.randn(2, 24, 3)
        batch = (xyz, torch.ones(2, 24, dtype=torch.long),
                 torch.nn.functional.normalize(torch.randn_like(xyz), dim=-1),
                 torch.ones(2, 24), torch.randn_like(xyz), torch.zeros(2, 24, dtype=torch.long))
        run = SimpleNamespace(id="test-io", log=mock.Mock(side_effect=OSError(5, "WandB I/O")))
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_dir = str(Path(directory) / "checkpoints")
            log_path = str(Path(directory) / "log.json")
            with (
                mock.patch("sys.stdout", BrokenConsole()),
                mock.patch("sys.stderr", BrokenConsole()),
                resilient_console(),
                mock.patch("functional.checkpoint_io.Path.write_text", side_effect=OSError(5, "log I/O")),
                mock.patch("functional.checkpoint_io.time.sleep"),
                mock.patch("functional.cst_pred_trainer.wandb_confusion_matrix", return_value="chart"),
            ):
                trainer = CstPredTrainer(CstPredWrapper("attn_3dgcn"), [batch], checkpoint_dir,
                                         log_path, 2, 1e-4, "io_test", wandb_run=run)
                trainer.start()
            self.assertEqual(trainer.global_step, 2)
            self.assertEqual(run.log.call_count, 2)
            last = Path(checkpoint_dir) / "last.pth"
            self.assertEqual(torch.load(last, map_location="cpu")["epoch"], 1)
            resumed = CstPredTrainer(CstPredWrapper("attn_3dgcn"), [batch], checkpoint_dir,
                                     log_path, 3, 1e-4, "io_test", checkpoint_action="resume",
                                     checkpoint_source=str(last))
            self.assertEqual(resumed.start_epoch, 2)
            resumed.start()
            self.assertEqual(torch.load(last, map_location="cpu")["epoch"], 2)
            self.assertTrue(Path(log_path).is_file())

    def test_model_errors_are_not_swallowed(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = CstPredTrainer(CstPredWrapper("attn_3dgcn"), [], directory,
                                     str(Path(directory) / "log.json"), 1, 1e-4, "test")
            with mock.patch.object(trainer, "process_epoch", side_effect=RuntimeError("model bug")):
                with self.assertRaisesRegex(RuntimeError, "model bug"):
                    trainer.start()


if __name__ == "__main__":
    unittest.main()
