from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from functional.checkpoint_io import CHECKPOINT_SAVE_ATTEMPTS, safe_torch_save


class SafeCheckpointIOTest(unittest.TestCase):
    def test_retries_then_skips_without_destroying_previous_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "last.pth"
            destination.write_bytes(b"previous-valid-checkpoint")
            messages = []
            with mock.patch(
                "functional.checkpoint_io.os.replace",
                side_effect=OSError(5, "Input/output error"),
            ) as replace, mock.patch(
                "functional.checkpoint_io.time.sleep"
            ) as sleep:
                saved = safe_torch_save(
                    {"epoch": 24}, destination, logger=messages.append
                )

            self.assertFalse(saved)
            self.assertEqual(replace.call_count, CHECKPOINT_SAVE_ATTEMPTS)
            self.assertEqual(sleep.call_count, CHECKPOINT_SAVE_ATTEMPTS - 1)
            self.assertEqual(destination.read_bytes(), b"previous-valid-checkpoint")
            self.assertFalse(
                destination.with_suffix(
                    destination.suffix + f".tmp.{os.getpid()}"
                ).exists()
            )
            self.assertIn("save skipped", messages[-1])

    def test_transient_replace_failure_succeeds_on_retry(self):
        real_replace = os.replace
        calls = 0

        def fail_once_then_replace(source, destination):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise OSError(5, "Input/output error")
            real_replace(source, destination)

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "last.pth"
            with mock.patch(
                "functional.checkpoint_io.os.replace",
                side_effect=fail_once_then_replace,
            ), mock.patch("functional.checkpoint_io.time.sleep"):
                saved = safe_torch_save({"epoch": 25}, destination)

            self.assertTrue(saved)
            self.assertEqual(torch.load(destination, map_location="cpu")["epoch"], 25)


if __name__ == "__main__":
    unittest.main()
