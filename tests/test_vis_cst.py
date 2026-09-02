from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import vis_cst
from data_utils.stage1_h5 import convert_stage1_txt_to_h5


class _FakeResponse:
    def __init__(self, *, url: str, status_code: int, headers=None, content=b""):
        self.url = url
        self.status_code = status_code
        self.headers = headers or {}
        self.content = content

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _FakeRangeClient:
    def __init__(self, blob: bytes):
        self.blob = blob

    def get(self, url, headers=None):
        byte_range = headers["Range"].removeprefix("bytes=")
        start_text, stop_text = byte_range.split("-", maxsplit=1)
        start, stop = int(start_text), int(stop_text)
        return _FakeResponse(
            url=url,
            status_code=206,
            headers={
                "content-range": f"bytes {start}-{stop}/{len(self.blob)}"
            },
            content=self.blob[start : stop + 1],
        )

    def close(self):
        pass


def _sample(point_count: int) -> np.ndarray:
    sample = np.zeros((point_count, 12), dtype=np.float32)
    sample[:, :3] = np.arange(point_count * 3, dtype=np.float32).reshape(-1, 3)
    sample[:, 3] = np.arange(point_count) % 5
    sample[:, 4:7] = (0.0, 0.0, 1.0)
    sample[:, 7] = 0.5
    sample[:, 8:11] = 0.25
    sample[:, 11] = np.arange(point_count) // 2
    return sample


class ConstraintVisualizationSourceTest(unittest.TestCase):
    def test_http_range_reader_supports_h5py_sample_reads(self):
        with tempfile.TemporaryDirectory(dir=".") as temporary:
            root = Path(temporary)
            txt_root = root / "txt"
            txt_root.mkdir()
            expected = _sample(12)
            np.savetxt(txt_root / "sample.txt", expected)
            shard = convert_stage1_txt_to_h5(
                txt_root,
                root / "h5",
                samples_per_shard=1,
                compression="lzf",
            )[0]
            blob = shard.read_bytes()
            fake_client = _FakeRangeClient(blob)

            with patch("httpx.Client", return_value=fake_client):
                with vis_cst.HttpRangeReader(
                    "https://example.test/sample.h5",
                    block_size=4096,
                    max_blocks=8,
                ) as reader:
                    fields, index, sample_count = vis_cst._read_h5_sample(reader, 0)
                    self.assertGreater(reader.request_count, 0)
                    self.assertGreater(reader.bytes_fetched, 0)

            self.assertEqual(index, 0)
            self.assertEqual(sample_count, 1)
            for actual, field_slice in zip(
                fields,
                (
                    expected[:, 0:3],
                    expected[:, 3],
                    expected[:, 4:7],
                    expected[:, 7],
                    expected[:, 8:11],
                    expected[:, 11],
                ),
            ):
                np.testing.assert_array_equal(actual, field_slice)

    def test_negative_sample_index_selects_from_end(self):
        self.assertEqual(vis_cst._normalize_sample_index(-1, 10), 9)
        with self.assertRaises(IndexError):
            vis_cst._normalize_sample_index(10, 10)


if __name__ == "__main__":
    unittest.main()
