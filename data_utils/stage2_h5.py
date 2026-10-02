"""Lossless, split-aware HDF5 storage for Stage 2 classification/segmentation."""
from __future__ import annotations

import argparse
import json
import os
from bisect import bisect_right
from pathlib import Path

import numpy as np

from data_utils.stage1_h5 import (
    _compression_kwargs,
    discover_stage1_h5_files,
    import_h5py,
)


def resolve_storage_format(root, storage_format):
    storage_format = str(storage_format).lower()
    if storage_format not in {"auto", "txt", "h5"}:
        raise ValueError("storage_format must be one of: auto, txt, h5")
    if storage_format == "auto":
        return "h5" if discover_stage1_h5_files(root) else "txt"
    return storage_format


class Stage2H5Store:
    """Index only metadata; lazily open independent handles in each worker."""

    def __init__(self, root, task, split):
        self.paths = []
        self.offsets = []
        self.cumulative = []
        self.source_paths = []
        self.class_ids = []
        self.metadata = None
        self._handles = {}
        self._pid = os.getpid()
        columns = 12 if task == "classification" else 14
        for path in discover_stage1_h5_files(root):
            with import_h5py().File(path, "r") as handle:
                if (handle.attrs.get("format") != f"cstnet2.stage2.{task}"
                        or handle.attrs.get("format_version") != 1):
                    raise ValueError(f"incompatible Stage 2 {task} HDF5 file: {path}")
                if handle.attrs.get("split") != split:
                    continue
                required = {"points", "offsets", "source_path"}
                if task == "classification":
                    required.add("class_id")
                if not required.issubset(handle):
                    raise ValueError(f"missing HDF5 fields in {path}: {required - set(handle)}")
                offsets = np.asarray(handle["offsets"], dtype=np.int64)
                if (offsets.ndim != 1 or len(offsets) < 2 or offsets[0] != 0
                        or np.any(np.diff(offsets) <= 0)):
                    raise ValueError(f"invalid sample offsets in {path}")
                count = len(offsets) - 1
                if (handle["points"].shape != (int(offsets[-1]), columns)
                        or handle["source_path"].shape != (count,)):
                    raise ValueError(f"inconsistent HDF5 sample shapes in {path}")
                metadata = json.loads(handle.attrs["metadata"])
                if self.metadata is not None and metadata != self.metadata:
                    raise ValueError(f"inconsistent HDF5 metadata in {path}")
                self.metadata = metadata
                if task == "classification":
                    ids = np.asarray(handle["class_id"])
                    if (ids.shape != (count,) or ids.dtype.kind not in "iu"
                            or not np.isin(ids, list(metadata["classes"].values())).all()):
                        raise ValueError(f"invalid HDF5 class ids in {path}")
                    self.class_ids.extend(ids.tolist())
                self.paths.append(path)
                self.offsets.append(offsets)
                self.source_paths.extend(Path(p) for p in handle["source_path"].asstr()[:])
                self.cumulative.append(len(self.source_paths))
        if not self.paths:
            raise FileNotFoundError(f"no Stage 2 {task} HDF5 samples for split {split!r} in {root}")

    def load(self, index):
        if index < 0:
            index += len(self.source_paths)
        if not 0 <= index < len(self.source_paths):
            raise IndexError(index)
        if self._pid != os.getpid():
            self.close()
            self._pid = os.getpid()
        shard = bisect_right(self.cumulative, index)
        local = index - (self.cumulative[shard - 1] if shard else 0)
        if shard not in self._handles:
            self._handles[shard] = import_h5py().File(self.paths[shard], "r")
        start, stop = self.offsets[shard][local:local + 2]
        return self._handles[shard]["points"][int(start):int(stop)]

    def close(self):
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handles"] = {}
        return state

    def __del__(self):
        self.close()


def _convert(datasets, output_dir, task, metadata, samples_per_shard, compression, overwrite):
    if samples_per_shard <= 0:
        raise ValueError("samples_per_shard must be positive")
    compression_kwargs = _compression_kwargs(compression)
    output = Path(output_dir).resolve()
    prefix = f"stage2-{task}"
    existing = sorted(output.glob(f"{prefix}-*-of-*.h5"))
    if existing and not overwrite:
        raise FileExistsError(f"output shards already exist in {output}; use overwrite=True")
    output.mkdir(parents=True, exist_ok=True)
    written = []
    for dataset in datasets:
        entries = (dataset.datapath if task == "classification"
                   else [(None, path) for path in dataset.files])
        shard_count = (len(entries) + samples_per_shard - 1) // samples_per_shard
        for shard in range(shard_count):
            batch = entries[shard * samples_per_shard:(shard + 1) * samples_per_shard]
            arrays = []
            for _, source in batch:
                if task == "classification":
                    from data_utils.constraint_dataset_common import load_constraint_point_file
                    array = load_constraint_point_file(source, task_name="classification HDF5 conversion")
                else:
                    array = dataset._load_array(source)
                if len(array) == 0:
                    raise ValueError(f"empty point cloud: {source}")
                arrays.append(array)
            offsets = np.concatenate(([0], np.cumsum([len(a) for a in arrays], dtype=np.int64)))
            path = output / f"{prefix}-{dataset.split}-{shard:05d}-of-{shard_count:05d}.h5"
            temporary = path.with_suffix(f".h5.tmp.{os.getpid()}")
            h5py = import_h5py()
            try:
                with h5py.File(temporary, "w") as handle:
                    handle.attrs.update({"format": f"cstnet2.stage2.{task}", "format_version": 1,
                                         "split": dataset.split, "metadata": json.dumps(metadata)})
                    handle.create_dataset("offsets", data=offsets)
                    points = handle.create_dataset("points", (int(offsets[-1]), arrays[0].shape[1]),
                                                   dtype=np.float32, **compression_kwargs)
                    for index, array in enumerate(arrays):
                        points[int(offsets[index]):int(offsets[index + 1])] = array
                    handle.create_dataset("source_path", data=[p.resolve().as_posix() for _, p in batch],
                                          dtype=h5py.string_dtype("utf-8"))
                    if task == "classification":
                        handle.create_dataset("class_id", data=[c for c, _ in batch], dtype=np.int64)
                os.replace(temporary, path)
            finally:
                if temporary.exists():
                    temporary.unlink()
            written.append(path)
            print(f"[{dataset.split} {shard + 1}/{shard_count}] wrote {path}")
    # Delete only obsolete shards belonging to this converter after all writes succeed.
    for path in existing:
        if path not in written:
            path.unlink()
    return written


def convert_classification_txt_to_h5(input_dir, output_dir, *, samples_per_shard=2000,
                                     compression="lzf", overwrite=False, test_ratio=0.2, split_seed=42):
    """Preserve class ids, existing split manifests, all 12 columns and all points."""
    from data_utils.classification_dataset import Stage2ClassificationDataset
    train = Stage2ClassificationDataset(input_dir, "train", storage_format="txt",
                                        test_ratio=test_ratio, split_seed=split_seed)
    test = Stage2ClassificationDataset(input_dir, "test", classes=train.classes, storage_format="txt")
    return _convert([train, test], output_dir, "classification", {"classes": train.classes},
                    samples_per_shard, compression, overwrite)


def convert_segmentation_txt_to_h5(input_dir, output_dir, *, samples_per_shard=2000,
                                   compression="lzf", overwrite=False, label_map_path=None):
    """Preserve train/val[/test] splits and all 14 columns, including face ids/labels."""
    from data_utils.mfcad_seg_dataset import DEFAULT_LABEL_MAP, Stage2SegmentationDataset
    kwargs = {"root": input_dir, "n_points": None, "storage_format": "txt",
              "label_map_path": label_map_path or DEFAULT_LABEL_MAP}
    datasets = [Stage2SegmentationDataset(split=split, **kwargs) for split in ("train", "val")]
    if (Path(input_dir) / "test").is_dir():
        datasets.append(Stage2SegmentationDataset(split="test", **kwargs))
    return _convert(datasets, output_dir, "segmentation", {"label_map": datasets[0].label_map},
                    samples_per_shard, compression, overwrite)


def main(task, argv=None):
    parser = argparse.ArgumentParser(description=f"Convert Stage 2 {task} TXT data to HDF5 shards")
    parser.add_argument("--input_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--samples_per_shard", type=int, default=2000)
    parser.add_argument("--compression", choices=("none", "lzf", "gzip"), default="lzf")
    parser.add_argument("--overwrite", action="store_true")
    if task == "classification":
        parser.add_argument("--test_ratio", type=float, default=0.2)
        parser.add_argument("--split_seed", type=int, default=42)
        converter = convert_classification_txt_to_h5
    else:
        parser.add_argument("--label_map_path", type=Path)
        converter = convert_segmentation_txt_to_h5
    converter(**vars(parser.parse_args(argv)))
