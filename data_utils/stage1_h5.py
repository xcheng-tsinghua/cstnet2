from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from data_utils.constraint_dataset_common import (
    discover_txt_files,
    load_constraint_point_file,
    split_constraint_columns,
)


STAGE1_H5_FORMAT = "cstnet2.stage1"
STAGE1_H5_VERSION = 1
STAGE1_H5_FIELDS = (
    "offsets",
    "xyz",
    "pmt",
    "direction",
    "dimension",
    "location",
    "affiliate_idx",
    "source_path",
)


def import_h5py():
    try:
        import h5py
    except ImportError as error:
        raise ImportError(
            "HDF5 support requires h5py. Install it with `pip install h5py` "
            "in the environment used for conversion and training."
        ) from error
    return h5py


def discover_stage1_h5_files(root: str | Path) -> list[Path]:
    """Find Stage 1 HDF5 files recursively in deterministic path order."""
    root = Path(root)
    if root.is_file():
        if root.suffix.lower() not in {".h5", ".hdf5"}:
            raise ValueError(f"expected an .h5 or .hdf5 file, got: {root}")
        return [root]
    if not root.is_dir():
        raise FileNotFoundError(f"dataset path not found: {root}")
    return sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in {".h5", ".hdf5"}
        ),
        key=lambda path: str(path).lower(),
    )


def _compression_kwargs(compression: str) -> dict[str, object]:
    compression = compression.lower()
    if compression == "none":
        return {}
    if compression not in {"lzf", "gzip"}:
        raise ValueError("compression must be one of: none, lzf, gzip")
    kwargs: dict[str, object] = {"compression": compression, "shuffle": True}
    if compression == "gzip":
        kwargs["compression_opts"] = 1
    return kwargs


def _write_shard(
    output_path: Path,
    source_names: list[str],
    samples: list[tuple[np.ndarray, ...]],
    *,
    compression: str,
) -> None:
    h5py = import_h5py()
    point_counts = np.asarray([sample[0].shape[0] for sample in samples], dtype=np.int64)
    offsets = np.empty(len(samples) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(point_counts, out=offsets[1:])
    total_points = int(offsets[-1])
    compression_kwargs = _compression_kwargs(compression)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")

    try:
        with h5py.File(temporary_path, "w") as h5_file:
            h5_file.attrs["format"] = STAGE1_H5_FORMAT
            h5_file.attrs["format_version"] = STAGE1_H5_VERSION
            h5_file.attrs["sample_count"] = len(samples)
            h5_file.attrs["point_count"] = total_points
            h5_file.create_dataset("offsets", data=offsets)
            datasets = {
                "xyz": h5_file.create_dataset(
                    "xyz", (total_points, 3), dtype=np.float32, **compression_kwargs
                ),
                "pmt": h5_file.create_dataset(
                    "pmt", (total_points,), dtype=np.uint8, **compression_kwargs
                ),
                "direction": h5_file.create_dataset(
                    "direction", (total_points, 3), dtype=np.float32, **compression_kwargs
                ),
                "dimension": h5_file.create_dataset(
                    "dimension", (total_points,), dtype=np.float32, **compression_kwargs
                ),
                "location": h5_file.create_dataset(
                    "location", (total_points, 3), dtype=np.float32, **compression_kwargs
                ),
                "affiliate_idx": h5_file.create_dataset(
                    "affiliate_idx", (total_points,), dtype=np.int32, **compression_kwargs
                ),
            }
            string_dtype = h5py.string_dtype(encoding="utf-8")
            h5_file.create_dataset(
                "source_path",
                data=np.asarray(source_names, dtype=object),
                dtype=string_dtype,
            )

            for sample_index, sample in enumerate(samples):
                start = int(offsets[sample_index])
                stop = int(offsets[sample_index + 1])
                for field_name, values in zip(
                    (
                        "xyz",
                        "pmt",
                        "direction",
                        "dimension",
                        "location",
                        "affiliate_idx",
                    ),
                    sample,
                ):
                    datasets[field_name][start:stop] = values
            h5_file.flush()
        os.replace(temporary_path, output_path)
    except BaseException:
        if temporary_path.exists():
            temporary_path.unlink()
        raise


def _normalize_input_roots(
    input_dir: str | Path | Sequence[str | Path],
) -> list[Path]:
    if isinstance(input_dir, (str, Path)):
        candidates = [input_dir]
    else:
        candidates = list(input_dir)
    if not candidates:
        raise ValueError("at least one input directory is required")

    roots: list[Path] = []
    seen_roots: set[Path] = set()
    for candidate in candidates:
        root = Path(candidate).resolve()
        if root in seen_roots:
            continue
        seen_roots.add(root)
        roots.append(root)
    return roots


def _discover_txt_files_from_roots(
    input_roots: Sequence[Path],
) -> tuple[list[Path], list[str]]:
    """Discover files in root order and remove duplicates from overlapping roots."""
    source_files: list[Path] = []
    source_names: list[str] = []
    seen_files: set[Path] = set()
    for root_index, input_root in enumerate(input_roots):
        root_label = f"root_{root_index:03d}_{input_root.name or 'root'}"
        for source_path in discover_txt_files(input_root):
            resolved_path = source_path.resolve()
            if resolved_path in seen_files:
                continue
            seen_files.add(resolved_path)
            source_files.append(resolved_path)
            source_names.append(
                f"{root_label}/{resolved_path.relative_to(input_root).as_posix()}"
            )
    return source_files, source_names


def convert_stage1_txt_to_h5(
    input_dir: str | Path | Sequence[str | Path],
    output_dir: str | Path,
    *,
    samples_per_shard: int = 2_000,
    compression: str = "lzf",
    overwrite: bool = False,
) -> list[Path]:
    """Recursively convert one or more Stage 1 TXT trees to HDF5 shards.

    All points are retained. Each input row must have at least 12 columns; only
    the first 12 are converted into the six fields returned by
    :class:`Stage1ConstraintDataset`.
    """
    input_roots = _normalize_input_roots(input_dir)
    output_root = Path(output_dir).resolve()
    if samples_per_shard <= 0:
        raise ValueError("samples_per_shard must be positive")
    source_files, source_names = _discover_txt_files_from_roots(input_roots)
    output_root.mkdir(parents=True, exist_ok=True)
    shard_count = (len(source_files) + samples_per_shard - 1) // samples_per_shard
    expected_paths = [
        output_root / f"stage1-{shard_index:05d}-of-{shard_count:05d}.h5"
        for shard_index in range(shard_count)
    ]
    existing_paths = sorted(output_root.glob("stage1-*-of-*.h5"))
    if existing_paths and not overwrite:
        raise FileExistsError(
            f"{len(existing_paths)} output shard(s) already exist; use overwrite=True "
            f"to replace them, for example: {existing_paths[0]}"
        )

    written_paths: list[Path] = []
    for shard_index, output_path in enumerate(expected_paths):
        shard_paths = source_files[
            shard_index * samples_per_shard : (shard_index + 1) * samples_per_shard
        ]
        shard_source_names = source_names[
            shard_index * samples_per_shard : (shard_index + 1) * samples_per_shard
        ]
        samples: list[tuple[np.ndarray, ...]] = []
        for source_path in shard_paths:
            point_set = load_constraint_point_file(
                source_path,
                task_name="Stage 1 HDF5 conversion",
                allow_extra_columns=True,
            )
            # Conversion is storage-preserving. Legacy sentinels are
            # canonicalized only when a consumer loads the resulting shard.
            samples.append(
                split_constraint_columns(
                    point_set, canonicalize_invalid=False
                )
            )
        _write_shard(
            output_path,
            shard_source_names,
            samples,
            compression=compression,
        )
        written_paths.append(output_path)
        print(
            f"[{shard_index + 1}/{shard_count}] wrote {output_path} "
            f"({len(shard_paths)} samples)"
        )

    manifest = {
        "format": STAGE1_H5_FORMAT,
        "format_version": STAGE1_H5_VERSION,
        "input_roots": [str(path) for path in input_roots],
        "sample_count": len(source_files),
        "samples_per_shard": samples_per_shard,
        "compression": compression,
        "shards": [path.name for path in written_paths],
    }
    if len(input_roots) == 1:
        # Preserve the original manifest field for existing single-root users.
        manifest["input_root"] = str(input_roots[0])
    manifest_path = output_root / "stage1_manifest.json"
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_manifest, manifest_path)
    expected_path_set = set(expected_paths)
    for stale_path in existing_paths:
        if stale_path not in expected_path_set:
            stale_path.unlink()
    return written_paths


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recursively convert Stage 1 TXT samples to HDF5 shards."
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        nargs="+",
        type=Path,
        help="one or more directories recursively containing Stage 1 TXT files",
    )
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--samples_per_shard", default=2_000, type=int)
    parser.add_argument(
        "--compression", default="lzf", choices=("none", "lzf", "gzip")
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    paths = convert_stage1_txt_to_h5(
        args.input_dir,
        args.output_dir,
        samples_per_shard=args.samples_per_shard,
        compression=args.compression,
        overwrite=args.overwrite,
    )
    print(f"converted {len(paths)} shard(s) into {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
