from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from data_utils.stage1_h5 import discover_stage1_h5_files, import_h5py
from functional.constraints import (
    _fit_cone,
    _fit_cylinder,
    canonicalize_directions,
    estimate_normals_pca,
)


PRIMITIVE_NAMES = {1: "cylinder", 2: "cone"}


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p90": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": _percentile(values, 90.0),
    }


def _canonical_target_direction(values: np.ndarray, device: torch.device) -> torch.Tensor:
    direction = torch.as_tensor(values, device=device, dtype=torch.float32).mean(dim=0)
    return canonicalize_directions(direction.view(1, 3)).view(3)


def _axis_error_deg(
    prediction: torch.Tensor, target: torch.Tensor, *, unsigned: bool = False
) -> float:
    prediction = canonicalize_directions(prediction.view(1, 3)).view(3)
    target = canonicalize_directions(target.view(1, 3)).view(3)
    cosine = torch.dot(prediction, target)
    if unsigned:
        cosine = cosine.abs()
    cosine = cosine.clamp(-1.0, 1.0)
    return float(torch.acos(cosine).item() * (180.0 / math.pi))


def _geometry_residual(
    primitive_type: int,
    points: torch.Tensor,
    direction: torch.Tensor,
    dimension: torch.Tensor,
    location: torch.Tensor,
) -> torch.Tensor:
    offset = points - location
    axial = offset @ direction
    radial = (offset - axial.unsqueeze(1) * direction).norm(dim=1)
    if primitive_type == 1:
        return (radial - dimension).abs()
    expected_radial = axial.abs() * torch.tan(dimension)
    return (radial - expected_radial).abs()


def _new_primitive_accumulator() -> dict[str, object]:
    return {
        "instances_seen": 0,
        "instances_fitted": 0,
        "instances_failed": 0,
        "instances_skipped_too_small": 0,
        "instances_skipped_mixed_labels": 0,
        "point_count": [],
        "direction_error_deg": [],
        "axis_error_deg_unsigned": [],
        "canonical_sign_mismatches": 0,
        "dimension_absolute_error": [],
        "location_distance_error": [],
        "geometry_residual_mean": [],
        "geometry_residual_median": [],
        "target_direction_spread_deg": [],
        "target_dimension_spread": [],
        "target_location_spread": [],
    }


def _finalize_primitive_accumulator(
    primitive_type: int, accumulator: dict[str, object]
) -> dict[str, object]:
    result: dict[str, object] = {
        key: int(value)
        for key, value in accumulator.items()
        if isinstance(value, int)
    }
    for key, value in accumulator.items():
        if isinstance(value, list):
            result[key] = _summary(value)
    if primitive_type == 2:
        dimension_errors = accumulator["dimension_absolute_error"]
        assert isinstance(dimension_errors, list)
        result["semi_angle_absolute_error_deg"] = _summary(
            [float(value) * (180.0 / math.pi) for value in dimension_errors]
        )
    return result


def evaluate_h5_fitting(
    input_path: str | Path,
    *,
    max_samples: int = 0,
    min_points: int = 12,
    max_points_per_instance: int = 2048,
    normal_k: int = 16,
    min_label_purity: float = 0.95,
    device: str = "auto",
    seed: int = 0,
) -> dict[str, object]:
    if min_points < 4:
        raise ValueError("min_points must be at least 4")
    if max_points_per_instance < min_points:
        raise ValueError("max_points_per_instance must be >= min_points")
    if normal_k < 2:
        raise ValueError("normal_k must be at least 2")
    if not 0.0 < min_label_purity <= 1.0:
        raise ValueError("min_label_purity must be in (0, 1]")

    if device == "auto":
        torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        torch_device = torch.device(device)
    h5_files = discover_stage1_h5_files(input_path)
    if not h5_files:
        raise FileNotFoundError(f"no HDF5 files found below: {input_path}")

    rng = np.random.default_rng(seed)
    accumulators = {primitive_type: _new_primitive_accumulator() for primitive_type in PRIMITIVE_NAMES}
    primitive_point_counts = np.zeros(5, dtype=np.int64)
    xyz_min = np.full(3, np.inf, dtype=np.float64)
    xyz_max = np.full(3, -np.inf, dtype=np.float64)
    xyz_outside_unit_cube_count = 0
    samples_evaluated = 0
    points_evaluated = 0
    started_at = time.perf_counter()
    h5py = import_h5py()

    with torch.no_grad():
        for h5_path in h5_files:
            with h5py.File(h5_path, "r") as h5_file:
                offsets = np.asarray(h5_file["offsets"], dtype=np.int64)
                for sample_index in range(len(offsets) - 1):
                    if max_samples > 0 and samples_evaluated >= max_samples:
                        break
                    start = int(offsets[sample_index])
                    stop = int(offsets[sample_index + 1])
                    xyz = np.asarray(h5_file["xyz"][start:stop], dtype=np.float32)
                    pmt = np.asarray(h5_file["pmt"][start:stop], dtype=np.int64)
                    direction = np.asarray(
                        h5_file["direction"][start:stop], dtype=np.float32
                    )
                    dimension = np.asarray(
                        h5_file["dimension"][start:stop], dtype=np.float32
                    )
                    location = np.asarray(
                        h5_file["location"][start:stop], dtype=np.float32
                    )
                    affiliate_idx = np.asarray(
                        h5_file["affiliate_idx"][start:stop], dtype=np.int64
                    )

                    samples_evaluated += 1
                    points_evaluated += int(len(xyz))
                    if len(xyz) > 0:
                        xyz_min = np.minimum(xyz_min, xyz.min(axis=0))
                        xyz_max = np.maximum(xyz_max, xyz.max(axis=0))
                        xyz_outside_unit_cube_count += int(
                            np.any(np.abs(xyz) > 1.0, axis=1).sum()
                        )
                        primitive_point_counts += np.bincount(pmt, minlength=5)[:5]

                    for instance_id in np.unique(affiliate_idx):
                        instance_indices = np.flatnonzero(affiliate_idx == instance_id)
                        if instance_indices.size == 0:
                            continue
                        instance_types = pmt[instance_indices]
                        counts = np.bincount(instance_types, minlength=5)
                        primitive_type = int(counts.argmax())
                        if primitive_type not in PRIMITIVE_NAMES:
                            continue
                        accumulator = accumulators[primitive_type]
                        accumulator["instances_seen"] += 1
                        purity = float(counts[primitive_type]) / float(instance_indices.size)
                        if purity < min_label_purity:
                            accumulator["instances_skipped_mixed_labels"] += 1
                            continue
                        instance_indices = instance_indices[instance_types == primitive_type]
                        if instance_indices.size < min_points:
                            accumulator["instances_skipped_too_small"] += 1
                            continue

                        if instance_indices.size > max_points_per_instance:
                            instance_indices = np.sort(
                                rng.choice(
                                    instance_indices,
                                    size=max_points_per_instance,
                                    replace=False,
                                )
                            )
                        accumulator["point_count"].append(float(instance_indices.size))
                        points = torch.as_tensor(
                            xyz[instance_indices],
                            device=torch_device,
                            dtype=torch.float32,
                        )
                        target_direction = _canonical_target_direction(
                            direction[instance_indices], torch_device
                        )
                        target_dimension = torch.as_tensor(
                            float(np.median(dimension[instance_indices])),
                            device=torch_device,
                            dtype=torch.float32,
                        )
                        target_location = torch.as_tensor(
                            np.median(location[instance_indices], axis=0),
                            device=torch_device,
                            dtype=torch.float32,
                        )

                        per_point_target_direction = canonicalize_directions(
                            torch.as_tensor(
                                direction[instance_indices],
                                device=torch_device,
                                dtype=torch.float32,
                            )
                        )
                        target_cosine = (
                            per_point_target_direction * target_direction
                        ).sum(dim=1).clamp(-1.0, 1.0)
                        accumulator["target_direction_spread_deg"].append(
                            float(
                                (torch.acos(target_cosine) * (180.0 / math.pi))
                                .max()
                                .item()
                            )
                        )
                        accumulator["target_dimension_spread"].append(
                            float(np.ptp(dimension[instance_indices]))
                        )
                        accumulator["target_location_spread"].append(
                            float(
                                np.linalg.norm(
                                    location[instance_indices]
                                    - np.median(location[instance_indices], axis=0),
                                    axis=1,
                                ).max()
                            )
                        )

                        try:
                            normals = estimate_normals_pca(
                                points.unsqueeze(0),
                                k=min(normal_k, points.shape[0] - 1),
                            )[0]
                            if primitive_type == 1:
                                prediction = _fit_cylinder(points, normals)
                            else:
                                prediction = _fit_cone(points, normals)
                            pred_direction, pred_dimension, pred_location = prediction
                            finite = (
                                torch.isfinite(pred_direction).all()
                                and torch.isfinite(pred_dimension)
                                and torch.isfinite(pred_location).all()
                            )
                            if not bool(finite):
                                raise RuntimeError("non-finite fit")
                        except (RuntimeError, ValueError):
                            accumulator["instances_failed"] += 1
                            continue

                        accumulator["instances_fitted"] += 1
                        direction_error = _axis_error_deg(
                            pred_direction, target_direction
                        )
                        unsigned_axis_error = _axis_error_deg(
                            pred_direction, target_direction, unsigned=True
                        )
                        accumulator["direction_error_deg"].append(direction_error)
                        accumulator["axis_error_deg_unsigned"].append(
                            unsigned_axis_error
                        )
                        if direction_error > 90.0:
                            accumulator["canonical_sign_mismatches"] += 1
                        accumulator["dimension_absolute_error"].append(
                            float((pred_dimension - target_dimension).abs().item())
                        )
                        accumulator["location_distance_error"].append(
                            float((pred_location - target_location).norm().item())
                        )
                        residual = _geometry_residual(
                            primitive_type,
                            points,
                            pred_direction,
                            pred_dimension,
                            pred_location,
                        )
                        accumulator["geometry_residual_mean"].append(
                            float(residual.mean().item())
                        )
                        accumulator["geometry_residual_median"].append(
                            float(residual.median().item())
                        )
            if max_samples > 0 and samples_evaluated >= max_samples:
                break

    elapsed_seconds = time.perf_counter() - started_at
    return {
        "input": str(Path(input_path).resolve()),
        "h5_files": [str(path.resolve()) for path in h5_files],
        "device": str(torch_device),
        "settings": {
            "max_samples": int(max_samples),
            "min_points": int(min_points),
            "max_points_per_instance": int(max_points_per_instance),
            "normal_k": int(normal_k),
            "min_label_purity": float(min_label_purity),
            "seed": int(seed),
        },
        "dataset": {
            "samples_evaluated": int(samples_evaluated),
            "points_evaluated": int(points_evaluated),
            "xyz_min": xyz_min.tolist(),
            "xyz_max": xyz_max.tolist(),
            "xyz_outside_unit_cube_count": int(xyz_outside_unit_cube_count),
            "xyz_outside_unit_cube_ratio": (
                float(xyz_outside_unit_cube_count) / max(points_evaluated, 1)
            ),
            "primitive_point_counts": primitive_point_counts.tolist(),
        },
        "elapsed_seconds": float(elapsed_seconds),
        "primitives": {
            PRIMITIVE_NAMES[primitive_type]: _finalize_primitive_accumulator(
                primitive_type, accumulator
            )
            for primitive_type, accumulator in accumulators.items()
        },
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate cylinder and cone fitting with GT primitive types and "
            "GT affiliate_idx labels from cstnet2 Stage 1 HDF5 shards."
        )
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--max_samples", default=0, type=int)
    parser.add_argument("--min_points", default=12, type=int)
    parser.add_argument("--max_points_per_instance", default=2048, type=int)
    parser.add_argument("--normal_k", default=16, type=int)
    parser.add_argument("--min_label_purity", default=0.95, type=float)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--output_json", type=Path)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    result = evaluate_h5_fitting(
        args.input,
        max_samples=args.max_samples,
        min_points=args.min_points,
        max_points_per_instance=args.max_points_per_instance,
        normal_k=args.normal_k,
        min_label_purity=args.min_label_purity,
        device=args.device,
        seed=args.seed,
    )
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
