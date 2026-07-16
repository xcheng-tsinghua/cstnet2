from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from data_utils.mfcad_seg_dataset import DEFAULT_LABEL_MAP, load_label_map


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    value = color.lstrip("#")
    if len(value) != 6:
        raise ValueError(f"expected #RRGGBB color, got {color!r}")
    return tuple(int(value[offset:offset + 2], 16) for offset in (0, 2, 4))


def colors_from_label_map(label_map: dict[str, Any]) -> np.ndarray:
    return np.asarray([_hex_to_rgb(label["color"]) for label in label_map["labels"]], dtype=np.uint8)


def write_colored_ply(
    path: str | Path,
    xyz: np.ndarray,
    colors: np.ndarray,
    labels: np.ndarray,
    face_ids: np.ndarray,
    confidence: np.ndarray,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    xyz = np.asarray(xyz, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.uint8)
    labels = np.asarray(labels, dtype=np.int64)
    face_ids = np.asarray(face_ids, dtype=np.int64)
    confidence = np.asarray(confidence, dtype=np.float32)
    count = xyz.shape[0]
    if not (
        xyz.shape == (count, 3)
        and colors.shape == (count, 3)
        and labels.shape == (count,)
        and face_ids.shape == (count,)
        and confidence.shape == (count,)
    ):
        raise ValueError("PLY properties must have matching point dimensions")

    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {count}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write("property int label\nproperty int face_id\nproperty float confidence\n")
        handle.write("end_header\n")
        for point, color, label, face_id, score in zip(
            xyz, colors, labels, face_ids, confidence
        ):
            handle.write(
                f"{point[0]:.7f} {point[1]:.7f} {point[2]:.7f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])} "
                f"{int(label)} {int(face_id)} {float(score):.7f}\n"
            )


def export_segmentation_sample(
    output_dir: str | Path,
    sample_id: str,
    xyz: np.ndarray,
    gt_label: np.ndarray,
    pred_label: np.ndarray,
    face_id: np.ndarray,
    confidence: np.ndarray,
    label_map: dict[str, Any],
) -> dict[str, str]:
    output_dir = Path(output_dir) / str(sample_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    xyz = np.asarray(xyz, dtype=np.float32)
    gt_label = np.asarray(gt_label, dtype=np.int64)
    pred_label = np.asarray(pred_label, dtype=np.int64)
    face_id = np.asarray(face_id, dtype=np.int64)
    confidence = np.asarray(confidence, dtype=np.float32)
    palette = colors_from_label_map(label_map)
    if gt_label.min() < 0 or pred_label.min() < 0:
        raise ValueError("visualization export does not support ignored negative labels")
    if gt_label.max() >= len(palette) or pred_label.max() >= len(palette):
        raise ValueError("visualization label is outside the metadata color map")

    npz_path = output_dir / "prediction.npz"
    np.savez_compressed(
        npz_path,
        xyz=xyz,
        gt_label=gt_label,
        pred_label=pred_label,
        face_id=face_id,
        confidence=confidence,
    )
    gt_path = output_dir / "gt.ply"
    pred_path = output_dir / "prediction.ply"
    error_path = output_dir / "error.ply"
    write_colored_ply(gt_path, xyz, palette[gt_label], gt_label, face_id, confidence)
    write_colored_ply(pred_path, xyz, palette[pred_label], pred_label, face_id, confidence)
    errors = pred_label != gt_label
    error_colors = np.full((xyz.shape[0], 3), (160, 160, 160), dtype=np.uint8)
    error_colors[errors] = (255, 0, 0)
    write_colored_ply(error_path, xyz, error_colors, pred_label, face_id, confidence)
    return {
        "npz": str(npz_path),
        "gt_ply": str(gt_path),
        "prediction_ply": str(pred_path),
        "error_ply": str(error_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Convert a saved MFCAD++ segmentation NPZ to PLY views")
    parser.add_argument("prediction_npz")
    parser.add_argument("--label_map", default=str(DEFAULT_LABEL_MAP))
    parser.add_argument("--output_dir", default="")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    source = Path(args.prediction_npz)
    data = np.load(source)
    label_map = load_label_map(args.label_map)
    output_dir = Path(args.output_dir) if args.output_dir else source.parent.parent
    paths = export_segmentation_sample(
        output_dir=output_dir,
        sample_id=source.parent.name,
        xyz=data["xyz"],
        gt_label=data["gt_label"],
        pred_label=data["pred_label"],
        face_id=data["face_id"],
        confidence=data["confidence"],
        label_map=label_map,
    )
    print(json.dumps(paths, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main(parse_args())

