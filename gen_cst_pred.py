"""Generate point-cloud files containing offline Stage 1 constraint predictions.

Only XYZ is used for inference. Replace columns 3:11 (pmt, mad, dim, loc),
preserving XYZ and every column from index 11 onward. TXT files retain their
relative paths. No clustering or fitting is run.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from data_utils.constraint_dataset_common import zero_invalid_constraint_components
from functional.direct_constraints import direct_constraints, validate_direct_checkpoint
from functional.point_features import stage1_forward, stage1_feature_dim
from networks.cst_pred_wrapper import CstPredWrapper


MODEL_NAMES = ("pointnet2", "pointnet", "attn_3dgcn")
GT_CORE_COLUMNS = 11
DEFAULT_CHECKPOINT = (
    Path(__file__).resolve().parent
    / "model_trained/stage1_direct/attn_3dgcn/joint/last.pth"
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run Stage 1 offline and mirror point clouds with predicted constraints."
    )
    parser.add_argument("--input_dir", default=r"/opt/data/private/data_set/pcd_cstnet2/mfcad_pcd", type=str)
    parser.add_argument("--output_dir", default=r"/opt/data/private/data_set/pcd_cstnet2/mfcad_pcd_pred", type=str)
    parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_CHECKPOINT),
        type=str,
        help="Stage 1 .pth file or directory; defaults to stage1_direct/attn_3dgcn/joint/last.pth.",
    )
    parser.add_argument(
        "--model", default="auto", choices=("auto",) + MODEL_NAMES,
        help="Default: read the model name from checkpoint metadata.",
    )
    parser.add_argument(
        "--device", default="auto", type=str,
        help="auto, cpu, cuda, or an explicit device such as cuda:1.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    return parser.parse_args(argv)


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {name}")
    return device


def resolve_checkpoint(path: str | os.PathLike[str]) -> Path:
    checkpoint = Path(path).expanduser()
    if checkpoint.is_file():
        return checkpoint
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Stage 1 checkpoint not found: {checkpoint}")
    candidates = (
        "best_constraint_score.pth",
        "best_pmt_miou.pth",
        "last.pth",
    )
    for name in candidates:
        candidate = checkpoint / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"no Stage 1 checkpoint found in {checkpoint}; tried {', '.join(candidates)}"
    )


def _extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise ValueError("Stage 1 checkpoint must be a dictionary")
    if isinstance(checkpoint.get("model"), dict):
        state = checkpoint["model"]
    elif isinstance(checkpoint.get("state_dict"), dict):
        state = checkpoint["state_dict"]
    elif checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
        state = checkpoint
    else:
        raise ValueError("checkpoint does not contain model/state_dict weights")
    if state and all(key.startswith("module.") for key in state):
        state = {key[len("module."):]: value for key, value in state.items()}
    return state


class Stage1Predictor:
    """Strictly loaded, inference-only Stage 1 constraint predictor."""

    def __init__(
        self,
        checkpoint_path: str | os.PathLike[str],
        device: torch.device,
        model_name: str = "auto",
    ):
        self.checkpoint_path = resolve_checkpoint(checkpoint_path)
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        checkpoint_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
        if not isinstance(checkpoint_args, dict):
            checkpoint_args = {}

        self.model_name = (
            str(checkpoint_args.get("model", "pointnet2"))
            if model_name == "auto" else model_name
        )
        if self.model_name not in MODEL_NAMES:
            raise ValueError(f"unsupported Stage 1 model: {self.model_name}")
        validate_direct_checkpoint(checkpoint_args)

        self.use_extra_features = _as_bool(
            checkpoint_args.get("use_extra_features", False)
        )
        self.feature_k = int(checkpoint_args.get("feature_k", 16))
        channel_fea = stage1_feature_dim(self.use_extra_features)
        self.model = CstPredWrapper(
            self.model_name,
            channel_fea=channel_fea,
        )
        self.model.load_state_dict(_extract_state_dict(checkpoint), strict=True)
        self.device = device
        self.model.to(device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.inference_mode()
    def predict(self, xyz_array: np.ndarray) -> dict[str, np.ndarray]:
        xyz = torch.as_tensor(
            np.ascontiguousarray(xyz_array), dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        model_output = stage1_forward(self.model, xyz, use_extra_features=self.use_extra_features, feature_k=self.feature_k)
        required_outputs = ("embedding", "log_pmt", "mad", "dim", "loc")
        if any(not torch.isfinite(model_output[name]).all() for name in required_outputs):
            raise FloatingPointError("Stage 1 output contains NaN or Inf")

        constraints = direct_constraints(model_output)
        return {
            "pmt": constraints["primitive_type"].argmax(dim=-1)[0].cpu().numpy(),
            "mad": constraints["direction"][0].cpu().numpy(),
            "dim": constraints["dimension"][0].cpu().numpy(),
            "loc": constraints["location"][0].cpu().numpy(),
        }


def build_output_array(
    input_array: np.ndarray,
    prediction: dict[str, np.ndarray],
) -> np.ndarray:
    if input_array.ndim != 2 or input_array.shape[1] < GT_CORE_COLUMNS:
        raise ValueError("expected at least 11 columns (xyz,pmt,mad,dim,loc)")
    count = input_array.shape[0]
    for name in ("pmt", "mad", "dim", "loc"):
        if len(prediction[name]) != count:
            raise ValueError(f"prediction {name} has {len(prediction[name])} rows, expected {count}")

    pmt = np.asarray(prediction["pmt"]).reshape(count)
    mad, dim = zero_invalid_constraint_components(
        pmt,
        np.asarray(prediction["mad"]).reshape(count, 3),
        np.asarray(prediction["dim"]).reshape(count),
    )
    core = np.concatenate(
        [
            input_array[:, :3],
            pmt.reshape(count, 1),
            mad,
            dim.reshape(count, 1),
            np.asarray(prediction["loc"]).reshape(count, 3),
        ],
        axis=1,
    )
    return np.concatenate([core, input_array[:, GT_CORE_COLUMNS:]], axis=1)


def load_point_file(path: Path) -> tuple[np.ndarray, str]:
    delimiter = " "
    try:
        array = np.loadtxt(path, dtype=np.float64, ndmin=2)
    except ValueError as whitespace_error:
        try:
            array = np.loadtxt(path, dtype=np.float64, delimiter=",", ndmin=2)
            delimiter = ","
        except ValueError:
            raise whitespace_error
    if array.shape[1] < GT_CORE_COLUMNS:
        raise ValueError(f"expected a 2D point array with at least 11 columns: {path}")
    if array.shape[0] < 3:
        raise ValueError(f"at least 3 points are required: {path}")
    if not np.isfinite(array[:, :3]).all():
        raise ValueError(f"XYZ contains NaN or Inf: {path}")
    return array, delimiter


def save_point_file(path: Path, array: np.ndarray, delimiter: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
            formats = ["%.6f"] * array.shape[1]
            formats[3] = "%d"
            np.savetxt(temporary, array, fmt=formats, delimiter=delimiter)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def iter_point_files(root: Path) -> Iterable[Path]:
    return (
        path for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix.lower() == ".txt"
    )


def validate_roots(input_dir: Path, output_dir: Path) -> None:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"input directory not found: {input_dir}")
    input_resolved = input_dir.resolve()
    output_resolved = output_dir.resolve()
    if input_resolved == output_resolved or input_resolved in output_resolved.parents:
        raise ValueError("output_dir must not be the input directory or one of its subdirectories")


def generate_dataset(args) -> None:
    input_dir = Path(args.input_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    validate_roots(input_dir, output_dir)
    files = list(iter_point_files(input_dir))
    if not files:
        raise FileNotFoundError(
            f"no .txt point cloud files found below {input_dir}"
        )

    # Reproduce the whole directory tree, including empty directories. Files
    # other than TXT point clouds are intentionally not copied.
    output_dir.mkdir(parents=True, exist_ok=True)
    for directory in sorted(path for path in input_dir.rglob("*") if path.is_dir()):
        (output_dir / directory.relative_to(input_dir)).mkdir(
            parents=True, exist_ok=True
        )

    device = resolve_device(args.device)
    predictor = Stage1Predictor(
        checkpoint_path=args.checkpoint,
        device=device,
        model_name=args.model,
    )
    print(
        "Stage 1 predictor: "
        f"checkpoint={predictor.checkpoint_path}; model={predictor.model_name}; "
        f"device={device}; "
        "constraint_route=direct_mlp_v1; preserve XYZ and extra attributes"
    )
    print(f"input files: {len(files)}; input={input_dir}; output={output_dir}")

    written = skipped = 0
    for index, input_path in enumerate(files, start=1):
        relative_path = input_path.relative_to(input_dir)
        output_path = output_dir / relative_path
        if output_path.exists() and not args.overwrite:
            skipped += 1
            print(f"[{index}/{len(files)}] skip existing: {relative_path}")
            continue
        input_array, delimiter = load_point_file(input_path)
        prediction = predictor.predict(input_array[:, :3])
        output_array = build_output_array(input_array, prediction)
        save_point_file(output_path, output_array, delimiter)
        written += 1
        print(
            f"[{index}/{len(files)}] saved: {relative_path} "
            f"({input_array.shape[0]} points, {output_array.shape[1]} columns)"
        )
    print(f"finished: written={written}, skipped={skipped}, total={len(files)}")


def main(argv=None) -> None:
    generate_dataset(parse_args(argv))


if __name__ == "__main__":
    main()
