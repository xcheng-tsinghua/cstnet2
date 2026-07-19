"""Generate point-cloud files containing offline Stage 1 constraint predictions.

Only the first three columns of every input point are used for inference.  The
output core follows ``CstNet2Dataset`` exactly:

    xyz, pmt, mad, dim, nor, loc, affiliate_idx

Unknown input columns are treated as opaque task attributes and are preserved
after that 15-column core.  Relative paths below the input directory are kept.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from functional.constraints import assemble_constraints_from_stage1
from functional.point_features import build_stage1_input_features, stage1_feature_dim
from networks.cst_pred_wrapper import CstPredWrapper


MODEL_NAMES = ("pointnet2", "pointnet", "attn_3dgcn")
STAGE1_MODES = ("baseline", "multitask")
GT_CORE_COLUMNS = 15


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run Stage 1 offline and mirror point clouds with predicted constraints."
    )
    parser.add_argument("--input_dir", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=str,
        help="Stage 1 .pth file or checkpoint directory.",
    )
    parser.add_argument(
        "--model", default="auto", choices=("auto",) + MODEL_NAMES,
        help="Default: read the model name from checkpoint metadata.",
    )
    parser.add_argument(
        "--stage1_mode", default="auto", choices=("auto",) + STAGE1_MODES,
        help="Default: read baseline/multitask from checkpoint metadata.",
    )
    parser.add_argument(
        "--device", default="auto", type=str,
        help="auto, cpu, cuda, or an explicit device such as cuda:1.",
    )
    parser.add_argument("--cluster_bandwidth", default=None, type=float)
    parser.add_argument("--normal_k", default=16, type=int)
    parser.add_argument(
        "--extensions", default=".txt", type=str,
        help="Comma-separated extensions, for example .txt,.xyz,.npy.",
    )
    parser.add_argument(
        "--input_layout", default="auto", choices=("auto", "raw", "gt"),
        help="raw appends all input columns after xyz; gt replaces columns 3:15.",
    )
    parser.add_argument("--overwrite", action="store_true")
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
        "best_cluster_ari.pth",
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
        stage1_mode: str = "auto",
        cluster_bandwidth: float | None = None,
        normal_k: int = 16,
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
        self.stage1_mode = (
            str(checkpoint_args.get("stage1_mode", "baseline"))
            if stage1_mode == "auto" else stage1_mode
        )
        if self.model_name not in MODEL_NAMES:
            raise ValueError(f"unsupported Stage 1 model: {self.model_name}")
        if self.stage1_mode not in STAGE1_MODES:
            raise ValueError(f"unsupported Stage 1 mode: {self.stage1_mode}")

        self.use_extra_features = _as_bool(
            checkpoint_args.get("use_extra_features", False)
        )
        self.normal_source = str(checkpoint_args.get("normal_source", "none"))
        self.feature_k = int(checkpoint_args.get("feature_k", 16))
        if self.use_extra_features and self.normal_source == "gt":
            raise ValueError(
                "this checkpoint was trained with GT normals, but gen_cst_pred.py "
                "is intentionally XYZ-only; use an XYZ/PCA checkpoint"
            )
        channel_fea = stage1_feature_dim(
            self.use_extra_features, self.normal_source
        )
        self.model = CstPredWrapper(
            self.model_name,
            channel_fea=channel_fea,
            stage1_mode=self.stage1_mode,
        )
        self.model.load_state_dict(_extract_state_dict(checkpoint), strict=True)
        self.device = device
        self.model.to(device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        saved_bandwidth = checkpoint_args.get("cluster_bandwidth", 0.35)
        self.cluster_bandwidth = float(
            saved_bandwidth if cluster_bandwidth is None else cluster_bandwidth
        )
        if self.cluster_bandwidth <= 0:
            raise ValueError("cluster_bandwidth must be positive")
        self.normal_k = int(normal_k)
        if self.normal_k < 2:
            raise ValueError("normal_k must be at least 2")

    def _extra_features(self, xyz: torch.Tensor) -> torch.Tensor | None:
        if not self.use_extra_features:
            return None
        return build_stage1_input_features(
            xyz,
            normals=None,
            use_normals=self.normal_source == "pca",
            use_curvature=True,
            k=self.feature_k,
        )

    @torch.inference_mode()
    def predict(self, xyz_array: np.ndarray) -> dict[str, np.ndarray]:
        xyz = torch.as_tensor(
            np.ascontiguousarray(xyz_array), dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        model_output = self.model(xyz, self._extra_features(xyz))
        if isinstance(model_output, dict):
            embedding = model_output["embedding"]
            log_pmt = model_output["log_pmt"]
        else:
            embedding, log_pmt = model_output
        if not torch.isfinite(embedding).all() or not torch.isfinite(log_pmt).all():
            raise FloatingPointError("Stage 1 output contains NaN or Inf")

        constraints = assemble_constraints_from_stage1(
            xyz=xyz,
            cluster_embedding=embedding,
            log_primitive=log_pmt,
            normals=None,
            cluster_bandwidth=self.cluster_bandwidth,
            normal_k=self.normal_k,
        )
        return {
            "pmt": constraints["primitive_type"].argmax(dim=-1)[0].cpu().numpy(),
            "mad": constraints["direction"][0].cpu().numpy(),
            "dim": constraints["dimension"][0].cpu().numpy(),
            "nor": constraints["continuity"][0].cpu().numpy(),
            "loc": constraints["location"][0].cpu().numpy(),
            "affiliate_idx": constraints["affiliate_idx"][0].cpu().numpy(),
        }


def _looks_like_gt_layout(array: np.ndarray) -> bool:
    if array.shape[1] < GT_CORE_COLUMNS:
        return False
    pmt = array[:, 3]
    affiliate = array[:, 14]
    return bool(
        np.isfinite(pmt).all()
        and np.isfinite(affiliate).all()
        and np.all((pmt >= 0) & (pmt <= 4) & (pmt == np.floor(pmt)))
        and np.all(affiliate == np.floor(affiliate))
    )


def build_output_array(
    input_array: np.ndarray,
    prediction: dict[str, np.ndarray],
    input_layout: str = "auto",
) -> np.ndarray:
    count = input_array.shape[0]
    for name in ("pmt", "mad", "dim", "nor", "loc", "affiliate_idx"):
        if len(prediction[name]) != count:
            raise ValueError(f"prediction {name} has {len(prediction[name])} rows, expected {count}")

    if input_layout == "gt" or (
        input_layout == "auto" and _looks_like_gt_layout(input_array)
    ):
        suffix = input_array[:, GT_CORE_COLUMNS:]
    elif input_layout in ("auto", "raw"):
        suffix = input_array[:, 3:]
    else:
        raise ValueError(f"unsupported input_layout: {input_layout}")

    core = np.concatenate(
        [
            input_array[:, :3],
            np.asarray(prediction["pmt"]).reshape(count, 1),
            np.asarray(prediction["mad"]).reshape(count, 3),
            np.asarray(prediction["dim"]).reshape(count, 1),
            np.asarray(prediction["nor"]).reshape(count, 3),
            np.asarray(prediction["loc"]).reshape(count, 3),
            np.asarray(prediction["affiliate_idx"]).reshape(count, 1),
        ],
        axis=1,
    )
    return np.concatenate([core, suffix], axis=1)


def load_point_file(path: Path) -> tuple[np.ndarray, str]:
    if path.suffix.lower() == ".npy":
        array = np.load(path, allow_pickle=False)
        delimiter = " "
    else:
        delimiter = " "
        try:
            array = np.loadtxt(path, dtype=np.float64)
        except ValueError as whitespace_error:
            try:
                array = np.loadtxt(path, dtype=np.float64, delimiter=",")
                delimiter = ","
            except ValueError:
                raise whitespace_error
    array = np.asarray(array)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[1] < 3:
        raise ValueError(f"expected a 2D point array with at least 3 columns: {path}")
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
            if path.suffix.lower() == ".npy":
                np.save(temporary, array)
            else:
                formats = ["%.9g"] * array.shape[1]
                formats[3] = "%d"
                formats[14] = "%d"
                np.savetxt(temporary, array, fmt=formats, delimiter=delimiter)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def parse_extensions(value: str) -> set[str]:
    extensions = set()
    for item in value.split(","):
        item = item.strip().lower()
        if not item:
            continue
        extensions.add(item if item.startswith(".") else f".{item}")
    if not extensions:
        raise ValueError("at least one file extension is required")
    return extensions


def iter_point_files(root: Path, extensions: set[str]) -> Iterable[Path]:
    return (
        path for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix.lower() in extensions
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
    extensions = parse_extensions(args.extensions)
    files = list(iter_point_files(input_dir, extensions))
    if not files:
        raise FileNotFoundError(
            f"no files with extensions {sorted(extensions)} found below {input_dir}"
        )

    # Reproduce the whole directory tree, including empty directories. Files
    # outside the selected point-cloud extensions are intentionally not copied.
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
        stage1_mode=args.stage1_mode,
        cluster_bandwidth=args.cluster_bandwidth,
        normal_k=args.normal_k,
    )
    print(
        "Stage 1 predictor: "
        f"checkpoint={predictor.checkpoint_path}; model={predictor.model_name}; "
        f"mode={predictor.stage1_mode}; device={device}; "
        f"cluster_bandwidth={predictor.cluster_bandwidth}"
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
        output_array = build_output_array(
            input_array, prediction, input_layout=args.input_layout
        )
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
