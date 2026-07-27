"""Evaluate one Stage 1 checkpoint over every TXT sample below a directory."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime

import torch
from colorama import Fore, init

from data_utils.datasets import CstNet2Dataset
from functional.cst_pred_evaluator import CstPredEvaluator
from functional.cst_pred_trainer import load_model_state_with_diagnostics
from functional.point_features import stage1_feature_dim
from networks.cst_pred_wrapper import CstPredWrapper


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        required=True,
        help="directory recursively containing every Stage 1 evaluation .txt sample",
    )
    parser.add_argument("--checkpoint", required=True, help="full Stage 1 checkpoint")
    parser.add_argument("--bs", type=int, default=16)
    parser.add_argument(
        "--n_points",
        type=int,
        default=None,
        help="points per sample; defaults to the checkpoint training value",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="deterministic per-file point sampling seed",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    parser.add_argument(
        "--model",
        choices=["auto", "pointnet2", "pointnet", "attn_3dgcn"],
        default="auto",
        help="defaults to the checkpoint model",
    )
    parser.add_argument(
        "--cluster_bandwidth",
        type=float,
        default=None,
        help="defaults to the checkpoint value",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="",
        help="evaluation report path; defaults to log/eval_cst_pred_<time>.json",
    )
    return parser.parse_args(argv)


def _load_checkpoint(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError("Stage 1 evaluation requires a dictionary checkpoint")
    missing = sorted({"model", "args"} - set(checkpoint))
    if missing:
        raise ValueError(
            f"Stage 1 evaluation checkpoint is missing fields: {missing}"
        )
    if not isinstance(checkpoint["args"], dict):
        raise ValueError("checkpoint args must be a dictionary")
    return checkpoint


def _resolve_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable")
    return torch.device(name)


def _checkpoint_value(checkpoint_args, name, default):
    value = checkpoint_args.get(name, default)
    return default if value is None else value


def main(args):
    checkpoint = _load_checkpoint(args.checkpoint)
    checkpoint_args = checkpoint["args"]
    model_name = (
        args.model
        if args.model != "auto"
        else _checkpoint_value(checkpoint_args, "model", "")
    )
    if model_name not in ("pointnet2", "pointnet", "attn_3dgcn"):
        raise ValueError(
            "checkpoint does not contain a supported model name; pass --model explicitly"
        )

    n_points = (
        args.n_points
        if args.n_points is not None
        else int(_checkpoint_value(checkpoint_args, "n_points", 2000))
    )
    use_extra_features = bool(
        _checkpoint_value(checkpoint_args, "use_extra_features", False)
    )
    feature_k = int(_checkpoint_value(checkpoint_args, "feature_k", 16))
    cluster_bandwidth = (
        float(args.cluster_bandwidth)
        if args.cluster_bandwidth is not None
        else float(_checkpoint_value(checkpoint_args, "cluster_bandwidth", 0.35))
    )
    train_phase = str(_checkpoint_value(checkpoint_args, "train_phase", "joint"))
    geom_start_epoch = int(
        _checkpoint_value(checkpoint_args, "geom_start_epoch", 20)
    )
    geom_ramp_epochs = int(
        _checkpoint_value(checkpoint_args, "geom_ramp_epochs", 20)
    )
    loss_weights = {
        name: float(_checkpoint_value(checkpoint_args, name, default))
        for name, default in {
            "w_pmt": 1.0,
            "w_cluster": 0.5,
            "w_mad": 0.02,
            "w_dim": 0.05,
            "w_loc": 0.02,
            "w_geom": 0.02,
            "w_inst": 0.005,
        }.items()
    }
    enabled_losses = {
        name: bool(
            _checkpoint_value(checkpoint_args, f"enable_{name}_loss", True)
        )
        for name in ("mad", "dim", "loc", "geom", "inst")
    }

    device = _resolve_device(args.device)
    model = CstPredWrapper(
        model_name,
        channel_fea=stage1_feature_dim(use_extra_features),
    ).to(device)
    load_model_state_with_diagnostics(
        model,
        checkpoint["model"],
        require_complete=True,
        source=args.checkpoint,
    )
    data_loader = CstNet2Dataset.create_directory_dataloader(
        root=args.data_root,
        bs=args.bs,
        n_points=n_points,
        num_workers=args.workers,
        shuffle=False,
        is_sample=False,
        sample_seed=args.seed,
    )
    evaluator = CstPredEvaluator(
        model,
        data_loader,
        loss_weights=loss_weights,
        train_phase=train_phase,
        enabled_losses=enabled_losses,
        geom_start_epoch=geom_start_epoch,
        geom_ramp_epochs=geom_ramp_epochs,
        use_extra_features=use_extra_features,
        feature_k=feature_k,
        cluster_bandwidth=cluster_bandwidth,
    )
    checkpoint_epoch = int(checkpoint.get("epoch", 0))
    loss_summary, metric_summary = evaluator.evaluate(checkpoint_epoch)

    output_json = args.output_json
    if not output_json:
        os.makedirs("log", exist_ok=True)
        output_json = os.path.join(
            "log",
            f"eval_cst_pred_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json",
        )
    output_json = os.path.abspath(output_json)
    output_dir = os.path.dirname(output_json)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    report = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "checkpoint_epoch": checkpoint_epoch,
        "data_root": os.path.abspath(args.data_root),
        "dataset_file_count": len(data_loader.dataset),
        "model": model_name,
        "device": str(device),
        "n_points": n_points,
        "use_extra_features": use_extra_features,
        "feature_k": feature_k,
        "cluster_bandwidth": cluster_bandwidth,
        "sampling_seed": args.seed,
        "loss": loss_summary,
        "metrics": metric_summary,
    }
    with open(output_json, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    print(Fore.GREEN + f"evaluation report saved to: {output_json}")
    return report


if __name__ == "__main__":
    init(autoreset=True)
    main(parse_args())
