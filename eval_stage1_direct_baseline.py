"""Evaluate a trained XYZ-only Stage 1 direct baseline checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from data_utils.stage1_dataset import Stage1ConstraintDataset
from functional.stage1_direct_evaluator import Stage1DirectEvaluator
from functional.stage1_direct_trainer import (
    DIRECT_CHECKPOINT_TASK,
    load_direct_checkpoint,
)
from networks.stage1_direct_baselines import build_stage1_direct_baseline


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Stage 1 direct baseline")
    parser.add_argument("checkpoint")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--n_points", type=int, default=None)
    parser.add_argument("--bs", "--batch_size", dest="bs", type=int, default=30)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--use_amp", action="store_true", default=False)
    parser.add_argument("--is_sample", action="store_true", default=False)
    parser.add_argument("--output_json", default="")
    return parser.parse_args(argv)


def _resolve_device(name: str) -> torch.device:
    if str(name).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def main(args: argparse.Namespace) -> dict:
    checkpoint = load_direct_checkpoint(args.checkpoint)
    if checkpoint.get("task") != DIRECT_CHECKPOINT_TASK:
        raise ValueError(
            f"checkpoint task is {checkpoint.get('task')!r}, "
            f"expected {DIRECT_CHECKPOINT_TASK!r}"
        )
    model_config = checkpoint.get("model_config")
    if not isinstance(model_config, dict):
        raise ValueError("Stage 1 direct checkpoint is missing model_config")
    checkpoint_args = checkpoint.get("args", {})
    n_points = (
        int(args.n_points)
        if args.n_points is not None
        else int(checkpoint_args.get("n_points", 2048))
    )
    device = _resolve_device(args.device)
    model = build_stage1_direct_baseline(model_config).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    loader = Stage1ConstraintDataset.create_dataloader(
        root=args.data_root,
        bs=args.bs,
        n_points=n_points,
        num_workers=args.workers,
        shuffle=False,
        is_sample=args.is_sample,
        sample_seed=int(checkpoint_args.get("seed", 2026)) + 2,
    )
    evaluator = Stage1DirectEvaluator(
        model=model,
        data_loader=loader,
        device=device,
        loss_weights=checkpoint.get("loss_weights", {}),
        use_amp=args.use_amp,
    )
    summary = evaluator.evaluate()
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.output_json:
        evaluator.save(summary, Path(args.output_json).expanduser())
        print(f"saved evaluation summary: {Path(args.output_json).expanduser()}")
    return summary


if __name__ == "__main__":
    main(parse_args())
