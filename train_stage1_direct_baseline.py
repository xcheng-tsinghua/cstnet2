"""Train an XYZ-only Stage 1 model that directly predicts four constraints."""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch

from data_utils.stage1_dataset import Stage1ConstraintDataset
from functional.stage1_direct_loss import DEFAULT_DIRECT_LOSS_WEIGHTS
from functional.stage1_direct_trainer import Stage1DirectTrainer
from functional.wandb_utils import (
    initialize_wandb_run,
    read_wandb_run_id_from_checkpoint,
)
from networks.stage1_direct_baselines import (
    DIRECT_BASELINE_MODEL_NAMES,
    build_stage1_direct_baseline,
    stage1_direct_model_config,
)


DEFAULT_DATA_ROOT = "/opt/data/private/data_set/pcd_cstnet2/abc_pcd"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an independent XYZ-only Stage 1 direct baseline"
    )
    parser.add_argument(
        "--model", default="attn3dgcn", choices=DIRECT_BASELINE_MODEL_NAMES
    )
    parser.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--val_data_root",
        default="",
        help="Optional held-out validation directory; strongly recommended for comparisons",
    )
    parser.add_argument("--n_points", type=int, default=2048)
    parser.add_argument("--bs", "--batch_size", dest="bs", type=int, default=30)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--epoch", "--epochs", dest="epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--scheduler_step", type=int, default=20)
    parser.add_argument("--scheduler_gamma", type=float, default=0.9)
    parser.add_argument("--gradient_clip_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--use_amp", action="store_true", default=False)
    parser.add_argument("--is_sample", action="store_true", default=False)
    parser.add_argument(
        "--device", default="auto", help="auto, cpu, cuda, or a concrete CUDA device"
    )

    parser.add_argument("--feature_dim", type=int, default=128)
    parser.add_argument("--head_hidden_dim", type=int, default=128)
    parser.add_argument("--head_dropout", type=float, default=0.2)
    parser.add_argument("--dgcnn_k", type=int, default=20)
    parser.add_argument("--attn_neighbors", type=int, default=20)
    parser.add_argument("--attn_k", type=int, default=16)
    parser.add_argument("--pointtransformer_k", type=int, default=16)
    parser.add_argument("--pointtransformer_width", type=int, default=64)
    parser.add_argument("--pointtransformer_depth", type=int, default=3)
    parser.add_argument("--pointmamba_tokens", type=int, default=128)
    parser.add_argument("--pointmamba_group_size", type=int, default=32)
    parser.add_argument("--pointmamba_width", type=int, default=64)
    parser.add_argument("--pointmamba_depth", type=int, default=2)
    parser.add_argument("--pointnext_k", type=int, default=24)
    parser.add_argument("--pointmlp_group_size", type=int, default=24)

    parser.add_argument("--w_pmt", type=float, default=DEFAULT_DIRECT_LOSS_WEIGHTS["w_pmt"])
    parser.add_argument("--w_mad", type=float, default=DEFAULT_DIRECT_LOSS_WEIGHTS["w_mad"])
    parser.add_argument("--w_dim", type=float, default=DEFAULT_DIRECT_LOSS_WEIGHTS["w_dim"])
    parser.add_argument("--w_loc", type=float, default=DEFAULT_DIRECT_LOSS_WEIGHTS["w_loc"])

    parser.add_argument(
        "--output_root",
        default=os.path.join("model_trained", "stage1_direct_baseline"),
    )
    parser.add_argument(
        "--resume",
        default="",
        help="Checkpoint path, or 'auto' to use the selected run's last.pth",
    )
    parser.add_argument("--wandb_project", default="cstnet2")
    parser.add_argument("--wandb_entity", default="")
    parser.add_argument("--wandb_run_name", default="")
    args = parser.parse_args(argv)

    if args.n_points <= 0 or args.bs <= 0 or args.workers < 0 or args.epochs <= 0:
        parser.error("n_points, bs, and epochs must be positive; workers must be non-negative")
    if args.lr <= 0 or args.weight_decay < 0:
        parser.error("lr must be positive and weight_decay must be non-negative")
    if args.scheduler_step <= 0 or not 0.0 < args.scheduler_gamma <= 1.0:
        parser.error("scheduler_step must be positive and scheduler_gamma must be in (0, 1]")
    if args.gradient_clip_norm <= 0:
        parser.error("gradient_clip_norm must be positive")
    stage1_direct_model_config(args)
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    normalized = str(name).strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(normalized)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {name}")
    return device


def resolve_output_and_resume(
    args: argparse.Namespace,
) -> tuple[Path, str]:
    output_dir = (
        Path(args.output_root).expanduser()
        / args.model
        / f"seed_{int(args.seed)}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    resume = str(args.resume or "").strip()
    if resume.lower() == "auto":
        candidate = output_dir / "last.pth"
        if not candidate.is_file():
            raise FileNotFoundError(f"auto-resume checkpoint not found: {candidate}")
        resume = str(candidate)
    elif resume:
        candidate = Path(resume).expanduser()
        if not candidate.is_file():
            raise FileNotFoundError(f"resume checkpoint not found: {candidate}")
        resume = str(candidate)
    return output_dir, resume


def main(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    device = resolve_device(args.device)
    model_config = stage1_direct_model_config(args)
    output_dir, resume_checkpoint = resolve_output_and_resume(args)
    train_loader = Stage1ConstraintDataset.create_dataloader(
        root=args.data_root,
        bs=args.bs,
        n_points=args.n_points,
        num_workers=args.workers,
        shuffle=True,
        is_sample=args.is_sample,
        sample_seed=args.seed,
    )
    val_loader = None
    if args.val_data_root:
        val_loader = Stage1ConstraintDataset.create_dataloader(
            root=args.val_data_root,
            bs=args.bs,
            n_points=args.n_points,
            num_workers=args.workers,
            shuffle=False,
            is_sample=args.is_sample,
            sample_seed=args.seed + 1,
        )

    model = build_stage1_direct_baseline(model_config).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma
    )
    loss_weights = {
        "w_pmt": args.w_pmt,
        "w_mad": args.w_mad,
        "w_dim": args.w_dim,
        "w_loc": args.w_loc,
    }
    checkpoint_args = {
        **vars(args),
        "model_config": model_config,
        "parameter_count": parameter_count,
        "device": str(device),
        "train_file_count": len(train_loader.dataset),
        "val_file_count": len(val_loader.dataset) if val_loader is not None else 0,
    }
    run_name = args.wandb_run_name or f"stage1_direct_{args.model}_seed{args.seed}"
    wandb_resume_id = read_wandb_run_id_from_checkpoint(resume_checkpoint)
    if resume_checkpoint and not wandb_resume_id:
        print(
            "WARNING: resume checkpoint has no wandb_run_id; a new WandB Run will be created"
        )
    run = initialize_wandb_run(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        run_id=wandb_resume_id,
        config=checkpoint_args,
    )
    print(
        f"Stage 1 direct baseline: model={args.model}, device={device}, "
        f"parameters={parameter_count:,}, output={output_dir}"
    )
    trainer = Stage1DirectTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=output_dir,
        device=device,
        epochs=args.epochs,
        loss_weights=loss_weights,
        gradient_clip_norm=args.gradient_clip_norm,
        use_amp=args.use_amp,
        checkpoint_args=checkpoint_args,
        wandb_run=run,
    )
    try:
        return trainer.fit(resume_checkpoint=resume_checkpoint or None)
    finally:
        run.finish()


if __name__ == "__main__":
    main(parse_args())
