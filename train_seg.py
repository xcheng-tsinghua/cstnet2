"""Train Stage 2 MFCAD++ point segmentation models."""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from data_utils.mfcad_seg_dataset import DEFAULT_LABEL_MAP, MFCADSegmentationDataset
from functional.cuda_runtime import preload_cuda_nvrtc
from functional.segmentation_loss import compute_training_class_statistics
from functional.stage2_seg_trainer import Stage2SegmentationTrainer
from functional.wandb_utils import initialize_wandb_run
from networks.segmentation_models import (
    DEFAULT_SEGMENTATION_MODEL,
    SEGMENTATION_MODEL_NAMES,
    build_segmentation_model,
    segmentation_model_config,
)


MODEL_OUTPUT_ROOT = Path("model_trained/seg")


def segmentation_run_name(model_config: dict[str, object]) -> str:
    model_name = str(model_config["model"])
    if model_name != DEFAULT_SEGMENTATION_MODEL and bool(
        model_config["baseline_use_constraints"]
    ):
        model_name += "_constraints"
    return model_name


def resolve_training_paths(
    model_config: dict[str, object],
    resume: bool,
) -> tuple[Path, Path | None]:
    output_dir = MODEL_OUTPUT_ROOT / segmentation_run_name(model_config)
    if not resume:
        return output_dir, None
    checkpoint_path = output_dir / "last.pth"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"resume=true but no checkpoint was found: {checkpoint_path.resolve()}"
        )
    return output_dir, checkpoint_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 2 MFCAD++ point segmentation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=r"/opt/data/private/data_set/pcd_cstnet2/mfcad_pcd",
        help="MFCAD++ root containing the train/val/test split directories",
    )
    parser.add_argument(
        "--label_map", type=str, default=str(DEFAULT_LABEL_MAP),
        help="JSON label metadata used by the dataset and checkpoints",
    )
    parser.add_argument(
        "--class_statistics_path",
        type=str,
        default=str(MODEL_OUTPUT_ROOT / "class_statistics.json"),
        help="Shared training-only class-statistics cache used by all model variants",
    )
    parser.add_argument(
        "--model",
        choices=SEGMENTATION_MODEL_NAMES,
        default=DEFAULT_SEGMENTATION_MODEL,
        help="Segmentation architecture used for training",
    )
    parser.add_argument(
        "--baseline_use_constraints",
        action="store_true",
        default=False,
        help="Concatenate the full 15D constraint vector with XYZ for baseline models",
    )
    parser.add_argument("--batch_size", type=int, default=20, help="Samples per batch")
    parser.add_argument("--n_points", type=int, default=2048, help="Points sampled per part")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader worker processes")
    parser.add_argument("--epochs", type=int, default=200, help="Total training epochs")
    parser.add_argument("--warmup_epochs", type=int, default=5, help="Linear LR warmup epochs")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Peak learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="AdamW weight decay")
    parser.add_argument(
        "--gradient_clip_norm", type=float, default=1.0,
        help="Maximum global gradient norm",
    )
    parser.add_argument(
        "--feature_dim", type=int, default=64,
        help="Initial per-stream width for the constraint_aware model",
    )
    parser.add_argument(
        "--norm_type", choices=("ln", "bn"), default="ln",
        help="Constraint-stream normalization",
    )
    parser.add_argument("--seed", type=int, default=2026, help="Base random seed")
    parser.add_argument(
        "--use_amp", action="store_true", default=False,
        help="Disable CUDA automatic mixed precision",
    )
    parser.add_argument(
        "--use_npy_cache", action="store_true", default=False,
        help="Read and write parsed point-cloud NPY caches",
    )
    parser.add_argument(
        "--not_resume",
        action="store_true",
        default=False,
        help="Resume from model_trained/seg/<model_name>/last.pth",
    )
    parser.add_argument(
        "--recompute_class_statistics", action="store_true", default=False,
        help="Ignore cached class statistics and scan the training split again",
    )
    parser.add_argument(
        "--wandb_project", type=str, default="cstnet2", help="WandB project name",
    )
    parser.add_argument(
        "--wandb_entity", type=str, default="", help="Optional WandB entity/team",
    )
    parser.add_argument(
        "--wandb_run_name", type=str, default="stage2_mfcad_seg", help="WandB run name",
    )
    return parser.parse_args(argv)


def log_training_configuration(
    args: argparse.Namespace,
    device: torch.device,
    output_dir: Path,
) -> None:
    print("Stage 2 segmentation configuration:")
    for name, value in sorted(vars(args).items()):
        print(f"  {name}={value!r}")
    print(f"  device={device}")
    print(f"  output_dir={output_dir}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def distributed_context() -> tuple[bool, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if distributed:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    return distributed, local_rank, device


def main(args: argparse.Namespace) -> None:
    distributed, local_rank, device = distributed_context()
    rank = dist.get_rank() if distributed else 0
    set_seed(args.seed + rank)
    model_config = segmentation_model_config(args)
    nvrtc_library = None
    if (
        device.type == "cuda"
        and model_config["model"] == DEFAULT_SEGMENTATION_MODEL
    ):
        nvrtc_library = preload_cuda_nvrtc()
    output_dir, resume_checkpoint = resolve_training_paths(model_config, not args.not_resume)
    if rank == 0:
        log_training_configuration(args, device, output_dir)
        if nvrtc_library is not None:
            print(f"CUDA NVRTC: preloaded {nvrtc_library}")

    train_loader, val_loader, test_loader = MFCADSegmentationDataset.create_dataloaders(
        root=args.data_root,
        batch_size=args.batch_size,
        n_points=args.n_points,
        num_workers=args.workers,
        label_map_path=args.label_map,
        use_npy_cache=args.use_npy_cache,
        distributed=distributed,
    )
    statistics_path = Path(args.class_statistics_path)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        compute_training_class_statistics(
            train_loader.dataset,
            statistics_path,
            force=args.recompute_class_statistics,
        )
    if distributed:
        dist.barrier()
    class_weights, _ = compute_training_class_statistics(
        train_loader.dataset,
        statistics_path,
        force=False,
    )

    model = build_segmentation_model(
        num_classes=train_loader.dataset.num_classes,
        config=model_config,
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if rank == 0:
        print(f"segmentation model: {model_config}; parameters={parameter_count:,}")
    if distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
        )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    wandb_run = None
    if rank == 0:
        wandb_run = initialize_wandb_run(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            config={
                **vars(args),
                "model_config": model_config,
                "parameter_count": parameter_count,
                "num_classes": train_loader.dataset.num_classes,
                "class_weights": class_weights.tolist(),
                "label_map": train_loader.dataset.label_map,
                "device": str(device),
                "world_size": dist.get_world_size() if distributed else 1,
            },
        )

    trainer = Stage2SegmentationTrainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=test_loader,
        class_weights=class_weights,
        label_map=train_loader.dataset.label_map,
        output_dir=output_dir,
        device=device,
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        gradient_clip_norm=args.gradient_clip_norm,
        use_amp=args.use_amp,
        checkpoint_args=vars(args),
        wandb_run=wandb_run,
    )
    try:
        trainer.fit(resume_checkpoint=resume_checkpoint)
    finally:
        if wandb_run is not None:
            wandb_run.finish()
        if distributed:
            dist.destroy_process_group()


if __name__ == "__main__":
    main(parse_args())

