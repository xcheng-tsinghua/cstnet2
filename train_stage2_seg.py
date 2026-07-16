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
from functional.segmentation_loss import compute_training_class_statistics
from functional.stage2_seg_trainer import Stage2SegmentationTrainer
from networks.stage2_segmentation import Stage2SegmentationModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Stage 2 MFCAD++ point segmentation")
    parser.add_argument(
        "--data_root",
        default=r"D:\document\DeepLearning\DataSet\pcd_cstnet2\mfcad_pcd",
    )
    parser.add_argument("--label_map", default=str(DEFAULT_LABEL_MAP))
    parser.add_argument("--output_dir", default="model_trained/stage2_mfcad_seg")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--n_points", type=int, default=2048)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--gradient_clip_norm", type=float, default=1.0)
    parser.add_argument("--feature_dim", type=int, default=64)
    parser.add_argument("--norm_type", choices=("ln", "bn"), default="ln")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--use_npy_cache", action="store_true")
    parser.add_argument("--resume", default="")
    parser.add_argument("--recompute_class_statistics", action="store_true")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", default="cstnet2")
    parser.add_argument("--wandb_run_name", default="stage2_mfcad_seg")
    return parser.parse_args()


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

    train_loader, val_loader, _ = MFCADSegmentationDataset.create_dataloaders(
        root=args.data_root,
        batch_size=args.batch_size,
        n_points=args.n_points,
        num_workers=args.workers,
        label_map_path=args.label_map,
        use_npy_cache=args.use_npy_cache,
        distributed=distributed,
    )
    output_dir = Path(args.output_dir)
    statistics_path = output_dir / "class_statistics.json"
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

    model = Stage2SegmentationModel(
        num_classes=train_loader.dataset.num_classes,
        feature_dim=args.feature_dim,
        norm_type=args.norm_type,
    ).to(device)
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
    if args.use_wandb and rank == 0:
        try:
            import wandb
        except ImportError:
            print("wandb is not installed; training continues without WandB")
        else:
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config=vars(args),
            )

    trainer = Stage2SegmentationTrainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        class_weights=class_weights,
        label_map=train_loader.dataset.label_map,
        output_dir=output_dir,
        device=device,
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        gradient_clip_norm=args.gradient_clip_norm,
        use_amp=not args.no_amp,
        checkpoint_args=vars(args),
        wandb_run=wandb_run,
    )
    trainer.fit(resume_checkpoint=args.resume or None)
    if wandb_run is not None:
        wandb_run.finish()
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main(parse_args())

