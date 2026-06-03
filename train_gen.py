"""
Train Stage 2 constraint-aware point cloud diffusion generation.

The default workflow follows AGENTS.md: add diffusion noise, pass the noisy
point cloud through frozen Stage 1, and train only the Stage 2 denoiser.
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm

from data_utils.datasets import CstNet2Dataset
from functional.constraints import ground_truth_constraints_to_tensor
from networks.stage1_extractor import FrozenStage1ConstraintExtractor
from networks.stage2 import CstNetStage2Diffusion


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return value.lower() in {"true", "1", "yes", "y"}


def parse_args():
    parser = argparse.ArgumentParser("Stage 2 diffusion training")
    parser.add_argument("--bs", type=int, default=16)
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--decay_rate", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--n_points", type=int, default=2000)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--beta_start", type=float, default=1e-4)
    parser.add_argument("--beta_end", type=float, default=0.02)
    parser.add_argument("--is_sample", default="False", choices=["True", "False"])
    parser.add_argument("--local", default="False", choices=["True", "False"])
    parser.add_argument("--root_sever", type=str, default=r"data/pcd_cstnet2/Param20K_Extend")
    parser.add_argument("--root_local", type=str, default=r"D:\document\DeepLearning\DataSet\pcd_cstnet2\Param20K_Extend")

    parser.add_argument("--constraint_source", choices=["stage1", "gt"], default="stage1")
    parser.add_argument("--stage1_model", choices=["pointnet2", "pointnet", "attn_3dgcn"], default="pointnet2")
    parser.add_argument("--stage1_ckpt", type=str, default=r"model_trained/pointnet2_pmt_prim_cluster.pth")
    parser.add_argument("--stage1_cluster_bandwidth", type=float, default=0.35)
    parser.add_argument("--stage1_use_gt_normals", default="False", choices=["True", "False"])

    parser.add_argument("--save_name", type=str, default="stage2_diffusion")
    parser.add_argument("--resume", default="False", choices=["True", "False"])
    return parser.parse_args()


def make_alpha_cumprod(timesteps: int, beta_start: float, beta_end: float, device: torch.device) -> torch.Tensor:
    betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
    alphas = 1.0 - betas
    return torch.cumprod(alphas, dim=0)


def q_sample(clean_xyz: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor, alpha_cumprod: torch.Tensor) -> torch.Tensor:
    alpha_bar = alpha_cumprod[timesteps].view(-1, 1, 1)
    return alpha_bar.sqrt() * clean_xyz + (1.0 - alpha_bar).sqrt() * noise


def build_constraints(
    data,
    noisy_xyz: torch.Tensor,
    device: torch.device,
    source: str,
    stage1: Optional[FrozenStage1ConstraintExtractor],
    use_gt_normals_for_stage1: bool,
) -> torch.Tensor:
    if source == "gt":
        return ground_truth_constraints_to_tensor(
            pmt=data[2].to(device, non_blocking=True),
            direction=data[3].float().to(device, non_blocking=True),
            dimension=data[4].float().to(device, non_blocking=True),
            continuity=data[5].float().to(device, non_blocking=True),
            location=data[6].float().to(device, non_blocking=True),
        )
    if stage1 is None:
        raise RuntimeError("Stage 1 extractor is required when constraint_source='stage1'")
    normals = data[5].float().to(device, non_blocking=True) if use_gt_normals_for_stage1 else None
    return stage1(noisy_xyz, normals=normals)


def process_epoch(model, loader, device, args, stage1, alpha_cumprod, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    losses = []
    desc = "train" if is_train else "eval"
    for data in tqdm(loader, total=len(loader), desc=desc):
        clean_xyz = data[0].float().to(device, non_blocking=True)
        bsz = clean_xyz.shape[0]
        t = torch.randint(0, args.timesteps, (bsz,), device=device)
        noise = torch.randn_like(clean_xyz)
        noisy_xyz = q_sample(clean_xyz, t, noise, alpha_cumprod)
        constraints = build_constraints(
            data,
            noisy_xyz,
            device,
            args.constraint_source,
            stage1,
            str2bool(args.stage1_use_gt_normals),
        )

        with torch.set_grad_enabled(is_train):
            pred_noise = model(noisy_xyz, constraints, t)
            loss = F.mse_loss(pred_noise, noise)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
        losses.append(loss.detach())

    return torch.stack(losses).mean().item() if losses else 0.0


def main(args):
    os.makedirs("model_trained", exist_ok=True)

    data_root = args.root_local if str2bool(args.local) else args.root_sever
    train_loader, test_loader = CstNet2Dataset.create_dataloader(
        root=data_root,
        bs=args.bs,
        n_points=args.n_points,
        num_workers=args.workers,
        is_sample=str2bool(args.is_sample),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    alpha_cumprod = make_alpha_cumprod(args.timesteps, args.beta_start, args.beta_end, device)

    stage1 = None
    if args.constraint_source == "stage1":
        stage1 = FrozenStage1ConstraintExtractor(
            model_name=args.stage1_model,
            checkpoint=args.stage1_ckpt,
            cluster_bandwidth=args.stage1_cluster_bandwidth,
        ).to(device)
        stage1.freeze()

    model = CstNetStage2Diffusion().to(device)
    save_path = os.path.join("model_trained", f"{args.save_name}.pth")
    if str2bool(args.resume) and os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device))
        print(f"resume Stage 2 diffusion model from {save_path}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=args.decay_rate,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epoch, 1))
    best_loss = float("inf")

    for epoch in range(args.epoch):
        train_loss = process_epoch(model, train_loader, device, args, stage1, alpha_cumprod, optimizer)
        with torch.no_grad():
            test_loss = process_epoch(model, test_loader, device, args, stage1, alpha_cumprod, optimizer=None)
        scheduler.step()

        torch.save(model.state_dict(), save_path)
        if test_loss <= best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), os.path.join("model_trained", f"{args.save_name}_best.pth"))

        print(
            f"epoch {epoch + 1}/{args.epoch} "
            f"train_noise_mse={train_loss:.6f} test_noise_mse={test_loss:.6f} best={best_loss:.6f}"
        )


if __name__ == "__main__":
    main(parse_args())
