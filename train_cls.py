"""
Train Stage 2 constraint-aware classification.

Stage 1 is loaded as a frozen constraint extractor by default. Ground-truth
constraints are still available as an oracle/debug option.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from typing import Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm

from data_utils.datasets import CstNet2Dataset
from functional.constraints import ground_truth_constraints_to_tensor
from networks.stage1_extractor import FrozenStage1ConstraintExtractor
from networks.stage2 import CstNetStage2Classifier
from networks.utils import all_metric_cls

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover - optional dependency
    try:
        from tensorboardX import SummaryWriter
    except ImportError:  # pragma: no cover - optional dependency
        SummaryWriter = None


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return value.lower() in {"true", "1", "yes", "y"}


def parse_args():
    parser = argparse.ArgumentParser("Stage 2 classification training")
    parser.add_argument("--bs", "--batch_size", dest="bs", type=int, default=32)
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--decay_rate", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--n_points", "--n_point", dest="n_points", type=int, default=2000)
    parser.add_argument("--is_sample", default="False", choices=["True", "False"])
    parser.add_argument("--local", default="False", choices=["True", "False"])
    parser.add_argument("--root_sever", type=str, default=r"data/pcd_cstnet2/Param20K_Extend")
    parser.add_argument("--root_local", type=str, default=r"D:\document\DeepLearning\DataSet\pcd_cstnet2\Param20K_Extend")

    parser.add_argument("--constraint_source", choices=["stage1", "gt"], default="stage1")
    parser.add_argument("--stage1_model", choices=["pointnet2", "pointnet", "attn_3dgcn"], default="pointnet2")
    parser.add_argument("--stage1_ckpt", type=str, default=r"model_trained/pointnet2_pmt_prim_cluster.pth")
    parser.add_argument("--stage1_cluster_bandwidth", type=float, default=0.35)
    parser.add_argument("--stage1_use_gt_normals", default="False", choices=["True", "False"])

    parser.add_argument("--save_name", type=str, default="stage2_cls")
    parser.add_argument("--resume", default="False", choices=["True", "False"])
    return parser.parse_args()


def build_constraints(
    data,
    device: torch.device,
    source: str,
    stage1: Optional[FrozenStage1ConstraintExtractor],
    use_gt_normals_for_stage1: bool,
) -> torch.Tensor:
    xyz = data[0].float().to(device, non_blocking=True)
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
    return stage1(xyz, normals=normals)


def evaluate(model, loader, device, args, stage1):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    with torch.no_grad():
        for data in tqdm(loader, total=len(loader), desc="eval"):
            xyz = data[0].float().to(device, non_blocking=True)
            target = data[1].long().to(device, non_blocking=True)
            constraints = build_constraints(
                data,
                device,
                args.constraint_source,
                stage1,
                str2bool(args.stage1_use_gt_normals),
            )
            pred = model(xyz, constraints)
            total_loss += F.nll_loss(pred, target).item()
            all_preds.append(pred.cpu().numpy())
            all_labels.append(target.cpu().numpy())
    metrics = all_metric_cls(all_preds, all_labels)
    return total_loss / max(len(loader), 1), metrics


def main(args):
    os.makedirs("log", exist_ok=True)
    os.makedirs("model_trained", exist_ok=True)

    data_root = args.root_local if str2bool(args.local) else args.root_sever
    train_loader, test_loader = CstNet2Dataset.create_dataloader(
        root=data_root,
        bs=args.bs,
        n_points=args.n_points,
        num_workers=args.workers,
        is_sample=str2bool(args.is_sample),
    )
    n_classes = train_loader.dataset.n_classes()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stage1 = None
    if args.constraint_source == "stage1":
        stage1 = FrozenStage1ConstraintExtractor(
            model_name=args.stage1_model,
            checkpoint=args.stage1_ckpt,
            cluster_bandwidth=args.stage1_cluster_bandwidth,
        ).to(device)
        stage1.freeze()

    model = CstNetStage2Classifier(n_classes=n_classes).to(device)
    save_path = os.path.join("model_trained", f"{args.save_name}.pth")
    if str2bool(args.resume) and os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device))
        print(f"resume Stage 2 classifier from {save_path}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=args.decay_rate,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

    writer = SummaryWriter(log_dir=os.path.join("log", args.save_name + "_" + datetime.now().strftime("%Y%m%d_%H%M%S"))) if SummaryWriter else None
    best_acc = 0.0

    for epoch in range(args.epoch):
        model.train()
        all_preds, all_labels = [], []
        running_loss = 0.0
        for data in tqdm(train_loader, total=len(train_loader), desc=f"train {epoch + 1}/{args.epoch}"):
            xyz = data[0].float().to(device, non_blocking=True)
            target = data[1].long().to(device, non_blocking=True)
            constraints = build_constraints(
                data,
                device,
                args.constraint_source,
                stage1,
                str2bool(args.stage1_use_gt_normals),
            )

            optimizer.zero_grad(set_to_none=True)
            pred = model(xyz, constraints)
            loss = F.nll_loss(pred, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            all_preds.append(pred.detach().cpu().numpy())
            all_labels.append(target.detach().cpu().numpy())

        scheduler.step()
        train_metrics = all_metric_cls(all_preds, all_labels)
        test_loss, test_metrics = evaluate(model, test_loader, device, args, stage1)
        train_loss = running_loss / max(len(train_loader), 1)

        if writer is not None:
            writer.add_scalar("loss/train", train_loss, epoch)
            writer.add_scalar("loss/test", test_loss, epoch)
            for prefix, metrics in (("train", train_metrics), ("test", test_metrics)):
                writer.add_scalar(f"{prefix}/ins_acc", metrics[0], epoch)
                writer.add_scalar(f"{prefix}/class_acc", metrics[1], epoch)
                writer.add_scalar(f"{prefix}/f1_macro", metrics[2], epoch)
                writer.add_scalar(f"{prefix}/f1_weighted", metrics[3], epoch)
                writer.add_scalar(f"{prefix}/map", metrics[4], epoch)

        torch.save(model.state_dict(), save_path)
        if test_metrics[0] >= best_acc:
            best_acc = test_metrics[0]
            torch.save(model.state_dict(), os.path.join("model_trained", f"{args.save_name}_best.pth"))

        print(
            f"epoch {epoch + 1}/{args.epoch} "
            f"train_loss={train_loss:.6f} test_loss={test_loss:.6f} "
            f"train_acc={train_metrics[0]:.4f} test_acc={test_metrics[0]:.4f} best={best_acc:.4f}"
        )

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main(parse_args())
