"""Train constraint-aware or baseline Stage 2 point-cloud classifiers."""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from colorama import init

from data_utils.datasets import CstNet2Dataset
from functional.constraints import ground_truth_constraints_to_tensor
from functional.cuda_runtime import preload_cuda_nvrtc
from functional.wandb_utils import flatten_wandb_metrics, initialize_wandb_run
from networks.classification_models import (
    CLASSIFICATION_MODEL_NAMES,
    DEFAULT_CLASSIFICATION_MODEL,
    build_classification_model,
    classification_model_config,
    classification_model_uses_constraints,
    classification_run_name,
)
from networks.utils import all_metric_cls


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "y", "on"}:
        return True
    if normalized in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean value, got {value!r}")


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser("Stage 2 classification training")
    parser.add_argument("--bs", "--batch_size", dest="bs", type=int, default=32)
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--decay_rate", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--n_points", "--n_point", dest="n_points", type=int, default=2000)
    parser.add_argument("--is_sample", default="False", choices=["True", "False"])
    parser.add_argument("--local", default="False", choices=["True", "False"])
    parser.add_argument("--root_sever", type=str, default=r"/opt/data/private/data_set/pcd_cstnet2/Param20K_pcd")
    parser.add_argument("--root_local", type=str, default=r"D:\document\DeepLearning\DataSet\pcd_cstnet2\Param20K_Extend")

    parser.add_argument(
        "--model",
        choices=CLASSIFICATION_MODEL_NAMES,
        default=DEFAULT_CLASSIFICATION_MODEL,
        help="Stage 2 classification architecture",
    )
    parser.add_argument(
        "--baseline_use_constraints",
        action="store_true",
        default=False,
        help="Use the 15D per-point constraint as baseline point features",
    )
    parser.add_argument(
        "--stage2_variant",
        choices=["baseline", "discriminative", "token_fusion"],
        default="baseline",
        help="baseline keeps the existing classifier; the other two use the new Stage2 classification heads.",
    )
    parser.add_argument("--stage2_norm", choices=["ln", "bn"], default="ln")
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--aux_loss_weight", type=float, default=0.1)
    parser.add_argument("--token_dim", type=int, default=256)
    parser.add_argument("--transformer_layers", type=int, default=3)
    parser.add_argument("--transformer_heads", type=int, default=8)
    parser.add_argument("--token_dropout", type=float, default=0.1)
    parser.add_argument("--stream_dropout", type=float, default=0.1)
    parser.add_argument("--use_stats_token", default="False", choices=["True", "False"])

    parser.add_argument("--save_name", type=str, default="stage2_cls")
    parser.add_argument("--wandb_project", type=str, default="cstnet2")
    parser.add_argument("--wandb_entity", type=str, default="")
    parser.add_argument("--wandb_run_name", type=str, default="")
    parser.add_argument(
        "--resume",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Load the selected model's existing weight file before training",
    )
    return parser.parse_args(argv)


def constraints_from_dataset_batch(data, device: torch.device) -> torch.Tensor:
    """Assemble constraints already stored in the selected point-cloud files."""
    return ground_truth_constraints_to_tensor(
        pmt=data[2].to(device, non_blocking=True),
        direction=data[3].float().to(device, non_blocking=True),
        dimension=data[4].float().to(device, non_blocking=True),
        continuity=data[5].float().to(device, non_blocking=True),
        location=data[6].float().to(device, non_blocking=True),
    )


def build_stage2_classifier(args, n_classes: int) -> torch.nn.Module:
    return build_classification_model(n_classes, classification_model_config(args))


def classification_loss(model, xyz, constraints, target, args, model_config):
    variant = model_config.get("stage2_variant", "baseline")
    use_aux = (
        model_config["model"] == DEFAULT_CLASSIFICATION_MODEL
        and variant != "baseline"
        and args.aux_loss_weight > 0.0
    )
    if not use_aux:
        log_probs = model(xyz, constraints)
        if args.label_smoothing > 0.0:
            loss = F.cross_entropy(log_probs, target, label_smoothing=args.label_smoothing)
        else:
            loss = F.nll_loss(log_probs, target)
        return log_probs, loss

    output = model(xyz, constraints, return_aux=True)
    loss = F.cross_entropy(
        output["main_logits"],
        target,
        label_smoothing=args.label_smoothing,
    )

    if variant == "discriminative":
        aux_keys = ("aux_component_logits", "aux_constraint_logits")
    else:
        aux_keys = ("aux_final_constraint_logits", "aux_component_token_logits")

    for key in aux_keys:
        loss = loss + args.aux_loss_weight * F.cross_entropy(
            output[key],
            target,
            label_smoothing=args.label_smoothing,
        )

    return output["log_probs"], loss


def evaluate(model, loader, device, use_constraints):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    with torch.no_grad():
        for data in tqdm(loader, total=len(loader), desc="eval"):
            xyz = data[0].float().to(device, non_blocking=True)
            target = data[1].long().to(device, non_blocking=True)
            constraints = None
            if use_constraints:
                constraints = constraints_from_dataset_batch(data, device)
            pred = model(xyz, constraints)
            total_loss += F.nll_loss(pred, target).item()
            all_preds.append(pred.cpu().numpy())
            all_labels.append(target.cpu().numpy())
    metrics = all_metric_cls(all_preds, all_labels)
    return total_loss / max(len(loader), 1), metrics


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
    n_classes = train_loader.dataset.n_classes()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_config = classification_model_config(args)
    use_constraints = classification_model_uses_constraints(model_config)

    nvrtc_library = None
    if (
        device.type == "cuda"
        and model_config["model"] == DEFAULT_CLASSIFICATION_MODEL
    ):
        nvrtc_library = preload_cuda_nvrtc()

    model = build_classification_model(n_classes, model_config).to(device)
    save_stem = args.save_name
    if args.save_name == "stage2_cls":
        save_stem = f"{args.save_name}_{classification_run_name(model_config)}"
    save_path = os.path.join("model_trained", f"{save_stem}.pth")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"classification model: {model_config}; parameters={parameter_count:,}")
    print(f"constraints required by model: {use_constraints}")
    if nvrtc_library is not None:
        print(f"CUDA NVRTC: preloaded {nvrtc_library}")
    if args.resume and os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device))
        print(f"checkpoint: loaded model weights from {save_path}")
    elif args.resume:
        raise FileNotFoundError(
            f"resume=true but no classification weights were found: {save_path}"
        )
    else:
        print(f"checkpoint: none provided; starting a new training run ({save_path})")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=args.decay_rate,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

    wandb_run = initialize_wandb_run(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name or save_stem,
        config={
            **vars(args),
            "model_config": model_config,
            "parameter_count": parameter_count,
            "num_classes": n_classes,
            "data_root": data_root,
            "constraint_storage": "point_file_fields",
            "device": str(device),
        },
    )
    best_acc = 0.0

    for epoch in range(args.epoch):
        epoch_learning_rate = float(optimizer.param_groups[0]["lr"])
        model.train()
        all_preds, all_labels = [], []
        running_loss = 0.0
        gradient_norm_sum = 0.0
        gradient_norm_max = 0.0
        gradient_norm_count = 0
        for data in tqdm(
            train_loader,
            total=len(train_loader),
            desc=f"train {epoch + 1}/{args.epoch}",
        ):
            xyz = data[0].float().to(device, non_blocking=True)
            target = data[1].long().to(device, non_blocking=True)
            constraints = None
            if use_constraints:
                constraints = constraints_from_dataset_batch(data, device)

            optimizer.zero_grad(set_to_none=True)
            pred, loss = classification_loss(
                model,
                xyz,
                constraints,
                target,
                args,
                model_config,
            )
            loss.backward()
            gradient_norm = float(
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            )
            if np.isfinite(gradient_norm):
                gradient_norm_sum += gradient_norm
                gradient_norm_max = max(gradient_norm_max, gradient_norm)
                gradient_norm_count += 1
            optimizer.step()

            running_loss += loss.item()
            all_preds.append(pred.detach().cpu().numpy())
            all_labels.append(target.detach().cpu().numpy())

        train_metrics = all_metric_cls(all_preds, all_labels)
        test_loss, test_metrics = evaluate(
            model,
            test_loader,
            device,
            use_constraints,
        )
        train_loss = running_loss / max(len(train_loader), 1)

        torch.save(model.state_dict(), save_path)
        if test_metrics["instance_accuracy"] >= best_acc:
            best_acc = test_metrics["instance_accuracy"]
            torch.save(model.state_dict(), os.path.join("model_trained", f"{save_stem}_best.pth"))

        wandb_payload = {
            "epoch": epoch + 1,
            "learning_rate": epoch_learning_rate,
            "loss/train": train_loss,
            "loss/test": test_loss,
            "best/test_instance_accuracy": best_acc,
            "train/optimization/gradient_norm_mean": (
                gradient_norm_sum / max(gradient_norm_count, 1)
            ),
            "train/optimization/gradient_norm_max": gradient_norm_max,
        }
        wandb_payload.update(flatten_wandb_metrics("train/metric", train_metrics))
        wandb_payload.update(flatten_wandb_metrics("test/metric", test_metrics))
        wandb_run.log(wandb_payload, step=epoch)

        print(
            f"epoch {epoch + 1}/{args.epoch} "
            f"train_loss={train_loss:.6f} test_loss={test_loss:.6f} "
            f"train_acc={train_metrics['instance_accuracy']:.4f} "
            f"test_acc={test_metrics['instance_accuracy']:.4f} best={best_acc:.4f}"
        )
        scheduler.step()

    wandb_run.finish()


if __name__ == "__main__":
    init(autoreset=True)
    main(parse_args())
