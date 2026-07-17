from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - progress bars are optional
    def tqdm(iterable, **_kwargs):
        return iterable

from data_utils.mfcad_seg_dataset import DEFAULT_LABEL_MAP, MFCADSegmentationDataset
from functional.segmentation_loss import WeightedSegmentationLoss
from functional.segmentation_metrics import SegmentationMetrics
from networks.segmentation_models import build_segmentation_model, segmentation_model_config
from tools.visualize_mfcad_seg import export_segmentation_sample


def load_full_checkpoint(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Evaluate Stage 2 MFCAD++ point segmentation")
    parser.add_argument("checkpoint")
    parser.add_argument(
        "--data_root",
        default=r"D:\document\DeepLearning\DataSet\pcd_cstnet2\mfcad_pcd",
    )
    parser.add_argument("--label_map", default=str(DEFAULT_LABEL_MAP))
    parser.add_argument("--split", choices=("val", "validation", "test"), default="test")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--n_points", type=int, default=2048)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output_json", default="")
    parser.add_argument("--prediction_dir", default="")
    parser.add_argument("--max_exports", type=int, default=20)
    return parser.parse_args()


@torch.no_grad()
def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = load_full_checkpoint(args.checkpoint)
    required = {"model", "args", "label_map", "class_weights"}
    missing = sorted(required.difference(checkpoint))
    if missing:
        raise ValueError(f"incomplete segmentation checkpoint; missing: {missing}")

    dataset = MFCADSegmentationDataset(
        root=args.data_root,
        split=args.split,
        n_points=args.n_points,
        label_map_path=args.label_map,
    )
    if dataset.label_map != checkpoint["label_map"]:
        raise ValueError("checkpoint and evaluation dataset use different label maps")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )

    saved_args = checkpoint.get("args", {})
    model_config = checkpoint.get("model_config") or segmentation_model_config(saved_args)
    model_config = segmentation_model_config(model_config)
    model = build_segmentation_model(
        num_classes=dataset.num_classes,
        config=model_config,
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    print(f"evaluation model: {model_config}")
    criterion = WeightedSegmentationLoss(checkpoint["class_weights"]).to(device)
    metrics = SegmentationMetrics(dataset.num_classes)
    total_loss = 0.0
    export_count = 0

    for raw_batch in tqdm(loader, desc=f"evaluate {dataset.split}"):
        xyz = raw_batch["xyz"].to(device, non_blocking=True)
        constraints = raw_batch["constraints"].to(device, non_blocking=True)
        masks = raw_batch["constraint_masks"].to(device, non_blocking=True)
        labels = raw_batch["labels"].to(device, non_blocking=True)
        face_ids = raw_batch["face_ids"].to(device, non_blocking=True)
        logits = model(xyz, constraints, masks)
        total_loss += float(criterion(logits, labels).item())
        metrics.update(logits, labels, face_ids)

        if args.prediction_dir and export_count < args.max_exports:
            probabilities = logits.softmax(dim=-1)
            confidence, predictions = probabilities.max(dim=-1)
            for batch_id in range(xyz.shape[0]):
                if export_count >= args.max_exports:
                    break
                export_segmentation_sample(
                    output_dir=args.prediction_dir,
                    sample_id=raw_batch["sample_id"][batch_id],
                    xyz=xyz[batch_id].cpu().numpy(),
                    gt_label=labels[batch_id].cpu().numpy(),
                    pred_label=predictions[batch_id].cpu().numpy(),
                    face_id=face_ids[batch_id].cpu().numpy(),
                    confidence=confidence[batch_id].cpu().numpy(),
                    label_map=dataset.label_map,
                )
                export_count += 1

    result = {
        "split": dataset.split,
        "num_samples": len(dataset),
        "loss": total_loss / max(len(loader), 1),
        **metrics.compute(),
    }
    output_json = Path(args.output_json) if args.output_json else Path(args.checkpoint).with_name(
        f"{dataset.split}_metrics.json"
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main(parse_args())
