"""Reproducible training/evaluation for cached-constraint Stage 2 ablations."""
from __future__ import annotations

import csv
import copy
from functools import wraps
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler
from tqdm import tqdm

from train_cls import classification_loss, constraints_from_dataset_batch
from networks.classification_models import classification_model_config
from functional.wandb_utils import (
    initialize_wandb_run, flatten_wandb_summary_metrics, wandb_confusion_matrix, wandb_run_id,
)
from functional.stage2_ablation_config import EXPERIMENTS
from networks.stage2_ablation import build_ablation_model
from networks.utils import all_metric_cls


def optional_disk_output(function):
    """Output failures must not interrupt training; other errors still propagate."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        try:
            result = function(*args, **kwargs)
            return True if result is None else result
        except (OSError, RuntimeError) as error:
            # PyTorch's zip writer reports disk failures as RuntimeError.
            if isinstance(error, RuntimeError) and not any(marker in str(error) for marker in (
                "PytorchStreamWriter failed", "unexpected pos", "could not be opened")):
                raise
            print(f"[disk warning] {function.__name__} failed for {args[0]}: {error}; "
                  "skipping this output and continuing.", flush=True)
            return False
    return wrapped


@optional_disk_output
def prepare_run_directory(directory):
    directory.mkdir(parents=True, exist_ok=True)


@optional_disk_output
def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class SeededSubset(Dataset):
    """Point sampling independent of model RNG consumption and worker count."""
    def __init__(self, dataset, indices, seed):
        self.dataset, self.indices, self.seed = dataset, list(indices), int(seed)
        self.epoch = 0

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        state = np.random.get_state()
        np.random.seed((self.seed + self.epoch * 1000003 + self.indices[index]) % (2**32))
        try:
            return self.dataset[self.indices[index]]
        finally:
            np.random.set_state(state)


def dataset_manifest(dataset):
    """Sample identities + storage metadata; records changes without hashing GBs."""
    paths = ([(int(label), str(path)) for label, path in dataset.datapath]
             if hasattr(dataset, "datapath") else [str(p) for p in dataset.files])
    store = getattr(dataset, "_h5_store", None)
    root = Path(dataset.root)
    if store is not None:
        files = sorted(p for p in ([root] if root.is_file() else root.rglob("*"))
                       if p.is_file() and p.suffix.lower() in (".h5", ".hdf5", ".json"))
    else:
        files = [Path(item[1] if isinstance(item, tuple) else item) for item in paths]
    storage = [(str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns) for p in files]
    return {"samples": paths, "storage_sha256": hashlib.sha256(
        json.dumps(storage, sort_keys=True).encode()).hexdigest()}


def build_datasets(args, seed):
    # Match train_cls.py's dataset factory, but require an existing split.
    from data_utils.classification_dataset import Stage2ClassificationDataset
    from data_utils.stage2_h5 import resolve_storage_format
    root = Path(args.data_root)
    storage = resolve_storage_format(root, args.data_format)
    if storage == "txt" and not (
        ((root / "train").is_dir() and (root / "test").is_dir())
        or any((root / name).is_file() for name in ("split_file.json", "split_file"))
    ):
        raise FileNotFoundError("existing train/test directories or split_file.json required; ablations do not create splits")
    train_loader, test_loader = Stage2ClassificationDataset.create_dataloaders(
        root=args.data_root, bs=args.batch_size, n_points=args.n_points,
        num_workers=args.workers, is_sample=False, storage_format=storage)
    train_base, test_base = train_loader.dataset, test_loader.dataset
    metadata = {"classes": train_base.classes, "train_source": dataset_manifest(train_base),
                "test_source": dataset_manifest(test_base)}
    train = SeededSubset(train_base, range(len(train_base)), seed)
    test = SeededSubset(test_base, range(len(test_base)), seed)
    return train, test, train_base.n_classes(), metadata


def make_loader(dataset, args, training, epoch=0):
    dataset.epoch = epoch if training else 0
    generator = torch.Generator().manual_seed(dataset.seed + epoch)
    sampler = None
    if args.is_sample:
        sampler = RandomSampler(dataset, replacement=False,
            num_samples=min(len(dataset), args.batch_size * (4 if training else 2)), generator=generator)
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=training and sampler is None,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=args.device != "cpu" and torch.cuda.is_available(),
        generator=generator, persistent_workers=False,
        drop_last=False)


def capture_rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def run_epoch(model, loader, device, args, optimizer=None):
    training = optimizer is not None
    model.train(training)
    predictions, labels = [], []
    loss_sum, batch_count = 0.0, 0
    gradient_norms = []
    with torch.set_grad_enabled(training):
        for raw in tqdm(loader, total=len(loader), desc="train" if training else "eval"):
            xyz = raw[0].float().to(device)
            target = raw[1].long().to(device)
            constraints = constraints_from_dataset_batch(raw, device)
            if training:
                scores, loss = classification_loss(model, xyz, constraints, target, args,
                                                   classification_model_config(args))
            else:
                scores = model(xyz, constraints)
                loss = F.nll_loss(scores, target)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("non-finite ablation loss")
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True))
                gradient_norms.append(norm)
                optimizer.step()
            predictions.append(scores.detach().cpu().numpy())
            labels.append(target.cpu().numpy())
            loss_sum += loss.detach().item()
            batch_count += 1
    if not batch_count:
        raise ValueError("empty data loader")
    result = all_metric_cls(predictions, labels)
    result["loss"] = loss_sum / batch_count
    if training:
        result["optimization/gradient_norm_mean"] = float(np.mean(gradient_norms))
        result["optimization/gradient_norm_max"] = float(max(gradient_norms))
    return result


def evaluate(model, dataset, device, args):
    # Fix test point/FPS sampling without consuming the training RNG stream.
    rng = capture_rng_state()
    try:
        seed_everything(dataset.seed)
        return run_epoch(model, make_loader(dataset, args, False), device, args)
    finally:
        restore_rng_state(rng)


def load_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=False)


@optional_disk_output
def save_checkpoint(path, payload):
    # A failed save is reported to W&B but does not stop training.
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def run_directory(args, experiment, seed):
    return Path(args.output_dir) / args.task / args.model / experiment / f"seed_{seed}"


def protocol_config(args, experiment, seed, metadata):
    ignored = {"mode", "experiments", "seed", "resume", "output_dir", "workers", "device",
               "list", "dry_run", "wandb_project", "wandb_entity", "wandb_run_name"}
    arguments = {k: v for k, v in vars(args).items() if k not in ignored}
    arguments["data_root"] = str(Path(args.data_root).resolve())
    return {"version": 2, "args": arguments, "experiment": experiment, "seed": seed,
            "intervention": EXPERIMENTS[experiment].to_dict(), "dataset": metadata}


@optional_disk_output
def clear_run_outputs(directory):
    """Remove this run's generated files before a fresh training run."""
    names = {"config.json", "parameters.json", "last.pth", "best.pth",
             "result.json", "evaluation_test.json"}
    for path in directory.iterdir():
        epoch_log = (path.suffix == ".json" and path.stem.startswith("epoch_")
                     and path.stem[6:].isdigit())
        if path.is_file() and (path.name in names or epoch_log):
            path.unlink()


def run_experiment(args, experiment, seed):
    directory = run_directory(args, experiment, seed)
    prepare_run_directory(directory)
    last_path, best_path = directory / "last.pth", directory / "best.pth"
    if args.mode == "evaluate":
        saved = load_checkpoint(best_path)
        # Reconstruct training configuration instead of silently applying CLI defaults.
        from argparse import Namespace
        saved_args = dict(vars(args))
        saved_args.update(saved["protocol"]["args"])
        for key in ("mode", "workers", "device", "output_dir"):
            saved_args[key] = getattr(args, key)
        args = Namespace(**saved_args)
    seed_everything(seed)
    train, test, count, metadata = build_datasets(args, seed)
    protocol = protocol_config(args, experiment, seed, metadata)
    device = torch.device(args.device if args.device != "auto" else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        from functional.cuda_runtime import preload_cuda_nvrtc
        preload_cuda_nvrtc()
    model = build_ablation_model(args.task, count, experiment, args.model, config=classification_model_config(args)).to(device)
    parameters = {"total": sum(p.numel() for p in model.parameters()),
                  "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)}
    if args.mode == "evaluate":
        if saved["protocol"] != protocol:
            raise ValueError("evaluation data or experiment differs from the saved protocol")
        model.load_state_dict(saved["model"], strict=True)
        result = evaluate(model, test, device, args)
        write_json(directory / "evaluation_test.json",
                   {"epoch": saved["epoch"], "metrics": result})
        print(json.dumps({k: v for k, v in result.items() if isinstance(v, (int, float))},
                         ensure_ascii=False))
        return result

    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad),
                                 lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    start, best, best_epoch = 0, 0.0, -1
    resume_id = ""
    best_metrics = None
    if args.resume:
        saved = load_checkpoint(last_path)
        if saved["protocol"] != protocol:
            raise ValueError("resume protocol mismatch (data, split, experiment, seed or hyperparameters)")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        start, best, best_epoch = saved["epoch"] + 1, saved["best"], saved["best_epoch"]
        best_metrics = saved.get("best_metrics")
        if best_metrics is None:
            # Compatibility with checkpoints written before metrics were retained.
            best_metrics = load_checkpoint(best_path)["test"]
        restore_rng_state(saved["rng"])
        resume_id = str(saved.get("wandb_run_id", "") or "")
    primary = "instance_accuracy"
    print(f"{directory}: parameters={parameters}; train={len(train)}, test={len(test)}")
    class_names = [name for name, _ in sorted(metadata["classes"].items(), key=lambda item: item[1])]
    run_name = f"{args.wandb_run_name or args.save_name}_{experiment}_seed_{seed}"
    rng_before_wandb = capture_rng_state()
    wandb_run = initialize_wandb_run(
        project=args.wandb_project, entity=args.wandb_entity, name=run_name, run_id=resume_id,
        config={**vars(args), "experiment": experiment, "seed": seed,
                "constraint_components": list(EXPERIMENTS[experiment].components),
                "model_config": classification_model_config(args), "parameter_count": parameters["total"],
                "num_classes": count, "class_names": class_names,
                "constraint_storage": "point_file_fields", "device": str(device)},
    )
    restore_rng_state(rng_before_wandb)
    try:
        if not args.resume:
            # Reset only after data/model/W&B initialization succeeds. In
            # particular, remove old higher-epoch logs and completed results
            # so a shorter or interrupted new run cannot expose stale output.
            clear_run_outputs(directory)
            write_json(directory / "config.json", protocol)
        write_json(directory / "parameters.json", parameters)
        for epoch in range(start, args.epochs):
            epoch_learning_rate = float(optimizer.param_groups[0]["lr"])
            train_metrics = run_epoch(model, make_loader(train, args, True, epoch), device, args,
                                     optimizer=optimizer)
            test_metrics = evaluate(model, test, device, args)
            scheduler.step()
            improved = test_metrics[primary] >= best
            if improved:
                best, best_epoch = test_metrics[primary], epoch
                best_metrics = copy.deepcopy(test_metrics)
            payload = {"protocol": protocol, "epoch": epoch, "model": model.state_dict(),
                       "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                       "rng": capture_rng_state(), "best": best, "best_epoch": best_epoch,
                       "test": test_metrics, "parameters": parameters,
                       "wandb_run_id": wandb_run_id(wandb_run),
                       "model_config": classification_model_config(args), "args": vars(args),
                       "best_acc": best, "best_metrics": best_metrics}
            best_saved = save_checkpoint(best_path, payload) if improved else False
            last_saved = save_checkpoint(last_path, payload)
            write_json(directory / f"epoch_{epoch + 1:04d}.json",
                       {"epoch": epoch + 1, "train": train_metrics, "test": test_metrics})
            log = {
                "epoch": epoch + 1, "learning_rate": epoch_learning_rate,
                "loss/train": train_metrics["loss"], "loss/test": test_metrics["loss"],
                "best/test_instance_accuracy": best,
                "checkpoint/last_saved": int(last_saved), "checkpoint/best_saved": int(best_saved),
                "train/optimization/gradient_norm_mean": train_metrics["optimization/gradient_norm_mean"],
                "train/optimization/gradient_norm_max": train_metrics["optimization/gradient_norm_max"],
            }
            for split, metrics in (("train", train_metrics), ("test", test_metrics)):
                metric_values = {k: v for k, v in metrics.items() if k != "loss" and not k.startswith("optimization/")}
                log.update(flatten_wandb_summary_metrics(f"{split}/metric", metric_values))
                log[f"{split}/confusion_matrix"] = wandb_confusion_matrix(
                    metrics["confusion_matrix"], class_names,
                    title=f"{split.title()} Classification Confusion Matrix")
            wandb_run.log(log, step=epoch)
            print(f"[{experiment} seed={seed}] {epoch + 1}/{args.epochs} "
                  f"loss={train_metrics['loss']:.6f} test/{primary}={test_metrics[primary]:.6f}", flush=True)
        # Use the metrics already evaluated at the best epoch. Never reload a
        # missing or stale checkpoint after a failed disk write.
        if best_metrics is None:
            raise ValueError("no best metrics available; training requires at least one epoch")
        result = {"task": args.task, "model": args.model,
                  "experiment": experiment, "seed": seed, "best_epoch": best_epoch + 1,
                  "test": best_metrics, "parameters": parameters,
                  "protocol": protocol}
        write_json(directory / "result.json", result)
        return result
    finally:
        wandb_run.finish()


@optional_disk_output
def summarize(output_dir, seed=42):
    """Export one row per completed experiment at the selected seed."""
    rows = []
    for path in sorted(Path(output_dir).rglob("result.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        if result["task"] != "cls" or result["seed"] != seed or result["experiment"] not in EXPERIMENTS:
            continue
        protocol = dict(result["protocol"])
        for key in ("experiment", "intervention"):
            protocol.pop(key)
        fingerprint = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()[:16]
        row = {k: result[k] for k in ("task", "model", "experiment", "seed", "best_epoch")}
        row["protocol_id"] = fingerprint
        row.update({k: v for k, v in result["test"].items() if isinstance(v, (float, int))})
        row.update({f"parameters_{k}": v for k, v in result["parameters"].items()})
        rows.append(row)
    if not rows:
        print(f"No completed ablation results found for seed={seed}.")
        return []
    with (Path(output_dir) / "summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({k for row in rows for k in row}))
        writer.writeheader()
        writer.writerows(rows)
    write_json(Path(output_dir) / "summary.json", rows)
    print(f"Summarized {len(rows)} experiments at seed={seed} in {Path(output_dir).resolve()}")
    return rows
