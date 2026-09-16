from __future__ import annotations

import math
import os
import sys
import warnings
from time import time

import torch
from colorama import Back, Fore
from tqdm import tqdm

from functional.direct_constraints import CONSTRAINT_ROUTE, direct_constraints, validate_direct_checkpoint
from functional.stage1_phase_loss import (
    LOSS_NAMES, TRAINING_RECIPE, stage1_active_losses, stage1_phase_loss,
)
from functional.point_features import stage1_forward
from functional.finite_checks import assert_finite_tensors
from functional.stage1_metrics import (
    CONSTRAINT_ATTRIBUTE_ACCUMULATOR_KEYS,
    aggregate_constraint_attribute_metrics,
    evaluate_constraint_attribute_metrics,
    evaluate_primitive_metrics,
    primitive_metrics_from_confusion,
    primitive_prediction_collapsed,
)
from functional.checkpoint_io import safe_torch_save, safe_json_save
from functional.console_io import ResilientTextStream, safe_print as print
from functional.wandb_utils import (
    flatten_wandb_summary_metrics,
    wandb_confusion_matrix,
    wandb_run_id,
)


PRIMITIVE_CLASS_NAMES = ("plane", "cylinder", "cone", "sphere", "other")
BEST_FILE_NAMES = {
    "pmt_miou": "best_pmt_miou.pth",
    "constraint_score": "best_constraint_score.pth",
}
RESUME_WARNING_ONLY_CONFIG_KEYS = frozenset({"point_count"})


class CstPredTrainer(object):
    """Train the multitask Stage 1 model without coupling it to Stage 2."""

    def __init__(
        self,
        model,
        train_loader,
        checkpoint_dir,
        log_savepth,
        max_epoch,
        lr,
        save_str,
        wandb_run=None,
        decay_rate=1e-4,
        loss_weights=None,
        use_extra_features=False,
        feature_k=16,
        overfit_one_batch=False,
        grad_clip=1.0,
        train_phase="semantic",
        checkpoint_action="scratch",
        checkpoint_source="",
        checkpoint_args=None,
    ):
        super().__init__()
        if checkpoint_action not in ("scratch", "resume", "init"):
            raise ValueError(f"unsupported checkpoint action: {checkpoint_action}")
        if checkpoint_action in ("resume", "init") and not checkpoint_source:
            raise ValueError(
                f"checkpoint action {checkpoint_action!r} requires a checkpoint source"
            )
        self.model = model
        self.device = next(self.model.parameters()).device
        self.train_loader = train_loader
        self.log_savepth = log_savepth
        self.max_epoch = int(max_epoch)
        self.save_str = save_str
        self.wandb_run = wandb_run
        self.decay_rate = decay_rate
        self.loss_weights = {} if loss_weights is None else dict(loss_weights)
        self.use_extra_features = bool(use_extra_features)
        self.feature_k = int(feature_k)
        self.overfit_one_batch = overfit_one_batch
        self.overfit_batch = None
        self.grad_clip = None if grad_clip is None else float(grad_clip)
        if self.grad_clip is not None and (
            not math.isfinite(self.grad_clip) or self.grad_clip < 0
        ):
            raise ValueError("grad_clip must be finite and non-negative")
        self.train_phase = train_phase
        self.checkpoint_action = checkpoint_action
        self.checkpoint_source = str(checkpoint_source) if checkpoint_source else ""
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_args = {} if checkpoint_args is None else dict(checkpoint_args)
        self.checkpoint_args.update(
            constraint_route=CONSTRAINT_ROUTE, train_phase=train_phase,
            training_recipe=TRAINING_RECIPE,
            loc_input=self.model.loc_input,
            mad_input=self.model.mad_input,
            dim_input=self.model.dim_input,
            **{f"w_{name}": float(self.loss_weights.get(f"w_{name}", 1.0)) for name in LOSS_NAMES},
        )
        self.optimizer = None
        self.scheduler = None
        self.start_epoch = 0
        self.global_step = 0
        self.best_metrics = {
            "pmt_miou": {"value": float("-inf"), "epoch": -1},
            "constraint_score": {"value": float("-inf"), "epoch": -1},
        }
        self.save_dict_train = self._new_save_dict()

        if not self.checkpoint_dir:
            raise ValueError("checkpoint_dir must be provided")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        print(f"checkpoints save to: {self.checkpoint_dir}, log save to: {self.log_savepth}")
        self._initialize_training(float(lr))

    @staticmethod
    def _new_save_dict():
        return {
            "loss": [],
            "metrics": [],
            "prim_loss": [],
            "clus_loss": [],
            "prim_acc": [],
        }

    def _initialize_training(self, lr):
        resume_state = None
        if self.checkpoint_action == "resume":
            resume_state = self._load_checkpoint_file(self.checkpoint_source)
            self._validate_checkpoint_mode(resume_state, self.checkpoint_source)
            self._validate_full_checkpoint(resume_state, self.checkpoint_source)
            self._validate_resume_config(resume_state)
            self._load_model_state(
                resume_state["model"], require_complete=True, source=self.checkpoint_source
            )
        elif self.checkpoint_action == "init":
            init_state = self._load_checkpoint_file(self.checkpoint_source)
            self._validate_checkpoint_mode(init_state, self.checkpoint_source)
            self._load_model_state(
                self._initial_model_state(init_state),
                require_complete=True,
                source=self.checkpoint_source,
            )
        else:
            print(Fore.BLACK + Back.CYAN + "training Stage 1 from scratch")

        self._configure_train_phase()
        self.make_optimizer_and_schedule(lr)

        if resume_state is not None:
            self.optimizer.load_state_dict(resume_state["optimizer"])
            self.scheduler.load_state_dict(resume_state["scheduler"])
            self.start_epoch = int(resume_state["epoch"]) + 1
            self.global_step = int(resume_state["global_step"])
            self.best_metrics = _normalize_best_metrics(resume_state["best_metrics"])
            schedule = resume_state.get("loss_schedule", {})
            saved_schedule_epoch = int(schedule.get("global_epoch", resume_state["epoch"]))
            if saved_schedule_epoch != int(resume_state["epoch"]):
                raise ValueError(
                    "checkpoint loss schedule is inconsistent: "
                    f"epoch={resume_state['epoch']} loss_schedule.global_epoch={saved_schedule_epoch}"
                )
            print(
                Fore.WHITE
                + Back.CYAN
                + f"resumed from {self.checkpoint_source}: next_epoch={self.start_epoch}, "
                f"global_step={self.global_step}, lr={self.current_lrs()}"
            )
        elif self.checkpoint_action == "init":
            print(
                Fore.WHITE
                + Back.CYAN
                + f"initialized model only from {self.checkpoint_source}; optimizer is new"
            )

    def _initial_model_state(self, checkpoint):
        """Reuse semantic weights, expanding only missing XYZ input columns."""
        state = dict(_extract_model_state(checkpoint))
        current_state = self.model.state_dict()
        for head in ("mad_head", "dim_head", "loc_head"):
            key = f"{head}.linear_layers.0.weight"
            current = current_state[key]
            incoming = state.get(key)
            if torch.is_tensor(incoming) and incoming.shape != current.shape:
                if not (self.train_phase == "geometry"
                        and checkpoint.get("args", {}).get("train_phase") == "semantic"
                        and incoming.shape == (current.shape[0], current.shape[1] - 3, 1)):
                    raise RuntimeError(
                        f"{head} input is incompatible with backbone + XYZ; "
                        "restart geometry from a semantic checkpoint, then retrain joint"
                    )
                # Semantic training never updated attribute heads. Preserve their
                # existing weights, with fresh initialization only for new columns.
                expanded = current.clone()
                expanded[:, :-3, :] = incoming
                state[key] = expanded
                print(f"Initializing geometry from legacy semantic weights: "
                      f"expanded {head} input with 3 new XYZ channels; optimizer starts fresh")
        return state

    @staticmethod
    def _load_checkpoint_file(path):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"checkpoint not found: {path}")
        return torch.load(path, map_location="cpu")

    @staticmethod
    def _validate_full_checkpoint(state, source):
        required = {
            "epoch", "global_step", "model", "optimizer", "scheduler",
            "best_metrics", "args",
        }
        if not isinstance(state, dict):
            raise ValueError(f"resume checkpoint must be a dict: {source}")
        missing = sorted(required - set(state.keys()))
        if missing:
            raise ValueError(
                f"resume checkpoint is incomplete ({source}); missing fields: {missing}. "
                "Automatic resume requires a complete Stage 1 training checkpoint."
            )

    @staticmethod
    def _validate_checkpoint_mode(state, source):
        if not isinstance(state, dict):
            raise ValueError(f"invalid checkpoint: {source}")
        validate_direct_checkpoint(state.get("args", {}), require_geometry=False)

    def _validate_resume_config(self, checkpoint):
        saved_config = checkpoint.get("checkpoint_config")
        if saved_config is None:
            saved_config = _critical_checkpoint_config(checkpoint.get("args", {}))
        else:
            saved_config = dict(saved_config)
            saved_config.pop("stage1_mode", None)
        current_config = _critical_checkpoint_config(self.checkpoint_args)
        differences = _config_differences(saved_config, current_config)
        warning_differences = [
            difference
            for difference in differences
            if difference[0] in RESUME_WARNING_ONLY_CONFIG_KEYS
        ]
        fatal_differences = [
            difference
            for difference in differences
            if difference[0] not in RESUME_WARNING_ONLY_CONFIG_KEYS
        ]
        for key, saved, current in warning_differences:
            warnings.warn(
                f"resume checkpoint {key} differs: checkpoint={saved!r}, "
                f"current={current!r}; continuing because this does not change "
                "checkpoint parameter shapes",
                RuntimeWarning,
                stacklevel=2,
            )
        if fatal_differences:
            print(Fore.RED + "resume checkpoint configuration mismatch:")
            for key, saved, current in fatal_differences:
                print(Fore.RED + f"  {key}: checkpoint={saved!r}, current={current!r}")
            raise ValueError("resume checkpoint configuration mismatch; use --checkpoint_policy restart to initialize from the previous phase with a fresh optimizer")
        if warning_differences:
            print(Fore.YELLOW + "resume checkpoint configuration: compatible with warnings")
        else:
            print(Fore.GREEN + "resume checkpoint configuration: exact match")

    def _load_model_state(self, incoming_state, require_complete, source):
        load_model_state_with_diagnostics(
            self.model,
            incoming_state,
            require_complete=require_complete,
            source=source,
        )

    def _configure_train_phase(self):
        if hasattr(self.model, "set_train_phase"):
            prefixes = self.model.set_train_phase(self.train_phase)
        else:
            raise TypeError("Stage 1 model must implement set_train_phase()")
        trainable = [(name, p) for name, p in self.model.named_parameters() if p.requires_grad]
        frozen_count = sum(p.numel() for p in self.model.parameters() if not p.requires_grad)
        trainable_count = sum(p.numel() for _, p in trainable)
        print(Fore.CYAN + f"train_phase={self.train_phase}")
        print(Fore.CYAN + f"trainable prefixes: {prefixes}")
        print(
            Fore.CYAN
            + f"trainable parameters: {trainable_count:,}; frozen parameters: {frozen_count:,}"
        )
        if not trainable:
            raise ValueError(f"train_phase={self.train_phase} has no trainable parameters")

    def make_optimizer_and_schedule(self, lr):
        named_trainable = [
            (name, param) for name, param in self.model.named_parameters() if param.requires_grad
        ]
        param_groups = [{
            "params": [p for _, p in named_trainable],
            "lr": lr,
            "group_name": self.train_phase,
        }]

        self.optimizer = torch.optim.Adam(
            param_groups,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=self.decay_rate,
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=20, gamma=0.9
        )
        print(Fore.CYAN + f"optimizer LR groups: {self.current_lrs()}")

    def current_lrs(self):
        return {
            group.get("group_name", str(index)): float(group["lr"])
            for index, group in enumerate(self.optimizer.param_groups)
        }

    def start(self):
        if self.start_epoch >= self.max_epoch:
            print(
                Fore.YELLOW
                + f"nothing to train: resume next_epoch={self.start_epoch} >= epoch={self.max_epoch}"
            )
            return

        for global_epoch in range(self.start_epoch, self.max_epoch):
            start_time = time()
            train_loss, train_metrics = self.process_epoch(global_epoch)
            # Higher is better, using the phase's active direct-prediction losses.
            train_metrics["constraint_score"] = -float(train_loss["loss_all"])
            self.append_save_dict(train_loss, train_metrics)
            train_time = time() - start_time
            print(Fore.BLUE + f"training time: {train_time:.4f} sec")
            improved = self._update_best_metrics(global_epoch, train_metrics)

            wandb_payload = {}
            wandb_payload.update(
                flatten_wandb_summary_metrics("loss", train_loss)
            )
            wandb_payload.update(
                flatten_wandb_summary_metrics("metric", train_metrics)
            )

            # The checkpoint contains the LR that will be used by the next epoch.
            self.scheduler.step()
            self.save(global_epoch, improved)
            if self.wandb_run is not None:
                try:
                    wandb_payload["confusion_matrix/primitive"] = (
                        wandb_confusion_matrix(
                            train_metrics["pmt_confusion_matrix"],
                            PRIMITIVE_CLASS_NAMES,
                            title="Train Primitive Confusion Matrix",
                        )
                    )
                    self.wandb_run.log(wandb_payload, step=global_epoch)
                except OSError as error:
                    print(f"WARNING: WandB logging I/O failed at epoch {global_epoch}: "
                          f"{error}; continuing training and retrying logging next epoch")

    def _update_best_metrics(self, epoch, train_metrics):
        candidates = {
            "pmt_miou": float(train_metrics.get("pmt_miou", float("-inf"))),
            "constraint_score": float(train_metrics.get("constraint_score", float("-inf"))),
        }
        improved = []
        for name, value in candidates.items():
            if value > float(self.best_metrics[name]["value"]):
                self.best_metrics[name] = {"value": value, "epoch": int(epoch)}
                improved.append(name)
        return improved

    def save(self, epoch, improved=None):
        improved = [] if improved is None else list(improved)
        payload = self._checkpoint_payload(epoch)
        last_path = os.path.join(self.checkpoint_dir, "last.pth")
        status = {"last": safe_torch_save(payload, last_path)}
        if status["last"]:
            print(Fore.GREEN + f"saved checkpoint: {last_path}")
        for metric_name in improved:
            best_path = os.path.join(self.checkpoint_dir, BEST_FILE_NAMES[metric_name])
            status[metric_name] = safe_torch_save(payload, best_path)
            if status[metric_name]:
                print(Fore.GREEN + f"saved best checkpoint: {best_path}")

        log_payload = {
            "run": {
                "start_epoch": self.start_epoch,
                "last_epoch": int(epoch),
                "global_step": self.global_step,
                "best_metrics": self.best_metrics,
                "best_metrics_source": "train",
                "checkpoint_dir": self.checkpoint_dir,
                "data_root": self.checkpoint_args.get("data_root", ""),
                "dataset_file_count": len(self.train_loader.dataset)
                if hasattr(self.train_loader, "dataset")
                else None,
            },
            "train": self.save_dict_train,
        }
        safe_json_save(log_payload, self.log_savepth)
        return status

    def _checkpoint_payload(self, epoch):
        checkpoint_config = _critical_checkpoint_config(self.checkpoint_args)
        payload = {
            "epoch": int(epoch),
            "global_step": int(self.global_step),
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "best_metrics": self.best_metrics,
            "best_metrics_source": "train",
            "args": self.checkpoint_args,
            "checkpoint_config": checkpoint_config,
            "wandb_run_id": wandb_run_id(self.wandb_run),
            "loss_schedule": {
                "global_epoch": int(epoch),
                "next_global_epoch": int(epoch) + 1,
                "global_step": int(self.global_step),
            },
        }
        return payload

    def append_save_dict(self, loss_summary, metric_summary):
        target = self.save_dict_train
        split = "train"
        target["loss"].append(loss_summary)
        target["metrics"].append(metric_summary)
        target["prim_loss"].append(float(loss_summary.get("raw/pmt", 0.0)))
        target["clus_loss"].append(float(loss_summary.get("raw/cluster", 0.0)))
        target["prim_acc"].append(float(metric_summary.get("pmt_acc", 0.0)))

        raw_text = ", ".join(
            f"{name}={float(loss_summary.get('raw/' + name, 0.0)):.5f}"
            for name in LOSS_NAMES if self._active_losses()[name]
        )
        weighted_text = ", ".join(
            f"{name}={float(loss_summary.get('weighted/' + name, 0.0)):.5f}"
            for name in LOSS_NAMES if self._active_losses()[name]
        )
        print(
            f"{split}: loss_all={loss_summary.get('loss_all', 0.0):.6f}; "
            f"raw[{raw_text}]; weighted[{weighted_text}]"
        )
        print(
            f"{split}: pmt_acc={metric_summary.get('pmt_acc', 0.0):.4f}, "
            f"pmt_macro_f1={metric_summary.get('pmt_macro_f1', 0.0):.4f}, "
            f"pmt_miou={metric_summary.get('pmt_miou', 0.0):.4f}"
        )
        if "direction_mean_angular_error_deg" in metric_summary:
            print(
                f"{split}: direction_angle_error="
                f"{metric_summary['direction_mean_angular_error_deg']:.4f} deg, "
                f"dimension_mae="
                f"{metric_summary['dimension_mean_absolute_error']:.6f}, "
                f"location_distance_error="
                f"{metric_summary['location_mean_distance_error']:.6f}"
            )
        print(f"{split}: gt primitive histogram={metric_summary.get('pmt_gt_histogram', [])}")
        print(f"{split}: predicted primitive histogram={metric_summary.get('pmt_pred_histogram', [])}")
        print(f"{split}: confusion matrix={metric_summary.get('pmt_confusion_matrix', [])}")
        print(f"{split}: per-class recall={metric_summary.get('pmt_per_class_recall', [])}")
        print(f"{split}: per-class precision={metric_summary.get('pmt_per_class_precision', [])}")
        print(f"{split}: per-class IoU={metric_summary.get('pmt_per_class_iou', [])}")

    def _epoch_iterable(self):
        if not self.overfit_one_batch:
            return self.train_loader, None
        if self.overfit_batch is None:
            self.overfit_batch = next(iter(self.train_loader))
            print(Fore.YELLOW + "overfit_one_batch=True: reuse one train batch every epoch")
        return [self.overfit_batch], 1

    def process_epoch(self, global_epoch):
        loss_batches = []
        metric_batches = []
        print(f"training global epoch {global_epoch}")
        self.model.train()
        if hasattr(self.model, "apply_train_phase_mode"):
            self.model.apply_train_phase_mode()

        loader, total_override = self._epoch_iterable()
        total = total_override if total_override is not None else len(loader)
        progress_bar = tqdm(
            loader, total=total, desc=f"[{global_epoch}/{self.max_epoch}]{self.save_str}",
            file=ResilientTextStream(sys.stderr),
        )
        for data in progress_bar:
            loss_dict, metric_dict = self.process_batch(data, global_epoch, True)
            # Reuse the CPU copies needed for epoch statistics in the progress bar.
            loss_dict = _detach_dict(loss_dict)
            metric_dict = _detach_dict(metric_dict)
            progress_bar.set_postfix({
                "pmt_acc": f"{_scalar(metric_dict, 'pmt_acc'):.4f}",
                **{f"{name}_loss": f"{_scalar(loss_dict, 'raw/' + name):.4f}"
                   for name in LOSS_NAMES if self._active_losses()[name]},
            }, refresh=False)
            loss_batches.append(loss_dict)
            metric_batches.append(metric_dict)

        loss_summary = _mean_dicts(loss_batches)
        metric_summary = _aggregate_metric_dicts(metric_batches)
        warn_if_primitive_collapsed(metric_summary, split="train", epoch=global_epoch)
        return loss_summary, metric_summary

    @staticmethod
    def _unpack_model_output(model_output):
        if not isinstance(model_output, dict):
            raise TypeError("Stage 1 model must return the multitask prediction dictionary")
        required = {"embedding", "log_pmt", "mad", "dim", "loc"}
        missing = sorted(required.difference(model_output))
        if missing:
            raise ValueError(f"Stage 1 model output is missing fields: {missing}")
        return model_output

    def _active_losses(self):
        return stage1_active_losses(self.train_phase)

    def process_batch(
        self,
        data_batch,
        global_epoch,
        is_train,
    ):
        """Stage1ConstraintDataset order: xyz, pmt, mad, dim, loc, affiliate_idx."""
        with torch.set_grad_enabled(is_train):
            if is_train:
                try:
                    self.optimizer.zero_grad(set_to_none=True)
                except TypeError:
                    self.optimizer.zero_grad()

            xyz = data_batch[0].float().to(self.device, non_blocking=True)
            pmt_gt = data_batch[1].long().to(self.device, non_blocking=True)
            mad_gt = data_batch[2].float().to(self.device, non_blocking=True)
            dim_gt = data_batch[3].float().to(self.device, non_blocking=True)
            loc_gt = data_batch[4].float().to(self.device, non_blocking=True)
            affiliate_idx = data_batch[-1].long().to(self.device, non_blocking=True)
            outputs = self._unpack_model_output(stage1_forward(
                self.model, xyz, use_extra_features=self.use_extra_features, feature_k=self.feature_k,
            ))
            self._assert_finite_outputs(outputs)
            loss, loss_dict = stage1_phase_loss(
                {name: value.float() for name, value in outputs.items()},
                pmt_gt, mad_gt, dim_gt, loc_gt, affiliate_idx,
                train_phase=self.train_phase, weights=self.loss_weights,
            )

            self._assert_finite_losses(loss_dict)
            if is_train:
                trainable_parameters = [
                    parameter
                    for parameter in self.model.parameters()
                    if parameter.requires_grad
                ]
                loss.backward()
                if self.grad_clip is not None and self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=self.grad_clip)
                self.optimizer.step()
                self.global_step += 1

            with torch.no_grad():
                primitive_metrics = evaluate_primitive_metrics(outputs["log_pmt"], pmt_gt)
                constraints = direct_constraints(outputs)
                attribute_metrics = evaluate_constraint_attribute_metrics(
                    mad_pred=constraints["direction"],
                    dim_pred=constraints["dimension"],
                    loc_pred=constraints["location"],
                    pmt_gt=pmt_gt,
                    mad_gt=mad_gt,
                    dim_gt=dim_gt,
                    loc_gt=loc_gt,
                )

            metric_dict = {}
            metric_dict.update(primitive_metrics)
            metric_dict.update(attribute_metrics)
            aggregation_weight = torch.tensor(
                float(xyz.shape[0]), device=xyz.device, dtype=torch.float32
            )
            loss_dict["_aggregation_weight"] = aggregation_weight
            metric_dict["_aggregation_weight"] = aggregation_weight
            return loss_dict, metric_dict

    @staticmethod
    def _assert_finite_outputs(outputs):
        assert_finite_tensors(outputs, "model outputs")

    @staticmethod
    def _assert_finite_losses(loss_dict):
        assert_finite_tensors(loss_dict, "loss terms")


def warn_if_primitive_collapsed(metric_summary, split="unknown", epoch=-1, threshold=0.95):
    histogram = metric_summary.get("pmt_pred_histogram", [])
    if primitive_prediction_collapsed(torch.as_tensor(histogram), threshold=threshold):
        print(
            Fore.RED
            + "primitive prediction collapsed to one class"
            + f" (split={split}, epoch={epoch}, predicted_histogram={histogram})"
        )
        return True
    return False


def load_model_state_with_diagnostics(
    model,
    incoming_state,
    *,
    require_complete,
    source,
):
    if not isinstance(incoming_state, dict):
        raise ValueError(f"model state in {source} is not a state_dict")
    current_state = model.state_dict()
    missing_keys = sorted(key for key in current_state if key not in incoming_state)
    unexpected_keys = sorted(key for key in incoming_state if key not in current_state)
    shape_mismatch = {}
    for key in sorted(set(current_state).intersection(incoming_state)):
        incoming_value = incoming_state[key]
        if not torch.is_tensor(incoming_value):
            shape_mismatch[key] = ("not-a-tensor", tuple(current_state[key].shape))
        elif tuple(incoming_value.shape) != tuple(current_state[key].shape):
            shape_mismatch[key] = (
                tuple(incoming_value.shape), tuple(current_state[key].shape)
            )

    complete = not missing_keys and not unexpected_keys and not shape_mismatch
    print(f"loading model from: {source}")
    print(f"missing_keys: {missing_keys}")
    print(f"unexpected_keys: {unexpected_keys}")
    print(f"shape_mismatch: {shape_mismatch}")
    print(f"model load complete: {complete}")
    if require_complete and not complete:
        raise RuntimeError(f"exact model state match required: {source}")

    compatible = {
        key: value
        for key, value in incoming_state.items()
        if key in current_state and key not in shape_mismatch
    }
    load_result = model.load_state_dict(compatible, strict=False)
    result_missing = sorted(load_result.missing_keys)
    result_unexpected = sorted(load_result.unexpected_keys)
    if require_complete and (result_missing or result_unexpected):
        raise RuntimeError(
            f"model load failed after preflight; missing={result_missing}, "
            f"unexpected={result_unexpected}"
        )
    return {
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
        "shape_mismatch": shape_mismatch,
        "complete": complete,
    }


def _extract_model_state(checkpoint):
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must be a dictionary")
    if "model" in checkpoint and isinstance(checkpoint["model"], dict):
        return checkpoint["model"]
    if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
        return checkpoint["state_dict"]
    if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
        return checkpoint
    raise ValueError("checkpoint does not contain model/state_dict weights")


def _critical_checkpoint_config(args):
    args = {} if args is None else args
    weights = {
        name: _normalize_config_value(args.get(name, 1.0))
        for name in ("w_pmt", "w_cluster", "w_mad", "w_dim", "w_loc")
    }
    return {
        "model": _normalize_config_value(args.get("model", "<missing>")),
        "train_phase": _normalize_config_value(args.get("train_phase", "<missing>")),
        "use_extra_features": _normalize_config_value(
            args.get("use_extra_features", "<missing>")
        ),
        "feature_k": _normalize_config_value(args.get("feature_k", "<missing>")),
        "point_count": _normalize_config_value(args.get("n_points", "<missing>")),
        "loss_weights": weights,
        "training_recipe": args.get("training_recipe", "legacy_regularized_v1"),
        "loc_input": args.get("loc_input", "backbone_only_v1"),
        "mad_input": args.get("mad_input", "backbone_only_v1"),
        "dim_input": args.get("dim_input", "backbone_only_v1"),
        "constraint_route": args.get("constraint_route", CONSTRAINT_ROUTE),
    }


def _normalize_config_value(value):
    if isinstance(value, str):
        lowered = value.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
    return value


def _config_differences(saved, current, prefix=""):
    differences = []
    keys = sorted(set(saved).union(current))
    for key in keys:
        path = f"{prefix}.{key}" if prefix else key
        saved_value = saved.get(key, "<missing>")
        current_value = current.get(key, "<missing>")
        if isinstance(saved_value, dict) and isinstance(current_value, dict):
            differences.extend(_config_differences(saved_value, current_value, path))
        elif saved_value != current_value:
            differences.append((path, saved_value, current_value))
    return differences


def _normalize_best_metrics(best_metrics):
    normalized = {}
    for name in BEST_FILE_NAMES:
        value = best_metrics.get(name, {"value": float("-inf"), "epoch": -1})
        if isinstance(value, dict):
            normalized[name] = {
                "value": float(value.get("value", float("-inf"))),
                "epoch": int(value.get("epoch", -1)),
            }
        else:
            normalized[name] = {"value": float(value), "epoch": -1}
    return normalized


def _aggregate_metric_dicts(dicts):
    if not dicts:
        return {}
    confusion_values = [
        item["pmt_confusion_matrix"]
        for item in dicts
        if "pmt_confusion_matrix" in item
    ]
    primitive_keys = {
        "pmt_acc", "pmt_gt_histogram", "pmt_pred_histogram",
        "pmt_confusion_matrix", "pmt_per_class_acc", "pmt_per_class_recall",
        "pmt_per_class_precision", "pmt_per_class_f1", "pmt_macro_f1",
        "pmt_per_class_iou", "pmt_miou",
    }
    excluded_keys = primitive_keys | CONSTRAINT_ATTRIBUTE_ACCUMULATOR_KEYS
    filtered = [
        {key: value for key, value in item.items() if key not in excluded_keys}
        for item in dicts
    ]
    output = _mean_dicts(filtered)
    if confusion_values:
        confusion = torch.stack([value.float() for value in confusion_values]).sum(dim=0)
        output.update(_to_python(primitive_metrics_from_confusion(confusion)))
    output.update(aggregate_constraint_attribute_metrics(dicts))
    return output

def _scalar(data, key):
    value = data.get(key, 0.0)
    if torch.is_tensor(value):
        return float(value.detach().float().mean().cpu())
    if isinstance(value, list):
        flat = torch.as_tensor(value).float()
        return float(flat.mean()) if flat.numel() else 0.0
    return float(value)


def _detach_dict(data):
    return {key: _detach_value(value) for key, value in data.items()}


def _detach_value(value):
    if torch.is_tensor(value):
        return value.detach().float().cpu()
    return value


def _to_python(value):
    if torch.is_tensor(value):
        value = value.detach().float().cpu()
        if value.numel() == 1:
            return float(value.item())
        return value.tolist()
    if isinstance(value, dict):
        return {key: _to_python(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_python(item) for item in value]
    return value


def _mean_dicts(dicts):
    if not dicts:
        return {}
    weights = [
        float(torch.as_tensor(item.get("_aggregation_weight", 1.0)).item())
        for item in dicts
    ]
    output = {}
    keys = sorted({
        key
        for item in dicts
        for key in item
        if key != "_aggregation_weight"
    })
    for key in keys:
        values_and_weights = [
            (item[key], weight)
            for item, weight in zip(dicts, weights)
            if key in item
        ]
        values = [value for value, _ in values_and_weights]
        available_weight = max(sum(weight for _, weight in values_and_weights), 1.0)
        first = values[0]
        if torch.is_tensor(first):
            weighted = sum(
                value.float() * weight
                for value, weight in values_and_weights
            ) / available_weight
            output[key] = _to_python(weighted)
        elif isinstance(first, (int, float)):
            output[key] = float(
                sum(float(value) * weight for value, weight in values_and_weights)
                / available_weight
            )
        else:
            output[key] = _to_python(first)
    return output
