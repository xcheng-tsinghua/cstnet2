from __future__ import annotations

import json
import os
from contextlib import nullcontext
from time import time

import torch
import torch.nn.functional as F
from colorama import Back, Fore
from tqdm import tqdm

from functional.loss import (
    constraint_loss,
    discriminative_loss,
    evaluate_clustering,
    linear_ramp,
)
from functional.point_features import build_stage1_input_features
from functional.stage1_metrics import (
    evaluate_predicted_clustering,
    evaluate_primitive_metrics,
    primitive_metrics_from_confusion,
    primitive_prediction_collapsed,
)
from functional.wandb_utils import flatten_wandb_metrics


LOSS_NAMES = ("pmt", "cluster", "mad", "dim", "nor", "loc", "geom", "inst")
BEST_FILE_NAMES = {
    "pmt_miou": "best_pmt_miou.pth",
    "cluster_ari": "best_cluster_ari.pth",
    "constraint_score": "best_constraint_score.pth",
}


class CstPredTrainer(object):
    """Train baseline or multitask Stage 1 without coupling it to Stage 2."""

    def __init__(
        self,
        model,
        train_loader,
        test_loader,
        checkpoint_dir,
        log_savepth,
        max_epoch,
        lr,
        save_str,
        wandb_run=None,
        decay_rate=1e-4,
        stage1_mode="baseline",
        loss_weights=None,
        geom_start_epoch=20,
        geom_ramp_epochs=20,
        use_extra_features=False,
        normal_source="gt",
        feature_k=16,
        cluster_bandwidth=0.35,
        overfit_one_batch=False,
        grad_clip=1.0,
        train_phase="semantic",
        enabled_losses=None,
        resume_checkpoint="",
        init_from_checkpoint="",
        checkpoint_args=None,
        joint_backbone_lr_scale=0.1,
        use_amp=False,
        enable_grad_diagnostics=True,
    ):
        super().__init__()
        if resume_checkpoint and init_from_checkpoint:
            raise ValueError("--resume_checkpoint and --init_from_checkpoint are mutually exclusive")
        self.model = model
        self.device = next(self.model.parameters()).device
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.log_savepth = log_savepth
        self.max_epoch = int(max_epoch)
        self.save_str = save_str
        self.wandb_run = wandb_run
        self.decay_rate = decay_rate
        self.stage1_mode = stage1_mode
        self.loss_weights = {} if loss_weights is None else dict(loss_weights)
        self.geom_start_epoch = int(geom_start_epoch)
        self.geom_ramp_epochs = int(geom_ramp_epochs)
        self.use_extra_features = bool(use_extra_features)
        self.normal_source = normal_source
        self.feature_k = int(feature_k)
        self.cluster_bandwidth = cluster_bandwidth
        self.overfit_one_batch = overfit_one_batch
        self.overfit_batch = None
        self.grad_clip = grad_clip
        self.train_phase = train_phase
        self.enabled_losses = {} if enabled_losses is None else dict(enabled_losses)
        self.resume_checkpoint = resume_checkpoint
        self.init_from_checkpoint = init_from_checkpoint
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_args = {} if checkpoint_args is None else dict(checkpoint_args)
        self.joint_backbone_lr_scale = float(joint_backbone_lr_scale)
        self.enable_grad_diagnostics = bool(enable_grad_diagnostics)
        self.use_amp = bool(use_amp and self.device.type == "cuda")
        if use_amp and not self.use_amp:
            print(Fore.YELLOW + "AMP requested without CUDA; AMP is disabled")
        self.amp_dtype = torch.float16
        self.scaler = torch.cuda.amp.GradScaler(enabled=True) if self.use_amp else None

        self.optimizer = None
        self.scheduler = None
        self.start_epoch = 0
        self.global_step = 0
        self.best_metrics = {
            "pmt_miou": {"value": float("-inf"), "epoch": -1},
            "cluster_ari": {"value": float("-inf"), "epoch": -1},
            "constraint_score": {"value": float("-inf"), "epoch": -1},
        }
        self.save_dict_train = self._new_save_dict()
        self.save_dict_test = self._new_save_dict()

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
            "clus_nmi": [],
            "clus_ari": [],
        }

    def _initialize_training(self, lr):
        resume_state = None
        if self.resume_checkpoint:
            resume_state = self._load_checkpoint_file(self.resume_checkpoint)
            self._validate_full_checkpoint(resume_state, self.resume_checkpoint)
            self._validate_resume_config(resume_state)
            self._load_model_state(
                resume_state["model"], require_complete=True, source=self.resume_checkpoint
            )
        elif self.init_from_checkpoint:
            init_state = self._load_checkpoint_file(self.init_from_checkpoint)
            self._load_model_state(
                _extract_model_state(init_state),
                require_complete=False,
                source=self.init_from_checkpoint,
            )
        else:
            print(Fore.BLACK + Back.CYAN + "training Stage 1 from scratch")

        self._configure_train_phase()
        self.make_optimizer_and_schedule(lr)

        if resume_state is not None:
            self.optimizer.load_state_dict(resume_state["optimizer"])
            self.scheduler.load_state_dict(resume_state["scheduler"])
            if self.scaler is not None:
                if "scaler" not in resume_state:
                    raise ValueError("AMP resume checkpoint is missing scaler state")
                self.scaler.load_state_dict(resume_state["scaler"])
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
                + f"resumed from {self.resume_checkpoint}: next_epoch={self.start_epoch}, "
                f"global_step={self.global_step}, lr={self.current_lrs()}"
            )
        elif self.init_from_checkpoint:
            print(
                Fore.WHITE
                + Back.CYAN
                + f"initialized model only from {self.init_from_checkpoint}; optimizer is new"
            )

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
                "Use --init_from_checkpoint for model-only/legacy weights."
            )

    def _validate_resume_config(self, checkpoint):
        saved_config = checkpoint.get("checkpoint_config")
        if saved_config is None:
            saved_config = _critical_checkpoint_config(checkpoint.get("args", {}))
        current_config = _critical_checkpoint_config(self.checkpoint_args)
        differences = _config_differences(saved_config, current_config)
        if differences:
            print(Fore.RED + "resume checkpoint configuration mismatch:")
            for key, saved, current in differences:
                print(Fore.RED + f"  {key}: checkpoint={saved!r}, current={current!r}")
            raise ValueError("resume checkpoint configuration mismatch")
        print(Fore.GREEN + "resume checkpoint configuration: exact match")

    def _load_model_state(self, incoming_state, require_complete, source):
        if not isinstance(incoming_state, dict):
            raise ValueError(f"model state in {source} is not a state_dict")
        current_state = self.model.state_dict()
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
            raise RuntimeError(f"resume requires an exact model state match: {source}")

        compatible = {
            key: value
            for key, value in incoming_state.items()
            if key in current_state and key not in shape_mismatch
        }
        load_result = self.model.load_state_dict(compatible, strict=False)
        result_missing = sorted(load_result.missing_keys)
        result_unexpected = sorted(load_result.unexpected_keys)
        if require_complete and (result_missing or result_unexpected):
            raise RuntimeError(
                f"model load failed after preflight; missing={result_missing}, "
                f"unexpected={result_unexpected}"
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
        if self.train_phase == "joint":
            backbone_params = [p for name, p in named_trainable if name.startswith("embedding.")]
            head_params = [p for name, p in named_trainable if not name.startswith("embedding.")]
            param_groups = []
            if backbone_params:
                param_groups.append({
                    "params": backbone_params,
                    "lr": lr * self.joint_backbone_lr_scale,
                    "group_name": "backbone_high",
                })
            if head_params:
                param_groups.append({"params": head_params, "lr": lr, "group_name": "heads"})
        else:
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
            epoch_lrs = self.current_lrs()
            start_time = time()
            train_loss, train_metrics = self.process_epoch(global_epoch, True)
            self.append_save_dict(train_loss, train_metrics, True)
            train_time = time() - start_time
            print(Fore.BLUE + f"training time: {train_time:.4f} sec")

            start_time = time()
            test_loss, test_metrics = self.process_epoch(global_epoch, False)
            self.append_save_dict(test_loss, test_metrics, False)
            test_time = time() - start_time
            print(Fore.BLUE + f"evaluation time: {test_time:.4f} sec")

            constraint_score = 0.5 * (
                float(test_metrics.get("pmt_miou", 0.0))
                + max(0.0, float(test_metrics.get("cluster_ari_real", 0.0)))
            )
            test_metrics["constraint_score"] = constraint_score
            improved = self._update_best_metrics(global_epoch, test_metrics)

            if self.wandb_run is not None:
                payload = {
                    "epoch": global_epoch,
                    "global_step": self.global_step,
                    "time/train_sec": train_time,
                    "time/test_sec": test_time,
                }
                for name, value in epoch_lrs.items():
                    payload[f"lr/{name}"] = value
                payload.update(flatten_wandb_metrics("train/loss", train_loss))
                payload.update(flatten_wandb_metrics("train/metric", train_metrics))
                payload.update(flatten_wandb_metrics("test/loss", test_loss))
                payload.update(flatten_wandb_metrics("test/metric", test_metrics))
                payload.update(flatten_wandb_metrics("best", self.best_metrics))
                self.wandb_run.log(payload, step=global_epoch)

            # The checkpoint contains the LR that will be used by the next epoch.
            self.scheduler.step()
            self.save(global_epoch, improved)

    def _update_best_metrics(self, epoch, test_metrics):
        candidates = {
            "pmt_miou": float(test_metrics.get("pmt_miou", float("-inf"))),
            "cluster_ari": float(test_metrics.get("cluster_ari_real", float("-inf"))),
            "constraint_score": float(test_metrics.get("constraint_score", float("-inf"))),
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
        _atomic_torch_save(payload, last_path)
        print(Fore.GREEN + f"saved checkpoint: {last_path}")
        for metric_name in improved:
            best_path = os.path.join(self.checkpoint_dir, BEST_FILE_NAMES[metric_name])
            _atomic_torch_save(payload, best_path)
            print(Fore.GREEN + f"saved best checkpoint: {best_path}")

        log_payload = {
            "run": {
                "start_epoch": self.start_epoch,
                "last_epoch": int(epoch),
                "global_step": self.global_step,
                "best_metrics": self.best_metrics,
                "checkpoint_dir": self.checkpoint_dir,
            },
            "train": self.save_dict_train,
            "test": self.save_dict_test,
        }
        with open(self.log_savepth, "w") as file:
            json.dump(log_payload, file, ensure_ascii=False, indent=4)

    def _checkpoint_payload(self, epoch):
        checkpoint_config = _critical_checkpoint_config(self.checkpoint_args)
        payload = {
            "epoch": int(epoch),
            "global_step": int(self.global_step),
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "best_metrics": self.best_metrics,
            "args": self.checkpoint_args,
            "checkpoint_config": checkpoint_config,
            "loss_schedule": {
                "global_epoch": int(epoch),
                "next_global_epoch": int(epoch) + 1,
                "global_step": int(self.global_step),
                "aux_progress": linear_ramp(
                    epoch, self.geom_start_epoch, self.geom_ramp_epochs
                ),
                "geom_start_epoch": self.geom_start_epoch,
                "geom_ramp_epochs": self.geom_ramp_epochs,
            },
        }
        if self.scaler is not None:
            payload["scaler"] = self.scaler.state_dict()
        return payload

    def append_save_dict(self, loss_summary, metric_summary, is_train):
        target = self.save_dict_train if is_train else self.save_dict_test
        split = "train" if is_train else "test"
        target["loss"].append(loss_summary)
        target["metrics"].append(metric_summary)
        target["prim_loss"].append(float(loss_summary.get("raw/pmt", 0.0)))
        target["clus_loss"].append(float(loss_summary.get("raw/cluster", 0.0)))
        target["prim_acc"].append(float(metric_summary.get("pmt_acc", 0.0)))
        target["clus_nmi"].append(float(metric_summary.get("cluster_nmi_real", 0.0)))
        target["clus_ari"].append(float(metric_summary.get("cluster_ari_real", 0.0)))

        raw_text = ", ".join(
            f"{name}={float(loss_summary.get('raw/' + name, 0.0)):.5f}"
            for name in LOSS_NAMES
        )
        weighted_text = ", ".join(
            f"{name}={float(loss_summary.get('weighted/' + name, 0.0)):.5f}"
            for name in LOSS_NAMES
        )
        print(
            f"{split}: loss_all={loss_summary.get('loss_all', 0.0):.6f}; "
            f"raw[{raw_text}]; weighted[{weighted_text}]"
        )
        print(
            f"{split}: pmt_acc={metric_summary.get('pmt_acc', 0.0):.4f}, "
            f"pmt_macro_f1={metric_summary.get('pmt_macro_f1', 0.0):.4f}, "
            f"pmt_miou={metric_summary.get('pmt_miou', 0.0):.4f}, "
            f"cluster_ari_real={metric_summary.get('cluster_ari_real', 0.0):.4f}, "
            f"cluster_nmi_real={metric_summary.get('cluster_nmi_real', 0.0):.4f}"
        )
        print(f"{split}: gt primitive histogram={metric_summary.get('pmt_gt_histogram', [])}")
        print(f"{split}: predicted primitive histogram={metric_summary.get('pmt_pred_histogram', [])}")
        print(f"{split}: confusion matrix={metric_summary.get('pmt_confusion_matrix', [])}")
        print(f"{split}: per-class recall={metric_summary.get('pmt_per_class_recall', [])}")
        print(f"{split}: per-class precision={metric_summary.get('pmt_per_class_precision', [])}")
        print(f"{split}: per-class IoU={metric_summary.get('pmt_per_class_iou', [])}")

    def _epoch_iterable(self, is_train):
        if not self.overfit_one_batch:
            return self.train_loader if is_train else self.test_loader, None
        if self.overfit_batch is None:
            self.overfit_batch = next(iter(self.train_loader))
            print(Fore.YELLOW + "overfit_one_batch=True: reuse one train batch every epoch")
        return [self.overfit_batch], 1

    def process_epoch(self, global_epoch, is_train):
        loss_batches = []
        metric_batches = []
        if is_train:
            print(f"training global epoch {global_epoch}")
            self.model.train()
            if hasattr(self.model, "apply_train_phase_mode"):
                self.model.apply_train_phase_mode()
        else:
            print(f"testing global epoch {global_epoch}")
            self.model.eval()

        loader, total_override = self._epoch_iterable(is_train)
        total = total_override if total_override is not None else len(loader)
        progress_bar = tqdm(
            loader, total=total, desc=f"[{global_epoch}/{self.max_epoch}]{self.save_str}"
        )
        for batch_index, data in enumerate(progress_bar):
            loss_dict, metric_dict = self.process_batch(
                data,
                global_epoch,
                is_train,
                diagnose_gradients=(
                    is_train and batch_index == 0 and self.enable_grad_diagnostics
                ),
            )
            progress_bar.set_postfix({
                "loss": f"{_scalar(loss_dict, 'loss_all'):.4f}",
                "pmt_acc": f"{_scalar(metric_dict, 'pmt_acc'):.4f}",
                "pmt_miou": f"{_scalar(metric_dict, 'pmt_miou'):.4f}",
                "ari_real": f"{_scalar(metric_dict, 'cluster_ari_real'):.4f}",
                "LR": f"{max(self.current_lrs().values()):.6f}",
            })
            loss_batches.append(_detach_dict(loss_dict))
            metric_batches.append(_detach_dict(metric_dict))

        loss_summary = _mean_dicts(loss_batches)
        metric_summary = _aggregate_metric_dicts(metric_batches)
        warn_if_primitive_collapsed(metric_summary, split="train" if is_train else "test", epoch=global_epoch)
        return loss_summary, metric_summary

    def _build_features(self, xyz, nor_gt):
        if not self.use_extra_features:
            return None
        if self.normal_source == "gt":
            normals, use_normals = nor_gt, True
        elif self.normal_source == "pca":
            normals, use_normals = None, True
        else:
            normals, use_normals = None, False
        with torch.no_grad():
            features = build_stage1_input_features(
                xyz,
                normals=normals,
                use_normals=use_normals,
                use_curvature=True,
                k=self.feature_k,
            )
        return features.detach()

    @staticmethod
    def _unpack_model_output(model_output):
        if isinstance(model_output, dict):
            return model_output
        embedding, log_pmt = model_output
        return {"embedding": embedding, "log_pmt": log_pmt}

    def _active_losses(self):
        active = {name: False for name in LOSS_NAMES}
        if self.stage1_mode == "baseline":
            active.update({"pmt": True, "cluster": True})
            return active
        if self.train_phase in ("semantic", "joint"):
            active["pmt"] = True
            active["cluster"] = True
        for name in ("mad", "dim", "nor", "loc", "geom", "inst"):
            phase_allows = self.train_phase in ("geometry", "joint")
            if name == "nor":
                phase_allows = self.train_phase in ("semantic", "geometry", "joint")
            active[name] = phase_allows and bool(self.enabled_losses.get(name, True))
        return active

    def process_batch(
        self,
        data_batch,
        global_epoch,
        is_train,
        diagnose_gradients=False,
    ):
        """CstNet2Dataset order: xyz, cls, pmt, mad, dim, nor, loc, affiliate_idx."""
        with torch.set_grad_enabled(is_train):
            if is_train:
                try:
                    self.optimizer.zero_grad(set_to_none=True)
                except TypeError:
                    self.optimizer.zero_grad()

            xyz = data_batch[0].float().to(self.device, non_blocking=True)
            pmt_gt = data_batch[2].long().to(self.device, non_blocking=True)
            mad_gt = data_batch[3].float().to(self.device, non_blocking=True)
            dim_gt = data_batch[4].float().to(self.device, non_blocking=True)
            nor_gt = data_batch[5].float().to(self.device, non_blocking=True)
            loc_gt = data_batch[6].float().to(self.device, non_blocking=True)
            affiliate_idx = data_batch[-1].long().to(self.device, non_blocking=True)
            extra_fea = self._build_features(xyz, nor_gt)
            amp_context = (
                torch.cuda.amp.autocast(dtype=self.amp_dtype)
                if self.use_amp
                else nullcontext()
            )
            with amp_context:
                outputs = self._unpack_model_output(self.model(xyz, extra_fea))
                self._assert_finite_outputs(outputs)
                active_losses = self._active_losses()
                if self.stage1_mode == "multitask" and all(
                    key in outputs for key in ("mad", "dim", "nor", "loc")
                ):
                    loss, loss_dict = constraint_loss(
                        xyz=xyz,
                        log_pmt_pred=outputs["log_pmt"].float(),
                        mad_pred=outputs["mad"].float(),
                        dim_pred=outputs["dim"].float(),
                        nor_pred=outputs["nor"].float(),
                        loc_pred=outputs["loc"].float(),
                        pmt_gt=pmt_gt,
                        mad_gt=mad_gt,
                        dim_gt=dim_gt,
                        nor_gt=nor_gt,
                        loc_gt=loc_gt,
                        affil_idx=affiliate_idx,
                        point_emb=outputs["embedding"].float(),
                        weights=self.loss_weights,
                        global_epoch=global_epoch,
                        geom_start_epoch=self.geom_start_epoch,
                        geom_ramp_epochs=self.geom_ramp_epochs,
                        enabled_losses=active_losses,
                    )
                else:
                    pmt_loss, cluster_loss = compute_loss(
                        outputs["embedding"].float(),
                        affiliate_idx,
                        outputs["log_pmt"].float(),
                        pmt_gt,
                    )
                    loss, loss_dict = _baseline_loss_dict(
                        pmt_loss, cluster_loss, self.loss_weights, xyz
                    )

            self._assert_finite_losses(loss_dict)
            gradient_metrics = {}
            if diagnose_gradients:
                gradient_metrics = self._gradient_diagnostics(loss_dict)

            if is_train:
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    if self.grad_clip is not None and self.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in self.model.parameters() if p.requires_grad],
                            max_norm=self.grad_clip,
                        )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    if self.grad_clip is not None and self.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in self.model.parameters() if p.requires_grad],
                            max_norm=self.grad_clip,
                        )
                    self.optimizer.step()
                self.global_step += 1

            with torch.no_grad():
                primitive_metrics = evaluate_primitive_metrics(outputs["log_pmt"], pmt_gt)
                real_cluster_metrics = evaluate_predicted_clustering(
                    affiliate_idx,
                    outputs["embedding"],
                    bandwidth=self.cluster_bandwidth,
                )
                oracle_acc, oracle_nmi, oracle_ari = evaluate_clustering(
                    affiliate_idx, outputs["embedding"]
                )

            metric_dict = {}
            metric_dict.update(primitive_metrics)
            metric_dict.update(real_cluster_metrics)
            metric_dict.update({
                "cluster_acc_oracle_optional": oracle_acc,
                "cluster_nmi_oracle_optional": oracle_nmi,
                "cluster_ari_oracle_optional": oracle_ari,
            })
            metric_dict.update(gradient_metrics)
            return loss_dict, metric_dict

    def _gradient_diagnostics(self, loss_dict):
        shared_params = [
            param
            for name, param in self.model.named_parameters()
            if name.startswith("embedding.") and param.requires_grad
        ]
        diagnostics = {}
        if not shared_params:
            for name in LOSS_NAMES:
                diagnostics[f"grad_norm/{name}"] = 0.0
            diagnostics.update({
                "grad_cosine/pmt_vs_cluster": 0.0,
                "grad_cosine/pmt_vs_geom": 0.0,
                "grad_cosine/pmt_vs_inst": 0.0,
            })
            return diagnostics

        gradient_cache = {}
        pmt_grads = _task_gradients(loss_dict["raw/pmt"], shared_params)
        gradient_cache["pmt"] = pmt_grads
        diagnostics["grad_norm/pmt"] = _gradient_norm(pmt_grads)
        for name in LOSS_NAMES[1:]:
            grads = _task_gradients(loss_dict[f"raw/{name}"], shared_params)
            diagnostics[f"grad_norm/{name}"] = _gradient_norm(grads)
            if name in ("cluster", "geom", "inst"):
                diagnostics[f"grad_cosine/pmt_vs_{name}"] = _gradient_cosine(
                    pmt_grads, grads
                )
        return diagnostics

    @staticmethod
    def _assert_finite_outputs(outputs):
        bad = [
            name
            for name, value in outputs.items()
            if torch.is_tensor(value) and not torch.isfinite(value).all()
        ]
        if bad:
            print(Fore.RED + f"non-finite model outputs: {bad}")
            raise FloatingPointError(f"non-finite model outputs: {bad}")

    @staticmethod
    def _assert_finite_losses(loss_dict):
        bad = [
            name
            for name, value in loss_dict.items()
            if torch.is_tensor(value) and not torch.isfinite(value).all()
        ]
        if bad:
            values = {
                name: _to_python(value.detach())
                for name, value in loss_dict.items()
                if torch.is_tensor(value)
            }
            print(Fore.RED + f"non-finite loss terms: {bad}; values={values}")
            raise FloatingPointError(f"non-finite loss terms: {bad}")


def compute_loss(pnt_fea, affiliate_idx, log_pmt, pmt_gt):
    """Baseline Stage 1 loss: primitive NLL plus instance discriminative loss."""
    pnt_fea = _nan_to_num(pnt_fea.float(), nan=0.0, posinf=1e4, neginf=-1e4)
    log_pmt = _nan_to_num(log_pmt.float(), nan=0.0, posinf=1e4, neginf=-1e4)
    cluster_loss = discriminative_loss(pnt_fea, affiliate_idx)
    pmt_loss = F.nll_loss(log_pmt.reshape(-1, log_pmt.shape[-1]), pmt_gt.reshape(-1))
    return pmt_loss, _nan_to_num(cluster_loss, nan=0.0, posinf=1e4, neginf=-1e4)


def _baseline_loss_dict(pmt_loss, cluster_loss, weights, reference):
    zero = reference.sum() * 0.0
    raw = {name: zero for name in LOSS_NAMES}
    raw["pmt"] = pmt_loss
    raw["cluster"] = cluster_loss
    target = {
        "pmt": float(weights.get("w_pmt", 1.0)),
        "cluster": float(weights.get("w_cluster", 0.5)),
    }
    weighted = {name: zero for name in LOSS_NAMES}
    weighted["pmt"] = raw["pmt"] * target["pmt"]
    weighted["cluster"] = raw["cluster"] * target["cluster"]
    loss = weighted["pmt"] + weighted["cluster"]
    output = {
        "loss_all": loss,
        "pmt_loss": pmt_loss,
        "cluster_loss": cluster_loss,
        "mad_loss": zero,
        "dim_loss": zero,
        "nor_loss": zero,
        "loc_loss": zero,
        "geom_loss": zero,
        "inst_loss": zero,
        "loss_plane": zero,
        "loss_cylinder": zero,
        "loss_cone": zero,
        "loss_sphere": zero,
        "aux_factor": zero,
        "schedule/aux_progress": zero,
    }
    for name in LOSS_NAMES:
        output[f"raw/{name}"] = raw[name]
        output[f"weighted/{name}"] = weighted[name]
        output[f"effective_weight/{name}"] = reference.new_tensor(
            target.get(name, 0.0)
        )
    return loss, output


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


def _task_gradients(loss, parameters):
    if not torch.is_tensor(loss) or not loss.requires_grad:
        return [None for _ in parameters]
    return list(torch.autograd.grad(
        loss,
        parameters,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    ))


def _gradient_norm(gradients):
    total = None
    for gradient in gradients:
        if gradient is None:
            continue
        term = gradient.detach().float().pow(2).sum()
        total = term if total is None else total + term
    if total is None:
        return 0.0
    return float(total.sqrt().cpu())


def _gradient_cosine(left, right):
    dot = None
    left_sq = None
    right_sq = None
    for left_grad, right_grad in zip(left, right):
        if left_grad is None or right_grad is None:
            continue
        left_grad = left_grad.detach().float()
        right_grad = right_grad.detach().float()
        current_dot = (left_grad * right_grad).sum()
        current_left = left_grad.pow(2).sum()
        current_right = right_grad.pow(2).sum()
        dot = current_dot if dot is None else dot + current_dot
        left_sq = current_left if left_sq is None else left_sq + current_left
        right_sq = current_right if right_sq is None else right_sq + current_right
    if dot is None or left_sq <= 0 or right_sq <= 0:
        return 0.0
    return float((dot / (left_sq.sqrt() * right_sq.sqrt() + 1e-12)).cpu())


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
        name: _normalize_config_value(args.get(name, "<missing>"))
        for name in (
            "w_pmt", "w_cluster", "w_mad", "w_dim",
            "w_nor", "w_loc", "w_geom", "w_inst",
        )
    }
    enabled = {
        name: _normalize_config_value(args.get(f"enable_{name}_loss", "<missing>"))
        for name in ("mad", "dim", "nor", "loc", "geom", "inst")
    }
    return {
        "model": _normalize_config_value(args.get("model", "<missing>")),
        "stage1_mode": _normalize_config_value(args.get("stage1_mode", "<missing>")),
        "train_phase": _normalize_config_value(args.get("train_phase", "<missing>")),
        "use_extra_features": _normalize_config_value(
            args.get("use_extra_features", "<missing>")
        ),
        "normal_source": _normalize_config_value(args.get("normal_source", "<missing>")),
        "feature_k": _normalize_config_value(args.get("feature_k", "<missing>")),
        "point_count": _normalize_config_value(args.get("n_points", "<missing>")),
        "loss_weights": weights,
        "enabled_losses": enabled,
        "geom_start_epoch": _normalize_config_value(
            args.get("geom_start_epoch", "<missing>")
        ),
        "geom_ramp_epochs": _normalize_config_value(
            args.get("geom_ramp_epochs", "<missing>")
        ),
        "joint_backbone_lr_scale": _normalize_config_value(
            args.get("joint_backbone_lr_scale", "<missing>")
        ),
        "use_amp": _normalize_config_value(args.get("use_amp", False)),
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


def _atomic_torch_save(payload, path):
    temporary_path = path + ".tmp"
    torch.save(payload, temporary_path)
    os.replace(temporary_path, path)


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
    filtered = [
        {key: value for key, value in item.items() if key not in primitive_keys}
        for item in dicts
    ]
    output = _mean_dicts(filtered)
    if confusion_values:
        confusion = torch.stack([value.float() for value in confusion_values]).sum(dim=0)
        output.update(_to_python(primitive_metrics_from_confusion(confusion)))
    return output

def _scalar(data, key):
    value = data.get(key, 0.0)
    if torch.is_tensor(value):
        return float(value.detach().float().mean().cpu())
    if isinstance(value, list):
        flat = torch.as_tensor(value).float()
        return float(flat.mean()) if flat.numel() else 0.0
    return float(value)


def _nan_to_num(value, nan=0.0, posinf=1e4, neginf=-1e4):
    if hasattr(torch, "nan_to_num"):
        return torch.nan_to_num(value, nan=nan, posinf=posinf, neginf=neginf)
    value = torch.where(torch.isnan(value), value.new_tensor(nan), value)
    value = torch.where(value == float("inf"), value.new_tensor(posinf), value)
    return torch.where(value == float("-inf"), value.new_tensor(neginf), value)


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
    output = {}
    keys = sorted({key for item in dicts for key in item})
    for key in keys:
        values = [item[key] for item in dicts if key in item]
        first = values[0]
        if torch.is_tensor(first):
            output[key] = _to_python(torch.stack([value.float() for value in values]).mean(dim=0))
        elif isinstance(first, (int, float)):
            output[key] = float(sum(float(value) for value in values) / len(values))
        else:
            output[key] = _to_python(first)
    return output
