import json
from contextlib import nullcontext
from time import time

import torch
import torch.nn.functional as F
from colorama import Back, Fore
from tqdm import tqdm

from functional.loss import constraint_loss, discriminative_loss, evaluate_clustering
from functional.point_features import build_stage1_input_features
from functional.stage1_metrics import evaluate_predicted_clustering, evaluate_primitive_metrics


class CstPredTrainer(object):
    """
    用于训练约束预测模块。支持旧 baseline 和增强 multitask Stage 1。
    Dataset batch order is confirmed from CstNet2Dataset:
    xyz, cls, pmt, mad, dim, nor, loc, affiliate_idx.
    """
    def __init__(
        self,
        model,
        train_loader,
        test_loader,
        model_savepth,
        log_savepth,
        max_epoch,
        lr,
        is_load_weight,
        save_str,
        wandb_run=None,
        decay_rate=1e-4,
        stage1_mode="baseline",
        loss_weights=None,
        geom_warmup_epoch=20,
        use_extra_features=False,
        normal_source="gt",
        feature_k=16,
        cluster_bandwidth=0.35,
        overfit_one_batch=False,
        grad_clip=1.0,
    ):
        super().__init__()
        print(f'weight save to: {model_savepth}, log save to: {log_savepth}')

        self.model = model
        self.device = next(self.model.parameters()).device
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model_savepth = model_savepth
        self.log_savepth = log_savepth
        self.max_epoch = max_epoch
        self.save_str = save_str
        self.wandb_run = wandb_run
        self.decay_rate = decay_rate
        self.stage1_mode = stage1_mode
        self.loss_weights = {} if loss_weights is None else loss_weights
        self.geom_warmup_epoch = geom_warmup_epoch
        self.use_extra_features = use_extra_features
        self.normal_source = normal_source
        self.feature_k = feature_k
        self.cluster_bandwidth = cluster_bandwidth
        self.overfit_one_batch = overfit_one_batch
        self.overfit_batch = None
        self.grad_clip = grad_clip
        self.use_amp = False
        self.amp_dtype = torch.bfloat16

        self.optimizer = None
        self.scheduler = None
        self.make_optimizer_and_schedule(lr)
        self.load_weight(is_load_weight)

        self.save_dict_train = self._new_save_dict()
        self.save_dict_test = self._new_save_dict()

    @staticmethod
    def _new_save_dict():
        return {
            'loss': [],
            'metrics': [],
            # Legacy-friendly mirrors for quick plotting.
            'prim_loss': [],
            'clus_loss': [],
            'prim_acc': [],
            'clus_nmi': [],
            'clus_ari': [],
        }

    def make_optimizer_and_schedule(self, lr):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=self.decay_rate
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.9)

    def load_weight(self, is_load_weight):
        if is_load_weight:
            try:
                state = torch.load(self.model_savepth, map_location=self.device)
                if isinstance(state, dict) and "state_dict" in state:
                    state = state["state_dict"]
                missing, unexpected = self.model.load_state_dict(state, strict=False)
                print(Fore.WHITE + Back.CYAN + 'training from exist model: ' + self.model_savepth)
                if missing:
                    print(Fore.YELLOW + f'missing weights initialized from scratch: {missing}')
                if unexpected:
                    print(Fore.YELLOW + f'unexpected weights ignored: {unexpected}')
            except Exception as exc:
                print(Fore.RED + Back.CYAN + f'no existing compatible model, training from scratch: {exc}')
        else:
            print(Fore.BLACK + Back.CYAN + 'does not load weight, training from scratch')

    def start(self):
        for epoch in range(self.max_epoch):
            start_time = time()
            train_loss, train_metrics = self.process_epoch(epoch, True)
            self.append_save_dict(train_loss, train_metrics, True)
            train_time = time() - start_time
            print(Fore.BLUE + f'训练耗时: {train_time:.4f} 秒')

            start_time = time()
            test_loss, test_metrics = self.process_epoch(epoch, False)
            self.append_save_dict(test_loss, test_metrics, False)
            test_time = time() - start_time
            print(Fore.BLUE + f'评估耗时: {test_time:.4f} 秒')

            if self.wandb_run is not None:
                log_payload = {
                    'epoch': epoch,
                    'lr': self.optimizer.param_groups[0]['lr'],
                    'time/train_sec': train_time,
                    'time/test_sec': test_time,
                }
                log_payload.update(_flatten_for_log('train/loss', train_loss))
                log_payload.update(_flatten_for_log('train/metric', train_metrics))
                log_payload.update(_flatten_for_log('test/loss', test_loss))
                log_payload.update(_flatten_for_log('test/metric', test_metrics))
                self.wandb_run.log(log_payload, step=epoch)

            self.save()
            self.scheduler.step()

    def save(self):
        print(Fore.GREEN + f'save model dict to {self.model_savepth}')
        torch.save(self.model.state_dict(), self.model_savepth)

        print(Fore.GREEN + f'save training dict to {self.log_savepth}')
        with open(self.log_savepth, 'w') as f:
            json.dump({'train': self.save_dict_train, 'test': self.save_dict_test}, f, ensure_ascii=False, indent=4)

    def append_save_dict(self, loss_summary, metric_summary, is_train):
        target = self.save_dict_train if is_train else self.save_dict_test
        split = 'train' if is_train else 'test'
        target['loss'].append(loss_summary)
        target['metrics'].append(metric_summary)
        target['prim_loss'].append(float(loss_summary.get('pmt_loss', 0.0)))
        target['clus_loss'].append(float(loss_summary.get('cluster_loss', 0.0)))
        target['prim_acc'].append(float(metric_summary.get('pmt_acc', 0.0)))
        target['clus_nmi'].append(float(metric_summary.get('cluster_nmi_real', 0.0)))
        target['clus_ari'].append(float(metric_summary.get('cluster_ari_real', 0.0)))

        print(
            f'{split}_metrics: '
            f'loss_all: {loss_summary.get("loss_all", 0.0):.6f}. '
            f'pmt_loss: {loss_summary.get("pmt_loss", 0.0):.6f}. '
            f'cluster_loss: {loss_summary.get("cluster_loss", 0.0):.6f}. '
            f'mad_loss: {loss_summary.get("mad_loss", 0.0):.6f}. '
            f'dim_loss: {loss_summary.get("dim_loss", 0.0):.6f}. '
            f'nor_loss: {loss_summary.get("nor_loss", 0.0):.6f}. '
            f'loc_loss: {loss_summary.get("loc_loss", 0.0):.6f}. '
            f'geom_loss: {loss_summary.get("geom_loss", 0.0):.6f}. '
            f'inst_loss: {loss_summary.get("inst_loss", 0.0):.6f}. '
            f'pmt_acc: {metric_summary.get("pmt_acc", 0.0):.4f}. '
            f'pmt_miou: {metric_summary.get("pmt_miou", 0.0):.4f}. '
            f'cluster_ari_real: {metric_summary.get("cluster_ari_real", 0.0):.4f}. '
            f'cluster_nmi_real: {metric_summary.get("cluster_nmi_real", 0.0):.4f}. '
            f'cluster_ari_oracle_optional: {metric_summary.get("cluster_ari_oracle_optional", 0.0):.4f}. '
            f'cluster_nmi_oracle_optional: {metric_summary.get("cluster_nmi_oracle_optional", 0.0):.4f}'
        )

    def _epoch_iterable(self, is_train):
        if not self.overfit_one_batch:
            return self.train_loader if is_train else self.test_loader, None

        if self.overfit_batch is None:
            self.overfit_batch = next(iter(self.train_loader))
            print(Fore.YELLOW + 'overfit_one_batch=True: cache one train batch and reuse it every epoch')
        return [self.overfit_batch], 1

    def process_epoch(self, current_epoch, is_train):
        loss_batches = []
        metric_batches = []

        if is_train:
            print('training epoch')
            self.model.train()
        else:
            print('testing epoch')
            self.model.eval()

        loader, total_override = self._epoch_iterable(is_train)
        total = total_override if total_override is not None else len(loader)
        progress_bar = tqdm(loader, total=total, desc=f'[{current_epoch}/{self.max_epoch}]{self.save_str}')
        for data in progress_bar:
            loss_dict, metric_dict = self.process_batch(data, current_epoch, is_train)

            progress_bar.set_postfix({
                'loss': f'{_scalar(loss_dict, "loss_all"):.4f}',
                'pmt_acc': f'{_scalar(metric_dict, "pmt_acc"):.4f}',
                'pmt_miou': f'{_scalar(metric_dict, "pmt_miou"):.4f}',
                'ari_real': f'{_scalar(metric_dict, "cluster_ari_real"):.4f}',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })

            loss_batches.append(_detach_dict(loss_dict))
            metric_batches.append(_detach_dict(metric_dict))

        return _mean_dicts(loss_batches), _mean_dicts(metric_batches)

    def _build_features(self, xyz, nor_gt):
        if not self.use_extra_features:
            return None
        if self.normal_source == 'gt':
            normals = nor_gt
            use_normals = True
        elif self.normal_source == 'pca':
            normals = None
            use_normals = True
        else:
            normals = None
            use_normals = False

        with torch.no_grad():
            fea = build_stage1_input_features(
                xyz,
                normals=normals,
                use_normals=use_normals,
                use_curvature=True,
                k=self.feature_k,
            )
        return fea.detach()

    @staticmethod
    def _unpack_model_output(model_output):
        if isinstance(model_output, dict):
            return model_output
        embedding, log_pmt = model_output
        return {
            "embedding": embedding,
            "log_pmt": log_pmt,
        }

    def process_batch(self, data_batch, current_epoch, is_train):
        """
        记录一个 batch 内的操作。
        CstNet2Dataset returns:
        0 xyz, 1 cls, 2 pmt, 3 mad, 4 dim, 5 nor, 6 loc, 7 affiliate_idx.
        """
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

            amp_ctx = torch.autocast(device_type='cuda', dtype=self.amp_dtype) if self.use_amp else nullcontext()
            with amp_ctx:
                outputs = self._unpack_model_output(self.model(xyz, extra_fea))
                self._assert_finite_outputs(outputs)

                if self.stage1_mode == 'multitask' and all(k in outputs for k in ('mad', 'dim', 'nor', 'loc')):
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
                        current_epoch=current_epoch,
                        geom_warmup_epoch=self.geom_warmup_epoch,
                    )
                else:
                    pmt_loss, cluster_loss = compute_loss(
                        outputs["embedding"].float(),
                        affiliate_idx,
                        outputs["log_pmt"].float(),
                        pmt_gt,
                    )
                    loss = self.loss_weights.get("w_pmt", 1.0) * pmt_loss + self.loss_weights.get("w_cluster", 1.0) * cluster_loss
                    z = loss.detach() * 0.0
                    loss_dict = {
                        "loss_all": loss,
                        "pmt_loss": pmt_loss,
                        "cluster_loss": cluster_loss,
                        "mad_loss": z,
                        "dim_loss": z,
                        "nor_loss": z,
                        "loc_loss": z,
                        "geom_loss": z,
                        "inst_loss": z,
                        "loss_plane": z,
                        "loss_cylinder": z,
                        "loss_cone": z,
                        "loss_sphere": z,
                        "aux_factor": z,
                    }

            self._assert_finite_losses(loss_dict)

            if is_train:
                loss.backward()
                if self.grad_clip is not None and self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
                self.optimizer.step()

            with torch.no_grad():
                primitive_metrics = evaluate_primitive_metrics(outputs["log_pmt"], pmt_gt)
                real_cluster_metrics = evaluate_predicted_clustering(
                    affiliate_idx,
                    outputs["embedding"],
                    bandwidth=self.cluster_bandwidth,
                )
                oracle_acc, oracle_nmi, oracle_ari = evaluate_clustering(affiliate_idx, outputs["embedding"])

            metric_dict = {}
            metric_dict.update(primitive_metrics)
            metric_dict.update(real_cluster_metrics)
            metric_dict.update({
                "cluster_acc_oracle_optional": oracle_acc,
                "cluster_nmi_oracle_optional": oracle_nmi,
                "cluster_ari_oracle_optional": oracle_ari,
            })
            return loss_dict, metric_dict

    @staticmethod
    def _assert_finite_outputs(outputs):
        bad = [name for name, value in outputs.items() if torch.is_tensor(value) and not torch.isfinite(value).all()]
        if bad:
            print(Fore.RED + f'non-finite model outputs: {bad}')
            raise FloatingPointError(f'non-finite model outputs: {bad}')

    @staticmethod
    def _assert_finite_losses(loss_dict):
        bad = [name for name, value in loss_dict.items() if torch.is_tensor(value) and not torch.isfinite(value).all()]
        if bad:
            values = {
                name: _to_python(value.detach())
                for name, value in loss_dict.items()
                if torch.is_tensor(value)
            }
            print(Fore.RED + f'non-finite loss terms: {bad}; values={values}')
            raise FloatingPointError(f'non-finite loss terms: {bad}')


def compute_loss(pnt_fea, affiliate_idx, log_pmt, pmt_gt):
    """
    Baseline Stage 1 loss: primitive type NLL + instance discriminative loss.
    """
    pnt_fea = _nan_to_num(pnt_fea.float(), nan=0.0, posinf=1e4, neginf=-1e4)
    log_pmt = _nan_to_num(log_pmt.float(), nan=0.0, posinf=1e4, neginf=-1e4)
    cluster_loss = discriminative_loss(pnt_fea, affiliate_idx)
    pmt_loss = F.nll_loss(log_pmt.reshape(-1, log_pmt.shape[-1]), pmt_gt.reshape(-1))
    return pmt_loss, _nan_to_num(cluster_loss, nan=0.0, posinf=1e4, neginf=-1e4)


def compute_seg_acc(pred, label):
    bs, n_points, _ = pred.size()
    n_items_batch = bs * n_points
    choice_meta_type = pred.argmax(dim=2)
    correct_meta_type = choice_meta_type.eq(label).sum()
    return correct_meta_type / n_items_batch


def _scalar(data, key):
    value = data.get(key, 0.0)
    if torch.is_tensor(value):
        return float(value.detach().float().mean().cpu())
    if isinstance(value, list):
        return float(sum(value) / max(1, len(value)))
    return float(value)


def _nan_to_num(value, nan=0.0, posinf=1e4, neginf=-1e4):
    if hasattr(torch, "nan_to_num"):
        return torch.nan_to_num(value, nan=nan, posinf=posinf, neginf=neginf)
    value = torch.where(torch.isnan(value), value.new_tensor(nan), value)
    value = torch.where(value == float("inf"), value.new_tensor(posinf), value)
    value = torch.where(value == float("-inf"), value.new_tensor(neginf), value)
    return value


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
        return {k: _to_python(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_python(v) for v in value]
    return value


def _mean_dicts(dicts):
    if len(dicts) == 0:
        return {}
    out = {}
    keys = sorted({key for item in dicts for key in item.keys()})
    for key in keys:
        values = [item[key] for item in dicts if key in item]
        first = values[0]
        if torch.is_tensor(first):
            out[key] = _to_python(torch.stack([v.float() for v in values]).mean(dim=0))
        elif isinstance(first, (int, float)):
            out[key] = float(sum(float(v) for v in values) / len(values))
        else:
            out[key] = _to_python(first)
    return out


def _flatten_for_log(prefix, data):
    flat = {}
    for key, value in data.items():
        if isinstance(value, list):
            for idx, item in enumerate(value):
                flat[f'{prefix}/{key}_{idx}'] = item
        elif isinstance(value, (int, float)):
            flat[f'{prefix}/{key}'] = value
    return flat
