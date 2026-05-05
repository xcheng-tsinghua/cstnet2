import torch
from tqdm import tqdm
import torch.nn.functional as F
import einops
from contextlib import nullcontext
from colorama import Fore, Back
from functional.loss import discriminative_loss, evaluate_clustering
import json
from time import time


class CstPredTrainer(object):
    """
    用于训练约束预测模块
    """
    def __init__(self, model, train_loader, test_loader, model_savepth, log_savepth, max_epoch, lr, is_load_weight, save_str, wandb_run=None):
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
        self.use_amp = False
        self.amp_dtype = torch.bfloat16

        self.optimizer = None
        self.scheduler = None
        self.make_optimizer_and_schedule(lr)
        self.load_weight(is_load_weight)

        self.save_dict_train = {
            'prim_loss': [],
            'clus_loss': [],
            'prim_acc': [],
            'clus_acc': [],
            'clus_nmi': [],
            'clus_ari': [],
        }

        self.save_dict_test = {
            'prim_loss': [],
            'clus_loss': [],
            'prim_acc': [],
            'clus_acc': [],
            'clus_nmi': [],
            'clus_ari': [],
        }

    def make_optimizer_and_schedule(self, lr):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=1e-4
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.9)

    def load_weight(self, is_load_weight):
        if is_load_weight:
            try:
                self.model.load_state_dict(torch.load(self.model_savepth, map_location=self.device))
                print(Fore.WHITE + Back.CYAN + 'training from exist model: ' + self.model_savepth)
            except:
                print(Fore.RED + Back.CYAN + 'no existing model, training from scratch')
        else:
            print(Fore.BLACK + Back.CYAN + 'does not load weight, training from scratch')

    def start(self):
        for epoch in range(self.max_epoch):
            # 训练一个 epoch
            start_time = time()
            pl, cl, pa, ca, cn, cr = self.process_epoch(epoch, True)
            self.append_save_dict(pl, cl, pa, ca, cn, cr, True)
            end_time = time()
            train_time = end_time - start_time
            print(Fore.BLUE + f'训练耗时: {end_time - start_time:.4f} 秒')

            # 测试一个 epoch
            start_time = time()
            pl, cl, pa, ca, cn, cr = self.process_epoch(epoch, False)
            self.append_save_dict(pl, cl, pa, ca, cn, cr, False)
            end_time = time()
            test_time = end_time - start_time
            print(Fore.BLUE + f'评估耗时: {end_time - start_time:.4f} 秒')

            if self.wandb_run is not None:
                self.wandb_run.log({
                    'epoch': epoch,
                    'lr': self.optimizer.param_groups[0]['lr'],
                    'time/train_sec': train_time,
                    'time/test_sec': test_time,
                    'train/prim_loss': self.save_dict_train['prim_loss'][-1],
                    'train/clus_loss': self.save_dict_train['clus_loss'][-1],
                    'train/prim_acc': self.save_dict_train['prim_acc'][-1],
                    'train/clus_acc': self.save_dict_train['clus_acc'][-1],
                    'train/clus_nmi': self.save_dict_train['clus_nmi'][-1],
                    'train/clus_ari': self.save_dict_train['clus_ari'][-1],
                    'test/prim_loss': self.save_dict_test['prim_loss'][-1],
                    'test/clus_loss': self.save_dict_test['clus_loss'][-1],
                    'test/prim_acc': self.save_dict_test['prim_acc'][-1],
                    'test/clus_acc': self.save_dict_test['clus_acc'][-1],
                    'test/clus_nmi': self.save_dict_test['clus_nmi'][-1],
                    'test/clus_ari': self.save_dict_test['clus_ari'][-1],
                }, step=epoch)

            # 保存权重和训练数据
            self.save()

            # 学习率调整器计数加一
            self.scheduler.step()

    def save(self):
        print(Fore.GREEN + f'save model dict to {self.model_savepth}')
        torch.save(self.model.state_dict(), self.model_savepth)

        print(Fore.GREEN + f'save training dict to {self.log_savepth}')
        with open(self.log_savepth, 'w') as f:
            json.dump({'train': self.save_dict_train, 'test': self.save_dict_test}, f, ensure_ascii=False, indent=4)

    def append_save_dict(self, pl, cl, pa, ca, cn, cr, is_train):
        if is_train:
            self.save_dict_train['prim_loss'].append(pl)
            self.save_dict_train['clus_loss'].append(cl)
            self.save_dict_train['prim_acc'].append(pa)
            self.save_dict_train['clus_acc'].append(ca)
            self.save_dict_train['clus_nmi'].append(cn)
            self.save_dict_train['clus_ari'].append(cr)
            print_start = 'train_metrics: '

        else:
            self.save_dict_test['prim_loss'].append(pl)
            self.save_dict_test['clus_loss'].append(cl)
            self.save_dict_test['prim_acc'].append(pa)
            self.save_dict_test['clus_acc'].append(ca)
            self.save_dict_test['clus_nmi'].append(cn)
            self.save_dict_test['clus_ari'].append(cr)
            print_start = 'test_metrics: '

        print(print_start + f'prim_loss: {pl:.6f}. clus_loss: {cl:.6f} .'
                            f'prim_acc: {pa:.4f}. clus_acc: {ca:.4f}. clus_nmi: {cn:.4f}. clus_ari: {cr:.4f}')

    def process_epoch(self, current_epoch, is_train):
        pl_lst = []
        cl_lst = []
        pa_lst = []
        ca_lst = []
        cn_lst = []
        cr_lst = []

        if is_train:
            print('training epoch')
            self.model.train()
            loader = self.train_loader

        else:
            print('testing epoch')
            self.model.eval()
            loader = self.test_loader

        progress_bar = tqdm(loader, total=len(loader), desc=f'[{current_epoch}/{self.max_epoch}]{self.save_str}')
        for data in progress_bar:
            pmt_loss, cluster_loss, pmt_acc, acc, nmi, ari = self.process_batch(data, is_train)

            # 更新进度条
            progress_bar.set_postfix({
                'pmt_acc': f'{pmt_acc:.4f}',
                'cluster_ari': f'{ari:.4f}',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.6f}'}
            )

            pl_lst.append(pmt_loss.float())
            cl_lst.append(cluster_loss.float())
            pa_lst.append(pmt_acc.float())
            ca_lst.append(acc.float())
            cn_lst.append(nmi.float())
            cr_lst.append(ari.float())

        pl = torch.stack(pl_lst).mean().item()
        cl = torch.stack(cl_lst).mean().item()
        pa = torch.stack(pa_lst).mean().item()
        ca = torch.stack(ca_lst).mean().item()
        cn = torch.stack(cn_lst).mean().item()
        cr = torch.stack(cr_lst).mean().item()
        return pl, cl, pa, ca, cn, cr

    def process_batch(self, data_batch, is_train):
        """
        记录一个 batch 内的操作
        Args:
            data_batch:
            is_train:

        Returns:

        """
        with torch.set_grad_enabled(is_train):
            # 清空梯度，否则梯度会累加
            if is_train:
                self.optimizer.zero_grad(set_to_none=True)

            xyz = data_batch[0].float().to(self.device, non_blocking=True)
            pmt_gt = data_batch[2].long().to(self.device, non_blocking=True)
            affiliate_idx = data_batch[-1].long().to(self.device, non_blocking=True)

            # 将数据输入模型进行推理
            amp_ctx = torch.autocast(device_type='cuda', dtype=self.amp_dtype) if self.use_amp else nullcontext()
            with amp_ctx:
                pnt_fea, log_pmt = self.model(xyz)

                # 计算损失
                pmt_loss, cluster_loss = compute_loss(pnt_fea.float(), affiliate_idx, log_pmt.float(), pmt_gt)
                loss = pmt_loss + cluster_loss

            if (not torch.isfinite(pnt_fea).all()) or (not torch.isfinite(log_pmt).all()):
                print(Fore.RED + 'non-finite model output, skip this batch')
                z = torch.zeros((), device=self.device, dtype=torch.float32)
                return z, z, z, z, z, z
            if not torch.isfinite(loss):
                print(Fore.RED + f'non-finite loss, skip this batch: pmt={float(pmt_loss):.6f}, clus={float(cluster_loss):.6f}')
                z = torch.zeros((), device=self.device, dtype=torch.float32)
                return z, z, z, z, z, z

            # 训练时需要梯度反向传播，计算各参数梯度，以及根据梯度更新权重
            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                # 优化器根据梯度进行权重更新
                self.optimizer.step()

            # 聚类指标（与测试阶段同一套 Torch 实现；训练时也计算，便于对齐观察）
            with torch.no_grad():
                acc, nmi, ari = evaluate_clustering(affiliate_idx, pnt_fea)
            pmt_acc = compute_seg_acc(log_pmt, pmt_gt)

            return pmt_loss.detach(), cluster_loss.detach(), pmt_acc.detach(), acc.detach(), nmi.detach(), ari.detach()


def compute_loss(pnt_fea, affiliate_idx, log_pmt, pmt_gt):
    """

    Args:
        pnt_fea: [bs, n_point, emb]
        affiliate_idx: [bs, n_point]
        log_pmt: [bs, n_point, emb]
        pmt_gt: [bs, n_point]

    Returns:

    """
    pnt_fea = torch.nan_to_num(pnt_fea.float(), nan=0.0, posinf=1e4, neginf=-1e4)
    log_pmt = torch.nan_to_num(log_pmt.float(), nan=0.0, posinf=1e4, neginf=-1e4)
    tri_loss = discriminative_loss(pnt_fea, affiliate_idx)

    log_pmt_fit_loss = einops.rearrange(log_pmt, 'b n c -> (b n) c')
    pmt_gt_fit_loss = einops.rearrange(pmt_gt, 'b n -> (b n)')
    pmt_loss = F.nll_loss(log_pmt_fit_loss, pmt_gt_fit_loss)
    pmt_loss = torch.nan_to_num(pmt_loss, nan=0.0, posinf=1e4, neginf=-1e4)

    return pmt_loss, tri_loss


def compute_seg_acc(pred, label):
    """

    Args:
        pred: [bs, n_point, emb]
        label: [bs, n_point]

    Returns:

    """
    bs, n_points, _ = pred.size()
    n_items_batch = bs * n_points

    choice_meta_type = pred.argmax(dim=2)
    correct_meta_type = choice_meta_type.eq(label).sum()

    seg_acc = correct_meta_type / n_items_batch
    return seg_acc

