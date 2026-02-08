import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F
import einops
from colorama import init, Fore, Back
from functional.loss import discriminative_loss, evaluate_clustering
import json


class CstPredTrainer(object):
    """
    用于训练约束预测模块
    """
    def __init__(self, model, train_loader, test_loader, model_savepth, log_savepth, max_epoch, lr, is_load_weight, save_str):
        super().__init__()
        print(f'weight save to: {model_savepth}, log save to: {log_savepth}')

        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model_savepth = model_savepth
        self.log_savepth = log_savepth
        self.max_epoch = max_epoch
        self.save_str = save_str

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
                self.model.load_state_dict(torch.load(self.model_savepth))
                print(Fore.WHITE + Back.CYAN + 'training from exist model: ' + self.model_savepth)
            except:
                print(Fore.RED + Back.CYAN + 'no existing model, training from scratch')
        else:
            print(Fore.BLACK + Back.CYAN + 'does not load weight, training from scratch')

    def start(self):
        for epoch in range(self.max_epoch):
            # 训练一个 epoch
            pl, cl, pa, ca, cn, cr = self.process_epoch(epoch, True)
            self.append_save_dict(pl, cl, pa, ca, cn, cr, True)

            # 测试一个 epoch
            pl, cl, pa, ca, cn, cr = self.process_epoch(epoch, False)
            self.append_save_dict(pl, cl, pa, ca, cn, cr, False)

            # 保存权重和训练数据
            self.save()

            # 学习率调整器计数加一
            self.scheduler.step()

    def save(self):
        print('save model dict')
        torch.save(self.model.state_dict(), self.model_savepth)

        print('save training dict')
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

        print(print_start + f'''
   prim_loss: {pl}
   clus_loss: {cl}
-> prim_acc: {pa}
   clus_acc: {ca}
   clus_nmi: {cn}
-> clus_ari: {cr}
''')

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

        progress_bar = tqdm(loader, total=len(loader), desc=f'[{current_epoch}/{self.max_epoch}|{self.save_str}]:')
        for data in progress_bar:
            pmt_loss, cluster_loss, pmt_acc, acc, nmi, ari = self.process_batch(data, is_train)

            # 更新进度条
            progress_bar.set_postfix({
                'pmt_acc': f'{pmt_acc:.4f}',
                'cluster_ari': f'{ari:.4f}',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.6f}'}
            )

            pl_lst.append(pmt_loss)
            cl_lst.append(cluster_loss)
            pa_lst.append(pmt_acc)
            ca_lst.append(acc)
            cn_lst.append(nmi)
            cr_lst.append(ari)

        pl = np.mean(pl_lst)
        cl = np.mean(cl_lst)
        pa = np.mean(pa_lst)
        ca = np.mean(ca_lst)
        cn = np.mean(cn_lst)
        cr = np.mean(cr_lst)

        return pl, cl, pa, ca, cn, cr

    def process_batch(self, data_batch, is_train):
        """
        记录一个 batch 内的操作
        Args:
            data_batch:
            progress_bar:
            is_train:

        Returns:

        """
        with torch.set_grad_enabled(is_train):
            # 清空梯度，否则梯度会累加
            if is_train:
                self.optimizer.zero_grad()

            xyz = data_batch[0].float().cuda()
            pmt_gt = data_batch[2].long().cuda()
            affiliate_idx = data_batch[-1].long().cuda()

            # 将数据输入模型进行推理
            pnt_fea, log_pmt = self.model(xyz)

            # 计算损失
            pmt_loss, cluster_loss = compute_loss(pnt_fea, affiliate_idx, log_pmt, pmt_gt)

            # 训练时需要梯度反向传播，计算各参数梯度，以及根据梯度更新权重
            if is_train:
                # 梯度反向传播
                loss = pmt_loss + cluster_loss
                loss.backward()

                # 优化器根据梯度进行权重更新
                self.optimizer.step()

            # 计算准确率
            acc, nmi, ari = evaluate_clustering(affiliate_idx, pnt_fea)
            pmt_acc = compute_seg_acc(log_pmt, pmt_gt)

            return pmt_loss.item(), cluster_loss.item(), pmt_acc.item(), acc.item(), nmi.item(), ari.item()


def compute_loss(pnt_fea, affiliate_idx, log_pmt, pmt_gt):
    """

    Args:
        pnt_fea: [bs, n_point, emb]
        affiliate_idx: [bs, n_point]
        log_pmt: [bs, n_point, emb]
        pmt_gt: [bs, n_point]

    Returns:

    """
    tri_loss = discriminative_loss(pnt_fea, affiliate_idx)

    log_pmt_fit_loss = einops.rearrange(log_pmt, 'b n c -> (b n) c')
    pmt_gt_fit_loss = einops.rearrange(pmt_gt, 'b n -> (b n)')
    pmt_loss = F.nll_loss(log_pmt_fit_loss, pmt_gt_fit_loss)

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

    choice_meta_type = pred.data.max(2)[1]
    correct_meta_type = choice_meta_type.eq(label.data).cpu().sum()

    seg_acc = correct_meta_type / n_items_batch
    return seg_acc

