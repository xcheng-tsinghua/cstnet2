"""
训练约束预测模块
"""
import os
import numpy as np
import torch
from tqdm import tqdm
from datetime import datetime
import argparse
from tensorboardX import SummaryWriter
import statistics
from typing import Union
import torch.nn.functional as F

from data_utils.datasets import CstNet2Dataset
from cst_pred.cst_pcd import CstPcd
from modules.loss import EmbeddingLoss
from modules.attn_3dgcn import Attn3DGCN
from colorama import init, Fore, Back, Style


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', type=int, default=32, help='batch size in training')
    parser.add_argument('--epoch', default=2000, type=int, help='number of epoch in training')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--n_points', type=int, default=2000, help='Point Number')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--workers', type=int, default=4, help='dataloader workers')
    parser.add_argument('--is_load_weight', default='False', choices=['True', 'False'], type=str)

    parser.add_argument('--local', default='False', choices=['True', 'False'], type=str)
    parser.add_argument('--root_sever', type=str, default=rf'/opt/data/private/data_set/pcd_cstnet2/Param20K_Extend')
    parser.add_argument('--root_local', type=str, default=rf'D:\document\DeepLearning\DataSet\pcd_cstnet2\Param20K_Extend')

    args = parser.parse_args()
    return args


def write_loss_dict(writer: SummaryWriter, loss_dict: Union[dict, list[dict], tuple[dict]], step: int, tag: str):

    if isinstance(loss_dict, (list, tuple)):
        loss_dict = {k: statistics.mean([d[k] for d in loss_dict]) for k in loss_dict[0]}

    for c_key, c_value in loss_dict.items():
        writer.add_scalar(f'{tag}/{c_key}', c_value, step)


def main(args):
    save_str = 'attn_gcn3d'

    # logger
    log_dir = os.path.join('log', save_str + datetime.now().strftime("%Y-%m-%d %H-%M-%S"))
    # log_dir = os.path.join('log', save_str)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(logdir=log_dir)

    # data
    data_root = args.root_local if eval(args.local) else args.root_sever
    train_loader, test_loader = CstNet2Dataset.create_dataloader(data_root, args.bs, args.n_points, args.workers)

    # model
    predictor = Attn3DGCN().cuda()

    model_savepth = 'model_trained/' + save_str + '.pth'

    if eval(args.is_load_weight):
        try:
            predictor.load_state_dict(torch.load(model_savepth))
            print(Fore.WHITE + Back.CYAN + 'training from exist model: ' + model_savepth)
        except:
            print(Fore.RED + Back.CYAN + 'no existing model, training from scratch')
    else:
        print(Fore.BLACK + Back.CYAN + 'does not load weight, training from scratch')

    # optimizer
    optimizer = torch.optim.Adam(
        predictor.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=args.decay_rate
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)
    emb_loss = EmbeddingLoss()

    # 训练
    train_batch = 0
    test_batch = 0
    for epoch in range(args.epoch):
        train_loss_list_pmt = []
        train_loss_list_tri = []

        # 设置为训练模式，启用 dropout、batchNormalization 等模块
        predictor = predictor.train()

        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
        for batch_id, data in progress_bar:
            xyz, pmt_gt, affiliate_idx = data[0].float().cuda(), data[2].long().cuda(), data[-1].long().cuda()

            # 清空梯度，否则梯度会累加
            optimizer.zero_grad()

            # 将数据输入模型进行推理
            xyz = xyz.transpose(1, 2)  # [bs, 3, n_points]
            log_pmt, pnt_fea = predictor(xyz)

            # 计算损失
            pmt_loss = F.nll_loss(log_pmt.reshape(-1, 5), pmt_gt.view(-1))
            tri_loss = emb_loss.triplet_loss(pnt_fea, affiliate_idx.cpu().numpy())
            loss = pmt_loss + tri_loss

            # 梯度反向传播
            loss.backward()

            # 优化器根据梯度进行权重更新
            optimizer.step()

            # 记录损失
            train_loss_list_pmt.append(pmt_loss.item())
            train_loss_list_tri.append(tri_loss.item())

            writer.add_scalar('train/loss_batch_pmt', pmt_loss.item(), train_batch)
            writer.add_scalar('train/loss_batch_tri', tri_loss.item(), train_batch)
            train_batch += 1

            # 更新进度条
            progress_bar.set_postfix({
                'LossPMT': f"{pmt_loss.item():.4f}",
                'LossTri': f"{tri_loss.item():.4f}",
                'LR': f"{optimizer.param_groups[0]['lr']:.6f}"}
            )

        train_loss_epoch_pmt = np.mean(train_loss_list_pmt).item()
        train_loss_epoch_tri = np.mean(train_loss_list_tri).item()
        writer.add_scalar('train/loss_epoch_pmt', train_loss_epoch_pmt, epoch)
        writer.add_scalar('train/loss_epoch_tri', train_loss_epoch_tri, epoch)

        # 学习率调整器计数加一
        scheduler.step()

        # 保存权重
        torch.save(predictor.state_dict(), model_savepth)

        # 测试
        with torch.no_grad():
            test_loss_list_pmt = []
            test_loss_list_tri = []

            # 设置为评估模式，禁用 dropout、batchNormalization 等模块
            predictor = predictor.eval()

            progress_bar = tqdm(enumerate(test_loader), total=len(test_loader))
            for batch_id, data in progress_bar:
                xyz, pmt_gt, affiliate_idx = data[0].float().cuda(), data[2].long().cuda(), data[-1].long().cuda()

                xyz = xyz.transpose(1, 2)  # [bs, 3, n_points]
                log_pmt, pnt_fea = predictor(xyz)

                pmt_loss = F.nll_loss(log_pmt.view(-1, 5), pmt_gt.view(-1))
                tri_loss = emb_loss.triplet_loss(pnt_fea, affiliate_idx.cpu().numpy())

                test_loss_list_pmt.append(pmt_loss.item())
                test_loss_list_tri.append(tri_loss.item())

                writer.add_scalar('test/loss_batch_pmt', pmt_loss.item(), test_batch)
                writer.add_scalar('test/loss_batch_tri', tri_loss.item(), test_batch)
                test_batch += 1

                # 更新进度条
                progress_bar.set_postfix({
                    'LossPMT': f"{pmt_loss.item():.4f}",
                    'LossTri': f"{tri_loss.item():.4f}",
                    'LR': f"{optimizer.param_groups[0]['lr']:.6f}"}
                )

            test_loss_epoch_pmt = np.mean(test_loss_list_pmt).item()
            test_loss_epoch_tri = np.mean(test_loss_list_tri).item()
            writer.add_scalar('test/loss_epoch_pmt', test_loss_epoch_pmt, epoch)
            writer.add_scalar('test/loss_epoch_tri', test_loss_epoch_tri, epoch)

            print(f'{epoch} / {args.epoch}: train_loss_epoch_pmt: {train_loss_epoch_pmt}, train_loss_epoch_tri: {train_loss_epoch_tri}. test_loss_epoch_pmt: {test_loss_epoch_pmt}, test_loss_epoch_tri: {test_loss_epoch_tri}')

    writer.close()


if __name__ == '__main__':
    init(autoreset=True)
    main(parse_args())

