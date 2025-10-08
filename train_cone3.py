
"""
用于测试是否能回归出几个属性, 使用额外的几何损失
"""

import torch
from datetime import datetime
import argparse
from tqdm import tqdm
import numpy as np
import os
from torch.nn import functional as F
from tensorboardX import SummaryWriter
from colorama import Fore, Back, init
import statistics

from data_utils.datasets import ConeDataset
from models.pointnet2 import PointNet2Reg


def perpendicular_loss_normalized(axis_pred, perp_label, eps=1e-8):
    """
    原点到垂足的向量与圆锥轴线方向垂直
    axis_pred: 预测主轴 [bs, 3]
    perp_label: 预测垂足 [bs, 3]
    """
    axis_pred = axis_pred / (torch.norm(axis_pred, dim=1, keepdim=True) + eps)
    perp_label = perp_label / (torch.norm(perp_label, dim=1, keepdim=True) + eps)
    dot = torch.sum(axis_pred * perp_label, dim=1)
    loss = torch.mean(dot ** 2)
    return loss


def point_on_cone_loss(xyz, axis_pred, semi_angle_pred, apex_pred):
    """
    :param xyz: [bs, point, 3]
    :param axis_pred: [bs, 3]
    :param semi_angle_pred: [bs,]
    :param apex_pred: [bs, 3]
    """
    # 从锥角到圆锥面上的点构成的向量与主方向之间的夹角等于主尺寸
    apex_to_xyz = xyz - apex_pred.unsqueeze(1)  # [bs, point, 3]
    dot1 = torch.einsum('bpc,bc->bp', apex_to_xyz, axis_pred)

    axis_pred_norm = axis_pred.norm(dim=1).unsqueeze(1)
    apex_to_xyz_norm = apex_to_xyz.norm(dim=2)
    dot2 = axis_pred_norm * apex_to_xyz_norm * torch.cos(semi_angle_pred.unsqueeze(1))
    semi_angle = (dot1 - dot2).abs().mean()

    return semi_angle


def safe_normalize(v, eps=1e-6, min_norm=0.1):
    """
    v: [bs, point, 3]
    防止向量长度过短
    """
    norm = v.norm(dim=-1, keepdim=True)
    norm = torch.clamp(norm, min=eps)
    v_normalized = v / norm

    # 只惩罚过短向量，防止不稳定
    length_loss = torch.relu(min_norm - norm).mean()
    return v_normalized, length_loss


def canonicalize_vectors_hard(v, eps=1e-6):
    """
    对每个三维向量执行字典序标准化反转，带容差判断。
    v: [bs, 3]
    eps: 浮点容差，用于判断“是否为 0”
    """
    x, y, z = v[:, 0], v[:, 1], v[:, 2]

    # “约等于 0”的判断
    z_zero = torch.abs(z) < eps
    y_zero = torch.abs(y) < eps
    x_zero = torch.abs(x) < eps

    # 按字典序判断是否要反转
    mask = (z < -eps) | \
           (z_zero & (y < -eps)) | \
           (z_zero & y_zero & (x < -eps))

    v_flipped = v.clone()
    v_flipped[mask] = -v_flipped[mask]
    return v_flipped


def cone_loss(xyz, pred, target, eps=1e-8):
    """
    xyz: [bs, n, 3]
    pred: [bs, 8]
    target: [bs, 11]
    """
    perp_pred = pred[:, :3]
    axis_pred = pred[:, 3:6]
    semi_angle_pred = pred[:, 6]
    beta_pred = pred[:, 7]

    apex_label = target[:, :3]
    axis_label = target[:, 3:6]
    perp_label = target[:, 6:9]
    semi_angle_label = target[:, 9]
    t_label = target[:, 10]

    # axis 为单位向量
    # axis_pred = axis_pred / (torch.norm(axis_pred, dim=1, keepdim=True) + eps)

    axis_pred, loss_axis_len = safe_normalize(axis_pred)

    # 将axis方向进行标准化
    axis_pred = canonicalize_vectors_hard(axis_pred)

    # beta = log(1 + t)
    # t = exp(beta) - 1
    # apex = perp_foot + t * axis
    t_pred = torch.exp(beta_pred) - 1.0
    apex_pred = perp_pred + t_pred.unsqueeze(1) * axis_pred

    loss_apex = F.mse_loss(apex_pred, apex_label)
    loss_axis = F.mse_loss(axis_pred, axis_label)
    loss_prep = F.mse_loss(perp_pred, perp_label)
    loss_semi_angle = F.mse_loss(semi_angle_pred, semi_angle_label)
    loss_t = F.mse_loss(t_pred, t_label)

    # 原点到 foot 的向量与 axis 垂直
    foot_axis_perp_loss = perpendicular_loss_normalized(axis_pred, perp_label)

    # 点位于圆锥上的几何损失
    on_cone_loss = point_on_cone_loss(xyz, axis_pred, semi_angle_pred, apex_pred)

    loss = loss_apex + loss_axis + loss_prep + loss_semi_angle + loss_t + foot_axis_perp_loss + on_cone_loss + loss_axis_len
    loss_branch = {'apex': loss_apex.item(),
                   'axis': loss_axis.item(),
                   'prep': loss_prep.item(),
                   'angle': loss_semi_angle.item(),
                   't': loss_t.item(),
                   'foot_prep': foot_axis_perp_loss.item(),
                   'on_cone': on_cone_loss.item(),
                   'axis_len': loss_axis_len.item(),
                   }

    return loss_branch, loss


def loss_branch_process_and_save(loss_branches: list[dict], writer, epoch, tag):
    """
    将 loss 求均值并且保存在 writer 中
    """
    mean_dict = {
        key: statistics.mean(d[key] for d in loss_branches)
        for key in loss_branches[0].keys()
    }

    for key, value in mean_dict.items():
        writer.add_scalar(f'{tag}/{key}', value, epoch)

    formatted = {k: f"{v:.6f}" for k, v in mean_dict.items()}
    print(formatted)


def parse_args():

    parser = argparse.ArgumentParser('training')
    parser.add_argument('--bs', type=int, default=200, help='batch size in training')
    parser.add_argument('--epoch', default=100000, type=int, help='number of epoch in training')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')

    parser.add_argument('--n_point', type=int, default=2000, help='Point Number')
    parser.add_argument('--is_load_weight', default='False', choices=['True', 'False'], type=str)

    parser.add_argument('--local', default='False', choices=['True', 'False'], type=str)
    parser.add_argument('--root_sever', type=str, default=rf'/opt/data/private/data_set/pcd_cstnet2/cone')
    parser.add_argument('--root_local', type=str, default=rf'D:\document\DeepLearning\DataSet\pcd_cstnet2\cone')

    return parser.parse_args()


def main(args):
    # parameters
    save_str = 'cone_geom_loss2_squeeze_log_'

    # logger
    log_dir = os.path.join('log', save_str + datetime.now().strftime("%Y-%m-%d %H-%M-%S"))
    # log_dir = os.path.join('log', save_str)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(logdir=log_dir)

    # datasets
    if args.local == 'True':
        data_root = args.root_local
    else:
        data_root = args.root_sever

    train_dataset = ConeDataset(root=data_root, npoints=args.n_point, is_train=True)
    test_dataset = ConeDataset(root=data_root, npoints=args.n_point, is_train=False)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=4, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.bs, shuffle=False, num_workers=4)

    # loading model
    # foot(3) + axis(3) + semi_angle(1) + beta(1)
    classifier = PointNet2Reg(8)

    if eval(args.is_load_weight):
        model_savepth = 'model_trained/' + save_str + '.pth'
        try:
            classifier.load_state_dict(torch.load(model_savepth))
            print('training from exist model: ' + model_savepth)
        except:
            print('no existing model, training from scratch')
    else:
        print('does not load weight, training from scratch')

    classifier = classifier.cuda()

    # optimizer
    optimizer = torch.optim.Adam(
        classifier.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=args.decay_rate
    )

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.95)

    # training
    for epoch in range(args.epoch):

        train_loss_branch = []
        train_loss = []

        classifier = classifier.train()
        for batch_id, data in tqdm(enumerate(train_loader), total=len(train_loader)):
            points = data[0].float().cuda()
            target = data[1].float().cuda()

            optimizer.zero_grad()

            pred = classifier(points)
            loss_branch, loss = cone_loss(points, pred, target)

            train_loss_branch.append(loss_branch)
            train_loss.append(loss.item())

            loss.backward()
            optimizer.step()

        scheduler.step()
        torch.save(classifier.state_dict(), 'model_trained/' + save_str + '.pth')

        with torch.no_grad():

            test_loss_branch = []
            test_loss = []

            classifier = classifier.eval()
            for j, data in tqdm(enumerate(test_loader), total=len(test_loader)):
                points = data[0].float().cuda()
                target = data[1].float().cuda()

                pred = classifier(points)
                loss_branch, loss = cone_loss(points, pred, target)

                test_loss_branch.append(loss_branch)
                test_loss.append(loss.item())

        print(f'--> {epoch} / {args.epoch} - {datetime.now().strftime("%Y-%m-%d %H-%M-%S")} <--')

        loss_branch_process_and_save(train_loss_branch, writer, epoch, 'train')
        train_loss_mean = np.mean(train_loss).item()
        writer.add_scalar('train/loss', train_loss_mean, epoch)

        loss_branch_process_and_save(test_loss_branch, writer, epoch, 'test')
        test_loss_mean = np.mean(test_loss).item()
        writer.add_scalar('test/loss', test_loss_mean, epoch)

        print(f'total: {train_loss_mean:.4f}, {test_loss_mean:.4f}')


if __name__ == '__main__':
    init(autoreset=True)
    os.makedirs('log', exist_ok=True)
    os.makedirs('model_trained', exist_ok=True)
    main(parse_args())



