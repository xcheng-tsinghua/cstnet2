
"""
用于测试是否能回归出几个属性, 使用额外的几何损失
进一步优化版
只挑部分有效的属性预测
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

from data_utils.datasets import ConeDataset
from models.pointnet2 import PointNet2Reg


def perpendicular_loss_normalized(axis_pred, perp_label, eps=1e-6):
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
    pred: [bs, 7]
    target: [bs, 11]
    """
    perp_pred = pred[:, :3]
    axis_pred = pred[:, 3:6]
    semi_angle_pred = pred[:, 6]

    apex_label = target[:, :3]
    axis_label = target[:, 3:6]
    perp_label = target[:, 6:9]
    semi_angle_label = target[:, 9]
    t_label = target[:, 10]

    # axis 为单位向量
    axis_pred = axis_pred / (torch.norm(axis_pred, dim=1, keepdim=True) + eps)

    # 将axis方向进行标准化
    axis_pred = canonicalize_vectors_hard(axis_pred)

    loss_axis = F.mse_loss(axis_pred, axis_label)
    loss_prep = F.mse_loss(perp_pred, perp_label)
    loss_semi_angle = F.mse_loss(semi_angle_pred, semi_angle_label)

    # axis_norm_loss = (1.0 - torch.norm(axis_pred, dim=1)).abs().mean()
    # axis_norm_loss = torch.tensor(0)

    # 原点到 foot 的向量与 axis 垂直
    foot_axis_perp_loss = perpendicular_loss_normalized(axis_pred, perp_label)

    loss_all = loss_axis + loss_prep + loss_semi_angle + foot_axis_perp_loss

    return loss_axis, loss_prep, loss_semi_angle, foot_axis_perp_loss, loss_all


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
    save_str = 'cone_loss_3'

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
    # foot(3) + axis(3) + semi_angle(1)
    classifier = PointNet2Reg(7)

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

        train_loss_axis = []
        train_loss_prep = []
        train_loss_semi_angle = []
        train_loss_foot_axis_perp = []

        train_loss = []

        classifier = classifier.train()
        for batch_id, data in tqdm(enumerate(train_loader), total=len(train_loader)):
            points = data[0].float().cuda()
            target = data[1].float().cuda()

            optimizer.zero_grad()

            pred = classifier(points)
            loss_axis, loss_prep, loss_semi_angle, foot_axis_perp_loss, loss = cone_loss(points, pred, target)

            train_loss_axis.append(loss_axis.item())
            train_loss_prep.append(loss_prep.item())
            train_loss_semi_angle.append(loss_semi_angle.item())
            train_loss_foot_axis_perp.append(foot_axis_perp_loss.item())
            train_loss.append(loss.item())

            loss.backward()
            optimizer.step()

        scheduler.step()
        torch.save(classifier.state_dict(), 'model_trained/' + save_str + '.pth')

        train_loss_axis_mean = np.mean(train_loss_axis).item()
        train_loss_prep_mean = np.mean(train_loss_prep).item()
        train_loss_semi_angle_mean = np.mean(train_loss_semi_angle).item()
        train_loss_foot_axis_perp_mean = np.mean(train_loss_foot_axis_perp).item()

        train_loss_mean = np.mean(train_loss).item()

        writer.add_scalar('train/axis', train_loss_axis_mean, epoch)
        writer.add_scalar('train/prep', train_loss_prep_mean, epoch)
        writer.add_scalar('train/semi_angle', train_loss_semi_angle_mean, epoch)
        writer.add_scalar('train/foot_axis_perp', train_loss_foot_axis_perp_mean, epoch)

        writer.add_scalar('train/loss', train_loss_mean, epoch)

        with torch.no_grad():

            test_loss_axis = []
            test_loss_prep = []
            test_loss_semi_angle = []
            test_loss_foot_axis_perp = []
            test_loss = []

            classifier = classifier.eval()
            for j, data in tqdm(enumerate(test_loader), total=len(test_loader)):
                points = data[0].float().cuda()
                target = data[1].float().cuda()

                pred = classifier(points)
                loss_axis, loss_prep, loss_semi_angle, foot_axis_perp_loss, loss = cone_loss(points, pred, target)

                test_loss_axis.append(loss_axis.item())
                test_loss_prep.append(loss_prep.item())
                test_loss_semi_angle.append(loss_semi_angle.item())
                test_loss_foot_axis_perp.append(foot_axis_perp_loss.item())
                test_loss.append(loss.item())

            test_loss_axis_mean = np.mean(test_loss_axis).item()
            test_loss_prep_mean = np.mean(test_loss_prep).item()
            test_loss_semi_angle_mean = np.mean(test_loss_semi_angle).item()
            test_loss_foot_axis_perp_mean = np.mean(test_loss_foot_axis_perp).item()
            test_loss_mean = np.mean(test_loss).item()

            writer.add_scalar('test/axis', test_loss_axis_mean, epoch)
            writer.add_scalar('test/prep', test_loss_prep_mean, epoch)
            writer.add_scalar('test/semi_angle', test_loss_semi_angle_mean, epoch)
            writer.add_scalar('test/foot_axis_perp', test_loss_foot_axis_perp_mean, epoch)
            writer.add_scalar('test/loss', test_loss_mean, epoch)

        print(f'{epoch} / {args.epoch} - {datetime.now().strftime("%Y-%m-%d %H-%M-%S")}')
        print(Fore.GREEN + '-type---train---test-')
        print(f' axis: {train_loss_axis_mean:.4f}, {test_loss_axis_mean:.4f}')
        print(f' prep: {train_loss_prep_mean:.4f}, {test_loss_prep_mean:.4f}')
        print(f'angle: {train_loss_semi_angle_mean:.4f}, {test_loss_semi_angle_mean:.4f}')

        print(f'total: {train_loss_mean:.4f}, {test_loss_mean:.4f}')


if __name__ == '__main__':
    init(autoreset=True)
    os.makedirs('log', exist_ok=True)
    os.makedirs('model_trained', exist_ok=True)
    main(parse_args())



