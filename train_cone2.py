
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
    # bs, n_points, _ = xyz.size()
    # xyz = xyz.view(bs * n_points, -1)
    # axis_pred = axis_pred.unsqueeze(1).repeat(1, n_points, 1).view(bs * n_points, -1)
    # semi_angle_pred = semi_angle_pred.unsqueeze(1).repeat(1, n_points).view(bs * n_points, -1)
    # apex_pred = apex_pred.unsqueeze(1).repeat(1, n_points, 1).view(bs * n_points, -1)

    # 从锥角到圆锥面上的点构成的向量与主方向之间的夹角等于主尺寸
    apex_to_xyz = xyz - apex_pred.unsqueeze(1)  # [bs, point, 3]
    dot1 = torch.einsum('bpc,bc->bp', apex_to_xyz, axis_pred)

    axis_pred_norm = axis_pred.norm(dim=1).unsqueeze(1)
    apex_to_xyz_norm = apex_to_xyz.norm(dim=2)
    dot2 = axis_pred_norm * apex_to_xyz_norm * torch.cos(semi_angle_pred.unsqueeze(1))
    semi_angle = (dot1 - dot2).abs().mean()

    return semi_angle


def cone_loss(xyz, pred, target):
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

    # 先使 beta 处于 [-pi/2, pi/2] 之间，以符合反正切定义域
    beta_pred = (torch.pi / 2) * torch.tanh(beta_pred)

    # 预测的 t, beta = arctan(t), t = tan(beta)
    t_pred = torch.tan(beta_pred)

    # apex = perp_foot + t * axis
    apex_pred = perp_pred + t_pred.unsqueeze(1) * axis_pred

    loss_apex = F.mse_loss(apex_pred, apex_label)
    loss_axis = F.mse_loss(axis_pred, axis_label)
    loss_prep = F.mse_loss(perp_pred, perp_label)
    loss_semi_angle = F.mse_loss(semi_angle_pred, semi_angle_label)
    loss_beta = F.mse_loss(t_pred, t_label)

    # axis 为单位向量
    axis_norm_loss = (1.0 - torch.norm(axis_pred, dim=1)).abs().mean()

    # 原点到 foot 的向量与 axis 垂直
    foot_axis_perp_loss = perpendicular_loss_normalized(axis_pred, perp_label)

    # 点位于圆锥上的几何损失
    on_cone_loss = point_on_cone_loss(xyz, axis_pred, semi_angle_pred, apex_pred)

    loss_all = loss_apex + loss_axis + loss_prep + loss_semi_angle + loss_beta + axis_norm_loss + foot_axis_perp_loss + on_cone_loss

    return loss_apex, loss_axis, loss_prep, loss_semi_angle, loss_beta, axis_norm_loss, foot_axis_perp_loss, on_cone_loss, loss_all


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
    save_str = 'cone_geom_loss_alt'

    # logger
    # log_dir = os.path.join('log', save_str + datetime.now().strftime("%Y-%m-%d %H-%M-%S"))
    log_dir = os.path.join('log', save_str)
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

        train_loss_apex = []
        train_loss_axis = []
        train_loss_prep = []
        train_loss_semi_angle = []

        train_loss_beta = []
        train_loss_axis_norm = []
        train_loss_foot_axis_perp = []
        train_loss_geom_cone = []

        train_loss = []

        classifier = classifier.train()
        for batch_id, data in tqdm(enumerate(train_loader), total=len(train_loader)):
            points = data[0].float().cuda()
            target = data[1].float().cuda()

            optimizer.zero_grad()

            pred = classifier(points)
            loss_apex, loss_axis, loss_prep, loss_semi_angle, loss_beta, axis_norm_loss, foot_axis_perp_loss, on_cone_loss, loss = cone_loss(points, pred, target)

            train_loss_apex.append(loss_apex.item())
            train_loss_axis.append(loss_axis.item())
            train_loss_prep.append(loss_prep.item())
            train_loss_semi_angle.append(loss_semi_angle.item())

            train_loss_beta.append(loss_beta.item())
            train_loss_axis_norm.append(axis_norm_loss.item())
            train_loss_foot_axis_perp.append(foot_axis_perp_loss.item())
            train_loss_geom_cone.append(on_cone_loss.item())

            train_loss.append(loss.item())

            loss.backward()
            optimizer.step()

        scheduler.step()
        torch.save(classifier.state_dict(), 'model_trained/' + save_str + '.pth')

        train_loss_apex_mean = np.mean(train_loss_apex).item()
        train_loss_axis_mean = np.mean(train_loss_axis).item()
        train_loss_prep_mean = np.mean(train_loss_prep).item()
        train_loss_semi_angle_mean = np.mean(train_loss_semi_angle).item()

        train_loss_beta_mean = np.mean(train_loss_beta).item()
        train_loss_axis_norm_mean = np.mean(train_loss_axis_norm).item()
        train_loss_foot_axis_perp_mean = np.mean(train_loss_foot_axis_perp).item()
        train_loss_geom_mean = np.mean(train_loss_geom_cone).item()

        train_loss_mean = np.mean(train_loss).item()

        writer.add_scalar('train/apex', train_loss_apex_mean, epoch)
        writer.add_scalar('train/axis', train_loss_axis_mean, epoch)
        writer.add_scalar('train/prep', train_loss_prep_mean, epoch)
        writer.add_scalar('train/semi_angle', train_loss_semi_angle_mean, epoch)

        writer.add_scalar('train/beta', train_loss_beta_mean, epoch)
        writer.add_scalar('train/axis_norm', train_loss_axis_norm_mean, epoch)
        writer.add_scalar('train/foot_axis_perp', train_loss_foot_axis_perp_mean, epoch)
        writer.add_scalar('train/geom', train_loss_geom_mean, epoch)

        writer.add_scalar('train/loss', train_loss_mean, epoch)

        with torch.no_grad():

            test_loss_apex = []
            test_loss_axis = []
            test_loss_prep = []
            test_loss_semi_angle = []

            test_loss_beta = []
            test_loss_axis_norm = []
            test_loss_foot_axis_perp = []
            test_loss_geom_cone = []

            test_loss = []

            classifier = classifier.eval()
            for j, data in tqdm(enumerate(test_loader), total=len(test_loader)):
                points = data[0].float().cuda()
                target = data[1].float().cuda()

                pred = classifier(points)
                loss_apex, loss_axis, loss_prep, loss_semi_angle, loss_beta, axis_norm_loss, foot_axis_perp_loss, on_cone_loss, loss = cone_loss(points, pred, target)

                test_loss_apex.append(loss_apex.item())
                test_loss_axis.append(loss_axis.item())
                test_loss_prep.append(loss_prep.item())
                test_loss_semi_angle.append(loss_semi_angle.item())

                test_loss_beta.append(loss_beta.item())
                test_loss_axis_norm.append(axis_norm_loss.item())
                test_loss_foot_axis_perp.append(foot_axis_perp_loss.item())
                test_loss_geom_cone.append(on_cone_loss.item())

                test_loss.append(loss.item())

            test_loss_apex_mean = np.mean(test_loss_apex).item()
            test_loss_axis_mean = np.mean(test_loss_axis).item()
            test_loss_prep_mean = np.mean(test_loss_prep).item()
            test_loss_semi_angle_mean = np.mean(test_loss_semi_angle).item()

            test_loss_beta_mean = np.mean(test_loss_beta).item()
            test_loss_axis_norm_mean = np.mean(test_loss_axis_norm).item()
            test_loss_foot_axis_perp_mean = np.mean(test_loss_foot_axis_perp).item()
            test_loss_geom_mean = np.mean(test_loss_geom_cone).item()

            test_loss_mean = np.mean(test_loss).item()

            writer.add_scalar('test/apex', test_loss_apex_mean, epoch)
            writer.add_scalar('test/axis', test_loss_axis_mean, epoch)
            writer.add_scalar('test/prep', test_loss_prep_mean, epoch)
            writer.add_scalar('test/semi_angle', test_loss_semi_angle_mean, epoch)

            writer.add_scalar('test/beta', test_loss_beta_mean, epoch)
            writer.add_scalar('test/axis_norm', test_loss_axis_norm_mean, epoch)
            writer.add_scalar('test/foot_axis_perp', test_loss_foot_axis_perp_mean, epoch)
            writer.add_scalar('test/geom', test_loss_geom_mean, epoch)

            writer.add_scalar('test/loss', test_loss_mean, epoch)

        print(f'{epoch} / {args.epoch} - {datetime.now().strftime("%Y-%m-%d %H-%M-%S")}')
        print(Fore.GREEN + '-type---train---test-')
        print(f' apex: {train_loss_apex_mean:.4f}, {test_loss_apex_mean:.4f}')
        print(f' axis: {train_loss_axis_mean:.4f}, {test_loss_axis_mean:.4f}')
        print(f' prep: {train_loss_prep_mean:.4f}, {test_loss_prep_mean:.4f}')
        print(f'angle: {train_loss_semi_angle_mean:.4f}, {test_loss_semi_angle_mean:.4f}')

        print(f' beta: {train_loss_beta_mean:.4f}, {test_loss_beta_mean:.4f}')
        print(f'axisn: {train_loss_axis_norm_mean:.4f}, {test_loss_axis_norm_mean:.4f}')
        print(f'fprep: {train_loss_foot_axis_perp_mean:.4f}, {test_loss_foot_axis_perp_mean:.4f}')
        print(f' geom: {train_loss_geom_mean:.4f}, {test_loss_geom_mean:.4f}')

        print(f'total: {train_loss_mean:.4f}, {test_loss_mean:.4f}')


if __name__ == '__main__':
    init(autoreset=True)
    os.makedirs('log', exist_ok=True)
    os.makedirs('model_trained', exist_ok=True)
    main(parse_args())



