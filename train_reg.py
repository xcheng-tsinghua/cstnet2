
"""
用于测试是否能回归出几个属性
"""

import torch
from datetime import datetime
import argparse
from tqdm import tqdm
import numpy as np
import os
from tensorboardX import SummaryWriter

from data_utils.datasets import RegressionDataset
from models.pointnet2 import PointNet2Reg


def plane_loss(points, pred_plane, lam=0.1):
    """
    points: [bs, n_point, 3]
    pred_plane: [bs, 4]  (a,b,c,d)
    """
    a, b, c, d = pred_plane[:, 0], pred_plane[:, 1], pred_plane[:, 2], pred_plane[:, 3]
    normal = torch.stack([a, b, c], dim=-1)  # [bs, 3]

    # [bs, n_point]
    residuals = (points @ normal.unsqueeze(-1)).squeeze(-1) + d.unsqueeze(-1)
    loss_point = (residuals ** 2).mean()

    # normalize constraint
    norm = torch.norm(normal, dim=-1)
    loss_norm = ((norm - 1) ** 2).mean()

    return loss_point + lam * loss_norm


def foot_loss(points, pred_foot):
    """
    points: [bs, n_point, 3]
    pred_plane: [bs, 3]  (-ad,-bd,-cd)
    """
    points_to_foot = points - pred_foot.unsqueeze(1)  # [bs, n, 3]

    # 点到垂足的向量与原点到垂足垂直
    residuals = torch.abs(torch.bmm(points_to_foot, pred_foot.unsqueeze(2)).squeeze(2)).mean()

    return residuals


def parse_args():

    parser = argparse.ArgumentParser('training')
    parser.add_argument('--bs', type=int, default=200, help='batch size in training')
    parser.add_argument('--epoch', default=100000, type=int, help='number of epoch in training')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')

    parser.add_argument('--n_point', type=int, default=2000, help='Point Number')
    parser.add_argument('--is_load_weight', default='False', choices=['True', 'False'], type=str)

    parser.add_argument('--local', default='False', choices=['True', 'False'], type=str)
    parser.add_argument('--root_sever', type=str, default=rf'/opt/data/private/data_set/pcd_cstnet2/plane_test')
    parser.add_argument('--root_local', type=str, default=rf'D:\document\DeepLearning\DataSet\pcd_cstnet2\plane_test')

    return parser.parse_args()


def main(args):
    # parameters
    save_str = 'pointnet_reg'

    # logger
    log_dir = os.path.join('log', save_str + datetime.now().strftime("%Y-%m-%d %H-%M-%S"))
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(logdir=log_dir)

    # datasets
    if args.local == 'True':
        data_root = args.root_local
    else:
        data_root = args.root_sever

    train_dataset = RegressionDataset(root=data_root, npoints=args.n_point, is_train=True)
    test_dataset = RegressionDataset(root=data_root, npoints=args.n_point, is_train=False)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=4, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.bs, shuffle=False, num_workers=4)

    # loading model
    classifier = PointNet2Reg()

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
        logstr_epoch = f'Epoch_{epoch + 1}/{args.epoch}:'

        train_loss = []
        classifier = classifier.train()
        for batch_id, data in tqdm(enumerate(train_loader), total=len(train_loader)):
            points = data[0].float().cuda()
            target = data[1].float().cuda()

            optimizer.zero_grad()

            pred = classifier(points)
            # loss = F.mse_loss(pred, target)
            # loss = plane_loss(points, pred)
            loss = foot_loss(points, pred)
            train_loss.append(loss.item())

            loss.backward()
            optimizer.step()

        scheduler.step()
        torch.save(classifier.state_dict(), 'model_trained/' + save_str + '.pth')
        train_loss_mean = np.mean(train_loss).item()

        with torch.no_grad():
            test_loss = []
            classifier = classifier.eval()
            for j, data in tqdm(enumerate(test_loader), total=len(test_loader)):
                points = data[0].float().cuda()
                target = data[1].float().cuda()

                pred = classifier(points)
                # loss = F.mse_loss(pred, target)
                # loss = plane_loss(points, pred)
                loss = foot_loss(points, pred)
                test_loss.append(loss.item())

            test_loss_mean = np.mean(test_loss).item()
            print(f'{epoch} / {args.epoch}: train_loss: {train_loss_mean}. test_loss: {test_loss_mean}')
            writer.add_scalar('train_loss', train_loss_mean, epoch)
            writer.add_scalar('train_loss', train_loss_mean, epoch)


if __name__ == '__main__':
    os.makedirs('log', exist_ok=True)
    os.makedirs('model_trained', exist_ok=True)
    main(parse_args())



