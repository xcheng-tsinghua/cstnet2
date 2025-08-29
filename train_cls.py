
"""
train classification
"""
import os.path

import torch
import torch.nn.functional as F
from datetime import datetime
import logging
import argparse
from tqdm import tqdm
from tensorboardX import SummaryWriter

from data_utils.datasets import CstNet2Dataset
from models.pointnet2 import PointNet2Cls
from models.utils import all_metric_cls


def parse_args():

    parser = argparse.ArgumentParser('training')
    parser.add_argument('--batch_size', type=int, default=160, help='batch size in training')
    parser.add_argument('--epoch', default=200, type=int, help='number of epoch in training')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')

    parser.add_argument('--n_primitive', type=int, default=4, help='number of considered meta type')
    parser.add_argument('--n_point', type=int, default=2000, help='Point Number')

    parser.add_argument('--local', default='False', choices=['True', 'False'], type=str)
    parser.add_argument('--root_sever', type=str, default=rf'/opt/data/private/data_set/pcd_cstnet2/Param20K_Extend')
    parser.add_argument('--root_local', type=str, default=rf'D:\document\DeepLearning\DataSet\pcd_cstnet2\Param20K_Extend')

    return parser.parse_args()


def main(args):
    # parameters
    save_str = 'pn2_cst_label'

    # logger
    log_dir = os.path.join('log', save_str)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(logdir=log_dir)

    # datasets
    data_root = args.root_local if eval(args.local) else args.root_sever
    train_dataset = CstNet2Dataset(root=data_root, npoints=args.n_point, is_train=True, data_augmentation=False)
    test_dataset = CstNet2Dataset(root=data_root, npoints=args.n_point, is_train=False, data_augmentation=False)
    num_class = len(train_dataset.classes)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # loading model
    classifier = PointNet2Cls(num_class, 5+3+1+3+3)

    model_savepth = 'model_trained/' + save_str + '.pth'
    try:
        classifier.load_state_dict(torch.load(model_savepth))
        print('training from exist model: ' + model_savepth)
    except:
        print('no existing model, training from scratch')

    classifier = classifier.cuda()

    # optimizer
    optimizer = torch.optim.Adam(
        classifier.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=args.decay_rate
    )

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

    # training
    for epoch in range(args.epoch):
        logstr_epoch = 'Epoch %d/%d:' % (epoch + 1, args.epoch)
        all_preds = []
        all_labels = []

        classifier = classifier.train()
        for batch_id, data in tqdm(enumerate(train_loader), total=len(train_loader)):
            points = data[0].cuda()
            target = data[1].long().cuda()

            pmt = data[2].long().cuda()  # 基元类型
            pmt = F.one_hot(pmt, 5)

            main_dir = data[3].cuda()  # 主方向
            main_dim = data[4].cuda()  # 主尺寸
            normal = data[5].cuda()  # 法线
            main_loc = data[6].cuda()  # 主位置
            affil_idx = data[7].long().cuda()  # 从属索引

            cst = torch.cat([pmt, main_dir, main_dim.unsqueeze(2), normal, main_loc], dim=2)

            optimizer.zero_grad()

            pred = classifier(points, cst)
            loss = F.nll_loss(pred, target)

            loss.backward()
            optimizer.step()

            all_preds.append(pred.detach().cpu().numpy())
            all_labels.append(target.detach().cpu().numpy())

        all_metric_train = all_metric_cls(all_preds, all_labels)

        writer.add_scalar('train/ins_acc', all_metric_train[0], epoch)
        writer.add_scalar('train/cla_acc', all_metric_train[1], epoch)
        writer.add_scalar('train/f1', all_metric_train[2], epoch)
        writer.add_scalar('train/map', all_metric_train[3], epoch)

        scheduler.step()
        torch.save(classifier.state_dict(), 'model_trained/' + save_str + '.pth')

        with torch.no_grad():
            all_preds = []
            all_labels = []

            classifier = classifier.eval()
            for j, data in tqdm(enumerate(test_loader), total=len(test_loader)):
                points = data[0].cuda()
                target = data[1].long().cuda()

                pmt = data[2].long().cuda()  # 基元类型
                pmt = F.one_hot(pmt, 5)

                main_dir = data[3].cuda()  # 主方向
                main_dim = data[4].cuda()  # 主尺寸
                normal = data[5].cuda()  # 法线
                main_loc = data[6].cuda()  # 主位置
                affil_idx = data[7].long().cuda()  # 从属索引

                cst = torch.cat([pmt, main_dir, main_dim.unsqueeze(2), normal, main_loc], dim=2)

                pred = classifier(points, cst)

                all_preds.append(pred.detach().cpu().numpy())
                all_labels.append(target.detach().cpu().numpy())

            all_metric_test = all_metric_cls(all_preds, all_labels)

            writer.add_scalar('test/ins_acc', all_metric_test[0], epoch)
            writer.add_scalar('test/cla_acc', all_metric_test[1], epoch)
            writer.add_scalar('test/f1', all_metric_test[2], epoch)
            writer.add_scalar('test/map', all_metric_test[3], epoch)

            accustr = f'test_instance_accuracy: {all_metric_test[0]}. test_class_accuracy: {all_metric_test[1]}. test_F1_Score: {all_metric_test[2]}. mAP: {all_metric_test[3]}'
            print(accustr)


if __name__ == '__main__':
    main(parse_args())

