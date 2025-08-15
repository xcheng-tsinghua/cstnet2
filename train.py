
"""
train classification
"""

import torch
import torch.nn.functional as F
from datetime import datetime
import logging
import argparse
from tqdm import tqdm
import numpy as np
import os

from data_utils.Param20KDataset import RegressionDataset
from models.pointnet import PointNet
from models.pointnet2 import PointNet2Regression


def parse_args():

    parser = argparse.ArgumentParser('training')
    parser.add_argument('--bs', type=int, default=200, help='batch size in training')
    parser.add_argument('--epoch', default=200, type=int, help='number of epoch in training')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')

    parser.add_argument('--n_point', type=int, default=2000, help='Point Number')
    parser.add_argument('--is_load_weight', default='True', choices=['True', 'False'], type=str)

    parser.add_argument('--local', default='False', choices=['True', 'False'], type=str)
    parser.add_argument('--root_sever', type=str, default=rf'/opt/data/private/data_set/pcd_cstnet2')
    parser.add_argument('--root_local', type=str, default=rf'D:\document\DeepLearning\DataSet\pcd_cstnet2')

    return parser.parse_args()


def main(args):
    # parameters
    save_str = 'pointnet'

    # logger
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler('log/' + save_str + f'-{datetime.now().strftime("%Y-%m-%d %H-%M-%S")}.txt')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

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
    classifier = PointNet2Regression()

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

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

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
            loss = F.mse_loss(pred, target)
            train_loss.append(loss.item())

            loss.backward()
            optimizer.step()

        scheduler.step()
        torch.save(classifier.state_dict(), 'model_trained/' + save_str + '.pth')
        train_accstr = f'train_loss:\t{np.mean(train_loss)}'

        with torch.no_grad():
            test_loss = []
            classifier = classifier.eval()
            for j, data in tqdm(enumerate(test_loader), total=len(test_loader)):
                points = data[0].float().cuda()
                target = data[1].float().cuda()

                pred = classifier(points)
                loss = F.mse_loss(pred, target)
                test_loss.append(loss.item())

            test_accstr = f'test_loss:\t{np.mean(test_loss)}'
            log_str = logstr_epoch + '\t' + train_accstr + '\t' + test_accstr

            print(log_str.replace('\t', ' ') + datetime.now().strftime("%Y-%m-%d %H-%M-%S"))
            logger.info(log_str)


if __name__ == '__main__':
    os.makedirs('log', exist_ok=True)
    os.makedirs('model_trained', exist_ok=True)
    main(parse_args())



