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

from data_utils.datasets import CstNet2Dataset
from models.cst_pcd import CstPcd
from models.loss import constraint_loss


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', type=int, default=32, help='batch size in training')
    parser.add_argument('--epoch', default=2000, type=int, help='number of epoch in training')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--npoints', type=int, default=2000, help='Point Number')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--workers', type=int, default=4, help='dataloader workers')
    parser.add_argument('--is_load_weight', default='True', choices=['True', 'False'], type=str)

    parser.add_argument('--local', default='False', choices=['True', 'False'], type=str)
    parser.add_argument('--root_sever', type=str, default=rf'/opt/data/private/data_set/pcd_cstnet2/Param20K_Extend')
    parser.add_argument('--root_local', type=str, default=rf'D:\document\DeepLearning\DataSet\pcd_cstnet2\Param20K_Extend')

    args = parser.parse_args()
    return args


def write_loss_dict(writer: SummaryWriter, loss_dict: dict, step: int, add_str: str):
    for c_key, c_value in loss_dict.items():
        writer.add_scalar(f'detailed/{add_str}/' + c_key, c_value, step)


def main(args):
    save_str = 'cst_pcd_3_'

    # logger
    log_dir = os.path.join('log', save_str + datetime.now().strftime("%Y-%m-%d %H-%M-%S"))
    # log_dir = os.path.join('log', save_str)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(logdir=log_dir)

    # data
    data_root = args.root_local if eval(args.local) else args.root_sever

    train_set = CstNet2Dataset(root=data_root, is_train=True, npoints=args.npoints)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.bs, shuffle=True, num_workers=args.workers)

    test_set = CstNet2Dataset(root=data_root, is_train=False, npoints=args.npoints)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.bs, shuffle=True, num_workers=args.workers)

    # model
    predictor = CstPcd(args.npoints).cuda()

    model_savepth = 'model_trained/' + save_str + '.pth'

    if eval(args.is_load_weight):
        try:
            predictor.load_state_dict(torch.load(model_savepth))
            print('training from exist model: ' + model_savepth)
        except:
            print('no existing model, training from scratch')
    else:
        print('does not load weight, training from scratch')

    # optimizer
    optimizer = torch.optim.Adam(
        predictor.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=args.decay_rate
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

    # train
    train_batch = 0
    test_batch = 0
    for epoch in range(args.epoch):
        train_loss_all = []
        predictor = predictor.train()

        for batch_id, data in tqdm(enumerate(train_loader), total=len(train_loader)):
            xyz, cls, pmt_gt, mad_gt, dim_gt, nor_gt, loc_gt, affil_idx = data

            xyz = xyz.float().cuda()
            pmt_gt = pmt_gt.long().cuda()
            mad_gt = mad_gt.float().cuda()
            dim_gt = dim_gt.float().cuda()
            nor_gt = nor_gt.float().cuda()
            loc_gt = loc_gt.float().cuda()
            affil_idx = affil_idx.long().cuda()

            optimizer.zero_grad()
            log_pmt_pred, mad_pred, dim_pred, nor_pred, loc_pred, = predictor(xyz)
            loss, loss_dict = constraint_loss(xyz, log_pmt_pred, mad_pred, dim_pred, nor_pred, loc_pred,
                                   pmt_gt, mad_gt, dim_gt, nor_gt, loc_gt, affil_idx)

            loss.backward()
            optimizer.step()

            c_loss = loss.item()
            train_loss_all.append(c_loss)
            writer.add_scalar('train/loss_batch', c_loss, train_batch)
            write_loss_dict(writer, loss_dict, train_batch, 'train')
            train_batch += 1

        train_loss_epoch = np.mean(train_loss_all).item()
        writer.add_scalar('train/loss_epoch', train_loss_epoch, epoch)

        scheduler.step()
        torch.save(predictor.state_dict(), model_savepth)

        # test
        with torch.no_grad():
            test_loss_all = []
            predictor = predictor.eval()

            for batch_id, data in tqdm(enumerate(test_loader), total=len(test_loader)):
                xyz, cls, pmt_gt, mad_gt, dim_gt, nor_gt, loc_gt, affil_idx = data

                xyz = xyz.float().cuda()
                pmt_gt = pmt_gt.long().cuda()
                mad_gt = mad_gt.float().cuda()
                dim_gt = dim_gt.float().cuda()
                nor_gt = nor_gt.float().cuda()
                loc_gt = loc_gt.float().cuda()
                affil_idx = affil_idx.long().cuda()

                log_pmt_pred, mad_pred, dim_pred, nor_pred, loc_pred, = predictor(xyz)
                loss, loss_dict = constraint_loss(xyz, log_pmt_pred, mad_pred, dim_pred, nor_pred, loc_pred,
                                       pmt_gt, mad_gt, dim_gt, nor_gt, loc_gt, affil_idx)

                c_loss = loss.item()
                test_loss_all.append(c_loss)
                writer.add_scalar('test/loss_batch', c_loss, epoch)
                write_loss_dict(writer, loss_dict, test_batch, 'test')
                test_batch += 1

            test_loss_epoch = np.mean(test_loss_all).item()
            writer.add_scalar('test/loss_epoch', test_loss_epoch, test_batch)

            print(f'{epoch} / {args.epoch}: train_loss: {train_loss_epoch}. test_loss: {test_loss_epoch}')

    writer.close()


if __name__ == '__main__':
    main(parse_args())

