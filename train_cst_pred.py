"""
训练约束预测模块
"""
import os
import argparse
from datetime import datetime

from data_utils.datasets import CstNet2Dataset
from functional.cst_pred_trainer import CstPredTrainer
from networks.cst_pred_wrapper import CstPredWrapper
from colorama import init, Fore, Back


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', type=int, default=32, help='batch size in training')
    parser.add_argument('--epoch', default=2000, type=int, help='number of epoch in training')
    parser.add_argument('--lr', default=0.0001, type=float, help='learning rate in training')
    parser.add_argument('--n_points', type=int, default=2000, help='Point Number')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--workers', type=int, default=4, help='dataloader workers')
    parser.add_argument('--is_load_weight', default='True', choices=['True', 'False'], type=str)
    parser.add_argument('--model', default='pointnet2', choices=['pointnet2', 'pointnet', 'attn_3dgcn'], type=str)
    parser.add_argument('--is_sample', default='False', choices=['True', 'False'], type=str)

    parser.add_argument('--local', default='False', choices=['True', 'False'], type=str)
    parser.add_argument('--root_sever', type=str, default=rf'/opt/data/private/data_set/pcd_cstnet2/Param20K_Extend')
    parser.add_argument('--root_local', type=str, default=rf'D:\document\DeepLearning\DataSet\pcd_cstnet2\Param20K_Extend')

    args = parser.parse_args()
    return args


def main(args):
    save_str = f'{args.model}_pmt_prim_cluster'
    print(Fore.BLUE + Back.CYAN + f'-> save str: {save_str} <-')

    os.makedirs('log', exist_ok=True)
    os.makedirs('model_trained', exist_ok=True)

    # data
    data_root = args.root_local if eval(args.local) else args.root_sever
    train_loader, test_loader = CstNet2Dataset.create_dataloader(
        root=data_root,
        bs=args.bs,
        n_points=args.n_points,
        num_workers=args.workers,
        is_sample=eval(args.is_sample)
    )

    # trainer
    trainer = CstPredTrainer(
        model = CstPredWrapper(args.model).cuda(),
        train_loader = train_loader,
        test_loader = test_loader,
        model_savepth = 'model_trained/' + save_str + '.pth',
        log_savepth = os.path.join('log', save_str + f'_{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}.json'),
        max_epoch = args.epoch,
        lr = args.lr,
        is_load_weight = eval(args.is_load_weight),
        save_str=save_str
    )
    trainer.start()


if __name__ == '__main__':
    init(autoreset=True)
    main(parse_args())

