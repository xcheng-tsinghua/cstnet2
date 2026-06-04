"""
训练约束预测模块
"""
import os
import argparse
from datetime import datetime
import torch

from data_utils.datasets import CstNet2Dataset
from functional.cst_pred_trainer import CstPredTrainer
from networks.cst_pred_wrapper import CstPredWrapper
from colorama import init, Fore, Back

try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency
    wandb = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', type=int, default=512, help='batch size in training')
    parser.add_argument('--epoch', default=500, type=int, help='number of epoch in training')
    parser.add_argument('--lr', default=0.00001, type=float, help='learning rate in training')
    parser.add_argument('--n_points', type=int, default=2000, help='Point Number')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--workers', type=int, default=16, help='dataloader workers')
    parser.add_argument('--is_load_weight', default='False', choices=['True', 'False'], type=str)
    parser.add_argument('--model', default='pointnet2', choices=['pointnet2', 'pointnet', 'attn_3dgcn'], type=str)
    parser.add_argument('--is_sample', default='False', choices=['True', 'False'], type=str)

    parser.add_argument('--local', default='False', choices=['True', 'False'], type=str)
    parser.add_argument('--root_sever', type=str, default=r'/opt/data/private/data_set/pcd_cstnet2/Param20K_pcd')
    parser.add_argument('--root_local', type=str, default=r'D:\document\DeepLearning\DataSet\pcd_cstnet2\Param20K_Extend')
    parser.add_argument('--use_wandb', default='False', choices=['True', 'False'], type=str)
    parser.add_argument('--wandb_project', type=str, default='cstnet2')
    parser.add_argument('--wandb_entity', type=str, default='')
    parser.add_argument('--wandb_run_name', type=str, default='')

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

    use_wandb = eval(args.use_wandb)
    if use_wandb and wandb is None:
        print(Fore.YELLOW + 'wandb is not installed; continue without wandb logging')
        use_wandb = False
    run = None
    if use_wandb:
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity if args.wandb_entity else None,
            name=args.wandb_run_name if args.wandb_run_name else save_str,
            config=vars(args)
        )

    # trainer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = CstPredTrainer(
        model = CstPredWrapper(args.model).to(device),
        train_loader = train_loader,
        test_loader = test_loader,
        model_savepth = 'model_trained/' + save_str + '.pth',
        log_savepth = os.path.join('log', save_str + f'_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.json'),
        max_epoch = args.epoch,
        lr = args.lr,
        is_load_weight = eval(args.is_load_weight),
        save_str=save_str,
        wandb_run=run
    )
    try:
        trainer.start()
    finally:
        if run is not None:
            run.finish()


if __name__ == '__main__':
    init(autoreset=True)
    main(parse_args())

