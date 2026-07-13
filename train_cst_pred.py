"""
训练约束预测模块
"""
import os
import argparse
from datetime import datetime
import torch

from data_utils.datasets import CstNet2Dataset
from functional.cst_pred_trainer import CstPredTrainer
from functional.point_features import stage1_feature_dim
from networks.cst_pred_wrapper import CstPredWrapper
from colorama import init, Fore, Back

try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency
    wandb = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', type=int, default=180, help='batch size in training')
    parser.add_argument('--epoch', default=500, type=int, help='number of epoch in training')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate in training')
    parser.add_argument('--n_points', type=int, default=2000, help='Point Number')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--workers', type=int, default=16, help='dataloader workers')
    parser.add_argument('--is_load_weight', default='True', choices=['True', 'False'], type=str)
    parser.add_argument('--model', default='attn_3dgcn', choices=['pointnet2', 'pointnet', 'attn_3dgcn'], type=str)
    parser.add_argument('--is_sample', default='False', choices=['True', 'False'], type=str)

    parser.add_argument('--local', default='False', choices=['True', 'False'], type=str)
    parser.add_argument('--root_sever', type=str, default=r'/opt/data/private/data_set/pcd_cstnet2/Param20K_pcd')
    parser.add_argument('--root_local', type=str, default=r'D:\document\DeepLearning\DataSet\pcd_cstnet2\Param20K_Extend')
    parser.add_argument('--use_wandb', default='False', choices=['True', 'False'], type=str)
    parser.add_argument('--wandb_project', type=str, default='cstnet2')
    parser.add_argument('--wandb_entity', type=str, default='')
    parser.add_argument('--wandb_run_name', type=str, default='')
    parser.add_argument('--stage1_mode', default='baseline', choices=['baseline', 'multitask'], type=str)
    parser.add_argument('--use_extra_features', default='False', choices=['True', 'False'], type=str)
    parser.add_argument('--normal_source', default='none', choices=['gt', 'pca', 'none'], type=str)
    parser.add_argument('--feature_k', default=16, type=int)
    parser.add_argument('--cluster_bandwidth', default=0.35, type=float)
    parser.add_argument('--overfit_one_batch', default='False', choices=['True', 'False'], type=str)
    parser.add_argument('--w_pmt', default=1.0, type=float)
    parser.add_argument('--w_cluster', default=1.0, type=float)
    parser.add_argument('--w_mad', default=0.2, type=float)
    parser.add_argument('--w_dim', default=0.2, type=float)
    parser.add_argument('--w_nor', default=0.2, type=float)
    parser.add_argument('--w_loc', default=0.1, type=float)
    parser.add_argument('--w_geom', default=0.2, type=float)
    parser.add_argument('--w_inst', default=0.05, type=float)
    parser.add_argument('--geom_warmup_epoch', default=20, type=int)

    args = parser.parse_args()
    return args


def main(args):
    save_str = f'{args.model}_pmt_prim_cluster' if args.stage1_mode == 'baseline' else f'{args.model}_multitask_pmt_prim_cluster'
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
    use_extra_features = eval(args.use_extra_features)
    channel_fea = stage1_feature_dim(use_extra_features, args.normal_source)
    loss_weights = {
        'w_pmt': args.w_pmt,
        'w_cluster': args.w_cluster,
        'w_mad': args.w_mad,
        'w_dim': args.w_dim,
        'w_nor': args.w_nor,
        'w_loc': args.w_loc,
        'w_geom': args.w_geom,
        'w_inst': args.w_inst,
    }
    trainer = CstPredTrainer(
        model = CstPredWrapper(
            args.model,
            channel_fea=channel_fea,
            stage1_mode=args.stage1_mode,
        ).to(device),
        train_loader = train_loader,
        test_loader = test_loader,
        model_savepth = 'model_trained/' + save_str + '.pth',
        log_savepth = os.path.join('log', save_str + f'_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.json'),
        max_epoch = args.epoch,
        lr = args.lr,
        decay_rate=args.decay_rate,
        is_load_weight = eval(args.is_load_weight),
        save_str=save_str,
        wandb_run=run,
        stage1_mode=args.stage1_mode,
        loss_weights=loss_weights,
        geom_warmup_epoch=args.geom_warmup_epoch,
        use_extra_features=use_extra_features,
        normal_source=args.normal_source,
        feature_k=args.feature_k,
        cluster_bandwidth=args.cluster_bandwidth,
        overfit_one_batch=eval(args.overfit_one_batch),
    )
    try:
        trainer.start()
    finally:
        if run is not None:
            run.finish()


if __name__ == '__main__':
    init(autoreset=True)
    main(parse_args())

