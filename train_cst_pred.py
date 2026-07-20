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
from functional.wandb_utils import (
    initialize_wandb_run,
    read_wandb_run_id_from_checkpoint,
)
from networks.cst_pred_wrapper import CstPredWrapper
from colorama import init, Fore, Back


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', type=int, default=50, help='batch size in training')
    parser.add_argument('--epoch', default=500, type=int, help='number of epoch in training')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate in training')
    parser.add_argument('--n_points', type=int, default=2000, help='Point Number')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--workers', type=int, default=16, help='dataloader workers')
    parser.add_argument('--model', default='attn_3dgcn', choices=['pointnet2', 'pointnet', 'attn_3dgcn'], type=str)
    parser.add_argument('--is_sample', action='store_true', default=False)

    parser.add_argument('--local', action='store_true', default=False)
    parser.add_argument('--root_sever', type=str, default=r'/opt/data/private/data_set/pcd_cstnet2/Param20K_pcd')
    parser.add_argument('--root_local', type=str, default=r'D:\document\DeepLearning\DataSet\pcd_cstnet2\Param20K_pcd')
    parser.add_argument('--wandb_project', type=str, default='cstnet2')
    parser.add_argument('--wandb_entity', type=str, default='')
    parser.add_argument('--wandb_run_name', type=str, default='')
    parser.add_argument('--train_phase', default='semantic', choices=['semantic', 'geometry', 'joint'])
    parser.add_argument('--use_extra_features', action='store_true', default=False)
    parser.add_argument('--normal_source', default='none', choices=['gt', 'pca', 'none'], type=str)
    parser.add_argument('--feature_k', default=16, type=int)
    parser.add_argument('--cluster_bandwidth', default=0.35, type=float)
    parser.add_argument('--overfit_one_batch', action='store_true', default=False)
    parser.add_argument('--resume_checkpoint', default='', type=str,
                        help='resume model, optimizer, scheduler, epoch and all training state')
    parser.add_argument('--init_from_checkpoint', default='', type=str,
                        help='load model weights only and create a new optimizer')
    parser.add_argument('--w_pmt', default=1.0, type=float)
    parser.add_argument('--w_cluster', default=0.5, type=float)
    parser.add_argument('--w_mad', default=0.02, type=float)
    parser.add_argument('--w_dim', default=0.05, type=float)
    parser.add_argument('--w_nor', default=0.1, type=float)
    parser.add_argument('--w_loc', default=0.02, type=float)
    parser.add_argument('--w_geom', default=0.02, type=float)
    parser.add_argument('--w_inst', default=0.005, type=float)
    parser.add_argument('--geom_start_epoch', default=20, type=int)
    parser.add_argument('--geom_ramp_epochs', default=20, type=int)
    parser.add_argument('--disable_mad_loss', dest='enable_mad_loss', action='store_false', default=True)
    parser.add_argument('--disable_dim_loss', dest='enable_dim_loss', action='store_false', default=True)
    parser.add_argument('--disable_nor_loss', dest='enable_nor_loss', action='store_false', default=True)
    parser.add_argument('--disable_loc_loss', dest='enable_loc_loss', action='store_false', default=True)
    parser.add_argument('--disable_geom_loss', dest='enable_geom_loss', action='store_false', default=True)
    parser.add_argument('--disable_inst_loss', dest='enable_inst_loss', action='store_false', default=True)
    parser.add_argument('--joint_backbone_lr_scale', default=0.1, type=float)
    parser.add_argument('--use_amp', action='store_true', default=False)
    parser.add_argument(
        '--disable_grad_diagnostics',
        dest='enable_grad_diagnostics',
        action='store_false',
        default=True,
    )

    args = parser.parse_args(argv)
    return args


def main(args):
    if args.resume_checkpoint and args.init_from_checkpoint:
        raise ValueError('--resume_checkpoint and --init_from_checkpoint are mutually exclusive')
    save_str = f'{args.model}_multitask_{args.train_phase}_pmt_prim_cluster'
    print(Fore.BLUE + Back.CYAN + f'-> save str: {save_str} <-')

    os.makedirs('log', exist_ok=True)
    os.makedirs('model_trained', exist_ok=True)

    # data
    data_root = args.root_local if args.local else args.root_sever
    train_loader, test_loader = CstNet2Dataset.create_dataloader(
        root=data_root,
        bs=args.bs,
        n_points=args.n_points,
        num_workers=args.workers,
        is_sample=args.is_sample
    )

    # trainer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_extra_features = args.use_extra_features
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
    enabled_losses = {
        'mad': args.enable_mad_loss,
        'dim': args.enable_dim_loss,
        'nor': args.enable_nor_loss,
        'loc': args.enable_loc_loss,
        'geom': args.enable_geom_loss,
        'inst': args.enable_inst_loss,
    }
    stage1_model = CstPredWrapper(
        args.model,
        channel_fea=channel_fea,
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in stage1_model.parameters())
    wandb_resume_id = read_wandb_run_id_from_checkpoint(args.resume_checkpoint)
    if args.resume_checkpoint and not wandb_resume_id:
        print(
            Fore.YELLOW
            + 'WARNING: resume checkpoint has no wandb_run_id; '
            'a new WandB Run will be created for this legacy checkpoint'
        )
    run = initialize_wandb_run(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name if args.wandb_run_name else save_str,
        run_id=wandb_resume_id,
        config={
            **vars(args),
            'parameter_count': parameter_count,
            'input_feature_dim': channel_fea,
            'device': str(device),
        },
    )
    trainer = CstPredTrainer(
        model=stage1_model,
        train_loader = train_loader,
        test_loader = test_loader,
        checkpoint_dir=os.path.join('model_trained', save_str),
        log_savepth = os.path.join('log', save_str + f'_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.json'),
        max_epoch = args.epoch,
        lr = args.lr,
        decay_rate=args.decay_rate,
        save_str=save_str,
        wandb_run=run,
        loss_weights=loss_weights,
        geom_start_epoch=args.geom_start_epoch,
        geom_ramp_epochs=args.geom_ramp_epochs,
        use_extra_features=use_extra_features,
        normal_source=args.normal_source,
        feature_k=args.feature_k,
        cluster_bandwidth=args.cluster_bandwidth,
        overfit_one_batch=args.overfit_one_batch,
        train_phase=args.train_phase,
        enabled_losses=enabled_losses,
        resume_checkpoint=args.resume_checkpoint,
        init_from_checkpoint=args.init_from_checkpoint,
        checkpoint_args=vars(args),
        joint_backbone_lr_scale=args.joint_backbone_lr_scale,
        use_amp=args.use_amp,
        enable_grad_diagnostics=args.enable_grad_diagnostics,
    )
    try:
        trainer.start()
    finally:
        run.finish()


if __name__ == '__main__':
    init(autoreset=True)
    main(parse_args())

