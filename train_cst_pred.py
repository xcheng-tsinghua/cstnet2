"""
训练约束预测模块
"""
import os
import math
import argparse
from datetime import datetime
import torch

from data_utils.stage1_dataset import Stage1ConstraintDataset
from data_utils.huggingface_dataset import resolve_stage1_data_root
from functional.stage1_phase_loss import TRAINING_RECIPE
from functional.console_io import resilient_console, safe_print
from functional.direct_constraints import CONSTRAINT_ROUTE
from functional.cst_pred_trainer import CstPredTrainer
from functional.point_features import stage1_feature_dim
from functional.stage1_checkpoint_policy import (
    CHECKPOINT_POLICIES,
    resolve_stage1_checkpoint,
)
from functional.wandb_utils import (
    initialize_wandb_run,
    read_wandb_run_id_from_checkpoint,
)
from networks.cst_pred_wrapper import CstPredWrapper
from colorama import init, Fore, Back


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', type=int, default=30, help='batch size in training')
    parser.add_argument('--epoch', default=100, type=int, help='number of epoch in training')
    parser.add_argument('--lr', default=None, type=float, help='default: 1e-4 for semantic/geometry, 1e-5 for joint')
    parser.add_argument('--n_points', type=int, default=2048, help='Point Number')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--workers', type=int, default=16, help='dataloader workers')
    parser.add_argument('--model', default='attn_3dgcn', choices=['pointnet2', 'pointnet', 'attn_3dgcn'], type=str)
    parser.add_argument('--is_sample', action='store_true', default=False)

    parser.add_argument(
        '--data_root',
        type=str,
        default='/opt/data/private/data_set/pcd_cstnet2/stage1_small2_h5',
        help='local Stage 1 directory/HDF5 file or Hugging Face dataset repository/tree URL',
    )
    parser.add_argument(
        '--hf_cache_dir', default=None, type=str,
        help='optional Hugging Face cache directory; defaults to the standard HF cache',
    )
    parser.add_argument(
        '--data_format',
        default='auto',
        choices=['auto', 'txt', 'h5'],
        help='Stage 1 storage format; auto prefers HDF5 when shards are present',
    )
    parser.add_argument('--wandb_project', type=str, default='cstnet2-s1')
    parser.add_argument('--wandb_entity', type=str, default='')
    parser.add_argument('--wandb_run_name', type=str, default='')
    parser.add_argument('--train_phase', default='semantic', choices=['semantic', 'geometry', 'joint'])
    parser.add_argument('--disable_extra_features', action='store_true', default=False)
    parser.add_argument('--feature_k', default=16, type=int)
    parser.add_argument('--overfit_one_batch', action='store_true', default=False)
    parser.add_argument(
        '--checkpoint_root',
        default=os.path.join('model_trained', 'stage1_direct'),
        type=str,
    )
    parser.add_argument(
        '--checkpoint_policy',
        default='auto',
        choices=CHECKPOINT_POLICIES,
        help='auto-resume, restart the selected phase, or require a resume checkpoint',
    )
    for name in ("pmt", "cluster", "mad", "dim", "loc"):
        parser.add_argument(f"--w_{name}", default=1.0, type=float)
    parser.add_argument(
        '--grad_clip',
        default=1.0,
        type=float,
        help='global gradient norm limit; set to 0 to disable clipping',
    )

    args = parser.parse_args(argv)
    if args.lr is None:
        args.lr = 1e-4 if args.train_phase == "joint" else 1e-4
    for name in ("pmt", "cluster", "mad", "dim", "loc"):
        weight = getattr(args, f"w_{name}")
        if not math.isfinite(weight) or weight <= 0:
            parser.error(f"--w_{name} must be finite and positive")
    return args


@resilient_console()
def main(args):
    if not args.data_root:
        raise ValueError('--data_root must be a local dataset path or Hugging Face dataset URL')
    save_str = f'{args.model}_direct_{args.train_phase}'
    print(Fore.BLUE + Back.CYAN + f'-> save str: {save_str} <-')

    checkpoint_resolution = resolve_stage1_checkpoint(
        checkpoint_root=args.checkpoint_root,
        model=args.model,
        phase=args.train_phase,
        policy=args.checkpoint_policy,
    )
    checkpoint_resolution.print_summary(
        model=args.model,
        phase=args.train_phase,
        policy=args.checkpoint_policy,
    )

    os.makedirs('log', exist_ok=True)
    os.makedirs('model_trained', exist_ok=True)

    # data
    resolved_data_root = resolve_stage1_data_root(
        args.data_root, cache_dir=args.hf_cache_dir, storage_format=args.data_format,
    )
    train_loader = Stage1ConstraintDataset.create_dataloader(
        root=resolved_data_root,
        bs=args.bs,
        n_points=args.n_points,
        num_workers=args.workers,
        shuffle=True,
        is_sample=args.is_sample,
        storage_format=args.data_format,
    )

    # trainer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_extra_features = not args.disable_extra_features
    channel_fea = stage1_feature_dim(use_extra_features)
    loss_weights = {
        'w_pmt': args.w_pmt,
        'w_cluster': args.w_cluster,
        'w_mad': args.w_mad,
        'w_dim': args.w_dim,
        'w_loc': args.w_loc,
    }
    stage1_model = CstPredWrapper(
        args.model,
        channel_fea=channel_fea,
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in stage1_model.parameters())
    resume_source = (
        checkpoint_resolution.source
        if checkpoint_resolution.action == 'resume'
        else None
    )
    wandb_resume_id = read_wandb_run_id_from_checkpoint(resume_source)
    if resume_source and not wandb_resume_id:
        print(
            Fore.YELLOW
            + 'WARNING: automatically selected resume checkpoint has no wandb_run_id; '
            'a new WandB Run will be created'
        )
    checkpoint_args = {
        **vars(args),
        'use_extra_features': use_extra_features,
        'constraint_route': CONSTRAINT_ROUTE,
        'training_recipe': TRAINING_RECIPE,
        'resolved_data_root': resolved_data_root,
    }
    run = initialize_wandb_run(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name if args.wandb_run_name else save_str,
        run_id=wandb_resume_id,
        config={
            **checkpoint_args,
            'parameter_count': parameter_count,
            'input_feature_dim': channel_fea,
            'device': str(device),
            'dataset_file_count': len(train_loader.dataset),
            'checkpoint_action': checkpoint_resolution.action,
            'checkpoint_source': (
                str(checkpoint_resolution.source)
                if checkpoint_resolution.source is not None
                else ''
            ),
            'checkpoint_dir': str(checkpoint_resolution.checkpoint_dir),
        },
    )
    trainer = CstPredTrainer(
        model=stage1_model,
        train_loader = train_loader,
        checkpoint_dir=str(checkpoint_resolution.checkpoint_dir),
        log_savepth = os.path.join('log', save_str + f'_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.json'),
        max_epoch = args.epoch,
        lr = args.lr,
        decay_rate=args.decay_rate,
        save_str=save_str,
        wandb_run=run,
        loss_weights=loss_weights,
        use_extra_features=use_extra_features,
        feature_k=args.feature_k,
        overfit_one_batch=args.overfit_one_batch,
        train_phase=args.train_phase,
        checkpoint_action=checkpoint_resolution.action,
        checkpoint_source=(
            str(checkpoint_resolution.source)
            if checkpoint_resolution.source is not None
            else ''
        ),
        checkpoint_args=checkpoint_args,
        grad_clip=args.grad_clip,
    )
    try:
        trainer.start()
    finally:
        try:
            run.finish()
        except OSError as error:
            safe_print(f"WARNING: WandB finish I/O failed: {error}")


if __name__ == '__main__':
    init(autoreset=True)
    main(parse_args())

