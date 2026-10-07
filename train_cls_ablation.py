"""Single CLI for CSTNet2 Stage 2 classification ablations.

Examples:
  python train_cls_ablation.py --list
  python train_cls_ablation.py --data_root PATH --experiments all --seed 42
  python train_cls_ablation.py --data_root PATH --experiments no_location --seed 42
"""
from __future__ import annotations

import argparse
import json
from colorama import init

from functional.stage2_ablation_config import EXPERIMENTS, SUITES, expand_experiments


def parse_args(argv=None):
    parser = argparse.ArgumentParser(__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--mode", choices=("train", "evaluate", "summarize"), default="train")
    parser.add_argument(
        "--experiments", nargs="+", default=["all"], choices=(*EXPERIMENTS, *SUITES),
        help=("Select one or more experiments; XYZ is always retained. "
              "all: run all nine experiments; "
              "xyz_only: use XYZ without constraints; "
              "no_primitive_type: remove primitive type; "
              "no_direction: remove direction; "
              "no_dimension: remove dimension; "
              "no_location: remove location; "
              "only_primitive_type: keep only primitive type constraints; "
              "only_direction: keep only direction constraints; "
              "only_dimension: keep only dimension constraints; "
              "only_location: keep only location constraints."))
    parser.add_argument("--seed", type=int, default=42, help="Single training seed shared by all selected experiments")
    parser.set_defaults(model="constraint_aware", task="cls")
    parser.add_argument("--data_root", default="/opt/data/private/data_set/pcd_cstnet2/tmcad_pcd", help="Existing frozen Stage 1 prediction cache")
    parser.add_argument("--is_sample", action="store_true")
    parser.add_argument("--save_name", default="stage2_cls_ablation")
    parser.add_argument("--wandb_project", default="cstnet2-s2-ablation")
    parser.add_argument("--wandb_entity", default="")
    parser.add_argument("--wandb_run_name", default="")
    parser.add_argument("--stage2_norm", choices=("ln", "bn"), default="ln")
    parser.add_argument("--token_dim", type=int, default=256)
    parser.add_argument("--transformer_layers", type=int, default=3)
    parser.add_argument("--transformer_heads", type=int, default=8)
    parser.add_argument("--token_dropout", type=float, default=0.1)
    parser.add_argument("--stream_dropout", type=float, default=0.1)
    parser.add_argument("--use_stats_token", action="store_true")
    parser.add_argument("--data_format", choices=("auto", "txt", "h5"), default="auto")
    parser.add_argument("--output_dir", default="model_trained/stage2_ablation")
    parser.add_argument("--bs", "--batch_size", dest="batch_size", type=int, default=20)
    parser.add_argument("--n_points", "--n_point", dest="n_points", type=int, default=2048)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--epoch", "--epochs", dest="epochs", type=int, default=200)
    parser.add_argument("--lr", "--learning_rate", dest="learning_rate", type=float, default=1e-4)
    parser.add_argument("--decay_rate", "--weight_decay", dest="weight_decay", type=float, default=1e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--aux_loss_weight", type=float, default=0.1)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda or cuda:0")
    parser.add_argument("--resume", default="auto", metavar="MODE_OR_PATH",
                        help="Empty string: train from scratch; auto: try this experiment's last.pth "
                             "and restart if restoration fails; checkpoint path: restore strictly "
                             "and raise on failure. Successful restoration continues the original W&B Run.")
    parser.add_argument("--list", action="store_true", help="List experiments without loading PyTorch")
    parser.add_argument("--dry_run", action="store_true", help="Print resolved experiment settings only")
    args = parser.parse_args(argv)
    if args.learning_rate is None:
        args.learning_rate = 1e-4
    args.experiments = expand_experiments(args.experiments)
    if not 0 <= args.seed < 2**32:
        parser.error("seed must lie in [0, 2**32)")
    if args.list or args.mode == "summarize":
        return args
    if args.batch_size < 2 or args.n_points < 2 or args.epochs < 1 or args.workers < 0:
        parser.error("batch_size/n_points must be >=2, epochs >=1, workers >=0")
    if args.learning_rate <= 0 or args.weight_decay < 0 or args.aux_loss_weight < 0:
        parser.error("learning_rate must be positive; weight_decay/aux_loss_weight nonnegative")
    if not 0 <= args.label_smoothing < 1:
        parser.error("label_smoothing must be in [0,1)")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.list:
        for name, spec in EXPERIMENTS.items():
            print(f"{name:24s} {spec.description}")
        for name, entries in SUITES.items():
            print(f"suite {name}: {', '.join(entries)}")
        return
    if args.dry_run:
        print(json.dumps({"args": vars(args), "experiments": {
            e: EXPERIMENTS[e].to_dict() for e in args.experiments}}, indent=2))
        return
    from functional.stage2_ablation_runner import run_experiment, summarize
    if args.mode != "summarize":
        for experiment in args.experiments:
            run_experiment(args, experiment, args.seed)
    summarize(args.output_dir, args.seed)


if __name__ == "__main__":
    init(autoreset=True)
    main()
