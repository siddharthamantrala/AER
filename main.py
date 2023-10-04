# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
import importlib
from datasets import NAMES as DATASET_NAMES
from models import get_all_models
from argparse import ArgumentParser
from utils.args import add_management_args, add_gcil_args, add_bpcn_args
from datasets import ContinualDataset
from datasets import get_dataset
from models import get_model
from utils.training import train
from utils.best_args import best_args
from utils.conf import set_random_seed
from bayes_pcn.util import DotDict, setup
from typing import Any
import wandb


def setup_env(args: Any) -> DotDict:
    """
    Setup wandb environment, create directories, set seeds, etc.
    """
    os.environ["WANDB_MODE"] = args.wandb_mode
    wandb.init(project="bayes_pcn", entity=args.wandb_account, config=args)
    wandb.define_metric("iteration/step")
    wandb.define_metric("iteration/*", step_metric="iteration/step")
    wandb.define_metric("epoch/step")
    wandb.define_metric("epoch/*", step_metric="epoch/step")

    args = DotDict(wandb.config)
    if args.run_name is None:
        args.run_name = f"{args.seed}h{args.h_dim}l{args.n_layers}m{args.n_models}_{wandb.run.id}"
    wandb.run.name = args.run_name
    args.path = f'runs/{args.run_group}/{args.run_name}'
    print(f"Saving models to directory: {args.path}")

    setup(args=args)
    return args


def main():
    parser = ArgumentParser(description='SAMC', allow_abbrev=False)
    parser.add_argument('--model', type=str, required=False, default='er_bpcn',
                        help='Model name.', choices=get_all_models())
    parser.add_argument('--dataset', type=str, required=False,
                        choices=DATASET_NAMES, default='seq-cifar10',
                        help='Which dataset to perform experiments on.')
    parser.add_argument('--load_best_args', action='store_true', default=True,
                        help='Loads the best arguments for each method, '
                             'dataset and memory buffer.')
    parser.add_argument('--theta', type=float, required=False, default=0.8,
                        help='Threshold for GradCAM masking.')
    parser.add_argument('--seed', type=int, default=0,
                        help='The random seed.')
    parser.add_argument('--cuda', type=str, default=True,
                        help='Use GPU or not')
    
    add_management_args(parser)
    # bayes-pcn configs
    add_bpcn_args(parser)
    args = parser.parse_known_args()[0]
    mod = importlib.import_module('models.' + args.model)

    if args.load_best_args:
        if hasattr(mod, 'Buffer'):
            parser.add_argument('--buffer_size', type=int, required=False, default=200,
                                help='The size of the memory buffer.')
        args = parser.parse_args()
        if args.model == 'joint':
            if args.dataset == 'gcil-cifar100':
                best = best_args[args.dataset]['sgd'][args.weight_dist]
            else:
                best = best_args[args.dataset]['sgd']
        else:
            if args.dataset == 'gcil-cifar100':
                best = best_args[args.dataset][args.model][args.weight_dist]
            else:
                best = best_args[args.dataset][args.model]
        if hasattr(args, 'buffer_size'):
            if args.model == 'joint' or args.model == 'sgd':
                best = best[-1]
            else:
                best = best[args.buffer_size]
        else:
            best = best[-1]
        for key, value in best.items():
            setattr(args, key, value)

    else:
        get_parser = getattr(mod, 'get_parser')
        parser = get_parser()
        if args.dataset == 'gcil-cifar100':
            add_gcil_args(parser)
        args = parser.parse_args()

        print(args)

    if args.seed is not None:
        set_random_seed(args.seed)

    if args.model == 'mer':
        setattr(args, 'batch_size', 1)

    # "setup_env" is inherited from bayes-pcn's code
    args = setup_env(args=args)

    print(args)

    dataset = get_dataset(args)
    backbone = dataset.get_backbone()
    loss = dataset.get_loss()
    model = get_model(args, backbone, loss, dataset.get_transform())

    if isinstance(dataset, ContinualDataset):
        train(model, dataset, args)


if __name__ == '__main__':
    main()
