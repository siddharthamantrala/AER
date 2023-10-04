# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from argparse import ArgumentParser
from datasets import NAMES as DATASET_NAMES
from models import get_all_models
from bayes_pcn.const import *


def add_gcil_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments required for GCIL-CIFAR100 Dataset.
    :param parser: the parser instance
    """
    # arguments for GCIL-CIFAR100
    parser.add_argument('--gil_seed', type=int, default=1993, help='Seed value for GIL-CIFAR task sampling')
    parser.add_argument('--pretrain', action='store_true', default=False, help='whether to use pretrain')
    parser.add_argument('--phase_class_upper', default=50, type=int, help='the maximum number of classes')
    parser.add_argument('--epoch_size', default=1000, type=int, help='Number of samples in one epoch')
    parser.add_argument('--pretrain_class_nb', default=0, type=int, help='the number of classes in first group')
    parser.add_argument('--weight_dist', default='longtail', type=str, help='what type of weight distribution assigned to classes to sample (unif or longtail)')


def add_experiment_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the models.
    :param parser: the parser instance
    """
    parser.add_argument('--dataset', type=str, required=True,
                        choices=DATASET_NAMES,
                        help='Which dataset to perform experiments on.')
    parser.add_argument('--tiny_imagenet_path', type=str, default='data')
    parser.add_argument('--model', type=str, required=True,
                        help='Model name.', choices=get_all_models())

    parser.add_argument('--lr', type=float, required=True,
                        help='Learning rate.')
    parser.add_argument('--batch_size', type=int, required=True,
                        help='Batch size.')
    parser.add_argument('--n_epochs', type=int, required=True,
                        help='The number of epochs for each task.')
    parser.add_argument('--save_interim', action='store_true', default=False,
                        help='The number of epochs for each task.')


def add_management_args(parser: ArgumentParser) -> None:
    parser.add_argument('--experiment_id', type=str, default='cl')
    parser.add_argument('--notes', type=str, default=None,
                        help='Notes for this run.')

    parser.add_argument('--csv_log', action='store_true', default=True, 
                        help='Enable csv logging')
    parser.add_argument('--tensorboard', action='store_true',
                        help='Enable tensorboard logging')
    parser.add_argument('--validation', action='store_true',
                        help='Test on the validation set')
    parser.add_argument('--output_dir', type=str, default='experiments')


def add_bpcn_args(parser: ArgumentParser) -> None:
    # general configs
    parser.add_argument('--run-name', type=str, default=None,
                        help='Name of this run. If not specified set to wandb run ID.')
    parser.add_argument('--run-group', type=str, default='default', help='Group this run is in.')
    parser.add_argument('--load-path', type=str, default=None)
    parser.add_argument('--wandb-account', type=str, default="wandb",
                        help='Weights and Biases account ID.')
    parser.add_argument('--wandb-mode', type=str, choices=['online', 'offline'], default='offline')
    parser.add_argument('--dtype', type=str, choices=['float32', 'float64'], default='float32',
                        help='Use float64 if deletion is needed for numerical stability.')

    # model configs
    # parser.add_argument('--n_memories', type=int, default=40,
    #                     help='number of memories per task')
    parser.add_argument('--memory_strength', default=1, type=float,
                        help='ajusting the trade off between past task loss and current task loss. (1 means equal)')
    parser.add_argument('--n-models', type=int, default=1)
    parser.add_argument('--n-layers', type=int, default=2)
    parser.add_argument('--h-dim', type=int, default=256)
    parser.add_argument('--sigma-prior', type=float, default=1.)
    parser.add_argument('--sigma-obs', type=float, default=0.01)
    parser.add_argument('--sigma-data', type=float, default=None)
    parser.add_argument('--beta-forget', type=float, default=0.1, help='between 0-1. 0 = no forget.')
    parser.add_argument('--beta-noise', type=float, default=0.1,
                        help='diffusion rate when using --layer-update-strat=noising.')
    parser.add_argument('--scale-layer', action='store_true', help='normalize layer activations.')
    parser.add_argument('--bias', action='store_true', help='Use bias alongside linear transform.')
    parser.add_argument('--n-elbo-particles', type=int, default=1,
                        help='# particles for ELBO estimation when --ensemble-proposal-strat=diag.')
    parser.add_argument('--act-fn', type=ActFn,
                        default=ActFn.RELU, choices=list(ActFn))
    parser.add_argument('--activation-init-strat', type=ActInitStrat,
                        default=ActInitStrat.RANDN, choices=list(ActInitStrat))
    parser.add_argument('--weight-init-strat', type=WeightInitStrat,
                        default=WeightInitStrat.RANDN, choices=list(WeightInitStrat))
    parser.add_argument('--layer-log-prob-strat', type=LayerLogProbStrat,
                        default=LayerLogProbStrat.P_PRED, choices=list(LayerLogProbStrat))
    parser.add_argument('--layer-sample-strat', type=LayerSampleStrat,
                        default=LayerSampleStrat.MAP, choices=list(LayerSampleStrat))
    parser.add_argument('--layer-update-strat', type=LayerUpdateStrat,
                        default=LayerUpdateStrat.BAYES, choices=list(LayerUpdateStrat))
    parser.add_argument('--ensemble-log-joint-strat', type=EnsembleLogJointStrat,
                        default=EnsembleLogJointStrat.SHARED,
                        choices=list(EnsembleLogJointStrat))
    parser.add_argument('--ensemble-proposal-strat', type=EnsembleProposalStrat,
                        default=EnsembleProposalStrat.MODE,
                        choices=list(EnsembleProposalStrat))
    parser.add_argument('--mhn-metric', type=MHNMetric,
                        default=MHNMetric.DOT, choices=list(MHNMetric))
    parser.add_argument('--kernel-type', type=Kernel, default=Kernel.RBF, choices=list(Kernel))

    # training configs
    parser.add_argument('--weight-lr', type=float, default=0.0001)
    parser.add_argument('--activation-lr', type=float, default=0.01)
    parser.add_argument('--activation-optim', choices=['adam', 'sgd'], default='adam')
    parser.add_argument('--T-infer', type=int, default=500)
    parser.add_argument('--n-proposal-samples', type=int, default=1)
    parser.add_argument('--n-repeat', type=int, default=1,
                        help='how much ICM iteration to perform at inference.')
    parser.add_argument('--resample', action='store_true', help='resample if using n-models > 1.')
    parser.add_argument('--forget-every', type=int, default=None,
                        help="Apply forget with strength --beta-forget every this # of iterations.")

def add_rehearsal_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the rehearsal-based methods
    :param parser: the parser instance
    """
    parser.add_argument('--buffer_size', type=int, required=True,
                        help='The size of the memory buffer.')
    parser.add_argument('--minibatch_size', type=int, required=True,
                        help='The batch size of the memory buffer.')
