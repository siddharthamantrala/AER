# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from torch.optim import SGD
from torchvision.models import resnet18
import torch.nn as nn
from utils.buffer import Buffer
from utils.training import batch_resize, batch_visualization

from utils.args import *
from models.utils.continual_model import ContinualModel
from datasets.utils.validation import ValidationDataset
from utils.status import progress_bar
import torch
import numpy as np
import math


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual Learning via'
                                        ' Progressive Neural Networks.')
    add_management_args(parser)
    add_experiment_args(parser)
    return parser


class Sgd(ContinualModel):
    NAME = 'sgd'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(Sgd, self).__init__(backbone, loss, args, transform)
        self.current_task = 0
        self.n_classes = backbone.num_classes
        self.net = resnet18(weights='IMAGENET1K_V1')
        # modify classifier
        self.net.fc = nn.Linear(512, self.n_classes)
        # check model architecture
        print(self.net)
        # only optimize the last FC layer
        self.opt = SGD([{'params': self.net.fc.parameters()}], lr=self.args.lr)
        self.net.to(self.device)

    def observe(self, inputs, labels, not_aug_inputs):
        self.opt.zero_grad()
        inputs = batch_resize(inputs)
        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)
        loss.backward()
        self.opt.step()

        return loss.item()
