# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from torch.optim import SGD
from torchvision.models import resnet18
from pytorch_grad_cam import GradCAM
import torch.nn as nn
from utils.buffer import Buffer
from utils.training import batch_resize, batch_visualization

import torch
from datasets import get_dataset
from utils.args import *
from models.utils.continual_model import ContinualModel


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Learning without Forgetting.')
    add_management_args(parser)
    add_experiment_args(parser)
    parser.add_argument('--alpha', type=float, required=True,
                        help='Penalty weight.')
    parser.add_argument('--softmax_temp', type=float, required=True,
                        help='Temperature of the softmax function.')
    parser.add_argument('--wd_reg', type=float, required=True,
                        help='Coefficient of the weight decay regularizer.')
    return parser


def smooth(logits, temp, dim):
    log = logits ** (1 / temp)
    return log / torch.sum(log, dim).unsqueeze(1)


def modified_kl_div(old, new):
    return -torch.mean(torch.sum(old * torch.log(new), 1))


class Lwf(ContinualModel):
    NAME = 'lwf'
    COMPATIBILITY = ['class-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super(Lwf, self).__init__(backbone, loss, args, transform)
        self.old_net = None
        self.soft = torch.nn.Softmax(dim=1)
        self.logsoft = torch.nn.LogSoftmax(dim=1)
        self.dataset = get_dataset(args)
        self.current_task = 0
        self.cpt = get_dataset(args).N_CLASSES_PER_TASK
        nc = get_dataset(args).N_TASKS * self.cpt
        self.eye = torch.tril(torch.ones((nc, nc))).bool().to(self.device)
        self.n_classes = backbone.num_classes
        # initial network
        self.net = resnet18(weights='IMAGENET1K_V1')
        # modify classifier
        self.net.fc = nn.Linear(512, self.n_classes)
        # check model architecture
        print(self.net)
        # only optimize the last FC layer
        self.opt = SGD([{'params': self.net.fc.parameters()}], lr=self.args.lr)
        self.net.to(self.device)
        # load GradCAM
        self.target_layer = self.net.layer4[-1]
        self.cam = GradCAM(model=self.net, target_layer=self.target_layer, 
                           use_cuda=self.args.cuda)
    
    def features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the non-activated output of the second-last layer. (512 dims)
        :param x: input tensor (batch_size, *input_shape)
        :return: output tensor (??)
        """
        # get feature map of x
        out = self.cam.get_feature_map(x)  # torch.Size([bsz, 3, 224, 224]) -> torch.Size([bsz, 512, 7, 7])
        if self.args.cuda:
            out = out.cuda()
        out = self.net.avgpool(out)
        feat = out.view(out.size(0), -1)
        return feat

    def get_params(self) -> torch.Tensor:
        """
        Returns all the parameters concatenated in a single tensor.
        :return: parameters tensor (??)
        """
        params = []
        for pp in list(self.net.fc.parameters()):
            params.append(pp.view(-1))
        return torch.cat(params)

    def begin_task(self, dataset):
        self.net.eval()
        if self.current_task > 0:
            # warm-up
            opt = SGD(self.net.fc.parameters(), lr=self.args.lr)
            for epoch in range(self.args.n_epochs):
                for i, data in enumerate(dataset.train_loader):
                    inputs, labels, not_aug_inputs = data
                    inputs = batch_resize(inputs)
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    with torch.no_grad():
                        feats = self.features(inputs)
                    mask = self.eye[(self.current_task + 1) * self.cpt - 1] ^ self.eye[self.current_task * self.cpt - 1]
                    opt.zero_grad()
                    outputs = self.net.fc(feats)[:, mask]
                    loss = self.loss(outputs, labels - self.current_task * self.cpt)
                    loss.backward()
                    opt.step()

            logits = []
            with torch.no_grad():
                for i in range(0, dataset.train_loader.dataset.data.shape[0], self.args.batch_size):
                    inputs = torch.stack([dataset.train_loader.dataset.__getitem__(j)[2]
                                          for j in range(i, min(i + self.args.batch_size,
                                                         len(dataset.train_loader.dataset)))])
                    inputs = batch_resize(inputs)
                    log = self.net(inputs.to(self.device)).cpu()
                    logits.append(log)
            setattr(dataset.train_loader.dataset, 'logits', torch.cat(logits))
        self.net.train()

        self.current_task += 1

    def observe(self, inputs, labels, not_aug_inputs, logits=None):
        self.opt.zero_grad()
        inputs = batch_resize(inputs)
        outputs = self.net(inputs)

        mask = self.eye[self.current_task * self.cpt - 1]
        loss = self.loss(outputs[:, mask], labels)
        if logits is not None:
            mask = self.eye[(self.current_task - 1) * self.cpt - 1]
            loss += self.args.alpha * modified_kl_div(smooth(self.soft(logits[:, mask]).to(self.device), 2, 1),
                                                      smooth(self.soft(outputs[:, mask]), 2, 1))

        loss += self.args.wd_reg * torch.sum(self.get_params() ** 2)
        loss.backward()
        self.opt.step()

        return loss.item()
