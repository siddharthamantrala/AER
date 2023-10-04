# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from utils.buffer import Buffer
from utils.args import *
from models.utils.continual_model import ContinualModel

from utils.training import batch_resize, batch_visualization
from torch.optim import SGD
from torchvision.models import resnet18
from pytorch_grad_cam import GradCAM
from bayes_pcn.const import *
from bayes_pcn.trainer import train_epoch, score_epoch, model_dispatcher
from models.agem_bpcn import sample_from_memory


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual Learning via'
                                        ' Meta-Experience Replay.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    # remove batch_size from parser
    for i in range(len(parser._actions)):
        if parser._actions[i].dest == 'batch_size':
            del parser._actions[i]
            break

    parser.add_argument('--beta', type=float, required=True,
                        help='Within-batch update beta parameter.')
    parser.add_argument('--gamma', type=float, required=True,
                        help='Across-batch update gamma parameter.')
    parser.add_argument('--batch_num', type=int, required=True,
                        help='Number of batches extracted from the buffer.')

    return parser


class MerBpcn(ContinualModel):
    NAME = 'mer_bpcn'
    COMPATIBILITY = ['class-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super(MerBpcn, self).__init__(backbone, loss, args, transform)
        self.n_inputs = 512 * 7 * 7
        self.n_classes = backbone.num_classes
        self.current_task = 0
        self.net = resnet18(weights='IMAGENET1K_V1')
        # modify classifier
        self.net.fc = nn.Linear(512, self.n_classes)
        # check model architecture
        print(self.net)
        # only optimize the last FC layer
        self.opt = SGD([{'params': self.net.fc.parameters()}], lr=self.args.lr)
        # load bayes-pcn
        dataset_info = {'x_dim': self.n_inputs}
        self.bpcn = model_dispatcher(args=self.args, dataset_info=dataset_info)
        if self.args.cuda and torch.cuda.device_count() > 0:
            self.bpcn.device = torch.device('cuda')
        # MSELoss for bpcn.infer
        self.mse = nn.MSELoss()
        # load GradCAM
        self.target_layer = self.net.layer4[-1]
        self.cam = GradCAM(model=self.net, target_layer=self.target_layer, 
                           use_cuda=self.args.cuda)
        # set the threshold for GradCAM masking
        self.theta = self.args.theta
        # set 
        self.alpha = self.args.memory_strength
        # the first K batches to skip
        self.first_K_batches = int((self.args.samples_per_task / self.args.batch_size) - 
                                   np.ceil(self.n_memories / self.args.batch_size))
        # allocate episodic memory
        self.memory_data = {}
        self.memory_labs = {}
        self.pxl_needed = {}
        # allocate counters
        self.observed_tasks = []
        self.old_task = -1
        self.mem_cnt = 0
        self.batch_index = 0
        self.nc_per_task = int(self.n_classes / self.args.n_tasks)
        # set parameters for dynamic memory
        self.max_pxl = self.n_memories * self.n_inputs
        self.pxl_stored = np.zeros(self.n_classes)
        self.img_stored = np.zeros(self.n_classes)

    def get_params(self) -> torch.Tensor:
        """
        Returns all the parameters concatenated in a single tensor.
        :return: parameters tensor (??)
        """
        params = []
        for pp in list(self.net.fc.parameters()):
            params.append(pp.view(-1))
        return torch.cat(params)

    def set_params(self, new_params: torch.Tensor) -> None:
        """
        Sets the parameters to a given value.
        :param new_params: concatenated values to be set (??)
        """
        assert new_params.size() == self.get_params().size()
        progress = 0
        for pp in list(self.net.fc.parameters()):
            cand_params = new_params[progress: progress +
                torch.tensor(pp.size()).prod()].view(pp.size())
            progress += torch.tensor(pp.size()).prod()
            pp.data = cand_params

    def forward(self, x, classifier_only):
        """Use Conv layers + FC layer or FC layer only"""
        if classifier_only:
            return self.net.fc(x)
        else:
            return self.net(x)

    def observe(self, inputs, labels, not_aug_inputs):
        """inputs: transformed data
        not_aug_inputs: original images
        """
        # use not_aug_inputs (original images)
        real_batch_size = inputs.shape[0]
        t = self.current_task

        # new task comes
        if t != self.old_task:
            self.observed_tasks.append(t)
            self.old_task = t
            # initialize episodic memory for the new task
            self.memory_data[t] = torch.FloatTensor(real_batch_size, self.n_inputs)
            self.memory_labs[t] = torch.LongTensor(real_batch_size)
            self.pxl_needed[t] = np.zeros(real_batch_size)
            if self.args.cuda:
                self.memory_data[t] = self.memory_data[t].cuda()
                self.memory_labs[t] = self.memory_labs[t].cuda()
        
        # get alpha (feature importance)
        inputs = batch_resize(inputs)  # torch.Size([bsz, 3, 32, 32]) -> torch.Size([bsz, 3, 224, 224])        
        target_category = labels.cpu().tolist()
        batch_feature_importance = self.cam.get_feature_importance(
            input_tensor=inputs, target_category=target_category)
        # get positive feature importance
        batch_feature_importance = np.maximum(batch_feature_importance, 0)
        # get feature map of x
        feature_maps = self.cam.get_feature_map(inputs)  # torch.Size([bsz, 3, 224, 224]) -> torch.Size([bsz, 512, 7, 7])
        if self.args.cuda:
            feature_maps = feature_maps.cuda()

        theta_A0 = self.get_params().data.clone()

        for i in range(self.args.batch_num):
            theta_Wi0 = self.get_params().data.clone()
            self.zero_grad()
            ptloss = 0.0
            # observe batches from memory
            if len(self.observed_tasks) > 1:
                buf_inputs, buf_labels = sample_from_memory(self.memory_data, self.memory_labs, real_batch_size)
                pt_bsz = len(buf_inputs)
                buf_inputs = buf_inputs.view(pt_bsz, 512, 7, 7)
                pt_output_batch = self.forward(
                        self.net.avgpool(buf_inputs).view(pt_bsz, -1),
                        classifier_only=True)
                ptloss = self.loss(pt_output_batch, buf_labels)
            # then observe current batch
            cur_output_batch = self.forward(
                    self.net.avgpool(feature_maps).view(real_batch_size, -1),
                    classifier_only=True)
            ctloss = self.loss(cur_output_batch, labels)

            loss = ctloss + ptloss * self.alpha
            loss.backward()
            self.opt.step()

            # within batch reptile meta-update
            new_params = theta_Wi0 + self.args.beta * (self.get_params() - theta_Wi0)
            self.set_params(new_params)

        # across batch reptile meta-update
        new_new_params = theta_A0 + self.args.gamma * (self.get_params() - theta_A0)
        self.set_params(new_new_params)

        # only train on the last minibatch
        if self.batch_index >= self.first_K_batches and self.theta > 0:
            # bayes-pcn forget previous samples
            if len(self.observed_tasks) > 0:
                self.bpcn.forget()
            # train bayes-pcn
            _ = self.bpcn.learn(X_obs=feature_maps.view(real_batch_size, -1))

        # define some auxiliary variables
        masked_x = torch.empty_like(feature_maps)
        fixed_index = torch.empty_like(feature_maps)
        # number of non-zero pixels for each image within this mini-batch
        pxl_needed = np.zeros(real_batch_size)
        # second, use Grad-CAM masks to mask images within the current mini-batch
        for i in range(real_batch_size):
            tmp_x = feature_maps[i]  # torch.Size([512, 7, 7])
            # get binary mask by threshold theta, where 1 indicates fixed pixel and 0 indicates dropped pixel
            feature_importance = batch_feature_importance[i]  # (512,)
            # percentile thresholding
            threshold = np.percentile(feature_importance, self.theta * 100)
            mask = np.where(feature_importance < threshold, 0, 1)
            # (512,) -> (512, 7, 7)
            mask_ones = np.ones_like(tmp_x.cpu())
            mask = mask[:, None, None] * mask_ones
            # mask the image
            mask = torch.tensor(mask, dtype=torch.bool)
            if self.args.cuda:
                mask = mask.cuda()
            tmp_masked_x = mask * tmp_x
            # recording
            masked_x[i] = tmp_masked_x
            fixed_index[i] = mask
            # calculate number of non-zero pixels of this image after applying the mask
            pxl_needed[i] = np.count_nonzero(tmp_masked_x.cpu())
        # last, reshape images & masks back to (bsz * feature) dimension and transfer to GPUs
        masked_x = masked_x.view(real_batch_size, -1)
        fixed_index = fixed_index.view(real_batch_size, -1)
        if self.args.cuda:
            masked_x = masked_x.cuda()
            fixed_index = fixed_index.cuda()
        total_pxl_needed = np.sum(pxl_needed)

        # Use bayes-pcn to recall images
        if self.batch_index >= self.first_K_batches and self.theta > 0:
            n_repeat = 1  # how much ICM iteration to perform at inference
            masked_x = self.bpcn.infer(
                X_obs=masked_x, n_repeat=n_repeat, fixed_indices=fixed_index)
            masked_x = masked_x.data
            recall_mse = self.mse(feature_maps.view(real_batch_size, -1), masked_x)
            print(f"Recall MSE on task {t}:", recall_mse.item())

        # now we begin to store the mini-batch into episodic memory with dynamic size
        if self.img_stored[t] == 0:
            self.memory_data[t].copy_(masked_x)
            self.img_stored[t] += real_batch_size
            self.pxl_stored[t] += total_pxl_needed
            self.memory_labs[t].copy_(labels)
            self.pxl_needed[t] = pxl_needed

        elif self.pxl_stored[t] + total_pxl_needed <= self.max_pxl:
            self.memory_data[t] = torch.cat((self.memory_data[t], masked_x), 0)
            self.img_stored[t] += real_batch_size
            self.pxl_stored[t] += total_pxl_needed
            self.memory_labs[t] = torch.cat((self.memory_labs[t], labels))
            self.pxl_needed[t] = np.concatenate(
                (self.pxl_needed[t], pxl_needed), axis=None)
        # if pxl_stored exceeds max_pxl
        else:
            pxl_released = 0
            for k in range(int(self.img_stored[t])):
                pxl_released += self.pxl_needed[t][k]
                if self.pxl_stored[t] + total_pxl_needed - pxl_released <= self.max_pxl:
                    # remove images up to the current one from memory
                    self.memory_data[t] = self.memory_data[t][k+1:,]
                    self.memory_labs[t] = self.memory_labs[t][k+1:]
                    self.pxl_needed[t] = self.pxl_needed[t][k+1:]
                    self.img_stored[t] -= (k + 1)
                    self.pxl_stored[t] -= pxl_released
                    # now store the current mini-batch into memory
                    self.memory_data[t] = torch.cat((self.memory_data[t], masked_x[:k+1]), 0)
                    self.memory_labs[t] = torch.cat((self.memory_labs[t], labels[:k+1]))
                    self.pxl_needed[t] = np.concatenate(
                        (self.pxl_needed[t], pxl_needed[:k+1]), axis=None)
                    self.img_stored[t] += (k + 1)
                    self.pxl_stored[t] += np.sum(pxl_needed[:k+1])
                    break
                else:
                    continue
        self.batch_index += 1
        return loss.item()

    def end_task(self, dataset) -> None:
        self.current_task += 1
        self.first_K_batches += int(self.args.samples_per_task / self.args.batch_size)
        