# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the gem_license file in the root of this source tree.

import quadprog

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from utils.buffer import Buffer
from utils.args import *
from utils.training import batch_resize, batch_visualization
from models.utils.continual_model import ContinualModel
from torch.optim import SGD
from torchvision.models import resnet18
from pytorch_grad_cam import GradCAM
from bayes_pcn.const import *
from bayes_pcn.trainer import train_epoch, score_epoch, model_dispatcher


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Gradient Episodic Memory.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    # remove minibatch_size from parser
    for i in range(len(parser._actions)):
        if parser._actions[i].dest == 'minibatch_size':
            del parser._actions[i]
            break

    parser.add_argument('--gamma', type=float, default=None,
                        help='Margin parameter for GEM.')
    return parser


def store_grad(params, grads, grad_dims):
    """
        This stores parameter gradients of past tasks.
        pp: parameters
        grads: gradients
        grad_dims: list with number of parameters per layers
    """
    # store the gradients
    grads.fill_(0.0)
    count = 0
    for param in params():
        if param.grad is not None:
            begin = 0 if count == 0 else sum(grad_dims[:count])
            end = np.sum(grad_dims[:count + 1])
            grads[begin: end].copy_(param.grad.data.view(-1))
        count += 1


def overwrite_grad(params, newgrad, grad_dims):
    """
        This is used to overwrite the gradients with a new gradient
        vector, whenever violations occur.
        pp: parameters
        newgrad: corrected gradient
        grad_dims: list storing number of parameters at each layer
    """
    count = 0
    for param in params():
        if param.grad is not None:
            begin = 0 if count == 0 else sum(grad_dims[:count])
            end = sum(grad_dims[:count + 1])
            this_grad = newgrad[begin: end].contiguous().view(
                param.grad.data.size())
            param.grad.data.copy_(this_grad)
        count += 1


def project2cone2(gradient, memories, margin=0.5, eps=1e-3):
    """
        Solves the GEM dual QP described in the paper given a proposed
        gradient "gradient", and a memory of task gradients "memories".
        Overwrites "gradient" with the final projected update.

        input:  gradient, p-vector
        input:  memories, (t * p)-vector
        output: x, p-vector
    """
    memories_np = memories.cpu().t().double().numpy()
    gradient_np = gradient.cpu().contiguous().view(-1).double().numpy()
    n_rows = memories_np.shape[0]
    self_prod = np.dot(memories_np, memories_np.transpose())
    self_prod = 0.5 * (self_prod + self_prod.transpose()) + np.eye(n_rows) * eps
    grad_prod = np.dot(memories_np, gradient_np) * -1
    G = np.eye(n_rows)
    h = np.zeros(n_rows) + margin
    v = quadprog.solve_qp(self_prod, grad_prod, G, h)[0]
    x = np.dot(v, memories_np) + gradient_np
    gradient.copy_(torch.from_numpy(x).view(-1, 1))


class GemBpcn(ContinualModel):
    NAME = 'gem_bpcn'
    COMPATIBILITY = ['class-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super(GemBpcn, self).__init__(backbone, loss, args, transform)
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
        # Allocate temporary synaptic memory
        self.grad_dims = []
        for pp in self.net.fc.parameters():
            self.grad_dims.append(pp.data.numel())
        self.grads_cs = []
        self.grads_da = torch.zeros(np.sum(self.grad_dims)).to(self.device)
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

    def forward(self, x, classifier_only):
        """Use Conv layers + FC layer or FC layer only"""
        if classifier_only:
            return self.net.fc(x)
        else:
            return self.net(x)

    def end_task(self, dataset):
        self.current_task += 1
        self.first_K_batches += int(self.args.samples_per_task / self.args.batch_size)
        
        self.grads_cs.append(torch.zeros(
            np.sum(self.grad_dims)).to(self.device))

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

        # compute gradient on previous tasks
        if len(self.observed_tasks) > 1:
            for tt in range(len(self.observed_tasks) - 1):
                # compute gradient on the memory buffer
                self.opt.zero_grad()
                # fwd/bwd on the examples in the memory
                past_task = self.observed_tasks[tt]
                pt_bsz = self.memory_data[past_task].size(0)
                # (pt_bsz, -1) -> (pt_bsz, 512, 7, 7)
                input_batch = self.memory_data[past_task].view(pt_bsz, 512, 7, 7)
                pt_output_batch = self.forward(
                        self.net.avgpool(input_batch).view(pt_bsz, -1),
                        classifier_only=True)
                penalty = self.loss(pt_output_batch, self.memory_labs[past_task])
                penalty.backward()
                store_grad(self.net.fc.parameters, self.grads_cs[tt], self.grad_dims)

        # now compute the grad on the current data
        self.opt.zero_grad()
        cur_output_batch = self.forward(
            self.net.avgpool(feature_maps).view(real_batch_size, -1),
            classifier_only=True)
        loss = self.loss(cur_output_batch, labels)
        loss.backward()

        # check if gradient violates buffer constraints
        if len(self.observed_tasks) > 1:
            # copy gradient
            store_grad(self.net.fc.parameters, self.grads_da, self.grad_dims)

            dot_prod = torch.mm(self.grads_da.unsqueeze(0),
                            torch.stack(self.grads_cs).T)
            if (dot_prod < 0).sum() != 0:
                project2cone2(self.grads_da.unsqueeze(1),
                              torch.stack(self.grads_cs).T, margin=self.args.gamma)
                # copy gradients back
                overwrite_grad(self.net.fc.parameters, self.grads_da,
                               self.grad_dims)

        self.opt.step()

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
