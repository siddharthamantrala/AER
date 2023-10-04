from copy import deepcopy
import torch
import os
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import functional as F
import random

from utils.buffer import Buffer
from utils.args import *
from utils.training import batch_resize, batch_visualization
from models.utils.continual_model import ContinualModel
from torch.optim import SGD
from torchvision.models import resnet18
from pytorch_grad_cam import GradCAM
from bayes_pcn.const import *
from bayes_pcn.trainer import train_epoch, score_epoch, model_dispatcher
from models.agem_bpcn import sample_from_memory


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Complementary Learning Systems Based Experience Replay')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)

    # Consistency Regularization Weight
    parser.add_argument('--reg_weight', type=float, default=0.1)

    # Stable Model parameters
    parser.add_argument('--stable_model_update_freq', type=float, default=0.70)
    parser.add_argument('--stable_model_alpha', type=float, default=0.999)

    # Plastic Model Parameters
    parser.add_argument('--plastic_model_update_freq', type=float, default=0.90)
    parser.add_argument('--plastic_model_alpha', type=float, default=0.999)

    return parser


# =============================================================================
# Mean-ER
# =============================================================================
class CLSERBpcn(ContinualModel):
    NAME = 'clser_bpcn'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(CLSERBpcn, self).__init__(backbone, loss, args, transform)
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
        # Initialize plastic and stable model
        self.plastic_model = deepcopy(self.net).to(self.device)
        self.stable_model = deepcopy(self.net).to(self.device)
        # set regularization weight
        self.reg_weight = args.reg_weight
        # set parameters for plastic model
        self.plastic_model_update_freq = args.plastic_model_update_freq
        self.plastic_model_alpha = args.plastic_model_alpha
        # set parameters for stable model
        self.stable_model_update_freq = args.stable_model_update_freq
        self.stable_model_alpha = args.stable_model_alpha

        self.consistency_loss = nn.MSELoss(reduction='none')
        self.current_task = 0
        self.global_step = 0

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

        self.opt.zero_grad()
        loss = 0

        if len(self.observed_tasks) > 1:
            
            buf_inputs, buf_labels = sample_from_memory(
                self.memory_data, self.memory_labs, real_batch_size)
            buf_inputs = buf_inputs.view(real_batch_size, 512, 7, 7)

            stable_model_logits = self.stable_model.fc(
                self.stable_model.avgpool(buf_inputs).view(real_batch_size, -1))
            plastic_model_logits = self.plastic_model.fc(
                self.plastic_model.avgpool(buf_inputs).view(real_batch_size, -1))

            stable_model_prob = F.softmax(stable_model_logits, 1)
            plastic_model_prob = F.softmax(plastic_model_logits, 1)

            label_mask = F.one_hot(buf_labels, num_classes=stable_model_logits.shape[-1]) > 0
            sel_idx = stable_model_prob[label_mask] > plastic_model_prob[label_mask]
            sel_idx = sel_idx.unsqueeze(1)

            ema_logits = torch.where(
                sel_idx,
                stable_model_logits,
                plastic_model_logits,
            )

            org_model_logits = self.net.fc(
                self.net.avgpool(buf_inputs).view(real_batch_size, -1))
            l_cons = torch.mean(self.consistency_loss(org_model_logits, ema_logits.detach()))
            l_reg = self.args.reg_weight * l_cons
            loss += l_reg

            if hasattr(self, 'writer'):
                self.writer.add_scalar(f'Task {self.current_task}/l_cons', l_cons.item(), self.iteration)
                self.writer.add_scalar(f'Task {self.current_task}/l_reg', l_reg.item(), self.iteration)

            combined_feature_maps = torch.cat((feature_maps, buf_inputs))
            combined_labels = torch.cat((labels, buf_labels))
            
            combined_outputs = self.net.fc(
                self.net.avgpool(combined_feature_maps).view(combined_feature_maps.size(0), -1))
            ce_loss = self.loss(combined_outputs, combined_labels)
            # Log values
            if hasattr(self, 'writer'):
                self.writer.add_scalar(f'Task {self.current_task}/l_reg', l_reg.item(), self.iteration)

        elif len(self.observed_tasks) == 1:
            outputs = self.net.fc(
                self.net.avgpool(feature_maps).view(real_batch_size, -1))
            ce_loss = self.loss(outputs, labels)
        
        loss += ce_loss

        # Log values
        if hasattr(self, 'writer'):
            self.writer.add_scalar(f'Task {self.current_task}/ce_loss', ce_loss.item(), self.iteration)
            self.writer.add_scalar(f'Task {self.current_task}/loss', loss.item(), self.iteration)

        loss.backward()
        self.opt.step()

        # Update the ema model
        self.global_step += 1
        if torch.rand(1) < self.plastic_model_update_freq:
            self.update_plastic_model_variables()

        if torch.rand(1) < self.stable_model_update_freq:
            self.update_stable_model_variables()

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
        
    def update_plastic_model_variables(self):
        alpha = min(1 - 1 / (self.global_step + 1), self.plastic_model_alpha)
        for ema_param, param in zip(self.plastic_model.fc.parameters(), self.net.fc.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

    def update_stable_model_variables(self):
        alpha = min(1 - 1 / (self.global_step + 1),  self.stable_model_alpha)
        for ema_param, param in zip(self.stable_model.fc.parameters(), self.net.fc.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
