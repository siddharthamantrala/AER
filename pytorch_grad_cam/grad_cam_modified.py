import cv2
import numpy as np
import torch
from pytorch_grad_cam.base_cam_gem import BaseCAM

class GradCAM_modified(BaseCAM):
    def __init__(self, model, target_layer, use_cuda=False, 
        reshape_transform=None):
        super(GradCAM_modified, self).__init__(model, target_layer, use_cuda, reshape_transform)

    def get_cam_weights(self,
                        input_tensor,
                        target_category,
                        activations,
                        grads):
        return np.mean(grads, axis=(2, 3))