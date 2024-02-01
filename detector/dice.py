from typing import Any

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from utils.dataset import get_dataloaders
from utils.func import get_feature_train
from .base import BasePostprocessor

normalizer = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True) + 1e-10


class DICE(BasePostprocessor):
    def __init__(self, p):
        super(DICE, self).__init__()
        self.w, self.b = None, None
        self.p = p
        self.mean_act = None
        self.masked_w = None

    @torch.no_grad()
    def setup(self, net: nn.Module, cfg, dataset_cfg):
        activation_log, _ = get_feature_train(net, cfg, dataset_cfg)
        self.mean_act = activation_log.mean(0)
        self.w, self.b = net.get_fc()
        self.calculate_mask()
        np.save(f"./output/{cfg.exp_space}_dice_mask.npy", self.masked_w.cpu().numpy())

    @torch.no_grad()
    def calculate_mask(self):
        contrib = self.mean_act[None, :] * self.w.data.squeeze().cpu().numpy()
        self.thresh = np.percentile(contrib, self.p)
        mask = torch.Tensor((contrib > self.thresh)).cuda()
        self.masked_w = self.w * mask

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        _, feature = net(data, return_feature=True)
        vote = feature[:, None, :] * self.masked_w
        output = vote.sum(2) + self.b
        _, pred = torch.max(torch.softmax(output, dim=1), dim=1)
        energyconf = torch.logsumexp(output.data.cpu(), dim=1)
        return pred, energyconf
