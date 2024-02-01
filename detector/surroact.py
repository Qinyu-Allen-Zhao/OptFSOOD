import os
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression

from utils.func import get_feature_train, get_surrogate_ood_features, get_fxw
from .base import BasePostprocessor



class SurrogateAct(BasePostprocessor):
    def __init__(self, is_gaussian=True, use_real=False):
        super().__init__()
        self.x_lim = None
        self.theta = None
        self.width = None
        self.left_boundary = None
        self.is_gaussian = is_gaussian
        self.use_real = use_real
        print(f"Use Gaussian: {self.is_gaussian}, Use Real: {self.use_real}")

    @torch.no_grad()
    def setup(self, net: nn.Module, cfg, dataset_cfg):
        # Get the features from the training set
        feature_train, pred_train = get_feature_train(net, cfg, dataset_cfg)
        samples = np.random.choice(range(len(feature_train)), min(50000, len(feature_train)),
                                   replace=False)
        feature_train = torch.from_numpy(feature_train[samples])
        pred_train = torch.from_numpy(pred_train[samples])

        # Boundaries
        left_b = np.quantile(feature_train, 1e-3)
        right_b = np.quantile(feature_train, 1 - 1e-3)
        print(f"Search Range: [{left_b:.2f}, {right_b:.2f})")
        self.x_lim = [left_b, right_b]
        self.width = (self.x_lim[1] - self.x_lim[0]) / 100
        self.left_boundary = np.arange(self.x_lim[0], self.x_lim[1], self.width)

        # Get corresponding weights
        w, _ = net.get_fc()
        self.w = w

        stored = f"./output/logit_contrib/{cfg.dataset.id_dataset}_{cfg.model_type}_train_further.pth"
        if False: #os.path.isfile(stored):
            fxw_train = torch.load(stored)
        else:
            fxw_train = get_fxw(w.cpu(), pred_train, feature_train, self.left_boundary, self.width, show_progress=True)
            torch.save(fxw_train, stored)

        # Surrogate OOD dataset
        pred_ood, feature_ood = get_surrogate_ood_features(cfg, net, dataset_cfg, self.is_gaussian, self.use_real)
        feature_ood = torch.from_numpy(feature_ood)
        pred_ood = torch.from_numpy(pred_ood)
        fxw_ood = get_fxw(w.cpu(), pred_ood, feature_ood, self.left_boundary, self.width, show_progress=True)

        x = np.concatenate([fxw_train.numpy(), fxw_ood.numpy()], axis=0)
        y = np.zeros(len(x))
        y[:len(fxw_train)] = 1
        classifier = LogisticRegression(random_state=42)
        classifier.fit(x, y)
        self.theta = torch.from_numpy(classifier.coef_).cuda()
        print(self.x_lim)
        print(self.theta)

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        output, feature = net(data, return_feature=True)
        score = torch.softmax(output, dim=1)
        _, pred = torch.max(score, dim=1)

        fxws = get_fxw(self.w, pred, feature, self.left_boundary, self.width)
        conf = torch.sum(fxws * self.theta, dim=1).cpu()

        return pred, conf
