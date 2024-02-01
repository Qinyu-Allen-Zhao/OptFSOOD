from typing import Any

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from utils.dataset import get_dataloaders
from utils.func import get_feature_train, get_surrogate_ood_features
from utils.metrics import cal_auroc_from_conf
from .base import BasePostprocessor


class React(BasePostprocessor):
    def __init__(self, use_surrogate=False):
        super(React, self).__init__()
        self.max_threshold = 1.014
        self.min_threshold = None
        self.percentile = 90
        self.use_surrogate = use_surrogate

    def setup(self, net: nn.Module, cfg, dataset_cfg):
        feature_train, pred_train = get_feature_train(net, cfg, dataset_cfg)
        feature_train_np = feature_train.flatten()
        if self.use_surrogate:
            # Get the features from the training set
            samples = np.random.choice(range(len(feature_train)), min(50000, len(feature_train)), replace=False)
            feature_train = torch.from_numpy(feature_train[samples]).cuda()

            # Surrogate OOD dataset
            pred_ood, feature_ood = get_surrogate_ood_features(cfg, net, dataset_cfg, is_gaussian=True, use_real=False)
            feature_ood = torch.from_numpy(feature_ood).cuda()

            best_auroc, best_p = -1, 5
            for p in np.arange(5, 100, 5):
                self.percentile = p
                threshold = np.percentile(feature_train_np, p)
                feature_train_p = feature_train.clip(max=threshold)
                output = net.fc(feature_train_p)
                conf_train = torch.logsumexp(output.data.cpu(), dim=1)

                feature_ood_p = feature_ood.clip(max=threshold)
                output = net.fc(feature_ood_p)
                conf_ood = torch.logsumexp(output.data.cpu(), dim=1)

                auroc = cal_auroc_from_conf(conf_train, conf_ood)
                print(f"p={p}, AUROC={auroc*100:.2f}")
                if auroc > best_auroc:
                    best_auroc = auroc
                    best_p = p
            print(f"Best p={best_p}")
            self.percentile = best_p

        self.max_threshold = np.percentile(feature_train_np, self.percentile)
        print('Threshold at percentile {:2d} over id data is: {}'.format(
              self.percentile, self.max_threshold))

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        _, feature = net(data, return_feature=True)
        feature = feature.clip(max=self.max_threshold)
        output = net.fc(feature)

        score = torch.softmax(output, dim=1)
        _, pred = torch.max(score, dim=1)

        energyconf = torch.logsumexp(output.data.cpu(), dim=1)
        return pred, energyconf
