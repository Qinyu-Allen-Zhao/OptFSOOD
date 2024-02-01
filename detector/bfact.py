from typing import Any

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from utils.dataset import get_dataloaders
from utils.func import get_feature_train, get_surrogate_ood_features
from utils.metrics import cal_auroc_from_conf
from .base import BasePostprocessor


class BFact(BasePostprocessor):
    def __init__(self, use_surrogate=False):
        super().__init__()
        self.n = 2
        self.threshold = 1e10
        self.use_surrogate = use_surrogate

    def setup(self, net: nn.Module, cfg, dataset_cfg):
        feature_train, _ = get_feature_train(net, cfg, dataset_cfg)
        feature_train_np = feature_train.flatten()

        if not self.use_surrogate:
            self.threshold = np.percentile(feature_train_np, 95)
            print('Threshold at percentile {:2d} over id data is: {}'.format(95, self.threshold))
            return

        # Get the features from the training set
        samples = np.random.choice(range(len(feature_train)), min(50000, len(feature_train)), replace=False)
        feature_train = torch.from_numpy(feature_train[samples]).cuda()

        # Surrogate OOD dataset
        pred_ood, feature_ood = get_surrogate_ood_features(cfg, net, dataset_cfg, is_gaussian=True, use_real=False)
        feature_ood = torch.from_numpy(feature_ood).cuda()

        best_auroc, best_p, best_n = -1, -1, -1
        for p in np.arange(5, 100, 5):
            for n in [1, 2, 3, 4]:
                threshold = np.percentile(feature_train_np, p)
                feature_train_p = self.softcap(feature_train, threshold, n)
                output = net.fc(feature_train_p)
                conf_train = torch.logsumexp(output.data.cpu(), dim=1)

                feature_ood_p = self.softcap(feature_ood, threshold, n)
                output = net.fc(feature_ood_p)
                conf_ood = torch.logsumexp(output.data.cpu(), dim=1)

                auroc = cal_auroc_from_conf(conf_train, conf_ood)
                print(f"p={p}, n={n}, AUROC={auroc * 100:.2f}")
                if auroc > best_auroc:
                    best_auroc = auroc
                    best_p, best_n = p, n

        self.threshold = np.percentile(feature_train_np, best_p)
        self.n = best_n
        print('n={}, Threshold at percentile {:2d} over id data is: {}'.format(best_n, best_p, self.threshold))

    def softcap(self, x, threshold, n):
        return (1 / ((1 + ((x / threshold) ** (2 * n))) ** 0.5)) * x

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        _, feature = net(data, return_feature=True)
        feature = self.softcap(feature, self.threshold, self.n)
        output = net.fc(feature)

        score = torch.softmax(output, dim=1)
        _, pred = torch.max(score, dim=1)

        energyconf = torch.logsumexp(output.data.cpu(), dim=1)

        return pred, energyconf
