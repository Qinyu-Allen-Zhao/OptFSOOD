from typing import Any

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from utils.dataset import get_dataloaders
from utils.func import get_feature_train, get_surrogate_ood_features
from utils.metrics import cal_auroc_from_conf
from .base import BasePostprocessor


class VRA_P(BasePostprocessor):
    def __init__(self, x1=0.5, x2=1.0, quantile=False, use_surrogate=False):
        super().__init__()
        self.x1 = x1
        self.x2 = x2
        self.quantile = False
        self.use_surrogate = use_surrogate

    def setup(self, net: nn.Module, cfg, dataset_cfg):
        if self.use_surrogate:
            feature_train, _ = get_feature_train(net, cfg, dataset_cfg)
            feature_train_np = feature_train.flatten()

            # Get the features from the training set
            samples = np.random.choice(range(len(feature_train)), min(50000, len(feature_train)), replace=False)
            feature_train = torch.from_numpy(feature_train[samples]).cuda()

            # Surrogate OOD dataset
            pred_ood, feature_ood = get_surrogate_ood_features(cfg, net, dataset_cfg, is_gaussian=True,
                                                                     use_real=False)
            feature_ood = torch.from_numpy(feature_ood).cuda()

            best_auroc, best_x1, best_x2 = -1, -1, -1

            for x1 in np.arange(0.0, 0.4, 0.1):
                for x2 in np.arange(0.5, 1.5, 0.1):
                    feature_train_p = self.feature_shape(feature_train, x1, x2)
                    output = net.fc(feature_train_p)
                    conf_train = torch.logsumexp(output.data.cpu(), dim=1)

                    feature_ood_p = self.feature_shape(feature_ood, x1, x2)
                    output = net.fc(feature_ood_p)
                    conf_ood = torch.logsumexp(output.data.cpu(), dim=1)

                    auroc = cal_auroc_from_conf(conf_train, conf_ood)
                    print(f"x1={x1}, x2={x2}, AUROC={auroc * 100:.2f}")
                    if auroc > best_auroc:
                        best_auroc = auroc
                        best_x1, best_x2 = x1, x2

            self.x1 = best_x1
            self.x2 = best_x2

        if self.quantile:
            train_loader = get_dataloaders(["train"], cfg, dataset_cfg, batch_size=128)[0]
            activation_log = []
            net.eval()
            with torch.no_grad():
                for data, label in tqdm(train_loader,
                                        desc='Eval: ',
                                        position=0,
                                        leave=True):
                    data = data.cuda().float()

                    batch_size = data.shape[0]

                    _, feature = net(data, return_feature=True)
                    dim = feature.shape[1]
                    activation_log.append(feature.data.cpu().numpy().reshape(
                        batch_size, dim, -1).mean(2))

            activation_log = np.concatenate(activation_log, axis=0)
            self.x1 = np.percentile(activation_log.flatten(), self.x1 * 100)
            self.x2 = np.percentile(activation_log.flatten(), self.x2 * 100)

        print(f'Thresholds: x1={self.x1}, x2={self.x2}')

    def feature_shape(self, feature, x1, x2):
        feature = feature.clip(max=x2)
        mask = torch.Tensor(feature > x1).int()
        feature = mask * feature

        return feature

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        _, feature = net(data, return_feature=True)
        feature = self.feature_shape(feature, self.x1, self.x2)
        mask = torch.Tensor(feature > self.x1).int()
        feature = mask * feature

        output = net.fc(feature)
        score = torch.softmax(output, dim=1)
        _, pred = torch.max(score, dim=1)
        energyconf = torch.logsumexp(output.data.cpu(), dim=1)

        return pred, energyconf
