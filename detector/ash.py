from typing import Any

import numpy as np
import torch
import torch.nn as nn

from utils.func import get_feature_train, get_surrogate_ood_features
from utils.metrics import cal_auroc_from_conf
from .base import BasePostprocessor


def ash_b(x, percentile=65):
    assert x.dim() == 4
    assert 0 <= percentile <= 100
    b, c, h, w = x.shape

    # calculate the sum of the input per sample
    s1 = x.sum(dim=[1, 2, 3])

    n = x.shape[1:].numel()
    k = n - int(np.round(n * percentile / 100.0))
    t = x.view((b, c * h * w))
    v, i = torch.topk(t, k, dim=1)
    fill = s1 / k
    fill = fill.unsqueeze(dim=1).expand(v.shape)
    t.zero_().scatter_(dim=1, index=i, src=fill)
    return x


def ash_p(x, percentile=65):
    assert x.dim() == 4
    assert 0 <= percentile <= 100

    b, c, h, w = x.shape

    n = x.shape[1:].numel()
    k = n - int(np.round(n * percentile / 100.0))
    t = x.view((b, c * h * w))
    v, i = torch.topk(t, k, dim=1)
    t.zero_().scatter_(dim=1, index=i, src=v)

    return x


def ash_s(x, percentile=65):
    assert x.dim() == 4
    assert 0 <= percentile <= 100
    b, c, h, w = x.shape

    # calculate the sum of the input per sample
    s1 = x.sum(dim=[1, 2, 3])
    n = x.shape[1:].numel()
    k = n - int(np.round(n * percentile / 100.0))
    t = x.view((b, c * h * w))
    v, i = torch.topk(t, k, dim=1)
    t.zero_().scatter_(dim=1, index=i, src=v)

    # calculate new sum of the input per sample after pruning
    s2 = x.sum(dim=[1, 2, 3])

    # apply sharpening
    scale = s1 / s2
    x = x * torch.exp(scale[:, None, None, None])

    return x


def ash_rand(x, percentile=75, r1=0, r2=10):
    assert x.dim() == 4
    assert 0 <= percentile <= 100
    b, c, h, w = x.shape

    n = x.shape[1:].numel()
    k = n - int(np.round(n * percentile / 100.0))
    t = x.view((b, c * h * w))
    v, i = torch.topk(t, k, dim=1)
    v = v.uniform_(r1, r2)
    t.zero_().scatter_(dim=1, index=i, src=v)
    return x


class ASH(BasePostprocessor):
    def __init__(self, shaping_func='ash_s', percentile=95, use_surrogate=False):
        super().__init__()
        self.percentile = percentile
        self.shaping_func = eval(shaping_func)
        self.use_surrogate = use_surrogate

    def feature_reshape(self, feature):
        feature_map = feature[:, :, None, None]
        feature_p = self.shaping_func(feature_map, self.percentile)
        return torch.flatten(feature_p, 1)

    def setup(self, net: nn.Module, cfg, dataset_cfg):
        if not self.use_surrogate:
            return
        # Get the features from the training set
        feature_train, pred_train = get_feature_train(net, cfg, dataset_cfg)
        samples = np.random.choice(range(len(feature_train)), min(50000, len(feature_train)), replace=False)
        feature_train = torch.from_numpy(feature_train[samples]).cuda()

        # Surrogate OOD dataset
        pred_ood, feature_ood = get_surrogate_ood_features(cfg, net, dataset_cfg, is_gaussian=True, use_real=False)
        feature_ood = torch.from_numpy(feature_ood).cuda()

        best_auroc, best_p = -1, 5
        for p in np.arange(5, 100, 5):
            self.percentile = p
            feature_train_p = self.feature_reshape(feature_train)
            output = net.fc(feature_train_p)
            conf_train = torch.logsumexp(output.data.cpu(), dim=1)

            feature_ood_p = self.feature_reshape(feature_ood)
            output = net.fc(feature_ood_p)
            conf_ood = torch.logsumexp(output.data.cpu(), dim=1)
            auroc = cal_auroc_from_conf(conf_train, conf_ood)

            print(f"p={p}, AUROC={auroc*100:.2f}")
            if auroc > best_auroc:
                best_auroc = auroc
                best_p = p
        print(f"Best p={best_p}")
        self.percentile = best_p

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        _, feature = net(data, return_feature=True)
        feature = self.feature_reshape(feature)
        output = net.fc(feature)

        score = torch.softmax(output, dim=1)
        _, pred = torch.max(score, dim=1)

        energyconf = torch.logsumexp(output.data.cpu(), dim=1)
        return pred, energyconf
