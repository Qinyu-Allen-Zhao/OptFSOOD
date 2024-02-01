import os
from typing import Any

import time
import numpy as np
import tqdm
import torch
import torch.nn as nn

from utils.func import get_feature_train
from .base import BasePostprocessor


class Ours(BasePostprocessor):
    def __init__(self, use_ood_score="vanilla"):
        super().__init__()
        self.left_boundary = None
        self.w = None
        self.b = None
        self.width = 0.1
        self.theta = None
        self.use_ood_score = use_ood_score
        self.dynamic_weight = False
        self.MAX_NUM_ITERATIONS = 10

    @torch.no_grad()
    def setup(self, net: nn.Module, cfg, dataset_cfg):
        w, b = net.get_fc()
        w_np, b_np = w.detach().cpu().numpy(), b[None, :].detach().cpu().numpy()
        self.w = w

        # Store the extracted results for speed up in experiments
        stored = f"./output/logit_contrib/{cfg.dataset.id_dataset}_{cfg.model_type}_train.npz"
        if os.path.isfile(stored):
            print("Loading stored results ... ")
            train_info = np.load(stored)
            self.left_boundary = train_info['boundaries']
            self.width = train_info['width'].item()
            lc_fv_list = train_info['lc_fv_list']
        else:
            # Get the features from the training set
            feature, pred = get_feature_train(net, cfg, dataset_cfg)
            # feature, pred = feature[:1000], pred[:1000]

            # Calculate Boundaries
            left_b = np.quantile(feature, 1e-3)
            right_b = np.quantile(feature, 1-1e-3)
            print(f"Search Range: [{left_b:.2f}, {right_b:.2f})")

            self.width = (right_b-left_b) / 100.0
            self.left_boundary = np.arange(left_b, right_b, self.width)

            if not self.dynamic_weight:
                print("Optimize with fixed weight ...")
                lc_fv_list = self.get_lc_fv_list(w_np, feature, pred)
                np.savez(stored, boundaries=self.left_boundary, width=np.array([self.width]), lc_fv_list=lc_fv_list)
            else:
                # Dynamic weight, please see the appendix of our paper
                feature_ori = feature.copy()

                feature = feature_ori[np.random.choice(range(len(feature_ori)), 10000, replace=False)]
                feat_p = feature.copy()
                for _ in tqdm.trange(self.MAX_NUM_ITERATIONS):
                    logit = feat_p @ w_np.T + b_np
                    pred = np.argmax(logit, axis=1)
                    lc_fv_list = self.get_lc_fv_list(w_np, feature, pred)
                    theta = lc_fv_list / np.linalg.norm(lc_fv_list, 2) * np.sqrt(len(lc_fv_list))
                    print(theta[:10])
                    
                    feature = feature_ori[np.random.choice(range(len(feature_ori)), 10000, replace=False)]
                    feat_p = np.zeros_like(feature)
                    for i, x in enumerate(self.left_boundary):
                        mask = (feature >= x) & (feature < x + self.width)
                        feat_p += mask * feature * theta[i]

        print("I(z): ", lc_fv_list)
        self.theta = lc_fv_list / np.linalg.norm(lc_fv_list, 2) * 1000  # * np.sqrt(len(lc_fv_list))
        print("theta: ", self.theta)
        print("Norm of theta: ", np.dot(self.theta, self.theta))

        # Calculate the ratio of changed weights
        print("Calculating the ratio of changed weights...")
        feature, pred = get_feature_train(net, cfg, dataset_cfg)
        sample = np.random.choice(range(len(feature)), 10000, replace=False)
        feature, pred = feature[sample], pred[sample]
        feat_p = np.zeros_like(feature)
        for i, x in enumerate(self.left_boundary):
            mask = (feature >= x) & (feature < x + self.width)
            feat_p += mask * feature * self.theta[i]
        logits = feat_p @ w_np.T + b_np
        pred_new = np.argmax(logits, axis=1)
        print(pred)
        print(pred_new)
        print(f"Ratio of changed weight indices: {np.sum(np.abs(pred - pred_new) > 0) / len(pred) * 100:.1f}")

        self.theta = torch.from_numpy(self.theta[np.newaxis, :]).cuda()

    def get_lc_fv_list(self, w, feature, pred):
        lc = w[pred] * feature
        lc_fv_list = []
        for b in tqdm.tqdm(self.left_boundary):
            mask = (feature >= b) & (feature < b + self.width)
            feat_masked = mask * lc
            res = np.mean(np.sum(feat_masked, axis=1))
            lc_fv_list.append(res)
        lc_fv_list = np.array(lc_fv_list)

        return lc_fv_list

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        time_start = time.time()
        output, feature = net(data, return_feature=True)
        score = torch.softmax(output, dim=1)
        _, pred = torch.max(score, dim=1)

        feat_p = torch.zeros_like(feature).cuda()
        for i, x in enumerate(self.left_boundary):
            mask = (feature >= x) & (feature < x + self.width)
            feat_p += mask * feature * self.theta[0][i]

        if self.use_ood_score == 'energy':
            # Energy:
            output = net.fc(feat_p)
            conf = torch.logsumexp(output, dim=1)
        else:
            # Vanilla
            conf = torch.sum(self.w[pred] * feat_p, dim=1).cpu()

        # MSP:
        # output = net.fc(feat_p)
        # score = torch.softmax(output, dim=1)
        # conf, _ = torch.max(score, dim=1)

        # # MLS
        # output = net.fc(feat_p)
        # conf, _ = torch.max(output, dim=1)

        time_end = time.time()
        print(f'time cost: {time_end - time_start} s')
        return pred, conf
