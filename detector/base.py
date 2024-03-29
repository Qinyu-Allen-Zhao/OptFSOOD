from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class BasePostprocessor:
    def __init__(self):
        pass

    def setup(self, net: nn.Module, cfg, dataset_cfg):
        pass

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        output = net(data)
        score = torch.softmax(output, dim=1)
        conf, pred = torch.max(score, dim=1)
        return pred, conf

    def inference(self, net: nn.Module, data_loader: DataLoader):
        pred_list, conf_list, label_list = [], [], []
        for data, label in tqdm(data_loader):
            data = data.cuda()
            label = label.cuda()
            pred, conf = self.postprocess(net, data)
            for idx in range(len(data)):
                pred_list.append(pred[idx].cpu().tolist())
                conf_list.append(conf[idx].cpu().tolist())
                label_list.append(label[idx].cpu().tolist())

        # convert values into numpy array
        pred_list = np.array(pred_list, dtype=int)
        conf_list = np.array(conf_list)
        label_list = np.array(label_list, dtype=int)

        return pred_list, conf_list, label_list
