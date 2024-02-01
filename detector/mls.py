from typing import Any

import torch
import torch.nn as nn

from detector.base import BasePostprocessor


class MaxLogit(BasePostprocessor):
    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        output = net(data)
        conf, pred = torch.max(output, dim=1)
        return pred, conf
