from typing import Any

import torch
import torch.nn as nn

from .base import BasePostprocessor


class EnergyBased(BasePostprocessor):
    def __init__(self):
        super().__init__()
        self.temperature = 1.0
        print(f"Using Temperature: {self.temperature}")

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        output = net(data)
        score = torch.softmax(output, dim=1)
        _, pred = torch.max(score, dim=1)
        conf = self.temperature * torch.logsumexp(output / self.temperature,
                                                  dim=1)
        return pred, conf
