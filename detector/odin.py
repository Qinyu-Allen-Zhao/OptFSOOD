"""Adapted from: https://github.com/facebookresearch/odin."""
from typing import Any

import torch
import torch.nn as nn

from .base import BasePostprocessor


class ODIN(BasePostprocessor):
    def __init__(self, temperature=1000, noise=0.004):
        super().__init__()
        self.temperature = temperature
        self.noise = noise
        print(f"Using Temperature: {self.temperature}, Noise: {self.noise}")

    def postprocess(self, net: nn.Module, data: Any):
        data.requires_grad = True
        output = net(data)

        # Calculating the perturbation we need to add, that is,
        # the sign of gradient of cross entropy loss w.r.t. input
        criterion = nn.CrossEntropyLoss()

        labels = output.detach().argmax(axis=1)

        # Using temperature scaling
        output = output / self.temperature

        loss = criterion(output, labels)
        loss.backward()

        # Normalizing the gradient to binary in {0, 1}
        gradient = torch.ge(data.grad.detach(), 0)
        gradient = (gradient.float() - 0.5) * 2

        # Scaling values taken from original code
        gradient[:, 0] = (gradient[:, 0]) / (63.0 / 255.0)
        gradient[:, 1] = (gradient[:, 1]) / (62.1 / 255.0)
        gradient[:, 2] = (gradient[:, 2]) / (66.7 / 255.0)

        # Adding small perturbations to images
        tempInputs = torch.add(data.detach(), gradient, alpha=-self.noise)
        output = net(tempInputs)
        output = output / self.temperature

        # Calculating the confidence after adding perturbations
        nnOutput = output.detach()
        nnOutput = nnOutput - nnOutput.max(dim=1, keepdims=True).values
        nnOutput = nnOutput.exp() / nnOutput.exp().sum(dim=1, keepdims=True)

        conf, pred = nnOutput.max(dim=1)

        return pred, conf
