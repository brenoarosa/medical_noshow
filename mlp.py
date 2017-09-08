"""
Multi Layer Perceptron Module
"""

from typing import Iterable
from copy import deepcopy
import torch.nn as nn

class MLP(nn.Module):
    """
    Feed-Forward Neural Network Module
    """

    def __init__(self, layer_sizes: Iterable[int]):
        super(MLP, self).__init__()

        self.layers = []

        for i in range(1, len(layer_sizes)):
            if i < (len(layer_sizes) - 1):
                layer = nn.Sequential(
                    nn.BatchNorm1d(layer_sizes[i-1]),
                    nn.Linear(layer_sizes[i-1], layer_sizes[i]),
                    nn.Relu())
            else: # last layer
                layer = nn.Sequential(
                    nn.BatchNorm1d(layer_sizes[i-1]),
                    nn.Linear(layer_sizes[i-1], layer_sizes[i]),
                    nn.Sigmoid())

            layer_name = "layer_{}".format(i)
            setattr(self, layer_name, layer)
            self.layers.append(layer_name)

        self._init_weigths()
        self.initial_weights = deepcopy(self.state_dict())
        return

    def _init_weigths(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform(module.weight.data, gain=1)
                nn.init.constant(module.bias.data, 0)

    def forward(self, x):
        for layer in self.layers:
            x = getattr(self, layer)(x)
        return x
