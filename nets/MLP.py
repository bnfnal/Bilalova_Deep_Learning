from torch import nn
import numpy as np


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        layers = []
        input_size = np.prod(cfg['input'])
        for layer, layer_size in cfg['up_stack'].items():
            layers.append(nn.Linear(input_size, layer_size))
            layers.append(nn.ReLU)
            input_size = layer_size
        layers.append(nn.Linear(input_size, cfg['output']))
        self.model = nn.Sequential(*layers)

    def forward(self, xb):
        return self.model(xb)