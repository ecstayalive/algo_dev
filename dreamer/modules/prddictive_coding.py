import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class PredictiveCodingNet(nn.Module):
    def __init__(
        self, features: int, n_layer: int, activation: str = "tanh", dtype=None, device=None
    ):
        super().__init__()
        factory_kwargs = {"dtype": dtype, "device": device}
        self.weight = nn.Parameter(torch.zeros(n_layer, features, features, **factory_kwargs))
        self.states = nn.Parameter(torch.zeros(n_layer, features, 1, **factory_kwargs))
        nn.init.xavier_uniform_(self.weight)
        # nn.init.xavier_uniform_(self.states)

    def forward(self, input: torch.Tensor):
        """
        Performs one full step of predictive coding:
        1. Fast inference dynamics to update mu.
        2. Slow learning dynamics to update W.
        """
        ...
