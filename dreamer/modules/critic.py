import torch
import torch.nn as nn

from dreamer.utils.utils import build_network, create_normal_dist, horizontal_forward


class CriticV(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config.parameters.dreamer.agent.critic
        self.stochastic_size = config.parameters.dreamer.stochastic_size
        self.deterministic_size = config.parameters.dreamer.deterministic_size

        self.network = build_network(
            self.stochastic_size + self.deterministic_size,
            self.config.hidden_size,
            self.config.num_layers,
            self.config.activation,
            1,
        )

    def forward(self, posterior: torch.Tensor, deterministic: torch.Tensor):
        x = horizontal_forward(self.network, posterior, deterministic, output_shape=(1,))
        return create_normal_dist(x, std=1, event_shape=1)


class CriticQ(nn.Module):
    def __init__(self, action_size: int, config: dict):
        super().__init__()
        self.action_size = action_size
        self.config = config.parameters.dreamer.agent.critic
        self.stochastic_size = config.parameters.dreamer.stochastic_size
        self.deterministic_size = config.parameters.dreamer.deterministic_size
        self.network = build_network(
            self.deterministic_size + self.stochastic_size + action_size,
            self.config.hidden_size,
            self.config.num_layers,
            self.config.activation,
            1,
        )

    def forward(self, prior: torch.Tensor, deterministic: torch.Tensor, action: torch.Tensor):
        x = torch.cat([prior, deterministic, action], dim=-1)
        batch_with_horizon_shape = x.shape[:-1]
        x = self.network(x.flatten(0, -2))
        x = x.reshape(*batch_with_horizon_shape, 1)
        return create_normal_dist(x, std=1, event_shape=1)
