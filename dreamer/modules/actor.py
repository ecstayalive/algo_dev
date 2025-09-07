import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import TanhTransform

from dreamer.utils.utils import build_network, create_normal_dist


class HabitActor(nn.Module):
    def __init__(self, discrete_action_bool, action_size, config):
        super().__init__()
        self.config = config.parameters.dreamer.agent.actor
        self.discrete_action_bool = discrete_action_bool
        self.stochastic_size = config.parameters.dreamer.stochastic_size
        self.deterministic_size = config.parameters.dreamer.deterministic_size

        action_size = action_size if discrete_action_bool else 2 * action_size

        self.network = build_network(
            self.stochastic_size + self.deterministic_size,
            self.config.hidden_size,
            self.config.num_layers,
            self.config.activation,
            action_size,
        )

    def forward(
        self,
        posterior: torch.Tensor,
        deterministic: torch.Tensor,
        return_dist=False,
        squashing=True,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.distributions.Distribution]:
        x = torch.cat((posterior, deterministic), -1)
        x = self.network(x)
        if self.discrete_action_bool:
            dist = torch.distributions.OneHotCategorical(logits=x)
            action = dist.sample() + dist.probs - dist.probs.detach()
        else:
            dist = create_normal_dist(
                x,
                mean_scale=self.config.mean_scale,
                init_std=self.config.init_std,
                min_std=self.config.min_std,
                activation=torch.tanh,
            )
            action = torch.distributions.Independent(dist, 1).rsample()
        if squashing:
            action = action.tanh()
        return (action, dist) if return_dist else action

    def log_prob(
        self,
        mean: torch.Tensor,
        stddev: torch.Tensor,
        action: torch.Tensor,
        squashing: bool = False,
    ):
        """
        Args:
            action (torch.Tensor): Unsquashing action

        Returns:
            torch.Tensor: The log probability of the action.
        """
        dist = create_normal_dist(mean, stddev)
        return (
            dist.log_prob(action) - 2.0 * (math.log(2.0) - action - F.softplus(-2.0 * action))
            if squashing
            else dist.log_prob(action)
        )


class ThinkingActor(nn.Module):
    def __init__(self, discrete_action_bool, action_size, config):
        super().__init__()
        self.config = config.parameters.dreamer.agent.actor
        self.discrete_action_bool = discrete_action_bool
        self.stochastic_size = config.parameters.dreamer.stochastic_size
        self.deterministic_size = config.parameters.dreamer.deterministic_size

        action_size = action_size if discrete_action_bool else 2 * action_size

        self.network = build_network(
            self.stochastic_size * 2 + self.deterministic_size,
            self.config.hidden_size,
            self.config.num_layers,
            self.config.activation,
            action_size,
        )

    def forward(
        self,
        posterior: torch.Tensor,
        deterministic: torch.Tensor,
        des_posterior: torch.Tensor,
        return_dist=False,
        squashing=True,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.distributions.Distribution]:
        x = torch.cat((posterior, deterministic, des_posterior), -1)
        x = self.network(x)
        if self.discrete_action_bool:
            dist = torch.distributions.OneHotCategorical(logits=x)
            action = dist.sample() + dist.probs - dist.probs.detach()
        else:
            dist = create_normal_dist(
                x,
                mean_scale=self.config.mean_scale,
                init_std=self.config.init_std,
                min_std=self.config.min_std,
                activation=torch.tanh,
            )
            action = torch.distributions.Independent(dist, 1).rsample()
        if squashing:
            action = action.tanh()
        return (action, dist) if return_dist else action

    def log_prob(self, mean: torch.Tensor, stddev: torch.Tensor, action: torch.Tensor):
        """
        Args:
            action (torch.Tensor): Unsquashing action

        Returns:
            torch.Tensor: The log probability of the action.
        """
        dist = create_normal_dist(mean, stddev)
        return 2.0 * (math.log(2.0) - action - F.softplus(-2.0 * action)) * dist.log_prob(action)


class Gating(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.stochastic_size = config.parameters.dreamer.stochastic_size
        self.deterministic_size = config.parameters.dreamer.deterministic_size
        self.network = nn.Sequential(
            nn.Linear(self.stochastic_size + self.deterministic_size, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, state: torch.Tensor, deterministic: torch.Tensor) -> torch.Tensor:
        x = torch.cat((state, deterministic), -1)
        return self.network(x)


def squashing(action: torch.Tensor):
    return torch.tanh(action)
