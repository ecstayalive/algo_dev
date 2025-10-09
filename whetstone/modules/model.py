import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from thunder.nn import LinearBlock
from whetstone.utils.utils import build_network, create_normal_dist, horizontal_forward


class RSSM(nn.Module):
    def __init__(self, action_size, config):
        super().__init__()
        self.config = config.parameters.dreamer.rssm

        self.recurrent_model = RecurrentModel(action_size, config)
        self.transition_model = TransitionModel(config)
        self.transition_ensemble_model = TransitionEnsembleModel(5, config)
        self.representation_model = RepresentationModel(config)

    def recurrent_model_input_init(self, batch_size):
        return self.transition_model.input_init(batch_size), self.recurrent_model.input_init(
            batch_size
        )


class InformationGainModel(nn.Module):
    def __init__(self, state_size: int, min_gain: float = 0.1):
        """_summary_

        Args:
            state_size (int): _description_
            min_gain (float, optional): _description_. Defaults to 0.1.
        """
        super().__init__()
        self.model = LinearBlock(state_size, 2, [128, 128], "relu")
        self.min_gain = min_gain

    def forward(self, x: torch.Tensor):
        log_params = self.model(x)
        return create_normal_dist(log_params, min_std=self.min_gain)


class RecurrentModel(nn.Module):
    def __init__(self, action_size, config):
        super().__init__()
        self.config = config.parameters.dreamer.rssm.recurrent_model
        self.device = config.operation.device
        self.stochastic_size = config.parameters.dreamer.stochastic_size
        self.deterministic_size = config.parameters.dreamer.deterministic_size

        self.activation = getattr(nn, self.config.activation)()

        self.linear = nn.Linear(self.stochastic_size + action_size, self.config.hidden_size)
        self.recurrent = nn.GRUCell(self.config.hidden_size, self.deterministic_size)

    def forward(self, embedded_state, action, deterministic):
        x = torch.cat((embedded_state, action), 1)
        x = self.activation(self.linear(x))
        x = self.recurrent(x, deterministic)
        return x

    def input_init(self, batch_size):
        return torch.zeros(batch_size, self.deterministic_size).to(self.device)


class TransitionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config.parameters.dreamer.rssm.transition_model
        self.device = config.operation.device
        self.stochastic_size = config.parameters.dreamer.stochastic_size
        self.deterministic_size = config.parameters.dreamer.deterministic_size
        self.network = build_network(
            self.deterministic_size,
            self.config.hidden_size,
            self.config.num_layers,
            self.config.activation,
            self.stochastic_size * 2,
        )

    def forward(self, x):
        x = self.network(x)
        prior_dist = create_normal_dist(x, min_std=self.config.min_std)
        prior = prior_dist.rsample()
        return prior_dist, prior

    def input_init(self, batch_size):
        return torch.zeros(batch_size, self.stochastic_size).to(self.device)


class TransitionEnsembleModel(nn.Module):
    def __init__(self, k: int, config):
        super().__init__()
        self.k = k
        self.config = config.parameters.dreamer.rssm.transition_model
        self.device = config.operation.device
        self.stochastic_size = config.parameters.dreamer.stochastic_size
        self.deterministic_size = config.parameters.dreamer.deterministic_size
        self.network = build_network(
            self.deterministic_size,
            self.config.hidden_size,
            self.config.num_layers,
            self.config.activation,
            self.k * self.stochastic_size * 2,
        )

    def forward(self, h: torch.Tensor) -> Normal:
        """
        Args:
            h (torch.Tensor): shape: (B, H_dim)

        Returns:
            torch.distributions.Normal: batch_shape: (K, B), event_shape: (S_dim)
        """
        B = h.shape[0]
        network_out = self.network(h)
        # (B, K, S_dim * 2)
        reshaped_out = network_out.view(B, self.k, self.stochastic_size * 2)
        # (K, B, S_dim * 2)
        transposed_out = reshaped_out.permute(1, 0, 2)

        return create_normal_dist(transposed_out, min_std=self.config.min_std)


class RepresentationModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config.parameters.dreamer.rssm.representation_model
        self.embedded_state_size = config.parameters.dreamer.embedded_state_size
        self.stochastic_size = config.parameters.dreamer.stochastic_size
        self.deterministic_size = config.parameters.dreamer.deterministic_size
        self.network = build_network(
            self.embedded_state_size + self.deterministic_size,
            self.config.hidden_size,
            self.config.num_layers,
            self.config.activation,
            self.stochastic_size * 2,
        )

    def forward(self, embedded_observation, deterministic):
        x = self.network(torch.cat((embedded_observation, deterministic), 1))
        posterior_dist = create_normal_dist(x, min_std=self.config.min_std)
        posterior = posterior_dist.rsample()
        return posterior_dist, posterior


class RewardModel(nn.Module):
    def __init__(self, config, num_rewards: int = 1):
        super().__init__()
        self.config = config.parameters.dreamer.reward
        self.stochastic_size = config.parameters.dreamer.stochastic_size
        self.deterministic_size = config.parameters.dreamer.deterministic_size
        self.in_features = self.stochastic_size + self.deterministic_size
        self.network = build_network(
            self.stochastic_size + self.deterministic_size,
            self.config.hidden_size,
            self.config.num_layers,
            self.config.activation,
            num_rewards,
        )

    def forward(self, posterior, deterministic):
        x = horizontal_forward(self.network, posterior, deterministic, output_shape=(1,))
        return create_normal_dist(x, std=1, event_shape=1)


class CustomRewardModel(nn.Module):
    def __init__(self, config, num_rewards: int = 1):
        super().__init__()
        self.config = config.parameters.dreamer.reward
        self.stochastic_size = config.parameters.dreamer.stochastic_size
        self.deterministic_size = config.parameters.dreamer.deterministic_size
        self.o_embed_size = config.parameters.dreamer.embedded_state_size
        self.num_rewards = num_rewards
        self.network = build_network(
            self.o_embed_size,
            self.config.hidden_size,
            self.config.num_layers,
            self.config.activation,
            num_rewards,
        )

    def forward(self, features: torch.Tensor):
        x = horizontal_forward(
            self.network,
            features,
            input_shape=(features.shape[-1],),
            output_shape=(self.num_rewards,),
        )
        return create_normal_dist(x, std=1, event_shape=1)


class ContinueModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config.parameters.dreamer.continue_
        self.stochastic_size = config.parameters.dreamer.stochastic_size
        self.deterministic_size = config.parameters.dreamer.deterministic_size

        self.network = build_network(
            self.stochastic_size + self.deterministic_size,
            self.config.hidden_size,
            self.config.num_layers,
            self.config.activation,
            1,
        )

    def forward(self, posterior, deterministic):
        x = horizontal_forward(self.network, posterior, deterministic, output_shape=(1,))
        return torch.distributions.Bernoulli(logits=x)
        return torch.distributions.Bernoulli(logits=x)
