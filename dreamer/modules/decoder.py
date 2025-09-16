from typing import Optional, Tuple

import torch
import torch.nn as nn

from dreamer.utils.utils import (
    create_normal_dist,
    horizontal_forward,
    initialize_weights,
)
from lightning.nn import LinearBlock


class Decoder(nn.Module):
    def __init__(self, observation_shape, config, o_embed_size: Optional[int] = None):
        super().__init__()
        self.config = config.parameters.dreamer.decoder
        self.stochastic_size = config.parameters.dreamer.stochastic_size
        self.deterministic_size = config.parameters.dreamer.deterministic_size

        activation = getattr(nn, self.config.activation)()
        self.observation_shape = observation_shape

        self.network = nn.Sequential(
            nn.Linear(
                (
                    (self.stochastic_size + self.deterministic_size)
                    if o_embed_size is None
                    else o_embed_size
                ),
                self.config.depth * 32,
            ),
            nn.Unflatten(1, (self.config.depth * 32, 1)),
            nn.Unflatten(2, (1, 1)),
            nn.ConvTranspose2d(
                self.config.depth * 32,
                self.config.depth * 4,
                self.config.kernel_size,
                self.config.stride,
            ),
            activation,
            nn.ConvTranspose2d(
                self.config.depth * 4,
                self.config.depth * 2,
                self.config.kernel_size,
                self.config.stride,
            ),
            activation,
            nn.ConvTranspose2d(
                self.config.depth * 2,
                self.config.depth * 1,
                self.config.kernel_size + 1,
                self.config.stride,
            ),
            activation,
            nn.ConvTranspose2d(
                self.config.depth * 1,
                self.observation_shape[0],
                self.config.kernel_size + 1,
                self.config.stride,
            ),
            # nn.Softsign(),
        )
        self.network.apply(initialize_weights)

    def forward(
        self,
        posterior: torch.Tensor,
        deterministic: Optional[torch.Tensor] = None,
        return_dist: bool = True,
    ):
        if deterministic is not None:
            x = horizontal_forward(
                self.network, posterior, deterministic, output_shape=self.observation_shape
            )
        else:
            x = horizontal_forward(
                self.network,
                posterior,
                input_shape=(posterior.shape[-1],),
                output_shape=self.observation_shape,
            )
        return (
            x
            if not return_dist
            else create_normal_dist(x, std=1, event_shape=len(self.observation_shape))
        )


class HierarchicalDecoder(nn.Module):
    def __init__(
        self,
        observation_shape: Tuple[int],
        config,
        hidden_shape: int = [128],
    ):
        super().__init__()
        self.config = config.parameters.dreamer.decoder
        self.stochastic_size = config.parameters.dreamer.stochastic_size
        self.deterministic_size = config.parameters.dreamer.deterministic_size
        self.o_embed_size = config.parameters.dreamer.embedded_state_size

        activation = getattr(nn, self.config.activation)()
        self.observation_shape = observation_shape
        self.dec1 = LinearBlock(
            self.stochastic_size + self.deterministic_size,
            self.o_embed_size,
            hidden_shape,
            "relu",
            activate_output=True,
        )
        self.dec2 = nn.Sequential(
            nn.Unflatten(1, (self.config.depth * 32, 1)),
            nn.Unflatten(2, (1, 1)),
            nn.ConvTranspose2d(
                self.config.depth * 32,
                self.config.depth * 4,
                self.config.kernel_size,
                self.config.stride,
            ),
            activation,
            nn.ConvTranspose2d(
                self.config.depth * 4,
                self.config.depth * 2,
                self.config.kernel_size,
                self.config.stride,
            ),
            activation,
            nn.ConvTranspose2d(
                self.config.depth * 2,
                self.config.depth * 1,
                self.config.kernel_size + 1,
                self.config.stride,
            ),
            activation,
            nn.ConvTranspose2d(
                self.config.depth * 1,
                self.observation_shape[0],
                self.config.kernel_size + 1,
                self.config.stride,
            ),
        )
        self.dec2.apply(initialize_weights)

    def forward(self, posterior: torch.Tensor, deterministic: torch.Tensor):
        o_embed: torch.Tensor = self.dec1(torch.cat([posterior, deterministic], dim=-1))
        image = horizontal_forward(
            self.dec2,
            o_embed,
            input_shape=(o_embed.shape[-1],),
            output_shape=self.observation_shape,
        )
        return o_embed, image

    def decode_o_embed(self, posterior: torch.Tensor, deterministic: torch.Tensor):
        return self.dec1(torch.cat([posterior, deterministic], dim=-1))

    def decode_image(self, o_embed: torch.Tensor):
        return horizontal_forward(self.dec2, o_embed, output_shape=self.observation_shape)
