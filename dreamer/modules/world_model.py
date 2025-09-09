from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba2

from dreamer.utils.utils import build_network


@dataclass(slots=True)
class WorldModelInfos:
    mu_p: torch.Tensor  # [B,L,S], prior mean
    std_p: torch.Tensor  # [B,L,S], prior std
    mu_q: torch.Tensor  # [B,L,S], posterior mean
    std_q: torch.Tensor  # [B,L,S], posterior std
    prior: torch.Tensor  # [B,L,S]
    post: torch.Tensor  # [B,L,S]
    hidden: torch.Tensor  # [B,L,D]


class WorldModel(nn.Module):
    """
    p(s_t|s_{t-1},a_{t-1})
    q(s_t|s_{t-1},a_{t-1},o_t)
    """

    def __init__(self, action_size, config):
        super().__init__()
        self.config = config.parameters.dreamer
        self.device = config.operation.device
        state_size = config.parameters.dreamer.stochastic_size
        hidden_size = config.parameters.dreamer.deterministic_size
        enc_size = config.parameters.dreamer.embedded_state_size
        self.Wa = nn.Sequential(nn.Linear(action_size, hidden_size), nn.ELU())
        # self.Ws = nn.Sequential(nn.Linear(state_size, hidden_size), nn.ELU())
        self.mamba = Mamba2(hidden_size, 4, bias=True)
        self.prior_head = build_network(hidden_size, 128, 2, "ELU", 2 * state_size)
        self.post_head = build_network(hidden_size + enc_size, 128, 2, "ELU", 2 * state_size)
        # self.multi_prior_head = build_network(hidden_size, 128, 2, "ELU", 2 * 5 * state_size)

    def forward(self, actions: torch.Tensor, s0: torch.Tensor, o_embed: torch.Tensor):
        """
        Args:
            actions: [B,L,A]
            s0: [B,S]
            enc_o: [B,L,E]
        """
        B, L, _ = actions.shape
        a_embed = self.Wa(actions)
        # s0_embed = self.Ws(s0).unsqueeze(1)  # [B, 1, D]
        # with torch.no_grad():
        #     h = self.mamba(a_embed.detach() + s0_embed.detach())
        #     mu_q, log_std_q = self.post_head(torch.cat([h, o_embed], dim=-1)).chunk(2, -1)
        #     posterior1 = mu_q + torch.randn_like(mu_q) * F.softplus(log_std_q)
        # s_embed = self.Ws(posterior1.detach())
        # s_embed = torch.cat((s0_embed, s_embed[:, :-1]), 1)  # [B, L, D]
        h = self.mamba(a_embed)
        mu_p, log_std_p = self.prior_head(h).chunk(2, -1)
        mu_q, log_std_q = self.post_head(torch.cat([h, o_embed], -1)).chunk(2, -1)
        std_p = F.softplus(log_std_p)
        std_q = F.softplus(log_std_q)
        z_prior = mu_p + torch.randn_like(mu_p) * std_p
        z_post = mu_q + torch.randn_like(mu_q) * std_q
        return WorldModelInfos(
            mu_p=mu_p, std_p=std_p, mu_q=mu_q, std_q=std_q, prior=z_prior, post=z_post, hidden=h
        )

    def step(self, action: torch.Tensor, s: torch.Tensor, o_embed: torch.Tensor):
        """
        Args:
            action: [B,A]
            s: [B,S]
            o_embed: [B,E]
        """
        a_embed = self.Wa(action)
        # s_embed = self.Ws(s)
        # u = a_embed + s_embed
        u = a_embed
        h = self.mamba(u.unsqueeze(1)).squeeze(1)
        mu_q, log_std_q = self.post_head(torch.cat([h, o_embed], -1)).chunk(2, -1)
        posterior = mu_q + torch.randn_like(mu_q) * F.softplus(log_std_q)
        return posterior, h

    def imagine(self, action: torch.Tensor, s: torch.Tensor):
        """
        Args:
            action: [B, A]
            s: [B, S]
        """
        a_embed = self.Wa(action)
        # s_embed = self.Ws(s)
        # u = a_embed + s_embed
        u = a_embed
        h = self.mamba(u.unsqueeze(1)).squeeze(1)
        mu_p, log_std_p = self.prior_head(h).chunk(2, -1)
        std_p = F.softplus(log_std_p)
        prior = mu_p + torch.randn_like(mu_p) * std_p
        return prior, mu_p, std_p, h

    def state0_dist(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.zeros(batch_size, self.config.stochastic_size, device=self.device), torch.ones(
            batch_size, self.config.stochastic_size, device=self.device
        )

    def deterministic0(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.zeros(
            batch_size, self.config.stochastic_size, device=self.device
        ), torch.zeros(batch_size, self.config.deterministic_size, device=self.device)
