from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba

from dreamer.utils.utils import build_network


@dataclass(slots=True)
class WorldModelInfos:
    mu_p: torch.Tensor  # [B,L,S], prior mean
    std_p: torch.Tensor  # [B,L,S], prior std
    mu_q: torch.Tensor  # [B,L,S], posterior mean
    std_q: torch.Tensor  # [B,L,S], posterior std
    z_prior: torch.Tensor  # [B,L,S]
    z_post: torch.Tensor  # [B,L,S]
    H1: torch.Tensor  # [B,L,D]
    # kl_loss: torch.Tensor  # [B,L]


class WorldModel(nn.Module):
    """
    p(s_{t+1}|s_t,a_t)
    q(s_t|s_{t-1},a_t,o_t)
    """

    def __init__(self, action_size, config):
        super().__init__()
        self.config = config.parameters.dreamer
        self.device = config.operation.device
        state_size = config.parameters.dreamer.stochastic_size
        hidden_size = config.parameters.dreamer.deterministic_size
        enc_size = config.parameters.dreamer.embedded_state_size
        self.Wa = nn.Sequential(nn.Linear(action_size, hidden_size), nn.ELU())
        self.Ws = nn.Sequential(nn.Linear(state_size, hidden_size), nn.ELU())
        self.We = nn.Sequential(nn.Linear(enc_size, hidden_size), nn.ELU())
        # self.z0_projector = nn.Sequential(nn.Linear(state_size, state_size))
        self.mamba = Mamba(hidden_size, 4)
        self.prior_head = build_network(hidden_size, 200, 2, "ELU", 2 * state_size)
        self.post_head = build_network(hidden_size + enc_size, 200, 2, "ELU", 2 * state_size)
        self.norm_in = nn.LayerNorm(hidden_size)
        self.norm_h = nn.LayerNorm(hidden_size)

    def forward(self, actions: torch.Tensor, s0: torch.Tensor, enc_o: torch.Tensor):
        B, L, _ = actions.shape
        Wa_act = self.Wa(actions)
        # --- Prepare the input sequence for Mamba ---
        # 1. Embed the initial state s0 to kickstart the sequence.
        #    This serves as the information from timestep t=-1.
        s0_embed = self.Ws(s0).unsqueeze(1)  # [B, 1, D]
        # 2. To predict state t, we use observation t-1.
        #    We shift the encoded observations and prepend the initial state embedding.
        #    enc_o_prev shape: [B, L, E] -> [B, 1, D] + [B, L-1, D] = [B, L, D]
        enc_o_embed = self.We(enc_o)
        enc_o_prev = torch.cat([s0_embed, enc_o_embed[:, :-1]], dim=1)

        # 3. The final input combines action and previous observation info.
        mamba_input = Wa_act + enc_o_prev

        # --- Run Mamba and Heads in Parallel ---
        # 4. Process the entire sequence in one go.
        H1 = self.mamba(mamba_input)
        H1 = self.norm_h(H1)
        # 5. Compute prior and posterior distributions for the whole sequence at once.
        prior_param = self.prior_head(H1)
        post_param = self.post_head(torch.cat([H1, enc_o], -1))
        mu_p, log_std_p = prior_param.chunk(2, -1)
        std_p = F.softplus(log_std_p)
        z_prior = mu_p + torch.randn_like(mu_p) * std_p
        mu_q, log_std_q = post_param.chunk(2, -1)
        std_q = F.softplus(log_std_q)
        z_post = mu_q + torch.randn_like(mu_q) * std_q
        return WorldModelInfos(
            mu_p=mu_p, std_p=std_p, mu_q=mu_q, std_q=std_q, z_prior=z_prior, z_post=z_post, H1=H1
        )

    @torch.no_grad()
    def imagine(self, actions: torch.Tensor, s0: torch.Tensor, window=None):
        if actions.dim() == 2:
            actions = actions.unsqueeze(1)
        B, L, _ = actions.shape
        Wa_act = self.Wa(actions)

        # Initialize state for imagination
        s_prev = s0
        # Mamba's internal state (h) can be passed explicitly, but mamba.py manages it.
        # For simplicity, we just feed sequences of length 1.
        mus, stds = [], []
        h_final = None

        for t in range(L):
            # 1. Mamba input combines previous stochastic state and current action
            u_t = self.Ws(s_prev) + Wa_act[:, t, :]

            # 2. Process a single timestep. Input shape: [B, 1, D]
            h_t = self.mamba(u_t.unsqueeze(1))
            h_t = self.norm_h(h_t.squeeze(1))  # Shape: [B, D]

            # 3. Predict prior distribution from the deterministic state
            prior_param = self.prior_head(h_t)
            mu_t, log_std_t = prior_param.chunk(2, -1)
            std_t = F.softplus(log_std_t)

            # 4. Sample the next state from the prior to feed into the next step
            s_prev = mu_t + torch.randn_like(mu_t) * std_t

            # 5. Store results
            mus.append(mu_t)
            stds.append(std_t)
            h_final = h_t  # Keep track of the last hidden state

        mu = torch.stack(mus, 1)
        std = torch.stack(stds, 1)
        h_final = (
            h_final.unsqueeze(1)
            if h_final is not None
            else torch.zeros(B, 1, self.config.deterministic_size, device=self.device)
        )

        return mu, std, h_final

    def state_dist0(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.zeros(batch_size, self.config.stochastic_size, device=self.device), torch.ones(
            batch_size, self.config.stochastic_size, device=self.device
        )

    def state0(self, batch_size: int) -> torch.Tensor:
        return torch.zeros(batch_size, self.config.stochastic_size, device=self.device)

    def latent0(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.zeros(
            batch_size, self.config.stochastic_size, device=self.device
        ), torch.zeros(batch_size, self.config.deterministic_size, device=self.device)
