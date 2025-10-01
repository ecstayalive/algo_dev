from typing import Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from lightning.nn import LinearBlock


class SpaceAttention(nn.Module):
    """
    A self-contained spatial attention module.
    It takes a controlling hidden state and generates attention parameters,
    computes the filter banks, and performs the read/write operation.
    """

    def __init__(
        self,
        state_size: int,
        obs_shape: Tuple[int, int, int],
        patch_size: int,
        num_patch: int,
        hidden: Iterable[int] = (128,),
    ):
        """
        Args:
            h_dim (int): The dimension of the controlling hidden state.
            obs_shape (Tuple[int, int, int]):
            patch_size (int):
            num_patch (int):
            hidden (Iterable[]): The side length of the attention patch.
        """
        super().__init__()
        self.channel, self.height, self.width = obs_shape
        self.patch_size = patch_size
        self.num_patch = num_patch
        self.fc_params = LinearBlock(state_size, num_patch * 5, hidden, activation="elu")

    def compute_params(self, state: torch.Tensor):
        B = state.size(0)
        # 修正: 明确网络输出log_sigma_sq
        params = self.fc_params(state).view(B, self.num_patch, 5)
        gx_raw, gy_raw, log_sigma_sq, log_delta, log_gamma = params.unbind(dim=-1)

        gx = torch.sigmoid(gx_raw) * (self.width - 1)
        gy = torch.sigmoid(gy_raw) * (self.height - 1)
        sigma_sq = torch.exp(log_sigma_sq).clamp(min=1e-4)  # 方差必须为正
        base_delta = (max(self.height, self.width) - 1) / (self.patch_size - 1)
        delta = base_delta * torch.exp(log_delta)
        gamma = torch.exp(log_gamma)

        return (
            gx.unsqueeze(-1),
            gy.unsqueeze(-1),
            delta.unsqueeze(-1),
            sigma_sq.unsqueeze(-1),
            gamma.unsqueeze(-1),
        )

    def gaussian_filter(
        self, gx: torch.Tensor, gy: torch.Tensor, delta: torch.Tensor, sigma_sq: torch.Tensor
    ):
        """Computes Gaussian filter matrices Fx and Fy for all glimpses."""
        B, G, _ = gx.shape
        device = gx.device

        i = torch.arange(self.patch_size, device=device, dtype=torch.float32).view(
            1, 1, self.patch_size
        )
        j = torch.arange(self.patch_size, device=device, dtype=torch.float32).view(
            1, 1, self.patch_size
        )

        mu_x = gx + (i - self.patch_size / 2 - 0.5) * delta
        mu_y = gy + (j - self.patch_size / 2 - 0.5) * delta

        a = torch.arange(self.width, device=device, dtype=torch.float32).view(1, 1, self.width)
        b = torch.arange(self.height, device=device, dtype=torch.float32).view(1, 1, self.height)

        Fx = torch.exp(-torch.pow(a - mu_x.unsqueeze(3), 2) / (2 * sigma_sq.unsqueeze(3)))
        Fy = torch.exp(-torch.pow(b - mu_y.unsqueeze(3), 2) / (2 * sigma_sq.unsqueeze(3)))

        Fx = Fx / (Fx.sum(dim=3, keepdim=True) + 1e-8)
        Fy = Fy / (Fy.sum(dim=3, keepdim=True) + 1e-8)
        return Fx, Fy

    def read(
        self,
        observation: torch.Tensor,
        gx: torch.Tensor,
        gy: torch.Tensor,
        delta: torch.Tensor,
        sigma_sq: torch.Tensor,
        gamma: torch.Tensor,
    ):
        """Performs the parallel read operation for all glimpses."""
        B, G, _ = gx.shape
        Fx, Fy = self.gaussian_filter(gx, gy, delta, sigma_sq)  # Fx: (B,G,P,W), Fy: (B,G,P,H)
        # 扩展 observation 以匹配 glimpse 维度
        # observation: (B, C, H, W) -> obs_expanded: (B, G, C, H, W)
        obs_expanded = observation.unsqueeze(1).expand(-1, G, -1, -1, -1)
        # --- 核心修正：使用 matmul 进行广播矩阵乘法 ---
        # Fy: (B, G, P, H) -> unsqueeze -> (B, G, 1, P, H)
        # obs_expanded: (B, G, C, H, W)
        # Fx.transpose: (B, G, W, P) -> unsqueeze -> (B, G, 1, W, P)
        # 广播规则:
        # 1. (B,G,1,P,H) @ (B,G,C,H,W) -> intermediate(B,G,C,P,W)
        # 2. intermediate(B,G,C,P,W) @ (B,G,1,W,P) -> patch(B,G,C,P,P)
        # Fy @ observation
        intermediate = torch.matmul(Fy.unsqueeze(2), obs_expanded)
        # intermediate @ Fx.T
        patch = torch.matmul(intermediate, Fx.transpose(-1, -2).unsqueeze(2))
        patch = patch.squeeze(2)  # Shape: (B, G, C, P, P)

        return patch * gamma.view(B, G, 1, 1, 1)

    def write(self, patches, gx, gy, delta, sigma_sq, gamma):
        B, G, C, N, _ = patches.shape
        Fx, Fy = self.gaussian_filter(gx, gy, delta, sigma_sq)
        # Fx 形状: (B, G, N, W)
        # Fy 形状: (B, G, N, H)

        # 目标操作: canvas_update = Fy.T @ patch @ Fx

        # --- 步骤 1: 计算 patches @ Fx ---
        # patches:         (B, G, C, N, N)
        # Fx (增加通道维): (B, G, 1, N, W)
        # 结果:            (B, G, C, N, W)
        intermediate = torch.matmul(patches, Fx.unsqueeze(2))

        Fy_t = Fy.transpose(-1, -2)

        canvas_updates = torch.matmul(Fy_t.unsqueeze(2), intermediate)

        # 应用逆 gamma 缩放
        # gamma.view(B, G, 1, 1, 1) 将正确广播到 (B, G, C, H, W)
        return canvas_updates / gamma.view(B, G, 1, 1, 1)


class AttentiveEncoder(nn.Module):
    def __init__(
        self,
        observation_shape: Tuple[int, int, int],
        state_size: int,
        patch_size: int,
        num_patch: int,
        embedding_size: int,
        attention_hidden: Iterable[int] = (128,),
    ):
        super().__init__()
        C, _, _ = observation_shape
        self.num_patch = num_patch
        self.channel = C
        self.patch_size = patch_size
        self.space_attention = SpaceAttention(
            state_size, observation_shape, patch_size, num_patch, attention_hidden
        )
        self.patch_encoder1 = nn.Sequential(
            nn.Conv2d(C, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy_input = torch.randn(1, C, patch_size, patch_size)
            dummy_features: torch.Tensor = self.patch_encoder1(dummy_input)
        self.linear = nn.Linear(num_patch * dummy_features.shape[-1], embedding_size)

    def forward(self, observation: torch.Tensor, state: torch.Tensor):
        B = observation.size(0)
        gx, gy, delta, sigma_sq, gamma = self.space_attention.compute_params(state)
        # Perform read without gamma for ground truth
        unscaled_patches = self.space_attention.read(
            observation, gx, gy, delta, sigma_sq, torch.ones_like(gamma)
        )
        # Apply gamma for the input to the encoder network
        scaled_patches = unscaled_patches
        return self.encode_patches(scaled_patches), unscaled_patches

    def encode_patches(self, patches: torch.Tensor):
        """_summary_

        Args:
            patches (torch.Tensor): [B, N, C, P, P]

        Returns:
            _type_: _description_
        """
        B, N, C, P, P = patches.shape
        patch_features = self.patch_encoder1(patches.view(B * N, C, P, P))
        embedding = self.linear(patch_features.view(B, -1))
        return embedding


class AttentiveDecoder(nn.Module):

    def __init__(self, state_size, observation_shape, patch_size, num_patch):
        super().__init__()
        C, self.height, self.width = observation_shape
        self.num_patch = num_patch
        self.channel = C
        self.patch_size = patch_size
        self.space_attention = SpaceAttention(state_size, observation_shape, patch_size, num_patch)
        encoder_cnn_for_shape_calc = nn.Sequential(
            nn.Conv2d(C, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        with torch.no_grad():
            dummy_input = torch.zeros(1, C, patch_size, patch_size)
            intermediate_shape = encoder_cnn_for_shape_calc(dummy_input).shape

        self.start_channels, self.start_H, self.start_W = intermediate_shape[1:]
        flattened_dim = self.start_channels * self.start_H * self.start_W

        self.linear = nn.Sequential(nn.Linear(state_size, num_patch * flattened_dim), nn.ReLU())
        self.patch_generator = nn.Sequential(
            nn.ConvTranspose2d(self.start_channels, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, C, kernel_size=3, stride=2, output_padding=1),
        )

    def generate_patches(self, state: torch.Tensor):
        B = state.size(0)
        features = self.linear(state)
        features_reshaped = features.view(
            B * self.num_patch, self.start_channels, self.start_H, self.start_W
        )
        generated_patches = self.patch_generator(features_reshaped)
        return generated_patches.view(
            B, self.num_patch, self.channel, self.patch_size, self.patch_size
        )

    def forward(
        self, posterior: torch.Tensor, deterministic: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        state = torch.cat([posterior, deterministic], dim=-1)

        patches = self.generate_patches(state)

        params = self.space_attention.compute_params(state)
        gx, gy, delta, sigma_sq, gamma = params

        canvas_updates = self.space_attention.write(patches, gx, gy, delta, sigma_sq, gamma)
        canvas = canvas_updates.sum(dim=1)
        return canvas, patches


class PatchAttention(nn.Module):
    def __init__(self, state_size: int, num_patches: int, k: int = 4, hidden: list[int] = [128]):
        """
        Args:
            state_size (int): (stochastic_size + deterministic_size)
            num_patches (int):
            k (int):
        """
        super().__init__()
        self.patch_sizeum_patches = num_patches
        self.k = k
        self.mlp = LinearBlock(state_size, num_patches, hidden, activation="relu")

    def forward(self, state: torch.Tensor, use_gumbel_softmax: bool = False):
        """
        Args:
            state (torch.Tensor): 输入状态，形状为 [B, state_size]
            use_gumbel_softmax (bool): 是否使用 Gumbel-Softmax 进行可微分的离散采样
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
            - logits (torch.Tensor): 每个补丁的重要性分数，形状为 [B, num_patches]
            - indices (torch.Tensor): 被选中的 top-K 补丁的索引，形状为 [B, num_glimpses]
        """
        logits = self.mlp(state)
        if self.training and use_gumbel_softmax:
            pass

        _, indices = torch.topk(logits, self.k, dim=1)
        indices = torch.sort(indices, dim=1).values

        return logits, indices


class PatchEncoder(nn.Module):
    """
    A CNN specifically designed to extract small image patches
    from attention mechanisms. It takes a small patch as input and
    outputs an embedding vector of fixed dimension.
    Args:
        input_shape: Glimpse image shape
        output_size:
    """

    def __init__(self, input_shape: Tuple[int], output_size: int):
        super().__init__()
        self._input_shape = input_shape
        self._output_size = output_size

        self.cnn = nn.Sequential(
            nn.Conv2d(self._input_shape[0], 16, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy_input = torch.zeros(1, *self._input_shape)
            flattened_size = self.cnn(dummy_input).shape[1]
        self.mlp = Linear(flattened_size, output_size, bias=False)

    def forward(self, glimpse):
        """
        Args:
            glimpse shape: [B, C, H, W]
        """
        x = self.cnn(glimpse)
        embedding = self.mlp(x)
        return embedding


class PatchDecoder(nn.Module):
    """
    Decoder 的基本单元。
    接收一个状态/嵌入向量，并将其解码为一个单一的图像补丁。
    这个模块是 PatchEncoder 的大致逆过程
    """

    def __init__(self, state_size: int, patch_output_shape: Tuple[int, int, int]):
        """
        Args:
            state_size (int):
            patch_output_shape (Tuple[int, int, int]): (C, patch_size, patch_size)
        """
        super().__init__()
        self.C, self.P, _ = patch_output_shape
        self.state_size = state_size

        self.flattened_size = 64 * 1 * 1
        self.in_shape = (64, 1, 1)
        self.mlp = Linear(self.state_size, self.flattened_size, bias=False)
        self.de_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1),  # -> [B, 32, 3, 3]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2),  # -> [B, 16, 7, 7]
            nn.ReLU(),
            nn.ConvTranspose2d(
                16, self.C, kernel_size=3, stride=2, output_padding=1
            ),  # -> [B, C, 16, 16]
        )

    def forward(self, state: torch.Tensor):
        """
        Args:
            state (torch.Tensor):
        Returns:
            torch.Tensor:
        """
        x = self.mlp(state)
        x = x.unflatten(-1, self.in_shape)
        generated_patch = self.de_conv(x)
        return generated_patch


class VisionPatchBasedEnc(nn.Module):
    """ """

    def __init__(
        self, observation_shape: Tuple[int, int, int], embedding_size: int, patch_size: int = 16
    ):
        super().__init__()
        C, H, W = observation_shape
        assert (
            H % patch_size == 0 and W % patch_size == 0
        ), "The image dimensions must be divisible by patch_size."
        self.observation_shape = observation_shape
        self.patch_size = patch_size
        self.embedding_size = embedding_size
        self.patch_sizeum_patches = (H // patch_size) * (W // patch_size)
        self.patch_shape = (observation_shape[0], self.patch_size, self.patch_size)
        self.patch_encoder = PatchEncoder(self.patch_shape, embedding_size)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.patch_sizeum_patches, embedding_size))
        # self.layer_norm = nn.LayerNorm(embedding_size)

    def forward(self, obs: torch.Tensor):
        """
        Args:
            obs: [B, C, H, W]
        """
        B, C, H, W = obs.shape
        p = self.patch_size
        # 'b c (h p1) (w p2) -> (b h w) c p1 p2'
        # B, C, (H/p), p, (W/p), p -> (B * num_patches), C, p, p
        patches = rearrange(obs, "b c (h p1) (w p2) -> (b h w) c p1 p2", p1=p, p2=p)

        # `patches`  [B * num_patches, C, patch_size, patch_size]
        patch_embeddings = self.patch_encoder(
            patches
        )  # Shape: [B * num_patches, glimpse_output_size]

        # [B * num_patches, glimpse_output_size] -> [B, num_patches, embedding_size]
        patch_embeddings = (
            patch_embeddings.view(B, self.patch_sizeum_patches, -1) + self.pos_embedding
        )
        global_embedding = torch.mean(patch_embeddings, dim=1)  # Shape: [B, glimpse_output_size]
        return global_embedding


class VisionPatchBasedDec(nn.Module):

    def __init__(
        self,
        state_size: int,
        num_patches: int,
        patch_output_shape: Tuple[int, int, int],
    ):
        """
        Args:
            state_size (int): 世界模型状态的维度。
            num_patches (int): 总的补丁数量。
            patch_output_shape (Tuple[int, int, int]): 单个补丁的输出形状。
        """
        super().__init__()

        # 为了让 Decoder 知道它正在生成哪个位置的补丁，
        # 我们给每个补丁的 ID 创建一个可学习的位置嵌入 (Positional Embedding)。
        # 这与 ViT 中的位置嵌入思想完全相同。
        self.state_size = state_size
        self.pos_embedding = nn.Embedding(num_patches, state_size)

        self.patch_decoder = PatchDecoder(
            state_size=state_size, patch_output_shape=patch_output_shape
        )

    def forward(self, state: torch.Tensor, indices: torch.Tensor | None = None):
        """
        Args:
            state (torch.Tensor): [B, state_size]
            indices (torch.Tensor): [B, k]

        Returns:
            torch.Tensor: [B, k, C, P, P]
        """
        B = state.shape[0]
        if indices is None:
            num_targets = self.pos_embedding.num_embeddings  # num_patches
            # indices: [B, N]
            indices = torch.arange(num_targets, device=state.device).unsqueeze(0).expand(B, -1)
        k = indices.shape[1]
        state_expanded = state.unsqueeze(1).expand(-1, k, -1)

        # [B, num_glimpses] -> [B, num_glimpses, state_size]
        pos_embeds = self.pos_embedding(indices)

        # 将全局状态和位置嵌入拼接起来，形成 PatchDecoder 的输入
        # 这告诉解码器：“基于当前这个全局情况，请生成这个位置的补丁”
        decoder_input = state_expanded + pos_embeds  # Shape: [B, num_glimpses, final_state_size]
        # Parallel generate
        # 为了使用单个 PatchDecoder 高效处理，我们将 batch 和 num_glimpses 维度合并
        decoder_input_flat = decoder_input.view(B * k, -1)

        generated_patches_flat = self.patch_decoder(decoder_input_flat)  # Shape: [B*k, C, P, P]

        C, P = self.patch_decoder.C, self.patch_decoder.P
        patches = generated_patches_flat.view(B, k, C, P, P)

        return patches


class HybridAttentiveEncoder(nn.Module):
    """
    A hybrid encoder integrating spatial attention and self-attention.
    TODO: Experimental Features
    """

    def __init__(self, observation_shape, config):
        super().__init__()
        self.config = config.parameters.dreamer
        self.attention_config = self.config.attention
        self.patch_sizeum_glimpses = self.attention_config.num_glimpses
        state_dim = self.config.deterministic_size

        self.attention_net = SpaceAttention(state_dim, config)

        glimpse_input_shape = (
            observation_shape[0],
            self.attention_config.patch_size,
            self.attention_config.patch_size,
        )
        embed_dim = self.config.embedding_size  # e.g., 1024

        self.glimpse_encoder = PatchEncoder(glimpse_input_shape, embed_dim, config)

        # --- MODIFICATION START: Integrating Transformer ---
        # 定义一个标准的 Transformer Encoder Layer
        transformer_layer = TransformerEncoderLayer(
            d_model=embed_dim,  # 嵌入维度
            nhead=self.attention_config.transformer_heads,  # 多头注意力的头数 (e.g., 8)
            dim_feedforward=self.attention_config.transformer_mlp_hidden,  # 前馈网络隐藏层维度
            dropout=self.attention_config.transformer_dropout,
            batch_first=True,  # <-- VERY IMPORTANT! Ensures input is [B, SeqLen, Dim]
        )

        # 将多个 layer 堆叠成一个完整的 Transformer Encoder
        self.transformer_encoder = TransformerEncoder(
            transformer_layer, num_layers=self.attention_config.transformer_layers  # e.g., 2-3
        )
        # --- MODIFICATION END ---

    def forward(self, obs, deterministic):
        B = obs.size(0)

        # STAGE 1: SPATIAL ATTENTION (The "Spotlight")
        # -----------------------------------------------
        # 1. 决定 N 个瞥见的位置
        theta, _ = self.attention_net(deterministic)

        # 2. 提取 N 个瞥见
        obs_expanded = obs.repeat_interleave(self.patch_sizeum_glimpses, dim=0)
        grid = F.affine_grid(
            theta,
            (
                B * self.patch_sizeum_glimpses,
                obs.size(1),
                self.attention_config.patch_size,
                self.attention_config.patch_size,
            ),
            align_corners=False,
        )
        glimpses = F.grid_sample(obs_expanded, grid, align_corners=False)

        # 3. 独立编码每个瞥见，得到一个向量集合
        embedded_glimpses = self.glimpse_encoder(glimpses)
        # Reshape to [B, N, embed_dim] - a perfect set for the Transformer
        embedded_glimpses = embedded_glimpses.view(B, self.patch_sizeum_glimpses, -1)

        # STAGE 2: SELF-ATTENTION (The "Cocktail Party")
        # ------------------------------------------------
        # 4. 让瞥见之间互相"交流"，理解彼此的关系
        contextualized_glimpses = self.transformer_encoder(embedded_glimpses)

        # 5. 聚合所有经过上下文增强的瞥见信息
        aggregated_embedding = torch.sum(contextualized_glimpses, dim=1)

        return aggregated_embedding
        return aggregated_embedding
