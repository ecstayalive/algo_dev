import torch

from dreamer.modules.mamba import InferenceCache, Mamba2, Mamba2Config

config = Mamba2Config(128, n_layer=1, chunk_size=50)
mamba = Mamba2(config, torch.device("cuda"))

x = torch.randn(1, 50, 128).cuda()
y, h = mamba(x)
y: torch.Tensor
h: InferenceCache
print(h.ssm_state.shape)
