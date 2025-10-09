import math

import torch

LOG2PIE = math.log(2 * math.pi * math.e)


def gaussian_entropy_diag(logvar: torch.Tensor):
    D = logvar.shape[-1]
    return 0.5 * (D * LOG2PIE + logvar.sum(-1))


def calculate_information_gain_proxy(means: torch.Tensor, logvars: torch.Tensor):
    vars_i = logvars.exp()
    mu_mix = means.mean(0)
    second_moment = (vars_i + means.pow(2)).mean(0)
    var_mix = (second_moment - mu_mix.pow(2)).clamp_min(1e-8)
    H_mix = gaussian_entropy_diag(var_mix.log())
    H_avg = gaussian_entropy_diag(logvars).mean(0)
    return (H_mix - H_avg).clamp_min(0.0)
