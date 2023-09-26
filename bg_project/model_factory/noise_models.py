import torch
import torch.nn as nn
import numpy as np


class GaussianNoise(nn.Module):
    def __init__(self, sigma: float = 0.1):
        super(GaussianNoise, self).__init__()
        self.sigma = sigma

    def forward(self, x, noise_scale: float = 1):
        if noise_scale != 0:
            sampled_noise = torch.randn(x.shape, device=x.device) * self.sigma
            return x + noise_scale * sampled_noise
        else:
            return x


class GaussianSignalDependentNoise(nn.Module):
    def __init__(self, sigma: float = 0.01):
        super(GaussianSignalDependentNoise, self).__init__()
        self.sigma = sigma
        self.noise = torch.tensor(0)

    def forward(self, x, noise_scale: float = 1):
        if noise_scale != 0:
            sampled_noise = self.noise.repeat(*x.size()).normal_() * self.sigma
            return (
                x
                + noise_scale
                * torch.sqrt(torch.abs(x + np.finfo(float).eps))
                * sampled_noise
            )
        else:
            return x
