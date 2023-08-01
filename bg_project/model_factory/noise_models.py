import torch
import torch.nn as nn
import numpy as np


class GaussainNoise(nn.Module):
    def __init__(self, sigma: float = .1):
        super(GaussainNoise, self).__init__()
        self.sigma = sigma
        self.noise = torch.tensor(0, dtype=torch.float32)

    def forward(self, x, noise_scale: float = 1):
        if noise_scale != 0:
            sampled_noise = self.noise.repeat(*x.size()).normal_() * self.sigma
            return x + noise_scale * sampled_noise
        else:
            return x


class GaussianSignalDependentNoise(nn.Module):
    def __init__(self, sigma: float = .01):
        super(GaussianSignalDependentNoise, self).__init__()
        self.sigma = sigma
        self.noise = torch.tensor(0)

    def forward(self, x, noise_scale: float = 1):
        if noise_scale != 0:
            sampled_noise = self.noise.repeat(*x.size()).normal_() * self.sigma
            return x + noise_scale * torch.sqrt(torch.abs(x + np.finfo(float).eps)) * sampled_noise
        else:
            return x


