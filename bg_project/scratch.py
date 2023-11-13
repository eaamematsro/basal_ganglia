import torch
import numpy as np
from model_factory.factory_utils import torchify
from model_factory.architectures import HRLNetwork

observation_size, batch_size, latent_dim = 5, 50, 10

model = HRLNetwork(observation_size=observation_size, latent_dim=latent_dim)
observations = torchify(np.random.randn(batch_size, observation_size))
model(observations)
