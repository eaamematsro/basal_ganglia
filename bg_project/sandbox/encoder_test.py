import torch
import pdb
import numpy as np
from model_factory.factory_utils import torchify
from model_factory.networks import EncoderNetwork


input_size, batch_size = 50, 20
inputs = torchify(np.random.randn(batch_size, input_size))
model = EncoderNetwork(input_dim=50)
model(inputs)
pdb.set_trace()
