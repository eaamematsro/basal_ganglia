import torch
import numpy as np
from model_factory.networks import ThalamicRNN
from model_factory.factory_utils import torchify
import pdb

batch_size, bg_size = 50, 5
network = ThalamicRNN(nbg=bg_size)
network.reset_state(batch_size=batch_size)
thalamic_input = torchify(np.random.randn(batch_size, bg_size))
network(thalamic_input)
pdb.set_trace()
