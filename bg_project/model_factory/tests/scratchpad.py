import torch
import numpy as np
from model_factory.networks import ThalamicRNN
from model_factory.architectures import RNNGMM
from model_factory.factory_utils import torchify
import pdb

batch_size, bg_size, bg_inputsize = 50, 5, 10
network = RNNGMM(nbg=bg_size, bg_input_size=bg_inputsize)
thalamic_input = torchify(np.random.randn(batch_size, bg_inputsize))
inputs = {"bg_inputs": thalamic_input}
network.rnn.reset_state(batch_size)
network(inputs)
pdb.set_trace()
