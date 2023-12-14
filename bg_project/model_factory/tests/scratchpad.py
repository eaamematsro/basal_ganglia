import torch
import numpy as np
from model_factory.networks import ThalamicRNN
from model_factory.architectures import RNNMultiContextInput
from model_factory.factory_utils import torchify
import pdb

batch_size, bg_size, nclasses, bg_inputsize, nneurons = 50, 5, 10, 10, 150
network = RNNMultiContextInput(
    nneurons=nneurons, nbg=bg_size, bg_input_size=bg_inputsize, n_classes=nclasses
)
thalamic_input = torchify(np.random.randn(batch_size, bg_inputsize))
inputs = {"bg_inputs": thalamic_input}
network.rnn.reset_state(batch_size)
network(inputs)
