import numpy as np
import torch

from architectures import RNNStaticBG, RNNFeedbackBG
from factory_utils import torchify

batches = 50
input_size = 5
bg_rank = 10
context_rank = 6

context = torch.randn((context_rank, batches))

bg_input = {
    'context': context
}

# bg_input = torch.randn((input_size, batches))
input_sources = {
    'trigger': (10, True),
}
rnn_input = {
    'trigger': torch.randn(10, batches)
}
test_model = RNNFeedbackBG(bg_input_size=input_size, rnn_input_sources=input_sources, context_rank=context_rank)
test_model.rnn.reset_state(batches)
out = test_model(bg_input, rnn_input)