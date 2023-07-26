import numpy as np
import torch
from networks import MultiHeadMLP, ThalamicRNN
from factory_utils import torchify

batches = 5
input_size = 5
bg_rank = 10

input_sources = {
    'trigger': (10, True),
}

inputs = {'trigger': torchify(np.random.randn(10, batches)),
            'go': torchify(np.random.randn(10, batches)),
          }
bg = torchify(np.random.randn(bg_rank, batches))

test_model = ThalamicRNN(
    nbg=bg_rank,
    input_sources=input_sources
)
test_model.reset_state(batch_size=batches)
test_model(bg, inputs, validate_inputs=True)