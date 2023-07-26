import abc

import torch
import numpy as np
import torch.nn as nn
from factory_utils import torchify
from typing import Callable, Optional, Dict, List, Tuple
from networks import (MLP, MultiHeadMLP, RNN)


class BaseArchitecture(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self):
        self.network = None

    @abc.abstractmethod
    def forward(self, **kwargs):
        """Define network forward pass"""


class VanillaRNN(BaseArchitecture):
    """Vanilla RNN class with no other areas"""
    def __init__(self, nneurons: int = 100, non_linearity: Optional[nn.Module] = None,
                 g0: float = 1.2, input_sources: Optional[Dict[str, Tuple[int, bool]]] = None,
                 device: Optional[torch.device] = None, dt: float = 5e-2, tau: float = .15):
        super(VanillaRNN, self).__init__()

        self.rnn = RNN(nneurons=nneurons, non_linearity=non_linearity,
                      g0=g0, input_sources=input_sources, device=device,
                      dt=dt, tau=tau)

    def forward(self):
        return self.rnn()


class RNNStaticBG(BaseArchitecture):
    def __init__(self):
        super(RNNStaticBG, self).__init__()

        self.rnn = RNN(

        )

        self.bg = MLP(

        )