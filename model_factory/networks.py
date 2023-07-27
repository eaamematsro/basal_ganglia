import pdb

import torch
import numpy as np
import torch.nn as nn
from factory_utils import torchify
from typing import Callable, Optional, Dict, List, Tuple


class RNN(nn.Module):
    """ Base class for recurrent neural networks"""

    def __init__(self, nneurons: int = 100, non_linearity: Optional[nn.Module] = None,
                 g0: float = 1.2, input_sources: Optional[Dict[str, Tuple[int, bool]]] = None,
                 device: Optional[torch.device] = None, dt: float = 5e-2, tau: float = .15):
        super(RNN, self).__init__()

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            device = self.device
        else:
            self.device = device

        if non_linearity is None:
            non_linearity = nn.Softplus()

        self.nonlinearity = non_linearity

        self.I = {}

        if input_sources is not None:
            for input_name, (input_size, learnable) in input_sources.items():
                input_mat = np.random.randn(nneurons, ) / np.sqrt(nneurons)
                input_tens = torchify(input_mat, device)
                if learnable:
                    self.I[input_name] = nn.Parameter(input_tens)
                else:
                    self.I[input_name] = input_tens

        J_mat = (g0 * np.random.randn(nneurons, nneurons) / np.sqrt(nneurons))
        J_mat = torchify(J_mat, device=device)
        self.input_names = set(list(self.I.keys()))
        self.J = nn.Parameter(J_mat)
        self.B = nn.Parameter(torchify(np.random.randn(nneurons, 1), device))
        self.x, self.r = None, None
        self.dt = dt
        self.tau = tau

    def forward(self, inputs: Dict[str, torch.Tensor], noise_scale: float = 0,
                validate_inputs: bool = False):
        if validate_inputs:
            input_names = set(list(inputs.keys()))
            assert input_names.intersection(self.input_names) == input_names
        out = 0
        for input_name, input_value in inputs.items():
            out += self.I[input_name] @ input_value

        x = self.x + self.dt / self.tau * (
                -self.x + self.J @ self.r + self.B + out
                + noise_scale * torch.randn(self.x.shape)
        )
        r = self.nonlinearity(x)
        self.x = x
        self.r = r
        return self.x, self.r

    def reset_state(self, batch_size: int = 10):
        self.x = torch.randn((self.J.shape[0], batch_size)) / np.sqrt(self.J.shape[0])
        self.r = self.nonlinearity(self.x)


class ThalamicRNN(nn.Module):
    """ Base class for recurrent neural networks"""

    def __init__(self, nneurons: int = 100, nbg: int = 20, non_linearity: Optional[nn.Module] = None,
                 g0: float = 1.2, input_sources: Optional[Dict[str, Tuple[int, bool]]] = None,
                 device: Optional[torch.device] = None, dt: float = 5e-2, tau: float = .15):
        super(ThalamicRNN, self).__init__()

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            device = self.device
        else:
            self.device = device

        if non_linearity is None:
            non_linearity = nn.Softplus()

        self.nonlinearity = non_linearity

        self.I = {}

        if input_sources is not None:
            for input_name, (input_size, learnable) in input_sources.items():
                input_mat = np.random.randn(nneurons, input_size) / np.sqrt(nneurons)
                input_tens = torchify(input_mat, device)
                if learnable:
                    self.I[input_name] = nn.Parameter(input_tens)
                else:
                    self.I[input_name] = input_tens

        J_mat = (g0 * np.random.randn(nneurons, nneurons) / np.sqrt(nneurons))
        J_mat = torchify(J_mat, device=device)
        self.input_names = set(list(self.I.keys()))
        self.J = nn.Parameter(J_mat)
        self.B = nn.Parameter(torchify(np.random.randn(nneurons, 1), device))
        self.U = torchify(np.random.randn(nneurons, nbg) / np.sqrt(nneurons), device)
        self.V = torchify(np.random.randn(nbg, nneurons) / np.sqrt(nneurons), device)
        self.x, self.r = None, None
        self.dt = dt
        self.tau = tau

    def forward(self, r_thalamic, inputs: Optional[Dict[str, torch.Tensor]] = None, noise_scale: float = 0,
                validate_inputs: bool = False):
        if inputs is None:
            inputs = {}
        if validate_inputs:
            input_names = set(list(inputs.keys()))
            assert input_names.intersection(self.input_names) == input_names
        out = 0
        for input_name, input_value in inputs.items():
            out += self.I[input_name] @ input_value

        v_batch = torch.einsum('jk, ji -> jik', r_thalamic, self.V)
        J = torch.einsum('ij, jlk -> ilk', self.U, v_batch)
        J_rec = J + self.J[:, :, None]
        rec_input = torch.einsum('ijk, jk -> ik', J_rec, self.r)
        x = self.x + self.dt / self.tau * (
                -self.x + rec_input + self.B + out
                + noise_scale * torch.randn(self.x.shape)
        )
        r = self.nonlinearity(x)
        self.x = x
        self.r = r
        return self.x, self.r

    def reset_state(self, batch_size: int = 10):
        self.x = torch.randn((self.J.shape[0], batch_size)) / np.sqrt(self.J.shape[0])
        self.r = self.nonlinearity(self.x)


class MLP(nn.Module):
    def __init__(self, layer_sizes: Optional[Tuple[int, ...]] = None, non_linearity: Optional[nn.Module] = None,
                 input_size: Optional[int] = 150, output_size: Optional[int] = 10):
        super(MLP, self).__init__()
        if layer_sizes is None or not (type(layer_sizes) == tuple):
            layer_sizes = (100, 50,)

        if non_linearity is None:
            non_linearity = nn.Softplus()

        modules = []
        layer_sizes = layer_sizes + (output_size,)
        previous_size = input_size
        for size in layer_sizes:
            modules.append(
                nn.Linear(previous_size, size)
            )
            modules.append(non_linearity)
            previous_size = size

        self.mlp = nn.Sequential(*modules)

    def forward(self, inputs: torch.Tensor):
        """

        :param inputs: [batch, inputs]
        :return:
        """
        y = self.mlp(inputs)
        return y


class MultiHeadMLP(nn.Module):
    def __init__(self, independent_layers: Optional[Dict[str, Tuple[Tuple[int, ...], int]]] = None,
                 shared_layer_sizes: Optional[Tuple[int, ...]] = None,
                 non_linearity: Optional[nn.Module] = None,
                 input_size: Optional[int] = 150, output_size: Optional[int] = 10):
        super(MultiHeadMLP, self).__init__()
        if independent_layers is None:
            independent_layers = {
                'input_1': ((100, 50), 10),
                'input_2': ((100, 50), 10)
            }

        assert len(independent_layers) > 1, 'There are less than one input heads. Consider using MLP instead.'
        self.input_names = set(list(independent_layers.keys()))
        self.input_mlps = {}
        for input_name, (input_sizes, layer_input) in independent_layers.items():
            self.input_mlps[input_name] = MLP(layer_sizes=input_sizes, output_size=input_size,
                                              input_size=layer_input, non_linearity=non_linearity)

        self.shared_mlp = MLP(layer_sizes=shared_layer_sizes, input_size=input_size,
                              output_size=output_size, non_linearity=non_linearity)

    def forward(self, inputs: Dict[str, torch.Tensor], validate_inputs: bool = True):
        """

        :param inputs: Dict of Inputs [batch, inputs]
        :param validate_inputs:
        :return:
        """
        if validate_inputs:
            input_names = set(list(inputs.keys()))
            assert input_names.intersection(self.input_names) == input_names

        combined_input = 0
        for input_name, value in inputs.items():
            combined_input += self.input_mlps[input_name](value)

        y = self.shared_mlp(combined_input)
        return y


