import abc
import pdb
import os
import re
import torch
import pickle
import numpy as np
import torch.nn as nn
from factory_utils import torchify
from datetime import date
from pathlib import Path
from typing import Callable, Optional, Dict, List, Tuple
from networks import (MLP, MultiHeadMLP, RNN, ThalamicRNN)


class BaseArchitecture(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, ):
        super(BaseArchitecture, self).__init__()
        self.network = None
        self.save_path = None
        self.text_path = None
        self.set_save_path()

    @abc.abstractmethod
    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Define network forward pass"""

    @abc.abstractmethod
    def description(self,):
        """"""

    def set_save_path(self):
        """"""
        cwd = os.getcwd()
        cwd_path = Path(cwd)
        model_path = cwd_path / 'data/models'
        model_path.mkdir(exist_ok=True)

        date_str = date.today().strftime("%Y-%m-%d")
        date_save_path = model_path / date_str
        date_save_path.mkdir(exist_ok=True)
        self.save_path = date_save_path

        reg_exp = '_'.join(['model', '\d+'])
        files = [x for x in date_save_path.iterdir() if x.is_dir() and re.search(reg_exp, str(x.stem))]
        folder_path = date_save_path / f"model_{len(files)}"
        folder_path.mkdir(exist_ok=True)
        self.save_path = folder_path / 'model.pickle'
        self.text_path = folder_path / 'params.json'

    def save_model(self):
        data_dict = {'network': self.network, 'full_model': self}
        with open(self.save_path, 'wb') as handle:
            pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


class VanillaRNN(BaseArchitecture):
    """Vanilla RNN class with no other areas"""
    def __init__(self, nneurons: int = 100, non_linearity: Optional[nn.Module] = None,
                 g0: float = 1.2, input_sources: Optional[Dict[str, Tuple[int, bool]]] = None,
                 device: Optional[torch.device] = None, dt: float = 5e-2, tau: float = .15,
                 **kwargs):
        super(VanillaRNN, self).__init__()

        self.rnn = RNN(nneurons=nneurons, non_linearity=non_linearity,
                      g0=g0, input_sources=input_sources, device=device,
                      dt=dt, tau=tau)

    def forward(self, rnn_inputs: Optional[Dict[str, torch.Tensor]] = None,
                **kwargs):
        r_hidden, r_act = self.rnn(rnn_inputs)
        return {'r_hidden': r_hidden, 'r_act': r_act}

    def description(self,):
        """"""
        print(
            "A basic RNN with inputs"
        )


class RNNStaticBG(BaseArchitecture):
    def __init__(self, nneurons: int = 100, nbg: int = 20, non_linearity: Optional[nn.Module] = None,
                 g0: float = 1.2, input_sources: Optional[Dict[str, Tuple[int, bool]]] = None,
                 dt: float = .01, tau: float = .15, device: Optional[torch.device] = None,
                 bg_layer_sizes: Optional[Tuple[int, ...]] = None, bg_nfn:  Optional[nn.Module] = None,
                 bg_input_size: Optional[int] = 1, **kwargs):
        super(RNNStaticBG, self).__init__()

        self.rnn = ThalamicRNN(nneurons=nneurons, nbg=nbg, non_linearity=non_linearity, g0=g0,
                               input_sources=input_sources, dt=dt, tau=tau, device=device)
        self.bg = MLP(layer_sizes=bg_layer_sizes, non_linearity=bg_nfn, input_size=bg_input_size,
                      output_size=nbg)

    def forward(self, bg_inputs: Dict[str, torch.Tensor],
                rnn_inputs: Optional[Dict[str, torch.Tensor]] = None, **kwargs):

        bg_input = next(iter(bg_inputs.values()))
        bg_act = self.bg(bg_input).T
        r_hidden, r_act = self.rnn(bg_act, inputs=rnn_inputs)
        return {'r_hidden': r_hidden, 'r_act': r_act, 'bg_act': bg_act}

    def description(self,):
        """"""
        print(
            "An RNN who's weights are multiplied by a static gain from the BG"
        )


class RNNFeedbackBG(BaseArchitecture):
    def __init__(self, nneurons: int = 100, nbg: int = 20, non_linearity: Optional[nn.Module] = None,
                 g0: float = 1.2, input_sources: Optional[Dict[str, Tuple[int, bool]]] = None,
                 dt: float = .01, tau: float = .15, device: Optional[torch.device] = None,
                 bg_ind_layer_sizes: Optional[Tuple[int, ...]] = None, shared_layer_sizes: Optional[Tuple[int, ...]]
                 = None, bg_nfn:  Optional[nn.Module] = None, bg_input_size: Optional[int] = 10, context_rank: int = 1,
                 **kwargs):
        super(RNNFeedbackBG, self).__init__()

        self.rnn = ThalamicRNN(nneurons=nneurons, nbg=nbg, non_linearity=non_linearity, g0=g0,
                               input_sources=input_sources, dt=dt, tau=tau, device=device)
        if bg_ind_layer_sizes is None:
            bg_ind_layer_sizes = ((25, 12), context_rank)
        else:
            bg_ind_layer_sizes = (bg_ind_layer_sizes, context_rank)

        bg_inputs = {
            'context': bg_ind_layer_sizes,
            'recurrent': ((50, 25), nneurons)
        }
        self.bg = MultiHeadMLP(independent_layers=bg_inputs, shared_layer_sizes=shared_layer_sizes,
                               non_linearity=bg_nfn, input_size=bg_input_size, output_size=nbg
        )

    def forward(self, bg_inputs: Dict[str, torch.Tensor],
                rnn_inputs: Optional[Dict[str, torch.Tensor]] = None, **kwargs):

        bg_inputs['recurrent'] = self.rnn.r.T
        bg_act = self.bg(bg_inputs).T
        r_hidden, r_act = self.rnn(bg_act, inputs=rnn_inputs)
        return {'r_hidden': r_hidden, 'r_act': r_act, 'bg_act': bg_act}

    def description(self,):
        """"""
        print(
            "An RNN who's weights are dynamically multiplied by the outputs of a BG module"
            "that receives inputs from the RNN itself."
        )


NETWORKS = {
            "VanillaRNN": VanillaRNN,
            "RNNStaticBG": RNNStaticBG,
            "RNNFeedbackBG": RNNFeedbackBG,
            }