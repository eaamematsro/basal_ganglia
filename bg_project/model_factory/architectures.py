import abc
import pdb
import os
import re
import torch
import pickle
import json
import torch.nn as nn
from datetime import date
from pathlib import Path
from typing import Callable, Optional, Dict, List, Tuple
from .networks import MLP, MultiHeadMLP, RNN, ThalamicRNN


class BaseArchitecture(nn.Module, metaclass=abc.ABCMeta):
    def __init__(
        self,
    ):
        super(BaseArchitecture, self).__init__()
        self.network = None
        self.save_path = None
        self.text_path = None
        self.params = {}
        self.output_names = None
        self.set_save_path()
        self.set_outputs()

    @abc.abstractmethod
    def set_outputs(self):
        """Define keys of the network output"""

    @abc.abstractmethod
    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Define network forward pass"""

    @abc.abstractmethod
    def description(
        self,
    ):
        """"""

    def set_save_path(self):
        """"""
        cwd = os.getcwd()
        cwd_path = Path(cwd)
        model_path = cwd_path / "data/models"
        model_path.mkdir(exist_ok=True)

        date_str = date.today().strftime("%Y-%m-%d")
        date_save_path = model_path / date_str
        date_save_path.mkdir(exist_ok=True)
        self.save_path = date_save_path

        reg_exp = "_".join(["model", "\d+"])

        files = [
            x
            for x in date_save_path.iterdir()
            if x.is_dir() and re.search(reg_exp, str(x.stem))
        ]
        folder_path = date_save_path / f"model_{len(files)}"
        folder_path.mkdir(exist_ok=True)
        pdb.set_trace()
        self.save_path = folder_path / "model.pickle"
        self.text_path = folder_path / "params.json"

    def save_model(self, **kwargs):
        data_dict = {"network": self, **kwargs}
        with open(self.save_path, "wb") as handle:
            pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(self.text_path, "w") as f:
            json.dump(self.params, f)


class VanillaRNN(BaseArchitecture):
    """Vanilla RNN class with no other areas"""

    def __init__(
        self,
        nneurons: int = 100,
        non_linearity: Optional[nn.Module] = None,
        g0: float = 1.2,
        input_sources: Optional[Dict[str, Tuple[int, bool]]] = None,
        dt: float = 0.05,
        tau: float = 0.15,
        **kwargs,
    ):
        super(VanillaRNN, self).__init__()
        self.params = {
            "n_hidden": nneurons,
            "inputs": input_sources,
            "network": type(self).__name__,
        }
        self.rnn = RNN(
            nneurons=nneurons,
            non_linearity=non_linearity,
            g0=g0,
            input_sources=input_sources,
            dt=dt,
            tau=tau,
        )

    def set_outputs(self):
        self.output_names = ["r_hidden", "r_act"]

    def forward(self, rnn_inputs: Optional[Dict[str, torch.Tensor]] = None, **kwargs):
        r_hidden, r_act = self.rnn.forward(rnn_inputs)
        return {"r_hidden": r_hidden, "r_act": r_act}

    def description(
        self,
    ):
        """"""
        print("A basic RNN with inputs")


class RNNMultiContextInput(BaseArchitecture):
    def __init__(
        self,
        nneurons: int = 100,
        nbg: int = 20,
        non_linearity: Optional[nn.Module] = None,
        g0: float = 1.2,
        input_sources: Optional[Dict[str, Tuple[int, bool]]] = None,
        dt: float = 0.05,
        tau: float = 0.15,
        bg_layer_sizes: Optional[Tuple[int, ...]] = None,
        bg_nfn: Optional[nn.Module] = None,
        bg_input_size: Optional[int] = 1,
        include_bias: bool = True,
        **kwargs,
    ):
        super(RNNMultiContextInput, self).__init__()
        self.params = {
            "n_hidden": nneurons,
            "nbg": nbg,
            "inputs": input_sources,
            "bg_layers": bg_layer_sizes,
            "network": type(self).__name__,
        }
        if input_sources is None:
            input_sources = {}

        input_sources.update({"contextual": (nbg, True)})
        self.rnn = RNN(
            nneurons=nneurons,
            non_linearity=non_linearity,
            g0=g0,
            input_sources=input_sources,
            dt=dt,
            tau=tau,
        )
        self.bg = MLP(
            layer_sizes=bg_layer_sizes,
            non_linearity=bg_nfn,
            input_size=bg_input_size,
            output_size=nbg,
            include_bias=include_bias,
        )

    def set_outputs(self):
        self.output_names = ["r_hidden", "r_act", "bg_act"]

    def forward(
        self,
        bg_inputs: Dict[str, torch.Tensor],
        rnn_inputs: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs,
    ):

        bg_input = next(iter(bg_inputs.values()))
        bg_act = self.bg.forward(bg_input)
        rnn_inputs.update({"contextual": bg_act})
        r_hidden, r_act = self.rnn.forward(inputs=rnn_inputs, **kwargs)
        return {"r_hidden": r_hidden, "r_act": r_act, "bg_act": bg_act}

    def description(
        self,
    ):
        """"""
        print(
            "A RNN designed for multitasking that receives contextual inputs"
            " via input vectors."
        )


class RNNStaticBG(BaseArchitecture):
    def __init__(
        self,
        nneurons: int = 100,
        nbg: int = 20,
        non_linearity: Optional[nn.Module] = None,
        g0: float = 1.2,
        input_sources: Optional[Dict[str, Tuple[int, bool]]] = None,
        dt: float = 0.05,
        tau: float = 0.15,
        bg_layer_sizes: Optional[Tuple[int, ...]] = None,
        bg_nfn: Optional[nn.Module] = None,
        bg_input_size: Optional[int] = 1,
        include_bias: bool = True,
        **kwargs,
    ):
        super(RNNStaticBG, self).__init__()
        self.params = {
            "n_hidden": nneurons,
            "nbg": nbg,
            "inputs": input_sources,
            "bg_layers": bg_layer_sizes,
            "network": type(self).__name__,
        }
        self.rnn = ThalamicRNN(
            nneurons=nneurons,
            nbg=nbg,
            non_linearity=non_linearity,
            g0=g0,
            input_sources=input_sources,
            dt=dt,
            tau=tau,
        )
        self.bg = MLP(
            layer_sizes=bg_layer_sizes,
            non_linearity=bg_nfn,
            input_size=bg_input_size,
            output_size=nbg,
            include_bias=include_bias,
        )

    def set_outputs(self):
        self.output_names = ["r_hidden", "r_act", "bg_act"]

    def forward(
        self,
        bg_inputs: Dict[str, torch.Tensor],
        rnn_inputs: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs,
    ):

        bg_input = next(iter(bg_inputs.values()))
        bg_act = self.bg(bg_input)
        r_hidden, r_act = self.rnn(bg_act, inputs=rnn_inputs, **kwargs)
        return {"r_hidden": r_hidden, "r_act": r_act, "bg_act": bg_act}

    def description(
        self,
    ):
        """"""
        print("An RNN who's weights are multiplied by a static gain from the BG")


class RNNFeedbackBG(BaseArchitecture):
    def __init__(
        self,
        nneurons: int = 100,
        nbg: int = 20,
        non_linearity: Optional[nn.Module] = None,
        g0: float = 1.2,
        input_sources: Optional[Dict[str, Tuple[int, bool]]] = None,
        dt: float = 0.05,
        tau: float = 0.15,
        bg_ind_layer_sizes: Optional[Tuple[int, ...]] = None,
        shared_layer_sizes: Optional[Tuple[int, ...]] = None,
        bg_nfn: Optional[nn.Module] = None,
        bg_input_size: Optional[int] = 10,
        context_rank: int = 1,
        include_bias: bool = True,
        **kwargs,
    ):
        super(RNNFeedbackBG, self).__init__()
        self.params = {
            "n_hidden": nneurons,
            "nbg": nbg,
            "inputs": input_sources,
            "bg_ind_layers": bg_ind_layer_sizes,
            "bg_shared_layers": shared_layer_sizes,
            "network": type(self).__name__,
        }
        self.rnn = ThalamicRNN(
            nneurons=nneurons,
            nbg=nbg,
            non_linearity=non_linearity,
            g0=g0,
            input_sources=input_sources,
            dt=dt,
            tau=tau,
        )
        if bg_ind_layer_sizes is None:
            bg_ind_layer_sizes = ((25, 12), context_rank)
        else:
            bg_ind_layer_sizes = (bg_ind_layer_sizes, context_rank)

        bg_inputs = {"context": bg_ind_layer_sizes, "recurrent": ((50, 25), nneurons)}
        self.bg = MultiHeadMLP(
            independent_layers=bg_inputs,
            shared_layer_sizes=shared_layer_sizes,
            non_linearity=bg_nfn,
            input_size=bg_input_size,
            output_size=nbg,
            include_bias=include_bias,
        )

    def set_outputs(self):
        self.output_names = ["r_hidden", "r_act", "bg_act"]

    def forward(
        self,
        bg_inputs: Dict[str, torch.Tensor],
        rnn_inputs: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs,
    ):

        bg_inputs["recurrent"] = self.rnn.r
        bg_act = self.bg(bg_inputs)
        r_hidden, r_act = self.rnn(bg_act, inputs=rnn_inputs, **kwargs)
        return {"r_hidden": r_hidden, "r_act": r_act, "bg_act": bg_act}

    def description(
        self,
    ):
        """"""
        print(
            "An RNN who's weights are dynamically multiplied by the outputs of a BG module"
            "that receives inputs from the RNN itself."
        )


NETWORKS = {
    "VanillaRNN": VanillaRNN,
    "RNNStaticBG": RNNStaticBG,
    "RNNFeedbackBG": RNNFeedbackBG,
    "RNNMultiContextInput": RNNMultiContextInput,
}
