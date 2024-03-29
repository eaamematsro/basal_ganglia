import abc
import pdb
import os
import re
from abc import ABC

import numpy as np
import torch
import pickle
import json
import torch.nn as nn
from datetime import date
from pathlib import Path
from typing import Callable, Optional, Dict, List, Tuple
from typing_extensions import override
from torch.distributions.multivariate_normal import MultivariateNormal
from model_factory.factory_utils import torchify
from .networks import (
    MLP,
    MultiHeadMLP,
    RNN,
    ThalamicRNN,
    BGRNN,
    InputRNN,
    EncoderNetwork,
    DecoderNetwork,
    GaussianMixtureModel,
)

# TODO: Implement a network with a set of input vectors


class BaseArchitecture(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, task: Optional[str] = None, **kwargs):
        super(BaseArchitecture, self).__init__()
        self.network = None
        self.save_path = None
        self.text_path = None
        self.folder_path = None
        self.task = task
        self.tags = None
        self.output_names = None
        self.set_save_path()
        self.set_outputs()
        self.learning_styles = {}

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
        if self.task is None:
            model_path = cwd_path / "data/models"
        else:
            model_path = cwd_path / f"data/models/{self.task}"
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
        self.folder_path = folder_path
        self.save_path = folder_path / "model.pickle"
        self.text_path = folder_path / "params.json"

    def save_model(self, **kwargs):
        self.folder_path.mkdir(exist_ok=True)
        data_dict = {"network": self, "tags": self.tags, **kwargs}
        with open(self.save_path, "wb") as handle:
            pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(self.text_path, "w") as f:
            json.dump(self.params, f)

    def swap_grad_state(
        self, grad_state: bool = True, params_to_swap: Optional[list] = None
    ):
        if params_to_swap is None:
            params_to_swap = self.parameters()

        for param_group in params_to_swap:
            if isinstance(param_group, torch.nn.parameter.Parameter):
                param_group.requires_grad = grad_state
            else:
                for param in param_group.parameters():
                    param.requires_grad = grad_state


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
        super(VanillaRNN, self).__init__(**kwargs)
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
        n_classes: int = 20,
        bg_nfn: Optional[nn.Module] = None,
        bg_input_size: int = 1,
        include_bias: bool = True,
        **kwargs,
    ):
        super(RNNMultiContextInput, self).__init__(**kwargs)
        self.params = {
            "n_hidden": nneurons,
            "latent_dim": nbg,
            "inputs": input_sources,
            "bg_layers": bg_layer_sizes,
            "network": type(self).__name__,
        }
        if input_sources is None:
            input_sources = {}

        input_sources.update({"contextual": (nbg, True)})
        self.rnn = InputRNN(
            nneurons=nneurons,
            nbg=nbg,
            non_linearity=non_linearity,
            g0=g0,
            input_sources=input_sources,
            dt=dt,
            tau=tau,
            bg_non_linearity=bg_nfn,
        )
        # self.bg = MLP(
        #     layer_sizes=bg_layer_sizes,
        #     non_linearity=bg_nfn,
        #     input_size=bg_input_size,
        #     output_size=latent_dim,
        #     include_bias=include_bias,
        # )

        self.classifier = MLP(
            input_size=bg_input_size,
            layer_sizes=bg_layer_sizes,
            output_size=n_classes,
            include_bias=include_bias,
            non_linearity=bg_nfn,
            std=1 / n_classes,
            return_nnl=False,
        )

        self.bg = GaussianMixtureModel(
            number_of_clusters=n_classes, latent_dimension=nbg
        )

    def set_outputs(self):
        self.output_names = ["r_hidden", "r_act", "bg_act"]

    def forward(
        self,
        bg_inputs: Dict[str, torch.Tensor],
        rnn_inputs: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs,
    ):

        # bg_input = next(iter(bg_inputs.values()))
        # bg_act = self.bg

        if "cluster_probs" in bg_inputs.keys():
            cluster_probs = bg_inputs["cluster_probs"]
        else:
            classifier_input = next(iter(bg_inputs.values()), None)
            logits = self.classifier(classifier_input)

            cluster_probs = nn.functional.softmax(
                logits,
            )

        bg_act = self.bg(cluster_probs)
        r_hidden, r_act = self.rnn(bg_act, inputs=rnn_inputs, **kwargs)
        return {
            "r_hidden": r_hidden,
            "r_act": r_act,
            "bg_act": bg_act,
            "cluster_probs": cluster_probs,
        }

    def description(
        self,
    ):
        """"""
        print(
            "A RNN designed for multitasking that receives contextual inputs"
            " via input vectors."
        )

    def swap_grad_state(
        self, grad_state: bool = True, params_to_swap: Optional[list] = None
    ):
        if params_to_swap is None:
            params_to_swap = [self.rnn.gained_I, self.bg, self.classifier]
        for param_group in params_to_swap:
            if isinstance(param_group, torch.nn.parameter.Parameter):
                param_group.requires_grad = grad_state
            else:
                for param in param_group.parameters():
                    param.requires_grad = grad_state

    def get_input_stats(
        self,
        classifier_input: torch.Tensor,
        cluster_probs: Optional[torch.Tensor] = None,
    ) -> (torch.Tensor, torch.Tensor):

        with torch.no_grad():
            if cluster_probs is None:
                cluster_logits = self.classifier(classifier_input)
                cluster_probs = nn.functional.softmax(cluster_logits, dim=1)
            cluster_ids = torch.argmax(cluster_probs, dim=1)
            cluster_means = self.bg.means(cluster_probs)
        return cluster_ids, cluster_means

    def one_step_update(
        self,
        initialization: torch.Tensor,
        gain_vector: Optional[torch.Tensor] = None,
        inputs: Optional[Dict[str, torch.Tensor]] = None,
    ):
        return self.rnn.next_state(initialization, gain_vector, inputs)


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
        bg_layer_sizes: Optional[Tuple[int, ...]] = (25, 15, 10),
        bg_nfn: Optional[nn.Module] = None,
        bg_input_size: Optional[int] = 1,
        include_bias: bool = False,
        **kwargs,
    ):
        super(RNNStaticBG, self).__init__(**kwargs)
        self.params = {
            "n_hidden": nneurons,
            "latent_dim": nbg,
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
            input_size=bg_input_size,
            include_bias=include_bias,
            output_size=nbg,
            non_linearity=bg_nfn,
            layer_sizes=bg_layer_sizes,
        )
        # self.bg = nn.Parameter(torchify(np.zeros((1, nneurons))))
        # if bg_nfn is None:
        #     self.bg_nfn = nn.Sigmoid()
        # else:
        #     self.bg_nfn = bg_nfn
        # self.bg = MLP(
        #     layer_sizes=bg_layer_sizes,
        #     non_linearity=bg_nfn,
        #     input_size=bg_input_size,
        #     output_size=latent_dim,
        #     include_bias=include_bias,
        # )

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


class RNNGMM(BaseArchitecture):
    def __init__(
        self,
        nneurons: int = 100,
        n_classes: int = 10,
        nbg: int = 20,
        non_linearity: Optional[nn.Module] = None,
        g0: float = 1.2,
        input_sources: Optional[Dict[str, Tuple[int, bool]]] = None,
        dt: float = 0.05,
        tau: float = 0.15,
        bg_layer_sizes: Optional[Tuple[int, ...]] = (25, 15),
        bg_nfn: Optional[nn.Module] = None,
        bg_input_size: Optional[int] = 1,
        include_bias: bool = False,
        **kwargs,
    ):
        super(RNNGMM, self).__init__(**kwargs)
        self.params = {
            "n_hidden": nneurons,
            "latent_dim": nbg,
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

        self.classifier = MLP(
            input_size=bg_input_size,
            layer_sizes=bg_layer_sizes,
            output_size=n_classes,
            include_bias=include_bias,
            non_linearity=bg_nfn,
            std=1 / n_classes,
            return_nnl=False,
        )

        self.bg = GaussianMixtureModel(
            number_of_clusters=n_classes, latent_dimension=nbg
        )

    def set_outputs(self):
        self.output_names = ["r_hidden", "r_act", "bg_act"]

    def forward(
        self,
        bg_inputs: Dict[str, torch.Tensor],
        rnn_inputs: Optional[Dict[str, torch.Tensor]] = None,
        tau: float = 1.0,
        hard: bool = True,
        **kwargs,
    ):

        if "cluster_probs" in bg_inputs.keys():
            cluster_probs = bg_inputs["cluster_probs"]
        else:
            classifier_input = next(iter(bg_inputs.values()), None)
            logits = self.classifier(classifier_input)

            # cluster_probs = nn.functional.gumbel_softmax(logits, tau=tau, hard=hard)
            cluster_probs = nn.functional.softmax(
                logits,
            )

        bg_act = self.bg(cluster_probs)
        try:
            r_hidden, r_act = self.rnn(bg_act, inputs=rnn_inputs, **kwargs)
        except RuntimeError:
            pdb.set_trace()
        return {
            "r_hidden": r_hidden,
            "r_act": r_act,
            "bg_act": bg_act,
            "cluster_probs": cluster_probs,
        }

    def description(
        self,
    ):
        """"""
        print("An RNN who's weights are multiplied by a static gain from the BG")

    def get_input_stats(
        self,
        classifier_input: torch.Tensor,
        cluster_probs: Optional[torch.Tensor] = None,
    ) -> (torch.Tensor, torch.Tensor):
        with torch.no_grad():
            if cluster_probs is None:
                cluster_logits = self.classifier(classifier_input)
                cluster_probs = nn.functional.softmax(cluster_logits, dim=1)
            cluster_ids = torch.argmax(cluster_probs, dim=1)
            cluster_means = self.bg.means(cluster_probs)
        return cluster_ids, cluster_means

    def one_step_update(
        self,
        initialization: torch.Tensor,
        gain_vector: Optional[torch.Tensor] = None,
        inputs: Optional[Dict[str, torch.Tensor]] = None,
    ):
        return self.rnn.next_state(initialization, gain_vector, inputs)


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
        super(RNNFeedbackBG, self).__init__(**kwargs)
        self.params = {
            "n_hidden": nneurons,
            "latent_dim": nbg,
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


class GMMVAE(BaseArchitecture):
    def __init__(
        self,
        number_of_clusters: int = 5,
        input_dim: int = 512,
        latent_dim: int = 10,
        task_encoder_layer_sizes: Optional[tuple] = (250, 150, 100),
        latent_encoder_layer_sizes: Optional[tuple] = (50, 25, 15),
        latent_decoder_layer_sizes: Optional[tuple] = (50, 25, 15),
        input_decoder_layter_sizes: Optional[tuple] = (50, 25, 15),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.params = {
            "latent_dimensionality": latent_dim,
            "input_dimensionality": input_dim,
            "number_of_clusters": number_of_clusters,
            "network": type(self).__name__,
        }

        self.encoder = EncoderNetwork(
            number_of_clusters=number_of_clusters,
            input_dim=input_dim,
            latent_dim=latent_dim,
            task_encoder_layer_sizes=task_encoder_layer_sizes,
            latent_encoder_layer_sizes=latent_encoder_layer_sizes,
        )

        self.decoder = DecoderNetwork(
            number_of_clusters=number_of_clusters,
            input_dim=input_dim,
            latent_dim=latent_dim,
            latent_decoder_layer_sizes=latent_decoder_layer_sizes,
            input_decoder_layter_sizes=input_decoder_layter_sizes,
        )

    def forward(
        self, x, tau: float = 1, hard: bool = False, **kwargs
    ) -> Dict[str, torch.Tensor]:
        x = x.view(x.size(0), -1)

        encoder_output = self.encoder(x)
        latent_state, clusters = encoder_output["latent"], encoder_output["cluster"]

        decoder_output = self.decoder(latent_state, clusters)

        output = {**decoder_output, **encoder_output}
        return output

    @classmethod
    def construct_gaussian(cls, mu: torch.Tensor, var: torch.Tensor):
        normal_distribution = MultivariateNormal(mu, torch.diag(var))
        return normal_distribution


class HRLNetwork(BaseArchitecture, ABC):
    def __init__(
        self,
        latent_dim: int = 10,
        n_clusters: int = 10,
        layer_sizes: Optional[Tuple[int, ...]] = None,
        non_linearity: Optional[nn.Module] = None,
        observation_size: Optional[int] = 150,
        include_bias: bool = True,
        noise_model: Optional[nn.Module] = None,
        action_dim: int = 2,
        action_layer_sizes: Optional[Tuple[int, ...]] = None,
        observation_layer_sizes: Optional[Tuple[int, ...]] = None,
    ):
        super().__init__()

        # Defining policy functions
        self.pi_k_o = MLP(
            output_size=n_clusters,
            layer_sizes=layer_sizes,
            noise_model=noise_model,
            non_linearity=non_linearity,
            input_size=observation_size,
            include_bias=include_bias,
        )

        self.pi_z_k = GaussianMixtureModel(
            number_of_clusters=n_clusters,
            latent_dimension=latent_dim,
        )

        self.q_k_a_o = MLP(
            input_size=action_dim + observation_size,
            layer_sizes=observation_layer_sizes,
            noise_model=noise_model,
            non_linearity=non_linearity,
            include_bias=include_bias,
            output_size=n_clusters,
        )

        self.pi_a_o_z = MultiHeadMLP(
            independent_layers={
                "observations": ((25, 10), observation_size),
                "latents": ((25, 10), latent_dim),
            },
            shared_layer_sizes=action_layer_sizes,
            output_size=action_dim,
        )

    def forward(self, observation: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:

        cluster_logits = self.pi_k_o(observation)
        clusters = nn.functional.gumbel_softmax(cluster_logits)
        cluster_ids = clusters.argmax(dim=1)
        latents = self.pi_z_k(cluster_ids)
        inputs = {"observations": observation, "latents": latents}
        actions = self.pi_a_o_z(inputs=inputs)
        pdb.set_trace()
        outputs = {
            "actions": actions,
            "cluster_ids": cluster_ids,
            "latents": latents,
            "cluster_probs": cluster_logits,
        }
        return actions

    def description(
        self,
    ):
        """"""

    def set_outputs(self):
        """"""

        self.output_names = ["actions", "cluster_ids", "latents", "cluster_probs"]


class PallidalRNN(BaseArchitecture, ABC):
    """A recurrent model of the Basal Ganglia - Thalamus - Cortex circuit"""

    def __init__(
        self,
        nneurons: int = 100,
        non_linearity: Optional[nn.Module] = None,
        g0: float = 1.2,
        input_sources: Optional[Dict[str, Tuple[int, bool]]] = None,
        dt: float = 0.05,
        tau: float = 0.15,
        tags: List[str] = None,
        **kwargs,
    ):
        super(PallidalRNN, self).__init__(**kwargs)
        self.tags = tags
        self.params = {
            "n_hidden": nneurons,
            "inputs": input_sources,
            "network": type(self).__name__,
            "tags": self.tags,
        }
        self.rnn = BGRNN(
            nneurons=nneurons,
            non_linearity=non_linearity,
            g0=g0,
            input_sources=input_sources,
            dt=dt,
            tau=tau,
        )

        learning_types = {
            "inputs": [self.rnn.I],
            "cortical": [self.rnn.J, self.rnn.B],
            "pallidal": [self.rnn.Wb],
            "thalamocortical": [self.rnn.U],
            "corticothalamo": [self.rnn.Vt],
        }

        self.learning_styles.update(learning_types)

    def set_outputs(self):
        self.output_names = ["r_hidden", "r_act"]

    def forward(self, rnn_inputs: Optional[Dict[str, torch.Tensor]] = None, **kwargs):
        r_hidden, r_act = self.rnn.forward(rnn_inputs)
        return {"r_hidden": r_hidden, "r_act": r_act}

    def description(
        self,
    ):
        """"""
        print("A basic implementation of the pallidal-thalamo-cortical loop.")


NETWORKS = {
    "VanillaRNN": VanillaRNN,
    "RNNStaticBG": RNNStaticBG,
    "RNNFeedbackBG": RNNFeedbackBG,
    "RNNMultiContextInput": RNNMultiContextInput,
    "GMM": RNNGMM,
    "PallidalRNN": PallidalRNN,
}
