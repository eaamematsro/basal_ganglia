import abc
import pdb
import torch
import numpy as np
import torch.nn as nn
from enum import Enum
from torch.distributions.multivariate_normal import MultivariateNormal
from .noise_models import GaussianNoise, GaussianSignalDependentNoise
from .factory_utils import torchify
from typing import Callable, Optional, Dict, List, Tuple
from statsmodels.stats.correlation_tools import cov_nearest


class VarianceTypes(Enum):
    UnitVariance = "unit_variance"
    Diagonal = "diagonal_variance"
    Full = "full"


class Module(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self):
        super(Module, self).__init__()

    def freeze_weights(self):
        for param in self.parameters():
            param.requires_grad = False

    def un_freeze_weights(self):
        for param in self.parameters():
            param.requires_grad = True


class RNN(Module):
    def __init__(
        self,
        nneurons: int = 100,
        non_linearity: Optional[nn.Module] = None,
        g0: float = 1.2,
        input_sources: Optional[Dict[str, Tuple[int, bool]]] = None,
        dt: float = 5e-2,
        tau: float = 0.15,
        noise_model: Optional[nn.Module] = None,
    ):
        """
        Base class for recurrent neural networks

        Args:
            nneurons: int, optional
                Number of recurrently connected neurons to use. by default 100
            non_linearity: nn.Module, optional
                Activation function to use for neural responses. by default SoftPlus
            g0: float, optional
                Gain multiplier of initial recurrent weights.
            input_sources: Dict[str, Tuple[int, bool], optional
                A dictionary that stores external inputs to the RNN. Keys are given by the
                input name and the values are a tuple consisting of an integer (corresponding to the
                dimensionality of the input) and a boolean (corresponding to whether these weights
                are learnable)
            dt: float, optional
                Network time step size, by default 0.05
            tau: float, optional
                Network membrane time constant, by default 0.15
            noise_model:  nn.Module, optional
                Model used to inject noise into neural activity, by default GaussianNoise
        """
        super(RNN, self).__init__()

        if noise_model is None:
            noise_model = GaussianNoise(sigma=np.sqrt(2 * dt / tau))

        if non_linearity is None:
            non_linearity = nn.Softplus()

        self.nonlinearity = non_linearity

        self.I = nn.ParameterDict({})

        if input_sources is not None:
            for input_name, (input_size, learnable) in input_sources.items():
                input_mat = np.random.randn(input_size, nneurons) / np.sqrt(nneurons)
                input_tens = torchify(input_mat)
                if learnable:
                    self.I[input_name] = nn.Parameter(input_tens)
                else:
                    self.I[input_name] = input_tens

        J_mat = g0 * np.random.randn(nneurons, nneurons) / np.sqrt(nneurons)
        J_mat = torchify(J_mat)
        self.input_names = set(list(self.I.keys()))
        self.J = nn.Parameter(J_mat)
        self.B = nn.Parameter(
            torchify(np.random.randn(1, nneurons) / np.sqrt(nneurons))
        )
        self.noise_model = noise_model
        self.x, self.r = None, None
        self.dt = dt
        self.tau = tau

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        noise_scale: float = 1,
        validate_inputs: bool = False,
        max_val: float = 1e3,
    ):
        if validate_inputs:
            input_names = set(list(inputs.keys()))
            assert input_names.intersection(self.input_names) == input_names
        out = 0
        for input_name, input_value in inputs.items():
            # print(input_name, input_value.shape, self.I[input_name].shape)
            out += input_value @ self.I[input_name]

        x = torch.clip(
            self.x
            + self.dt
            / self.tau
            * (
                self.noise_model(
                    -self.x + self.r @ self.J + self.B + out,
                    noise_scale,
                )
            ),
            -max_val,
            max_val,
        )

        r = self.nonlinearity(x)
        self.x = x
        self.r = r
        return self.x, self.r

    def reset_state(self, batch_size: int = 10):

        self.x = torch.randn(
            (batch_size, self.J.shape[0]), device=self.J.device
        ) / np.sqrt(self.J.shape[0])
        self.r = self.nonlinearity(self.x)


class ThalamicRNN(Module):
    """Base class for recurrent neural networks"""

    def __init__(
        self,
        nneurons: int = 100,
        nbg: int = 20,
        non_linearity: Optional[nn.Module] = None,
        g0: float = 1.2,
        input_sources: Optional[Dict[str, Tuple[int, bool]]] = None,
        dt: float = 5e-2,
        tau: float = 0.15,
        noise_model: Optional[nn.Module] = None,
    ):
        super(ThalamicRNN, self).__init__()

        if non_linearity is None:
            non_linearity = nn.Softplus()

        if noise_model is None:
            noise_model = GaussianNoise(sigma=np.sqrt(2 * dt / tau))

        self.nonlinearity = non_linearity

        self.I = nn.ParameterDict({})

        if input_sources is not None:
            for input_name, (input_size, learnable) in input_sources.items():
                input_mat = np.random.randn(input_size, nneurons) / np.sqrt(nneurons)
                input_tens = torchify(input_mat)
                if learnable:
                    self.I[input_name] = nn.Parameter(input_tens)
                else:
                    self.I[input_name] = input_tens

        J_mat = g0 * np.random.randn(nneurons, nneurons) / np.sqrt(nneurons)
        J_mat = torchify(J_mat)
        self.input_names = set(list(self.I.keys()))
        self.J = nn.Parameter(J_mat)
        self.B = nn.Parameter(torchify(np.random.randn(1, nneurons)))
        # self.U = nn.Parameter(
        #     torchify(np.random.randn(nneurons, nbg) / np.sqrt(nneurons)),
        #     requires_grad=False,
        # )
        # self.V = nn.Parameter(
        #     torchify(np.random.randn(nbg, nneurons) / np.sqrt(nneurons)),
        #     requires_grad=False,
        # )
        U, V = self.generate_bg_weights(nneurons=nneurons, rank=nbg)
        self.U = nn.Parameter(torchify(U), requires_grad=True)
        self.V = nn.Parameter(torchify(V), requires_grad=False)
        self.x, self.r = None, None
        self.dt = dt
        self.tau = tau
        self.noise_model = noise_model

    def generate_bg_weights(
        self, nneurons: int = 100, overlap: float = 0.5, rank: int = 5
    ):
        sigmas = np.eye(2 * rank)
        sig_vals = 0.1 * np.random.randn(rank, rank)
        np.fill_diagonal(sig_vals, overlap)
        sigmas[rank:, :rank] = sig_vals
        sigmas[:rank, rank:] = sig_vals.T
        cov = cov_nearest(sigmas)
        L = np.linalg.cholesky(cov)
        samples = L @ np.random.randn(2 * rank, nneurons)
        U = samples[:rank].T * 1 / np.sqrt(nneurons)
        V = samples[rank:] * 1 / np.sqrt(nneurons)
        return U, V

    def reconfigure_u_v(
        self, g1: float = 0, g2: float = 0, requires_grad: bool = False
    ):
        J = self.J.detach().cpu().numpy()
        U, S, Vh = np.linalg.svd(J)
        scale_u = np.sqrt(self.U.detach().cpu().numpy().var() / U.var())
        scale_v = np.sqrt(self.V.detach().cpu().numpy().var() / Vh.var())

        U *= scale_u
        Vh *= scale_v

        bg_rank = self.U.shape[1]

        u_grad = self.U.requires_grad
        v_grad = self.V.requires_grad
        self.U = nn.Parameter(
            torchify(
                (np.sqrt(1 - g1**2)) * U[:, :bg_rank]
                + g1**2 * np.random.randn(J.shape[0], bg_rank) / np.sqrt(J.shape[0])
            ),
            requires_grad=(requires_grad if requires_grad else u_grad),
        )

        self.V = nn.Parameter(
            torchify(
                (np.sqrt(1 - g2**2)) * Vh[:bg_rank]
                + g2**2 * np.random.randn(bg_rank, J.shape[0]) / np.sqrt(J.shape[0])
            ),
            requires_grad=(requires_grad if requires_grad else v_grad),
        )

    def forward(
        self,
        r_thalamic,
        inputs: Optional[Dict[str, torch.Tensor]] = None,
        noise_scale: float = 1,
        validate_inputs: bool = False,
    ):
        if inputs is None:
            inputs = {}
        if validate_inputs:
            input_names = set(list(inputs.keys()))
            assert input_names.intersection(self.input_names) == input_names
        out = 0
        for input_name, input_value in inputs.items():
            out += input_value @ self.I[input_name]

        r_mat = torch.diag_embed(r_thalamic)
        bg_tensor = torch.matmul(self.U, torch.matmul(r_mat, self.V))
        rec_input = torch.matmul(bg_tensor, self.r.unsqueeze(2)).squeeze()

        # rec_input = torch.einsum(
        #     "ij, kj, jl, ki -> kl", self.U, r_thalamic, self.V, self.r
        # )

        x = self.x + self.dt / self.tau * (
            self.noise_model(
                -self.x + self.r @ self.J + rec_input + self.B + out,
                noise_scale,
            )
        )

        r = self.nonlinearity(x)
        self.x = x
        self.r = r
        return self.x, self.r

    def reset_state(self, batch_size: int = 10):
        self.x = torch.randn(
            (batch_size, self.J.shape[0]), device=self.J.device
        ) / np.sqrt(self.J.shape[0])
        self.r = self.nonlinearity(self.x)


class MLP(Module):
    def __init__(
        self,
        layer_sizes: Optional[Tuple[int, ...]] = None,
        non_linearity: Optional[nn.Module] = None,
        input_size: Optional[int] = 150,
        output_size: Optional[int] = 10,
        include_bias: bool = True,
        noise_model: Optional[nn.Module] = None,
    ):
        super(MLP, self).__init__()
        if layer_sizes is None or not (type(layer_sizes) == tuple):
            layer_sizes = (
                150,
                100,
                50,
            )

        if non_linearity is None:
            non_linearity = nn.Softplus()

        if noise_model is None:
            noise_model = GaussianNoise(sigma=0.25)

        modules = []
        layer_sizes = layer_sizes + (output_size,)
        previous_size = input_size
        for size in layer_sizes:
            modules.append(nn.Linear(previous_size, size, bias=include_bias))
            modules.append(non_linearity)
            previous_size = size

        self.weights_init(modules)

        self.mlp = nn.Sequential(*modules)
        self.noise_model = noise_model

    def forward(self, inputs: torch.Tensor, noise_scale: float = 1):
        """


        Args:
            inputs:
            noise_scale:
        """
        y = self.noise_model(self.mlp(inputs), noise_scale)
        return y

    def weights_init(self, modules):
        for m in modules:
            if isinstance(m, nn.Linear):
                # nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('sigmoid'))
                nn.init.normal_(
                    m.weight, std=np.sqrt(1 / (m.weight.shape[0] * m.weight.shape[1]))
                )
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=np.sqrt(1 / (m.bias.shape[0])))


class MultiHeadMLP(Module):
    def __init__(
        self,
        independent_layers: Optional[Dict[str, Tuple[Tuple[int, ...], int]]] = None,
        shared_layer_sizes: Optional[Tuple[int, ...]] = None,
        non_linearity: Optional[nn.Module] = None,
        input_size: Optional[int] = 150,
        output_size: Optional[int] = 10,
        include_bias: bool = True,
        noise_model: Optional[nn.Module] = None,
    ):
        super(MultiHeadMLP, self).__init__()

        if independent_layers is None:
            independent_layers = {
                "input_1": ((100, 50), 10),
                "input_2": ((100, 50), 10),
            }

        assert (
            len(independent_layers) > 1
        ), "There are less than one input heads. Consider using MLP instead."
        self.input_names = set(list(independent_layers.keys()))
        self.input_mlps = nn.ParameterDict({})
        for input_name, (input_sizes, layer_input) in independent_layers.items():
            self.input_mlps[input_name] = MLP(
                layer_sizes=input_sizes,
                output_size=input_size,
                input_size=layer_input,
                non_linearity=non_linearity,
                include_bias=include_bias,
                noise_model=noise_model,
            )

        self.shared_mlp = MLP(
            layer_sizes=shared_layer_sizes,
            input_size=input_size,
            output_size=output_size,
            non_linearity=non_linearity,
            include_bias=include_bias,
            noise_model=noise_model,
        )

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        validate_inputs: bool = True,
        noise_scale: float = 0.05,
    ):
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
            combined_input += self.input_mlps[input_name](value, noise_scale)

        y = self.shared_mlp(combined_input, noise_scale)
        return y


class Gaussian(Module):
    def __init__(
        self,
        input_dim: int = 50,
        output_dim: int = 10,
    ):
        super().__init__()
        self.mu = nn.Linear(input_dim, output_dim)
        self.var = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.Softplus(),
        )

    @classmethod
    def reparameterize(self, mu, var):
        std = torch.sqrt(var + torch.finfo().eps)
        noise = torch.randn_like(var)
        z = mu + std * noise
        return z

    def forward(self, x):
        mu = self.mu(x)
        var = self.var(x)
        z = self.reparameterize(mu, var)
        return mu, var, z


class EncoderNetwork(Module):
    def __init__(
        self,
        number_of_clusters: int = 5,
        input_dim: int = 512,
        latent_dim: int = 10,
        task_encoder_layer_sizes: Optional[tuple] = (250, 150, 100),
        latent_encoder_layer_sizes: Optional[tuple] = (50, 25, 15),
        **kwargs
    ):
        super().__init__()
        self.task_encoder = MLP(
            layer_sizes=task_encoder_layer_sizes,
            input_size=input_dim,
            output_size=number_of_clusters,
        )

        self.latent_encoder = nn.Sequential(
            MLP(
                layer_sizes=latent_encoder_layer_sizes,
                input_size=number_of_clusters + input_dim,
                output_size=latent_dim,
            ),
            Gaussian(latent_dim, latent_dim),
        )

    def forward(self, x, tau: float = 1, hard: bool = True) -> Dict:
        """

        Args:
            x: inputs to cluster
            tau: float, non-negative scalar temperature for softmax
            hard:  if True, the returned samples will be discretized as one-hot vectors,
             but will be differentiated as if it is the soft sample in autograd

        Returns:

        """
        logits = self.task_encoder(x)
        probs = nn.functional.softmax(logits, dim=-1)
        y = nn.functional.gumbel_softmax(logits, tau=tau, hard=hard)
        mu, var, z = self.latent_encoder(y)

        output = {
            "mean": mu,
            "variance": var,
            "latent": z,
            "cluster_prob": probs,
            "cluster": y,
        }
        return output


class DecoderNetwork(Module):
    def __init__(
        self,
        number_of_clusters: int = 5,
        input_dim: int = 512,
        latent_dim: int = 10,
        latent_decoder_layer_sizes: Optional[tuple] = (50, 25, 15),
        input_decoder_layter_sizes: Optional[tuple] = (50, 25, 15),
        **kwargs
    ):
        super().__init__()

        self.latent_mu = MLP(
            layer_sizes=latent_decoder_layer_sizes,
            input_size=number_of_clusters,
            output_size=latent_dim,
        )
        self.latent_var = nn.Sequential(
            MLP(
                layer_sizes=latent_decoder_layer_sizes,
                input_size=number_of_clusters,
                output_size=latent_dim,
            ),
            nn.Softplus(),
        )

        self.input_decoder = MLP(
            layer_sizes=input_decoder_layter_sizes,
            input_size=latent_dim,
            output_size=input_dim,
        )

    def latent_distribution(self, y):
        z_mu = self.latent_mu(y)
        z_var = self.latent_var(y)
        return z_mu, z_var

    def input_generator(self, z):
        input = self.input_decoder(z)
        return input

    def forward(self, z, y) -> Dict:
        """

        Args:
            z: torch.Tensor, Latent state
            y: torch.Tensor, cluster label

        Returns:
            output: Dict,
        """

        z_mu, z_var = self.latent_distribution(y)

        reconstructed_input = self.input_generator(z)
        output = {
            "cluster_latent_mean": z_mu,
            "cluster_latent_var": z_var,
            "reconstruction": reconstructed_input,
        }
        return output


# TODO(eamematsro): Add a feedforward multicontext network


def transfer_network_weights(
    target_model: Module, source: Module, freeze: bool = False
) -> Module:
    source_state_dict = source.state_dict()
    target_state_dict = target_model.state_dict()

    source_keys = list(source_state_dict.keys())

    [
        target_state_dict.update({key: source_state_dict[key]})
        for key in target_state_dict.keys()
        if key in source_keys
    ]

    target_model.load_state_dict(target_state_dict)
    source_names = [name for name, _ in source.named_parameters()]

    if freeze:
        for name, param in target_model.named_parameters():
            if name in source_names:
                param.requires_grad = False

    return target_model
