import abc
import pdb
import os
import re
import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from itertools import product
from sklearn.decomposition import PCA
from model_factory.architectures import NETWORKS, BaseArchitecture
from typing import Optional, Any, Tuple, Callable
from scipy.ndimage import gaussian_filter1d
from model_factory.factory_utils import torchify
from datetime import date
from pathlib import Path


class Task(pl.LightningModule, metaclass=abc.ABCMeta):
    def __init__(
        self,
        network: str,
        lr: float = 1e-3,
        wd: float = 0,
        task: Optional[str] = None,
        **kwargs,
    ):
        super(Task, self).__init__()
        self.network = NETWORKS[network](task=task, **kwargs)
        self.type = network
        self.lr = lr
        self.wd = wd
        self.optimizer = None
        self.lr_scheduler = None
        self.network.test_loss = []

    def configure_optimizers(self):
        """"""
        self.optimizer = torch.optim.Adam(
            self.network.parameters(), weight_decay=self.wd, lr=self.lr
        )
        # self.lr_scheduler = {
        #     "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
        #         self.optimizer,
        #     ),
        #     "monitor": "val_loss",
        #     "name": "lr_rate",
        # }
        self.lr_scheduler = {
            "scheduler": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=50, T_mult=2
            ),
        }
        return [self.optimizer], [self.lr_scheduler]
        # return self.optimizer

    def add_noise_to_parameters(self, scale: float = 0.01):
        """

        Args:
            scale:

        Returns:

        """
        for param in self.parameters():
            if param.requires_grad:
                std = param.data.std()
                param.data = param.data + scale * std * torch.randn_like(param.data)

    @abc.abstractmethod
    def compute_loss(self, **kwargs):
        """"""

    def save_model(self):
        # TODO: Save network should
        self.network.save_model(task=self)

    def count_parameters(self):
        """Returns the number of trainable parameters in a model"""
        running_count = 0
        for param in self.network.parameters():
            if param.requires_grad:
                running_count += np.prod(param.shape)
        return running_count


class GenerateSine(Task):
    def __init__(
        self,
        network: Optional[str] = "RNNFeedbackBG",
        nneurons: int = 250,
        duration: int = 500,
        nbg: int = 10,
        **kwargs,
    ):

        kwargs["ncontext"] = 1
        rnn_input_source = {"go": (1, True)}

        super(GenerateSine, self).__init__(
            network=network,
            nneurons=nneurons,
            nbg=nbg,
            input_sources=rnn_input_source,
            include_bias=False,
            **kwargs,
        )

        self.network.params.update({"task": "SineGeneration"})
        self.network.params.update(kwargs)

        self.network.Wout = nn.Parameter(
            torchify(np.random.randn(nneurons, 1) / np.sqrt(nneurons))
        )
        self.duration = duration
        self.dt = self.network.rnn.dt
        self.replay_buffer = None
        self.Pulses = None
        self.fixation = None
        self.pulse_times = None
        self.total_time = None
        self.Loss = None
        self.g1, self.g2 = None, None
        self.kappaog = None
        self.new_targ_params = None
        self.optimizer = None
        self.create_gos_and_targets()
        self.configure_optimizers()
        self.results_path = set_results_path(type(self).__name__)

    def create_gos_and_targets(
        self,
        n_unique_pulses: int = 10,
        pulse_width: int = 10,
        frequency: float = 1,
        delay: int = 0,
        amplitude: int = 1,
    ):
        total_time = self.duration
        pulse_times = np.linspace(
            total_time * 1 / 10, total_time * 4 / 10, n_unique_pulses
        ).astype(int)
        pulses = np.zeros((total_time, n_unique_pulses))
        fixation = np.ones((total_time, n_unique_pulses))
        targets = np.zeros((total_time, n_unique_pulses))
        times = np.arange(total_time)
        period = int(2 * np.pi / self.dt)

        for idx, pulse_start in enumerate(pulse_times):
            pulses[pulse_start : pulse_start + pulse_width, idx] = 1
            fixation[pulse_start:, idx] = 0
            targets[:, idx] = amplitude * np.sin(
                (times - pulse_start - delay) * frequency * self.dt
            )
            targets[: (pulse_start + delay), idx] = 0
            targets[(pulse_start + delay + 3 * period) :, idx] = 0
            pulses[
                (pulse_start + 3 * period) : (pulse_start + pulse_width + 3 * period),
                idx,
            ] = -1

        pulses = torchify(gaussian_filter1d(pulses, sigma=1, axis=0))
        fixation = torchify(gaussian_filter1d(fixation, sigma=1, axis=0))
        targets = torchify(targets)
        self.Pulses = pulses
        self.fixation = fixation
        self.pulse_times = pulse_times
        self.total_time = total_time
        self.Loss = []
        self.Targets = targets
        return pulses

    def forward(
        self,
        go_cues: torch.Tensor,
    ):

        if go_cues.ndim == 1:
            go_cues = go_cues[:, None]
        batch_size = go_cues.shape[1]
        # pdb.set_trace()
        position_store = torch.zeros(
            self.duration, batch_size, 1, device=self.network.Wout.device
        )
        context_input = torch.zeros((batch_size, 1))
        bg_inputs = {"context": context_input}
        self.network.rnn.reset_state(batch_size)

        for ti in range(self.duration):
            rnn_input = {"go": go_cues[ti][:, None]}
            outputs = self.network(bg_inputs=bg_inputs, rnn_inputs=rnn_input)
            position_store[ti] = outputs["r_act"] @ self.network.Wout

        return position_store

    def eval_network(self, ax):
        batch = np.random.randint(10)
        go_cues = self.Pulses[:, batch]
        with torch.no_grad():
            position_store = self.forward(go_cues)
        ax.cla()
        ax.plot(self.Targets[:, batch], label="Target")
        ax.plot(go_cues.cpu(), label="Go Cue")
        ax.plot(position_store[:, 0].detach().cpu(), label="Actual")
        plt.legend()
        return position_store

    def compute_loss2(self, targ_num: np.ndarray, position: torch.Tensor):
        """"""
        Targets = torchify(self.Targets[:, targ_num])[:, :, None]
        loss = ((Targets - position) ** 2).mean()
        return loss

    def compute_loss(self, batch):

        if isinstance(batch, dict):
            batch = batch["X"]
        position = self.forward(batch[0])

        if batch[1].ndim == 1:
            loss = ((batch[1][:, None, None] - position) ** 2).mean()
        else:
            loss = ((batch[1][:, :, None] - position) ** 2).mean()

        losses = {"total": loss}
        return losses

    def configure_optimizers(self):
        """"""
        self.optimizer = torch.optim.Adam(self.network.parameters(), weight_decay=1e-3)
        return self.optimizer

    def _log_loss(self, result_dict, stage: str) -> None:
        for k, v in result_dict.items():
            self.log(
                f"{stage}_{k}",
                v,
                prog_bar=True,
                on_epoch=True,
                on_step=False,
                sync_dist=True,
            )

    def training_step(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Update and log beta, compute losses, and log them
        """
        self.network.rnn.reset_state(device=self.device)
        # Evaluate loss
        result_dict = self.compute_loss(batch)
        pdb.set_trace()
        self._log_loss(result_dict, "train")

        return result_dict["total"]

    def training_loop(
        self,
        data_loader,
        niterations: int = 500,
        plot_freq: int = 5,
        clip_grad: bool = True,
    ):
        fig_loss, ax_loss = plt.subplots(1, 2, figsize=(16, 8))
        for iteration in range(niterations):
            for idx, (x, y) in enumerate(data_loader):
                self.optimizer.zero_grad()

                position = self.forward(x.T)
                loss = nn.functional.mse_loss(position.squeeze(), y.T)
                # loss = self.compute_loss(y, position)
                loss.backward()
                self.Loss.append(loss.item())
                if clip_grad:
                    nn.utils.clip_grad_norm_(
                        self.network.parameters(),
                        1,
                        norm_type=2.0,
                        error_if_nonfinite=False,
                    )
                self.optimizer.step()

            if (iteration % plot_freq) == 0:
                self.eval_network(ax=ax_loss[1])
                ax_loss[0].cla()
                ax_loss[0].plot(self.Loss)
                ax_loss[0].set_yscale("log")
            plt.pause(0.01)
            # pdb.set_trace()

    def plot_different_gains(self, batch_size: int = 50, cmap: mcolors.Colormap = None):
        if cmap is None:
            cmap = plt.cm.inferno
        context_input = torch.linspace(-1, 1, batch_size)[:, None]
        position_store = torch.zeros(self.duration, batch_size, 1)
        bg_inputs = {"context": context_input}
        self.network.rnn.reset_state(batch_size)
        go_cues = self.Pulses[:, np.zeros(batch_size).astype(int)]
        normalization = mcolors.Normalize(vmin=0, vmax=batch_size - 1)
        with torch.no_grad():
            for ti in range(self.duration):
                rnn_input = {"go": go_cues[ti][:, None]}
                outputs = self.network(
                    bg_inputs=bg_inputs, rnn_inputs=rnn_input, noise_scale=0
                )
                position_store[ti] = outputs["r_act"] @ self.network.Wout
        fig, ax = plt.subplots()
        ax.plot(self.Targets[:, 0], label="Target", color="black", ls="--")
        ax.plot(go_cues[:, 0], label="Go Cue", color="red")
        for batch in range(batch_size):
            ax.plot(position_store[:, batch], color=cmap(normalization(batch)))
        ax.legend()
        file_name = self.results_path / "gain_interpolation"
        fig.savefig(file_name)
        plt.show()

        return position_store


class GenerateSinePL(Task):
    def __init__(
        self,
        network: Optional[str] = "RNNFeedbackBG",
        nneurons: int = 150,
        duration: int = 300,
        nbg: int = 10,
        n_context: int = 1,
        provide_probs: bool = True,
        **kwargs,
    ):

        rnn_input_source = {
            "cues": (2, True),
            "target_parameters": (2, True),
        }

        super(GenerateSinePL, self).__init__(
            network=network,
            nneurons=nneurons,
            nbg=nbg,
            input_sources=rnn_input_source,
            include_bias=False,
            bg_input_size=2,
            task="SineGeneration",
            **kwargs,
        )
        self.save_hyperparameters()
        self.network.params.update({"task": "SineGeneration"})
        self.network.params.update({"ncontexts": n_context})
        self.network.params.update(kwargs)

        self.network.Wout = nn.Parameter(
            torchify(np.random.randn(nneurons, 1) / np.sqrt(1))
        )
        self.duration = duration
        self.ncontext = n_context
        self.dt = self.network.rnn.dt
        self.train_loss = []
        self.test_loss = []
        self.replay_buffer = None
        self.Pulses = None
        self.fixation = None
        self.pulse_times = None
        self.total_time = None
        self.Loss = None
        self.g1, self.g2 = None, None
        self.kappaog = None
        self.new_targ_params = None
        self.optimizer = None
        self.configure_optimizers()
        self.param_normalizers = None
        self.cluster_labels = {}
        self.results_path = set_results_path(type(self).__name__)[-1]

        if self.ncontext == 1:
            self.provide_probs = True
        else:
            self.provide_probs = provide_probs

        if self.provide_probs & hasattr(self.network, "bg"):
            self.network.swap_grad_state(
                params_to_swap=[self.network.classifier], grad_state=False
            )

    def reset(self):
        self.train_loss = []
        self.test_loss = []

    def forward(self, inputs: dict, return_clusters: bool = False, **kwargs):

        cues = inputs["cues"]
        parameters = inputs["parameters"]
        batch_size = cues.shape[0]
        position_store = torch.zeros(
            self.duration, batch_size, 1, device=self.network.Wout.device
        )
        cluster_ids = torch.zeros(
            self.duration, batch_size, device=self.network.Wout.device
        )

        if hasattr(self.network, "bg"):
            if self.provide_probs:
                if self.ncontext == 1:
                    cluster_probs = torch.zeros(
                        (batch_size, self.network.bg.nclusters),
                        device=self.network.Wout.device,
                    )
                    cluster_probs[:, 0] = 1
                    bg_inputs = {"cluster_probs": cluster_probs}
                else:
                    parameters_amp = np.unique(parameters.cpu()[:, 0, 0])
                    parameters_freq = np.unique(parameters.cpu()[:, 1, 0])
                    tuples = [
                        (round(amp, 4), round(freq, 4))
                        for amp, freq in product(parameters_amp, parameters_freq)
                    ]
                    cluster_keys = list(self.cluster_labels.keys())
                    [
                        self.cluster_labels.update({tup: len(self.cluster_labels)})
                        for tup in tuples
                        if tup not in cluster_keys
                    ]

                    cluster_probs = torch.zeros(
                        (batch_size, self.network.bg.nclusters),
                        device=self.network.Wout.device,
                    )

                    for batch in range(batch_size):
                        batch_tup = (
                            round(parameters.cpu().numpy()[batch, 0, 0], 4),
                            round(parameters.cpu().numpy()[batch, 1, 0], 4),
                        )
                        cluster_probs[batch, self.cluster_labels[batch_tup]] = 1
                    bg_inputs = {"cluster_probs": cluster_probs}
            else:
                bg_inputs = {}
        else:
            bg_inputs = {}
        self.network.rnn.reset_state(batch_size)

        for ti in range(self.duration):
            bg_inputs.update({"context": parameters[:, :, ti]})
            rnn_input = {
                "cues": cues[:, :, ti],
                "target_parameters": parameters[:, :, ti],
            }
            outputs = self.network(bg_inputs=bg_inputs, rnn_inputs=rnn_input, **kwargs)
            position_store[ti] = outputs["r_act"] @ self.network.Wout
            outuput_keys = outputs.keys()

            if "cluster_probs" in outuput_keys:
                cluster_ids[ti] = torch.argmax(outputs["cluster_probs"], dim=1)
            else:
                cluster_ids = None

        if return_clusters:
            return position_store, cluster_ids
        else:
            return position_store

    # def configure_optimizers(self):
    #     """"""
    #     self.optimizer = torch.optim.Adam(self.network.parameters(), weight_decay=1e-3)
    #     return self.optimizer

    def _log_loss(self, result_dict, stage: str) -> None:
        for k, v in result_dict.items():
            self.log(
                f"{stage}_{k}",
                v,
                prog_bar=True,
                on_epoch=True,
                on_step=False,
                sync_dist=True,
            )

    def compute_loss(self, **kwargs):
        raise NotImplementedError

    def training_step(self, batch: torch.Tensor, batch_idx) -> torch.Tensor:
        """
        Update and log beta, compute losses, and log them
        """
        (timing_cues, contexts), y = batch
        inputs = {"cues": timing_cues, "parameters": contexts}
        positions = self.forward(inputs)
        loss = torch.log(((positions.squeeze() - y.T) ** 2).sum(dim=0).mean() + 1)
        self.train_loss.append(loss.detach().cpu().numpy())
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx) -> torch.Tensor:
        """
        Update and log beta, compute losses, and log them
        """
        (timing_cues, contexts), y = batch
        inputs = {"cues": timing_cues, "parameters": contexts}
        positions = self.forward(inputs, noise_scale=0)
        loss = torch.log(((positions.squeeze() - y.T) ** 2).sum(dim=0).mean() + 1)
        self.test_loss.append(loss.detach().cpu().numpy())
        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )
        self.log("hp/metric_1", loss, sync_dist=True)
        return loss

    def test_step(self, batch: torch.Tensor, batch_idx) -> torch.Tensor:
        """
        Update and log beta, compute losses, and log them
        """
        (timing_cues, contexts), y = batch
        inputs = {"cues": timing_cues, "parameters": contexts}
        positions = self.forward(inputs)
        loss = torch.log(
            ((positions.squeeze() - y.squeeze().T) ** 2).sum(dim=0).mean() + 1
        )
        self.log(
            "test_loss",
            loss,
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )
        self.log("hp/metric_2", loss, sync_dist=True)
        self.log("hp_metric", loss, sync_dist=True)
        return loss

    def evaluate_training(self, batch, original_network: Optional[Task] = None):
        (timing_cues, contexts), y = batch
        inputs = {"cues": timing_cues, "parameters": contexts}
        with torch.no_grad():
            positions, cluster_labels = self.forward(
                inputs, return_clusters=True, noise_scale=0
            )

        targets = y.detach().cpu().numpy()
        outputs = positions.squeeze().detach().cpu().numpy().T
        timing_cues = timing_cues.detach().cpu().numpy()

        plt.figure()
        fig, ax = plt.subplots(1, sharex="col")
        ax.plot(timing_cues[0, 0], label="Go Cue", ls="--", color="green")
        ax.plot(timing_cues[0, 1], label="Stop Cue", ls="--", color="red")
        ax.plot(targets[0], label="Target")
        ax.plot(outputs[0], label="Trained Model Output")
        # ax[1].plot(cluster_labels[:, 0])
        # ax[1].set_ylim([0, cluster_labels.max()])
        plt.legend()
        plt.pause(0.01)

    def get_cluster_means(
        self,
    ):

        context_keys = list(self.cluster_labels.keys())
        tuples = [
            (amp / self.param_normalizers[0], freq / self.param_normalizers[1])
            for (amp, freq) in context_keys
        ]
        n_contexts = len(tuples)
        parameters = np.zeros((n_contexts, 2))
        for idx, (amplitude, frequency) in enumerate(tuples):
            parameters[idx] = np.array(
                [
                    amplitude * self.param_normalizers[0],
                    frequency * self.param_normalizers[1],
                ]
            )
        parameters = torchify(parameters).to(self.device)
        # tuples = [
        #     (round(amp, 4), round(freq, 4))
        #     for amp, freq in product(parameters_amp, parameters_freq)
        # ]

        cluster_probs = torch.zeros(
            (n_contexts, self.network.bg.nclusters),
            device=self.network.Wout.device,
        )

        for batch in range(n_contexts):
            batch_tup = (
                round(parameters.cpu().numpy()[batch, 0], 4),
                round(parameters.cpu().numpy()[batch, 1], 4),
            )
            try:
                cluster_probs[batch, self.cluster_labels[batch_tup]] = 1
            except KeyError:
                print(f"{batch_tup} not in keys")

        cluster_ids, cluster_means = self.network.get_input_stats(
            parameters, cluster_probs=cluster_probs
        )
        unnormalized_params = parameters.clone()
        for i, norm in enumerate(self.param_normalizers):
            unnormalized_params[:, i] /= norm
        return (
            unnormalized_params.cpu().numpy(),
            cluster_ids.cpu().numpy(),
            cluster_means.cpu().numpy(),
        )

    def evaluate_network_clusters(self, go_cues: torch.Tensor):
        n_clusters = self.network.bg.nclusters

        parameters = torch.zeros((n_clusters, 2), device=self.network.Wout.device)
        cluster_probs = torch.zeros(
            (n_clusters, n_clusters), device=self.network.Wout.device
        )
        for key, value in self.cluster_labels.items():
            parameters[value] = torch.from_numpy(np.asarray(key))
            cluster_probs[value, value] = 1
        duration = go_cues.shape[-1]
        go_cues = go_cues[:n_clusters].to(self.network.Wout.device)
        position_store = torch.zeros(
            duration, n_clusters, 1, device=self.network.Wout.device
        )
        activity_store = torch.zeros(
            duration,
            n_clusters,
            self.network.rnn.J.shape[0],
            device=self.network.Wout.device,
        )

        hidden_units_store = torch.zeros_like(activity_store)
        bg_inputs = {"cluster_probs": cluster_probs}
        self.network.rnn.reset_state(n_clusters)
        with torch.no_grad():
            for time in range(duration):
                rnn_input = {
                    "cues": torch.tile(go_cues[0, :, time], dims=(n_clusters, 1)),
                    "target_parameters": parameters,
                }
                outputs = self.network(
                    bg_inputs=bg_inputs, rnn_inputs=rnn_input, noise_scale=0
                )

                position_store[time] = outputs["r_act"] @ self.network.Wout
                activity_store[time] = outputs["r_act"]
                hidden_units_store[time] = outputs["r_hidden"]

        return (
            position_store.squeeze().cpu().numpy(),
            activity_store.squeeze().cpu().numpy(),
            hidden_units_store.squeeze().cpu().numpy(),
        )


class MultiGainPacMan(Task):
    def __init__(
        self,
        network: Optional[str] = "RNNFeedbackBG",
        number_of_neurons: int = 250,
        duration: int = 150,
        nbg: int = 10,
        ncontext: int = 4,
        apply_energy_penalty: Optional[Tuple[str, ...]] = None,
        energy_penalty: float = 1e-2,
        output_weight_penalty: float = 1e-3,
        **kwargs,
    ):
        if apply_energy_penalty is None:
            apply_energy_penalty = ()

        kwargs["ncontext"] = ncontext
        rnn_input_source = {
            "current_height": (1, True),
            "target_derivative": (1, True),
            "target_height": (1, True),
            "environment_params": (ncontext, True),
        }

        super(MultiGainPacMan, self).__init__(
            network=network,
            nneurons=number_of_neurons,
            nbg=nbg,
            input_sources=rnn_input_source,
            include_bias=True,
            task="MultiGainPacMan",
            **kwargs,
        )

        self.save_hyperparameters(ignore=["original_model"])
        self.network.params.update({"task": "MultiGainPacMan"})
        self.network.params.update(kwargs)
        self.network.Wout = nn.Parameter(
            torchify(np.random.randn(number_of_neurons, 1) / np.sqrt(1))
        )
        self.penalize_activity = apply_energy_penalty
        self.duration = duration
        self.dt = self.network.rnn.dt
        self.optimizer = None
        self.energy_penalty = energy_penalty
        self.output_weight_penalty = output_weight_penalty

    def forward(
        self,
        contexts: torch.Tensor,
        targets: torch.Tensor,
        max_pos: float = 10,
        **kwargs,
    ) -> Any:
        """"""
        if contexts.ndim == 1:
            contexts = contexts[:, None]
        batch_size = contexts.shape[1]

        position_store = torch.zeros(
            self.duration,
            batch_size,
            1,
            device=self.network.Wout.device,
            requires_grad=False,
        )
        force_store = torch.zeros(
            self.duration, batch_size, 1, device=self.network.Wout.device
        )
        # position = torch.zeros(batch_size, 1, device=self.network.Wout.device)
        position = targets[0].unsqueeze(1)
        velocity = torch.zeros(batch_size, 1, device=self.network.Wout.device)
        bg_inputs = {"context": contexts.T}
        self.network.rnn.reset_state(batch_size)
        energies = {}
        [energies.update({key: None}) for key in self.penalize_activity]

        for ti in range(self.duration):

            rnn_input = {
                "current_height": position_store[ti - 1].clone(),
                "target_derivative": ((targets[ti] - targets[ti - 1]))[:, None],
                "target_height": targets[ti][:, None],
                "environment_params": contexts.T,
            }
            outputs = self.network(
                bg_inputs=bg_inputs,
                rnn_inputs=rnn_input,
                **kwargs,
            )
            force_store[ti] = torch.clamp(
                outputs["r_act"] @ self.network.Wout, -1e4, 1e4
            )
            # acceleration = (
            #     (outputs["r_act"] @ self.network.Wout) * (contexts[2])[:, None]
            #     - velocity * contexts[1][:, None]
            # ) / (contexts[0][:, None])

            acceleration = (
                (outputs["r_act"] @ self.network.Wout)
                / contexts[0].unsqueeze(1)
                * contexts[2].unsqueeze(1)
                - contexts[1].unsqueeze(1) * velocity
                - contexts[3].unsqueeze(1) * position
            )
            velocity = velocity + self.dt / self.network.rnn.tau * acceleration
            position = torch.clip(
                position + self.dt / self.network.rnn.tau * velocity,
                -max_pos,
                max_pos,
            )
            position_store[ti] = position
            for key in self.penalize_activity:
                if energies[key] is None:
                    energies[key] = torch.zeros(
                        (self.duration, *outputs[key].shape), device=outputs[key].device
                    )
                energies[key][ti] = outputs[key]

        return position_store, force_store, energies

    def compute_loss(
        self,
        target: torch.Tensor,
        model_output: torch.Tensor,
        network_energy: Optional[dict] = None,
        optimal_forces: Optional[torch.Tensor] = None,
        use_optimal: bool = False,
    ) -> dict:
        if optimal_forces is None:
            use_optimal = False
        if use_optimal:
            trajectory_loss = torch.log(
                ((optimal_forces - model_output[:-1].squeeze()) ** 2).sum(axis=0) ** 1
                / 2
                + 1
            ).mean()
        else:
            trajectory_loss = torch.log(
                torch.pow(model_output - target, 2).sum(axis=0) ** 1 / 2 + 1
            ).mean()
        # trajectory_loss = nn.functional.mse_loss(model_output.T, target.T)
        output_weight_loss = self.output_weight_penalty * torch.linalg.norm(
            self.network.Wout
        )
        energy_loss = 0
        if network_energy is not None:
            for key in self.penalize_activity:
                energy_loss = (
                    energy_loss + torch.linalg.norm(network_energy[key], dim=-1).mean()
                )
        total_loss = (
            trajectory_loss + self.energy_penalty * energy_loss + output_weight_loss
        )
        loss = {
            "energy": self.energy_penalty * energy_loss,
            "trajectory": trajectory_loss,
            "output_weight": output_weight_loss,
            "total": total_loss,
        }
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            pdb.set_trace()
            loss = {
                "energy": None,
                "trajectory": None,
                "output_weight": None,
                "total": None,
            }
        return loss

    def get_optimal_forces(
        self,
        targets,
        contexts,
        positions,
    ):
        velocities = torch.diff(positions.squeeze(), dim=0)
        errors = (targets - positions.squeeze())[:-1]
        optimal_force = (
            contexts[0]
            * self.network.rnn.tau
            / self.dt
            * (
                errors / (contexts[2] * self.dt / self.network.rnn.tau)
                - (1 - self.dt / self.network.rnn.tau * contexts[1]) * velocities
            )
        )
        return optimal_force

    def training_step(self, batch: torch.Tensor, batch_idx) -> torch.Tensor:

        # self.add_noise_to_parameters()
        x, y = batch
        positions, forces, energies = self.forward(x.T, y.T, noise_scale=0.25)
        optimal_paths = self.get_optimal_forces(y.T, x.T, positions)
        loss = self.compute_loss(
            y.T,
            positions.squeeze(),
            energies,
            optimal_forces=optimal_paths,
        )
        self.log(
            "train_loss",
            loss["total"],
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )
        self.log(
            "energy_loss",
            loss["energy"],
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )
        return loss["total"]

    def validation_step(self, batch: torch.Tensor, batch_idx) -> torch.Tensor:

        x, y = batch
        positions, forces, energies = self.forward(x.T, y.T, noise_scale=0)
        optimal_paths = self.get_optimal_forces(y.T, x.T, positions)
        loss = self.compute_loss(
            y.T,
            positions.squeeze(),
            energies,
            optimal_forces=optimal_paths,
        )
        self.log(
            "val_loss",
            loss["total"],
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )
        self.log("hp/metric_1", loss["trajectory"], sync_dist=True)
        return loss["total"]

    def test_step(self, batch: torch.Tensor, batch_idx) -> torch.Tensor:

        x, y = batch
        positions, forces, energies = self.forward(x.T, y.T, noise_scale=0)
        optimal_paths = self.get_optimal_forces(y.T.squeeze(), x.T, positions)
        # loss = self.compute_loss(
        #     y.T,
        #     forces.T,
        #     energies,
        #     optimal_forces=optimal_paths,
        # )
        loss = self.compute_loss(
            y.T.squeeze(),
            positions.squeeze(),
            energies,
            optimal_forces=optimal_paths,
        )
        self.log(
            "test_loss",
            loss["total"],
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )
        self.log("hp/metric_2", loss["trajectory"], sync_dist=True)
        self.log("hp_metric", loss["trajectory"], sync_dist=True)
        self.network.test_loss.append(loss["trajectory"].cpu().numpy())
        return loss["total"]

    def evaluate_training(
        self, batch, original_network: Optional[Task] = None, noise_scale: float = 0.05
    ):
        x, y = batch
        self.duration = y.shape[1]
        with torch.no_grad():
            positions, forces, energies = self.forward(
                x.T, y.T, noise_scale=noise_scale
            )
            if hasattr(self.network, "bg"):
                bg_outputs = self.network.bg.detach()  # (x)
            if original_network is not None:
                positions_initial, _, _ = original_network(
                    x.T, y.T, noise_scale=noise_scale
                )
                outputs_initial = positions_initial.squeeze().detach().cpu().numpy().T
            else:
                positions_initial = None
        targets = y.detach().cpu().numpy()
        outputs = positions.squeeze().detach().cpu().numpy().T
        loss = self.compute_loss(y.T, positions.squeeze())["trajectory"]
        plt.figure()
        plt.plot(
            np.arange(self.duration) * self.dt, targets[0], label="Target", ls="--"
        )
        plt.plot(
            np.arange(self.duration) * self.dt, outputs[0], label="Trained Model Output"
        )
        if positions_initial is not None:
            plt.plot(
                np.arange(self.duration) * self.dt,
                outputs_initial[0],
                label="Original Model Output",
                ls="--",
            )
        plt.xlabel("Time (s)")
        plt.ylabel("Agent Height")
        plt.legend()

        if hasattr(self.network, "bg"):
            fig, ax = plt.subplots(1, 2, figsize=(12, 8))
            ax[0].set_title("Contexts")
            g = ax[0].imshow(x, aspect="auto")
            plt.colorbar(g, ax=ax[0], label="Value")
            ax[0].set_xlabel("Context")
            ax[0].set_xticks(
                ticks=range(3), labels=["Mass", "Viscositiy", "Polarity"], rotation=45
            )
            ax[0].set_ylabel("Trial")
            ax[1].set_title("Basal Gangial Output")

            g = ax[1].imshow(bg_outputs, aspect="auto")
            plt.colorbar(g, ax=ax[1], label="Neural activity")
            ax[1].set_xlabel("Neuron")
            ax[1].set_ylabel("Trial")
            plt.pause(0.1)
            bg_pca = PCA().fit_transform(bg_outputs.detach().numpy())
            min_vals, _ = x.min(axis=0)
            new_x = x - min_vals
            max_x, _ = new_x.max(axis=0)
            normed_x = (new_x / max_x).detach().numpy()
        # fig, ax = plt.subplots(1, 1, figsize=(12, 8), subplot_kw={"projection": "3d"})
        # ax.scatter(bg_pca[:, 0], bg_pca[:, 1], bg_pca[:, 2], c=normed_x)
        # ax.set_xlabel("PC1")
        # ax.set_ylabel("PC2")
        # ax.set_zlabel("PC3")
        # plt.pause(0.1)
        return loss

    def change_context(self, batch, new_context: Tuple = (1, 0, -1)):

        x, y = batch
        x[:] = torchify(np.asarray(new_context))

        with torch.no_grad():
            positions, _, energies = self.forward(x.T, y.T)

        targets = y.detach().cpu().numpy()
        outputs = positions.squeeze().detach().cpu().numpy().T
        plt.figure()
        plt.plot(targets[0], label="Target")
        plt.plot(outputs[0], label="Model Output")
        plt.legend()
        plt.pause(0.1)


class TwoChoiceDecision(Task):
    def __init__(
        self,
        network: Optional[str] = "PallidalRNN",
        number_of_neurons: int = 250,
        duration: int = 150,
        nbg: int = 10,
        ncontext: int = 4,
        apply_energy_penalty: Optional[Tuple[str, ...]] = None,
        energy_penalty: float = 1e-2,
        **kwargs,
    ):
        if apply_energy_penalty is None:
            apply_energy_penalty = ()

        kwargs["ncontext"] = ncontext
        rnn_input_source = {
            "evidence": (1, True),
        }

        super(TwoChoiceDecision, self).__init__(
            network=network,
            nneurons=number_of_neurons,
            nbg=nbg,
            input_sources=rnn_input_source,
            include_bias=True,
            task="TwoChoice",
            **kwargs,
        )

        self.save_hyperparameters(ignore=["original_model"])
        self.network.params.update({"task": "TwoChoice"})
        self.network.params.update(kwargs)
        self._set_difficulty_level(**kwargs)
        self.network.Wout = nn.Parameter(
            torchify(np.random.randn(number_of_neurons, 1) / np.sqrt(1))
        )
        self.penalize_activity = apply_energy_penalty
        self.duration = duration
        self.dt = self.network.rnn.dt
        self.optimizer = None
        self.energy_penalty = energy_penalty

    def _set_difficulty_level(self, difficulty: float = 0.1, **kwargs):
        self.difficulty = difficulty

    def forward(
        self,
        evidence: torch.Tensor,
        **kwargs,
    ) -> Any:
        """"""
        batch_size = evidence.shape[0]

        choice_store = torch.zeros(
            self.duration, batch_size, 1, device=self.network.Wout.device
        )

        self.network.rnn.reset_state(batch_size)
        energies = {}
        [energies.update({key: None}) for key in self.penalize_activity]

        stimulus_noise = torch.randn_like(evidence) * np.sqrt(self.difficulty * 2 / 3)

        for ti in range(self.duration):

            rnn_input = {
                "evidence": evidence + stimulus_noise,
            }

            outputs = self.network(
                rnn_inputs=rnn_input,
                **kwargs,
            )
            choice_store[ti] = torch.clamp(
                outputs["r_act"] @ self.network.Wout, -1e1, 1e1
            )

            for key in self.penalize_activity:
                if energies[key] is None:
                    energies[key] = torch.zeros(
                        (self.duration, *outputs[key].shape), device=outputs[key].device
                    )
                energies[key][ti] = outputs[key]
        return choice_store, energies

    def compute_loss(
        self,
        target_choice: torch.Tensor,
        model_output: torch.Tensor,
        mask: torch.Tensor,
        network_energy: Optional[dict] = None,
    ) -> dict:
        """
        Compute the loss for the batch
        Args:
            target_choice: Target choice that model should reproduce
            model_output: Actual model choice
            mask: Mask determining which choice epochs to weight in loss
            network_energy: Internal energy of the model

        Returns:

        """

        decision_loss = (
            ((model_output.squeeze().T - target_choice) ** 2 * mask).sum(axis=1).mean()
        )

        energy_loss = 0
        if network_energy is not None:
            for key in self.penalize_activity:
                energy_loss = (
                    energy_loss + torch.linalg.norm(network_energy[key], dim=-1).mean()
                )
        total_loss = decision_loss + self.energy_penalty * energy_loss
        loss = {
            "energy": self.energy_penalty * energy_loss,
            "decision": decision_loss,
            "total": total_loss,
        }

        return loss

    def training_step(self, batch: torch.Tensor, batch_idx) -> torch.Tensor:

        evidence, choice, mask = batch
        model_output, energies = self.forward(evidence, noise_scale=0.25)
        loss = self.compute_loss(
            choice,
            model_output,
            mask,
            energies,
        )
        self.log(
            "train_loss",
            loss["total"],
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )
        self.log(
            "energy_loss",
            loss["energy"],
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )
        return loss["total"]

    def validation_step(self, batch: torch.Tensor, batch_idx) -> torch.Tensor:

        evidence, choice, mask = batch
        model_output, energies = self.forward(evidence, noise_scale=0)
        loss = self.compute_loss(
            choice,
            model_output,
            mask,
            energies,
        )

        self.log(
            "val_loss",
            loss["total"],
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )
        self.log("hp/metric_1", loss["decision"], sync_dist=True)
        return loss["total"]

    def test_step(self, batch: torch.Tensor, batch_idx) -> torch.Tensor:
        evidence, choice, mask = batch
        model_output, energies = self.forward(evidence, noise_scale=0)
        loss = self.compute_loss(
            choice,
            model_output,
            mask,
            energies,
        )

        self.log(
            "test_loss",
            loss["total"],
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )

        self.log("hp/metric_2", loss["decision"], sync_dist=True)
        self.log("hp_metric", loss["decision"], sync_dist=True)
        self.network.test_loss.append(loss["decision"].cpu().numpy())
        return loss["total"]

    def calculate_psychometrics(self, samples: int = 100):
        evidence = torchify(np.linspace(-1, 1)).unsqueeze(1)
        choice = np.zeros((samples, evidence.shape[0]))

        for sample in range(samples):
            with torch.no_grad():
                model_output, energies = self.forward(evidence, noise_scale=0.25)
            choice[sample] = model_output.squeeze()[-1].cpu().numpy()
        mean_choice = choice.mean(axis=0)
        evidence = evidence.squeeze().cpu().numpy()
        plt.scatter(evidence, mean_choice)
        plt.xlabel("Evidence Strength")
        plt.ylabel("Average Choice")
        plt.pause(0.1)


def set_results_path(task_name: str):
    cwd = os.getcwd()
    cwd_path = Path(cwd)
    task_path = cwd_path / f"results/{task_name}"
    task_path.mkdir(exist_ok=True)

    date_str = date.today().strftime("%Y-%m-%d")
    date_save_path = task_path / date_str
    date_save_path.mkdir(exist_ok=True)
    reg_exp = "_".join(["Trial", "\d+"])
    files = [
        x
        for x in date_save_path.iterdir()
        if x.is_dir() and re.search(reg_exp, str(x.stem))
    ]
    folder_path = date_save_path / f"Trial_{len(files)}"
    folder_path.mkdir(exist_ok=True)
    return task_path, date_save_path, folder_path
