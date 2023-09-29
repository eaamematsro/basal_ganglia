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
from model_factory.architectures import NETWORKS, BaseArchitecture
from typing import Optional, Any, Tuple, Callable
from scipy.ndimage import gaussian_filter1d
from model_factory.factory_utils import torchify
from datetime import date
from pathlib import Path


class Task(pl.LightningModule, metaclass=abc.ABCMeta):
    def __init__(self, network: str,
                 lr: float = 1e-3,
                 **kwargs):
        super(Task, self).__init__()
        self.network = NETWORKS[network](**kwargs)
        self.type = network
        self.lr = lr
        self.optimizer = None
        self.lr_scheduler = None

    def configure_optimizers(self):
        """"""
        self.optimizer = torch.optim.Adam(self.network.parameters(), weight_decay=1e-3, lr=self.lr)
        self.lr_scheduler = {
            'scheduler': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, 50),
            'name': "lr_rate",
        }
        return [self.optimizer], [self.lr_scheduler]
    @abc.abstractmethod
    def compute_loss(self, **kwargs):
        """"""

    def save_model(self):
        # TODO: Save network should
        self.network.save_model()


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
        duration: int = 500,
        nbg: int = 10,
        **kwargs,
    ):

        kwargs["ncontext"] = 1
        rnn_input_source = {"go": (1, True)}

        super(GenerateSinePL, self).__init__(
            network=network,
            nneurons=nneurons,
            nbg=nbg,
            input_sources=rnn_input_source,
            include_bias=False,
            **kwargs,
        )
        self.save_hyperparameters()
        self.network.params.update({"task": "SineGeneration"})
        self.network.params.update(kwargs)

        self.network.Wout = nn.Parameter(
            torchify(np.random.randn(nneurons, 1) / np.sqrt(1))
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
        self.configure_optimizers()
        self.results_path = set_results_path(type(self).__name__)[-1]

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
        context_input = torch.ones((batch_size, 1), device=self.network.Wout.device)
        bg_inputs = {"context": context_input}
        self.network.rnn.reset_state(batch_size)

        for ti in range(self.duration):
            rnn_input = {"go": go_cues[ti][:, None]}
            outputs = self.network(bg_inputs=bg_inputs, rnn_inputs=rnn_input)
            position_store[ti] = outputs["r_act"] @ self.network.Wout

        return position_store

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

    def compute_loss(self, **kwargs):
        raise NotImplementedError

    def training_step(self, batch: torch.Tensor, batch_idx) -> torch.Tensor:
        """
        Update and log beta, compute losses, and log them
        """
        x, y = batch
        positions = self.forward(x.T)
        loss = nn.functional.mse_loss(positions.squeeze(), y.T)
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
        x, y = batch
        positions = self.forward(x.T)
        loss = nn.functional.mse_loss(positions.squeeze(), y.T)
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
        x, y = batch
        positions = self.forward(x.T)
        loss = nn.functional.mse_loss(positions.squeeze(), y.T.squeeze())
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


class MultiGainPacMan(Task):
    def __init__(
        self,
        network: Optional[str] = "RNNFeedbackBG",
        number_of_neurons: int = 250,
        duration: int = 150,
        nbg: int = 10,
        ncontext: int = 3,
        apply_energy_penalty: Optional[Tuple[str, ...]] = None,
        energy_penalty: float = 1e-3,
        output_weight_penalty: float = 1e-3,
        **kwargs,
    ):
        if apply_energy_penalty is None:
            apply_energy_penalty = ()

        kwargs["ncontext"] = ncontext
        rnn_input_source = {"current_height": (1, True),
                            "target_derivative": (1, True),
                            "target_height": (1, True)}

        super(MultiGainPacMan, self).__init__(
            network=network,
            nneurons=number_of_neurons,
            nbg=nbg,
            input_sources=rnn_input_source,
            include_bias=True,
            **kwargs,
        )

        self.save_hyperparameters()
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
        self, contexts: torch.Tensor, targets: torch.Tensor, max_pos: float = 10
    ) -> Any:
        """"""
        if contexts.ndim == 1:
            contexts = contexts[:, None]
        batch_size = contexts.shape[1]

        position_store = torch.zeros(
            self.duration, batch_size, 1, device=self.network.Wout.device
        )
        position = torch.zeros(batch_size, 1, device=self.network.Wout.device)
        velocity = torch.zeros(batch_size, 1, device=self.network.Wout.device)
        bg_inputs = {"context": contexts.T}
        self.network.rnn.reset_state(batch_size)
        energies = {}
        [energies.update({key: None}) for key in self.penalize_activity]

        for ti in range(self.duration):

            rnn_input = {
                "current_height": position_store[ti - 1].clone(),
                "target_derivative": ((targets[ti] - targets[ti - 1]))[:, None],
                "target_height": targets[ti][:, None]
            }
            outputs = self.network(bg_inputs=bg_inputs, rnn_inputs=rnn_input)
            acceleration = (
                (outputs["r_act"] @ self.network.Wout) * (contexts[2])[:, None]
                - velocity * contexts[1][:, None]
            ) / (contexts[0][:, None])
            velocity = velocity + self.dt * acceleration
            position_store[ti] = torch.clip(
                position + self.dt * velocity, -max_pos, max_pos
            )
            for key in self.penalize_activity:
                if energies[key] is None:
                    energies[key] = torch.zeros(
                        (self.duration, *outputs[key].shape), device=outputs[key].device
                    )
                energies[key][ti] = outputs[key]

        return position_store, energies

    def compute_loss(
        self, target: torch.Tensor, model_output: torch.Tensor, network_energy: dict
    ) -> dict:
        # pdb.set_trace()
        trajectory_loss = torch.log(torch.pow(model_output - target, 2).sum(axis=0)).mean()
        # trajectory_loss = nn.functional.mse_loss(model_output.T, target.T)
        output_weight_loss = self.output_weight_penalty * torch.linalg.norm(
            self.network.Wout
        )
        energy_loss = 0
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
        return loss

    def training_step(self, batch: torch.Tensor, batch_idx) -> torch.Tensor:

        x, y = batch
        positions, energies = self.forward(x.T, y.T)
        loss = self.compute_loss(y.T, positions.squeeze(), energies)
        self.log(
            "train_loss",
            loss["total"],
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )
        return loss["total"]

    def validation_step(self, batch: torch.Tensor, batch_idx) -> torch.Tensor:

        x, y = batch
        positions, energies = self.forward(x.T, y.T)
        loss = self.compute_loss(y.T, positions.squeeze(), energies)
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
        positions, energies = self.forward(x.T, y.T)
        loss = self.compute_loss(y.T.squeeze(), positions.squeeze(), energies)
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
        return loss["total"]

    def evaluate_training(self, batch, original_network: Optional[Task] = None):
        x, y = batch
        with torch.no_grad():
            positions, energies = self.forward(x.T, y.T)
            bg_outputs = self.network.bg(x)
            if original_network is not None:
                positions_initial, _ = original_network(x.T, y.T)
                outputs_initial = positions_initial.squeeze().detach().cpu().numpy().T
            else:
                positions_initial = None
        targets = y.detach().cpu().numpy()
        outputs = positions.squeeze().detach().cpu().numpy().T
        plt.figure()
        plt.plot(targets[0], label='Target')
        plt.plot(outputs[0], label='Trained Model Output')
        if positions_initial is not None:
            plt.plot(outputs_initial[0], label='Original Model Output', ls='--')
        plt.legend()
        fig, ax = plt.subplots(1, 2, figsize=(12, 8))
        ax[0].set_title('Contexts')
        g = ax[0].imshow(x, aspect='auto')
        plt.colorbar(g, ax=ax[0], label='Value')
        ax[0].set_xlabel('Context')
        ax[0].set_xticks(ticks=range(3), labels=['Mass', 'Viscositiy', 'Polarity'], rotation=45)
        ax[0].set_ylabel('Trial')
        ax[1].set_title('Basal Gangial Output')
        g = ax[1].imshow(bg_outputs, aspect='auto')
        plt.colorbar(g, ax=ax[1], label='Neural activity')
        ax[1].set_xlabel('Neuron')
        ax[1].set_ylabel('Trial')
        plt.pause(.1)


    def change_context(self, batch, new_context: Tuple = (1, 0, -1)):

        x, y = batch
        x[:] = torchify(np.asarray(new_context))

        with torch.no_grad():
            positions, energies = self.forward(x.T, y.T)

        targets = y.detach().cpu().numpy()
        outputs = positions.squeeze().detach().cpu().numpy().T
        plt.figure()
        plt.plot(targets[0], label='Target')
        plt.plot(outputs[0], label='Model Output')
        plt.legend()
        plt.pause(.1)


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
