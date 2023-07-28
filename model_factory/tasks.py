import abc
import pdb
import os
import re
import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from .architectures import NETWORKS
from typing import Optional, Any
from scipy.ndimage import gaussian_filter1d
from .factory_utils import torchify
from datetime import date
from pathlib import Path



class Task(pl.LightningModule, metaclass=abc.ABCMeta):
    def __init__(self, network: str, device: Optional[torch.device] = None,
                 **kwargs):
        super(Task, self).__init__()
        self.network = NETWORKS[network](**kwargs)
        self.type = network

    @abc.abstractmethod
    def configure_optimizers(self):
        """"""

    @abc.abstractmethod
    def compute_loss(self, **kwargs):
        """"""

    @abc.abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """"""

    def save_model(self):
        self.network.save_model()


class ConsolidationNetwork(nn.Module):
    def __init__(self, nneurons: int = 100, nganglia: int = 20, save_path: str = None, lr: float = 1e-3,
                 device: torch.device = None, ncontexts: int = 2, dt: float = 5e-2, tau: float = .15):
        super().__init__()

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            device = self.device
        else:
            self.device = device

        J_mat = (1.2 * np.random.randn(nneurons, nneurons) / np.sqrt(nneurons))
        self.bg_rank = nganglia
        self.J = nn.Parameter(torch.from_numpy(J_mat.astype(np.float32)).to(device))
        self.U = torch.from_numpy((np.random.randn(nneurons, nganglia) / np.sqrt(nneurons)).astype(np.float32)
                                  ).to(device)
        self.V = torch.from_numpy((np.random.randn(nganglia, nneurons) / np.sqrt(nneurons)).astype(np.float32)
                                  ).to(device)
        self.B_m1 = nn.Parameter(torch.from_numpy(np.random.randn(nneurons, 1).astype(np.float32)).to(device))
        self.B_bg = nn.Parameter(torch.from_numpy(np.zeros(nganglia).astype(np.float32)).to(device))
        self.Wout = nn.Parameter(
            torch.from_numpy((np.random.randn(1, nneurons) / np.sqrt(nneurons)).astype(np.float32)).to(device))
        self.I_go = nn.Parameter(torch.from_numpy(np.random.randn(nneurons, 1).astype(np.float32)).to(device))
        # self.I_fix = nn.Parameter(torch.from_numpy(np.random.randn(nneurons, 1).astype(np.float32)).to(device))
        # self.I_speed = nn.Parameter(torch.from_numpy(np.random.randn(nneurons, 1).astype(np.float32)).to(device))
        self.neural_nonlinearity = nn.Softplus()
        self.dt = dt
        self.tau = tau
        self.opt = torch.optim.Adam(self.parameters(), lr=lr)
        self.save_path = save_path
        self.replay_buffer = None
        self.Pulses = None
        self.fixation = None
        self.pulse_times = None
        self.total_time = None
        self.Loss = None
        self.g1, self.g2 = None, None
        self.kappaog = None
        self.new_targ_params = None
        self.create_gos_and_targets()
        # self.create_targets()

    def create_gos(self, total_time: int = 500, n_unique_pulses: int = 10):
        device = self.device
        pulse_times = np.linspace(total_time * 1 / 10, total_time * 4 / 10, n_unique_pulses).astype(int)

        pulses = np.zeros((total_time, n_unique_pulses))
        fixation = np.ones((total_time, n_unique_pulses))

        for idx, pulse_start in enumerate(pulse_times):
            pulses[pulse_start: pulse_start + 10, idx] = 1
            fixation[pulse_start:, idx] = 0
        pulses = gaussian_filter1d(pulses, sigma=1, axis=0)
        pulses = torch.from_numpy(pulses.astype(np.float32)).to(device)
        fixation = gaussian_filter1d(fixation, sigma=1, axis=0)
        fixation = torch.from_numpy(fixation.astype(np.float32)).to(device)
        self.Pulses = pulses
        self.fixation = fixation
        self.pulse_times = pulse_times
        self.total_time = total_time
        self.Loss = []
        return pulses

    def create_targets(self, frequency: float = 1, delay: int = 0,
                       amplitude: int = 1, amplitude_modulated: bool = False,
                       **kwargs):
        pulse_times = self.pulse_times
        targets = np.zeros((self.total_time, pulse_times.shape[0]))
        times = np.arange(self.total_time)
        period = int(2 * np.pi / (self.dt))
        if amplitude_modulated:
            for idx, pulse in enumerate(pulse_times):
                amplitudes = amplitude * np.sin((times - pulse - delay) * (frequency / 2) * self.dt)
                targets[:, idx] = amplitudes * np.sin((times - pulse - delay) * frequency * self.dt)
                targets[:(pulse + delay), idx] = 0
                targets[(pulse + delay + 3 * period):, idx] = 0
        else:
            for idx, pulse in enumerate(pulse_times):
                targets[:, idx] = amplitude * np.sin((times - pulse - delay) * frequency * self.dt)
                targets[:(pulse + delay), idx] = 0
                targets[(pulse + delay + 3 * period):, idx] = 0
        self.Targets = targets

    def create_gos_and_targets(self, total_time: int = 500, n_unique_pulses: int = 10, pulse_width: int = 10,
                               frequency: float = 1, delay: int = 0, amplitude: int = 1):
        device = self.device
        pulse_times = np.linspace(total_time * 1 / 10, total_time * 4 / 10, n_unique_pulses).astype(int)

        pulses = np.zeros((total_time, n_unique_pulses))
        fixation = np.ones((total_time, n_unique_pulses))
        targets = np.zeros((total_time, n_unique_pulses))
        times = np.arange(total_time)
        period = int(2 * np.pi / (self.dt))

        for idx, pulse_start in enumerate(pulse_times):
            pulses[pulse_start: pulse_start + pulse_width, idx] = 1
            fixation[pulse_start:, idx] = 0
            targets[:, idx] = amplitude * np.sin((times - pulse_start - delay) * frequency * self.dt)
            targets[:(pulse_start + delay), idx] = 0
            targets[(pulse_start + delay + 3 * period):, idx] = 0
            pulses[(pulse_start + 3 * period): (pulse_start + pulse_width + 3 * period), idx] = -1

        pulses = gaussian_filter1d(pulses, sigma=1, axis=0)
        pulses = torch.from_numpy(pulses.astype(np.float32)).to(device)
        fixation = gaussian_filter1d(fixation, sigma=1, axis=0)
        fixation = torch.from_numpy(fixation.astype(np.float32)).to(device)
        self.Pulses = pulses
        self.fixation = fixation
        self.pulse_times = pulse_times
        self.total_time = total_time
        self.Loss = []
        self.Targets = targets
        return pulses

    def forward(self, targets: np.ndarray = None, triggers: np.ndarray = None,
                include_bg: bool = True, speed: int = 1,
                noise_scale: float = 0.15):
        assert targets.shape[1] == triggers.shape[0], 'The number of targets must be equal to the number of go cues.'
        device = self.device
        batch_size = targets.shape[1]
        Targets = torch.from_numpy(targets.astype(np.float32))
        go_cues = self.Pulses[:, triggers]
        position_store = torch.zeros(self.total_time, batch_size)
        xm1 = torch.randn((self.J.shape[0], batch_size), device=device) / np.sqrt(self.J.shape[0])
        rm1 = self.neural_nonlinearity(xm1)
        if include_bg:
            rthal = torch.diag((self.B_bg))
        else:
            rthal = torch.diag(torch.zeros(self.U.shape[1], device=device))

        for ti in range(self.total_time):
            xm1 = xm1 + self.dt / self.tau * (-xm1 + (self.J + self.U @ rthal @ self.V) @ rm1 + self.B_m1 +
                                              self.I_go * go_cues[ti] +
                                              torch.randn(rm1.shape, device=device) * np.sqrt(2 * noise_scale ** 2 *
                                                                                              (self.tau / self.dt)))
            rm1 = self.neural_nonlinearity(xm1)
            position_store[ti] = self.Wout @ rm1
        loss = ((Targets - position_store) ** 2).mean(axis=0).mean()
        plt.clf()
        plt.plot(targets[:, 0], label='Target')
        plt.plot(go_cues[:, 0].cpu(), label='Go Cue')
        plt.plot(position_store[:, 0].detach().cpu(), label='Actual')
        plt.legend()
        plt.pause(0.01)
        return loss

    def plot_different_gains(self, trigger: int = 0, save_path: str = None):
        device = self.device
        target = self.Targets[:, trigger]
        gains = np.linspace(-1, 1)
        position_store = torch.zeros(self.total_time, gains.shape[0])

        go_cues = self.Pulses[:, trigger].unsqueeze(1)
        N = gains.shape[0]
        og_cycler = plt.rcParams["axes.prop_cycle"]
        plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.viridis(np.linspace(0, 1, N)))
        plt.figure()
        plt.plot(target, label='Target', color='black', ls='--')
        plt.plot(go_cues.cpu(), label='Go Cue', color='red')
        for idx, gain in enumerate(gains):
            rthal = torch.diag(gain * (self.B_bg))
            xm1 = torch.randn((self.J.shape[0], gains.shape[0]), device=device) / np.sqrt(self.J.shape[0])
            rm1 = self.neural_nonlinearity(xm1)
            for ti in range(self.total_time):
                xm1 = xm1 + self.dt / self.tau * (-xm1 + (self.J + self.U @ rthal @ self.V)
                                                  @ rm1 + self.B_m1 +
                                                  self.I_go * go_cues[ti])
                rm1 = self.neural_nonlinearity(xm1)
                position_store[ti, idx] = (self.Wout @ rm1)[:, 0]

            plt.plot(position_store[:, idx].detach().cpu(), alpha=.5)
        plt.legend()
        f_path = save_path + '/output_gain_interpolation'
        plt.savefig(f_path)
        plt.pause(1)
        plt.rcParams["axes.prop_cycle"] = og_cycler

    def plot_variances(self, trigger: int = 0, samples: int = 100):
        target = self.Targets[:, trigger]
        device = self.device
        position_store = torch.zeros(self.total_time, samples)
        magnitude = torch.linalg.norm(self.B_bg)
        go_cues = self.Pulses[:, trigger].unsqueeze(1)
        plt.figure()
        plt.plot(target, label='Target', color='black', ls='--')
        plt.plot(go_cues.cpu(), label='Go Cue', color='red')

        for idx in range(samples):
            rthal = torch.diag(torch.randn(self.B_bg.shape, device=device) * magnitude + self.B_bg)
            xm1 = torch.randn((self.J.shape[0], samples), device=device) / np.sqrt(self.J.shape[0])
            rm1 = self.neural_nonlinearity(xm1)
            for ti in range(self.total_time):
                xm1 = xm1 + self.dt / self.tau * (-xm1 + (self.J + self.U @ rthal @ self.V)
                                                  @ rm1 + self.B_m1 +
                                                  self.I_go * go_cues[ti])
                rm1 = self.neural_nonlinearity(xm1)
                position_store[ti, idx] = (self.Wout @ rm1)[:, 0]

            plt.plot(position_store[:, idx].detach().cpu(), alpha=.5)
        plt.legend()
        plt.pause(1)

        vars = position_store.var(axis=1).detach().cpu().numpy()
        plt.figure()
        plt.plot(vars)
        plt.xlabel('Time'
                   )
        plt.ylabel('Variance')
        plt.pause(1)
        pdb.set_trace()


class GenerateSine(Task):
    def __init__(self, network: Optional[str] = "RNNFeedbackBG",
                 nneurons: int = 150, duration: int = 500,
                 nbg: int = 10,

                 **kwargs):

        kwargs['ncontext'] = 1
        rnn_input_source = {
            'go': (1, True)
        }

        super(GenerateSine, self).__init__(network=network, nneurons=nneurons, nbg=nbg,
                                           input_sources=rnn_input_source,
                                           **kwargs)

        self.network.params.update({'task': "SineGeneration"})
        self.network.params.update(kwargs)

        self.Wout = nn.Parameter(
            torchify(np.random.randn(1, nneurons) / np.sqrt(nneurons))
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

    def create_gos_and_targets(self, n_unique_pulses: int = 10, pulse_width: int = 10,
                               frequency: float = 1, delay: int = 0, amplitude: int = 1):
        total_time = self.duration
        pulse_times = np.linspace(total_time * 1 / 10, total_time * 4 / 10, n_unique_pulses).astype(int)
        pulses = np.zeros((total_time, n_unique_pulses))
        fixation = np.ones((total_time, n_unique_pulses))
        targets = np.zeros((total_time, n_unique_pulses))
        times = np.arange(total_time)
        period = int(2 * np.pi / self.dt)

        for idx, pulse_start in enumerate(pulse_times):
            pulses[pulse_start: pulse_start + pulse_width, idx] = 1
            fixation[pulse_start:, idx] = 0
            targets[:, idx] = amplitude * np.sin((times - pulse_start - delay) * frequency * self.dt)
            targets[:(pulse_start + delay), idx] = 0
            targets[(pulse_start + delay + 3 * period):, idx] = 0
            pulses[(pulse_start + 3 * period): (pulse_start + pulse_width + 3 * period), idx] = -1

        pulses = torchify(gaussian_filter1d(pulses, sigma=1, axis=0))
        fixation = torchify(gaussian_filter1d(fixation, sigma=1, axis=0))
        self.Pulses = pulses
        self.fixation = fixation
        self.pulse_times = pulse_times
        self.total_time = total_time
        self.Loss = []
        self.Targets = targets
        return pulses

    def forward(self, targ_num: np.ndarray,):
        batch_size = targ_num.shape[0]
        position_store = torch.zeros(self.duration, batch_size)
        context_input = torch.ones((batch_size, 1))
        bg_inputs = {'context': context_input}
        self.network.rnn.reset_state(batch_size)
        go_cues = self.Pulses[:, targ_num]

        for ti in range(self.duration):
            rnn_input = {'go': go_cues[ti][None, :]}
            outputs = self.network(bg_inputs=bg_inputs, rnn_inputs=rnn_input)
            position_store[ti] = self.Wout @ outputs['r_act']

        return position_store

    def eval_network(self, targ_num: np.ndarray, ax):
        go_cues = self.Pulses[:, targ_num]
        with torch.no_grad():
            position_store = self.forward(targ_num)
        ax.cla()
        ax.plot(self.Targets[:, targ_num[0]], label='Target')
        ax.plot(go_cues[:, 0].cpu(), label='Go Cue')
        ax.plot(position_store[:, 0].detach().cpu(), label='Actual')
        plt.legend()
        return position_store

    def compute_loss(self, targ_num: np.ndarray, position: torch.Tensor):
        """"""
        Targets = torchify(self.Targets[:, targ_num])
        loss = ((Targets - position) ** 2).mean()
        return loss

    def configure_optimizers(self):
        """"""
        self.optimizer = torch.optim.Adam(self.network.parameters())

    def training_loop(self, niterations: int = 500, batch_size: int = 50,
                      plot_freq: int = 50):
        fig_loss, ax_loss = plt.subplots(1, 2, figsize=(16, 8))
        # fig_perf, ax_perf = plt.subplots()
        for iteration in range(niterations):
            self.optimizer.zero_grad()
            targ_num = np.random.randint(0, self.Targets.shape[1], batch_size)
            position = self.forward(targ_num)
            loss = self.compute_loss(targ_num, position)
            loss.backward()
            self.Loss.append(loss.item())
            self.optimizer.step()

            if iteration % plot_freq == 0:
                self.eval_network(targ_num, ax=ax_loss[1])
                ax_loss[0].cla()
                ax_loss[0].plot(self.Loss)
            plt.pause(.01)

    def plot_different_gains(self, batch_size: int = 5,
                             cmap: mcolors.Colormap = None):
        if cmap is None:
            cmap = plt.cm.inferno
        trigger = np.ones(batch_size).astype(int)
        context_input = torch.linspace(0, 1, batch_size)[:, None]
        position_store = torch.zeros(self.duration, batch_size)
        bg_inputs = {'context': context_input}
        self.network.rnn.reset_state(batch_size)
        go_cues = self.Pulses[:, trigger]
        normalization = mcolors.Normalize(vmin=0, vmax=batch_size-1)
        with torch.no_grad():
            for ti in range(self.duration):
                rnn_input = {'go': go_cues[ti][None, :]}
                outputs = self.network(bg_inputs=bg_inputs, rnn_inputs=rnn_input)
                position_store[ti] = self.Wout @ outputs['r_act']
        fig, ax = plt.subplots()
        ax.plot(self.Targets[:, 0], label='Target', color='black', ls='--')
        ax.plot(go_cues[:, 0], label='Go Cue', color='red')
        for batch in range(batch_size):
            ax.plot(position_store[:, batch], color=cmap(normalization(batch)))
        ax.legend()
        file_name = self.results_path / 'gain_interpolation'
        fig.savefig(file_name)
        plt.show()


        return position_store


def set_results_path(task_name: str):
    cwd = os.getcwd()
    cwd_path = Path(cwd)
    task_path = cwd_path / f"results/{task_name}"
    task_path.mkdir(exist_ok=True)

    date_str = date.today().strftime("%Y-%m-%d")
    date_save_path = task_path / date_str
    date_save_path.mkdir(exist_ok=True)
    reg_exp = '_'.join(['Trial', '\d+'])
    files = [x for x in date_save_path.iterdir() if x.is_dir() and re.search(reg_exp, str(x.stem))]
    folder_path = date_save_path / f"Trial_{len(files)}"
    folder_path.mkdir(exist_ok=True)
    return folder_path
