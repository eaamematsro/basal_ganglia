import pdb
import copy
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from datasets.loaders import SineDataset
from torch.utils.data import DataLoader, RandomSampler, random_split
from torch.optim import Adam
from model_factory.factory_utils import torchify
from model_factory.architectures import RNNGMM
from typing import Sequence
from pathlib import Path
from itertools import product
from SineGeneration.analyze_models import set_plt_params, make_axis_nice


def split_dataset(dataset, fractions: Sequence = (0.05, 0.65, 0.2)):
    train_set, val_set, test_set = random_split(dataset, fractions)
    train_sampler = RandomSampler(train_set)
    val_sampler = RandomSampler(train_set)
    train = {"data": train_set, "sampler": train_sampler}
    val = {"data": val_set, "sampler": val_sampler}

    return train, val, test_set


set_plt_params()

## Train network to produce 1 sine
duration = 300
training_steps = 25
batch_size = 1
amplitudes = (0.5, 1.5)
frequencies = (0.5, 1.5)
dataset = SineDataset(duration=duration, amplitudes=amplitudes, frequencies=frequencies)
dt = 0.01
tau = 0.15
lr, wd = 1e-3, 1e-6
plot_freq = 25

train_set, val_set, test_set = split_dataset(dataset, (0.01, 0.79, 0.2))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_loader = DataLoader(
    train_set["data"],
    batch_size=batch_size,
    sampler=train_set["sampler"],
    num_workers=10,
)

model_path = Path(
    "/home/elom/Documents/basal_ganglia/data/models/SineGeneration/2023-12-10/model_0/model.pickle"
)
train_bg = True
train_J = True
train_I = True
if model_path.exists():
    with open(model_path, "rb") as h:
        pickled_data = pickle.load(h)
    trained_task: RNNGMM = copy.deepcopy(pickled_data["task"])
    trained_task.network.rnn.reset_state(batch_size=10)

    if hasattr(trained_task.network, "bg"):
        n_clusters = trained_task.network.bg.nclusters
        latent_dim = trained_task.network.bg.latent_dim
        trained_task.cluster_labels = {}
    means = torch.zeros(n_clusters, latent_dim)
    cov = 0.25 * torch.ones(latent_dim)
    beta = 0.9

    loss = []
    loss_var = []

    loss_j = []
    loss_j_var = []

    loss_i = []
    loss_i_var = []

    ## Train BG via RL ##
    if train_bg:
        for epoch in range(training_steps):
            epoch_errors = []
            for batch in train_loader:
                (timing_cues, contexts), y = batch

                inputs = {"cues": timing_cues, "parameters": contexts}

                trained_task.network.rnn.reset_state(2 * batch_size)
                position_store = torch.zeros(
                    duration, 2 * batch_size, 1, device=trained_task.network.Wout.device
                )

                parameters_amp = contexts[0, 0, 0].cpu().numpy().item()
                parameters_freq = contexts[0, 1, 0].cpu().numpy().item()
                tuples = ((round(parameters_amp, 4), round(parameters_freq, 4)),)
                cluster_keys = list(trained_task.cluster_labels.keys())
                [
                    trained_task.cluster_labels.update(
                        {tup: len(trained_task.cluster_labels)}
                    )
                    for tup in tuples
                    if tup not in cluster_keys
                ]
                batch_tup = (
                    round(contexts.cpu().numpy()[0, 0, 0], 4),
                    round(contexts.cpu().numpy()[0, 1, 0], 4),
                )

                cluster_label = trained_task.cluster_labels[batch_tup]
                bg_act = torch.zeros(
                    (2 * batch_size, latent_dim),
                    device=trained_task.network.Wout.device,
                )
                bg_act[0] = means[cluster_label]
                bg_act[1] = means[cluster_label] + cov * torch.randn(
                    latent_dim, device=trained_task.network.Wout.device
                )
                with torch.no_grad():
                    for ti in range(duration):
                        rnn_inputs = {
                            "cues": torch.tile(timing_cues[:, :, ti], dims=(2, 1)),
                            "target_parameters": torch.tile(
                                contexts[:, :, ti], dims=(2, 1)
                            ),
                        }
                        r_hidden, r_act = trained_task.network.rnn(
                            bg_act, inputs=rnn_inputs
                        )
                        position_store[ti] = r_act @ trained_task.network.Wout
                error = ((y.squeeze()[:, None] - position_store.squeeze()) ** 2).sum(
                    axis=0
                )
                if error[1] <= error[0]:
                    means[cluster_label] = (
                        beta * means[cluster_label] + (1 - beta) * bg_act[1]
                    )
                epoch_errors.append(np.min(error.numpy()))
            loss.append(np.mean(epoch_errors))
            loss_var.append(np.std(epoch_errors) / np.sqrt(len(epoch_errors)))

    ## Train recurrent weights via RL ##
    if train_J:
        trained_task: RNNGMM = copy.deepcopy(pickled_data["task"])
        for epoch in range(training_steps):
            epoch_errors = []
            old_network = copy.deepcopy(trained_task)
            for batch in train_loader:
                (timing_cues, contexts), y = batch

                inputs = {"cues": timing_cues, "parameters": contexts}
                trained_task.network.rnn.reset_state(batch_size)
                old_network.network.rnn.reset_state(batch_size)

                trained_task.network.rnn.J = copy.deepcopy(old_network.network.rnn.J)
                position_store = torch.zeros(
                    duration, 2 * batch_size, 1, device=trained_task.network.Wout.device
                )
                trained_task.network.rnn.J += 0.01 * torch.randn_like(
                    trained_task.network.rnn.J
                )
                with torch.no_grad():
                    for ti in range(duration):
                        rnn_inputs = {
                            "cues": timing_cues[:, :, ti],
                            "target_parameters": contexts[:, :, ti],
                        }
                        _, r_act = trained_task.network.rnn(inputs=rnn_inputs)
                        _, r_act_old = old_network.network.rnn(inputs=rnn_inputs)
                        position_store[ti, 1] = r_act @ trained_task.network.Wout
                        position_store[ti, 0] = r_act_old @ trained_task.network.Wout

                error = ((y.squeeze()[:, None] - position_store.squeeze()) ** 2).sum(
                    axis=0
                )
                if error[1] <= error[0]:
                    old_network.network.rnn.J *= beta
                    old_network.network.rnn.J += (1 - beta) * trained_task.network.rnn.J

                epoch_errors.append(np.min(error.numpy()))
            loss_j.append(np.mean(epoch_errors))
            loss_j_var.append(np.std(epoch_errors) / np.sqrt(len(epoch_errors)))

    ## Train Input weights via RL ##
    if train_I:
        trained_task: RNNGMM = copy.deepcopy(pickled_data["task"])
        for epoch in range(training_steps):
            epoch_errors = []
            old_network = copy.deepcopy(trained_task)
            for batch in train_loader:
                (timing_cues, contexts), y = batch

                inputs = {"cues": timing_cues, "parameters": contexts}
                trained_task.network.rnn.reset_state(batch_size)
                old_network.network.rnn.reset_state(batch_size)
                position_store = torch.zeros(
                    duration, 2 * batch_size, 1, device=trained_task.network.Wout.device
                )

                trained_task.network.rnn.I["target_parameters"] = copy.deepcopy(
                    old_network.network.rnn.I["target_parameters"]
                )

                trained_task.network.rnn.I[
                    "target_parameters"
                ] += 0.01 * torch.randn_like(
                    trained_task.network.rnn.I["target_parameters"]
                )

                with torch.no_grad():
                    for ti in range(duration):
                        rnn_inputs = {
                            "cues": timing_cues[:, :, ti],
                            "target_parameters": contexts[:, :, ti],
                        }
                        _, r_act = trained_task.network.rnn(inputs=rnn_inputs)
                        _, r_act_old = old_network.network.rnn(inputs=rnn_inputs)
                        position_store[ti, 1] = r_act @ trained_task.network.Wout
                        position_store[ti, 0] = r_act_old @ trained_task.network.Wout

                error = ((y.squeeze()[:, None] - position_store.squeeze()) ** 2).sum(
                    axis=0
                )
                if error[1] <= error[0]:
                    old_network.network.rnn.I["target_parameters"] *= beta
                    old_network.network.rnn.I["target_parameters"] += (
                        1 - beta
                    ) * trained_task.network.rnn.I["target_parameters"].data

                epoch_errors.append(np.min(error.numpy()))
            loss_i.append(np.mean(epoch_errors))
            loss_i_var.append(np.std(epoch_errors) / np.sqrt(len(epoch_errors)))

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, ax = plt.subplots()
    for idx, (loss_data, vars, label) in enumerate(
        zip(
            [loss, loss_i, loss_j],
            [loss_var, loss_i_var, loss_j_var],
            ["BG", "Inputs", "J"],
        )
    ):
        ax.scatter(
            range(len(loss_data)), loss_data, label=label, color=color_cycle[idx]
        )
        ax.plot(range(len(loss_data)), loss_data, color=color_cycle[idx], ls='--')
        ax.fill_between(
            range(len(loss_data)),
            np.array(loss_data) - np.array(vars),
            np.array(loss_data) + np.array(vars),
            alpha=0.5,
            color=color_cycle[idx],
        )

    plt.legend()
    ax.set_yscale("log")
    ax.set_ylabel("Error")
    cwd = Path(__file__).parent.parent.parent / "results/GenerateSinePL"
    file_name = cwd / "rl_learning_curve"
    ax.set_xlabel("Training Epochs")
    make_axis_nice(fig)
    fig.savefig(file_name)
    plt.pause(0.1)
    pdb.set_trace()
