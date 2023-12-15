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
from model_factory.architectures import RNNGMM, RNNMultiContextInput
from typing import Sequence, Union
from pathlib import Path
from itertools import product
from SineGeneration.analyze_models import set_plt_params, make_axis_nice


def split_dataset(dataset, fractions: Sequence = (0.05, 0.65, 0.2)):
    train_set, val_set, test_set = random_split(dataset, fractions)
    train_sampler = RandomSampler(train_set)
    val_sampler = RandomSampler(val_set)
    train = {"data": train_set, "sampler": train_sampler}
    val = {"data": val_set, "sampler": val_sampler}

    return train, val, test_set


set_plt_params()

## Train network to produce 1 sine
duration = 300
training_steps = 15
batch_size = 1
amplitudes = (0.5, 1.5)
frequencies = (0.5, 1.5)
dataset = SineDataset(duration=duration, amplitudes=amplitudes, frequencies=frequencies)
dt = 0.01
tau = 0.15
lr, wd = 1e-3, 1e-6
plot_freq = 25

fraction = 0.01
train_set, val_set, test_set = split_dataset(dataset, (fraction, 0.8 - fraction, 0.2))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_loader = DataLoader(
    train_set["data"],
    batch_size=batch_size,
    sampler=train_set["sampler"],
    num_workers=10,
)

network_data = {
    "RNNGMM": {"label": "Basal Ganglia", "data": []},
    "RNNMultiContextInput": {"label": "Input Only", "data": []},
}


model_path = Path(
    "/home/elom/Documents/basal_ganglia/data/models/SineGeneration/2023-12-10/model_0/model.pickle"
)

input_model_path = Path(
    "//home/elom/Documents/basal_ganglia/data/models/SineGeneration/2023-12-14/model_0/model.pickle"
)

model_path_names = [
    "//home/elom/Documents/basal_ganglia/data/models/SineGeneration/2023-12-14/model_10/model.pickle",
    "//home/elom/Documents/basal_ganglia/data/models/SineGeneration/2023-12-14/model_8/model.pickle",
    "/home/elom/Documents/basal_ganglia/data/models/SineGeneration/2023-12-10/model_1/model.pickle",
    "/home/elom/Documents/basal_ganglia/data/models/SineGeneration/2023-12-10/model_0/model.pickle",
]

model_paths = [Path(name) for name in model_path_names]

train_bg = True
train_J = False
train_I = False

for model_path in model_paths:
    if model_path.exists():
        with open(model_path, "rb") as h:
            pickled_data = pickle.load(h)

        trained_task: Union[RNNGMM, RNNMultiContextInput] = copy.deepcopy(
            pickled_data["task"]
        )  # load trained network in memory
        network_type = trained_task.network.params["network"]
        if trained_task.network.params["ncontexts"] == 2:

            trained_task.network.rnn.reset_state(batch_size=10)
            if hasattr(trained_task.network, "bg"):
                n_clusters = trained_task.network.bg.nclusters
                latent_dim = trained_task.network.bg.latent_dim
                trained_task.cluster_labels = {}
            means = torch.zeros(n_clusters, latent_dim)
            cov = 0.25 * torch.ones(latent_dim)
            beta = 0.9

            loss = []

            ## Train BG via RL ##
            if train_bg:
                for epoch in range(training_steps):
                    epoch_errors = []
                    for batch in train_loader:
                        (timing_cues, contexts), y = batch

                        inputs = {"cues": timing_cues, "parameters": contexts}

                        trained_task.network.rnn.reset_state(2 * batch_size)
                        position_store = torch.zeros(
                            duration,
                            2 * batch_size,
                            1,
                            device=trained_task.network.Wout.device,
                        )

                        parameters_amp = contexts[0, 0, 0].cpu().numpy().item()
                        parameters_freq = contexts[0, 1, 0].cpu().numpy().item()
                        tuples = (
                            (round(parameters_amp, 4), round(parameters_freq, 4)),
                        )
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
                                    "cues": torch.tile(
                                        timing_cues[:, :, ti], dims=(2, 1)
                                    ),
                                    "target_parameters": torch.tile(
                                        contexts[:, :, ti], dims=(2, 1)
                                    ),
                                }
                                r_hidden, r_act = trained_task.network.rnn(
                                    bg_act, inputs=rnn_inputs
                                )
                                position_store[ti] = r_act @ trained_task.network.Wout
                        error = (
                            (y.squeeze()[:, None] - position_store.squeeze()) ** 2
                        ).sum(axis=0)
                        if error[1] <= error[0]:
                            means[cluster_label] = (
                                beta * means[cluster_label] + (1 - beta) * bg_act[1]
                            )
                        epoch_errors.append(np.min(error.numpy()))
                    loss.append(np.mean(epoch_errors))
            network_data[network_type]["data"].append(loss)

color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
fig, ax = plt.subplots()

for idx, (key, value) in enumerate(network_data.items()):
    label = value["label"]
    losses = np.vstack(value["data"])
    mean_loss = losses.mean(axis=0)
    loss_std = losses.std(axis=0)

    ax.scatter(
        range(mean_loss.shape[0]), mean_loss, label=label, color=color_cycle[idx]
    )
    ax.plot(range(mean_loss.shape[0]), mean_loss, color=color_cycle[idx], ls="--")
    ax.fill_between(
        range(mean_loss.shape[0]),
        mean_loss - loss_std,
        mean_loss + loss_std,
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
# fig.savefig(file_name)
plt.pause(0.1)
pdb.set_trace()
