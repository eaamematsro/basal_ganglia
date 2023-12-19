import pdb
import re
import copy
import torch
import pickle
import numpy as np
import pandas as pd
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
import seaborn as sns


def split_dataset(dataset, fractions: Sequence = (0.01, 0.79, 0.2)):
    train_set, val_set, test_set = random_split(dataset, fractions)
    train_sampler = RandomSampler(train_set)
    val_sampler = RandomSampler(val_set)
    train = {"data": train_set, "sampler": train_sampler}
    val = {"data": val_set, "sampler": val_sampler}

    return train, val, test_set


set_plt_params()

## Train network to produce 1 sine

reg_ex = "(model_)([\d]+)"
duration = 300
training_steps = 15
consolidation_epochs = 50
consolidation_freq = 1
batch_size = 1
amplitudes = (0.5, 1.5)
frequencies = (0.5, 1.5)
n_replays = 128
dataset = SineDataset(duration=duration, amplitudes=amplitudes, frequencies=frequencies)
dt = 0.01
tau = 0.15
lr, wd = 1e-5, 1e0
plot_freq = 25

fraction = 0.6
train_set, val_set, test_set = split_dataset(dataset, (fraction, 0.8 - fraction, 0.2))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_loader = DataLoader(
    train_set["data"],
    batch_size=batch_size,
    sampler=train_set["sampler"],
    num_workers=10,
)

train_loader_consolidation = DataLoader(
    train_set["data"],
    batch_size=n_replays,
    sampler=train_set["sampler"],
    num_workers=10,
)

allowed_networks = ["RNNGMM"]


csv_file_path = Path(__file__).parent / "rl_consolidation_training.csv"

if csv_file_path.exists():
    training_storer = pd.read_csv(csv_file_path, index_col=None)
else:
    training_storer = pd.DataFrame(
        {
            "date": [],
            "model_num": [],
            "epoch_num": [],
            "loss": [],
            "network": [],
            "pretrained": [],
            "train_fraction": [],
        }
    )
# model_path = Path(
#     "/home/elom/Documents/basal_ganglia/data/models/SineGeneration/2023-12-10/model_0/model.pickle"
# )

train_bg = True
task = "SineGeneration"

cwd = Path().cwd()
data_path = cwd / f"data/models/{task}"

date_folders = [x for x in data_path.iterdir() if x.is_dir()]

folders = [[x for x in folder.iterdir() if x.is_dir()] for folder in date_folders]

set_plt_params()
model_store_paths = []
for data in folders:
    model_store_paths.extend(data)

for model_path in model_store_paths:
    model_picke_path = model_path / "model.pickle"
    model_name = model_path.stem
    date = model_path.parent.stem

    if model_path.exists():
        with open(model_picke_path, "rb") as h:
            pickled_data = pickle.load(h)

        reg_result = re.match(reg_ex, model_name)
        model_num = int(reg_result.group(2))

        trained_task: RNNGMM = copy.deepcopy(pickled_data["task"])

        network_type = trained_task.network.params["network"]

        if network_type in allowed_networks:
            trained_task.network.rnn.reset_state(batch_size=10)

            n_clusters = trained_task.network.bg.nclusters
            latent_dim = trained_task.network.bg.latent_dim
            trained_task.cluster_labels = {}

            means = torch.zeros(n_clusters, latent_dim)
            cov = 0.25 * torch.ones(latent_dim)
            beta = 0.9

            loss = []
            loss_no_bg = []
            loss_var = []
            loss_var_no_bg = []

            ## Train BG via RL ##
            if train_bg:
                for epoch in range(training_steps):
                    epoch_errors = []
                    epoch_errors_no_bg = []
                    for batch in train_loader:
                        means.requires_grad = False
                        (timing_cues, contexts), y = batch
                        inputs = {"cues": timing_cues, "parameters": contexts}
                        trained_task.network.rnn.reset_state(2 * batch_size)
                        position_store = torch.zeros(
                            duration,
                            3 * batch_size,
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
                            (3 * batch_size, latent_dim),
                            device=trained_task.network.Wout.device,
                        )

                        bg_act[0] = means[cluster_label]
                        bg_act[1] = means[cluster_label] + cov * torch.randn(
                            latent_dim, device=trained_task.network.Wout.device
                        )

                        trained_task.network.rnn.reset_state(3 * batch_size)
                        with torch.no_grad():
                            for ti in range(duration):
                                rnn_inputs = {
                                    "cues": torch.tile(
                                        timing_cues[:, :, ti], dims=(3, 1)
                                    ),
                                    "target_parameters": torch.tile(
                                        contexts[:, :, ti], dims=(3, 1)
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

                        epoch_errors.append(np.min(error[:2].numpy()))
                        epoch_errors_no_bg.append(error[2])

                    loss.append(np.mean(epoch_errors))
                    loss_var.append(np.std(epoch_errors) / np.sqrt(len(epoch_errors)))
                    loss_no_bg.append(np.mean(epoch_errors_no_bg))
                    loss_var_no_bg.append(
                        np.std(epoch_errors_no_bg) / np.sqrt(len(epoch_errors_no_bg))
                    )
                    epoch_data = [
                        {
                            "date": date,
                            "model_num": model_num,
                            "epoch_num": epoch,
                            "loss": np.mean(epoch_errors),
                            "network": "BG",
                            "pretrained": int(
                                trained_task.network.params["ncontexts"] == 2
                            ),
                            "train_fraction": fraction,
                        },
                        {
                            "date": date,
                            "model_num": model_num,
                            "epoch_num": epoch,
                            "loss": np.mean(epoch_errors_no_bg),
                            "network": "No-BG",
                            "pretrained": int(
                                trained_task.network.params["ncontexts"] == 2
                            ),
                            "train_fraction": fraction,
                        },
                    ]
                    temp_frame = pd.DataFrame(epoch_data, index=[0, 1])
                    training_storer = pd.concat(
                        [training_storer, temp_frame], ignore_index=True
                    )
                    training_storer.to_csv(csv_file_path, index=None)

                    loss_store = []
                    means.requires_grad = True
                    optimizer = torch.optim.Adam(
                        [
                            trained_task.network.rnn.J,
                            means,
                        ],
                        lr=lr,
                    )
                    trained_task.network.rnn.J.requires_grad = True
                    if (epoch % consolidation_freq) == 0:
                        for replay_index, replay_batch in enumerate(
                            train_loader_consolidation
                        ):
                            optimizer.zero_grad()
                            (timing_cues, contexts), y = replay_batch
                            inputs = {"cues": timing_cues, "parameters": contexts}
                            trained_task.network.rnn.reset_state(timing_cues.shape[0])

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

                            cluster_labels = []
                            for idx in range(timing_cues.shape[0]):
                                batch_tup = (
                                    round(contexts.cpu().numpy()[idx, 0, 0], 4),
                                    round(contexts.cpu().numpy()[idx, 1, 0], 4),
                                )
                                cluster_labels.append(
                                    trained_task.cluster_labels[batch_tup]
                                )
                            bg_act = means[cluster_labels]
                            bg_act_consolidation = torch.zeros_like(bg_act)
                            consolidation_target_store = torch.zeros(
                                duration,
                                timing_cues.shape[0],
                                1,
                                device=trained_task.network.Wout.device,
                            )

                            with torch.no_grad():
                                for ti in range(duration):
                                    rnn_inputs = {
                                        "cues": timing_cues[:, :, ti],
                                        "target_parameters": contexts[:, :, ti],
                                    }

                                    r_hidden, r_act = trained_task.network.rnn(
                                        bg_act, inputs=rnn_inputs
                                    )
                                    consolidation_target_store[ti] = (
                                        r_act @ trained_task.network.Wout
                                    )

                            output_store = torch.zeros_like(consolidation_target_store)
                            trained_task.network.rnn.reset_state(timing_cues.shape[0])
                            for ti in range(duration):
                                rnn_inputs = {
                                    "cues": timing_cues[:, :, ti],
                                    "target_parameters": contexts[:, :, ti],
                                }
                                r_hidden, r_act = trained_task.network.rnn(
                                    bg_act_consolidation, inputs=rnn_inputs
                                )
                                output_store[ti] = r_act @ trained_task.network.Wout

                            loss_consolidation = (
                                (output_store - consolidation_target_store) ** 2
                            ).sum(axis=[1, 2]).mean() + wd * torch.linalg.norm(means)
                            loss_consolidation.backward()
                            optimizer.step()
                            loss_store.append(loss_consolidation.item())
                            # means = torch.zeros(n_clusters, latent_dim)

            # color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
            # fig, ax = plt.subplots()
            # for idx, (loss_data, vars, label) in enumerate(
            #     zip(
            #         [loss, loss_no_bg],
            #         [loss_var, loss_var_no_bg],
            #         ["BG", "No BG"],
            #     )
            # ):
            #     ax.scatter(
            #         range(len(loss_data)),
            #         loss_data,
            #         label=label,
            #         color=color_cycle[idx],
            #     )
            #     ax.plot(
            #         range(len(loss_data)), loss_data, color=color_cycle[idx], ls="--"
            #     )
            #     ax.fill_between(
            #         range(len(loss_data)),
            #         np.array(loss_data) - np.array(vars),
            #         np.array(loss_data) + np.array(vars),
            #         alpha=0.5,
            #         color=color_cycle[idx],
            #     )
            #
            # plt.legend()
            # ax.set_yscale("log")
            # ax.set_ylabel("Error")
            # cwd = Path(__file__).parent.parent.parent / "results/GenerateSinePL"
            # file_name = cwd / "rl_learning_curve_with_consolidation"
            # ax.set_xlabel("Training Epochs")
            # make_axis_nice(fig)
            # fig.savefig(file_name)
            # plt.pause(0.1)
            # pdb.set_trace()
