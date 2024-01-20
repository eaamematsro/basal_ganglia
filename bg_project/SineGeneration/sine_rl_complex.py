import pdb
import copy
import torch
import pickle
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datasets.loaders import SineDataset
from torch.utils.data import DataLoader, RandomSampler, random_split
from torch.optim import Adam
from model_factory.factory_utils import torchify
from model_factory.architectures import RNNGMM, RNNMultiContextInput, PallidalRNN
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


def gram_schmidt(matrix: np.ndarray):
    """Create an orthonormal matrix from a set of vectors

    Args:
        matrix: collection of vectors to use to create orthonormal matrix
            [vector, dimension]
    Returns:
        gram_matrix: orthonormal matrix spanning the space defined by the input vectors
    """
    nconstraints, dims = matrix.shape
    Y = []
    Y.append(matrix[0] / np.linalg.norm(matrix[0]))

    for idx in range(dims - 1):
        if idx + 1 < nconstraints:
            temp_vec = matrix[idx + 1]
        else:
            temp_vec = np.random.randn(dims)

        for orth_vector in Y:
            temp_vec -= (
                np.dot(temp_vec, orth_vector)
                / np.dot(orth_vector, orth_vector)
                * orth_vector
            )

        temp_vec /= np.linalg.norm(temp_vec)
        Y.append(temp_vec)

    gram_matrix = np.vstack(Y).T
    return gram_matrix


# TODO: Only train models that haven't already been trained


set_plt_params()

## Train network to produce 1 sine
reg_ex = "(model_)([\d]+)"
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

fraction = 0.6
train_set, val_set, test_set = split_dataset(dataset, (fraction, 0.8 - fraction, 0.2))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_loader = DataLoader(
    train_set["data"],
    batch_size=batch_size,
    sampler=train_set["sampler"],
    num_workers=10,
)

task = "SineGeneration"
allowed_networks = [
    "PallidalRNN",
]

cwd = Path().cwd()
data_path = cwd / f"data/models/{task}"

date_folders = [x for x in data_path.iterdir() if x.is_dir()]

folders = [[x for x in folder.iterdir() if x.is_dir()] for folder in date_folders]

set_plt_params()
model_store_paths = []
for data in folders:
    model_store_paths.extend(data)

train_bg = True
train_J = False
train_I = False

csv_file_path = Path(__file__).parent / "rl_complex_training.csv"

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

for model_path in model_store_paths:
    if model_path.exists():
        model_picke_path = model_path / "model.pickle"
        model_name = model_path.stem
        date = model_path.parent.stem
        reg_result = re.match(reg_ex, model_name)
        model_num = int(reg_result.group(2))
        spec = training_storer.loc[
            (training_storer["date"] == date)
            & (training_storer["model_num"] == model_num)
        ]
        # skip iteration if data already exists
        if len(spec) > 0:
            continue

        with open(model_picke_path, "rb") as h:
            pickled_data = pickle.load(h)

        trained_task: Union[RNNGMM, RNNMultiContextInput, PallidalRNN] = copy.deepcopy(
            pickled_data["task"]
        )  # load trained network in memory
        network_type = trained_task.network.params["network"]

        if network_type in allowed_networks:
            Vt = trained_task.network.rnn.Vt.detach().cpu().numpy()
            nneurons, latent_dim = Vt.shape
            gram_matrix = gram_schmidt(Vt.T).T
            gram_matrix = torchify(gram_matrix).to(trained_task.network.Wout.device)
            sigma = torch.zeros(
                (2, latent_dim, nneurons), device=trained_task.network.Wout.device
            )
            # for i in range(latent_dim):
            #     sigma[:, i, i] = 1

            trained_task.network.rnn.reset_state(batch_size=10)

            cov = 0.05 * torch.ones(
                latent_dim, nneurons, device=trained_task.network.Wout.device
            )
            beta = 0.9

            loss = []

            ## Train BG via RL ##
            if train_bg:
                for epoch in range(training_steps):
                    sigma[1] += cov * torch.randn(
                        latent_dim, nneurons, device=trained_task.network.Wout.device
                    )
                    epoch_errors = []
                    for batch in train_loader:
                        (timing_cues, contexts), y = batch
                        timing_cues = timing_cues.to(trained_task.network.Wout.device)
                        contexts = contexts.to(trained_task.network.Wout.device)
                        inputs = {"cues": timing_cues, "parameters": contexts}

                        position_store = torch.zeros(
                            duration,
                            2 * batch_size,
                            1,
                            device=trained_task.network.Wout.device,
                        )

                        parameters_amp = contexts[0, 0, 0].cpu().numpy().item()
                        parameters_freq = contexts[0, 1, 0].cpu().numpy().item()

                        with torch.no_grad():
                            for run in range(2):
                                trained_task.network.rnn.reset_state(batch_size)
                                trained_task.network.rnn.Wb.data = (
                                    sigma[run] @ gram_matrix
                                ).T
                                for ti in range(duration):
                                    rnn_inputs = {
                                        "cues": timing_cues[:, :, ti],
                                        "target_parameters": contexts[:, :, ti],
                                    }
                                    r_hidden, r_act = trained_task.network.rnn(
                                        inputs=rnn_inputs
                                    )
                                    position_store[ti, run] = (
                                        r_act @ trained_task.network.Wout
                                    )
                        error = (
                            (y.squeeze()[:, None] - position_store.squeeze()) ** 2
                        ).sum(axis=0)
                        if error[1] <= error[0]:
                            sigma[0] = beta * sigma[0] + (1 - beta) * sigma[1]
                        epoch_errors.append(np.min(error.numpy()))

                    loss.append(np.mean(epoch_errors))
                    epoch_data = {
                        "date": date,
                        "model_num": model_num,
                        "epoch_num": epoch,
                        "loss": np.mean(epoch_errors),
                        "network": network_type,
                        "pretrained": int(
                            trained_task.network.params["ncontexts"] == 2
                        ),
                        "train_fraction": fraction,
                    }

                    temp_frame = pd.DataFrame(epoch_data, index=[0])

                    training_storer = pd.concat(
                        [training_storer, temp_frame], ignore_index=True
                    )
                pdb.set_trace()
                training_storer.to_csv(csv_file_path, index=None)
