import pdb
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from scipy import linalg
from torch.optim import Adam
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from datasets.loaders import SineDataset
from pacman.multigain_pacman import split_dataset
from torch.utils.data import DataLoader, RandomSampler, random_split
from itertools import product, combinations
from matplotlib.colors import Normalize
from typing import Tuple, Union, List


if __name__ == "__main__":

    project_3d = False
    task = "SineGeneration"

    cwd = Path().cwd()
    data_path = cwd / f"data/models/{task}"

    date_folders = [x for x in data_path.iterdir() if x.is_dir()]

    folders = [[x for x in folder.iterdir() if x.is_dir()] for folder in date_folders]

    model_store_paths = []
    for data in folders:
        model_store_paths.extend(data)
    training_outputs = []
    allowed_networks = [
        "RNNGMM",
    ]
    target_amplitudes = (0.5, 1.5)
    target_frequencies = (1.5, 0.75)
    max_components = 40
    n_slow_points = 25
    training_steps = 2500
    # amp_norm = 1 / np.std(target_amplitudes)
    # freq_norm = 1 / np.std(target_amplitudes)
    #
    # target_amplitudes = tuple(value * freq_norm for value in target_frequencies)
    # target_frequencies = tuple(value * freq_norm for value in target_frequencies)

    pairwise_distance_store = []
    parameter_store = []
    cluster_center = []
    frequency_score = []
    pc1_freq = []
    data_store = []
    duration = 300
    neurons_to_plot = 5
    dataset = SineDataset(
        duration=duration,
    )
    train_set, val_set, test_set = split_dataset(dataset, (0.6, 0.2, 0.2))
    val_loader = DataLoader(val_set["data"], batch_size=50, num_workers=10)
    (go_cues, contexts), targets = next(iter(val_loader))
    results_path = Path(__file__).parent.parent.parent / "results/GenerateSinePL"
    for file_path in model_store_paths:
        model_path = file_path / "model.pickle"
        date_str = file_path.parent.resolve().stem
        date_results_path = results_path / date_str
        if model_path.exists():
            with open(model_path, "rb") as h:
                pickled_data = pickle.load(h)
            trained_task = pickled_data["task"]
            initial_states = torch.nn.Parameter(
                torch.randn(
                    (n_slow_points, trained_task.network.Wout.shape[0]),
                    device=trained_task.network.Wout.device,
                )
            )
            optimizer = Adam([initial_states])
            output_norms = []
            for train_step in range(training_steps):
                optimizer.zero_grad()
                output = trained_task.network.one_step_update(initial_states)
                loss = ((output) ** 2).sum(axis=1).mean()
                output_norms.append(loss.item())
                loss.backward()
                optimizer.step()

            initial_states = torch.nn.Parameter(
                torch.randn(
                    (n_slow_points, trained_task.network.Wout.shape[0]),
                    device=trained_task.network.Wout.device,
                )
            )
            optimizer = Adam([initial_states])
            output_norms_gained = []
            gain_vector = 1 * torch.ones((n_slow_points, 10))
            for train_step in range(training_steps):
                optimizer.zero_grad()
                output = trained_task.network.one_step_update(
                    initial_states, gain_vector
                )
                loss = ((output) ** 2).sum(axis=1).mean()
                output_norms_gained.append(loss.item())
                loss.backward()
                optimizer.step()

            plt.figure()
            plt.scatter(range(len(output_norms)), output_norms)
            plt.scatter(range(len(output_norms)), output_norms_gained)

            plt.pause(0.1)
            pdb.set_trace()
