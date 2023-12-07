import pdb
import pickle
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
from datasets.loaders import SineDataset
from pacman.multigain_pacman import split_dataset
from torch.utils.data import DataLoader, RandomSampler, random_split
from itertools import product
from matplotlib.colors import Normalize


def linear_map(data: np.ndarray, n_dimensions: int = 2):
    color_scales = np.zeros((data.shape[0], 3))

    for dim in range(n_dimensions):
        dim_max = data[:, dim].max()
        color_scales[:, dim] = data[:, dim] / dim_max

    return color_scales


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

# amp_norm = 1 / np.std(target_amplitudes)
# freq_norm = 1 / np.std(target_amplitudes)
#
# target_amplitudes = tuple(value * freq_norm for value in target_frequencies)
# target_frequencies = tuple(value * freq_norm for value in target_frequencies)


pairwise_distance_store = []
parameter_store = []
cluster_center = []
data_store = []
duration = 300
dataset = SineDataset(
    duration=duration,
)
train_set, val_set, test_set = split_dataset(dataset, (0.6, 0.2, 0.2))
val_loader = DataLoader(val_set["data"], batch_size=10, num_workers=10)
(go_cues, contexts), targets = next(iter(val_loader))
for file_path in model_store_paths:
    model_path = file_path / "model.pickle"
    if model_path.exists():
        with open(model_path, "rb") as h:
            pickled_data = pickle.load(h)
        trained_task = pickled_data["task"]
        trained_task.network.rnn.reset_state(batch_size=10)
        outputs = trained_task.evaluate_network_clusters(go_cues)
        fig, ax = plt.subplots(10, figsize=(12, 20), sharex="col")
        for idx, axes in enumerate(ax):
            axes.plot(go_cues[0, 0], label="Go", c="green", ls="--")
            axes.plot(go_cues[0, 1], label="Stop", c="red", ls="--")
            axes.plot(
                outputs[:, idx],
            )
            axes.set_ylim([-1.75, 1.75])
        ax[-1].set_xlabel("Time")
        fig.tight_layout()
        plt.pause(0.1)
        parameters, cluster_ids, cluster_centers = trained_task.get_cluster_means()
        pairwise_distance = pairwise_distances(cluster_centers)
        pairwise_distance_store.append(pairwise_distance / pairwise_distance.max())
        parameter_store.append(parameters)
        augmented_data_matrix = np.zeros((np.product(pairwise_distance.shape), 5))

        for idx, (row, col) in enumerate(
            product(
                range(pairwise_distance.shape[0]), range(pairwise_distance.shape[1])
            )
        ):
            augmented_data_matrix[idx, 0] = (
                pairwise_distance[row, col] / pairwise_distance.max()
            )
            augmented_data_matrix[idx, 1:3] = parameters[row]
            augmented_data_matrix[idx, 3:] = parameters[col]

        test = np.hstack(
            [
                (pairwise_distance / pairwise_distance.max()).flatten(),
                parameters.flatten(),
            ]
        )
        data_store.append(augmented_data_matrix)
        center_pca = PCA()
        transformed = center_pca.fit_transform(cluster_centers)
        plt.figure()
        plt.scatter(
            transformed[:, 0],
            transformed[:, 1],
            c=linear_map(parameters, n_dimensions=1),
        )
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.pause(0.1)
        pdb.set_trace()


grouped_data = np.vstack(data_store)
fig, ax = plt.subplots()
processed_data = np.zeros((grouped_data.shape[0], 3))
processed_data[:, 0] = grouped_data[:, 0]
processed_data[:, 1] = (grouped_data[:, 1] - grouped_data[:, 3]) ** 2
processed_data[:, 2] = (grouped_data[:, 2] - grouped_data[:, 4]) ** 2
norm = Normalize(vmin=0, vmax=1)
g = ax.scatter(
    processed_data[:, 1],
    processed_data[:, 2],
    c=processed_data[:, 0],
    norm=norm,
    cmap="copper",
)
ax.set_xlabel("Amplitude Distance")
ax.set_ylabel("Frequency Distance")
plt.colorbar(g, ax=ax, label="Normed Euclidean Distance")
plt.pause(0.1)
pdb.set_trace()
