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

        dist_pca = PCA()
        pdb.set_trace()
