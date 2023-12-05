import pdb
import pickle
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import pairwise_distances

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
target_frequencies = (0.75, 1.5)

amp_norm = 1 / np.std(target_amplitudes)
freq_norm = 1 / np.std(target_amplitudes)

target_amplitudes = tuple(value * freq_norm for value in target_frequencies)
target_frequencies = tuple(value * freq_norm for value in target_frequencies)
pairwise_distance_store = []
for file_path in model_store_paths:
    model_path = file_path / "model.pickle"
    if model_path.exists():
        with open(model_path, "rb") as h:
            pickled_data = pickle.load(h)
        trained_task = pickled_data["task"]

        parameters, cluster_ids, cluster_centers = trained_task.get_cluster_means(
            amplitudes=target_amplitudes, frequencies=target_frequencies
        )

        pairwise_distance = pairwise_distances(cluster_centers)
        pairwise_distance_store.append(pairwise_distance / pairwise_distance.max())
        pdb.set_trace()
