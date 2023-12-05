import pdb
import pickle

import matplotlib.pyplot as plt

from pathlib import Path

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
for file_path in model_store_paths:
    model_path = file_path / "model.pickle"
    if model_path.exists():
        with open(model_path, "rb") as h:
            pickled_data = pickle.load(h)
        trained_task = pickled_data["task"]

        parameters, cluster_ids, cluster_centers = trained_task.get_cluster_means(
            amplitudes=target_amplitudes, frequencies=target_frequencies
        )
        pdb.set_trace()
