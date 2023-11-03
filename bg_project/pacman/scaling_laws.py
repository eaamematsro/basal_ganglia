import pdb
import pickle
import json

import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from datasets.tasks import MultiGainPacMan, Task
from multigain_pacman import split_dataset
from datasets.loaders import PacmanDataset
from torch.utils.data import DataLoader, RandomSampler, random_split
from pytorch_lightning import Trainer


def get_testing_loss(dataset: DataLoader, model: MultiGainPacMan, duration: int = 150):
    """Evaluate a trained model on some heldout testing data

    Args:
        dataset: Dataloader object containig testing data
        model: Trained task model to evaluate performance on.
        duration: Duration of a trial

    Returns:
        mean_loss: Testing loss averaged over batches.
    """
    losses = []
    model.duration = duration
    for batch in dataset:
        x, y = batch
        with torch.no_grad():
            positions, _ = model.forward(x.T, y.T, noise_scale=0)
            loss = model.compute_loss(y.T, positions.squeeze())
        losses.append(loss['trajectory'].numpy())
    mean_loss = np.mean(losses)
    return mean_loss


task = "MultiGainPacMan"

test_set = DataLoader(PacmanDataset(), batch_size=64)

cwd = Path().cwd()
data_path = cwd / f"data/models/{task}"


date_folders = [
    x for x in data_path.iterdir()
    if x.is_dir()
]

folders = [[
    x for x in folder.iterdir()
    if x.is_dir()] for folder in date_folders
]

model_store_paths = []
for data in folders:
    model_store_paths.extend(data)

training_outputs = []
for file_path in model_store_paths:
    model_path = file_path / "model.pickle"
    if model_path.exists():
        with open(model_path, 'rb') as h:
            pickled_data = pickle.load(h)
        params_path = file_path / "params.json"
        with open(params_path, 'rb') as h:
            model_params = json.load(h)
        trained_model = pickled_data['task']
        test_loss = get_testing_loss(test_set, trained_model)
        model_params.update(
            {
                'n_params': trained_model.count_parameters(),
                'loss': test_loss
            })
        training_outputs.append(model_params)

results = pd.DataFrame(training_outputs)
plt_params = {
    'data': results,
    'x': 'n_params',
    'y': 'loss',
    'hue': 'network'
}
sns.lineplot(**plt_params)
plt.xscale('log')
plt.pause(0.01)
pdb.set_trace()