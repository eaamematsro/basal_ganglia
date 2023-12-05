import pdb

import torch
import numpy as np
import matplotlib.pyplot as plt
from datasets.loaders import SineDataset
from torch.utils.data import DataLoader, RandomSampler, random_split
from torch.optim import Adam
from model_factory.factory_utils import torchify
from typing import Sequence


def split_dataset(dataset, fractions: Sequence = (0.6, 0.2, 0.2)):
    train_set, val_set, test_set = random_split(dataset, fractions)
    train_sampler = RandomSampler(train_set)
    val_sampler = RandomSampler(train_set)
    train = {"data": train_set, "sampler": train_sampler}
    val = {"data": val_set, "sampler": val_sampler}

    return train, val, test_set


## Train network to produce 1 sine
duration = 300
training_steps = 500
batch_size = 64
dataset = SineDataset(duration=duration)
nneurons = 250
dt = 0.01
tau = 0.15
lr, wd = 1e-3, 1e-6
plot_freq = 25

train_set, val_set, test_set = split_dataset(dataset, (0.6, 0.2, 0.2))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nonlinearity = torch.nn.Tanh()
losses = []

## Define network parameters
J_rec = torchify(1.2 * np.random.randn(nneurons, nneurons) / np.sqrt(nneurons)).to(
    device
)
W_cue = torchify(np.random.randn(nneurons, 2)).to(device)
bias = torchify(np.random.randn(nneurons, 1) / np.sqrt(1)).to(device)
W_out = torchify(np.random.randn(1, nneurons) / np.sqrt(nneurons)).to(device)
W_context = torchify(np.random.randn(nneurons, 2) / np.sqrt(2)).to(device)

optimizer = Adam([J_rec, W_cue, W_out, bias, W_context], lr=lr, weight_decay=wd)
train_loader = DataLoader(
    train_set["data"],
    batch_size=batch_size,
    sampler=train_set["sampler"],
    num_workers=10,
)
for t_step in range(training_steps):
    for batch_idx, batch in enumerate(train_loader):
        optimizer.zero_grad()
        (timing_cues, contexts), targets = batch
        x = torch.randn((nneurons, timing_cues.shape[0]), requires_grad=True).to(
            device
        ) / np.sqrt(nneurons)
        r = nonlinearity(x)
        timing_cues = timing_cues.to(device)
        contexts = contexts.to(device)
        targets = targets.to(device)
        outputs = torch.zeros((timing_cues.shape[0], duration), device=device)
        for time in range(duration):
            x = x + dt / tau * (
                -x
                + J_rec @ r
                + W_cue @ timing_cues[:, :, time].T
                + W_context @ contexts[:, :, time].T
                + bias
                + 0.1 * torch.randn_like(x)
            )
            r = nonlinearity(x)
            outputs[:, time] = W_out @ r
        loss = ((outputs - targets) ** 2).sum(dim=1).mean()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    if (t_step % plot_freq) == 0:
        plt.close("all")
        fig, ax = plt.subplots(2, 1)
        ax[0].plot(targets[0].detach().cpu(), label="Target")
        ax[0].plot(outputs[0].detach().cpu())
        ax[1].plot(losses)
        plt.pause(0.1)
pdb.set_trace()
