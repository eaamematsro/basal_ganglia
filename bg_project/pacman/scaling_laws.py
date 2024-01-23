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
from matplotlib.colors import TwoSlopeNorm, LogNorm
from SineGeneration.analyze_models_sl import make_axis_nice, set_plt_params


def get_testing_loss(model: MultiGainPacMan):
    losses = model.network.test_loss
    return np.mean(losses)


# def get_testing_loss(dataset: DataLoader, model: MultiGainPacMan, duration: int = 150):
#     """Evaluate a trained model on some heldout testing data
#
#     Args:
#         dataset: Dataloader object containig testing data
#         model: Trained task model to evaluate performance on.
#         duration: Duration of a trial
#
#     Returns:
#         mean_loss: Testing loss averaged over batches.
#     """
#     losses = []
#     model.duration = duration
#     for batch in dataset:
#         x, y = batch
#         with torch.no_grad():
#             positions, _ = model.forward(x.T, y.T, noise_scale=0)
#             loss = model.compute_loss(y.T, positions.squeeze())
#         losses.append(loss["trajectory"].numpy())
#     mean_loss = np.mean(losses)
#     return mean_loss


task = "MultiGainPacMan"

test_set = DataLoader(PacmanDataset(), batch_size=64)

cwd = Path().cwd()
data_path = cwd / f"data/models/{task}"
hsave_path = Path(__file__).parent / "results"
hsave_path.mkdir(exist_ok=True)


date_folders = [x for x in data_path.iterdir() if x.is_dir()]

folders = [[x for x in folder.iterdir() if x.is_dir()] for folder in date_folders]

model_store_paths = []
for data in folders:
    model_store_paths.extend(data)
training_outputs = []
allowed_networks = ["RNNStaticBG", "VanillaRNN"]
skip_plots = True
for file_path in model_store_paths:
    model_path = file_path / "model.pickle"
    if model_path.exists():
        with open(model_path, "rb") as h:
            pickled_data = pickle.load(h)
        params_path = file_path / "params.json"
        fig_folder = file_path / "results"
        fig_folder.mkdir(exist_ok=True)
        with open(params_path, "rb") as h:
            model_params = json.load(h)
        model_dataset = PacmanDataset(
            polarity=(model_params["polarity"],),
            viscosity=(model_params["viscosity"],),
            masses=(model_params["mass"],),
        )

        og_dataset = PacmanDataset(
            polarity=(1,),
            viscosity=(0,),
            masses=(1,),
        )
        _, val_set, _ = split_dataset(model_dataset, (0.6, 0.2, 0.2))
        val_loader = DataLoader(val_set["data"], batch_size=5, num_workers=10)

        _, og_val_set, _ = split_dataset(og_dataset, (0.6, 0.2, 0.2))
        pert_val_loader = DataLoader(og_val_set["data"], batch_size=5, num_workers=10)

        if "original_model" in model_params.keys():

            trained_model = pickled_data["task"]
            original_model = model_params["original_model"]
            test_dataloader = trained_model.test_dataloader
            if not (hasattr(trained_model.network, "bg") or skip_plots):
                set_plt_params()
                og_loss = []
                for batch_idx, batch in enumerate(val_loader):
                    loss = trained_model.evaluate_training(batch, original_network=None)
                    make_axis_nice()
                    save_name = fig_folder / f"original_training_{batch_idx}"
                    plt.savefig(save_name)
                    plt.pause(0.1)
                    og_loss.append(loss)

                plt.close("all")
                pert_loss = []

                try:
                    for batch_idx, batch in enumerate(pert_val_loader):
                        loss = trained_model.evaluate_training(
                            batch, original_network=None
                        )
                        make_axis_nice()
                        save_name = fig_folder / f"perturbed_training_{batch_idx}"
                        plt.savefig(save_name)
                        plt.pause(0.1)
                        pert_loss.append(loss)
                except RuntimeError:
                    continue
                del og_dataset, model_dataset
                plt.close("all")

                test_loss = get_testing_loss(trained_model)
                model_params.update(
                    {
                        "n_params": trained_model.count_parameters(),
                        "loss": test_loss,
                        "original_loss": np.mean(og_loss),
                        "perturbed_loss": np.mean(pert_loss),
                    }
                )
            else:
                test_loss = get_testing_loss(trained_model)
                model_params.update(
                    {
                        "n_params": trained_model.count_parameters(),
                        "loss": test_loss,
                    }
                )
            training_outputs.append(model_params)
results = pd.DataFrame(training_outputs)
joined_pd = None
gains = [1, -1]
set_plt_params()
for gain in gains:
    for net_type in allowed_networks:
        grouped = results.loc[
            (results["network"] == net_type) & (results["polarity"] == gain)
        ].groupby(
            [
                "n_hidden",
                "polarity",
                "mass",
                "viscosity",
            ]
        )
        test = grouped["loss"].mean()
        r_pd = test.reset_index()
        r_pd = r_pd.rename(columns={"loss": f"{net_type}_loss"})
        if joined_pd is None:
            joined_pd = r_pd
        else:
            joined_pd = joined_pd.merge(r_pd)
    joined_pd["ratio"] = np.exp(
        joined_pd[f"{allowed_networks[1]}_loss"]
        - joined_pd[f"{allowed_networks[0]}_loss"]
    )
    div_norm = TwoSlopeNorm(vmin=0, vmax=2, vcenter=1)

    data = joined_pd[["mass", "viscosity", "ratio"]].values
    fig, ax = plt.subplots(1, 2)
    v_min, v_max = results["loss"].min(), results["loss"].max()
    for idx, net_type in enumerate(allowed_networks):
        network_data = joined_pd[["mass", "viscosity", f"{net_type}_loss"]].values
        ax[idx].set_title(net_type)
        g = ax[idx].hexbin(
            network_data[:, 0],
            network_data[:, 1],
            C=network_data[:, 2],
            cmap="seismic_r",
            xscale="log",
            norm=LogNorm(vmin=v_min, vmax=v_max),
        )
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(g, cax=cbar_ax)
    fig.tight_layout()
    make_axis_nice(fig)
    file_name = hsave_path / f"network_compare_gain_{gain}"
    fig.savefig(file_name)
    plt.figure()
    plt.scatter(1, 0, s=20, color="Green")
    g = plt.hexbin(
        data[:, 0],
        data[:, 1],
        C=data[:, 2],
        cmap="seismic_r",
        norm=div_norm,
        xscale="log",
    )
    plt.xlabel("Mass")
    plt.ylabel("Viscosity")
    plt.xscale("log")
    plt.colorbar(g, label="Loss Ratio", extend="max")
    make_axis_nice()
    file_name = hsave_path / f"loss_ratios_gain_{gain}"
    plt.savefig(file_name)
    plt.pause(0.1)

    pdb.set_trace()
# plt_params = {"data": joined_pd, "x": "mass", "y": "viscosity", "hue": "ratio"}
# sns.scatterplot(**plt_params)
# # plt.xscale("log")
# plt.pause(0.01)
# pdb.set_trace()
