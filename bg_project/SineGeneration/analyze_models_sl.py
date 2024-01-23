import pdb
import pickle
import torch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from scipy import linalg
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from datasets.loaders import SineDataset
from pacman.multigain_pacman import split_dataset
from model_factory.factory_utils import torchify
from torch.optim import Adam
from torch.utils.data import DataLoader, RandomSampler, random_split
from itertools import product, combinations
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from typing import Tuple, Union, List, Optional


def set_plt_params(
    font_size: int = 16,
    legend_size: int = 16,
    axes_label_size: int = 20,
    linewidth: float = 1.5,
    axes_title_size: int = 12,
    xtick_label_size: int = 16,
    ytick_label_size: int = 16,
    ticksize: float = 5,
    fig_title_size: int = 34,
    style: str = "fast",
    font: str = "avenir",
    file_format: str = "svg",
    fig_dpi: int = 500,
    figsize: Tuple[float, float] = (11, 8),
    auto_method: str = None,
    x_margin: float = None,
    y_margin: float = None,
    render_path: bool = False,
):
    """
    This function sets the plot parameters for a plot
    :param font_size:
    :param legend_size:
    :param axes_label_size:
    :param linewidth:
    :param axes_title_size:
    :param xtick_label_size:
    :param ytick_label_size:
    :param ticksize:
    :param fig_title_size:
    :param style:
    :param font:
    :param file_format:
    :param fig_dpi:
    :param figsize
    :return:
    """
    plt.rcParams.update(plt.rcParamsDefault)
    plt.style.use(style)
    plt.rcParams["savefig.format"] = file_format
    plt.rcParams["backend"] = file_format
    plt.rcParams["savefig.dpi"] = fig_dpi
    plt.rcParams["figure.figsize"] = figsize
    if not auto_method is None:
        plt.rcParams["axes.autolimit_mode"] = auto_method
    if not x_margin is None:
        assert (x_margin >= 0) & (x_margin <= 1)
        plt.rcParams["axes.xmargin"] = x_margin
    if not y_margin is None:
        assert (y_margin >= 0) & (y_margin <= 1)
        plt.rcParams["axes.ymargin"] = y_margin

    plt.rc("font", size=font_size)
    plt.rc("axes", titlesize=axes_title_size)
    plt.rc("axes", labelsize=axes_label_size)
    plt.rc("xtick", labelsize=xtick_label_size)
    plt.rc("ytick", labelsize=ytick_label_size)
    plt.rc("legend", fontsize=legend_size)
    plt.rc("figure", titlesize=fig_title_size)
    plt.rc("lines", linewidth=linewidth)
    plt.rcParams["font.family"] = font
    plt.rcParams["xtick.major.size"] = ticksize
    plt.rcParams["ytick.major.size"] = ticksize
    plt.rcParams["xtick.minor.size"] = ticksize / 3
    plt.rcParams["ytick.minor.size"] = ticksize / 3

    if not render_path:
        plt.rcParams["svg.fonttype"] = "none"

    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.left"] = True
    plt.rcParams["axes.spines.bottom"] = True


def make_axis_nice(
    ax: Union[plt.Axes, plt.Figure] = None,
    offset: int = 10,
    line_width: float = 0.5,
    spines: List = None,
    color: str = None,
):
    """Makes axis pretty
    This function modifies the x and y axis so that there is a vertical and horizontal gap between them
    Args:
        ax: This is the axis (axes) that need to be changed. When this argument is a fig all axes of the fig
        get modified
        offset: Size of the gap.
        line_width: Linew width of the new axes.
        spines:
        color:
    """
    if ax is None:
        ax = plt.gca()

    if spines is None:
        spines = ["left", "bottom"]
    if type(ax) == plt.Figure:
        ax_list = ax.axes
    else:
        ax_list = [ax]

    for ax in ax_list:
        for spine in spines:
            ax.spines[spine].set_linewidth(line_width)
            if color is not None:
                ax.spines[spine].set_color(color)
            ax.spines[spine].set_position(("outward", offset))


def linear_map(data: np.ndarray, n_dimensions: int = 2):
    color_scales = np.zeros((data.shape[0], 3))

    for dim in range(n_dimensions):
        dim_max = data[:, dim].max()
        color_scales[:, dim] = data[:, dim] / dim_max

    return color_scales


def abline(
    slope,
    intercept,
    color: str = "red",
    label: str = None,
    axes=None,
    ls: str = "--",
    exponential: bool = False,
    **kwargs,
):
    """Plot a line from slope and intercept"""
    if axes is None:
        axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    if not exponential:
        y_vals = intercept + slope * x_vals
    else:
        y_vals = np.exp(intercept + slope * x_vals)

    if label is None:
        axes.plot(x_vals, y_vals, ls=ls, color=color, **kwargs)
    else:
        axes.plot(x_vals, y_vals, ls=ls, label=label, color=color, **kwargs)


def param_counter(model: torch.nn.Module):
    count = 0
    for param in model.parameters():
        if param.requires_grad:
            count += np.prod(param.shape)
    return count


if __name__ == "__main__":
    date = "2024-01-22"
    task = "SineGeneration"
    path = Path().cwd()
    target_folder = path / "data/models" / task / date

    folders = [[x for x in target_folder.iterdir() if x.is_dir()]]
    model_store_paths = []
    for data in folders:
        model_store_paths.extend(data)

    window_size = 10

    run_dicts = []
    for file_path in model_store_paths:
        model_path = file_path / "model.pickle"
        if model_path.exists():
            with open(model_path, "rb") as h:
                pickled_data = pickle.load(h)
            trained_task = pickled_data["task"]
            params = trained_task.network.params
            param_count = param_counter(trained_task.network)
            model_loss = trained_task.test_loss
            print(model_path)
            batch_num = np.arange(len(model_loss))
            decay_rate, intercept = np.polyfit(batch_num, np.log(model_loss), 1)
            # if params.get("ncontexts", 1) == 2:
            #     plt.scatter(np.arange(len(model_loss)), model_loss)
            #     # plt.scatter(
            #     #     np.arange(len(model_loss)), np.exp(decay_rate * batch_num + intercept)
            #     # )
            #     #
            #     plt.pause(0.1)
            # pdb.set_trace()
            run_dict = {}

            run_dicts.extend(
                {
                    "train_type": params.get("train_type", "full"),
                    "final_loss": np.mean(model_loss[-window_size:]),
                    "loss": float(loss_value),
                    "grad_step": step,
                    "learning_rate": -decay_rate,
                    "number_of_parameters": param_count,
                    "nbg": params.get("nbg", 10),
                    "amplitude": params.get("amp", 1),
                    "frequency": params.get("freq", 1),
                    "pretrained": (params.get("ncontexts", 1) == 2),
                }
                for step, loss_value in enumerate(model_loss)
            )

    result_store = pd.DataFrame(run_dicts)
    set_plt_params()
    plt_params = {
        "x": "grad_step",
        "y": "loss",
        "hue": "train_type",
        "data": result_store.loc[result_store["pretrained"]],
    }
    plt.figure()
    sns.lineplot(**plt_params)
    plt.xlabel("Gradient Step")
    plt.xscale("log")
    plt.ylabel("Loss")
    make_axis_nice()
    plt.pause(0.1)

    plt_params = {
        "x": "number_of_parameters",
        "y": "final_loss",
        "hue": "train_type",
        "data": result_store.loc[result_store["pretrained"]],
    }
    plt.figure()
    sns.scatterplot(**plt_params)
    plt.xlabel("Number of Parameters")
    plt.xscale("log")
    plt.ylabel("Loss")
    make_axis_nice()
    plt.pause(0.1)
    pdb.set_trace()
