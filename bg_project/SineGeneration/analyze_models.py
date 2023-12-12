import pdb
import pickle
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from datasets.loaders import SineDataset
from pacman.multigain_pacman import split_dataset
from torch.utils.data import DataLoader, RandomSampler, random_split
from itertools import product
from matplotlib.colors import Normalize
from typing import Tuple, Union, List


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


if __name__ == "__main__":

    task = "SineGeneration"

    cwd = Path().cwd()
    data_path = cwd / f"data/models/{task}"

    date_folders = [x for x in data_path.iterdir() if x.is_dir()]

    folders = [[x for x in folder.iterdir() if x.is_dir()] for folder in date_folders]

    set_plt_params()
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
            trained_task.network.rnn.reset_state(batch_size=10)

            if hasattr(trained_task, "results_path"):
                date_results_path = trained_task.results_path

            outputs, activity = trained_task.evaluate_network_clusters(go_cues)
            var_activity = np.var(activity, axis=0).mean(axis=0)
            prob_neuron = var_activity / var_activity.sum()
            fig, ax = plt.subplots(10, ncols=2, figsize=(12, 20), sharex="col")
            neurons = np.random.choice(
                activity.shape[2], neurons_to_plot, p=prob_neuron, replace=False
            )
            for idx, axes in enumerate(ax):
                if idx == 0:
                    axes[0].set_title("Behavior")
                    axes[1].set_title("Neural Activity")
                axes[0].plot(go_cues[0, 0], label="Go", c="green", ls="--")
                axes[0].plot(go_cues[0, 1], label="Stop", c="red", ls="--")
                axes[0].plot(
                    outputs[:, idx],
                )
                axes[0].set_ylim([-1.75, 1.75])
                axes[1].plot(go_cues[0, 0], label="Go", c="green", ls="--")
                axes[1].plot(go_cues[0, 1], label="Stop", c="red", ls="--")
                axes[1].plot(activity[:, idx, neurons])
            ax[-1, 0].set_xlabel("Time")
            ax[-1, 1].set_xlabel("Time")

            fig.tight_layout()
            file_name = date_results_path / "behavior_plot"
            make_axis_nice(fig)
            fig.savefig(file_name)
            plt.pause(0.1)
            pdb.set_trace()
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
            gains = np.linspace(-1, 1)
            # amp_model = Ridge()
            # amp_model.fit(transformed[:, :2] ** 2, parameters[:, 0] ** 2)
            # amp_coeff = amp_model.coef_ / np.linalg.norm(amp_model.coef_)
            # amp_score = amp_model.score(transformed[:, :2], parameters[:, 0])

            freq_model = Ridge()
            freq_model.fit(transformed[:, :2], parameters[:, 1])
            freq_score = freq_model.score(transformed[:, :2], parameters[:, 1])
            freq_coeff = freq_model.coef_ / np.linalg.norm(freq_model.coef_)
            frequency_score.append(freq_score)
            pc1_freq.append(freq_coeff[0] ** 2 / (freq_coeff**2).sum())
            freq_vec = gains[:, None] * freq_coeff
            plt.figure()
            plt.title(f"{freq_score: 0.3f}")

            plt.scatter(
                transformed[:, 0],
                transformed[:, 1],
                c=linear_map(parameters, n_dimensions=1),
            )
            abline(freq_coeff[1] / freq_coeff[0], label="Freq Encoding", intercept=0)

            plt.xlabel("PC1")
            plt.ylabel("PC2")
            plt.legend()
            file_name = date_results_path / "PCA_encoding_plot"
            plt.savefig(file_name)
            plt.pause(0.1)
            plt.close("all")

    plt.figure()
    plt.hist(freq_score, density=True)
    plt.xlabel("Goodness of Fit")
    file_name = results_path / "goodness_of_fit_dist"
    plt.savefig(file_name)
    plt.pause(0.1)

    plt.figure()
    plt.hist(pc1_freq, density=True)
    plt.xlabel("PC1 Fractional Contribution")
    file_name = results_path / "Fractional_contribution"
    plt.savefig(file_name)
    plt.pause(0.1)

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
    file_name = results_path / "Distance_representation"
    plt.savefig(file_name)
    plt.pause(0.1)
    pdb.set_trace()
