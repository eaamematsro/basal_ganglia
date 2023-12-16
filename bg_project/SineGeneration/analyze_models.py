import pdb
import pickle
import torch

import numpy as np
import matplotlib.pyplot as plt

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


def get_fixed_points(
    model,
    gain_vector: Optional[torch.Tensor] = None,
    n_slow_points: int = 10,
    max_training_epochs: int = 500000,
    output_min: float = 1e-1,
):
    initial_states = torch.nn.Parameter(
        1
        * torch.randn(
            (n_slow_points, model.network.Wout.shape[0]),
            device=model.network.Wout.device,
        )
    )
    optimizer = Adam([initial_states])
    output_norms_gained = []
    output_val = 1e4
    epoch = 0

    while (output_val > output_min) and (epoch < max_training_epochs):
        optimizer.zero_grad()
        output = model.network.one_step_update(initial_states, gain_vector)
        loss = ((output) ** 2).sum(axis=1).mean()
        output_norms_gained.append(loss.item())
        output_val = loss.item()
        epoch += 1
        loss.backward()
        optimizer.step()

    return model.network.rnn.nonlinearity(initial_states).detach().cpu().numpy()


def estimate_flow_field(
    model,
    pca_fit,
    gain_vector,
    n_divs: int = 15,
    max_val: float = 40,
    min_val: float = -20,
    ndims: int = 2,
    trajectory: Optional = None,
    axis: Optional = None,
):
    """Estimate the local flow field around a set of trajectories"""

    # Need to add means back

    assert ndims in [2, 3]
    pc_vals = []
    for dim in range(ndims):
        pc_vals.append(np.linspace(min_val, max_val * (1 / 2) ** dim, n_divs))

    all_pcs = np.zeros((n_divs**ndims, pca_fit.n_components_))
    gain_vector = torch.tile(gain_vector[0][None, :], dims=(n_divs**ndims, 1))
    for idx, pcs in enumerate(product(*pc_vals)):
        all_pcs[idx, :ndims] = pcs
    full_d = pca_fit.inverse_transform(all_pcs)
    initial_state = torchify(full_d)
    with torch.no_grad():
        flows = model.network.one_step_update(initial_state, gain_vector).cpu().numpy()
        base_flows = model.network.one_step_update(initial_state).cpu().numpy()

    latent_flows = (components[:ndims] @ flows.T).T
    base_latent_flows = (components[:ndims] @ base_flows.T).T

    delta_flow = latent_flows - base_latent_flows

    if ndims == 3:
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    else:
        fig, ax = plt.subplots()

    if ndims == 3:
        plot_freq = 1
        sample_index = np.random.choice(
            all_pcs.shape[0],
            int(all_pcs.shape[0] / plot_freq),
            replace=False,
        )
        ax.quiver(
            all_pcs[sample_index, 0],
            all_pcs[sample_index, 1],
            all_pcs[sample_index, 2],
            latent_flows[sample_index, 0],
            latent_flows[sample_index, 1],
            latent_flows[sample_index, 2],
            color="k",
            arrow_length_ratio=0.5,
            # length=1,
        )
        if trajectory is not None:
            ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], lw=2)
            ax.scatter(
                trajectory[0, 0],
                trajectory[0, 1],
                trajectory[0, 2],
                color="green",
                s=32,
            )

    else:
        ax.quiver(
            all_pcs[:, 0],
            all_pcs[:, 1],
            latent_flows[:, 0],
            latent_flows[:, 1],
            color="k",
        )

        if trajectory is not None:
            ax.plot(trajectory[:, 0], trajectory[:, 1], lw=2)
            ax.scatter(
                trajectory[0, 0],
                trajectory[0, 1],
                color="green",
                s=32,
            )
            ax.scatter(
                trajectory[-1, 0],
                trajectory[-1, 1],
                color="red",
                s=32,
            )

    if axis is not None:
        axis.quiver(
            all_pcs[:, 0],
            all_pcs[:, 1],
            delta_flow[:, 0],
            delta_flow[:, 1],
            # color=cmap(norm(delta_flow)),
        )

    return fig


def estimate_perturbed_flows(
    model,
    base_activity,
    pca_fit,
    gain_vector,
    noise_scale: float = 0.25,
    num_iterations: int = 50,
    ndims: int = 2,
    axis: Optional = None,
    plot_freq: int = 50,
):
    based_hidden_activity = torchify(base_activity)
    perturbed_activities = []
    flows = []
    gain_vector = torch.tile(gain_vector[0][None, :], dims=(base_activity.shape[0], 1))

    for trial in range(num_iterations):
        perturbed_activity = based_hidden_activity + (
            noise_scale * torch.sqrt(torch.abs(based_hidden_activity)) + 1
        ) * torch.randn_like(based_hidden_activity)
        with torch.no_grad():
            resultant_flow = (
                model.network.one_step_update(perturbed_activity, gain_vector)
                .cpu()
                .numpy()
            )
        perturbed_activities.append(perturbed_activity.cpu().numpy())
        flows.append(resultant_flow)

    stacked_flows = np.vstack(flows)
    stacked_inits = np.vstack(perturbed_activities)

    latent_flows = pca_fit.transform(stacked_flows)
    latent_inits = pca_fit.transform(stacked_inits)

    sample_index = np.random.choice(
        latent_flows.shape[0],
        int(latent_flows.shape[0] / plot_freq),
        replace=False,
    )
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.quiver(
        latent_inits[sample_index, 0],
        latent_inits[sample_index, 1],
        latent_inits[sample_index, 2],
        latent_flows[sample_index, 0],
        latent_flows[sample_index, 1],
        latent_flows[sample_index, 2],
    )
    plt.pause(0.1)
    pdb.set_trace()


def update_projection(ax, axi, projection="rectilinear", fig=None):
    if fig is None:
        fig = plt.gcf()
    rows, cols, start, stop = axi.get_subplotspec().get_geometry()
    ax.flat[start].remove()
    ax.flat[start] = fig.add_subplot(rows, cols, start + 1, projection=projection)


if __name__ == "__main__":

    project_3d = False
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
        "RNNMultiContextInput",
    ]
    target_amplitudes = (0.5, 1.5)
    target_frequencies = (1.5, 0.75)
    max_components = 40
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
            if trained_task.network.params["network"] in allowed_networks:
                if trained_task.param_normalizers is not None:
                    task_parameters = np.zeros((len(trained_task.cluster_labels), 2))
                    for (amp, freq), value in trained_task.cluster_labels.items():
                        task_parameters[value] = [amp, freq]

                    # get all combinations that have a matching amplitude

                    same_amp_idx = [
                        (idx1, idx2)
                        for idx1, idx2 in combinations(
                            range(task_parameters.shape[0]), 2
                        )
                        if task_parameters[idx1, 0] == task_parameters[idx2, 0]
                    ]
                    # get all combinations that have a matching frequency

                    same_freq_idx = [
                        (idx1, idx2)
                        for idx1, idx2 in combinations(
                            range(task_parameters.shape[0]), 2
                        )
                        if task_parameters[idx1, 1] == task_parameters[idx2, 1]
                    ]
                    (
                        outputs,
                        activity,
                        hidden_activity,
                    ) = trained_task.evaluate_network_clusters(go_cues)

                    # create N x CT matrix
                    stacked_activity = hidden_activity.transpose(2, 1, 0).reshape(
                        hidden_activity.shape[-1], -1
                    )
                    neural_pca = PCA()
                    neural_pca.fit(stacked_activity.T)

                    all_trajectories = neural_pca.transform(stacked_activity.T)

                    # get principal components [component, features]

                    components = neural_pca.components_

                    inverse_matrix = components.T
                    var_activity = np.var(activity, axis=0).mean(axis=0)
                    prob_neuron = var_activity / var_activity.sum()

                    condition_components = []
                    readout = trained_task.network.Wout.cpu().numpy()
                    readout_statistics = np.zeros((task_parameters.shape[0], 3))
                    for idx in range(activity.shape[1]):
                        condition_pca = PCA()
                        condition_pca.fit(hidden_activity[:, idx])
                        condition_components.append(condition_pca.components_)

                        overlap_with_output = np.min(
                            np.rad2deg(
                                linalg.subspace_angles(
                                    condition_pca.components_[:max_components].T,
                                    readout,
                                )
                            )
                        )
                        readout_statistics[idx] = [
                            overlap_with_output,
                            task_parameters[idx, 0] / trained_task.param_normalizers[0],
                            task_parameters[idx, 1] / trained_task.param_normalizers[1],
                        ]

                    fig_overlap, ax_overlap = plt.subplots(1, 2, sharey="row")

                    ax_overlap[0].scatter(
                        readout_statistics[:, 1], readout_statistics[:, 0]
                    )
                    ax_overlap[1].scatter(
                        readout_statistics[:, 2], readout_statistics[:, 0]
                    )
                    ax_overlap[0].set_xlabel("Amplitude")
                    ax_overlap[1].set_xlabel("Frequency")
                    fig_overlap.supylabel("Readout Overlap")
                    fig_overlap.tight_layout()
                    make_axis_nice(fig_overlap)
                    file_name = date_results_path / "readout_alignment_plot"
                    fig_overlap.savefig(file_name)

                    if project_3d:
                        fig, ax = plt.subplots(
                            4,
                            ncols=10,
                            figsize=(20, 24),
                            sharex="row",
                            sharey="row",
                            subplot_kw={"projection": "3d"},
                        )
                    else:
                        fig, ax = plt.subplots(
                            3,
                            ncols=10,
                            figsize=(20, 24),
                            sharex="row",
                            sharey="row",
                        )
                    neurons = np.random.choice(
                        activity.shape[2], neurons_to_plot, p=prob_neuron, replace=False
                    )
                    for idx, axes in enumerate(ax.T):
                        # if idx == 0:
                        #     axes[0].set_title("Behavior")
                        #     axes[1].set_title("Neural Activity")
                        cluster_prob = torch.zeros(
                            (outputs.shape[1]), device=trained_task.network.Wout.device
                        )
                        cluster_prob[idx] = 1
                        with torch.no_grad():
                            gain_vector = trained_task.network.bg(cluster_prob)

                        # slow_points = get_fixed_points(trained_task, gain_vector)

                        if project_3d:
                            for axis in axes[[0, 1, 3]]:
                                update_projection(ax, axis, fig=fig)

                        axes[0].plot(go_cues[0, 0], label="Go", c="green", ls="--")
                        axes[0].plot(go_cues[0, 1], label="Stop", c="red", ls="--")
                        axes[0].plot(
                            outputs[:, idx],
                        )
                        axes[0].set_ylim([-1.75, 1.75])
                        axes[1].plot(go_cues[0, 0], label="Go", c="green", ls="--")
                        axes[1].plot(go_cues[0, 1], label="Stop", c="red", ls="--")
                        axes[1].plot(activity[:, idx, neurons])
                        # projections = components[:3] @ hidden_activity[:, idx].T
                        projections = neural_pca.transform(hidden_activity[:, idx])
                        fig_flow = estimate_flow_field(
                            trained_task,
                            neural_pca,
                            gain_vector,
                            axis=axes[2],
                            trajectory=projections[:, :3],
                            min_val=all_trajectories.min(),
                            max_val=all_trajectories.max(),
                        )
                        make_axis_nice(fig_flow)
                        fig_flow.supxlabel("PC1")
                        fig_flow.supylabel("PC2")
                        file_name = date_results_path / f"solution_flowfield_plot_{idx}"
                        fig_overlap.savefig(file_name)
                        # slow_point_projections = components[:3] @ slow_points.T
                        # if project_3d:
                        #     axes[2].plot(projections[0], projections[1], projections[2])
                        # else:
                        #     axes[2].plot(projections[0], projections[1])
                        #     # axes[2].scatter(
                        #     #     slow_point_projections[0],
                        #     #     slow_point_projections[1],
                        #     #     color="red",
                        #     # )

                    make_axis_nice(fig)
                    excursion_means = []
                    comparisons = []

                    for (idx1, components_1), (idx2, components_2) in combinations(
                        enumerate(condition_components), 2
                    ):
                        means = []

                        for included_components in np.arange(1, max_components):
                            subspace_angles = linalg.subspace_angles(
                                components_1[:included_components].T,
                                components_2[:included_components].T,
                            )
                            means.append(np.rad2deg(subspace_angles).mean())
                        excursion_means.append(np.asarray(means))
                        comparisons.append((idx1, idx2))

                    idx_amp_match = [
                        idx
                        for idx, value in enumerate(comparisons)
                        if value in same_amp_idx
                    ]

                    idx_freq_match = [
                        idx
                        for idx, value in enumerate(comparisons)
                        if value in same_freq_idx
                    ]

                    excursions = np.vstack(excursion_means)
                    mean_values = excursions.mean(axis=0)
                    deviations = excursions.std(axis=0)

                    mean_amp_match = excursions[idx_amp_match].mean(axis=0)
                    deviations_amp_match = excursions[idx_amp_match].std(axis=0)

                    mean_freq_match = excursions[idx_freq_match].mean(axis=0)
                    deviations_freq_match = excursions[idx_freq_match].std(axis=0)

                    fig_alignment, ax_alignment = plt.subplots()
                    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

                    for idx, (mean, variance, label) in enumerate(
                        zip(
                            [mean_values, mean_amp_match, mean_freq_match],
                            [deviations, deviations_amp_match, deviations_freq_match],
                            ["All", "Matched Amplitude", "Matched Frequency"],
                        )
                    ):
                        ax_alignment.plot(
                            mean, label=label, color=color_cycle[idx], ls="--"
                        )
                        ax_alignment.fill_between(
                            range(mean.shape[0]),
                            mean - variance,
                            mean + variance,
                            color=color_cycle[idx],
                            alpha=0.5,
                        )

                    fig_alignment.legend()
                    ax_alignment.set_ylim([0, 90])
                    ax_alignment.set_xlabel("Number of Components")
                    ax_alignment.set_ylabel("Average Excursion Angle")
                    make_axis_nice(fig_alignment)
                    file_name = date_results_path / "alignment_plot"
                    fig_alignment.savefig(file_name)

                    fig.tight_layout()
                    file_name = date_results_path / "behavior_plot"
                    make_axis_nice(fig)
                    fig.savefig(file_name)
                    (
                        parameters,
                        cluster_ids,
                        cluster_centers,
                    ) = trained_task.get_cluster_means()
                    pairwise_distance = pairwise_distances(cluster_centers)
                    pairwise_distance_store.append(
                        pairwise_distance / pairwise_distance.max()
                    )
                    parameter_store.append(parameters)
                    augmented_data_matrix = np.zeros(
                        (np.product(pairwise_distance.shape), 5)
                    )

                    for idx, (row, col) in enumerate(
                        product(
                            range(pairwise_distance.shape[0]),
                            range(pairwise_distance.shape[1]),
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
                    abline(
                        freq_coeff[1] / freq_coeff[0],
                        label="Freq Encoding",
                        intercept=0,
                    )

                    plt.xlabel("PC1")
                    plt.ylabel("PC2")
                    plt.legend()
                    file_name = date_results_path / "PCA_encoding_plot"
                    plt.savefig(file_name)
                    plt.pause(0.1)
                    pdb.set_trace()
                    plt.close("all")

    plt.figure()
    plt.hist(frequency_score, density=True)
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
    # fig, ax = plt.subplots()
    # processed_data = np.zeros((grouped_data.shape[0], 3))
    # processed_data[:, 0] = grouped_data[:, 0]
    # processed_data[:, 1] = (grouped_data[:, 1] - grouped_data[:, 3]) ** 2
    # processed_data[:, 2] = (grouped_data[:, 2] - grouped_data[:, 4]) ** 2
    # norm = Normalize(vmin=0, vmax=1)
    # g = ax.scatter(
    #     processed_data[:, 1],
    #     processed_data[:, 2],
    #     c=processed_data[:, 0],
    #     norm=norm,
    #     cmap="copper",
    # )
    # ax.set_xlabel("Amplitude Distance")
    # ax.set_ylabel("Frequency Distance")
    # plt.colorbar(g, ax=ax, label="Normed Euclidean Distance")
    # file_name = results_path / "Distance_representation"
    # plt.savefig(file_name)
    # plt.pause(0.1)
    pdb.set_trace()
