import pdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from SineGeneration.analyze_models_sl import make_axis_nice, set_plt_params
from pathlib import Path


if __name__ == "__main__":
    current_dir = Path(__file__).parent
    csv_file = current_dir / "model_dataframe.csv"

    results = pd.read_csv(csv_file, index_col=None)

    set_plt_params()

    grouped = results.groupby(
        [
            "n_hidden",
            "polarity",
            "mass",
            "viscosity",
            "spring_constant",
            "learning_style",
            "n_params",
        ]
    )
    test = grouped["loss"].mean()
    mean_pd = test.reset_index()
    # r_pd = r_pd.rename(columns={"loss": f"{net_type}_loss"})

    plt_params = {
        "x": "n_params",
        "y": "loss",
        "hue": "learning_style",
        "data": mean_pd,
    }

    g = sns.lineplot(**plt_params)
    sns.scatterplot(**plt_params)
    g.set(xscale="log")
    make_axis_nice()
    g = sns.FacetGrid(
        row="learning_style", data=mean_pd.loc[mean_pd["spring_constant"] == 0.5]
    )
    g.map(
        sns.scatterplot,
        x="mass",
        y="viscosity",
        data=mean_pd.loc[mean_pd["spring_constant"] == 0.5],
        hue="loss",
    )
    plt.pause(0.1)
    learning_options = np.unique(mean_pd["learning_style"].tolist())
    n_opts = len(learning_options)
    fig, ax = plt.subplots(n_opts, sharex=True, sharey=True)
    for idx, learning_style in enumerate(learning_options):
        reduced_data = mean_pd.loc[
            (mean_pd["learning_style"] == learning_style)
            & (mean_pd["spring_constant"] == 0.5)
        ][["mass", "viscosity", "spring_constant", "loss"]].values
        ax[idx].set_title(learning_style)
        ax[idx].hexbin(
            reduced_data[:, 0],
            reduced_data[:, 1],
            C=reduced_data[:, -1],
            vmin=0,
            vmax=3,
            xscale="log",
        )
        # ax[idx].set_xscale("log")
        # ax[idx].set_yscale("log")
    ax[-1].set_xlabel("Mass")
    ax[-1].set_ylabel("Viscosity")
    make_axis_nice(fig)
    fig.tight_layout()
    plt.pause(0.1)
    pdb.set_trace()
