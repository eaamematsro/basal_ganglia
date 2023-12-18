import pdb

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from SineGeneration.analyze_models import set_plt_params, make_axis_nice


simple_rl_csv = Path(__file__).parent / "rl_simple_training.csv"

consolidation_rl_csv = Path(__file__).parent / "rl_consolidation_training.csv"
set_plt_params(style="fast")

results_path = Path(__file__).parent.parent.parent / "results/GenerateSinePL"

if simple_rl_csv.exists():
    simple_data = pd.read_csv(simple_rl_csv, index_col=None)

    plt_params = {
        "x": "epoch_num",
        "y": "loss",
        "hue": "network",
        "data": simple_data.loc[simple_data["pretrained"] == 1],
    }

    g = sns.lineplot(**plt_params)
    g.set_yscale("log")
    g.set_xlabel("Epoch")
    g.set_ylabel("Loss")
    make_axis_nice()
    file_name = results_path / f"rl_learning_curve"
    plt.savefig(file_name)
    plt.pause(0.1)
    pdb.set_trace()


if consolidation_rl_csv.exists():
    consolidation_data = pd.read_csv(consolidation_rl_csv, index_col=None)

    plt_params = {
        "x": "epoch_num",
        "y": "loss",
        "hue": "network",
        "data": consolidation_data.loc[consolidation_data["pretrained"] == 1],
    }

    g = sns.lineplot(**plt_params)
    g.set_yscale("log")
    g.set_xlabel("Epoch")
    g.set_ylabel("Loss")
    make_axis_nice()
    file_name = results_path / f"rl_learning_consolidation_curve"
    plt.savefig(file_name)
    plt.pause(0.1)
pdb.set_trace()
