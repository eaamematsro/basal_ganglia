import pdb

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from SineGeneration.analyze_models_sl import make_axis_nice, set_plt_params


def non_linearity(x, k: float = 100, amp: float = 1):
    return x  # np.maximum(0, x)


def non_linearity2(x, k: float = 100, amp: float = 1):
    return np.tanh(x)


n_trials, nsamples, ndim = 500, 5000, 250
hidden_dim = 100
ground_truth = []
estimates = []
for trial in range(n_trials):
    W = np.random.randn(1, ndim)
    g_test = np.random.rand()
    xs = np.random.randn(ndim, nsamples)
    bg = non_linearity((1 - g_test) * W @ xs + np.random.randn(1, nsamples))
    th = W @ xs
    g_estimate = 1 - np.dot(bg, th.T) / (np.dot(th, th.T))
    ground_truth.append(g_test)
    estimates.append(g_estimate)


save_path = Path(__file__).parent
set_plt_params()
plt.figure()
plt.scatter(ground_truth, estimates)
plt.xlabel("Ground Truth")
plt.ylabel("Model Estimate")
make_axis_nice()
file_name = save_path / "g_estimate"
plt.savefig(file_name)
plt.pause(0.1)
