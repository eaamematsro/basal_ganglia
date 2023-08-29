import pdb
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
from model_factory.factory_utils import torchify
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import cdist
from typing import Sequence
from itertools import product


class SineDataset(Dataset):
    def __init__(self, n_unique_pulses: int = 250, pulse_width: int = 10,
                frequency: float = 1, delay: int = 0, amplitude: int = 1,
                duration: int = 500, dt: float = 5e-2):
        
        super(SineDataset, self).__init__()
        pulse_times = np.linspace(duration * 1 / 10, duration * 4 / 10, n_unique_pulses).astype(int)
        pulses = np.zeros((duration, n_unique_pulses))
        targets = np.zeros((duration, n_unique_pulses))
        times = np.arange(duration)
        period = int(2 * np.pi / dt)

        for idx, pulse_start in enumerate(pulse_times):
            pulses[pulse_start: pulse_start + pulse_width, idx] = 1
            targets[:, idx] = amplitude * np.sin((times - pulse_start - delay) * frequency * dt)
            targets[:(pulse_start + delay), idx] = 0
            targets[(pulse_start + delay + 3 * period):, idx] = 0
            pulses[(pulse_start + 3 * period): (pulse_start + pulse_width + 3 * period), idx] = -1

        pulses = torchify(gaussian_filter1d(pulses, sigma=1, axis=0))
        targets = torchify(targets)
        self.x = pulses
        self.y = targets
        self.samples = n_unique_pulses

    def __getitem__(self, item):
        ""
        return self.x[:, item], self.y[:, item]
        
    def __len__(self):
        ""
        return self.samples


class PacmanDataset(Dataset):
    def __init__(self, trial_duration: int = 500, sigma: float = 25, n_samples: int = 100,
                 gains: Sequence = None, viscosity: Sequence = None, polarity: Sequence = None):
        super(PacmanDataset, self).__init__()
        if gains is None:
            gains = (0.25, 0.5, 1)

        if viscosity is None:
            viscosity = (0, 0.5, 1)

        if polarity is None:
            polarity = (-1, 1)

        xs = np.expand_dims(np.linspace(0, trial_duration, trial_duration), 1)
        self.kernel = self.exponentiated_quadratic(xs, xs, sigma=sigma)
        targets = self.sample_gaussian_process(n_samples=n_samples)

        n_entries = n_samples * len(polarity) * len(viscosity) * len(gains)
        contexts = np.zeros((n_entries, 3))
        out_targets = np.zeros((n_entries, trial_duration))

        for idx, (gain, visc, polar, trajectory) in enumerate(product(gains, viscosity, polarity, targets)):
            contexts[idx] = np.asarray([gain, visc, polar])
            out_targets[idx] = trajectory

        self.x = torchify(contexts)
        self.y = torchify(out_targets)
        self.samples = n_entries

    def __getitem__(self, item):
        """

        Args:
            item:

        Returns:

        """
        return self.x[item], self.y[item]

    def __len__(self):
        """"""
        return self.samples

    @classmethod
    def exponentiated_quadratic(cls, xa: np.ndarray, xb: np.ndarray, sigma: float = 1):
        """
        Exponentiated quadratic pairwise distances
        Args:
            xa: np.ndarray,

            xb: np.ndarray,

            sigma: float, optional


        Returns:

        """
        sq_norm = -1/(2 * sigma **2) * cdist(xa, xb, 'sqeuclidean')
        return np.exp(sq_norm)

    def sample_gaussian_process(self, n_samples: int = 100):
        ys = np.random.multivariate_normal(
            mean=np.zeros(self.kernel.shape[0]), cov=self.kernel,
            size=n_samples)
        return ys


if __name__ == '__main__':
    data_set = PacmanDataset()
    first_data = data_set[0]
    plt.plot(first_data[1])
    plt.pause(1)
    pdb.set_trace()