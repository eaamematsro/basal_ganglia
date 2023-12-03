"""This module contains classes corresponding to all the tasks that networks will be trained to perform.

The classes contained in this model all generate torch datasets that can be used by a torch data
loader to sample training data more efficiently.

Returns:
    SineDataset:
    PacmanDataset:

Examples:
    data_set = PacmanDataset()
    first_data = data_set[0]
"""
import pdb
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from model_factory.factory_utils import torchify
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import cdist
from typing import Sequence, Optional
from itertools import product


class SineDataset(Dataset):
    """Class for constructing sine generation dataset.


    Attributes:
        pulses: Torch tensor that stores the value of the go cue for each sinusoid.
        heights: Torch tensor that stores the desired value of the output for each sinusoid.
        samples: An integer that stores the total number of unique sinusoids.
    """

    def __init__(
        self,
        n_unique_pulses: int = 25,
        pulse_width: int = 10,
        frequencies: tuple = (1,),
        delay: int = 0,
        amplitudes: tuple = (1,),
        duration: int = 500,
        dt: float = 5e-2,
    ):
        """Initializes the sine dataset based on desired frequency, amplitude, etc.

        Args:
            n_unique_pulses: Number of unique go times.
            pulse_width: Duration of the go cue.
            frequencies: Frequencies of the sinusoidal output.
            delay:
            amplitudes: Amplitudes of the sinusoidal output.
            duration: Duration of a single trial.
            dt: Duration of each time step.
        """

        super(SineDataset, self).__init__()
        start_times = np.linspace(
            duration * 1 / 10, duration * 4 / 10, n_unique_pulses
        ).astype(int)

        stop_times = np.linspace(
            duration * 7 / 10, duration * 9 / 10, n_unique_pulses
        ).astype(int)

        nsamples = n_unique_pulses**2 * len(amplitudes) * len(frequencies)

        timing_pulses = np.zeros((2, duration, nsamples))
        targets = np.zeros((duration, nsamples))
        parameters = np.zeros((2, duration, nsamples))
        times = np.arange(duration)
        for idx, (pulse_start, pulse_stop, amplitude, frequency) in enumerate(
            product(start_times, stop_times, amplitudes, frequencies)
        ):
            timing_pulses[
                0,
                pulse_start - int(pulse_width / 2) : pulse_start + int(pulse_width / 2),
                idx,
            ] = 1
            timing_pulses[
                1,
                pulse_stop - int(pulse_width / 2) : pulse_stop + int(pulse_width / 2),
                idx,
            ] = 1
            targets[:, idx] = amplitude * np.sin(
                (times - pulse_start - delay) * frequency * dt
            )
            targets[: (pulse_start + delay), idx] = 0
            targets[(pulse_stop + delay) :, idx] = 0
            parameters[:, :, idx] = np.array([amplitude, frequency])[:, None]

        pulses = torchify(gaussian_filter1d(timing_pulses, sigma=5, axis=1))
        parameters = torchify(parameters)
        targets = torchify(targets)
        self.pulses = pulses
        self.contexts = parameters
        self.heights = targets
        self.samples = nsamples

    def __getitem__(self, item):
        """Returns a sample from the dataset"""
        return (
            self.pulses[:, :, item],
            self.contexts[:, :, item],
        ), self.heights[:, item]

    def __len__(self):
        """Returns the number of samples in the dataset"""
        return self.samples


class PacmanDataset(Dataset):
    """Class for generating samples for feedback driven pacman task.

    This class samples continuous trajectories from a gaussian process. It also
    returns a tuple relating to object mass, environment viscosity, and pedal polarity.

    Attributes:
        kernel: Gaussian process kernel used to draw target trajectories
        contexts: Torch tensor of contexts, where each context is given by
            (object mass, environment viscosity, pedal polarity)
        heights: Torch tensor of target heights.
        samples: Number of trajectories in dataset.
    """

    def __init__(
        self,
        trial_duration: int = 500,
        sigma_fraction: float = 0.025,
        n_samples: int = 100,
        masses: Optional[Sequence] = None,
        viscosity: Optional[Sequence] = None,
        polarity: Optional[Sequence] = None,
    ):
        """Instantiates target trajectories for pacman task

        Args:
            trial_duration: Duration of a pacman trial
            sigma_fraction: fraction of the trial duration to use as
             standard deviation of autocorrelation kernel
            n_samples: Number of trajectories to sample
            masses: Set of masses of dot. Must be greater than 0.
            viscosity: Set of environment viscosities. Must be nonnegative.
            polarity: Set of polarities to draw from. Must be -1 or 1
        """
        super(PacmanDataset, self).__init__()
        if masses is None:
            masses = (0.5, 1, 2)

        if viscosity is None:
            viscosity = (0, 1, 2, 3)

        if polarity is None:
            polarity = (1, -1)

        sigma = sigma_fraction * trial_duration
        xs = np.expand_dims(np.linspace(0, trial_duration, trial_duration), 1)
        self.kernel = self.exponentiated_quadratic(xs, xs, sigma=sigma)
        targets = self.sample_gaussian_process(n_samples=n_samples)

        n_entries = n_samples * len(polarity) * len(viscosity) * len(masses)
        contexts = np.zeros((n_entries, 3))
        out_targets = np.zeros((n_entries, trial_duration))

        for idx, (mass, visc, polar, trajectory) in enumerate(
            product(masses, viscosity, polarity, targets)
        ):
            contexts[idx] = np.asarray([mass, visc, polar])
            out_targets[idx] = trajectory

        self.contexts: torch.Tensor = torchify(contexts).T
        self.heights: torch.Tensor = torchify(out_targets).T
        self.samples = n_entries

    def __getitem__(self, item):
        """

        Args:
            item:

        Returns:

        """
        return self.contexts[:, item], self.heights[:, item]

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
        sq_norm = -1 / (2 * sigma**2) * cdist(xa, xb, "sqeuclidean")
        return np.exp(sq_norm)

    def sample_gaussian_process(self, n_samples: int = 100):
        ys = np.random.multivariate_normal(
            mean=np.zeros(self.kernel.shape[0]), cov=self.kernel, size=n_samples
        )
        return ys
