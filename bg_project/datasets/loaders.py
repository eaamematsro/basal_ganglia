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
            duration * 1 / 10, duration * 2 / 10, n_unique_pulses
        ).astype(int)

        stop_times = np.linspace(
            duration * 7 / 10, duration * 9 / 10, n_unique_pulses
        ).astype(int)

        nsamples = n_unique_pulses**2 * len(amplitudes) * len(frequencies)

        timing_pulses = np.zeros((2, duration, nsamples))
        targets = np.zeros((duration, nsamples))
        parameters = np.zeros((2, duration, nsamples))
        times = np.arange(duration)

        amp_var = np.std(amplitudes)
        freq_var = np.std(frequencies)

        if amp_var == 0:
            amp_var = 1
        if freq_var == 0:
            freq_var = 1

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

        parameters[0] /= amp_var
        parameters[1] /= freq_var

        pulses = torchify(gaussian_filter1d(timing_pulses, sigma=5, axis=1))
        parameters = torchify(parameters)
        targets = torchify(targets)
        self.pulses = pulses
        self.contexts = parameters
        self.heights = targets
        self.samples = nsamples
        self.normalizers = (1 / amp_var, 1 / freq_var)

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
        spring_constant: Optional[Sequence] = None,
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
            spring_constant: Set of possible spring constants to draw from. Must be nonnegative.
        """
        super(PacmanDataset, self).__init__()
        if masses is None:
            masses = (0.5, 1, 2)

        if viscosity is None:
            viscosity = (0, 1, 2, 3)

        if polarity is None:
            polarity = (1, -1)

        if spring_constant is None:
            spring_constant = (0.1, 0.5)

        sigma = sigma_fraction * trial_duration
        xs = np.expand_dims(np.linspace(0, trial_duration, trial_duration), 1)
        self.kernel = self.exponentiated_quadratic(xs, xs, sigma=sigma)
        targets = self.sample_gaussian_process(n_samples=n_samples)

        n_entries = (
            n_samples
            * len(polarity)
            * len(viscosity)
            * len(masses)
            * len(spring_constant)
        )
        contexts = np.zeros((n_entries, 4))
        out_targets = np.zeros((n_entries, trial_duration))

        for idx, (mass, visc, polar, trajectory, spring_k) in enumerate(
            product(masses, viscosity, polarity, targets, spring_constant)
        ):
            contexts[idx] = np.asarray([mass, visc, polar, spring_k])
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


class TwoChoiceDataset(Dataset):
    """Class for generating training data for two choice decision-making task

    This class samples continuous trajectories from a gaussian process. It also
    returns a tuple relating to object mass, environment viscosity, and pedal polarity.

    Attributes:
        choice: Torch tensor of target choices for each sample
        evidence: Torch tensor of stimulus evidence for each sample.
        masks: Torch tensor of masked values to include in loss computation
        samples: Number of episodes in dataset.
    """

    def __init__(
        self,
        trial_duration: int = 500,
        n_samples: int = 5000,
        min_delay_frac: float = 0.7,
        max_delay_frac: float = 0.9,
        delay_duration_frac: float = 0.5,
        evidence_gain: float = 3,
        evidence_bias: float = 0,
    ):
        """Instantiates model class

        Args:
            trial_duration: Duration of a trial in network time steps
            n_samples: Number of trials in dataset
            min_delay_frac: Minimum time at which delay ends
            max_delay_frac: Maximum time at which delay ends
            delay_duration_frac: Length of minimum delay period
            evidence_gain: Strength of evidence on choice
            evidence_bias: Choice bias, if negative choices are biased towards -1 even with low evidence
        """

        super(TwoChoiceDataset, self).__init__()
        mask = np.zeros((n_samples, trial_duration))
        choice = np.zeros((n_samples, trial_duration))
        evidence = np.zeros((n_samples, trial_duration))

        evidence_base = np.random.rand(n_samples) * 2 - 1

        min_delay = int(min_delay_frac * trial_duration)
        max_delay = int(max_delay_frac * trial_duration)
        delay_start = int(delay_duration_frac * trial_duration)
        for sample in range(n_samples):
            report_time = np.random.randint(min_delay, max_delay)
            evidence[sample, :delay_start] = evidence_base[sample]
            choice[sample] = np.sign(evidence_base[sample])
            mask[sample, report_time:] = 1 / (trial_duration - report_time)

        self.masks: torch.Tensor = torchify(mask)
        self.evidence: torch.Tensor = torchify(evidence)
        self.choice: torch.Tensor = torchify(choice)
        self.samples = n_samples

    def __getitem__(self, item):
        """

        Args:
            item:

        Returns:

        """
        return self.evidence[item], self.choice[item], self.masks[item]

    def __len__(self):
        """"""
        return self.samples
