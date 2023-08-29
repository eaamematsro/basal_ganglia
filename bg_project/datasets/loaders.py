import pdb

import numpy as np
from torch.utils.data import Dataset, DataLoader
from model_factory.factory_utils import torchify
from scipy.ndimage import gaussian_filter1d


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


if __name__ == '__main__':
    data_set = SineDataset()
    first_data = data_set[0]
    pdb.set_trace()