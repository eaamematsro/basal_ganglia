import numpy as np
import torch
from typing import Optional


def torchify(
    input_array: np.ndarray, device: Optional[torch.device] = None
) -> torch.Tensor:
    array = torch.from_numpy(input_array).to(torch.float)
    if device is not None:
        array.to(device)

    return array
