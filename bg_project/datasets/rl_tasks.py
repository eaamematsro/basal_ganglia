import abc
import torch
import numpy as np
from typing import Optional, Any, Tuple, Callable

from model_factory.architectures import BaseArchitecture, NETWORKS


class Task(metaclass=abc.ABCMeta):
    def __init__(
        self, network: str, lr: float = 1e-3, task: Optional[str] = None, **kwargs
    ):
        super(Task, self).__init__()
        self.network: BaseArchitecture = NETWORKS[network](task=task, **kwargs)
        self.type = network
        self.lr = lr
        self.optimizer = None
        self.lr_scheduler = None

    @abc.abstractmethod
    def get_reward(self, **kwargs):
        """"""

    @abc.abstractmethod
    def get_next_state(self):
        """"""

    def save_model(self):
        # TODO: Save network should
        self.network.save_model(task=self)
