import abc
import pdb
import numpy as np
import gymnasium
from gymnasium import Env
from gymnasium.core import ObsType, ActType

from rl_games.pygames.rl_games import PyGame
from typing import Optional, Any, SupportsFloat, Callable
import pygame


class CustomEnv(Env, metaclass=abc.ABCMeta):
    """"""

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 60,
    }

    def __init__(
        self, render_mode: Optional[str] = None, pygame: Optional[Callable] = None
    ):
        super().__init__()
        if pygame is not None:
            self.pygame: PyGame = pygame()
        else:
            self.pygame = pygame
        self._initialize_spaces()
        self.render_mode = render_mode
        self.clock = None

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        """"""
        self.pygame.reset()
        obs = self.pygame.observe()
        if self.render_mode == "human":
            self.render()
        return obs, {}

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """"""
        self.pygame.act(action)
        obs = self.pygame.observe()
        reward = self.pygame.evaluate()
        terminated = self.pygame.is_done()
        info = self.pygame.get_info()

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, False, info

    def render(self):
        """"""
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
            )
            return None

        if self.pygame.display is None:
            pygame.init()

            if self.render_mode == "human":
                pygame.display.init()
            else:
                self.pygame.display = pygame.Surface(
                    (self.pygame.width, self.pygame.height)
                )

        if self.pygame.clock is None:
            return None

        self.pygame.view()

        if self.render_mode == "human":
            pygame.event.pump()
            self.pygame.clock.tick(self.metadata["render_fps"])
            pygame.display.update()
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.pygame.display)), axes=(1, 0, 2)
            )

    def close(self):
        self.pygame.close()

    @abc.abstractmethod
    def _initialize_spaces(self):
        """"""
