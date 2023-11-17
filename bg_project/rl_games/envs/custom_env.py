import abc
from gymnasium import Env
from gymnasium import spaces
from gymnasium.core import ObsType, ActType

from rl_games.pygames.rl_games import PyGame
from typing import Optional, Any, SupportsFloat


class CustomEnv(Env, metaclass=abc.ABCMeta):
    """"""

    def __init__(self):
        super().__init__()
        self.pygame: Optional[PyGame] = None
        self._initialize_spaces()
        self.render_mode = None

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        """"""
        self.pygame.reset()
        obs = self.pygame.observe()
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

        return obs, reward, terminated, False, info

    def render(self, render_mode="human"):
        """"""
        self.pygame.view()

    def close(self):
        self.pygame.close()

    @abc.abstractmethod
    def _initialize_spaces(self):
        """"""
