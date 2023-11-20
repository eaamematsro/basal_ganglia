import numpy as np
from rl_games.envs.custom_env import CustomEnv
from rl_games.pygames.rl_games import GridWorld, MultiWorldGridWorld
from gymnasium import spaces


class GridWorldEnv(CustomEnv):
    def _initialize_spaces(self):
        self.action_space = spaces.Box(low=-20, high=20, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=0,
            high=np.array(
                [
                    self.pygame.width,
                    self.pygame.width,
                    self.pygame.height,
                    self.pygame.height,
                ]
            ),
            shape=(4,),
            dtype=np.float32,
        )

    def __init__(self, **kwargs):
        super().__init__(pygame=GridWorld, **kwargs)
        self.pygame.reset()


class MultiWorldGridWorldEnv(CustomEnv):
    def _initialize_spaces(self):
        self.action_space = spaces.Box(low=-20, high=20, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=0,
            high=np.array(
                [
                    self.pygame.width,
                    self.pygame.width,
                    self.pygame.height,
                    self.pygame.height,
                ]
            ),
            shape=(4,),
            dtype=np.float32,
        )

    def __init__(self, **kwargs):
        super().__init__(pygame=MultiWorldGridWorld, **kwargs)
        self.pygame.reset()
