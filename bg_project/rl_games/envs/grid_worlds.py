import pdb

import numpy as np
from rl_games.envs.custom_env import CustomEnv
from rl_games.pygames.rl_games import GridWorld, MultiWorldGridWorld, MultiRoomGridWorld, DegenerateGridWorld
from gymnasium import spaces


class GridWorldEnv(CustomEnv):
    def _initialize_spaces(self, image_obs: bool = True):
        self.action_space = spaces.Box(low=-20, high=20, shape=(2,), dtype=np.float32)
        if image_obs:
            self.observation_space = spaces.Box(
                low=0, high=255, shape=(self.pygame.width, self.pygame.height, 3)
                , dtype=np.uint8)
        else:
            self.observation_space = spaces.Box(
                low=-0.5,
                high=0.5,
                shape=(4,),
                dtype=np.float32,
            )

    def __init__(self, **kwargs):
        super().__init__(pygame=GridWorld, **kwargs)
        self.pygame.reset()


class DegenerateGridWorldEnv(CustomEnv):
    def _initialize_spaces(self, image_obs: bool = False, action_dim: int = 2, **kwargs):
        self.action_space = spaces.Box(low=-20, high=20, shape=(action_dim,), dtype=np.float32)
        if image_obs:
            self.observation_space = spaces.Box(
                low=0, high=255, shape=(self.pygame.width, self.pygame.height, 3)
                , dtype=np.uint8)
        else:
            self.observation_space = spaces.Box(
                low=-0.5,
                high=0.5,
                shape=(4,),
                dtype=np.float32,
            )

    def __init__(self, **kwargs):
        super().__init__(pygame=DegenerateGridWorld, **kwargs)
        self.pygame.reset()


class MultiRoomGridWorldEnv(CustomEnv):
    def _initialize_spaces(self, image_obs: bool = False):
        self.action_space = spaces.Box(low=-20, high=20, shape=(2,), dtype=np.float32)
        if image_obs:
            self.observation_space = spaces.Box(
                low=0, high=255, shape=(self.pygame.width, self.pygame.height, 3)
                , dtype=np.uint8)
        else:
            self.observation_space = spaces.Box(
                low=-0.5,
                high=0.5,
                shape=(4,),
                dtype=np.float32,
            )
    def __init__(self, **kwargs):
        super().__init__(pygame=MultiRoomGridWorld, **kwargs)
        self.pygame.reset()


class MultiWorldGridWorldEnv(CustomEnv):
    def _initialize_spaces(self, image_obs: bool = True):
        self.action_space = spaces.Box(low=-20, high=20, shape=(2,), dtype=np.float32)
        if image_obs:
            self.observation_space = spaces.Box(
                low=0, high=255, shape=(self.pygame.width, self.pygame.height, 3)
                , dtype=np.uint8)
        else:
            self.observation_space = spaces.Box(
                low=-0.5,
                high=0.5,
                shape=(4,),
                dtype=np.float32,
            )
    def __init__(self, **kwargs):
        super().__init__(pygame=MultiWorldGridWorld, **kwargs)
        self.pygame.reset()


if __name__ == '__main__':

    test_world = DegenerateGridWorldEnv()
