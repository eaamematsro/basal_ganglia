import numpy as np
from rl_games.envs.custom_env import CustomEnv
from rl_games.pygames.rl_games import SineGeneration
from gymnasium import spaces


class SineGenerationEnv(CustomEnv):
    def _initialize_spaces(self):
        self.action_space = spaces.Box(low=-20, high=20, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=0,
            high=10,
            shape=(5,),
            dtype=np.float32,
        )

    def __init__(self, **kwargs):
        super().__init__(pygame=SineGeneration, **kwargs)
        self.pygame.reset()
