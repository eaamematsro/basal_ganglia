import pdb

import gymnasium
import time
import matplotlib.pyplot as plt

from rl_factory.optimization import PPO


def make_env(gym_id):
    def thunk():
        env = gymnasium.make(gym_id)
        env = gymnasium.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


name = "CartPole-v1"
envs = gymnasium.vector.SyncVectorEnv([make_env(name) for _ in range(4)])
model = PPO(envs)
model.rollout()

# record video
# env = gymnasium.wrappers.RecordVideo(
#     env,
#     "videos",
# )

obs = envs.reset()
for _ in range(200):
    action = envs.action_space.sample()
    obs, reward, done, _, info = envs.step(action)
    envs.render()
