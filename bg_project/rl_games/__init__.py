from gymnasium.envs.registration import register

register(
    id="GridWorld-v0",
    entry_point="rl_games.envs:GridWorld",
)
