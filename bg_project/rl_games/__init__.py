from gymnasium.envs.registration import register

register(id="GridWorld-v0", entry_point="rl_games.envs.grid_worlds:GridWorldEnv")
register(
    id="MultiGridWorld-v0",
    entry_point="rl_games.envs.grid_worlds:MultiWorldGridWorldEnv",
)

register(
    id="SineGeneration-v0",
    entry_point="rl_games.envs.sine_generation:SineGenerationEnv",
)
