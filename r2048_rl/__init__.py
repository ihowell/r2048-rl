from gym.envs.registration import register

register(
    id='r2048-v0',
    entry_point='r2048_rl.envs:R2048Env',
)
