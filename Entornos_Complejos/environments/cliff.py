import gymnasium as gym

def make_cliff_env(render_mode="ansi", seed=None):
    env = gym.make("CliffWalking-v1", render_mode=render_mode)
    if seed is not None:
        env.reset(seed=seed)
    return env