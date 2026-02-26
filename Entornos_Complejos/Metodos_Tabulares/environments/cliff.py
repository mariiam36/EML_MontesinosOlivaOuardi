import gymnasium as gym

def make_cliff_env(render_mode="ansi", seed=None, steps=-1):
    env = gym.make("CliffWalking-v1", render_mode=render_mode, max_episode_steps=steps)
    if seed is not None:
        env.reset(seed=seed)
    return env