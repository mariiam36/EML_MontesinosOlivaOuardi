import gymnasium as gym

def make_mountain_car_env(render_mode="ansi", seed=None, steps=-1):
    env = gym.make("MountainCar-v0", render_mode=render_mode)
    if seed is not None:
        env.reset(seed=seed)
    return env