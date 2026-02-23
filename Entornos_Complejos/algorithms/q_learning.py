import numpy as np
from tqdm import tqdm
from .base import BaseAlgorithm
from policies.epsilon_greedy import epsilon_greedy_action


class QLearning(BaseAlgorithm):
    """
    Q-Learning (off-policy TD control)
    """

    def __init__(self, env, alpha=0.1, discount_factor=0.99, epsilon=0.3, decay=True):
        super().__init__(env)
        self.alpha = alpha
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.decay = decay

    def train(self, num_episodes=5000):

        episode_lengths = []
        rewards = []

        for t in tqdm(range(num_episodes), desc="Entrenando Q-Learning"):

            state, _ = self.env.reset()
            done = False

            epsilon = (
                max(0.05, self.epsilon * (0.995 ** t))
                if self.decay else self.epsilon
            )

            episode_reward = 0
            steps = 0

            while not done:

                action = epsilon_greedy_action(self.Q, state, epsilon)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                best_next_action = np.argmax(self.Q[next_state])

                self.Q[state, action] += self.alpha * (
                    reward
                    + self.discount_factor * self.Q[next_state, best_next_action]
                    - self.Q[state, action]
                )

                state = next_state
                episode_reward += reward
                steps += 1

            rewards.append(episode_reward)
            episode_lengths.append(steps)

        return rewards, episode_lengths