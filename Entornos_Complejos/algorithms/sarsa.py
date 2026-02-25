import numpy as np
from tqdm import tqdm
from .base import BaseAlgorithm
from policies.epsilon_greedy import epsilon_greedy_action


class SARSA(BaseAlgorithm):
    """
    SARSA (on-policy TD control)
    """

    def __init__(self, env, alpha=0.5, discount_factor=1.0, epsilon=0.1, decay=True):
        super().__init__(env)
        self.alpha = alpha
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.decay = decay

    def train(self, num_episodes=5000):

        rewards = []
        episode_lengths = []
        
        epsilon_start = self.epsilon
        
        decay_steps = num_episodes * 0.8 
        decay_rate = (epsilon_start - 0.05) / decay_steps

        for t in tqdm(range(num_episodes), desc="Entrenando SARSA"):
            
            state, _ = self.env.reset()
            done = False

            epsilon = (max(0.05, epsilon_start - (decay_rate * t))) if self.decay else self.epsilon

            action = epsilon_greedy_action(self.Q, state, epsilon)

            total_reward = 0
            steps = 0

            while not done:
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                next_action = epsilon_greedy_action(self.Q, next_state, epsilon)

                self.Q[state, action] += self.alpha * (
                    reward
                    + self.discount_factor * self.Q[next_state, next_action]
                    - self.Q[state, action]
                )

                state = next_state
                action = next_action

                total_reward += reward
                steps += 1

            rewards.append(total_reward)
            episode_lengths.append(steps)

        return rewards, episode_lengths