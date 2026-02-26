import numpy as np
from tqdm import tqdm
from .base import BaseApproxControl
from policies.epsilon_greedy import epsilon_greedy


class SARSAApprox:
    """
    SARSA semi-gradiente (aproximación lineal)
    """

    def __init__(self,
                 env,
                 feature_dim,
                 n_actions,
                 alpha=0.001,
                 discount_factor=0.99,
                 epsilon=0.3,
                 decay=True):

        self.env = env
        self.alpha = alpha
        self.gamma = discount_factor

        self.epsilon = epsilon
        self.decay = decay

        self.nA = n_actions

        # Parámetros del aproximador lineal
        self.theta = np.zeros((n_actions, feature_dim))


    def featurize(self, state):
        feature = np.zeros(self.theta.shape[1])
        idx = np.ravel_multi_index(state, self.env.observation_space.nvec)
        feature[idx] = 1
        return feature

    def train(self, num_episodes=5000):

        rewards = []
        steps = []

        epsilon_start = self.epsilon
        decay_steps = num_episodes * 0.8
        decay_rate = (epsilon_start - 0.05) / decay_steps

        for t in tqdm(range(num_episodes), desc="SARSA semi-gradiente"):

            state, _ = self.env.reset()
            done = False

            epsilon = (
                max(0.05, epsilon_start - decay_rate * t)
                if self.decay else self.epsilon
            )

            action = epsilon_greedy(self, state, epsilon)

            total_reward = 0
            steps_episode = 0

            while not done:

                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                next_action = epsilon_greedy(self, next_state, epsilon)

                x = self.featurize(state)
                x_next = self.featurize(next_state)

                q_sa = np.dot(self.theta[action], x)
                q_next = np.dot(self.theta[next_action], x_next) if not done else 0

                td_error = reward + self.gamma * q_next - q_sa

                # Semi-gradient update
                self.theta[action] += self.alpha * td_error * x

                state = next_state
                action = next_action

                total_reward += reward
                steps_episode += 1

            rewards.append(total_reward)
            steps.append(steps_episode)

        return rewards, steps