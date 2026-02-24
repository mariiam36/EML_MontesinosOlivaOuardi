import numpy as np
from tqdm import tqdm
from .base import BaseAlgorithm
from policies.epsilon_greedy import epsilon_greedy_action


class MCOnPolicy(BaseAlgorithm):
    """
    Monte Carlo On-Policy Control (all-visit, incremental).
    Política: epsilon-greedy (con posible decaimiento).
    """

    def __init__(self, env, epsilon=0.4, discount_factor=1.0, decay=True):
        super().__init__(env)
        self.epsilon = epsilon
        self.discount_factor = discount_factor
        self.decay = decay

        # Contador de visitas (para versión incremental)
        self.n_visits = np.zeros((self.nS, self.nA))

    def train(self, num_episodes=5000):

        rewards_per_episode = []
        episode_lengths = []

        for t in tqdm(range(num_episodes), desc="Entrenando MC on-policy"):

            state, _ = self.env.reset()
            done = False

            epsilon = (
                max(0.05, self.epsilon * (0.995 ** t))
                if self.decay else self.epsilon
            )

            episode = []
            total_reward = 0.0
            #factor = 1        # Acumula la potencia del descuento
            # Generar episodio
            while not done:
                action = epsilon_greedy_action(self.Q, state, epsilon)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                episode.append((state, action, reward))
                total_reward += reward
                #factor *= self.discount_factor
                state = next_state

            # Actualización backward
            G = 0.0
            for (state, action, reward) in reversed(episode):
                G = self.discount_factor * G + reward

                self.n_visits[state, action] += 1
                alpha = 1.0 / self.n_visits[state, action]

                self.Q[state, action] += alpha * (G - self.Q[state, action])

            rewards_per_episode.append(total_reward)
            episode_lengths.append(len(episode))

        return rewards_per_episode, episode_lengths