import numpy as np
from tqdm import tqdm
from .base import BaseAlgorithm
from policies.epsilon_greedy import epsilon_soft_distribution


class MCOffPolicy(BaseAlgorithm):
    """
    Monte Carlo Off-Policy Control (Importance Sampling, all-visit).
    Target policy: greedy
    Behavior policy: epsilon-soft
    """

    def __init__(self, env, discount_factor=0.95, epsilon=0.3, max_steps=50):
        super().__init__(env)
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.max_steps = max_steps

        self.C = np.zeros((self.nS, self.nA))  # acumulador pesos

    def train(self, num_episodes=5000):

        episode_rewards = []

        prob_greedy = 1.0 - self.epsilon + (self.epsilon / self.nA)
        inv_prob = 1.0 / prob_greedy
        
        for _ in tqdm(range(num_episodes), desc="Entrenando MC off-policy"):
            state, _ = self.env.reset()
            done = False
            step = 0

            states = []
            actions = []
            rewards = []

            total_reward = 0

            # Generar episodio (behavior policy)
            while not done and step < self.max_steps:

                probs = epsilon_soft_distribution(
                    self.Q, state, self.epsilon
                )
                action = np.random.choice(self.nA, p=probs)

                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                states.append(state)
                actions.append(action)
                rewards.append(reward)

                total_reward += reward
                state = next_state
                step += 1

            # Importance sampling backward
            G = 0.0
            W = 1.0

            for i in reversed(range(step)):

                s = states[i]
                a = actions[i]
                r = rewards[i]

                G = self.discount_factor * G + r

                self.C[s, a] += W
                self.Q[s, a] += (W / self.C[s, a]) * (G - self.Q[s, a])

                greedy_action = np.argmax(self.Q[s])

                if a != greedy_action:
                    break

                W *= inv_prob

            episode_rewards.append(total_reward)

        return episode_rewards
    

    def greedy_policy_analysis(self):
        """
        Versión especializada para off-policy.

        Devuelve:
        - pi_star
        - actions_str
        - grid_str
        """

        best_actions = np.argmax(self.Q, axis=1)

        pi_star = np.zeros((self.nS, self.nA), dtype=int)

        for s in range(self.nS):
            pi_star[s, best_actions[s]] = best_actions[s]

        actions_str = ", ".join(str(a) for a in best_actions)

        grid_str = self.env.render()

        return pi_star, actions_str, grid_str