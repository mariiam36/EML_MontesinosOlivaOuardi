from abc import ABC, abstractmethod
import numpy as np

class BaseApproxControl(ABC):
    """
    Base para algoritmos de control con aproximación de función.
    """

    def __init__(self, env, feature_dim, n_actions, alpha=0.001,
                 discount_factor=0.99, epsilon=0.1):

        self.env = env
        self.nA = n_actions
        self.alpha = alpha
        self.gamma = discount_factor
        self.epsilon = epsilon

        # Parámetros del aproximador lineal
        self.theta = np.zeros((n_actions, feature_dim))


    def featurize(self, state):
        """
        Convierte estado discreto en vector de features.
        Aquí usamos encoding one-hot.
        """
        feature = np.zeros(self.theta.shape[1])
        idx = hash(state) % len(feature)
        feature[idx] = 1
        return feature


    @abstractmethod
    def train(self, num_episodes):
        pass


    # Cada algoritmo define cómo calcula la acción greedy
    @abstractmethod
    def greedy_action(self, state):
        pass


    # Trayectoria greedy común
    def greedy_trajectory(self):

        state, _ = self.env.reset()
        done = False

        actions = []

        while not done:

            action = self.greedy_action(state)
            actions.append(str(action))

            state, _, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

        return ", ".join(actions)