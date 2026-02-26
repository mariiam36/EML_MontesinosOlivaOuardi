from abc import ABC, abstractmethod
import numpy as np

class BaseAlgorithm(ABC):
    """
    Clase base abstracta para algoritmos de control en entornos discretos.
    """

    def __init__(self, env):
        self.env = env
        self.nS = env.observation_space.n
        self.nA = env.action_space.n
        self.Q = np.zeros((self.nS, self.nA))

    @abstractmethod
    def train(self, num_episodes: int):
        pass

    def greedy_policy(self):
        return np.argmax(self.Q, axis=1)
    

    # Política greedy trayectoria
    def greedy_policy_trajectory(self):
        """
        Simula el comportamiento greedy desde el estado inicial.

        Returns
        -------
        pi_star : np.ndarray
            Matriz [nS, nA] con la política marcada.
        actions : str
            String con la secuencia de acciones.
        grid_str : str
            Render ASCII del entorno.
        """

        done = False
        pi_star = np.zeros((self.nS, self.nA), dtype=int)
        state, _ = self.env.reset()
        actions = []

        while not done:
            action = np.argmax(self.Q[state])
            actions.append(str(action))
            pi_star[state, action] = action
            state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

        actions_str = ", ".join(actions)
        grid_str = self.env.render()

        return pi_star, actions_str, grid_str