import gymnasium as gym
import numpy as np

# Definimos una clase que hereda de gym.ObservationWrapper, la cual nos permite modificar las observaciones que devuelve el entorno.
class StateAggregationEnv(gym.ObservationWrapper):


    def __init__(self, env, bins, low, high):
        # Constructor de la clase. Recibe:
        # - env: el entorno original de Gym que se va a envolver.
        # - bins: un array (o lista) que indica el número de intervalos (o "cubos") para discretizar en cada dimensión.
        # - low: los valores mínimos para cada dimensión de la observación.
        # - high: los valores máximos para cada dimensión de la observación.
        # Llamamos al constructor de la clase padre (ObservationWrapper) pasando el entorno original.
        super().__init__(env)  # Hay que invocar siempre al super()

        # En el caso del coche:
        # low=[-1.2, -0.07], high=[0.6, 0.7], bins=[20, 20]

        # Creamos "cubos" o "buckets" para cada dimensión.
        # Usamos np.linspace para generar una secuencia de números entre el valor mínimo (j) y el valor máximo (k)
        # Se crean (l - 1) divisiones, donde l es el número de bins especificados para esa dimensión.
        # La función zip(low, high, bins) recorre en paralelo cada valor mínimo, máximo y cantidad de bins.
        self.buckets = [np.linspace(j, k, l - 1) for j, k, l in zip(low, high, bins)]

        # En el caso del coche, habrá "2 linspace (arrays)"
        # Un array es [-1.2, -1,1, ...., 0.5, 0.6] Con 20 valores
        # Otro array es [-0.07, -0.06, ..., 0.06, 0.07]  Con 20 valores

        # Definimos el espacio de observación discreto. Usamos gym.spaces.MultiDiscrete,
        # el cual define un espacio de observaciones con múltiples dimensiones discretas.
        # MultiDiscrete es un producto cartesiano de distintos espacios discretos. Aquí aplicamos dos.
        # nvec toma como argumento un vector o lista con el número de valores discretos que habrá en cada dimensión.
        # Convertimos bins a lista.
        self.observation_space = gym.spaces.MultiDiscrete(nvec=bins.tolist())

    # Método que se encarga de transformar la observación continua del entorno original
    # a una observación discretizada según los "cubos" definidos.
    def observation(self, obs):  # Hay  que sobreescribir necesariamente este método.
        # Para cada dimensión de la observación, usamos np.digitize para encontrar en qué intervalo (bucket) cae el valor.
        # np.digitize devuelve el índice del cubo al que pertenece el valor.
        # La función zip(obs, self.buckets) recorre cada valor de la observación y su correspondiente bucket.
        indices = tuple(np.digitize(i, b) for i, b in zip(obs, self.buckets))

        # Retornamos la tupla de índices, que representa el estado discretizado.
        return indices
