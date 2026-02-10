"""
Module: algorithms/softmax.py
Description: Implementación del algoritmo softmax para el problema de los k-brazos.

Author: Luis Daniel Hernández Molinero
Email: ldaniel@um.es
Date: 2025/01/29

This software is licensed under the GNU General Public License v3.0 (GPL-3.0),
with the additional restriction that it may not be used for commercial purposes.

For more details about GPL-3.0: https://www.gnu.org/licenses/gpl-3.0.html
"""

import numpy as np

from algorithms.algorithm import Algorithm

class Softmax(Algorithm):

    def __init__(self, k: int, tau: float = 0.1):
        """
        Inicializa el algoritmo softmax.

        :param k: Número de brazos.
        :param tau: .
        :raises ValueError: Si tau no es mayor que 0.
        """
        assert tau > 0, "tau debe ser mayor que 0"

        super().__init__(k)
        self.tau = tau

    def select_arm(self) -> int:
        """
        Selecciona un brazo basado en la política softmax.

        :return: índice del brazo seleccionado.
        """

        # Probar cada brazo una vez
        if np.any(self.counts == 0):
            untried_arms = np.where(self.counts == 0)[0]
            return np.random.choice(untried_arms)

        # Calcular probabilidades softmax
        exp_values = np.exp(self.values / self.tau)
        probabilities = exp_values / np.sum(exp_values)

        # Elegir según esa distribución
        chosen_arm = np.random.choice(self.k, p=probabilities)

        return chosen_arm

