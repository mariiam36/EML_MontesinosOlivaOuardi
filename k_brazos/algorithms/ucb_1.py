"""
Module: algorithms/ucb_1.py
Description: Implementación del algoritmo UCB1 para el problema de los k-brazos.

Author: Luis Daniel Hernández Molinero
Email: ldaniel@um.es
Date: 2025/01/29

This software is licensed under the GNU General Public License v3.0 (GPL-3.0),
with the additional restriction that it may not be used for commercial purposes.

For more details about GPL-3.0: https://www.gnu.org/licenses/gpl-3.0.html
"""

import numpy as np

from algorithms.algorithm import Algorithm

class UCB1(Algorithm):

    def __init__(self, k: int):
        """
        Inicializa el algoritmo UCB1.

        :param k: Número de brazos.
        """
        super().__init__(k)

    def select_arm(self) -> int:
        """
        Selecciona un brazo basado en el algoritmo UCB1.

        :return: índice del brazo seleccionado.
        """

        # Asegurar que cada brazo se selecciona una vez
        if np.any(self.counts == 0):
            # Si hay brazos que no han sido seleccionados, elige uno de ellos al azar
            untried_arms = np.where(self.counts == 0)[0]
            return np.random.choice(untried_arms)
        
        # Número total de tiradas
        total_counts = np.sum(self.counts)

        # Calcular el valor UCB para cada brazo
        bonus = np.sqrt((2 * np.log(total_counts)) / self.counts)
        ucb_values = self.values + bonus

        return np.argmax(ucb_values)
