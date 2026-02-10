"""
Module: arms/armbernoulli.py
Description: Contains the implementation of the ArmBernoulli class for the Bernoulli distribution arm.

Author: Luis Daniel Hernández Molinero
Email: ldaniel@um.es
Date: 2025/01/29

This software is licensed under the GNU General Public License v3.0 (GPL-3.0),
with the additional restriction that it may not be used for commercial purposes.

For more details about GPL-3.0: https://www.gnu.org/licenses/gpl-3.0.html
"""


import numpy as np

from arms import Arm


class ArmBernoulli(Arm):
    def __init__(self, p: float, n: int = 1):
        """
        Inicializa el brazo con distribución Bernoulli.

        :param p: Probabilidad de éxito en cada ensayo.
        """
        assert 0 <= p <= 1, "La probabilidad p debe estar entre 0 y 1."
        assert n == 1, "El número de ensayos n debe ser 1 para una distribución Bernoulli."

        self.p = p
        self.n = n

    def pull(self):
        """
        Genera una recompensa siguiendo una distribución Bernoulli.

        :return: Recompensa obtenida del brazo.
        """
        reward = np.random.binomial(self.n, self.p)
        return reward

    def get_expected_value(self) -> float:
        """
        Devuelve el valor esperado de la distribución Bernoulli.

        :return: Valor esperado de la distribución.
        """

        return self.p

    def __str__(self):
        """
        Representación en cadena del brazo Bernoulli.

        :return: Descripción detallada del brazo Bernoulli.
        """
        return f"ArmBernoulli(p={self.p}, n={self.n})"

    @classmethod
    def generate_arms(cls, k: int, p_min: float = 0.1, p_max: float = 0.9):
        """
        Genera k brazos Bernoulli con probabilidades únicas en el rango [p_min, p_max].

        :param k: Número de brazos a generar.
        :param p_min: Probabilidad mínima de éxito.
        :param p_max: Probabilidad máxima de éxito.
        :return: Lista de brazos generados.
        """
        assert k > 0, "El número de brazos k debe ser mayor que 0."
        assert 0 <= p_min < p_max <= 1, "Las probabilidades deben estar en [0,1]."

        p_values = set()
        while len(p_values) < k:
            p = np.random.uniform(p_min, p_max)
            p = round(p, 2)
            p_values.add(p)

        p_values = list(p_values)

        arms = [ArmBernoulli(p) for p in p_values]

        return arms



