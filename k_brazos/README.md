<!-- # Un estudio básico comparativo entre distintos algoritmos epsilon-greedy

Punto inicial para estudiar distintas soluciones al problema del bandido de k-brazos.  -->

# Parte 1: Bandido de k-Brazos

## Descripción

Esta carpeta contiene la implementación y experimentación asociada al problema del bandido de k-brazos. Se estudian distintos algoritmos de selección de acciones bajo diferentes distribuciones de recompensa, analizando su capacidad de exploración y explotación.


## Estructura

```
k_brazos/
├── algorithms/
├── arms/
├── plotting/
├── bandit_experiment_epsilon_greedy.ipynb
├── bandit_experiment_softmax.ipynb
├── bandit_experiment_ucb_1.ipynb
├── main.ipynb
└── README.md
```

- `algorithms/`: implementaciones de los algoritmos estudiados ($\epsilon$-greedy, Softmax y UCB1).
- `arms/`: implementación de los distintos tipos de brazos con diferentes distribuciones de recompensa.
- `plotting/`: funciones auxiliares para la generación de gráficas.
- `bandit_experiment_epsilon_greedy.ipynb`: comparación de tipos de brazo bajo $\epsilon$-greedy.
- `bandit_experiment_softmax.ipynb`: comparación bajo política Softmax.
- `bandit_experiment_ucb_1.ipynb`: comparación bajo UCB1.
- `main.ipynb`: comparación conjunta de todos los algoritmos bajo distintas distribuciones.
- `README.md`: este fichero.


## Tecnologías

Las mismas que las detalladas en el fichero `README.md` principal.
