# Parte 2A: Métodos Tabulares

## Descripción

En esta sección se implementan y comparan distintos métodos tabulares de aprendizaje por refuerzo en el entorno discreto *Cliff Walking* de Gymnasium. Se estudian métodos Monte Carlo (on-policy y off-policy) y métodos de Diferencia Temporal (SARSA y Q-Learning).



## Estructura

```
Metodos_Tabulares/
├── algorithms/
├── environments/
├── policies/
├── utils/
├── MC_AllVisits_CliffWalking.ipynb
├── MC_AllVisitsOff_CliffWalking.ipynb
├── Sarsa_CliffWalking.ipynb
├── QLearning_CliffWalking.ipynb
├── main.ipynb
└── README.md
```

- `algorithms/`: implementaciones de Monte Carlo, SARSA y Q-Learning.
- `environments/`: definición y configuración de los entornos utilizados.
- `policies/`: implementación de políticas ($\epsilon$-greedy).
- `utils/`: funciones auxiliares para visualización.
- `MC_AllVisits_CliffWalking.ipynb`: ejecución de Monte Carlo on-policy.
- `MC_AllVisitsOff_CliffWalking.ipynb`: ejecución de Monte Carlo off-policy.
- `Sarsa_CliffWalking.ipynb`: ejecución del algoritmo SARSA.
- `QLearning_CliffWalking.ipynb`: ejecución del algoritmo Q-Learning.
- `main.ipynb`: comparación conjunta de todos los métodos.
- `README.md`: este fichero.


## Tecnologías

Las mismas que las detalladas en el fichero `README.md` principal.
