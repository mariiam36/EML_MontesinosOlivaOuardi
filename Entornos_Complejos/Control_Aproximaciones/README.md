# Parte 2B: Control con Aproximaciones

## Descripción

Esta sección aborda el problema del control en espacios de estados continuos mediante aproximación de funciones. Se implementan y comparan SARSA semi-gradiente y Deep Q-Learning en el entorno *Mountain Car* de Gymnasium.

## Estructura

```
Control_Aproximaciones/
├── algorithms/
├── environments/
├── policies/
├── utils/
├── Sarsa_SemiGradiente.ipynb
├── Deep_Q-Learning.ipynb
├── main.ipynb
└── README.md
```

- `algorithms/`: implementaciones de SARSA semi-gradiente y Deep Q-Learning.
- `environments/`: configuración del entorno *Mountain Car*.
- `policies/`: implementación de la política ε-greedy con decaimiento.
- `utils/`: funciones auxiliares para discretización y visualización.
- `Sarsa_SemiGradiente.ipynb`: entrenamiento de SARSA con aproximación lineal.
- `Deep_Q-Learning.ipynb`: entrenamiento de Deep Q-Learning con red neuronal y *replay buffer*.
- `main.ipynb`: comparación de ambos métodos.
- `README.md`: este fichero.


## Tecnologías

Las mismas que las detalladas en el fichero `README.md` principal.

