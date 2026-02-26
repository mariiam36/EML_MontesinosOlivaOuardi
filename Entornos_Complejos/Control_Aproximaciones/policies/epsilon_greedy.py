import numpy as np

def epsilon_greedy(algorithm, state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(algorithm.nA)

    q_values = compute_q(algorithm, state)
    return np.argmax(q_values)

def compute_q(algorithm, state):
    x = algorithm.featurize(state)
    return np.array([np.dot(algorithm.theta[a], x)
                        for a in range(algorithm.nA)])