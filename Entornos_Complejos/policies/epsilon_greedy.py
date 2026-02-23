import numpy as np

def epsilon_soft_distribution(Q, state, epsilon):
    nA = Q.shape[1]
    probs = np.ones(nA, dtype=float) * epsilon / nA
    best_action = np.argmax(Q[state])
    probs[best_action] += (1.0 - epsilon)
    return probs

def epsilon_greedy_action(Q, state, epsilon):
    probs = epsilon_soft_distribution(Q, state, epsilon)
    return np.random.choice(len(probs), p=probs)

def greedy_action(Q, state):
    return np.argmax(Q[state])