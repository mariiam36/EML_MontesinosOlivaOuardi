import random
import numpy as np
from collections import deque

class ReplayBuffer:

    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s_next, done):
        self.buffer.append((s, a, r, s_next, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)

        s, a, r, s_next, d = map(np.array, zip(*batch))
        return s, a, r, s_next, d

    def __len__(self):
        return len(self.buffer)