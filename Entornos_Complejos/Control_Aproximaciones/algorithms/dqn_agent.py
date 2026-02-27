from tqdm import tqdm
import numpy as np
from .dqn_network import QNetwork
from .replay_buffer import ReplayBuffer
import torch
import torch.optim as optim
import torch.nn as nn

class DQNAgent:

    def __init__(self,
                 env,
                 state_dim,
                 n_actions,
                 lr=1e-3,
                 gamma=0.99,
                 epsilon=1.0,
                 epsilon_min=0.05,
                 epsilon_decay=0.995,
                 batch_size=64,
                 target_update=50):

        self.env = env

        self.n_actions = n_actions
        self.gamma = gamma

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.batch_size = batch_size
        self.target_update = target_update

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Networks
        self.q_net = QNetwork(state_dim, n_actions).to(self.device)
        self.target_net = QNetwork(state_dim, n_actions).to(self.device)

        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        self.buffer = ReplayBuffer()

        self.step_counter = 0


    def select_action(self, state):

        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values = self.q_net(state_tensor)

        return torch.argmax(q_values).item()


    def train_step(self):

        if len(self.buffer) < self.batch_size:
            return

        s, a, r, s_next, d = self.buffer.sample(self.batch_size)

        s = torch.FloatTensor(s).to(self.device)
        a = torch.LongTensor(a).to(self.device)
        r = torch.FloatTensor(r).to(self.device)
        s_next = torch.FloatTensor(s_next).to(self.device)
        d = torch.FloatTensor(d).to(self.device)

        # Current Q
        q_values = self.q_net(s)
        q_sa = q_values.gather(1, a.unsqueeze(1)).squeeze()

        # Target Q
        with torch.no_grad():
            next_q = self.target_net(s_next).max(dim=1)[0]

            target = r + self.gamma * next_q * (1 - d)

        loss = nn.MSELoss()(q_sa, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def train(self, episodes=500):

        rewards_history = []
        steps_history = []

        for episode in tqdm(range(episodes), desc="DQN Training"):

            state, _ = self.env.reset()
            done = False

            episode_reward = 0
            steps = 0

            while not done:

                action = self.select_action(state)

                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                self.buffer.push(state, action, reward, next_state, done)

                self.train_step()

                state = next_state
                episode_reward += reward
                steps += 1

                self.step_counter += 1

                # Target network update
                if self.step_counter % self.target_update == 0:
                    self.target_net.load_state_dict(
                        self.q_net.state_dict()
                    )

            # Epsilon decay
            self.epsilon = max(
                self.epsilon_min,
                self.epsilon * self.epsilon_decay
            )
            steps_history.append(steps)  
            rewards_history.append(episode_reward)

        return rewards_history, steps_history
    
    
    def greedy_action(self, state):
        state_tensor = torch.FloatTensor(
            np.array(state, dtype=np.float32)
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values = self.q_net(state_tensor)

        return torch.argmax(q_values).item()
    
    def greedy_trajectory(self):

        state, _ = self.env.reset()
        done = False

        actions = []

        while not done:

            action = self.greedy_action(state)
            actions.append(str(action))

            state, _, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

        return ", ".join(actions)