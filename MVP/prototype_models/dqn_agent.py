import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class DuelingDQN(nn.Module):
    def __init__(self, state_dim=8, action_dim=8, hidden_dim=64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.value_stream     = nn.Sequential(nn.Linear(hidden_dim, 32), nn.ReLU(), nn.Linear(32, 1))
        self.advantage_stream = nn.Sequential(nn.Linear(hidden_dim, 32), nn.ReLU(), nn.Linear(32, action_dim))

    def forward(self, x):
        shared = self.shared(x)
        V = self.value_stream(shared)
        A = self.advantage_stream(shared)
        return V + (A - A.mean(dim=1, keepdim=True))


class DQNAgent:
    def __init__(self, state_dim=8, action_dim=8, lr=1e-4, gamma=0.95):
        self.device       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.action_dim   = action_dim
        self.gamma        = gamma
        self.epsilon      = 0.30   # more exploration for demo
        self.epsilon_min  = 0.01
        self.epsilon_decay= 0.999
        self.update_counter = 0

        self.model        = DuelingDQN(state_dim, action_dim).to(self.device)
        self.target_model = DuelingDQN(state_dim, action_dim).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())

        self.optimizer    = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn      = nn.SmoothL1Loss()
        self.replay_buffer= deque(maxlen=5000)

    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1), random.random() * 0.3

        state_t  = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.model(state_t).detach().cpu().numpy()[0]
        action   = int(np.argmax(q_values))
        # softmax confidence
        exp_q    = np.exp(q_values - q_values.max())
        confidence = float(exp_q[action] / exp_q.sum())
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        return action, confidence

    def train(self, batch_size=64):
        if len(self.replay_buffer) < batch_size:
            return None

        batch  = random.sample(self.replay_buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states      = torch.FloatTensor(states).to(self.device)
        actions     = torch.LongTensor(actions).to(self.device)
        rewards     = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones       = torch.FloatTensor(dones).to(self.device)

        current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q    = self.target_model(next_states).max(dim=1)[0]
        target_q  = rewards + self.gamma * next_q * (1 - dones)

        loss = self.loss_fn(current_q, target_q.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_counter += 1
        if self.update_counter % 10 == 0:
            for p, tp in zip(self.model.parameters(), self.target_model.parameters()):
                tp.data.copy_(0.99 * tp.data + 0.01 * p.data)

        return float(loss.item())

    def save(self, path='models/pretrained_dqn.pt'):
        torch.save(self.model.state_dict(), path)

    def load(self, path='models/pretrained_dqn.pt'):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.target_model.load_state_dict(self.model.state_dict())