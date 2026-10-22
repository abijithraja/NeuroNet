"""Dueling DQN starter: replay buffer, target network, epsilon-greedy policy."""
from collections import deque
import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class DuelingQNetwork(nn.Module):
    def __init__(self, state_dim, n_actions, hidden_dim=128):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.value = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, 1))
        self.advantage = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, n_actions))

    def forward(self, x):
        f = self.feature(x)
        v = self.value(f)
        a = self.advantage(f)
        return v + a - a.mean(dim=1, keepdim=True)

class ReplayBuffer:
    def __init__(self, capacity=10_000):
        self.data = deque(maxlen=capacity)
    def add(self, state, action, reward, next_state, done):
        self.data.append((np.array(state, dtype=np.float32), int(action), float(reward),
                          np.array(next_state, dtype=np.float32), float(done)))
    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (np.asarray(s), np.asarray(a), np.asarray(r, dtype=np.float32),
                np.asarray(ns), np.asarray(d, dtype=np.float32))
    def __len__(self):
        return len(self.data)

class DQNAgent:
    def __init__(self, state_dim, n_actions, hidden_dim=128, capacity=10_000,
                 batch_size=64, gamma=0.97, lr=1e-3, device=None,
                 target_update_every=250, seed=42):
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.gamma = gamma
        self.target_update_every = target_update_every
        self.online = DuelingQNetwork(state_dim, n_actions, hidden_dim).to(self.device)
        self.target = DuelingQNetwork(state_dim, n_actions, hidden_dim).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=lr)
        self.replay = ReplayBuffer(capacity)
        self.steps = 0
        self.learn_steps = 0

    def q_values(self, state):
        with torch.no_grad():
            x = torch.as_tensor(np.asarray(state, dtype=np.float32), device=self.device).unsqueeze(0)
            return self.online(x).squeeze(0).cpu().numpy()

    def act(self, state, epsilon=0.0):
        if random.random() < epsilon:
            return random.randrange(self.n_actions)
        return int(np.argmax(self.q_values(state)))

    def heuristic_confidence(self, state):
        """Heuristic only; not calibrated epistemic uncertainty."""
        q = self.q_values(state)
        shifted = q - np.max(q)
        exp = np.exp(np.clip(shifted, -50, 50))
        p = exp / (exp.sum() + 1e-9)
        return float(np.max(p))

    def remember(self, state, action, reward, next_state, done):
        self.replay.add(state, action, reward, next_state, done)
        self.steps += 1

    def learn(self):
        if len(self.replay) < self.batch_size:
            return None
        states, actions, rewards, next_states, dones = self.replay.sample(self.batch_size)
        s = torch.as_tensor(states, device=self.device)
        a = torch.as_tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        r = torch.as_tensor(rewards, device=self.device).unsqueeze(1)
        ns = torch.as_tensor(next_states, device=self.device)
        d = torch.as_tensor(dones, device=self.device).unsqueeze(1)

        q = self.online(s).gather(1, a)
        with torch.no_grad():
            # Double-DQN target: online chooses next action, target evaluates it.
            next_actions = self.online(ns).argmax(dim=1, keepdim=True)
            next_q = self.target(ns).gather(1, next_actions)
            target = r + self.gamma * next_q * (1.0 - d)
        loss = F.smooth_l1_loss(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 5.0)
        self.optimizer.step()
        self.learn_steps += 1
        if self.learn_steps % self.target_update_every == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.item())

    def save(self, path):
        torch.save({"online": self.online.state_dict(), "target": self.target.state_dict(),
                    "state_dim": self.online.feature[0].in_features,
                    "n_actions": self.n_actions}, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.online.load_state_dict(checkpoint["online"])
        self.target.load_state_dict(checkpoint.get("target", checkpoint["online"]))
        return self
