"""Dueling Deep Q-Network (DQN) with Double-DQN learning and confidence estimation."""
import random
from collections import deque
from typing import Tuple, List, Optional, Union
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class DuelingQNetwork(nn.Module):
    def __init__(self, state_dim: int = 14, n_actions: int = 8, hidden_dim: int = 64):
        super().__init__()
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared = self.shared(x)
        val = self.value_stream(shared)
        adv = self.advantage_stream(shared)
        # Identifiable Dueling aggregation: Q(s, a) = V(s) + (A(s, a) - mean(A(s, :)))
        return val + (adv - adv.mean(dim=1, keepdim=True))


class ReplayBuffer:
    def __init__(self, capacity: int = 10_000):
        self.data = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.data.append((
            np.asarray(state, dtype=np.float32),
            int(action),
            float(reward),
            np.asarray(next_state, dtype=np.float32),
            float(done),
        ))

    def sample(self, batch_size: int):
        batch = random.sample(self.data, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (
            np.asarray(s, dtype=np.float32),
            np.asarray(a, dtype=np.int64),
            np.asarray(r, dtype=np.float32),
            np.asarray(ns, dtype=np.float32),
            np.asarray(d, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.data)


class DQNAgent:
    def __init__(
        self,
        state_dim: int = 14,
        n_actions: int = 8,
        hidden_dim: int = 64,
        capacity: int = 10_000,
        batch_size: int = 64,
        gamma: float = 0.95,
        lr: float = 1e-3,
        device: Optional[str] = None,
        target_update_every: int = 200,
        seed: int = 42,
    ):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.state_dim = state_dim
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

    def q_values(self, state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            s = np.asarray(state, dtype=np.float32)
            # Adapt if state dimension mismatches (e.g. 8 vs 14)
            if s.shape[0] != self.state_dim:
                if s.shape[0] < self.state_dim:
                    pad = np.zeros(self.state_dim - s.shape[0], dtype=np.float32)
                    s = np.concatenate([s, pad])
                else:
                    s = s[: self.state_dim]
            x = torch.as_tensor(s, device=self.device).unsqueeze(0)
            return self.online(x).squeeze(0).cpu().numpy()

    def act(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        if random.random() < epsilon:
            return random.randrange(self.n_actions)
        q = self.q_values(state)
        return int(np.argmax(q))

    def act_with_confidence(self, state: np.ndarray, epsilon: float = 0.0) -> Tuple[int, float, np.ndarray]:
        q = self.q_values(state)
        if random.random() < epsilon:
            chosen = random.randrange(self.n_actions)
            conf = float(random.uniform(0.15, 0.45))
            return chosen, conf, q

        chosen = int(np.argmax(q))
        shifted = q - np.max(q)
        exp = np.exp(np.clip(shifted, -40.0, 40.0))
        probs = exp / (np.sum(exp) + 1e-9)
        conf = float(probs[chosen])
        return chosen, conf, q

    def heuristic_confidence(self, state: np.ndarray) -> float:
        _, conf, _ = self.act_with_confidence(state, epsilon=0.0)
        return conf

    def remember(self, state, action, reward, next_state, done):
        s = np.asarray(state, dtype=np.float32)
        ns = np.asarray(next_state, dtype=np.float32)
        if s.shape[0] != self.state_dim:
            if s.shape[0] < self.state_dim:
                s = np.concatenate([s, np.zeros(self.state_dim - s.shape[0], dtype=np.float32)])
            else:
                s = s[: self.state_dim]
        if ns.shape[0] != self.state_dim:
            if ns.shape[0] < self.state_dim:
                ns = np.concatenate([ns, np.zeros(self.state_dim - ns.shape[0], dtype=np.float32)])
            else:
                ns = ns[: self.state_dim]
        self.replay.add(s, action, reward, ns, done)
        self.steps += 1

    def learn(self) -> Optional[float]:
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
            # Double DQN: online chooses next action, target evaluates it
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

    def save(self, path: Union[str, Path]):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "state_dim": self.state_dim,
            "n_actions": self.n_actions,
        }, str(path))

    def load(self, path: Union[str, Path]):
        path = Path(path)
        if not path.exists():
            return self
        checkpoint = torch.load(str(path), map_location=self.device, weights_only=False)
        if isinstance(checkpoint, dict) and "online" in checkpoint:
            state_dict = checkpoint["online"]
        else:
            state_dict = checkpoint

        # Check if shape of first linear layer matches
        first_weight = state_dict.get("shared.0.weight")
        if first_weight is None:
            first_weight = state_dict.get("feature.0.weight")
        if first_weight is not None and first_weight.shape[1] != self.state_dim:
            # Rebuild network to match saved state_dim
            ckpt_dim = first_weight.shape[1]
            self.state_dim = ckpt_dim
            self.online = DuelingQNetwork(ckpt_dim, self.n_actions, hidden_dim=first_weight.shape[0]).to(self.device)
            self.target = DuelingQNetwork(ckpt_dim, self.n_actions, hidden_dim=first_weight.shape[0]).to(self.device)
            self.optimizer = torch.optim.Adam(self.online.parameters(), lr=1e-3)

        # Map 'feature' -> 'shared' if needed
        renamed = {}
        for k, v in state_dict.items():
            new_k = k.replace("feature.", "shared.").replace("value.", "value_stream.").replace("advantage.", "advantage_stream.")
            renamed[new_k] = v

        try:
            self.online.load_state_dict(renamed, strict=False)
            self.target.load_state_dict(renamed, strict=False)
        except Exception:
            pass
        return self
