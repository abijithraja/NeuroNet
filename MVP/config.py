from dataclasses import dataclass
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
LOG_DIR = ROOT / "logs"

@dataclass
class Config:
    seed: int = 42
    num_nodes: int = 20
    state_dim: int = 8
    hidden_dim: int = 128
    replay_capacity: int = 10_000
    batch_size: int = 64
    gamma: float = 0.97
    learning_rate: float = 1e-3
    target_update_every: int = 250
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 5_000
    confidence_threshold: float = 0.75
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

CONFIG = Config()
