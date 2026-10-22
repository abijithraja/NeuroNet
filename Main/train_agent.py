"""Train the NeuroNet Dueling DQN agent across realistic network failure scenarios."""
import sys
from pathlib import Path

# Ensure project root is in sys.path so imports work from any working directory
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import numpy as np

from config import CONFIG, MODEL_DIR, LOG_DIR
from src.neuronet.actions import N_ACTIONS, ACTION_NAMES
from src.neuronet.network_sim import RealisticNetworkEnv
from src.neuronet.toy_env import ToyNetworkEnv
from src.neuronet.dqn_agent import DQNAgent
from src.neuronet.telemetry import make_state
from main import train_baseline_detector

def train(
    episodes: int = 250,
    env_type: str = "realistic",
    seed: int = 42,
    save_path: Path = None,
):
    print("\n" + "=" * 65)
    print("      NEURONET : DUELING DQN POLICY TRAINING PIPELINE")
    print("=" * 65)
    print(f" Environment : {env_type.upper()}")
    print(f" Episodes    : {episodes}")
    print(f" Device      : {CONFIG.device}")
    print("=" * 65 + "\n")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    out_model = save_path or (MODEL_DIR / "dueling_dqn.pt")

    # 1. Train detector
    print("[1/2] Fitting LSTM Autoencoder + Isolation Forest...")
    detector = train_baseline_detector(seed)

    # 2. Setup Environment & Agent
    if env_type == "toy":
        env = ToyNetworkEnv(seed=seed, num_nodes=CONFIG.num_nodes)
    else:
        env = RealisticNetworkEnv(seed=seed, num_nodes=CONFIG.num_nodes)

    agent = DQNAgent(
        state_dim=CONFIG.state_dim,
        n_actions=N_ACTIONS,
        hidden_dim=CONFIG.hidden_dim,
        capacity=CONFIG.replay_capacity,
        batch_size=CONFIG.batch_size,
        gamma=CONFIG.gamma,
        lr=CONFIG.learning_rate,
        device=CONFIG.device,
        target_update_every=CONFIG.target_update_every,
        seed=seed,
    )

    returns = []
    losses = []
    print("\n[2/2] Training Dueling DQN Agent...")
    print(f"{'Episode':<10} | {'Epsilon':<8} | {'Recent Mean Reward':<20} | {'Loss':<10} | {'Latest Action':<15}")
    print("-" * 75)

    for ep in range(1, episodes + 1):
        state, info = env.reset(scenario=None)  # Random scenario each episode
        metrics_before = info["metrics"]

        # Anomaly detection & state construction
        anomaly = detector.predict(metrics_before)
        state_vec = make_state(metrics_before, anomaly_info=anomaly, recent_actions=env.recent_actions)

        # Exploration epsilon decay
        epsilon = CONFIG.epsilon_end + (CONFIG.epsilon_start - CONFIG.epsilon_end) * np.exp(
            -ep / (episodes * 0.40)
        )

        action = agent.act(state_vec, epsilon=float(epsilon))
        next_state, reward, done, truncated, step_info = env.step(action, anomaly_info=anomaly)

        agent.remember(state_vec, action, reward, next_state, done)
        loss = agent.learn()

        returns.append(reward)
        if loss is not None:
            losses.append(loss)

        if ep % 25 == 0 or ep == 1 or ep == episodes:
            recent_avg = float(np.mean(returns[-25:]))
            loss_str = f"{loss:.4f}" if loss is not None else "Replay warm-up"
            act_name = ACTION_NAMES.get(action, str(action))
            print(f"{ep:<10d} | {epsilon:<8.3f} | {recent_avg:<+20.2f} | {loss_str:<10} | {act_name:<15}")

    agent.save(out_model)
    # Also save to pretrained_dqn.pt so it updates the default
    agent.save(MODEL_DIR / "pretrained_dqn.pt")
    print("-" * 75)
    print(f"\nTraining completed successfully across {episodes} episodes!")
    print(f"Overall Mean Reward: {np.mean(returns):+.2f}")
    print(f"Model checkpoint saved to: {out_model}")

def main():
    parser = argparse.ArgumentParser(description="Train NeuroNet DQN Agent")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--env", type=str, choices=["realistic", "toy"], default="realistic")
    parser.add_argument("--seed", type=int, default=CONFIG.seed)
    args = parser.parse_args()
    train(episodes=args.episodes, env_type=args.env, seed=args.seed)

if __name__ == "__main__":
    main()
