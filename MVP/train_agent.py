import argparse
from pathlib import Path
import numpy as np
from config import CONFIG, MODEL_DIR
from src.neuronet.actions import N_ACTIONS
from src.neuronet.toy_env import ToyNetworkEnv
from src.neuronet.dqn_agent import DQNAgent

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--seed", type=int, default=CONFIG.seed)
    args = p.parse_args()
    env = ToyNetworkEnv(seed=args.seed, num_nodes=CONFIG.num_nodes)
    agent = DQNAgent(CONFIG.state_dim, N_ACTIONS, hidden_dim=CONFIG.hidden_dim,
        capacity=CONFIG.replay_capacity, batch_size=CONFIG.batch_size,
        gamma=CONFIG.gamma, lr=CONFIG.learning_rate, device=CONFIG.device,
        target_update_every=CONFIG.target_update_every, seed=args.seed)
    total_steps = max(1, args.episodes)
    returns = []
    for ep in range(args.episodes):
        state, info = env.reset()
        epsilon = CONFIG.epsilon_end + (CONFIG.epsilon_start - CONFIG.epsilon_end) * np.exp(
            -agent.steps / CONFIG.epsilon_decay_steps)
        action = agent.act(state, epsilon=float(epsilon))
        next_state, reward, done, _, step_info = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        loss = agent.learn()
        returns.append(reward)
        if (ep + 1) % 20 == 0 or ep == 0:
            avg = float(np.mean(returns[-20:]))
            print(f"episode={ep+1:4d}/{args.episodes} epsilon={epsilon:.3f} "
                  f"recent_mean_reward={avg:.3f} loss={loss}")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    out = MODEL_DIR / "dueling_dqn.pt"
    agent.save(out)
    print(f"Saved starter weights to {out}")
    print("A saved checkpoint is not proof of convergence; compare seeded evaluation runs.")

if __name__ == "__main__":
    main()
