import argparse
from collections import defaultdict
import numpy as np
from config import CONFIG, MODEL_DIR
from src.neuronet.actions import N_ACTIONS, ACTION_NAMES, Action
from src.neuronet.toy_env import ToyNetworkEnv
from src.neuronet.dqn_agent import DQNAgent
from src.neuronet.safety import apply_safety_gate

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--model", default=str(MODEL_DIR / "dueling_dqn.pt"))
    p.add_argument("--seed", type=int, default=123)
    args = p.parse_args()
    agent = DQNAgent(CONFIG.state_dim, N_ACTIONS, hidden_dim=CONFIG.hidden_dim,
                     device=CONFIG.device, seed=args.seed)
    agent.load(args.model)
    env = ToyNetworkEnv(seed=args.seed, num_nodes=CONFIG.num_nodes)
    results = defaultdict(list)
    for ep in range(args.episodes):
        state, info = env.reset()
        action_id = agent.act(state, epsilon=0.0)
        confidence = agent.heuristic_confidence(state)
        gate = apply_safety_gate(action_id, confidence, CONFIG.confidence_threshold)
        action = Action(action_id) if gate.approved else Action.ESCALATE
        _, reward, _, _, step = env.step(action)
        results[info["fault"]].append((reward, gate.approved, ACTION_NAMES[action]))
    print("Evaluation on synthetic toy environment (not ns-3):")
    for fault, rows in sorted(results.items()):
        print(f"{fault:14} n={len(rows):3} mean_reward={np.mean([r[0] for r in rows]):8.3f} "
              f"approved_rate={np.mean([r[1] for r in rows]):.2f} "
              f"actions={dict((a, sum(1 for x in rows if x[2] == a)) for a in set(x[2] for x in rows))}")
    print("Run multiple seeds and compare with a rule-based baseline before making claims.")

if __name__ == "__main__":
    main()
