# NeuroNet — Autonomous Self-Healing Network Control Plane

**NeuroNet** is an AI-driven, GPU-accelerated reinforcement learning control plane designed to transform reactive network management into an autonomous, closed-loop self-healing system.

```text
Telemetry Ingestion (ns-3 / Realistic Topology)
       │
       ▼
Anomaly Detection (LSTM Autoencoder + Isolation Forest Ensemble)
       │
       ▼
State Encoding (14-D Telemetry & Attribution Vector)
       │
       ▼
RL Decision Engine (Dueling DQN Policy)
       │
       ▼
Safety Gate (Confidence Threshold >= 75% & Anti-Thrashing Guard)
       ├── [Approved] ──> Autonomous Remediation Execution (<50ms)
       └── [Escalated] ─> Human NOC Operator Review
       │
       ▼
Multi-Objective Reward Feedback & Continuous Experience Replay Learning
       │
       ▼
Explainability (SHAP-style Feature Attribution) & Streamlit Live Dashboard
```

---

## 6-Phase Architecture

* **Phase 1 — Preserve & Strengthen Baseline Foundation**: Maintained and unified the Streamlit dashboard, telemetry emulator, LSTM Autoencoder, Isolation Forest, Dueling DQN agent, reward function, and JSON/JSONL audit logging into a single cohesive package.
* **Phase 2 — Realistic Network Telemetry & ns-3**: Built a dynamic 20-node network simulation environment (`network_sim.py`) that models M/M/1 queuing delays, buffer overflows, and 6 realistic failure scenarios (Congestion, Link Failure, Routing Flap, DDoS Attack, Degradation, and Nominal baseline), backed by ns-3 FlowMonitor trace adapters (`ns3_csv.py` and `neuronet-telemetry.cc`).
* **Phase 3 — Intelligent Decision-Making Integration**: Standardized state representation (14 dimensions), discrete action space (8 remediation actions), and unified anomaly intelligence directly into the Dueling DQN policy.
* **Phase 4 — Safe Self-Healing & Governance**: Added a confidence-gated safety gate (`safety.py`) enforcing confidence thresholds ($\ge 75\%$), anti-thrashing/oscillation guards, and physical risk checks, escalating uncertain decisions to human operators.
* **Phase 5 — Explainability & Audit History**: Integrated SHAP-inspired feature attribution breakdowns (`explainability.py`) and natural language justifications with dual audit storage (`logs/decisions.jsonl` and `logs/decision_log.json`).
* **Phase 6 — Training, Evaluation & Benchmarking**: Added multi-scenario training (`train_agent.py`) and side-by-side benchmarking (`evaluate_agent.py`) comparing NeuroNet against a Static Rule Engine, Random Agent, and Passive Monitor, generating markdown evaluation reports (`logs/EVALUATION_REPORT.md`).

---

## Folder Structure

```text
Main/
├── README.md               # System documentation and instructions
├── requirements.txt        # Python package dependencies
├── config.py               # Central settings and hyperparameters
├── main.py                 # End-to-end autonomous incident and remediation loop
├── train_agent.py          # Dueling DQN multi-scenario training pipeline
├── evaluate_agent.py       # Benchmarks against Rule Engine, Random, and Passive baselines
├── app.py                  # Interactive Streamlit dashboard
├── run.py                  # One-click launcher (populates telemetry & launches UI)
├── decision_log.json       # Live decision audit log for UI consumption
├── src/
│   └── neuronet/
│       ├── __init__.py
│       ├── actions.py          # 8 discrete actions, costs, and inverse mappings
│       ├── telemetry.py        # Telemetry validation & 14-D state constructor
│       ├── anomaly_detector.py # LSTM Autoencoder + Isolation Forest ensemble
│       ├── dqn_agent.py        # Dueling DQN, replay buffer, Double-DQN target network
│       ├── network_sim.py      # Realistic 20-node network queuing & scenario simulator
│       ├── toy_env.py          # Lightweight baseline toy environment
│       ├── safety.py           # Safety gate, confidence gating, anti-thrashing
│       ├── reward.py           # Multi-objective bounded reward calculation
│       ├── explainability.py   # SHAP-inspired attribution & decision rationale
│       ├── audit.py            # JSONL and JSON audit log handlers
│       └── ns3_csv.py          # ns-3 FlowMonitor CSV ingestion
├── ns3/
│   └── scratch/
│       └── neuronet-telemetry.cc # ns-3 C++ traffic scenario
├── tests/
│   └── test_core.py            # Unit test suite covering all 6 phases
├── data/
│   └── ns3_telemetry_sample.csv # Pre-generated sample ns-3 FlowMonitor trace
├── models/
│   ├── dueling_dqn.pt          # Trained Dueling DQN checkpoint
│   └── pretrained_dqn.pt       # Pretrained weights checkpoint
└── logs/
    ├── decision_log.json       # Structured decision records
    ├── decisions.jsonl         # Append-only line audit trail
    └── EVALUATION_REPORT.md    # Generated empirical benchmark report
```

---

## How to Run

### 1. Run Autonomous Incident Loop (CLI)
```powershell
python main.py --episodes 15 --env realistic --scenario all
```

### 2. Train Policy on Realistic Scenarios
```powershell
python train_agent.py --episodes 200 --env realistic
```

### 3. Run Benchmark Evaluation vs Baseline Strategies
```powershell
python evaluate_agent.py --episodes 10
```

### 4. Launch Interactive Streamlit Dashboard
```powershell
# Option A: One-click launcher
python run.py

# Option B: Direct Streamlit command
streamlit run app.py
```

### 5. Run Unit Tests
```powershell
python -m unittest tests/test_core.py
```
