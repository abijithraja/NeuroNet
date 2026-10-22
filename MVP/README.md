# NeuroNet — Starter Implementation

**Project status:** starter code for a new implementation. This repository does not claim that NeuroNet is already validated, GPU-accelerated, or connected to a live network.

NeuroNet's intended loop is:

`Telemetry → Anomaly Detection → Dueling DQN → Safety Gate → Simulated Remediation → Reward Feedback`

The starter provides:
- A 20-node **Python toy environment** to test the RL loop immediately.
- Isolation Forest anomaly detection (trained on synthetic baseline samples).
- A Dueling DQN with experience replay and target network.
- Confidence-gated action handling and a JSONL audit log.
- An ns-3 CSV adapter and an example ns-3 C++ scratch program that writes flow-level metrics.

**Important:** The Python toy environment is not ns-3. Use it for code development only. For network-simulation evidence, build and run the included ns-3 scenario, or export telemetry from your own ns-3 experiment and use the adapter.

## Folder layout

```text
NeuroNet/
├── README.md
├── requirements.txt
├── .gitignore
├── config.py
├── main.py
├── train_agent.py
├── evaluate_agent.py
├── src/
│   └── neuronet/
│       ├── __init__.py
│       ├── actions.py
│       ├── toy_env.py
│       ├── telemetry.py
│       ├── anomaly_detector.py
│       ├── dqn_agent.py
│       ├── safety.py
│       ├── reward.py
│       ├── audit.py
│       └── ns3_csv.py
├── ns3/
│   └── scratch/
│       └── neuronet-telemetry.cc
├── tests/
│   └── test_core.py
├── data/
│   └── .gitkeep
├── models/
│   └── .gitkeep
└── logs/
    └── .gitkeep
```

## 1. Create a Python environment on Windows

Install Python 3.10 or 3.11 from python.org and enable **Add Python to PATH** during installation. Open PowerShell in this folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that terminal, then activate again.

## 2. Run the starter loop (Python toy environment)

```powershell
python main.py --episodes 10
```

This tests the software pipeline and saves decisions to `logs/decisions.jsonl`. It does **not** use ns-3 telemetry.

## 3. Train the DQN

```powershell
python train_agent.py --episodes 200
```

The trained weights are written to `models/dueling_dqn.pt`. Training quality is not guaranteed; inspect the evaluation output rather than assuming the agent learned useful behaviour.

## 4. Evaluate a saved model

```powershell
python evaluate_agent.py --episodes 30
```

## 5. Run tests

```powershell
python -m unittest discover -s tests -v
```

## 6. ns-3 integration

The included C++ example is a small ns-3 scenario using a point-to-point link and UDP traffic. It writes `data/ns3_telemetry.csv` containing flow-level throughput, delay and packet-loss estimates from FlowMonitor.

1. Install ns-3 following the official ns-3 installation guide for your OS.
2. Copy `ns3/scratch/neuronet-telemetry.cc` into the `scratch/` directory of your ns-3 source tree.
3. Build/run it from the ns-3 source directory using the build runner, for example:

   ```bash
   ./ns3 run scratch/neuronet-telemetry
   ```

   Exact commands can vary by ns-3 release and installation method.
4. Copy the generated `data/ns3_telemetry.csv` into this project's `data/` directory (or change the output path in the C++ example).
5. Inspect it with:

   ```powershell
   python -c "from src.neuronet.ns3_csv import load_ns3_csv; print(load_ns3_csv('data/ns3_telemetry.csv').head())"
   ```

The included adapter normalizes the example CSV into columns expected by the telemetry/anomaly modules. This is an initial offline trace bridge, not live control of ns-3.

## What is and is not implemented

| Component | Status in this starter |
|---|---|
| Dueling DQN, replay buffer, target network | Implemented starter code |
| Synthetic 20-node environment | Implemented toy environment |
| Isolation Forest anomaly detector | Implemented on synthetic baseline data |
| Confidence/safety gate | Implemented conservative prototype; its confidence is a heuristic |
| Decision audit log | Implemented |
| ns-3 C++ traffic/FlowMonitor example | Included; must be built and run in ns-3 |
| CSV trace adapter | Implemented |
| Real network remediation/OpenFlow/Kubernetes/Ansible | Not implemented |
| LSTM autoencoder + SHAP explanations | Not included yet |
| Verified performance claims | None; collect experiments before reporting results |

## Suggested development order

1. Get the Python toy loop and tests running.
2. Run the ns-3 example and inspect its CSV.
3. Replace toy telemetry with ns-3 trace episodes.
4. Train/evaluate DQN across repeatable seeded fault scenarios.
5. Add stronger confidence calibration, rollback simulation, and explainability.
6. Only then consider a dashboard or real SDN controller integration.

Never connect experimental actions to a production network. Keep the first version entirely simulated.
