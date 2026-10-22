# NeuroNet MVP Function Report (Current State)

Date: April 13, 2026

## 1) What this report covers

This report explains what the MVP files currently do, how the full flow works end-to-end, and what is already implemented versus what is still design/vision.

Reviewed areas:
- MVP docs and architecture docs
- Simulation core code in Models
- Dashboard and launcher scripts
- Runtime behavior from an actual simulation run

## 2) MVP in one sentence

Your current MVP is a local simulation of a self-healing network loop:

Telemetry generation -> anomaly scoring -> RL action selection -> confidence gate (auto/escalate) -> reward calculation -> replay training -> decision logging + dashboard visualization.

## 3) How the current code flow works

### Step A: Telemetry generation

File: Models/telemetry_emulator.py

What it does now:
- Creates 20 simulated links (link_0 to link_19).
- Keeps baseline metrics per link:
  - latency: 5 ms
  - throughput: 8 Gbps
  - packet loss: 0.1%
- Generates noisy normal telemetry.
- Can inject 3 anomaly types:
  - congestion
  - link_failure
  - degradation
- Outputs one metrics dictionary with fields like latency_ms, throughput_gbps, packet_loss_pct, congestion, and is_anomaly.

### Step B: Anomaly detection

File: Models/anomaly_detector.py

What it does now:
- Uses two detectors together:
  - LSTM Autoencoder reconstruction error
  - Isolation Forest score
- Trains on benign windows (fast demo training loop).
- Uses a sliding metrics window (default size = 5).
- Computes:
  - lstm_score
  - iso_score
  - anomaly_score = 0.6 * lstm_score + 0.4 * iso_score
- Flags anomaly if score > 0.7.
- Returns useful feature flags:
  - latency_anomalous
  - loss_anomalous
  - congestion_high

### Step C: RL state encoding

File: Models/main_simulation.py

What it does now:
- Encodes detector output into an 8-dimensional state vector:
  1. anomaly_score
  2. lstm_error
  3. iso_score
  4. latency_anomalous (0/1)
  5. loss_anomalous (0/1)
  6. congestion_high (0/1)
  7. sin(hour) time feature
  8. lightweight action-history feature

Important note:
- The docs often describe a larger production-like state (for example 35 dimensions), but current MVP runtime uses this 8-dim compact state.

### Step D: RL decision model

File: Models/dqn_agent.py

What it does now:
- Implements a Dueling DQN network.
- Supports 8 actions:
  0 reroute
  1 qos_adjust
  2 scale_slice
  3 throttle
  4 traffic_shaping
  5 failover
  6 escalate
  7 monitor
- Action selection policy:
  - epsilon-greedy exploration (starts 0.30, decays)
  - predicted-action confidence via softmax on Q-values
- Uses experience replay buffer (size 5000).
- Trains with SmoothL1Loss (Huber) and soft target updates.

### Step E: Safety gate and action effect simulation

File: Models/main_simulation.py

What it does now:
- Uses confidence threshold 0.75.
- If confidence < 0.75, marks decision as escalated to NOC.
- Applies fixed action effect tables (latency/throughput/loss deltas) to produce post-action metrics.
- This is simulated execution (not real OpenFlow/K8s/Ansible dispatch yet).

### Step F: Reward computation

File: Models/reward_function.py

What it does now:
- Computes reward from metric deltas and action cost:
  - latency improvement
  - throughput improvement
  - packet loss reduction
  - action penalty/cost
  - optional SLA penalty
- Combines weighted terms and clips final reward to [-100, 100].

### Step G: Learning loop + artifact output

File: Models/main_simulation.py

What it does now:
- Runs episodes (default 30 in fast demo mode).
- For each episode:
  - remembers transition in replay buffer
  - trains mini-batch
  - appends detailed decision log record
- Saves outputs:
  - decision_log.json
  - models/pretrained_dqn.pt

## 4) Dashboard and run flow

### Streamlit dashboard

File: Models/app.py

What it does now:
- Provides a dashboard with periodic auto-refresh.
- Sidebar button runs main_simulation.py.
- Loads decision_log.json and computes "Latency Improvement (ms)".
- Shows:
  - KPI cards (incidents, autonomy rate, avg confidence, avg latency fix)
  - reward and latency charts
  - action distribution and escalation by anomaly type
  - filter by anomaly type
  - full styled decision log table

### Unified launcher

File: Models/run.py

What it does now:
- Runs simulation first.
- Then launches Streamlit app automatically.

## 5) Actual runtime behavior observed (validated run)

Command executed:
- python main_simulation.py (inside neuro-MVP/Models)

Observed outcomes:
- Simulation ran successfully for 30 episodes.
- Anomaly detector training completed.
- Decision engine executed and printed per-episode action/reward/confidence.
- Final summary reported by anomaly type:
  - CONGESTION: mean reward +5.18, mean confidence 0.672, escalation 40%
  - DEGRADATION: mean reward +5.12, mean confidence 0.840, escalation 20%
  - LINK_FAILURE: mean reward +5.56, mean confidence 0.625, escalation 40%
- Artifacts saved successfully:
  - decision_log.json
  - models/pretrained_dqn.pt

## 6) MVP docs vs current implementation (important clarity)

Your documentation in Flow and neuro-MVP describes a broad production architecture (Kafka, FastAPI, OpenFlow, Kubernetes operators, SHAP-heavy explainability, cloud deployment, MANET extension, etc.).

Current implemented MVP code is a simulation-first proof:
- Implemented now:
  - telemetry simulator
  - anomaly detection model combo
  - Dueling DQN training/inference loop
  - confidence-based escalation logic
  - reward feedback learning cycle
  - JSON decision audit trail
  - Streamlit dashboard visualization
- Not yet implemented in runtime code:
  - real network telemetry ingestion (Kafka/Prometheus pipeline)
  - live infrastructure dispatch (OpenFlow/Ansible/K8s operators)
  - production API layer (FastAPI serving endpoints)
  - distributed/multi-GPU production training workflow

So the MVP successfully proves the decision loop logic, while full infra integration remains future build scope.

## 7) File-by-file purpose map (MVP-relevant)

### In neuro-MVP
- MVP.md: high-level MVP concept, loop, limitations, and expected output.
- neuronet_modulator MVP version.md: detailed architecture narrative and benchmark-style positioning.
- README.md: quick summary of pipeline and run instructions.

### In neuro-MVP/Models
- telemetry_emulator.py: synthetic telemetry + anomaly injection.
- anomaly_detector.py: LSTM autoencoder + Isolation Forest detector.
- dqn_agent.py: Dueling DQN agent, replay memory, training.
- reward_function.py: reward calculation with action costs.
- main_simulation.py: orchestrates full MVP episode loop and output persistence.
- app.py: Streamlit dashboard for viewing outcomes.
- run.py: simulation + dashboard launcher.
- decision_log.json: generated incident/action audit output.
- models/pretrained_dqn.pt: saved trained model weights.
- requirements.txt: required Python packages.

### In Flow
- MVP.md: short architecture summary.
- ARCHITECTURE.md: full architecture layers and CLI-first positioning.
- neuronet_modulator MVP version.md: expanded product-level narrative.
- neuronet MVP.html: visual architecture flowchart for presentations.

## 8) Current maturity assessment

Status today:
- MVP core loop is working and runnable.
- Learning + logging + visualization are integrated.
- The system demonstrates autonomy with confidence-based escalation.
- It is still simulation-grade, not production-integrated network control yet.

This is a solid proof-of-concept stage that validates the AI control logic and provides measurable outputs for demo/judging.
