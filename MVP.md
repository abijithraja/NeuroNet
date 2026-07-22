# NeuroNet Modulator (MVP)

## Overview

NeuroNet Modulator MVP is a **simulation-based implementation** of an AI-driven self-healing network system.

It demonstrates the core intelligence loop:

> Detect → Decide → Act → Learn

---

## What This MVP Shows

* Simulated network telemetry
* Anomaly detection
* Reinforcement learning-based decisions
* Automated action execution
* Feedback-driven improvement

---

## Core Pipeline

Simulated Network
→ Anomaly Detection
→ RL Decision
→ Action Execution
→ Metrics Improvement

---

## Features

* Detects anomalies in latency, packet loss, congestion
* Selects actions (reroute, QoS, scaling, etc.)
* Improves network performance using rewards
* Logs decisions and outputs

---

## Project Structure

```
NeuroNet/
├── main_simulation.py
├── telemetry_emulator.py
├── anomaly_detector.py
├── dqn_agent.py
├── reward_function.py
├── decision_log.json
└── models/
```

---

## How to Run

### 1. Install Dependencies

```bash
pip install torch scikit-learn numpy
```

---

### 2. Run Simulation

```bash
python main_simulation.py
```

---

## Expected Output

* Anomaly detected
* RL agent selects action
* Metrics improve
* Reward calculated

Example:

```
Anomaly: congestion
Action: reroute
Latency: 30ms → 12ms
Reward: +41.5
```

---

## Key Insight

This MVP proves that:

* AI can detect network issues
* RL can make intelligent decisions
* Automated actions improve performance

---

## Limitations

* Uses simulated network (no real infrastructure)
* Simplified RL implementation
* No Kafka / Kubernetes integration

---

## Future Scope

* Real telemetry integration (Kafka, Prometheus)
* Kubernetes-based deployment
* SDN-based execution (OpenFlow)
* Scalable RL training

---

## Architecture Reference

For full system design, see:
 `ARCHITECTURE.md`

---

## Conclusion

This MVP validates the feasibility of a **self-healing network using reinforcement learning**, forming the foundation for a production-scale NeuroNet system.



## Reinforcement Learning (MVP Design)

### State Representation (Input)

In the MVP, the RL agent uses a simplified state vector based on core network metrics.

**Format:**

```python
state = [
    latency,
    packet_loss,
    congestion,
    anomaly_score,
    last_action
]
```

---

### Action Output

The agent selects a discrete action with a confidence score.

**Format:**

```python
{
    "action": str,
    "confidence": float
}
```

---

### Action Space (MVP)

| ID | Action              |
| -- | ------------------- |
| 0  | Reroute             |
| 1  | QoS adjustment      |
| 2  | Scale resources     |
| 3  | Monitor (no action) |

---

### Decision Timing

* Simulation loop interval: **1–2 seconds**
* RL inference latency: **few milliseconds**

**Decision Loop:**
The system continuously evaluates and acts in a fast simulation loop to demonstrate real-time behavior.

---

### Design Note

The MVP preserves the core RL decision-making logic while reducing state complexity and action space for faster execution and demonstration.
