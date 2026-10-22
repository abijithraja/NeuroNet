# NeuroNet — Documentation & Research Index

This directory contains research papers, technical architecture documents, hackathon submission proposals, and interactive visual flowcharts for the **NeuroNet** GPU-accelerated, reinforcement learning-driven self-healing network control plane.

---

## Quick Reference Table

| File Name | Format | Primary Role | Core Highlights |
| :--- | :--- | :--- | :--- |
| [ARCHITECTURE.md](file:///c:/Users/ABIJITH%20RAJA%20B/Desktop/Neuronet/Documentation%20and%20Research/ARCHITECTURE.md) | Markdown | Production System Architecture | 6-layer architecture, Prometheus/Kafka/InfluxDB, OpenFlow/K8s/Ansible execution, CLI-first design. |
| [MVP.md](file:///c:/Users/ABIJITH%20RAJA%20B/Desktop/Neuronet/Documentation%20and%20Research/MVP.md) | Markdown | MVP Specification & RL Design | 5-dim state vector, 4 discrete actions, simulation loop, limitations, and verification workflow. |
| [Neuronet Rl agent working.pdf](file:///c:/Users/ABIJITH%20RAJA%20B/Desktop/Neuronet/Documentation%20and%20Research/Neuronet%20Rl%20agent%20working.pdf) | PDF (10 pages) | Academic Technical Paper | Formal MDP formulation, 35-dim state with SHAP, 8-action Dueling DQN, 3-head ensemble safety gate, Mininet benchmarks. |
| [Neuronet modulator Final.docx](file:///c:/Users/ABIJITH%20RAJA%20B/Desktop/Neuronet/Documentation%20and%20Research/Neuronet%20modulator%20Final.docx) | Word (.docx) | Project Documentation | Final year research project document. |
| [Neuronet modulator Final.md](file:///c:/Users/ABIJITH%20RAJA%20B/Desktop/Neuronet/Documentation%20and%20Research/Neuronet%20modulator%20Final.md) | Markdown | Complete Submission Document | 14-part full proposal: market gap, architecture, cold-start strategy, Mininet benchmarks, SmartGPU mapping, and run guide. |
| [neuronet_modulator original.md](file:///c:/Users/ABIJITH%20RAJA%20B/Desktop/Neuronet/Documentation%20and%20Research/neuronet_modulator%20original.md) | Markdown | Master Reference & Deep Dive | Exhaustive technical manuscript containing comprehensive background, algorithms, equations, and full architectural rationale. |
| [neuronet_modulator MVP version.md](file:///c:/Users/ABIJITH%20RAJA%20B/Desktop/Neuronet/Documentation%20and%20Research/neuronet_modulator%20MVP%20version.md) | Markdown | MVP-Focused Project Spec | Tailored hackathon version linking production design with working MVP code compromises and CLI workflows. |
| [Neuronet original.html](file:///c:/Users/ABIJITH%20RAJA%20B/Desktop/Neuronet/Documentation%20and%20Research/Neuronet%20original.html) | HTML / CSS | Interactive Pipeline Flowchart | Web-based visual diagram of the end-to-end telemetry ingestion, anomaly detection, RL inference, and feedback loop. |
| [neuronet MVP.html](file:///c:/Users/ABIJITH%20RAJA%20B/Desktop/Neuronet/Documentation%20and%20Research/neuronet%20MVP.html) | HTML / CSS | Enhanced Flowchart & KPI Banner | Adds executive impact metrics (47–52ms MTTR, 85% SLA breach reduction, 87% auto-resolution) and telemetry metric chips. |
| [neuronet_agent.html](file:///c:/Users/ABIJITH%20RAJA%20B/Desktop/Neuronet/Documentation%20and%20Research/neuronet_agent.html) | HTML / JS / Canvas | Interactive Node-Graph Studio | Visual node editor with draggable components (Kafka, LSTM, DQN, OpenFlow), live execution simulations, and telemetry inspector. |

---

## Detailed File Summaries

### 1. [ARCHITECTURE.md](file:///c:/Users/ABIJITH%20RAJA%20B/Desktop/Neuronet/Documentation%20and%20Research/ARCHITECTURE.md)
* **Format:** Markdown (`.md`)
* **Purpose:** Concise engineering blueprint of the full-scale NeuroNet production platform.
* **Gist:**
  - Explains the central paradigm: **Detect → Decide → Act → Learn**.
  - Details the **6 functional system layers**:
    1. *Telemetry Ingestion Layer*: Prometheus (15s scrape), Kafka (event streaming), InfluxDB (time-series retention).
    2. *Anomaly Detection Layer*: LSTM Autoencoder (temporal sequences) + Isolation Forest (outliers) with SHAP attribution.
    3. *RL Intelligence Core*: CUDA-accelerated Dueling DQN / PPO policy with Redis replay buffers.
    4. *Execution Layer*: OpenFlow (SDN routing), Kubernetes (5G slice autoscaling), and Ansible (hardware QoS).
    5. *API Layer*: FastAPI microservices powering both the CLI and dashboards.
    6. *Observation Layer*: Real-time React / Grafana monitoring.
  - Highlights a **CLI-first philosophy** enabling headless DevOps and automated network pipelines with sub-50ms decision latency.

---

### 2. [MVP.md](file:///c:/Users/ABIJITH%20RAJA%20B/Desktop/Neuronet/Documentation%20and%20Research/MVP.md)
* **Format:** Markdown (`.md`)
* **Purpose:** Implementation specification for the self-contained prototype and demonstration pipeline.
* **Gist:**
  - Defines the simplified RL framework for offline testing without requiring full-scale SDN or hardware:
    - **State Vector (5-D)**: `[latency, packet_loss, congestion, anomaly_score, last_action]`.
    - **Discrete Actions (4 options)**: `Reroute (0)`, `QoS adjustment (1)`, `Scale resources (2)`, `Monitor/No action (3)`.
    - **Decision Cycle**: 1–2 second emulation tick with millisecond inference.
  - Outlines the step-by-step execution flow, simulated network telemetry, reward computation, and file layout (`main_simulation.py`, `telemetry_emulator.py`, `anomaly_detector.py`, `dqn_agent.py`, `reward_function.py`).
  - Identifies prototype boundaries (synthetic traffic, absence of Kafka/Kubernetes clusters) and the path toward production integration.

---

### 3. [Neuronet Rl agent working.pdf](file:///c:/Users/ABIJITH%20RAJA%20B/Desktop/Neuronet/Documentation%20and%20Research/Neuronet%20Rl%20agent%20working.pdf)
* **Format:** Academic PDF Document (10 pages)
* **Title:** *Design and Evaluation of a Reinforcement-Learning Agent for Self-Healing Network Control: NeuroNet*
* **Purpose:** Formal academic paper detailing the reinforcement learning agent's mathematical modeling, architecture, and empirical Mininet validation.
* **Gist:**
  - **MDP Formulation:** Formulates network self-healing as a Markov Decision Process $(S, A, P, R, \gamma)$.
  - **Perception Pipeline:** Ingests streaming metrics via Prometheus/Kafka, evaluates anomalies using an LSTM Autoencoder + Isolation Forest ensemble, and computes SHAP feature attributions.
  - **State Space (35 dimensions):** Anomaly deltas, SHAP weights, node ID, impacted flow counts, sinusoidal time-of-day encoding, and recent action memory.
  - **Action Space (8 actions):** ECMP rerouting, QoS queue priority, 5G slice scale-up, cross-slice interference reduction, traffic shaping, local failover, NOC human escalation, and monitoring.
  - **Dueling DQN Architecture:** Value stream $V(s)$ and Advantage stream $A(s,a)$ combined via mean subtraction; trained with Huber loss and experience replay.
  - **Ensemble Safety Gate:** 3-head DQN variance estimation; actions execute autonomously only if confidence $\ge 0.75$, with 15% safe escalation to NOC.
  - **Multi-Component Reward Function:** Balances latency drop, throughput gain, packet loss reduction, congestion alleviation, SLA penalties, action execution costs, and a $-10$ oscillation penalty.
  - **4-Phase Training & Scaling Results:** Benchmarked in Mininet across 20-node linear chain and 50/80-node fat-tree topologies, confirming 47–51 ms median GPU latency and $>89\%$ anomaly detection accuracy under link congestion, BGP flapping, and link failure scenarios.

---

### 4. [Neuronet modulator Final.docx](file:///c:/Users/ABIJITH%20RAJA%20B/Desktop/Neuronet/Documentation%20and%20Research/Neuronet%20modulator%20Final.docx)
* **Format:** Microsoft Word (`.docx`)
* **Purpose:** Final year research project document.
* **Gist:**
  - The formatted document equivalent of `Neuronet modulator Final.md`.
  - Structured for project review, detailing executive summaries, problem statements, business impact tables, architectural diagrams, empirical benchmarks, and future commercialization avenues.

---

### 5. [Neuronet modulator Final.md](file:///c:/Users/ABIJITH%20RAJA%20B/Desktop/Neuronet/Documentation%20and%20Research/Neuronet%20modulator%20Final.md)
* **Format:** Markdown (`.md`)
* **Purpose:** Comprehensive 14-part hackathon project submission document.
* **Gist:**
  - **Part 1 — Problem Statement:** Details reactive NOC timelines, static configs, and CPU-bound rule engine bottlenecks ($9M–$45M annual breach costs for 2M subscriber telcos).
  - **Part 2 — Solution Architecture:** Detailed breakdown of the 5 core layers and component workflows.
  - **Part 3 — Cold-Start & Training Strategy:** 4 phases from 5,000 synthetic Mininet episodes to live micro-batch retraining every 100 incidents.
  - **Part 4 — Experimental Evaluation:** In-depth benchmarks across 20-, 50-, and 80-node topologies, detailing p50/p95 latency and scenario-specific recovery metrics.
  - **Part 5 — Progressive Autonomy Model:** Phase 1 (100% human review) $\rightarrow$ Phase 2 (assisted, 85% autonomous, 15% escalated) $\rightarrow$ Phase 3 (full autonomy, $<5\%$ escalation).
  - **Part 6 — MANET Positioning:** Extending topological graph features to Mobile Ad-Hoc Networks.
  - **Part 7–14 — MVP Scope, Benchmarks, Tech Stack, Safety Rules & Quick Start:** Complete implementation checklist, failure modes, rollback mechanisms, and execution instructions.

---

### 6. [neuronet_modulator original.md](file:///c:/Users/ABIJITH%20RAJA%20B/Desktop/Neuronet/Documentation%20and%20Research/neuronet_modulator%20original.md)
* **Format:** Markdown (`.md`)
* **Purpose:** The foundational, unabridged master reference document of the NeuroNet research project.
* **Gist:**
  - Contains extensive technical background, mathematical loss equations, SmartGPU architecture transfer theory, and granular system diagrams.
  - Covers theoretical comparisons between rule-based heuristics, classical supervised ML models, and deep reinforcement learning control loops.
  - Serves as the primary source from which the hackathon proposal and academic paper were distilled.

---

### 7. [neuronet_modulator MVP version.md](file:///c:/Users/ABIJITH%20RAJA%20B/Desktop/Neuronet/Documentation%20and%20Research/neuronet_modulator%20MVP%20version.md)
* **Format:** Markdown (`.md`)
* **Purpose:** Hackathon documentation version oriented around the runnable MVP code deliverables.
* **Gist:**
  - Bridges the production architecture with the concrete code implementation in the workspace.
  - Contrasts full production components (Kafka, InfluxDB, OpenFlow controllers) with their MVP counterparts (synthetic telemetry generators, scikit-learn models, local PyTorch checkpoints, and JSON logs).
  - Provides step-by-step setup and evaluation guides tailored for demonstration judges.

---

### 8. [Neuronet original.html](file:///c:/Users/ABIJITH%20RAJA%20B/Desktop/Neuronet/Documentation%20and%20Research/Neuronet%20original.html)
* **Format:** HTML / CSS
* **Purpose:** Clean, web-based visual pipeline and architecture presentation diagram.
* **Gist:**
  - Renders a styled vertical flowchart of the NeuroNet data lifecycle using modern typography and dark-mode styling.
  - Illustrates inputs, processing layers, decision points, GPU-accelerated steps, action dispatchers, and feedback/retraining channels.

---

### 9. [neuronet MVP.html](file:///c:/Users/ABIJITH%20RAJA%20B/Desktop/Neuronet/Documentation%20and%20Research/neuronet%20MVP.html)
* **Format:** HTML / CSS
* **Purpose:** Enhanced, presentation-ready dashboard flowchart optimized for demo showcases.
* **Gist:**
  - Adds a prominent **Executive Impact KPI Banner**:
    - **47–52 ms**: RL-driven MTTR
    - **85%**: SLA breach reduction
    - **87%**: Anomalies auto-resolved
    - **<50 ms**: GPU inference latency
  - Expands each architectural card with telemetry metric tags (RTT latency, packet loss ratio, buffer utilization, device temperature) and execution pathway timings.

---

### 10. [neuronet_agent.html](file:///c:/Users/ABIJITH%20RAJA%20B/Desktop/Neuronet/Documentation%20and%20Research/neuronet_agent.html)
* **Format:** Single-page Interactive Web Application (HTML / CSS / JavaScript Canvas)
* **Purpose:** Visual interactive node-graph studio and real-time execution workbench.
* **Gist:**
  - **Interactive Canvas:** Draggable, connectable nodes representing each component in the NeuroNet ecosystem:
    - *Telemetry*: Prometheus Scraper, Kafka Ingestion Broker, SNMP Listener.
    - *Anomaly Detection*: LSTM Autoencoder & Isolation Forest ensemble, SHAP Explainer.
    - *RL Engine*: Dueling DQN Core, Confidence Safety Gate, Experience Replay Buffer.
    - *Execution*: OpenFlow SDN Dispatcher, Kubernetes Slice Autoscaler, Ansible Playbook Runner.
    - *Governance*: ElasticSearch Audit Logger, Human NOC Escalation Queue.
  - **Live Simulation Controls:** Run and inspect step-by-step decisions, view node connection states, review synthetic telemetry graphs, and observe real-time reward outputs directly inside the browser.
