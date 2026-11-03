# NeuroNet Phase 1: Code Audit & Implementation Matrix

## 1. Code Audit of the 8 Key Subsystems

| # | File | Current Implementation | Critical Finding / Limitation | Action Required |
|---|---|---|---|---|
| **1** | [`src/neuronet/toy_env.py`](file:///c:/Users/ABIJITH%20RAJA%20B/Desktop/Neuronet/MVP/src/neuronet/toy_env.py) | Maintains scalar `Metrics` (`latency_ms`, `throughput_mbps`, `packet_loss_pct`, `congestion`, `link_health`). Steps multiply scalars by hardcoded factors (e.g., `latency *= 0.58`). | **No real node or queue objects.** The 20 nodes only exist as an integer count (`num_nodes=20`). No queue accumulation, packet flows, or routing paths exist. | Replace with modular `network_engine/simulation.py` simulating genuine node queues, link bandwidths, and traffic matrix. |
| **2** | [`live_runner/server.py`](file:///c:/Users/ABIJITH%20RAJA%20B/Desktop/Neuronet/MVP/live_runner/server.py) | 20-node dictionary array (x, y coords), REST endpoints. When fault injected, assigns predetermined metrics (`latency=114.5ms`, `node buffer=92%`). In `agent_decide`, manually boosts Q-values (`q_raw[1] += 25.0`, `q_raw[4] += 50.0`). | **Artificial AI bias and synthetic recovery.** The server forces the AI's decision based on the fault button rather than the agent inferring the action from observed state features. Recovery resets all nodes to nominal with random floats. | Remove manual Q-value manipulation completely. Hook `server.py` to `network_engine` so telemetry and recovery reflect actual physical queue/routing state changes. |
| **3** | [`live_runner/web/app.js`](file:///c:/Users/ABIJITH%20RAJA%20B/Desktop/Neuronet/MVP/live_runner/web/app.js) | 60 FPS HTML5 canvas drawing 20 nodes, 32 links, animated particles, HUD gauges, inspector popup, and REST polling every 1.5s. | Node status is driven by server state. When healed, clears badges. Particles stall on cut links. | Enhance canvas to visualize actual dynamic routes (highlighting current primary vs. active rerouted path) and display real queue depth / drop counts in the inspector. |
| **4** | [`src/neuronet/actions.py`](file:///c:/Users/ABIJITH%20RAJA%20B/Desktop/Neuronet/MVP/src/neuronet/actions.py) | Defines `Action(IntEnum)` with 8 actions (`MONITOR`, `REROUTE`, `QOS_ADJUST`, `TRAFFIC_SHAPE`, `FAILOVER`, `SCALE_RESOURCE`, `ROLLBACK`, `ESCALATE`). | Enumeration is clean and complete, but execution logic was scattered across `server.py` and `toy_env.py` without separate modular handlers. | Retain `Action` enum; build dedicated `src/neuronet/remediation.py` with distinct operational logic for every action. |
| **5** | [`src/neuronet/reward.py`](file:///c:/Users/ABIJITH%20RAJA%20B/Desktop/Neuronet/MVP/src/neuronet/reward.py) | Multi-objective reward balancing latency drop ($0.20$), throughput gain ($0.12$), loss drop ($1.2$), congestion drop ($8.0$), health gain ($5.0$), action costs, and repetition penalty ($-2.0$). | Formula is solid, but previously fed synthetic/hardcoded before-and-after values. | Retain reward formula; feed it real calculated before/after values from the dynamic simulation and Mininet. |
| **6** | [`src/neuronet/dqn_agent.py`](file:///c:/Users/ABIJITH%20RAJA%20B/Desktop/Neuronet/MVP/src/neuronet/dqn_agent.py) | PyTorch `DuelingQNetwork` with shared 2-layer feature extractor (128 units), 1-unit Value stream $V(s)$, and 8-unit Advantage stream $A(s,a)$. Double-DQN target network, Huber loss, ReplayBuffer. | Network architecture is sound. `heuristic_confidence` uses Softmax entropy. Model needs to evaluate unmanipulated states independently. | Retain architecture and weights; integrate with calibrated state vector derived from genuine node/link telemetry. |
| **7** | [`real_mininet/topology.py`](file:///c:/Users/ABIJITH%20RAJA%20B/Desktop/Neuronet/MVP/real_mininet/topology.py) | 20 Open vSwitch switches: 4 Core (`c1..c4`), 6 Aggregation (`a1..a6`), 10 Edge (`e1..e10`), 2 hosts (`h1` on `e1`, `h2` on `e5`). | Link definitions: Path A uses `e1-a1-c1-c2-a3-e5`. Path B uses `e1-a6-c3-c4-a4-e5`. Loop-free standalone design. | Retain topology; ensure exact mapping between 20 dashboard nodes (`C1-C4`, `A1-A6`, `E1-E10`) and Mininet switches. |
| **8** | [`real_mininet/ovs_controller.py`](file:///c:/Users/ABIJITH%20RAJA%20B/Desktop/Neuronet/MVP/real_mininet/ovs_controller.py) | Dispatches OpenFlow flows via `ovs-ofctl` and controls administrative port states (`mod-port e1 e1-eth2 up/down`) for Path A / Path B failover. | `SCALE_RESOURCE` previously only cleared tc rules. `inject_congestion` used netem delay/loss rather than true queue saturation. | Implement real queue limits with TBF/HTB for buffer saturation; implement bandwidth/queue scaling for `SCALE_RESOURCE`. |

---

## 2. Implementation Matrix: Retain, Modify, and Develop

```
+---------------------------------------------------------------------------------------+
|                                     NEURONET                                          |
+---------------------------------------------------------------------------------------+
|  RETAIN (100% Preserved)         MODIFY (Upgraded)              DEVELOP (New Modules) |
|  -----------------------         -----------------              --------------------- |
|  • Dueling DQN PyTorch model     • live_runner/server.py        • network_engine/     |
|  • Pre-trained weights (.pt)       - Remove Q-value boost         - node.py           |
|  • LSTM-AE + Isolation Forest      - Connect to real engine       - link.py           |
|  • SHAP feature attribution      • live_runner/web/app.js         - queue_manager.py  |
|  • Action Enum (8 actions)         - Dynamic path visualizer      - routing_engine.py |
|  • Multi-objective reward math     - Real queue inspector         - simulation.py     |
|  • Mininet 20-switch topology    • real_mininet/ovs_controller  • fault_engine/       |
|  • HTML5 Canvas NOC UI             - Real queue saturation        - buffer_congestion |
|  • JSONL audit logging             - Genuine resource scaling     - link_failure      |
|                                  • src/neuronet/toy_env.py        - ddos_flood        |
|                                    - Delegate to network_engine   - route_flapping    |
|                                                                   - optical_degrade   |
|                                                                 • src/neuronet/       |
|                                                                   - remediation.py    |
+---------------------------------------------------------------------------------------+
```

---

## 3. Phase 2: Network Engine Architecture

### 3.1 Mathematical Queue Model (Fluid Queue Dynamics)
For every node $i \in \{0, \dots, 19\}$ at discrete simulation step $t$:
$$Q_{i, t+1} = \min\left(Q_{i, \max}, \max\left(0, Q_{i, t} + A_{i, t} - S_{i, t}\right)\right)$$
Where:
* $Q_{i, t}$: Current queue occupancy (packets).
* $Q_{i, \max}$: Maximum buffer capacity (default: 1,000 packets).
* $A_{i, t} = \frac{\lambda_{i, t} \cdot \Delta t}{P_{\text{size}}}$: Incoming packets during interval $\Delta t$ at arrival rate $\lambda_{i, t}$ (Mbps).
* $S_{i, t} = \frac{\mu_{i, t} \cdot \Delta t}{P_{\text{size}}}$: Processed packets at service processing capacity $\mu_{i, t}$ (Mbps).
* Dropped Packets:
  $$D_{i, t} = \max\left(0, Q_{i, t} + A_{i, t} - S_{i, t} - Q_{i, \max}\right)$$
* Queuing Latency (Little's Law approximation):
  $$d_{\text{queue}, i} = \frac{Q_{i, t}}{\mu_{i, t} / P_{\text{size}}} \times 1000 \quad (\text{ms})$$
* Total Path Latency:
  $$L_{\text{path}} = \sum_{e \in \text{edges}} d_{\text{prop}, e} + \sum_{n \in \text{nodes}} \left(d_{\text{proc}, n} + d_{\text{queue}, n}\right)$$

### 3.2 Dynamic Routing Engine (Constrained Shortest Path)
Using `networkx.DiGraph`:
* **Cost Function per edge $(u, v)$**:
  $$\text{weight}(u, v) = \alpha \cdot \text{latency}(u, v) + \beta \cdot \text{utilization}(u, v) + \gamma \cdot \text{queue\_congestion}(v)$$
  If link status is `cut`, weight is $\infty$.
* **Rerouting**: When node $v$ is congested ($\text{queue} > 80\%$), the routing engine computes the optimal alternate path $P^* = \text{shortest\_path}(G, \text{source}, \text{target}, \text{weight})$ that bypasses $v$, immediately diverting arrival rate $\lambda_{v}$ to alternate aggregation nodes and allowing $Q_v$ to drain.
