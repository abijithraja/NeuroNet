# NeuroNet — Empirical Evaluation & Benchmarking Report

**Evaluated Checkpoint:** `C:\Users\ABIJITH RAJA B\Desktop\Neuronet\Main\models\dueling_dqn.pt`

## Overall Performance Comparison

| Strategy | Mean Reward | Latency Recovery (ms) | Packet Loss Reduction (%) | Autonomy / Approval Rate |
| :--- | :---: | :---: | :---: | :---: |
| **NeuroNet RL Agent** | +1.49 | +19.54 ms | +2.47% | 55.0% |
| **Static Rule Engine** | +9.03 | +39.99 ms | +3.02% | 100.0% |
| **Random Action** | -1.97 | +21.83 ms | +1.89% | 100.0% |
| **Passive Monitor** | -14.15 | -1.75 ms | +0.05% | 100.0% |

## Scenario-by-Scenario Breakdown (NeuroNet RL Agent)

| Failure Scenario | Mean Reward | Latency Recovered | Loss Reduced | Autonomy Rate |
| :--- | :---: | :---: | :---: | :---: |
| `normal` | -0.20 | +0.00 ms | +0.00% | 0.0% |
| `congestion` | +6.74 | +34.70 ms | +3.03% | 100.0% |
| `link_failure` | +7.22 | +32.05 ms | +5.71% | 100.0% |
| `routing_flap` | -3.54 | +8.57 ms | +0.78% | 30.0% |
| `attack_ddos` | -1.08 | +41.92 ms | +5.30% | 100.0% |
| `degradation` | -0.20 | +0.00 ms | +0.00% | 0.0% |
