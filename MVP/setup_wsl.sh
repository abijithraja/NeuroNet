#!/usr/bin/env bash
# ==============================================================================
# NeuroNet - 1-Click WSL2 / Ubuntu Setup for Real Mininet Network Simulation
# ==============================================================================
set -e

echo "=== 1. Updating APT Package Lists ==="
sudo apt-get update -y

echo "=== 2. Installing Mininet, Open vSwitch, and Linux Networking Tools ==="
sudo apt-get install -y mininet openvswitch-switch iperf3 iproute2 python3 python3-pip python3-numpy python3-scipy

echo "=== 3. Starting Open vSwitch Service ==="
sudo service openvswitch-switch start
sudo ovs-vsctl show || true

echo "=== 4. Installing Python Machine Learning Dependencies ==="
pip3 install torch scikit-learn pandas --break-system-packages 2>/dev/null || pip3 install torch scikit-learn pandas

echo "=== 5. Running Mininet Sanity Check ==="
sudo mn --test pingall

echo "=== 6. Launching NeuroNet Real Mininet Simulation ==="
sudo python3 run_mininet.py --episodes 5 --model models/mithu_pretrained_dqn.pt

echo "================================================================="
echo "  SUCCESS: Real Mininet simulation completed."
echo "  Inspect decision logs at: logs/mininet_decisions.jsonl"
echo "================================================================="

