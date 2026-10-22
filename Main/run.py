"""Unified launcher for NeuroNet simulation and interactive dashboard."""
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
SIMULATION_FILE = BASE_DIR / "main.py"
DASHBOARD_FILE = BASE_DIR / "app.py"

def main():
    print("=" * 60)
    print("        LAUNCHING NEURONET SYSTEM")
    print("=" * 60)
    print("1. Running initial simulation loop to populate live telemetry...")
    subprocess.run([sys.executable, str(SIMULATION_FILE), "--episodes", "15"], cwd=str(BASE_DIR), check=False)
    
    print("\n2. Launching interactive Streamlit Dashboard...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(DASHBOARD_FILE)], cwd=str(BASE_DIR))

if __name__ == "__main__":
    main()
