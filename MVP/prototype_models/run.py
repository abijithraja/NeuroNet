import subprocess
import sys
from pathlib import Path

from main_simulation import main

BASE_DIR = Path(__file__).resolve().parent

if __name__ == "__main__":
    # Step 1: Run the AI simulation (generates decision_log.json)
    main()

    # Step 2: Auto-launch the Streamlit dashboard
    print("\n[OK] Launching NeuroNet Dashboard...")
    print("     Open your browser at: http://localhost:8501\n")
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(BASE_DIR / "app.py")],
        cwd=str(BASE_DIR),
        check=True,
    )