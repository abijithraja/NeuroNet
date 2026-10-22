import json
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
SIMULATION_FILE = BASE_DIR / "main_simulation.py"
DECISION_LOG_FILE = BASE_DIR / "decision_log.json"


def _is_streamlit_runtime() -> bool:
    return any(
        name.startswith("streamlit.runtime.scriptrunner") or name.startswith("streamlit.web")
        for name in sys.modules
    )


# If users run "python app.py", relaunch correctly in Streamlit mode.
if __name__ == "__main__" and "--from-streamlit" not in sys.argv and not _is_streamlit_runtime():
    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(BASE_DIR / "app.py"),
            "--",
            "--from-streamlit",
        ],
        cwd=str(BASE_DIR),
        check=False,
    )
    raise SystemExit(0)

import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# 1. Page Config
st.set_page_config(page_title="NeuroNet Dashboard", layout="wide")

# Auto-refresh every 2 seconds for real-time feel
st_autorefresh(interval=2000, key="data_refresh")

st.title("NeuroNet AI Decision Engine")
st.markdown("Live telemetry and reinforcement learning action log.")
# --- Live Mode Button ---
st.sidebar.header("Controls")
if st.sidebar.button("Run Simulation", type="primary"):
    with st.spinner("Running AI simulation..."):
        try:
            subprocess.run(
                [sys.executable, str(SIMULATION_FILE)],
                cwd=str(BASE_DIR),
                check=True,
            )
            st.success("Simulation complete! Dashboard will refresh automatically.")
        except subprocess.CalledProcessError as exc:
            st.error(f"Simulation failed with exit code {exc.returncode}.")

st.sidebar.divider()

# --- "Why This Matters" Panel ---
st.sidebar.header("Why This Matters")
st.sidebar.markdown(
    """
    **NeuroNet AI** autonomously manages network anomalies using reinforcement learning.

    - **Reduces downtime** -- AI responds in milliseconds
    - **Automates network ops** -- fewer manual interventions
    - **Improves latency** -- intelligent action selection
    - **Reduces human load** -- only escalates when unsure
    """
)

try:
    # 2. Load the data from your MVP run
    with DECISION_LOG_FILE.open() as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)

    # Calculate actual latency improvement (Before - After)
    df["Latency Improvement (ms)"] = df["metrics_before"].apply(lambda x: x["latency"]) - df["metrics_after"].apply(lambda x: x["latency"])
    
    # 3. Top Metrics Row
    st.subheader("Network Health Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    total_incidents = len(df)
    autonomous = len(df[df["escalated"] == False])
    avg_confidence = df["confidence"].mean() * 100
    avg_latency = df["Latency Improvement (ms)"].mean()

    col1.metric("Total Incidents", total_incidents)
    col2.metric("Autonomous Actions", f"{autonomous}", f"{(autonomous/total_incidents)*100:.0f}% rate")
    col3.metric("Avg AI Confidence", f"{avg_confidence:.1f}%")
    col4.metric("Avg Latency Fixed", f"{avg_latency:.1f} ms")

    st.divider()

    # 4. Interactive Charts
    st.subheader("Performance Trajectory")
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.markdown("**Agent Reward (Learning Curve)**")
        st.line_chart(df["reward"], color="#00ff00")
        
    with chart_col2:
        st.markdown("**Latency Improvement (ms)**")
        st.area_chart(df["Latency Improvement (ms)"], color="#00d4ff")

    st.divider()

    # 5. Action Distribution (Pie/Bar Chart)
    st.subheader("Action Distribution")
    action_col1, action_col2 = st.columns(2)

    with action_col1:
        st.markdown("**Actions taken by the AI agent**")
        st.bar_chart(df["action"].value_counts())

    with action_col2:
        st.markdown("**Escalation by Anomaly Type**")
        escalation_data = df.groupby("anomaly_type")["escalated"].mean() * 100
        st.bar_chart(escalation_data, color="#ff4b4b")

    st.divider()

    # 6. Filter by Anomaly Type
    st.subheader("Explore by Anomaly Type")
    atype = st.selectbox("Select Anomaly", df["anomaly_type"].unique())
    filtered_df = df[df["anomaly_type"] == atype]

    fcol1, fcol2, fcol3 = st.columns(3)
    fcol1.metric("Episodes", len(filtered_df))
    fcol2.metric("Avg Reward", f"{filtered_df['reward'].mean():+.2f}")
    fcol3.metric("Escalation Rate", f"{filtered_df['escalated'].mean()*100:.0f}%")

    st.dataframe(
        filtered_df[["episode", "action", "confidence", "escalated", "reward", "Latency Improvement (ms)"]],
        width="stretch"
    )

    st.divider()

    # 7. Full Decision Log
    st.subheader("Autonomous Decision Log")
    
    clean_df = df[["episode", "timestamp", "anomaly_type", "action", "confidence", "escalated", "reward", "Latency Improvement (ms)"]]
    
    def highlight_status(val):
        color = '#ff4b4b' if val else '#00c04b'
        return f'background-color: {color}'
    
    st.dataframe(clean_df.style.map(highlight_status, subset=['escalated']), width="stretch")

except FileNotFoundError:
    st.warning("No decision_log.json found. Click 'Run Simulation' in the sidebar to generate data.")
