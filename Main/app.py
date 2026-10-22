"""NeuroNet AI Decision Engine — Interactive Streamlit Dashboard.

Provides real-time network health overview, AI learning curves, SHAP explainability,
safety gate audit logs, and live simulation controls.
"""
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
SIMULATION_FILE = BASE_DIR / "main.py"
DECISION_LOG_FILE = BASE_DIR / "logs" / "decision_log.json"
ROOT_LOG_FILE = BASE_DIR / "decision_log.json"
EVAL_REPORT_FILE = BASE_DIR / "logs" / "EVALUATION_REPORT.md"

def _is_streamlit_runtime() -> bool:
    return any(
        name.startswith("streamlit.runtime.scriptrunner") or name.startswith("streamlit.web")
        for name in sys.modules
    )

# If run directly as 'python app.py', relaunch under streamlit
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

# Page Setup
st.set_page_config(
    page_title="NeuroNet — Self-Healing Network Control",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling (Dark Glassmorphism)
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.8));
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        margin-bottom: 12px;
    }
    .metric-val {
        font-size: 28px;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-label {
        font-size: 13px;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .badge-approved {
        background-color: rgba(52, 211, 153, 0.15);
        color: #34d399;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        border: 1px solid rgba(52, 211, 153, 0.3);
    }
    .badge-escalated {
        background-color: rgba(248, 113, 113, 0.15);
        color: #f87171;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        border: 1px solid rgba(248, 113, 113, 0.3);
    }
</style>
""", unsafe_allow_html=True)

st.title("⚡ NeuroNet — Autonomous Self-Healing Control Plane")
st.caption("GPU-Accelerated Reinforcement Learning Control Loop with Confidence Safety Gating and Anomaly Intelligence")

# Sidebar Controls
st.sidebar.header("🕹️ Simulation Controls")
sim_env = st.sidebar.selectbox("Network Backend", ["realistic", "ns3", "toy"], index=0)
sim_scenario = st.sidebar.selectbox("Scenario Filter", ["all", "congestion", "link_failure", "routing_flap", "attack_ddos", "degradation"], index=0)
sim_episodes = st.sidebar.slider("Number of Incidents", min_value=5, max_value=50, value=15, step=5)
sim_threshold = st.sidebar.slider("Safety Gate Threshold", min_value=0.50, max_value=0.95, value=0.75, step=0.05)

if st.sidebar.button("▶️ Run Incident Loop", type="primary", use_container_width=True):
    with st.spinner("Executing NeuroNet AI Perception-Decision-Action loop..."):
        try:
            cmd = [
                sys.executable,
                str(SIMULATION_FILE),
                "--episodes", str(sim_episodes),
                "--env", sim_env,
                "--scenario", sim_scenario,
                "--threshold", str(sim_threshold),
            ]
            subprocess.run(cmd, cwd=str(BASE_DIR), check=True)
            st.sidebar.success(f"Successfully processed {sim_episodes} incidents!")
            st.rerun()
        except subprocess.CalledProcessError as e:
            st.sidebar.error(f"Execution failed with code {e.returncode}")

st.sidebar.divider()
st.sidebar.subheader("ℹ️ System Architecture")
st.sidebar.markdown("""
**Loop:** Telemetry $\\rightarrow$ Anomaly $\\rightarrow$ Dueling DQN $\\rightarrow$ Safety Gate $\\rightarrow$ Execution $\\rightarrow$ Reward

- **Anomaly Detection:** LSTM Autoencoder + Isolation Forest
- **RL Agent:** Dueling DQN (Double-DQN with Replay Buffer)
- **Safety Gate:** Confidence Gating ($\ge 75\%$) & Anti-Oscillation
- **Explainability:** SHAP-style Feature Attribution
""")

# Load Data
data = []
target_log = DECISION_LOG_FILE if DECISION_LOG_FILE.exists() else ROOT_LOG_FILE
if target_log.exists():
    try:
        with target_log.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        st.error(f"Error loading decision log: {e}")

if not data:
    st.info("💡 No incident logs detected yet. Click **'▶️ Run Incident Loop'** in the sidebar to generate decisions!")
    st.stop()

df = pd.DataFrame(data)

# Compute Latency Improvement if not present
if "Latency Improvement (ms)" not in df.columns:
    if "latency_improvement_ms" in df.columns:
        df["Latency Improvement (ms)"] = df["latency_improvement_ms"]
    elif "metrics_before" in df.columns and "metrics_after" in df.columns:
        df["Latency Improvement (ms)"] = df.apply(
            lambda r: r["metrics_before"].get("latency_ms", 0) - r["metrics_after"].get("latency_ms", 0), axis=1
        )
    else:
        df["Latency Improvement (ms)"] = 0.0

total_incidents = len(df)
autonomous_df = df[df["escalated"] == False]
autonomous_count = len(autonomous_df)
auto_rate = (autonomous_count / max(1, total_incidents)) * 100.0
avg_conf = df["confidence"].mean() * 100.0
avg_lat_fix = df["Latency Improvement (ms)"].mean()

# KPI Row
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Total Incidents</div>
        <div class="metric-val">{total_incidents}</div>
    </div>
    """, unsafe_allow_html=True)
with k2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Autonomous Actions</div>
        <div class="metric-val">{autonomous_count} <span style="font-size:16px; color:#34d399;">({auto_rate:.1f}%)</span></div>
    </div>
    """, unsafe_allow_html=True)
with k3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Average Confidence</div>
        <div class="metric-val">{avg_conf:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)
with k4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Mean Latency Recovered</div>
        <div class="metric-val">{avg_lat_fix:+.1f} ms</div>
    </div>
    """, unsafe_allow_html=True)

# Tabs
tab_overview, tab_explain, tab_safety, tab_benchmarks, tab_logs = st.tabs([
    "📈 Performance Trajectory",
    "🧠 Anomaly & Explainability",
    "🛡️ Safety Gate & Governance",
    "📊 Strategy Benchmarks",
    "📋 Complete Audit Log",
])

with tab_overview:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Episode Rewards (Learning Curve)")
        if "reward" in df.columns:
            st.line_chart(df["reward"], color="#38bdf8")
    with c2:
        st.subheader("Latency Recovery (ms)")
        st.area_chart(df["Latency Improvement (ms)"], color="#34d399")

    st.divider()
    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Remediation Action Distribution")
        st.bar_chart(df["action"].value_counts(), color="#818cf8")
    with c4:
        st.subheader("Escalation Rate by Incident Scenario (%)")
        scen_col = "scenario" if "scenario" in df.columns else "anomaly_type"
        if scen_col in df.columns:
            esc_by_scen = df.groupby(scen_col)["escalated"].mean() * 100.0
            st.bar_chart(esc_by_scen, color="#f472b6")

with tab_explain:
    st.subheader("🧠 SHAP-Inspired Feature Attribution & Action Justification")
    st.markdown("Inspect which telemetry metrics triggered the anomaly and why the RL agent selected the remediation action.")

    selected_idx = st.selectbox(
        "Select Incident Episode to Inspect",
        range(len(df)),
        format_func=lambda i: f"Episode {df.iloc[i].get('episode', i+1)}: {df.iloc[i].get('anomaly_type', 'incident')} ({df.iloc[i].get('action', 'act')})",
        index=len(df) - 1,
    )
    incident = df.iloc[selected_idx]
    exp = incident.get("explanation", {})

    col_exp1, col_exp2 = st.columns([1, 1])
    with col_exp1:
        st.markdown(f"**Anomaly Status:** `{incident.get('anomaly_type', 'N/A')}` (Score: `{incident.get('anomaly_score', 0.0):.2f}`)")
        st.info(exp.get("anomaly_summary", "Anomaly analysis available."))
        st.markdown(f"**Action Selected:** `{incident.get('action', 'N/A')}` (Confidence: `{incident.get('confidence', 0.0):.1%}`)")
        st.success(exp.get("action_rationale", "Remediation action rationale."))

    with col_exp2:
        st.markdown("**Metric Contribution Breakdown (% of Anomaly Score)**")
        attr = exp.get("feature_attributions", {})
        if attr:
            attr_df = pd.DataFrame(list(attr.items()), columns=["Metric", "Contribution"]).sort_values("Contribution", ascending=True)
            st.bar_chart(attr_df.set_index("Metric"), color="#fb923c")
        else:
            st.write("Attribution weights recorded nominally.")

with tab_safety:
    st.subheader("🛡️ Safety Gate & Progressive Autonomy")
    st.markdown("""
    Under NeuroNet's progressive autonomy model:
    - **High-confidence decisions ($\ge 75\%$)** execute autonomously in under 50 ms.
    - **Low-confidence or high-risk decisions** safely escalate to the Network Operations Center (NOC).
    """)

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.metric("Total Autonomous Actions", f"{autonomous_count} / {total_incidents}")
        st.metric("NOC Human Escalations", f"{total_incidents - autonomous_count}")
    with col_s2:
        st.markdown("**Decision Gate Outcomes**")
        gate_counts = df["escalated"].map({False: "Autonomous (Approved)", True: "Escalated to Human NOC"}).value_counts()
        st.bar_chart(gate_counts, color="#34d399")

with tab_benchmarks:
    st.subheader("📊 Comparative Benchmark: RL Agent vs Traditional Baselines")
    if EVAL_REPORT_FILE.exists():
        try:
            with EVAL_REPORT_FILE.open("r", encoding="utf-8") as f:
                st.markdown(f.read())
        except Exception:
            st.info("Evaluation report available.")
    else:
        st.info("Run `python evaluate_agent.py` to generate side-by-side benchmark comparison against Static Rule Engine, Random Agent, and Passive Monitor.")

with tab_logs:
    st.subheader("📋 Decision & Audit Trail")
    filter_type = st.selectbox("Filter by Incident Type", ["ALL"] + list(df["anomaly_type"].unique()))
    filtered_df = df if filter_type == "ALL" else df[df["anomaly_type"] == filter_type]

    cols_to_show = ["episode", "timestamp", "anomaly_type", "action", "confidence", "escalated", "reward", "Latency Improvement (ms)"]
    avail_cols = [c for c in cols_to_show if c in filtered_df.columns]

    st.dataframe(filtered_df[avail_cols], use_container_width=True)

    csv_data = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Export Decision Log as CSV",
        data=csv_data,
        file_name="neuronet_decision_log.csv",
        mime="text/csv",
    )
