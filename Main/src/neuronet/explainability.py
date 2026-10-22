"""Explainability engine providing SHAP-inspired feature attributions and natural language decision justifications."""
from typing import Dict, Any, List, Optional
import numpy as np
from .actions import Action, ACTION_NAMES, ACTION_DESCRIPTIONS

def generate_decision_explanation(
    metrics_before: Dict[str, float],
    anomaly_info: Dict[str, Any],
    proposed_action: Action,
    confidence: float,
    gate_result: Dict[str, Any],
    q_values: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Generates detailed feature attribution and human-readable justification for an RL decision."""
    attributions = anomaly_info.get("feature_attributions", {
        "latency": 0.35, "packet_loss": 0.25, "congestion": 0.25, "cpu": 0.15
    })
    
    # Identify top driver
    top_driver = max(attributions.items(), key=lambda x: x[1])
    action_name = ACTION_NAMES[proposed_action]
    action_desc = ACTION_DESCRIPTIONS[proposed_action]

    # Generate anomaly text
    score = anomaly_info.get("anomaly_score", 0.0)
    if anomaly_info.get("is_anomaly", False):
        anomaly_summary = (
            f"Anomaly detected (Score: {score:.2f}, LSTM: {anomaly_info.get('lstm_error', 0.0):.2f}, "
            f"IsoForest: {anomaly_info.get('iso_score', 0.0):.2f}). "
            f"Primary metric driver: {top_driver[0].upper()} ({top_driver[1]:.1%} contribution)."
        )
    else:
        anomaly_summary = f"Telemetry within nominal bounds (Anomaly score: {score:.2f})."

    # Generate action rationale
    if proposed_action == Action.REROUTE:
        rationale = f"Selected '{action_name}' to bypass high latency ({metrics_before.get('latency_ms', 0):.1f}ms) and buffer congestion ({metrics_before.get('congestion', 0):.1%}) via alternate paths."
    elif proposed_action == Action.FAILOVER:
        rationale = f"Selected '{action_name}' due to critical link health deterioration ({metrics_before.get('link_health', 1.0):.1%}) to avoid link failure packet blackhole."
    elif proposed_action == Action.QOS_ADJUST:
        rationale = f"Selected '{action_name}' to prioritize delay-sensitive flows and mitigate packet loss ({metrics_before.get('packet_loss_pct', 0):.2f}%)."
    elif proposed_action == Action.SCALE_RESOURCE:
        rationale = f"Selected '{action_name}' to increase 5G slice capacity and absorb high demand without queue overflows."
    elif proposed_action == Action.THROTTLE:
        rationale = f"Selected '{action_name}' to curtail low-priority background flows and relieve high CPU load ({metrics_before.get('cpu_util_pct', 0):.1f}%)."
    elif proposed_action == Action.TRAFFIC_SHAPE:
        rationale = f"Selected '{action_name}' to smooth bursty packet arrivals and prevent queue buffer exhaustion."
    elif proposed_action == Action.ESCALATE:
        rationale = f"Selected '{action_name}' because the RL policy deferred autonomous action for manual inspection."
    else:
        rationale = f"Selected '{action_name}' as current telemetry fluctuations do not warrant active intervention."

    # Safety status
    approved = gate_result.get("approved", False)
    safety_summary = (
        f"Safety Gate APPROVED: Action dispatched autonomously (Confidence: {confidence:.1%})."
        if approved
        else f"Safety Gate ESCALATED: {gate_result.get('reason', 'Policy deferred')} (Confidence: {confidence:.1%})."
    )

    return {
        "anomaly_summary": anomaly_summary,
        "action_rationale": rationale,
        "action_description": action_desc,
        "safety_summary": safety_summary,
        "top_contributing_feature": top_driver[0],
        "top_contributing_weight": top_driver[1],
        "feature_attributions": attributions,
        "q_values": [float(q) for q in q_values] if q_values is not None else [],
    }
