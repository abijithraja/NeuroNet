"""Standardized remediation action space for NeuroNet."""
from enum import IntEnum
from typing import Dict, Any

class Action(IntEnum):
    MONITOR = 0         # Observe telemetry, take no active intervention
    REROUTE = 1         # Dynamic ECMP / alternate path rerouting
    QOS_ADJUST = 2      # Prioritize queues and DSCP / CoS markings
    TRAFFIC_SHAPE = 3   # Rate-limiting / token bucket burst smoothing
    FAILOVER = 4        # Instant hardware link/interface failover
    SCALE_RESOURCE = 5  # Scale 5G network slice / bandwidth allocation
    THROTTLE = 6        # Throttle low-priority background slice traffic
    ESCALATE = 7        # Defer to human NOC operator for manual intervention

ACTION_NAMES: Dict[Action, str] = {
    Action.MONITOR: "monitor",
    Action.REROUTE: "reroute",
    Action.QOS_ADJUST: "qos_adjust",
    Action.TRAFFIC_SHAPE: "traffic_shape",
    Action.FAILOVER: "failover",
    Action.SCALE_RESOURCE: "scale_resource",
    Action.THROTTLE: "throttle",
    Action.ESCALATE: "escalate_to_human",
}

ACTION_DESCRIPTIONS: Dict[Action, str] = {
    Action.MONITOR: "Continue observing telemetry without active network modification",
    Action.REROUTE: "Distribute traffic load onto alternate ECMP paths via OpenFlow",
    Action.QOS_ADJUST: "Promote degraded flow queues to high priority",
    Action.TRAFFIC_SHAPE: "Smooth bursty packet transmission rates using token buckets",
    Action.FAILOVER: "Switch traffic immediately to designated redundant standby link",
    Action.SCALE_RESOURCE: "Dynamically allocate additional compute/bandwidth to 5G slice",
    Action.THROTTLE: "Rate-limit bulk background traffic to protect latency-critical flows",
    Action.ESCALATE: "Safety gate escalation to human NOC operator due to low confidence/high risk",
}

# Inherent operational cost/overhead of executing each action
ACTION_COSTS: Dict[Action, float] = {
    Action.MONITOR: -0.1,
    Action.REROUTE: -1.0,
    Action.QOS_ADJUST: -0.5,
    Action.TRAFFIC_SHAPE: -0.8,
    Action.FAILOVER: -1.8,
    Action.SCALE_RESOURCE: -2.0,
    Action.THROTTLE: -1.2,
    Action.ESCALATE: -0.2,
}

# Predefined inverse action for rollback capability
INVERSE_ACTIONS: Dict[Action, Action] = {
    Action.MONITOR: Action.MONITOR,
    Action.REROUTE: Action.REROUTE,
    Action.QOS_ADJUST: Action.TRAFFIC_SHAPE,
    Action.TRAFFIC_SHAPE: Action.QOS_ADJUST,
    Action.FAILOVER: Action.FAILOVER,
    Action.SCALE_RESOURCE: Action.THROTTLE,
    Action.THROTTLE: Action.SCALE_RESOURCE,
    Action.ESCALATE: Action.MONITOR,
}

N_ACTIONS = len(Action)
