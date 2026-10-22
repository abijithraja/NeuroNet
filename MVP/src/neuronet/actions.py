from enum import IntEnum

class Action(IntEnum):
    MONITOR = 0
    REROUTE = 1
    QOS_ADJUST = 2
    TRAFFIC_SHAPE = 3
    FAILOVER = 4
    SCALE_RESOURCE = 5
    ROLLBACK = 6
    ESCALATE = 7

ACTION_NAMES = {
    Action.MONITOR: "monitor",
    Action.REROUTE: "reroute",
    Action.QOS_ADJUST: "qos_adjust",
    Action.TRAFFIC_SHAPE: "traffic_shape",
    Action.FAILOVER: "failover",
    Action.SCALE_RESOURCE: "scale_resource",
    Action.ROLLBACK: "rollback",
    Action.ESCALATE: "escalate_to_human",
}
N_ACTIONS = len(Action)
