"""Conservative safety gate for a simulated environment only."""
from dataclasses import dataclass
from .actions import Action, ACTION_NAMES

@dataclass
class GateResult:
    approved: bool
    action: str
    confidence: float
    reason: str

def apply_safety_gate(action, confidence, threshold=0.75, allow_autonomy=True):
    action = Action(int(action))
    confidence = float(confidence)
    if not 0 <= confidence <= 1:
        return GateResult(False, ACTION_NAMES[Action.ESCALATE], confidence, "Invalid confidence")
    if action == Action.ESCALATE:
        return GateResult(False, ACTION_NAMES[action], confidence, "Policy requested human escalation")
    if not allow_autonomy:
        return GateResult(False, ACTION_NAMES[Action.ESCALATE], confidence, "Autonomous actions disabled")
    if confidence < threshold:
        return GateResult(False, ACTION_NAMES[Action.ESCALATE], confidence,
                          f"Confidence below configured threshold {threshold:.2f}")
    return GateResult(True, ACTION_NAMES[action], confidence, "Approved for simulated execution")
