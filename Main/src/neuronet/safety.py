"""Safety Gate for Confidence Verification, Constraint Checking, and Automated Escalation."""
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from .actions import Action, ACTION_NAMES, INVERSE_ACTIONS

@dataclass
class GateResult:
    approved: bool
    action: str
    action_id: int
    confidence: float
    reason: str
    inverse_action: str
    risk_level: str  # "LOW", "MEDIUM", "HIGH"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "approved": self.approved,
            "action": self.action,
            "action_id": self.action_id,
            "confidence": float(self.confidence),
            "reason": self.reason,
            "inverse_action": self.inverse_action,
            "risk_level": self.risk_level,
        }


def apply_safety_gate(
    action: Action,
    confidence: float,
    threshold: float = 0.75,
    metrics_before: Optional[Dict[str, float]] = None,
    recent_actions: Optional[List[int]] = None,
    allow_autonomy: bool = True,
) -> GateResult:
    """Verifies confidence, operational constraints, and safety policies."""
    action = Action(int(action))
    confidence = float(confidence)
    inverse_act = ACTION_NAMES[INVERSE_ACTIONS.get(action, Action.MONITOR)]

    if not 0.0 <= confidence <= 1.0:
        return GateResult(
            approved=False,
            action=ACTION_NAMES[Action.ESCALATE],
            action_id=int(Action.ESCALATE),
            confidence=confidence,
            reason="Invalid or uncalibrated confidence value",
            inverse_action=ACTION_NAMES[Action.MONITOR],
            risk_level="HIGH",
        )

    if action == Action.ESCALATE:
        return GateResult(
            approved=False,
            action=ACTION_NAMES[Action.ESCALATE],
            action_id=int(Action.ESCALATE),
            confidence=confidence,
            reason="RL Policy explicitly requested NOC human review",
            inverse_action=ACTION_NAMES[Action.MONITOR],
            risk_level="LOW",
        )

    if not allow_autonomy:
        return GateResult(
            approved=False,
            action=ACTION_NAMES[Action.ESCALATE],
            action_id=int(Action.ESCALATE),
            confidence=confidence,
            reason="Autonomous remediation disabled by administrative policy",
            inverse_action=inverse_act,
            risk_level="MEDIUM",
        )

    # 1. Confidence threshold safety check
    if confidence < threshold:
        return GateResult(
            approved=False,
            action=ACTION_NAMES[Action.ESCALATE],
            action_id=int(Action.ESCALATE),
            confidence=confidence,
            reason=f"Confidence {confidence:.1%} is below required autonomy threshold {threshold:.1%}",
            inverse_action=inverse_act,
            risk_level="MEDIUM",
        )

    # 2. Oscillation / Thrashing guard
    history = recent_actions or []
    if len(history) >= 2 and history[-1] == int(action) and history[-2] == int(action):
        return GateResult(
            approved=False,
            action=ACTION_NAMES[Action.ESCALATE],
            action_id=int(Action.ESCALATE),
            confidence=confidence,
            reason="Action thrashing detected: identical action repeated 3x consecutively",
            inverse_action=inverse_act,
            risk_level="HIGH",
        )

    # 3. Domain-specific physical risk conditions
    if metrics_before:
        # Severe physical link degradation: taking 'monitor' is unsafe
        if metrics_before.get("link_health", 1.0) < 0.25 and action == Action.MONITOR:
            return GateResult(
                approved=False,
                action=ACTION_NAMES[Action.ESCALATE],
                action_id=int(Action.ESCALATE),
                confidence=confidence,
                reason="Unsafe action: 'monitor' proposed during critical link breakdown",
                inverse_action=inverse_act,
                risk_level="HIGH",
            )
        # Extreme buffer congestion: traffic shaping alone might be insufficient
        if metrics_before.get("congestion", 0.0) > 0.95 and action == Action.MONITOR:
            return GateResult(
                approved=False,
                action=ACTION_NAMES[Action.ESCALATE],
                action_id=int(Action.ESCALATE),
                confidence=confidence,
                reason="Unsafe action: passive monitoring proposed while buffer is 95%+ full",
                inverse_action=inverse_act,
                risk_level="HIGH",
            )

    return GateResult(
        approved=True,
        action=ACTION_NAMES[action],
        action_id=int(action),
        confidence=confidence,
        reason=f"Confidence {confidence:.1%} meets safety standards with valid risk clearance",
        inverse_action=inverse_act,
        risk_level="LOW",
    )
