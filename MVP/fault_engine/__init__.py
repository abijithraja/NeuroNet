"""NeuroNet Fault Injection Engine Package."""

from .faults import (
    NetworkFault,
    BufferCongestionFault,
    CoreLinkFailureFault,
    DDoSSYNFloodFault,
    RouteFlappingFault,
    OpticalDegradationFault,
    FAULTS,
)

__all__ = [
    "NetworkFault",
    "BufferCongestionFault",
    "CoreLinkFailureFault",
    "DDoSSYNFloodFault",
    "RouteFlappingFault",
    "OpticalDegradationFault",
    "FAULTS",
]
