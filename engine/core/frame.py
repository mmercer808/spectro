"""
Frame State

Immutable state passed to systems each frame.
Contains timing info and frame identification.
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class FrameState:
    """
    Immutable frame information passed to all systems.
    """
    frame_id: int   # Monotonically increasing frame counter
    dt: float       # Delta time since last frame (seconds)
    t: float        # Total elapsed time (seconds)
    
    @property
    def fps(self) -> float:
        """Estimated FPS from delta time."""
        return 1.0 / max(1e-6, self.dt)
