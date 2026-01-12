from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class FrameState:
    frame_id: int
    dt: float
    t: float
