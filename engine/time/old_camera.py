from enum import Enum, auto
from dataclasses import dataclass, field


class TimeCameraMode(Enum):
    FOLLOW_PLAYHEAD = auto()
    FREE_SCROLL = auto()
    SNAP_TO_BARS = auto()

@dataclass
class TimeCameraConfig:
        playhead_ratio: float = 0.3  # Where playhead sits (0-1 from left)
        min_px_per_beat: float = 10.0
        max_px_per_beat = 500.0
        follow_strength: float = 8.0
        inertia_friction: float = 0.95

@dataclass
class TimeCamera:
    left_beat: float = 0.0
    window_beats: float = 16.0
    mode: TimeCameraMode = TimeCameraMode.FOLLOW_PLAYHEAD
    _panel_width_px: float = 800.0
    config: TimeCameraConfig = field(default_factory=TimeCameraConfig)

    @property
    def _px_per_beat(self) -> float:
        return self._panel_width_px / self.window_beats

    def beat_to_px(self, beat: float) -> float:
        return (beat - self.left_beat) * self._px_per_beat

    def px_to_beat(self, px: float) -> float:
        return self.left_beat + (px / self._px_per_beat)

    def is_beat_visible(self, beat: float) -> bool:
        return beat >= self.left_beat and beat < self.left_beat + self.window_beats

    def get_visible_range(self) -> Tuple[float, float]:
        return (self.left_beat, self.left_beat + self.window_beats)

    def snap_to_grid(self, beat: float, subdivision: int = 4) -> float:
        return round(beat / subdivision) * subdivision

    def begin_drag(self, mouse_x: float):
        self.user_scroll_active = True
        self._drag_start_left_beat = self.left_beat
        self._drag_start_x = mouse_x
        self.scroll_velocity = 0.0

    def update_drag(self, mouse_x: float):
        if not self.user_scroll_active:
            return
        dx = mouse_x - self._drag_start_x
        delta_beats = -dx / self._px_per_beat
        self.left_beat = self._drag_start_left_beat + delta_beats

    def end_drag(self, mouse_x: float, velocity: float):
        self.user_scroll_active = False

        