"""
SequenceTensor - Grid state as tensors for GPU-native sequencing.

The grid is a tensor with RGBA channels:
    R = active (0 or 1)
    G = velocity (0-1) 
    B = timing offset (-1 to 1, micro-timing/swing)
    A = subdivision level (1, 2, 3, 4, 6, 8, 12, 16...)

Subdivision tensors are allocated on demand for cells that need them.
The alpha channel acts as both a render hint and a data structure selector.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, List, Set
from enum import IntEnum

# Optional torch support - fall back to numpy
try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False


class Channel(IntEnum):
    """Tensor channel indices."""
    ACTIVE = 0      # R - is there an event here?
    VELOCITY = 1    # G - how hard? (0-1)
    TIMING = 2      # B - micro-timing offset (-1 to 1)
    SUBDIVISION = 3 # A - subdivision level (1 = none, 2+ = subdivided)


@dataclass
class Subdivision:
    """Quantization result for placing an onset in the grid."""
    level: int      # subdivision resolution (1, 2, 3, 4, 6, 8, 12, 16)
    index: int      # position within subdivision (0 to level-1)
    error: float    # quantization error (0 = perfect fit)
    
    @property
    def fractional_position(self) -> float:
        """Position as fraction of beat (0.0 to 1.0)."""
        return self.index / self.level if self.level > 0 else 0.0


@dataclass
class RhythmSet:
    """
    Define subdivision as a set of positions within the beat.
    
    Common rhythm sets encode different feels:
    - STRAIGHT_16: 1-e-&-a in 16th grid
    - SWING: swung 16ths
    - TRIPLET: triplet feel
    """
    positions: Set[int]
    resolution: int = 16
    name: str = ""
    
    # Common presets
    STRAIGHT_8 = None   # Will be initialized below
    STRAIGHT_16 = None
    SWING_8 = None
    SWING_16 = None
    TRIPLET = None
    CLAVE_3_2 = None
    
    def __post_init__(self):
        self.positions = set(self.positions)
    
    def to_subdivision_level(self) -> int:
        """Get the subdivision level needed for this set."""
        return self.resolution
    
    def matches(self, frac: float, tolerance: float = 0.05) -> Optional[int]:
        """
        Check if a fractional position matches any position in this set.
        Returns the matching position index or None.
        """
        for pos in self.positions:
            expected = pos / self.resolution
            if abs(frac - expected) <= tolerance:
                return pos
        return None
    
    def to_tensor(self, backend='numpy') -> np.ndarray:
        """
        Expand set into a subdivision tensor.
        Shape: (4, resolution, 1) matching SequenceTensor channel layout.
        """
        if backend == 'torch' and HAS_TORCH:
            t = torch.zeros((4, self.resolution, 1), dtype=torch.float32)
            for pos in self.positions:
                t[Channel.ACTIVE, pos, 0] = 1.0
            return t
        else:
            t = np.zeros((4, self.resolution, 1), dtype=np.float32)
            for pos in self.positions:
                t[Channel.ACTIVE, pos, 0] = 1.0
            return t


# Initialize preset rhythm sets
RhythmSet.STRAIGHT_8 = RhythmSet({0, 2, 4, 6, 8, 10, 12, 14}, 16, "Straight 8ths")
RhythmSet.STRAIGHT_16 = RhythmSet({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, 16, "Straight 16ths")
RhythmSet.SWING_8 = RhythmSet({0, 5, 8, 13}, 16, "Swing 8ths")
RhythmSet.SWING_16 = RhythmSet({0, 3, 5, 8, 11, 13}, 16, "Swing 16ths")
RhythmSet.TRIPLET = RhythmSet({0, 4, 8}, 12, "Triplets")
RhythmSet.CLAVE_3_2 = RhythmSet({0, 3, 6, 10, 12}, 16, "3-2 Son Clave")


class SequenceTensor:
    """
    Grid state as a tensor - GPU native, shader compatible.
    
    The grid represents a cross-section of the timeline where:
    - Columns = beats (time on X axis)
    - Rows = instruments/pitches (frequency on Y axis)
    - Channels = event properties (RGBA)
    - Subdivision tensors = nested grids within cells
    
    Usage:
        grid = SequenceTensor(rows=8, cols=16)
        grid.toggle(row=0, col=4)  # Toggle kick on beat 5
        grid.set_velocity(row=0, col=4, velocity=0.8)
        grid.set_subdivision(row=1, col=2, level=4)  # 16th notes in cell
        
        # Upload to shader
        texture_data = grid.to_rgba_bytes()
    """
    
    def __init__(
        self,
        rows: int = 8,
        cols: int = 16,
        backend: str = 'numpy'  # 'numpy' or 'torch'
    ):
        self.rows = rows
        self.cols = cols
        self.backend = backend if (backend == 'torch' and HAS_TORCH) else 'numpy'
        
        # Core state tensor: (4, rows, cols) = (channels, instruments, beats)
        if self.backend == 'torch':
            self.state = torch.zeros((4, rows, cols), dtype=torch.float32)
        else:
            self.state = np.zeros((4, rows, cols), dtype=np.float32)
        
        # Subdivision tensors - sparse, allocated on demand
        # Key: (row, col) → tensor of shape (4, subdiv_level, 1)
        self.subdivisions: Dict[Tuple[int, int], np.ndarray] = {}
        
        # Track which cells have been modified (for incremental upload)
        self._dirty_cells: Set[Tuple[int, int]] = set()
        self._fully_dirty = True
    
    # =========================================================================
    # BASIC CELL OPERATIONS
    # =========================================================================
    
    def toggle(self, row: int, col: int) -> bool:
        """Toggle cell active state. Returns new state."""
        if not self._in_bounds(row, col):
            return False
        current = self.state[Channel.ACTIVE, row, col]
        new_val = 0.0 if current > 0.5 else 1.0
        self.state[Channel.ACTIVE, row, col] = new_val
        self._mark_dirty(row, col)
        return new_val > 0.5
    
    def set_active(self, row: int, col: int, active: bool = True):
        """Set cell active state."""
        if self._in_bounds(row, col):
            self.state[Channel.ACTIVE, row, col] = 1.0 if active else 0.0
            self._mark_dirty(row, col)
    
    def is_active(self, row: int, col: int) -> bool:
        """Check if cell is active."""
        if not self._in_bounds(row, col):
            return False
        return self.state[Channel.ACTIVE, row, col] > 0.5
    
    def set_velocity(self, row: int, col: int, velocity: float):
        """Set cell velocity (0.0 to 1.0)."""
        if self._in_bounds(row, col):
            self.state[Channel.VELOCITY, row, col] = np.clip(velocity, 0.0, 1.0)
            self._mark_dirty(row, col)
    
    def get_velocity(self, row: int, col: int) -> float:
        """Get cell velocity."""
        if not self._in_bounds(row, col):
            return 0.0
        return float(self.state[Channel.VELOCITY, row, col])
    
    def set_timing(self, row: int, col: int, offset: float):
        """Set micro-timing offset (-1.0 to 1.0)."""
        if self._in_bounds(row, col):
            self.state[Channel.TIMING, row, col] = np.clip(offset, -1.0, 1.0)
            self._mark_dirty(row, col)
    
    def get_timing(self, row: int, col: int) -> float:
        """Get micro-timing offset."""
        if not self._in_bounds(row, col):
            return 0.0
        return float(self.state[Channel.TIMING, row, col])
    
    # =========================================================================
    # SUBDIVISION OPERATIONS  
    # =========================================================================
    
    def set_subdivision(self, row: int, col: int, level: int):
        """
        Set subdivision level for a cell.
        Level 1 = no subdivision, 2+ = that many sub-cells.
        """
        if not self._in_bounds(row, col):
            return
        
        level = max(1, level)
        self.state[Channel.SUBDIVISION, row, col] = float(level)
        
        # Allocate subdivision tensor if needed
        if level > 1:
            key = (row, col)
            if key not in self.subdivisions or self.subdivisions[key].shape[1] != level:
                if self.backend == 'torch':
                    self.subdivisions[key] = torch.zeros((4, level, 1), dtype=torch.float32)
                else:
                    self.subdivisions[key] = np.zeros((4, level, 1), dtype=np.float32)
        
        self._mark_dirty(row, col)
    
    def get_subdivision_level(self, row: int, col: int) -> int:
        """Get subdivision level for a cell."""
        if not self._in_bounds(row, col):
            return 1
        return max(1, int(self.state[Channel.SUBDIVISION, row, col]))
    
    def get_subdivision_tensor(self, row: int, col: int) -> Optional[np.ndarray]:
        """Get the subdivision tensor for a cell, if it exists."""
        return self.subdivisions.get((row, col))
    
    def set_subdivision_cell(
        self,
        row: int,
        col: int,
        sub_index: int,
        active: bool = True,
        velocity: float = 0.8
    ):
        """Set a specific cell within a subdivision."""
        key = (row, col)
        if key not in self.subdivisions:
            return
        
        sub_tensor = self.subdivisions[key]
        if 0 <= sub_index < sub_tensor.shape[1]:
            sub_tensor[Channel.ACTIVE, sub_index, 0] = 1.0 if active else 0.0
            sub_tensor[Channel.VELOCITY, sub_index, 0] = velocity
            self._mark_dirty(row, col)
    
    def apply_rhythm_set(self, row: int, col: int, rhythm_set: RhythmSet):
        """Apply a rhythm set to a cell's subdivision."""
        self.set_subdivision(row, col, rhythm_set.resolution)
        
        key = (row, col)
        if key in self.subdivisions:
            # Clear existing
            self.subdivisions[key][:] = 0.0
            # Apply rhythm set
            for pos in rhythm_set.positions:
                if pos < self.subdivisions[key].shape[1]:
                    self.subdivisions[key][Channel.ACTIVE, pos, 0] = 1.0
                    self.subdivisions[key][Channel.VELOCITY, pos, 0] = 0.8
        
        self._mark_dirty(row, col)
    
    # =========================================================================
    # BATCH OPERATIONS
    # =========================================================================
    
    def toggle_batch(self, coords: List[Tuple[int, int]]):
        """Toggle multiple cells at once."""
        for row, col in coords:
            self.toggle(row, col)
    
    def clear(self):
        """Clear entire grid."""
        if self.backend == 'torch':
            self.state.zero_()
        else:
            self.state.fill(0.0)
        self.subdivisions.clear()
        self._fully_dirty = True
    
    def clear_row(self, row: int):
        """Clear all cells in a row."""
        if 0 <= row < self.rows:
            self.state[:, row, :] = 0.0
            # Remove subdivisions for this row
            keys_to_remove = [k for k in self.subdivisions if k[0] == row]
            for k in keys_to_remove:
                del self.subdivisions[k]
            self._fully_dirty = True
    
    def clear_column(self, col: int):
        """Clear all cells in a column (beat)."""
        if 0 <= col < self.cols:
            self.state[:, :, col] = 0.0
            # Remove subdivisions for this column
            keys_to_remove = [k for k in self.subdivisions if k[1] == col]
            for k in keys_to_remove:
                del self.subdivisions[k]
            self._fully_dirty = True
    
    # =========================================================================
    # PATTERN TRANSFORMATIONS
    # =========================================================================
    
    def shift_time(self, amount: int):
        """Shift entire pattern in time (columns)."""
        if self.backend == 'torch':
            self.state = torch.roll(self.state, shifts=amount, dims=2)
        else:
            self.state = np.roll(self.state, shift=amount, axis=2)
        
        # Shift subdivision keys
        new_subdivs = {}
        for (row, col), tensor in self.subdivisions.items():
            new_col = (col + amount) % self.cols
            new_subdivs[(row, new_col)] = tensor
        self.subdivisions = new_subdivs
        self._fully_dirty = True
    
    def reverse(self):
        """Reverse pattern in time."""
        if self.backend == 'torch':
            self.state = torch.flip(self.state, dims=[2])
        else:
            self.state = np.flip(self.state, axis=2).copy()
        
        # Reverse subdivision keys
        new_subdivs = {}
        for (row, col), tensor in self.subdivisions.items():
            new_col = self.cols - 1 - col
            new_subdivs[(row, new_col)] = tensor
        self.subdivisions = new_subdivs
        self._fully_dirty = True
    
    def rotate_rows(self, shift: int):
        """Circular shift instruments/rows."""
        if self.backend == 'torch':
            self.state = torch.roll(self.state, shifts=shift, dims=1)
        else:
            self.state = np.roll(self.state, shift=shift, axis=1)
        
        # Rotate subdivision keys
        new_subdivs = {}
        for (row, col), tensor in self.subdivisions.items():
            new_row = (row + shift) % self.rows
            new_subdivs[(new_row, col)] = tensor
        self.subdivisions = new_subdivs
        self._fully_dirty = True
    
    def double_time(self):
        """Compress pattern to half length, then repeat."""
        # Take every other column
        half = self.state[:, :, ::2].copy() if not HAS_TORCH else self.state[:, :, ::2].clone()
        
        # Repeat
        if self.backend == 'torch':
            self.state = torch.cat([half, half], dim=2)
        else:
            self.state = np.concatenate([half, half], axis=2)
        
        # Subdivisions get more complex - for now, clear them
        self.subdivisions.clear()
        self._fully_dirty = True
    
    def half_time(self):
        """Expand pattern to double length (first half only populated)."""
        # Stretch columns
        new_state = np.zeros((4, self.rows, self.cols), dtype=np.float32)
        for col in range(self.cols // 2):
            new_state[:, :, col * 2] = self.state[:, :, col]
        
        if self.backend == 'torch':
            self.state = torch.from_numpy(new_state)
        else:
            self.state = new_state
        
        self.subdivisions.clear()
        self._fully_dirty = True
    
    # =========================================================================
    # QUERY OPERATIONS
    # =========================================================================
    
    def get_active_at_beat(self, col: int) -> np.ndarray:
        """
        Get which rows are active at a specific beat.
        Returns boolean array of shape (rows,).
        """
        if not 0 <= col < self.cols:
            return np.zeros(self.rows, dtype=bool)
        
        active = self.state[Channel.ACTIVE, :, col]
        if self.backend == 'torch':
            return (active > 0.5).cpu().numpy()
        return active > 0.5
    
    def get_velocities_at_beat(self, col: int) -> np.ndarray:
        """Get velocities for all rows at a specific beat."""
        if not 0 <= col < self.cols:
            return np.zeros(self.rows, dtype=np.float32)
        
        vel = self.state[Channel.VELOCITY, :, col]
        if self.backend == 'torch':
            return vel.cpu().numpy()
        return vel.copy()
    
    def query_at_time(self, beat: float) -> List[Tuple[int, float, float]]:
        """
        Query what should fire at an exact beat position.
        
        Returns list of (row, velocity, timing_offset) for events
        that should fire at this moment, accounting for subdivisions.
        """
        col = int(beat)
        frac = beat - col
        
        if col < 0 or col >= self.cols:
            return []
        
        results = []
        
        for row in range(self.rows):
            if self.state[Channel.ACTIVE, row, col] < 0.5:
                continue
            
            velocity = float(self.state[Channel.VELOCITY, row, col])
            timing = float(self.state[Channel.TIMING, row, col])
            subdiv_level = int(self.state[Channel.SUBDIVISION, row, col])
            
            if subdiv_level <= 1:
                # No subdivision - fire at beat start
                if frac < 0.01:  # Small tolerance
                    results.append((row, velocity, timing))
            else:
                # Check subdivision
                key = (row, col)
                if key in self.subdivisions:
                    sub_tensor = self.subdivisions[key]
                    sub_idx = int(frac * subdiv_level)
                    sub_frac = (frac * subdiv_level) - sub_idx
                    
                    if sub_idx < sub_tensor.shape[1] and sub_frac < 0.01:
                        if sub_tensor[Channel.ACTIVE, sub_idx, 0] > 0.5:
                            sub_vel = float(sub_tensor[Channel.VELOCITY, sub_idx, 0])
                            results.append((row, sub_vel * velocity, timing))
        
        return results
    
    # =========================================================================
    # SHADER INTERFACE
    # =========================================================================
    
    def to_rgba_bytes(self) -> bytes:
        """
        Convert to RGBA texture bytes for shader upload.
        Layout: (rows, cols, 4) as contiguous RGBA pixels.
        """
        # Permute from (C, H, W) to (H, W, C)
        if self.backend == 'torch':
            data = self.state.permute(1, 2, 0).cpu().numpy()
        else:
            data = self.state.transpose(1, 2, 0)
        
        return data.astype(np.float32).tobytes()
    
    def to_rgba_array(self) -> np.ndarray:
        """
        Convert to RGBA array for shader upload.
        Shape: (rows, cols, 4)
        """
        if self.backend == 'torch':
            return self.state.permute(1, 2, 0).cpu().numpy().astype(np.float32)
        return self.state.transpose(1, 2, 0).astype(np.float32)
    
    def texture_size(self) -> Tuple[int, int]:
        """Return (width, height) for texture creation."""
        return (self.cols, self.rows)
    
    # =========================================================================
    # DIRTY TRACKING
    # =========================================================================
    
    def _in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.rows and 0 <= col < self.cols
    
    def _mark_dirty(self, row: int, col: int):
        self._dirty_cells.add((row, col))
    
    def is_dirty(self) -> bool:
        return self._fully_dirty or len(self._dirty_cells) > 0
    
    def get_dirty_cells(self) -> Set[Tuple[int, int]]:
        return self._dirty_cells.copy()
    
    def clear_dirty(self):
        self._dirty_cells.clear()
        self._fully_dirty = False
    
    # =========================================================================
    # SERIALIZATION
    # =========================================================================
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        if self.backend == 'torch':
            state_list = self.state.cpu().numpy().tolist()
        else:
            state_list = self.state.tolist()
        
        subdivs = {}
        for key, tensor in self.subdivisions.items():
            if self.backend == 'torch':
                subdivs[f"{key[0]},{key[1]}"] = tensor.cpu().numpy().tolist()
            else:
                subdivs[f"{key[0]},{key[1]}"] = tensor.tolist()
        
        return {
            'rows': self.rows,
            'cols': self.cols,
            'state': state_list,
            'subdivisions': subdivs
        }
    
    @classmethod
    def from_dict(cls, data: dict, backend: str = 'numpy') -> 'SequenceTensor':
        """Deserialize from dictionary."""
        grid = cls(data['rows'], data['cols'], backend)
        
        if backend == 'torch' and HAS_TORCH:
            grid.state = torch.tensor(data['state'], dtype=torch.float32)
        else:
            grid.state = np.array(data['state'], dtype=np.float32)
        
        for key_str, tensor_list in data.get('subdivisions', {}).items():
            row, col = map(int, key_str.split(','))
            if backend == 'torch' and HAS_TORCH:
                grid.subdivisions[(row, col)] = torch.tensor(tensor_list, dtype=torch.float32)
            else:
                grid.subdivisions[(row, col)] = np.array(tensor_list, dtype=np.float32)
        
        return grid
    
    def __repr__(self) -> str:
        active_count = int((self.state[Channel.ACTIVE] > 0.5).sum())
        subdiv_count = len(self.subdivisions)
        return f"SequenceTensor({self.rows}×{self.cols}, {active_count} active, {subdiv_count} subdivisions)"


# =============================================================================
# SHADER EVENT QUEUE
# =============================================================================

class ShaderEventType(IntEnum):
    """Event types emitted from shader → CPU."""
    TRIGGER = 1       # Playhead crossed an active cell
    LANDED = 2        # Event arrived at destination (from timeline)
    COLLISION = 3     # Two events overlapped
    THRESHOLD = 4     # Intensity crossed a threshold


@dataclass
class ShaderEvent:
    """Event emitted from shader."""
    type: ShaderEventType
    row: int
    col: int
    data0: float = 0.0
    data1: float = 0.0
    timestamp: float = 0.0


class ShaderEventQueue:
    """
    Ring buffer for GPU → CPU event communication.
    
    The shader writes events to an SSBO, the host polls and drains them.
    """
    
    def __init__(self, capacity: int = 64, backend: str = 'numpy'):
        self.capacity = capacity
        self.backend = backend
        
        # Each event: [type, row, col, data0, data1, timestamp]
        if backend == 'torch' and HAS_TORCH:
            self.events = torch.zeros((capacity, 6), dtype=torch.float32)
            self.head = torch.zeros(1, dtype=torch.int32)  # Write position
            self.tail = torch.zeros(1, dtype=torch.int32)  # Read position
        else:
            self.events = np.zeros((capacity, 6), dtype=np.float32)
            self.head = np.zeros(1, dtype=np.int32)
            self.tail = np.zeros(1, dtype=np.int32)
    
    def poll(self) -> List[ShaderEvent]:
        """Drain pending events from the queue."""
        h = int(self.head[0])
        t = int(self.tail[0])
        
        if h == t:
            return []
        
        results = []
        
        # Read events
        if h > t:
            event_data = self.events[t:h]
        else:
            # Wrapped around
            if self.backend == 'torch' and HAS_TORCH:
                event_data = torch.cat([self.events[t:], self.events[:h]])
            else:
                event_data = np.concatenate([self.events[t:], self.events[:h]])
        
        # Parse into ShaderEvent objects
        for row_data in event_data:
            if self.backend == 'torch' and HAS_TORCH:
                row_data = row_data.cpu().numpy()
            
            results.append(ShaderEvent(
                type=ShaderEventType(int(row_data[0])),
                row=int(row_data[1]),
                col=int(row_data[2]),
                data0=float(row_data[3]),
                data1=float(row_data[4]),
                timestamp=float(row_data[5])
            ))
        
        # Update tail to match head (queue drained)
        self.tail[0] = h
        
        return results
    
    def clear(self):
        """Clear the queue."""
        self.head[0] = 0
        self.tail[0] = 0
    
    def to_ssbo_bytes(self) -> bytes:
        """Get buffer data for SSBO upload."""
        # Pack head + events into contiguous buffer
        if self.backend == 'torch' and HAS_TORCH:
            head_data = self.head.cpu().numpy().astype(np.int32)
            event_data = self.events.cpu().numpy().astype(np.float32)
        else:
            head_data = self.head.astype(np.int32)
            event_data = self.events.astype(np.float32)
        
        return head_data.tobytes() + event_data.tobytes()
    
    def from_ssbo_bytes(self, data: bytes):
        """Update queue from SSBO readback."""
        head_bytes = 4  # int32
        head_data = np.frombuffer(data[:head_bytes], dtype=np.int32)
        event_data = np.frombuffer(data[head_bytes:], dtype=np.float32)
        event_data = event_data.reshape(self.capacity, 6)
        
        if self.backend == 'torch' and HAS_TORCH:
            self.head = torch.from_numpy(head_data.copy())
            self.events = torch.from_numpy(event_data.copy())
        else:
            self.head = head_data.copy()
            self.events = event_data.copy()


# =============================================================================
# QUANTIZATION UTILITIES
# =============================================================================

def quantize_to_subdivision(frac: float, candidates: List[int] = None) -> Subdivision:
    """
    Find best-fit subdivision for a fractional beat position.
    
    Args:
        frac: Position within beat (0.0 to 1.0)
        candidates: List of subdivision levels to try
        
    Returns:
        Subdivision with level, index, and quantization error
    """
    if candidates is None:
        candidates = [1, 2, 3, 4, 6, 8, 12, 16]
    
    best = Subdivision(level=1, index=0, error=frac if frac < 0.5 else 1.0 - frac)
    
    for n in candidates:
        # Find closest position in this subdivision
        idx = round(frac * n)
        if idx >= n:
            idx = n - 1
        
        quantized = idx / n
        error = abs(frac - quantized)
        
        if error < best.error:
            best = Subdivision(level=n, index=idx, error=error)
    
    return best


def detect_rhythm_set(positions: List[float], tolerance: float = 0.05) -> Optional[RhythmSet]:
    """
    Try to match a list of fractional positions to a known rhythm set.
    
    Args:
        positions: List of positions within beat (0.0 to 1.0)
        tolerance: Matching tolerance
        
    Returns:
        Best matching RhythmSet or None
    """
    known_sets = [
        RhythmSet.STRAIGHT_8,
        RhythmSet.STRAIGHT_16,
        RhythmSet.SWING_8,
        RhythmSet.SWING_16,
        RhythmSet.TRIPLET,
        RhythmSet.CLAVE_3_2
    ]
    
    best_set = None
    best_score = 0
    
    for rs in known_sets:
        matches = 0
        for pos in positions:
            if rs.matches(pos, tolerance) is not None:
                matches += 1
        
        score = matches / len(positions) if positions else 0
        if score > best_score:
            best_score = score
            best_set = rs
    
    return best_set if best_score > 0.8 else None
