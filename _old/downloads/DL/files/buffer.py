"""
GPU Buffer System

Persistent GPU buffers with dirty tracking for efficient partial updates.
This is the foundation for instanced rendering, particle systems, and any
mass data visualization.

Key principles:
1. Buffer is persistent - created once, updated incrementally
2. CPU staging array is the source of truth for edits
3. Dirty tracking enables partial uploads
4. Agnostic to data meaning - just slots of structured data
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any
import numpy as np
import moderngl


# =============================================================================
# Standard Instance Layout
# =============================================================================

# Default layout for instanced rendering (64 bytes per instance, good alignment)
INSTANCE_DTYPE = np.dtype([
    ('position',  np.float32, 3),   # 12 bytes - world position
    ('scale',     np.float32, 3),   # 12 bytes - scale (can be non-uniform)
    ('rotation',  np.float32, 4),   # 16 bytes - quaternion (w, x, y, z)
    ('color',     np.float32, 4),   # 16 bytes - RGBA
    ('value',     np.float32),      # 4 bytes  - generic scalar for shaders
    ('flags',     np.uint32),       # 4 bytes  - visibility, selection, etc.
])

INSTANCE_STRIDE = INSTANCE_DTYPE.itemsize  # 64 bytes


# Flag bits for the 'flags' field
class InstanceFlags:
    VISIBLE = 1 << 0
    SELECTED = 1 << 1
    HIGHLIGHTED = 1 << 2
    CAST_SHADOW = 1 << 3


# =============================================================================
# GPU Buffer
# =============================================================================

class GPUBuffer:
    """
    Persistent GPU buffer with CPU staging and dirty tracking.
    
    The buffer holds structured data (numpy structured array on CPU,
    raw bytes on GPU). Edits go to the CPU staging array and mark
    dirty ranges. sync() uploads only the dirty bytes to GPU.
    
    This class is agnostic to what the data represents - it just
    manages slots of structured data efficiently.
    """
    
    def __init__(
        self, 
        ctx: moderngl.Context, 
        capacity: int,
        dtype: np.dtype = INSTANCE_DTYPE,
        name: str = "unnamed"
    ):
        self.ctx = ctx
        self.capacity = capacity
        self.dtype = dtype
        self.stride = dtype.itemsize
        self.name = name
        
        # CPU staging buffer - source of truth for edits
        self.data = np.zeros(capacity, dtype=dtype)
        
        # GPU buffer - updated via sync()
        self.gpu: moderngl.Buffer = ctx.buffer(reserve=capacity * self.stride)
        
        # Dirty tracking (None = clean)
        self._dirty_min: Optional[int] = None
        self._dirty_max: Optional[int] = None
        
        # Stats
        self._total_syncs = 0
        self._total_bytes_uploaded = 0
    
    # =========================================================================
    # Properties
    # =========================================================================
    
    @property
    def is_dirty(self) -> bool:
        return self._dirty_min is not None
    
    @property
    def dirty_range(self) -> Optional[Tuple[int, int]]:
        if self._dirty_min is None:
            return None
        return (self._dirty_min, self._dirty_max)
    
    # =========================================================================
    # Field Accessors (views into structured array)
    # =========================================================================
    
    @property
    def positions(self) -> np.ndarray:
        """View of all positions. Shape: (capacity, 3)"""
        return self.data['position']
    
    @property
    def scales(self) -> np.ndarray:
        """View of all scales. Shape: (capacity, 3)"""
        return self.data['scale']
    
    @property
    def rotations(self) -> np.ndarray:
        """View of all rotations (quaternions). Shape: (capacity, 4)"""
        return self.data['rotation']
    
    @property
    def colors(self) -> np.ndarray:
        """View of all colors. Shape: (capacity, 4)"""
        return self.data['color']
    
    @property
    def values(self) -> np.ndarray:
        """View of all values. Shape: (capacity,)"""
        return self.data['value']
    
    @property
    def flags(self) -> np.ndarray:
        """View of all flags. Shape: (capacity,)"""
        return self.data['flags']
    
    # =========================================================================
    # Read Access (no sync needed)
    # =========================================================================
    
    def __getitem__(self, idx):
        """Direct read: buf[5] or buf[10:20]"""
        return self.data[idx]
    
    def __len__(self) -> int:
        return self.capacity
    
    # =========================================================================
    # Write Access (marks dirty)
    # =========================================================================
    
    def __setitem__(self, idx, value):
        """Direct write: buf[5] = data or buf[10:20] = array"""
        self.data[idx] = value
        
        if isinstance(idx, int):
            self._mark_dirty(idx, idx + 1)
        elif isinstance(idx, slice):
            start = idx.start or 0
            stop = idx.stop if idx.stop is not None else self.capacity
            self._mark_dirty(start, stop)
        elif isinstance(idx, np.ndarray):
            # Array indexing
            self._mark_dirty(int(idx.min()), int(idx.max()) + 1)
    
    def write(self, start: int, values: np.ndarray):
        """Write values starting at index."""
        end = start + len(values)
        self.data[start:end] = values
        self._mark_dirty(start, end)
    
    def write_field(self, field: str, start: int, end: int, values: np.ndarray):
        """Write a single field for a range of slots."""
        self.data[field][start:end] = values
        self._mark_dirty(start, end)
    
    def set_slot(self, idx: int, **fields):
        """
        Set multiple fields on a single slot.
        
        Usage:
            buf.set_slot(5, position=[1,2,3], color=[1,0,0,1])
        """
        for field, value in fields.items():
            self.data[field][idx] = value
        self._mark_dirty(idx, idx + 1)
    
    def set_range(self, start: int, end: int, **fields):
        """
        Set multiple fields for a range of slots.
        
        Usage:
            buf.set_range(0, 100, position=positions_array, color=colors_array)
        """
        for field, value in fields.items():
            self.data[field][start:end] = value
        self._mark_dirty(start, end)
    
    # =========================================================================
    # Dirty Tracking
    # =========================================================================
    
    def _mark_dirty(self, start: int, end: int):
        """Mark a range as dirty."""
        if self._dirty_min is None:
            self._dirty_min = start
            self._dirty_max = end
        else:
            self._dirty_min = min(self._dirty_min, start)
            self._dirty_max = max(self._dirty_max, end)
    
    def mark_all_dirty(self):
        """Mark entire buffer as dirty (for initial upload)."""
        self._mark_dirty(0, self.capacity)
    
    def clear_dirty(self):
        """Clear dirty flags without uploading."""
        self._dirty_min = None
        self._dirty_max = None
    
    # =========================================================================
    # GPU Sync
    # =========================================================================
    
    def sync(self) -> int:
        """
        Upload dirty region to GPU.
        
        Returns number of slots uploaded (0 if clean).
        Call this once per frame before rendering.
        """
        if self._dirty_min is None:
            return 0
        
        start = self._dirty_min
        end = self._dirty_max
        
        byte_offset = start * self.stride
        byte_data = self.data[start:end].tobytes()
        
        self.gpu.write(byte_data, offset=byte_offset)
        
        slots_uploaded = end - start
        self._total_syncs += 1
        self._total_bytes_uploaded += len(byte_data)
        
        self._dirty_min = None
        self._dirty_max = None
        
        return slots_uploaded
    
    def upload_all(self):
        """Force upload entire buffer (for initialization)."""
        self.gpu.write(self.data.tobytes())
        self._total_syncs += 1
        self._total_bytes_uploaded += self.data.nbytes
        self.clear_dirty()
    
    # =========================================================================
    # Initialization Helpers
    # =========================================================================
    
    def init_defaults(self, start: int = 0, end: Optional[int] = None):
        """
        Initialize slots with sensible defaults.
        
        - position: [0, 0, 0]
        - scale: [1, 1, 1]
        - rotation: [1, 0, 0, 0] (identity quaternion)
        - color: [1, 1, 1, 1] (white)
        - value: 0
        - flags: VISIBLE
        """
        if end is None:
            end = self.capacity
        
        self.data['position'][start:end] = [0, 0, 0]
        self.data['scale'][start:end] = [1, 1, 1]
        self.data['rotation'][start:end] = [1, 0, 0, 0]  # w, x, y, z
        self.data['color'][start:end] = [1, 1, 1, 1]
        self.data['value'][start:end] = 0
        self.data['flags'][start:end] = InstanceFlags.VISIBLE
        
        self._mark_dirty(start, end)
    
    # =========================================================================
    # VAO Binding
    # =========================================================================
    
    def get_instance_format(self) -> str:
        """
        Get moderngl format string for instance attributes.
        
        For INSTANCE_DTYPE: '3f 3f 4f 4f 1f 1u'
        The '/i' suffix (per-instance) is added during VAO creation.
        """
        # Build format from dtype
        formats = []
        for name in self.dtype.names:
            field_dtype, *shape = self.dtype.fields[name]
            base = field_dtype.base
            
            # Determine count
            if len(shape) > 0 and len(shape[0]) > 0:
                count = shape[0][0]
            else:
                count = 1
            
            # Determine type character
            if base == np.float32:
                fmt_char = 'f'
            elif base == np.uint32:
                fmt_char = 'u'
            elif base == np.int32:
                fmt_char = 'i'
            else:
                fmt_char = 'f'  # fallback
            
            formats.append(f'{count}{fmt_char}')
        
        return ' '.join(formats)
    
    def get_attribute_names(self) -> List[str]:
        """Get attribute names prefixed for shaders (inst_position, etc.)"""
        return [f'inst_{name}' for name in self.dtype.names]
    
    # =========================================================================
    # Stats / Debug
    # =========================================================================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        return {
            'name': self.name,
            'capacity': self.capacity,
            'stride': self.stride,
            'total_bytes': self.capacity * self.stride,
            'is_dirty': self.is_dirty,
            'dirty_range': self.dirty_range,
            'total_syncs': self._total_syncs,
            'total_bytes_uploaded': self._total_bytes_uploaded,
        }
    
    # =========================================================================
    # Cleanup
    # =========================================================================
    
    def release(self):
        """Release GPU resources."""
        if self.gpu:
            self.gpu.release()
            self.gpu = None


# =============================================================================
# Buffer Registry
# =============================================================================

class BufferRegistry:
    """
    Central registry for GPU buffers.
    
    Provides named access to buffers and handles lifecycle.
    """
    
    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        self._buffers: Dict[str, GPUBuffer] = {}
    
    def create(
        self, 
        name: str, 
        capacity: int, 
        dtype: np.dtype = INSTANCE_DTYPE
    ) -> GPUBuffer:
        """Create and register a new buffer."""
        if name in self._buffers:
            raise ValueError(f"Buffer '{name}' already exists")
        
        buf = GPUBuffer(self.ctx, capacity, dtype, name)
        self._buffers[name] = buf
        return buf
    
    def get(self, name: str) -> GPUBuffer:
        """Get a buffer by name."""
        return self._buffers[name]
    
    def has(self, name: str) -> bool:
        """Check if buffer exists."""
        return name in self._buffers
    
    def sync_all(self) -> int:
        """Sync all dirty buffers. Returns total slots uploaded."""
        total = 0
        for buf in self._buffers.values():
            total += buf.sync()
        return total
    
    def release_all(self):
        """Release all buffers."""
        for buf in self._buffers.values():
            buf.release()
        self._buffers.clear()
    
    def get_all_stats(self) -> List[Dict[str, Any]]:
        """Get stats for all buffers."""
        return [buf.get_stats() for buf in self._buffers.values()]
