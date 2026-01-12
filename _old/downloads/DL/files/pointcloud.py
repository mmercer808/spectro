"""
Point Cloud System

A PointCloud is a collection of instances that owns a contiguous range
in a GPUBuffer. It's the mass-instance equivalent of EntityNode.

EntityNode: Single object with Transform + MeshRenderer
PointCloud: Many instances sharing the same mesh, each with position/color/etc.

Key principles:
1. PointCloud owns a range [start, start+count) in a GPUBuffer
2. Edits go through PointCloud, which writes to the buffer and marks dirty
3. PointCloud doesn't own the buffer - multiple clouds can share one buffer
4. Animation systems update PointCloud, buffer.sync() uploads to GPU
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Callable, Any, Dict
import numpy as np

from engine.render.buffer import GPUBuffer, InstanceFlags


# =============================================================================
# Point Cloud
# =============================================================================

class PointCloud:
    """
    A collection of instances that owns a range in a GPUBuffer.
    
    Think of it as "1000 cubes" rather than "1000 EntityNodes each with a cube".
    All instances share the same mesh and pipeline.
    """
    
    def __init__(
        self,
        name: str,
        buffer: GPUBuffer,
        start: int,
        count: int,
        mesh_id: str = "cube",
        pipeline_id: str = "lit_color_instanced",
    ):
        self.name = name
        self.buffer = buffer
        self.start = start
        self.count = count
        self.mesh_id = mesh_id
        self.pipeline_id = pipeline_id
        
        # Metadata
        self.visible = True
        self.user_data: Dict[str, Any] = {}
    
    # =========================================================================
    # Range Properties
    # =========================================================================
    
    @property
    def end(self) -> int:
        return self.start + self.count
    
    @property
    def slice(self) -> slice:
        """Slice object for this cloud's range."""
        return slice(self.start, self.end)
    
    # =========================================================================
    # Field Views (read/write into buffer)
    # =========================================================================
    
    @property
    def positions(self) -> np.ndarray:
        """View of this cloud's positions. Shape: (count, 3)"""
        return self.buffer.positions[self.start:self.end]
    
    @positions.setter
    def positions(self, values: np.ndarray):
        self.buffer.write_field('position', self.start, self.end, values)
    
    @property
    def scales(self) -> np.ndarray:
        """View of this cloud's scales. Shape: (count, 3)"""
        return self.buffer.scales[self.start:self.end]
    
    @scales.setter
    def scales(self, values: np.ndarray):
        self.buffer.write_field('scale', self.start, self.end, values)
    
    @property
    def rotations(self) -> np.ndarray:
        """View of this cloud's rotations. Shape: (count, 4)"""
        return self.buffer.rotations[self.start:self.end]
    
    @rotations.setter
    def rotations(self, values: np.ndarray):
        self.buffer.write_field('rotation', self.start, self.end, values)
    
    @property
    def colors(self) -> np.ndarray:
        """View of this cloud's colors. Shape: (count, 4)"""
        return self.buffer.colors[self.start:self.end]
    
    @colors.setter
    def colors(self, values: np.ndarray):
        self.buffer.write_field('color', self.start, self.end, values)
    
    @property
    def values(self) -> np.ndarray:
        """View of this cloud's values. Shape: (count,)"""
        return self.buffer.values[self.start:self.end]
    
    @values.setter
    def values(self, values: np.ndarray):
        self.buffer.write_field('value', self.start, self.end, values)
    
    @property
    def flags(self) -> np.ndarray:
        """View of this cloud's flags. Shape: (count,)"""
        return self.buffer.flags[self.start:self.end]
    
    @flags.setter
    def flags(self, values: np.ndarray):
        self.buffer.write_field('flags', self.start, self.end, values)
    
    # =========================================================================
    # Single Point Access
    # =========================================================================
    
    def get_point(self, local_idx: int) -> np.ndarray:
        """Get full data for a single point (by local index)."""
        return self.buffer[self.start + local_idx]
    
    def set_point(self, local_idx: int, **fields):
        """
        Set fields on a single point.
        
        Usage:
            cloud.set_point(5, position=[1,2,3], color=[1,0,0,1])
        """
        global_idx = self.start + local_idx
        self.buffer.set_slot(global_idx, **fields)
    
    def get_position(self, local_idx: int) -> np.ndarray:
        """Get position of a single point."""
        return self.buffer.positions[self.start + local_idx]
    
    def set_position(self, local_idx: int, pos: np.ndarray):
        """Set position of a single point."""
        global_idx = self.start + local_idx
        self.buffer.positions[global_idx] = pos
        self.buffer._mark_dirty(global_idx, global_idx + 1)
    
    # =========================================================================
    # Bulk Operations
    # =========================================================================
    
    def set_all_positions(self, positions: np.ndarray):
        """Set all positions at once. Shape: (count, 3)"""
        assert len(positions) == self.count
        self.positions = positions
    
    def set_all_colors(self, colors: np.ndarray):
        """Set all colors at once. Shape: (count, 4) or (4,) for uniform."""
        if colors.ndim == 1:
            # Broadcast single color to all
            colors = np.tile(colors, (self.count, 1))
        assert len(colors) == self.count
        self.colors = colors
    
    def set_uniform_scale(self, scale: float):
        """Set all instances to the same uniform scale."""
        self.scales[:] = [scale, scale, scale]
        self.buffer._mark_dirty(self.start, self.end)
    
    def set_all_visible(self, visible: bool = True):
        """Set visibility for all instances."""
        if visible:
            self.flags[:] = self.flags | InstanceFlags.VISIBLE
        else:
            self.flags[:] = self.flags & ~InstanceFlags.VISIBLE
        self.buffer._mark_dirty(self.start, self.end)
    
    # =========================================================================
    # Initialization
    # =========================================================================
    
    def init_defaults(self):
        """Initialize all points with default values."""
        self.buffer.init_defaults(self.start, self.end)
    
    def init_grid(self, nx: int, ny: int, nz: int, spacing: float = 1.0, center: bool = True):
        """
        Initialize as a 3D grid of points.
        
        Args:
            nx, ny, nz: Grid dimensions
            spacing: Distance between points
            center: If True, center grid at origin
        """
        assert nx * ny * nz <= self.count, "Grid too large for cloud capacity"
        
        positions = np.zeros((self.count, 3), dtype=np.float32)
        
        idx = 0
        for iz in range(nz):
            for iy in range(ny):
                for ix in range(nx):
                    if idx >= self.count:
                        break
                    positions[idx] = [ix * spacing, iy * spacing, iz * spacing]
                    idx += 1
        
        if center:
            center_offset = np.array([
                (nx - 1) * spacing / 2,
                (ny - 1) * spacing / 2,
                (nz - 1) * spacing / 2,
            ], dtype=np.float32)
            positions -= center_offset
        
        self.init_defaults()
        self.positions = positions
    
    def init_random(self, bounds: float = 10.0):
        """Initialize with random positions in a cube."""
        positions = (np.random.rand(self.count, 3).astype(np.float32) - 0.5) * bounds * 2
        self.init_defaults()
        self.positions = positions
    
    # =========================================================================
    # Subset Operations (for partial updates)
    # =========================================================================
    
    def update_subset(self, local_indices: np.ndarray, **fields):
        """
        Update specific points by local index.
        
        Usage:
            cloud.update_subset([0, 5, 10], position=new_positions)
        """
        global_indices = self.start + local_indices
        
        for field_name, values in fields.items():
            self.buffer.data[field_name][global_indices] = values
        
        if len(local_indices) > 0:
            self.buffer._mark_dirty(
                self.start + int(local_indices.min()),
                self.start + int(local_indices.max()) + 1
            )
    
    # =========================================================================
    # Animation Helpers
    # =========================================================================
    
    def animate_positions(self, fn: Callable[[np.ndarray, float], np.ndarray], t: float):
        """
        Apply animation function to positions.
        
        Args:
            fn: Function(current_positions, time) -> new_positions
            t: Current time
        """
        new_positions = fn(self.positions.copy(), t)
        self.positions = new_positions
    
    def animate_colors(self, fn: Callable[[np.ndarray, float], np.ndarray], t: float):
        """Apply animation function to colors."""
        new_colors = fn(self.colors.copy(), t)
        self.colors = new_colors
    
    # =========================================================================
    # Debug
    # =========================================================================
    
    def __repr__(self) -> str:
        return f"PointCloud(name={self.name!r}, range=[{self.start}:{self.end}], count={self.count})"


# =============================================================================
# Point Cloud Manager
# =============================================================================

class PointCloudManager:
    """
    Manages multiple PointClouds sharing a GPUBuffer.
    
    Handles allocation of ranges within the buffer.
    """
    
    def __init__(self, buffer: GPUBuffer):
        self.buffer = buffer
        self._clouds: Dict[str, PointCloud] = {}
        self._next_slot = 0
    
    def create(
        self,
        name: str,
        count: int,
        mesh_id: str = "cube",
        pipeline_id: str = "lit_color_instanced",
    ) -> PointCloud:
        """
        Allocate a new PointCloud with the given number of instances.
        
        Raises ValueError if not enough space.
        """
        if name in self._clouds:
            raise ValueError(f"PointCloud '{name}' already exists")
        
        if self._next_slot + count > self.buffer.capacity:
            raise ValueError(
                f"Not enough space: need {count}, have {self.buffer.capacity - self._next_slot}"
            )
        
        cloud = PointCloud(
            name=name,
            buffer=self.buffer,
            start=self._next_slot,
            count=count,
            mesh_id=mesh_id,
            pipeline_id=pipeline_id,
        )
        
        self._clouds[name] = cloud
        self._next_slot += count
        
        return cloud
    
    def get(self, name: str) -> PointCloud:
        """Get a cloud by name."""
        return self._clouds[name]
    
    def has(self, name: str) -> bool:
        """Check if cloud exists."""
        return name in self._clouds
    
    def all(self) -> List[PointCloud]:
        """Get all clouds."""
        return list(self._clouds.values())
    
    @property
    def total_instances(self) -> int:
        """Total instances across all clouds."""
        return sum(c.count for c in self._clouds.values())
    
    @property
    def allocated_slots(self) -> int:
        """Number of buffer slots allocated."""
        return self._next_slot
    
    @property
    def free_slots(self) -> int:
        """Number of buffer slots remaining."""
        return self.buffer.capacity - self._next_slot
    
    def get_stats(self) -> Dict[str, Any]:
        """Get manager statistics."""
        return {
            'cloud_count': len(self._clouds),
            'total_instances': self.total_instances,
            'allocated_slots': self.allocated_slots,
            'free_slots': self.free_slots,
            'buffer_capacity': self.buffer.capacity,
        }
