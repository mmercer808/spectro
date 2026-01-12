"""
RenderWorld Batching Layer

This module transforms raw render items into efficient batched draw commands.
It's the bridge between "what to render" (snapshot) and "how to render" (commands).

Key concepts:
1. Material Buckets: Group items by (pipeline, mesh, material) for batching
2. Instancing: Multiple items with same mesh/material → single instanced draw
3. Sorting: Opaque front-to-back, transparent back-to-front
4. Instance Buffers: Packed transform + per-instance data for GPU upload
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
import numpy as np

from engine.core.snapshot import GraphSnapshot, SnapshotRenderItem, SnapshotCamera, CameraViewport
from engine.render.commands import (
    CommandList, Command,
    CmdSetViewport, CmdSetScissor, CmdClear, CmdDrawMesh, CmdDrawMeshInstanced
)


@dataclass
class MaterialKey:
    """
    Unique identifier for a material configuration.
    Items with the same MaterialKey can be batched together.
    """
    pipeline_id: str
    mesh_id: str
    # Future: texture_ids, blend_mode, etc.
    
    def __hash__(self):
        return hash((self.pipeline_id, self.mesh_id))
    
    def __eq__(self, other):
        if not isinstance(other, MaterialKey):
            return False
        return self.pipeline_id == other.pipeline_id and self.mesh_id == other.mesh_id


@dataclass
class InstanceData:
    """Per-instance data for batched rendering."""
    entity_id: int
    world_matrix: np.ndarray  # 4x4 float32
    color: Tuple[float, float, float, float]
    # Future: custom instance attributes


@dataclass
class RenderBucket:
    """
    A bucket of instances that share the same material.
    One draw call per bucket (instanced if count > 1).
    """
    key: MaterialKey
    instances: List[InstanceData] = field(default_factory=list)
    
    @property
    def instance_count(self) -> int:
        return len(self.instances)
    
    def build_instance_buffer(self) -> np.ndarray:
        """
        Pack instance data into a GPU-uploadable buffer.
        
        Layout per instance (stride = 80 bytes = 20 floats):
        - mat4 world (16 floats)
        - vec4 color (4 floats)
        """
        count = len(self.instances)
        if count == 0:
            return np.array([], dtype=np.float32)
        
        # 16 floats for mat4 + 4 floats for color = 20 floats per instance
        buffer = np.zeros((count, 20), dtype=np.float32)
        
        for i, inst in enumerate(self.instances):
            buffer[i, :16] = inst.world_matrix.flatten()
            buffer[i, 16:20] = inst.color
        
        return buffer.flatten()
    
    def add(self, item: SnapshotRenderItem):
        """Add a render item to this bucket."""
        self.instances.append(InstanceData(
            entity_id=item.entity_id,
            world_matrix=item.get_world_matrix(),
            color=item.color,
        ))


class RenderWorld:
    """
    The render-friendly representation of a scene snapshot.
    
    Transforms a GraphSnapshot into efficiently organized buckets
    ready for command emission.
    """
    
    def __init__(self, snapshot: GraphSnapshot):
        self.snapshot = snapshot
        self.buckets: Dict[MaterialKey, RenderBucket] = {}
        
        # Build buckets from snapshot
        self._bucket_items()
    
    def _bucket_items(self):
        """Sort items into material buckets."""
        for item in self.snapshot.items:
            key = MaterialKey(
                pipeline_id=item.pipeline_id,
                mesh_id=item.mesh_id,
            )
            
            if key not in self.buckets:
                self.buckets[key] = RenderBucket(key=key)
            
            self.buckets[key].add(item)
    
    def get_bucket_count(self) -> int:
        return len(self.buckets)
    
    def get_instance_count(self) -> int:
        return sum(b.instance_count for b in self.buckets.values())
    
    def emit_commands_for_camera(
        self,
        camera: SnapshotCamera,
        viewport: CameraViewport,
        target_h: int,
        use_instancing: bool = True,
        instancing_threshold: int = 4,
    ) -> CommandList:
        """
        Emit draw commands for a single camera.
        
        Args:
            camera: The camera to render from
            viewport: The sub-viewport region
            target_h: Total render target height (for GL coordinate flip)
            use_instancing: Whether to use instanced draws for batches
            instancing_threshold: Minimum instances to trigger instancing
        
        Returns:
            CommandList ready for execution
        """
        cmdlist = CommandList()
        
        # Calculate view and projection matrices
        V = self._look_at(camera.get_eye(), camera.get_target(), camera.get_up())
        aspect = max(1e-6, viewport.w / max(1.0, float(viewport.h)))
        P = self._perspective(camera.fov_y_deg, aspect, camera.near, camera.far)
        VP = P @ V
        
        # Set viewport and scissor (flip Y for GL)
        gl_y = target_h - (viewport.y + viewport.h)
        cmdlist.add(CmdSetViewport(viewport.x, gl_y, viewport.w, viewport.h))
        cmdlist.add(CmdSetScissor(viewport.x, gl_y, viewport.w, viewport.h, enabled=True))
        
        # Emit draws per bucket
        for key, bucket in self.buckets.items():
            count = bucket.instance_count
            if count == 0:
                continue
            
            if use_instancing and count >= instancing_threshold:
                # Instanced draw
                cmdlist.add(self._emit_instanced(bucket, VP, camera.index))
            else:
                # Individual draws
                for inst in bucket.instances:
                    mvp = (VP @ inst.world_matrix).astype(np.float32)
                    cmdlist.add(CmdDrawMesh(
                        pipeline_id=key.pipeline_id,
                        mesh_id=key.mesh_id,
                        uniforms={"u_mvp": mvp, "u_color": np.array(inst.color, dtype=np.float32)},
                        entity_id=inst.entity_id,
                    ))
        
        # Clear scissor
        cmdlist.add(CmdSetScissor(0, 0, 0, 0, enabled=False))
        
        return cmdlist
    
    def _emit_instanced(
        self,
        bucket: RenderBucket,
        VP: np.ndarray,
        camera_index: int
    ) -> CmdDrawMeshInstanced:
        """Emit an instanced draw command for a bucket."""
        # Build instance buffer key (unique per bucket per frame)
        buffer_key = f"inst_{bucket.key.pipeline_id}_{bucket.key.mesh_id}_{self.snapshot.snapshot_id}"
        
        # Pack instance data
        instance_buffer = bucket.build_instance_buffer()
        
        # Collect entity IDs for picking
        entity_ids = [inst.entity_id for inst in bucket.instances]
        
        return CmdDrawMeshInstanced(
            pipeline_id=bucket.key.pipeline_id,
            mesh_id=bucket.key.mesh_id,
            instance_count=bucket.instance_count,
            instance_buffer_key=buffer_key,
            instance_data=instance_buffer,
            view_proj=VP.astype(np.float32),
            entity_ids=tuple(entity_ids),
            camera_index=camera_index,
        )
    
    @staticmethod
    def _look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
        """Compute view matrix."""
        f = target - eye
        f = f / (np.linalg.norm(f) + 1e-6)
        u = up / (np.linalg.norm(up) + 1e-6)
        s = np.cross(f, u)
        s = s / (np.linalg.norm(s) + 1e-6)
        u2 = np.cross(s, f)
        
        M = np.eye(4, dtype=np.float32)
        M[0, :3] = s
        M[1, :3] = u2
        M[2, :3] = -f
        T = np.eye(4, dtype=np.float32)
        T[:3, 3] = -eye[:3]
        return M @ T
    
    @staticmethod
    def _perspective(fov_y_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
        """Compute perspective projection matrix."""
        f = 1.0 / np.tan(np.deg2rad(fov_y_deg) / 2.0)
        M = np.zeros((4, 4), dtype=np.float32)
        M[0, 0] = f / max(1e-6, aspect)
        M[1, 1] = f
        M[2, 2] = (far + near) / (near - far)
        M[2, 3] = (2 * far * near) / (near - far)
        M[3, 2] = -1.0
        return M


def build_render_commands(
    snapshot: GraphSnapshot,
    use_instancing: bool = True,
    instancing_threshold: int = 4,
) -> Tuple[List[CommandList], List[CameraViewport]]:
    """
    Build command lists from a snapshot.
    
    This is the main entry point for workers.
    Takes an immutable snapshot, returns pure-data command lists.
    
    Args:
        snapshot: Immutable scene snapshot
        use_instancing: Whether to batch draws with instancing
        instancing_threshold: Minimum instances to trigger instancing
    
    Returns:
        Tuple of (list of command lists per camera, camera viewport rects)
    """
    world = RenderWorld(snapshot)
    cmdlists: List[CommandList] = []
    
    for camera, viewport in zip(snapshot.cameras, snapshot.camera_viewports):
        cl = world.emit_commands_for_camera(
            camera=camera,
            viewport=viewport,
            target_h=snapshot.viewport_h,
            use_instancing=use_instancing,
            instancing_threshold=instancing_threshold,
        )
        cmdlists.append(cl)
    
    return cmdlists, list(snapshot.camera_viewports)


# Stats/debug helpers

@dataclass
class RenderWorldStats:
    """Statistics about the render world."""
    bucket_count: int
    total_instances: int
    largest_bucket: int
    unique_pipelines: Set[str]
    unique_meshes: Set[str]


def get_render_world_stats(world: RenderWorld) -> RenderWorldStats:
    """Get statistics about a render world for debugging."""
    pipelines = set()
    meshes = set()
    largest = 0
    
    for key, bucket in world.buckets.items():
        pipelines.add(key.pipeline_id)
        meshes.add(key.mesh_id)
        largest = max(largest, bucket.instance_count)
    
    return RenderWorldStats(
        bucket_count=world.get_bucket_count(),
        total_instances=world.get_instance_count(),
        largest_bucket=largest,
        unique_pipelines=pipelines,
        unique_meshes=meshes,
    )
