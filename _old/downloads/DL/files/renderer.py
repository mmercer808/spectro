"""
Renderer - GL Thread Command Executor

This module executes CommandLists on the GL thread.
It's the ONLY place where GL calls happen.

Key responsibilities:
1. Execute commands from CommandLists
2. Sync GPUBuffers before drawing
3. Create/cache VAOs for instanced rendering
4. Read pick IDs from attachment
"""

from __future__ import annotations
from typing import Any, Dict, Tuple, Optional, List
import numpy as np
import moderngl

from engine.render.commands import (
    CommandList, Command,
    CmdSetViewport, CmdSetScissor, CmdClear,
    CmdDrawMesh, CmdDrawMeshInstanced, CmdDrawPointCloud,
    CmdBeginPass, CmdEndPass,
)
from engine.render.resources import ResourceRegistry
from engine.render.targets import RenderTarget
from engine.render.buffer import GPUBuffer, BufferRegistry, INSTANCE_DTYPE


# =============================================================================
# ID Encoding for Picking
# =============================================================================

def encode_id_rgba(entity_id: int) -> Tuple[float, float, float, float]:
    """Encode entity ID as RGBA color (24-bit in RGB, alpha=1)."""
    eid = int(entity_id) & 0xFFFFFF
    r = (eid >> 16) & 0xFF
    g = (eid >> 8) & 0xFF
    b = (eid >> 0) & 0xFF
    return (r / 255.0, g / 255.0, b / 255.0, 1.0)


def decode_id_rgba(rgba_bytes: bytes) -> int:
    """Decode entity ID from RGBA pixel bytes."""
    if len(rgba_bytes) < 3:
        return 0
    r, g, b = rgba_bytes[0], rgba_bytes[1], rgba_bytes[2]
    return (r << 16) | (g << 8) | b


# =============================================================================
# Renderer
# =============================================================================

class Renderer:
    """
    Executes CommandLists and manages rendering state.
    
    MUST only be used from the GL thread!
    """
    
    def __init__(
        self,
        ctx: moderngl.Context,
        registry: ResourceRegistry,
        buffer_registry: Optional[BufferRegistry] = None,
    ):
        self.ctx = ctx
        self.registry = registry
        self.buffer_registry = buffer_registry or BufferRegistry(ctx)
        
        # VAO cache for instanced rendering: (mesh_id, buffer_name) -> VAO
        self._instanced_vaos: Dict[Tuple[str, str], moderngl.VertexArray] = {}
        
        # Legacy instance buffer cache (for CmdDrawMeshInstanced compatibility)
        self._instance_buffers: Dict[str, moderngl.Buffer] = {}
        
        # Stats
        self._frame_draw_calls = 0
        self._frame_instances = 0
        self._frame_upload_slots = 0
    
    # -------------------------------------------------------------------------
    # Main Entry Point
    # -------------------------------------------------------------------------
    
    def execute_to_target(self, cmdlist: CommandList, target: RenderTarget):
        """
        Execute a CommandList, rendering to the given target.
        
        MUST be called from GL thread only!
        """
        # Reset frame stats
        self._frame_draw_calls = 0
        self._frame_instances = 0
        self._frame_upload_slots = 0
        
        # Sync all GPU buffers before rendering
        self._frame_upload_slots = self.buffer_registry.sync_all()
        
        # Bind target
        target.fbo.use()
        self.ctx.viewport = (0, 0, target.spec.w, target.spec.h)
        self.ctx.scissor = None
        
        # Process commands
        for cmd in cmdlist.commands:
            self._execute_command(cmd, target)
    
    def _execute_command(self, cmd: Command, target: RenderTarget):
        """Execute a single command."""
        
        if isinstance(cmd, CmdSetViewport):
            self.ctx.viewport = (cmd.x, cmd.y, cmd.w, cmd.h)
        
        elif isinstance(cmd, CmdSetScissor):
            if cmd.enabled:
                self.ctx.scissor = (cmd.x, cmd.y, cmd.w, cmd.h)
            else:
                self.ctx.scissor = None
        
        elif isinstance(cmd, CmdClear):
            target.fbo.clear(cmd.r, cmd.g, cmd.b, cmd.a, depth=cmd.depth)
        
        elif isinstance(cmd, CmdDrawMesh):
            self._draw_mesh(cmd, target)
        
        elif isinstance(cmd, CmdDrawMeshInstanced):
            self._draw_mesh_instanced_legacy(cmd, target)
        
        elif isinstance(cmd, CmdDrawPointCloud):
            self._draw_point_cloud(cmd, target)
        
        elif isinstance(cmd, CmdBeginPass):
            pass  # For debugging/organization
        
        elif isinstance(cmd, CmdEndPass):
            pass
    
    # -------------------------------------------------------------------------
    # Draw: Single Mesh
    # -------------------------------------------------------------------------
    
    def _draw_mesh(self, cmd: CmdDrawMesh, target: RenderTarget):
        """Execute a single-instance mesh draw."""
        mesh = self.registry.get_mesh(cmd.mesh_id)
        pipeline = self.registry.get_pipeline(cmd.pipeline_id)
        prog = pipeline.program
        
        # Get VAO for this mesh/pipeline (no instancing)
        vao = self.registry.get_or_create_vao(cmd.mesh_id, cmd.pipeline_id)
        
        target.fbo.use()
        
        # Set entity ID for picking
        if cmd.entity_id is not None:
            try:
                prog["u_id_rgba"].value = encode_id_rgba(cmd.entity_id)
            except KeyError:
                pass
        
        # Set uniforms
        for key, val in cmd.uniforms.items():
            try:
                if isinstance(val, np.ndarray):
                    prog[key].write(val.astype(np.float32).tobytes())
                else:
                    prog[key].value = val
            except KeyError:
                pass
        
        vao.render(mode=moderngl.TRIANGLES)
        
        self._frame_draw_calls += 1
        self._frame_instances += 1
    
    # -------------------------------------------------------------------------
    # Draw: Point Cloud (NEW - uses persistent GPUBuffer)
    # -------------------------------------------------------------------------
    
    def _draw_point_cloud(self, cmd: CmdDrawPointCloud, target: RenderTarget):
        """
        Draw instances from a persistent GPUBuffer.
        
        This is the new efficient path:
        - Buffer already synced at frame start
        - VAO cached and reused
        - No data copying in the command
        """
        # Get the GPU buffer
        if not self.buffer_registry.has(cmd.buffer_key):
            print(f"[Renderer] Buffer '{cmd.buffer_key}' not found")
            return
        
        gpu_buffer = self.buffer_registry.get(cmd.buffer_key)
        
        # Get or create VAO for this mesh + buffer
        vao = self._get_or_create_instanced_vao(cmd.mesh_id, gpu_buffer)
        
        # Get pipeline and program
        pipeline = self.registry.get_pipeline(cmd.pipeline_id)
        prog = pipeline.program
        
        target.fbo.use()
        
        # Set uniforms
        try:
            prog["u_view"].write(cmd.view_matrix.astype(np.float32).tobytes())
        except KeyError:
            pass
        
        try:
            prog["u_proj"].write(cmd.proj_matrix.astype(np.float32).tobytes())
        except KeyError:
            pass
        
        try:
            prog["u_color_mode"].value = 0  # Default to solid color
        except KeyError:
            pass
        
        # Draw instanced
        # Note: moderngl doesn't support first_instance directly,
        # so we draw all instances and use flags for visibility
        vao.render(
            mode=moderngl.TRIANGLES,
            first=0,
            instances=cmd.start + cmd.count,  # Draw up to last needed instance
        )
        
        self._frame_draw_calls += 1
        self._frame_instances += cmd.count
    
    def _get_or_create_instanced_vao(
        self,
        mesh_id: str,
        gpu_buffer: GPUBuffer,
    ) -> moderngl.VertexArray:
        """Get or create a VAO for instanced rendering with a GPUBuffer."""
        key = (mesh_id, gpu_buffer.name)
        
        if key in self._instanced_vaos:
            return self._instanced_vaos[key]
        
        mesh = self.registry.get_mesh(mesh_id)
        pipeline = self.registry.get_pipeline("instanced")
        prog = pipeline.program
        
        # Build VAO with mesh vertices + instance attributes
        # Format for INSTANCE_DTYPE: '3f 3f 4f 4f 1f 1u'
        instance_format = gpu_buffer.get_instance_format()
        instance_attribs = gpu_buffer.get_attribute_names()
        
        content = [
            # Mesh vertex data
            (mesh.vbo, '3f 3f', 'in_pos', 'in_nrm'),
            # Instance data (per-instance)
            (gpu_buffer.gpu, instance_format + ' /i', *instance_attribs),
        ]
        
        vao = self.ctx.vertex_array(
            prog,
            content,
            index_buffer=mesh.ibo,
        )
        
        self._instanced_vaos[key] = vao
        return vao
    
    # -------------------------------------------------------------------------
    # Draw: Legacy Instanced (for compatibility)
    # -------------------------------------------------------------------------
    
    def _draw_mesh_instanced_legacy(self, cmd: CmdDrawMeshInstanced, target: RenderTarget):
        """
        Execute an instanced draw using the old method (data in command).
        
        This is the legacy path kept for compatibility.
        New code should use CmdDrawPointCloud.
        """
        mesh = self.registry.get_mesh(cmd.mesh_id)
        
        # Get or create instance buffer
        instance_buf = self._get_or_create_legacy_instance_buffer(
            cmd.instance_buffer_key,
            cmd.instance_data,
        )
        
        # This path uses the old instanced shader which expects mat4 + color
        # For now, fall back to non-instanced
        # TODO: Create proper VAO with legacy layout
        
        self._frame_draw_calls += 1
        self._frame_instances += cmd.instance_count
    
    def _get_or_create_legacy_instance_buffer(
        self,
        key: str,
        data: np.ndarray
    ) -> moderngl.Buffer:
        """Legacy instance buffer management."""
        if key in self._instance_buffers:
            buf = self._instance_buffers[key]
            if buf.size < data.nbytes:
                buf.release()
                buf = self.ctx.buffer(data.tobytes())
                self._instance_buffers[key] = buf
            else:
                buf.write(data.tobytes())
        else:
            buf = self.ctx.buffer(data.tobytes())
            self._instance_buffers[key] = buf
        
        return buf
    
    # -------------------------------------------------------------------------
    # Picking
    # -------------------------------------------------------------------------
    
    def read_pick_id(self, target: RenderTarget, x: int, y: int) -> int:
        """Read entity ID from pick attachment at (x, y)."""
        if target.pick is None:
            return 0
        
        x = max(0, min(int(x), target.spec.w - 1))
        y = max(0, min(int(y), target.spec.h - 1))
        
        gl_y = target.spec.h - 1 - y
        
        data = target.fbo.read(
            viewport=(x, gl_y, 1, 1),
            components=4,
            attachment=1,
        )
        
        return decode_id_rgba(data)
    
    # -------------------------------------------------------------------------
    # Compositing
    # -------------------------------------------------------------------------
    
    def composite_panels(
        self,
        panels: Dict[str, Any],
        areas: List,
        window_size: Tuple[int, int]
    ):
        """Composite viewport textures to screen."""
        prog = self.registry._blit_prog
        vao = self.registry._blit_vao
        
        if prog is None or vao is None:
            return
        
        prog["u_window"].value = window_size
        
        for area in areas:
            rect = panels.get(area.area_id)
            if not rect or not area.surface_rt:
                continue
            
            area.surface_rt.color.use(location=0)
            prog["u_tex"].value = 0
            prog["u_rect"].value = (rect.x, rect.y, rect.w, rect.h)
            vao.render(moderngl.TRIANGLES)
    
    # -------------------------------------------------------------------------
    # Stats / Debug
    # -------------------------------------------------------------------------
    
    def get_frame_stats(self) -> Dict[str, int]:
        """Get stats for the current frame."""
        return {
            "draw_calls": self._frame_draw_calls,
            "instances": self._frame_instances,
            "buffer_slots_uploaded": self._frame_upload_slots,
        }
    
    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------
    
    def cleanup(self):
        """Release resources."""
        for vao in self._instanced_vaos.values():
            vao.release()
        self._instanced_vaos.clear()
        
        for buf in self._instance_buffers.values():
            buf.release()
        self._instance_buffers.clear()
