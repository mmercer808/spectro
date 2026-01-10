"""
Renderer - GL Thread Command Executor

This module executes CommandLists on the GL thread.
It's the ONLY place where GL calls happen.

Key responsibilities:
1. Execute commands from CommandLists
2. Manage instance buffers for batched draws
3. Read pick IDs from attachment
4. Drain upload queue with budget
"""

from __future__ import annotations
from typing import Any, Dict, Tuple, Optional, List
import numpy as np
import moderngl

from engine.render.commands import (
    CommandList, Command,
    CmdSetViewport, CmdSetScissor, CmdClear, 
    CmdDrawMesh, CmdDrawMeshInstanced,
    CmdBeginPass, CmdEndPass,
)
from engine.render.resources import ResourceRegistry
from engine.render.targets import RenderTarget
from engine.render.uploader import UploadQueue, UploadJob, UploadKind, DrainResult


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
        upload_queue: Optional[UploadQueue] = None
    ):
        self.ctx = ctx
        self.registry = registry
        self.upload_queue = upload_queue or UploadQueue()
        
        # Instance buffer cache (key -> buffer)
        self._instance_buffers: Dict[str, moderngl.Buffer] = {}
        
        # Stats for this frame
        self._frame_draw_calls = 0
        self._frame_instances = 0
        self._frame_upload_bytes = 0
    
    # -------------------------------------------------------------------------
    # Command Execution
    # -------------------------------------------------------------------------
    
    def execute_to_target(self, cmdlist: CommandList, target: RenderTarget):
        """
        Execute a CommandList, rendering to the given target.
        
        MUST be called from GL thread only!
        """
        # Validate commands don't contain GL objects (debug build)
        errors = cmdlist.validate()
        if errors:
            for err in errors[:5]:  # Limit spam
                print(f"[Renderer] CommandList validation error: {err}")
        
        # Reset frame stats
        self._frame_draw_calls = 0
        self._frame_instances = 0
        
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
            # Use fbo.clear() to clear all attachments (color + pick)
            target.fbo.clear(cmd.r, cmd.g, cmd.b, cmd.a, depth=cmd.depth)
        
        elif isinstance(cmd, CmdDrawMesh):
            self._draw_mesh(cmd, target)
        
        elif isinstance(cmd, CmdDrawMeshInstanced):
            self._draw_mesh_instanced(cmd, target)
        
        elif isinstance(cmd, CmdBeginPass):
            # Currently just for debugging/organization
            pass
        
        elif isinstance(cmd, CmdEndPass):
            pass
    
    def _draw_mesh(self, cmd: CmdDrawMesh, target: RenderTarget):
        """Execute a single-instance mesh draw."""
        mesh = self.registry.get_mesh(cmd.mesh_id)
        pipeline = self.registry.get_pipeline(cmd.pipeline_id)
        prog = pipeline.program
        
        # Ensure we're rendering to all attachments
        target.fbo.use()
        
        # Set entity ID for picking (if shader supports it)
        if cmd.entity_id is not None:
            try:
                prog["u_id_rgba"].value = encode_id_rgba(cmd.entity_id)
            except KeyError:
                pass  # Shader doesn't have picking support
        
        # Set uniforms
        for key, val in cmd.uniforms.items():
            try:
                if isinstance(val, np.ndarray):
                    prog[key].write(val.astype(np.float32).tobytes())
                else:
                    prog[key].value = val
            except KeyError:
                pass  # Uniform not in shader
        
        # Draw
        mesh.vao_lit.render(mode=moderngl.TRIANGLES)
        
        self._frame_draw_calls += 1
        self._frame_instances += 1
    
    def _draw_mesh_instanced(self, cmd: CmdDrawMeshInstanced, target: RenderTarget):
        """Execute an instanced mesh draw."""
        mesh = self.registry.get_mesh(cmd.mesh_id)
        pipeline = self.registry.get_pipeline(cmd.pipeline_id + "_instanced")
        
        # Get or create instance buffer
        instance_buffer = self._get_or_create_instance_buffer(
            cmd.instance_buffer_key,
            cmd.instance_data,
        )
        
        # Create instanced VAO (or get cached)
        vao = self._create_instanced_vao(
            pipeline.program,
            mesh,
            instance_buffer,
            cmd.instance_buffer_key,
        )
        
        # Set view-projection uniform
        prog = pipeline.program
        try:
            prog["u_view_proj"].write(cmd.view_proj.tobytes())
        except KeyError:
            pass
        
        # Draw instanced
        target.fbo.use()
        vao.render(mode=moderngl.TRIANGLES, instances=cmd.instance_count)
        
        self._frame_draw_calls += 1
        self._frame_instances += cmd.instance_count
    
    def _get_or_create_instance_buffer(
        self, 
        key: str, 
        data: np.ndarray
    ) -> moderngl.Buffer:
        """Get or create an instance buffer with the given data."""
        # For now, just create/update every frame
        # Future: Use upload queue with versioning
        
        if key in self._instance_buffers:
            buf = self._instance_buffers[key]
            # Resize if needed
            if buf.size < data.nbytes:
                buf.release()
                buf = self.ctx.buffer(data.tobytes())
                self._instance_buffers[key] = buf
            else:
                buf.write(data.tobytes())
        else:
            buf = self.ctx.buffer(data.tobytes())
            self._instance_buffers[key] = buf
        
        self._frame_upload_bytes += data.nbytes
        return buf
    
    def _create_instanced_vao(
        self,
        prog: moderngl.Program,
        mesh,
        instance_buffer: moderngl.Buffer,
        key: str,
    ) -> moderngl.VertexArray:
        """Create a VAO with per-instance attributes."""
        # This is a simplified version - full implementation would cache VAOs
        # and handle attribute layout more robustly
        
        # For now, fall back to non-instanced path
        # TODO: Implement proper instanced VAO creation
        return mesh.vao_lit
    
    # -------------------------------------------------------------------------
    # Upload Queue Processing
    # -------------------------------------------------------------------------
    
    def drain_uploads(self, byte_budget: int = 1024 * 1024) -> DrainResult:
        """
        Process pending uploads up to byte budget.
        
        MUST be called from GL thread!
        """
        def processor(job: UploadJob) -> bool:
            return self._process_upload(job)
        
        return self.upload_queue.drain(byte_budget, processor)
    
    def _process_upload(self, job: UploadJob) -> bool:
        """Process a single upload job. Returns True on success."""
        try:
            if job.kind == UploadKind.INSTANCE_BUFFER:
                buf = self.ctx.buffer(job.payload)
                self._instance_buffers[job.resource_key] = buf
            
            elif job.kind == UploadKind.MESH_VERTICES:
                # Would update mesh VBO
                pass
            
            elif job.kind == UploadKind.TEXTURE_2D:
                # Would update texture
                pass
            
            return True
            
        except Exception as e:
            print(f"[Renderer] Upload failed for {job.resource_key}: {e}")
            return False
    
    # -------------------------------------------------------------------------
    # Picking
    # -------------------------------------------------------------------------
    
    def read_pick_id(self, target: RenderTarget, x: int, y: int) -> int:
        """
        Read entity ID from pick attachment at (x, y).
        
        Coordinates are in render target space (top-left origin in our system,
        but we flip for GL's bottom-left).
        """
        if target.pick is None:
            return 0
        
        # Clamp coordinates
        x = max(0, min(int(x), target.spec.w - 1))
        y = max(0, min(int(y), target.spec.h - 1))
        
        # Flip Y for GL (our Y is top-down, GL is bottom-up)
        gl_y = target.spec.h - 1 - y
        
        # Read pixel from pick attachment
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
        """
        Composite viewport textures to screen.
        
        This draws each viewport's render target texture to its panel rect.
        """
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
    # Stats/Debug
    # -------------------------------------------------------------------------
    
    def get_frame_stats(self) -> Dict[str, int]:
        """Get stats for the current frame."""
        return {
            "draw_calls": self._frame_draw_calls,
            "instances": self._frame_instances,
            "upload_bytes": self._frame_upload_bytes,
        }
    
    def cleanup(self):
        """Release resources."""
        for buf in self._instance_buffers.values():
            buf.release()
        self._instance_buffers.clear()
