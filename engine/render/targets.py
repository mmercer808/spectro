"""
Render Targets

Manages render-to-texture targets with optional picking attachment.
Uses a pool to avoid allocation thrashing during resize.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List
import moderngl


@dataclass
class RenderTargetSpec:
    """
    Specification for a render target.
    Used as a key for pooling.
    """
    w: int
    h: int
    color_format: str = "rgba8"
    depth: bool = True
    samples: int = 0  # MSAA samples (0 = no MSAA)
    picking: bool = False  # Include pick ID attachment


@dataclass
class RenderTarget:
    """
    A render target with color, optional depth, and optional pick attachment.
    """
    spec: RenderTargetSpec
    fbo: moderngl.Framebuffer
    color: moderngl.Texture
    depth: Optional[moderngl.Renderbuffer] = None
    pick: Optional[moderngl.Texture] = None
    
    def release(self, ctx: moderngl.Context):
        """Release all GL resources."""
        self.fbo.release()
        self.color.release()
        if self.depth:
            self.depth.release()
        if self.pick:
            self.pick.release()


class RenderTargetPool:
    """
    Pool of render targets for reuse.
    
    Avoids allocation thrashing during window resize.
    Targets are keyed by their spec (size, format, etc.).
    """
    
    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        self._free: Dict[Tuple, List[RenderTarget]] = {}
        self._allocated: List[RenderTarget] = []
    
    def _make_key(self, spec: RenderTargetSpec) -> Tuple:
        """Create a hashable key from spec."""
        return (spec.w, spec.h, spec.depth, spec.samples, spec.picking)
    
    def acquire(self, spec: RenderTargetSpec) -> RenderTarget:
        """
        Get a render target matching the spec.
        May return a pooled target or create a new one.
        """
        key = self._make_key(spec)
        bucket = self._free.get(key)
        
        if bucket:
            rt = bucket.pop()
            return rt
        
        # Create new target
        rt = self._create_target(spec)
        self._allocated.append(rt)
        return rt
    
    def release(self, rt: RenderTarget):
        """Return a render target to the pool for reuse."""
        key = self._make_key(rt.spec)
        self._free.setdefault(key, []).append(rt)
    
    def _create_target(self, spec: RenderTargetSpec) -> RenderTarget:
        """Create a new render target."""
        # Color attachment
        color = self.ctx.texture(
            (spec.w, spec.h),
            components=4,
            dtype="f1",  # 8-bit per channel
        )
        color.filter = (moderngl.LINEAR, moderngl.LINEAR)
        
        attachments = [color]
        
        # Pick attachment (for entity ID readback)
        pick = None
        if spec.picking:
            pick = self.ctx.texture(
                (spec.w, spec.h),
                components=4,
                dtype="f1",
            )
            pick.filter = (moderngl.NEAREST, moderngl.NEAREST)
            attachments.append(pick)
        
        # Depth attachment
        depth = None
        if spec.depth:
            depth = self.ctx.depth_renderbuffer((spec.w, spec.h))
        
        # Framebuffer
        fbo = self.ctx.framebuffer(
            color_attachments=attachments,
            depth_attachment=depth,
        )
        
        return RenderTarget(
            spec=spec,
            fbo=fbo,
            color=color,
            depth=depth,
            pick=pick,
        )
    
    def cleanup(self):
        """Release all pooled targets."""
        for bucket in self._free.values():
            for rt in bucket:
                rt.release(self.ctx)
        self._free.clear()
    
    def get_stats(self) -> Dict[str, int]:
        """Get pool statistics."""
        total_free = sum(len(b) for b in self._free.values())
        return {
            "total_allocated": len(self._allocated),
            "total_free": total_free,
            "bucket_count": len(self._free),
        }
