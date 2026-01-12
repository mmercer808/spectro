from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import moderngl

@dataclass
class RenderTargetSpec:
    w: int
    h: int
    color_format: str = "rgba8"
    depth: bool = True
    samples: int = 0
    picking: bool = False

@dataclass
class RenderTarget:
    spec: RenderTargetSpec
    fbo: moderngl.Framebuffer
    color: moderngl.Texture
    depth: Optional[moderngl.Renderbuffer] = None
    pick: Optional[moderngl.Texture] = None

class RenderTargetPool:
    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        self._free: Dict[Tuple, list[RenderTarget]] = {}

    def _key(self, spec: RenderTargetSpec) -> Tuple:
        return (spec.w, spec.h, spec.depth, spec.samples, spec.picking)

    def acquire(self, spec: RenderTargetSpec) -> RenderTarget:
        key = self._key(spec)
        bucket = self._free.get(key)
        if bucket:
            return bucket.pop()

        color = self.ctx.texture((spec.w, spec.h), components=4, dtype="f1")
        color.filter = (moderngl.LINEAR, moderngl.LINEAR)
        attachments = [color]

        pick = None
        if spec.picking:
            pick = self.ctx.texture((spec.w, spec.h), components=4, dtype="f1")
            pick.filter = (moderngl.NEAREST, moderngl.NEAREST)
            attachments.append(pick)

        depth = self.ctx.depth_renderbuffer((spec.w, spec.h)) if spec.depth else None
        fbo = self.ctx.framebuffer(color_attachments=attachments, depth_attachment=depth)
        return RenderTarget(spec=spec, fbo=fbo, color=color, depth=depth, pick=pick)

    def release(self, rt: RenderTarget):
        self._free.setdefault(self._key(rt.spec), []).append(rt)
