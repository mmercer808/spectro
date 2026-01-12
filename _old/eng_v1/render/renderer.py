from __future__ import annotations
from typing import Any, Dict, Tuple
import numpy as np
import moderngl

from engine.render.commands import (
    CommandList, CmdSetViewport, CmdSetScissor, CmdClear, CmdDrawMesh
)
from engine.render.resources import ResourceRegistry
from engine.render.targets import RenderTarget
from engine.viewport.viewport import ViewportArea

def encode_id_rgba(entity_id: int) -> tuple[float,float,float,float]:
    eid = int(entity_id) & 0xFFFFFF
    r = (eid >> 16) & 0xFF
    g = (eid >> 8) & 0xFF
    b = (eid >> 0) & 0xFF
    return (r/255.0, g/255.0, b/255.0, 1.0)

def decode_id_rgba(rgba_bytes: bytes) -> int:
    if len(rgba_bytes) < 3:
        return 0
    r, g, b = rgba_bytes[0], rgba_bytes[1], rgba_bytes[2]
    return (r << 16) | (g << 8) | b

class Renderer:
    def __init__(self, ctx: moderngl.Context, registry: ResourceRegistry):
        self.ctx = ctx
        self.registry = registry

    def execute_to_target(self, cmdlist: CommandList, target: RenderTarget):
        target.fbo.use()
        self.ctx.viewport = (0, 0, target.spec.w, target.spec.h)
        self.ctx.scissor = None

        for cmd in cmdlist.commands:
            if isinstance(cmd, CmdSetViewport):
                self.ctx.viewport = (cmd.x, cmd.y, cmd.w, cmd.h)
            elif isinstance(cmd, CmdSetScissor):
                self.ctx.scissor = (cmd.x, cmd.y, cmd.w, cmd.h) if cmd.enabled else None
            elif isinstance(cmd, CmdClear):
                self.ctx.clear(cmd.r, cmd.g, cmd.b, cmd.a, depth=cmd.depth)
                if cmd.clear_pick and target.pick is not None:
                    target.fbo.clear(attachments=[1], color=(0.0, 0.0, 0.0, 0.0))
            elif isinstance(cmd, CmdDrawMesh):
                mesh = self.registry.get_mesh(cmd.mesh_id)

                if cmd.pipeline_id == "pick_id":
                    prog = self.registry.get_pipeline("pick_id").program
                    vao = mesh.vao_pick
                    if cmd.entity_id:
                        prog["u_id_rgba"].value = encode_id_rgba(cmd.entity_id)
                else:
                    prog = self.registry.get_pipeline(cmd.pipeline_id).program
                    vao = mesh.vao_lit

                for k, v in cmd.uniforms.items():
                    if isinstance(v, np.ndarray):
                        prog[k].write(v.astype(np.float32).tobytes())
                    else:
                        prog[k].value = v

                vao.render(mode=moderngl.TRIANGLES)

    def composite_panels(self, panels: Dict[str, Any], areas: list[ViewportArea], window_size: Tuple[int,int]):
        prog = self.registry._blit_prog
        vao = self.registry._blit_vao
        assert prog is not None and vao is not None
        prog["u_window"].value = window_size

        for area in areas:
            rect = panels[area.area_id]
            if not area.surface_rt:
                continue
            area.surface_rt.color.use(location=0)
            prog["u_tex"].value = 0
            prog["u_rect"].value = (rect.x, rect.y, rect.w, rect.h)
            vao.render(moderngl.TRIANGLES)

    def read_pick_id(self, target: RenderTarget, x: int, y: int) -> int:
        if target.pick is None:
            return 0
        x = max(0, min(int(x), target.spec.w - 1))
        y = max(0, min(int(y), target.spec.h - 1))
        by = target.spec.h - 1 - y
        data = target.fbo.read(viewport=(x, by, 1, 1), components=4, attachment=1)
        return decode_id_rgba(data)
