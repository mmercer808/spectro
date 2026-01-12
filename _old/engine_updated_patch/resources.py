from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
import moderngl

@dataclass
class MeshGPU:
    vao_lit: moderngl.VertexArray
    vao_pick: moderngl.VertexArray
    vbo: moderngl.Buffer
    ibo: moderngl.Buffer
    index_count: int

@dataclass
class PipelineGPU:
    program: moderngl.Program

class ResourceRegistry:
    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        self.meshes: Dict[str, MeshGPU] = {}
        self.pipelines: Dict[str, PipelineGPU] = {}

        # compositor
        self._blit_prog: Optional[moderngl.Program] = None
        self._blit_vao: Optional[moderngl.VertexArray] = None

    def bootstrap_defaults(self):
        self._create_pipelines()
        self._create_meshes()
        self._create_blitter()

    def _create_pipelines(self):
        # lit
        vs = """
        #version 330
        in vec3 in_pos;
        in vec3 in_nrm;
        uniform mat4 u_mvp;
        void main() { gl_Position = u_mvp * vec4(in_pos, 1.0); }
        """
        fs = """
        #version 330
        uniform vec4 u_color;
        uniform vec4 u_id_rgba;
        layout(location=0) out vec4 fragColor;
        layout(location=1) out vec4 pickOut;
        void main() {
            fragColor = u_color;
            pickOut = u_id_rgba;
        }
        """
        self.pipelines["lit_color"] = PipelineGPU(program=self.ctx.program(vertex_shader=vs, fragment_shader=fs))

        # pick: write to attachment 1
        pfs = """
        #version 330
        uniform vec4 u_id_rgba;
        layout(location=0) out vec4 fragColor;
        layout(location=1) out vec4 pickOut;
        void main() {
            fragColor = vec4(0.0);
            pickOut = u_id_rgba;
        }
        """
        self.pipelines["pick_id"] = PipelineGPU(program=self.ctx.program(
            vertex_shader=vs,
            fragment_shader=pfs
        ))

    def _create_meshes(self):
        positions, normals, indices = [], [], []
        def add_face(p0,p1,p2,p3,n):
            base = len(positions)
            positions.extend([p0,p1,p2,p3])
            normals.extend([n,n,n,n])
            indices.extend([base+0, base+1, base+2, base+0, base+2, base+3])
        s = 0.5
        add_face((-s,-s, s),( s,-s, s),( s, s, s),(-s, s, s),(0,0,1))
        add_face(( s,-s,-s),(-s,-s,-s),(-s, s,-s),( s, s,-s),(0,0,-1))
        add_face(( s,-s, s),( s,-s,-s),( s, s,-s),( s, s, s),(1,0,0))
        add_face((-s,-s,-s),(-s,-s, s),(-s, s, s),(-s, s,-s),(-1,0,0))
        add_face((-s, s, s),( s, s, s),( s, s,-s),(-s, s,-s),(0,1,0))
        add_face((-s,-s,-s),( s,-s,-s),( s,-s, s),(-s,-s, s),(0,-1,0))

        v = np.array(positions, dtype=np.float32)
        n = np.array(normals, dtype=np.float32)
        interleaved = np.hstack([v, n]).astype(np.float32)
        idx = np.array(indices, dtype=np.uint32)

        vbo = self.ctx.buffer(interleaved.tobytes())
        ibo = self.ctx.buffer(idx.tobytes())

        lit_prog = self.pipelines["lit_color"].program
        pick_prog = self.pipelines["pick_id"].program

        vao_lit = self.ctx.vertex_array(lit_prog, [(vbo, "3f 3f", "in_pos", "in_nrm")], index_buffer=ibo)
        vao_pick = self.ctx.vertex_array(pick_prog, [(vbo, "3f 3f", "in_pos", "in_nrm")], index_buffer=ibo)

        self.meshes["cube"] = MeshGPU(vao_lit=vao_lit, vao_pick=vao_pick, vbo=vbo, ibo=ibo, index_count=len(indices))

    def _create_blitter(self):
        vs = """
        #version 330
        in vec2 in_pos;
        in vec2 in_uv;
        out vec2 v_uv;
        uniform vec2 u_window;
        uniform vec4 u_rect; // x,y,w,h in pixels (top-left origin)
        void main() {
            vec2 p = in_pos;
            vec2 px = vec2(u_rect.x + p.x * u_rect.z, u_rect.y + p.y * u_rect.w);
            vec2 ndc = vec2((px.x / u_window.x) * 2.0 - 1.0, 1.0 - (px.y / u_window.y) * 2.0);
            gl_Position = vec4(ndc, 0.0, 1.0);
            v_uv = in_uv;
        }
        """
        fs = """
        #version 330
        in vec2 v_uv;
        uniform sampler2D u_tex;
        out vec4 fragColor;
        void main() { fragColor = texture(u_tex, v_uv); }
        """
        self._blit_prog = self.ctx.program(vertex_shader=vs, fragment_shader=fs)

        quad = np.array([
            0,0, 0,0,
            1,0, 1,0,
            1,1, 1,1,
            0,0, 0,0,
            1,1, 1,1,
            0,1, 0,1,
        ], dtype=np.float32)
        vbo = self.ctx.buffer(quad.tobytes())
        self._blit_vao = self.ctx.vertex_array(self._blit_prog, [(vbo, "2f 2f", "in_pos", "in_uv")])

    def get_mesh(self, mesh_id: str) -> MeshGPU:
        return self.meshes[mesh_id]

    def get_pipeline(self, pipeline_id: str) -> PipelineGPU:
        return self.pipelines[pipeline_id]
