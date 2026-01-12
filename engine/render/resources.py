"""
Resource Registry

Central registry for GPU resources (meshes, pipelines, textures).
Separates resource management from rendering.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple
import numpy as np
import moderngl


# =============================================================================
# GPU Resource Types
# =============================================================================

@dataclass
class MeshGPU:
    """GPU-side mesh data."""
    vao_lit: moderngl.VertexArray
    vao_pick: moderngl.VertexArray
    vbo: moderngl.Buffer
    ibo: moderngl.Buffer
    index_count: int
    vertex_count: int = 0
    
    def release(self):
        self.vao_lit.release()
        self.vao_pick.release()
        self.vbo.release()
        self.ibo.release()


@dataclass
class PipelineGPU:
    """GPU-side shader pipeline."""
    program: moderngl.Program
    
    def release(self):
        self.program.release()


# =============================================================================
# Resource Registry
# =============================================================================

class ResourceRegistry:
    """
    Central registry for all GPU resources.
    
    Resources are identified by string keys.
    The registry owns the GPU objects and handles cleanup.
    """
    
    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        self.meshes: Dict[str, MeshGPU] = {}
        self.pipelines: Dict[str, PipelineGPU] = {}
        self.textures: Dict[str, moderngl.Texture] = {}
        
        # Compositor resources
        self._blit_prog: Optional[moderngl.Program] = None
        self._blit_vao: Optional[moderngl.VertexArray] = None
        self._blit_vbo: Optional[moderngl.Buffer] = None
    
    def bootstrap_defaults(self):
        """Create default resources (pipelines, meshes, blitter)."""
        self._create_default_pipelines()
        self._create_default_meshes()
        self._create_blitter()
    
    # -------------------------------------------------------------------------
    # Pipelines
    # -------------------------------------------------------------------------
    
    def _create_default_pipelines(self):
        """Create built-in shader pipelines."""
        
        # Lit color shader with MRT picking output
        lit_vs = """
        #version 330
        
        in vec3 in_pos;
        in vec3 in_nrm;
        
        uniform mat4 u_mvp;
        
        out vec3 v_normal;
        
        void main() {
            gl_Position = u_mvp * vec4(in_pos, 1.0);
            v_normal = in_nrm;
        }
        """
        
        lit_fs = """
        #version 330
        
        in vec3 v_normal;
        
        uniform vec4 u_color;
        uniform vec4 u_id_rgba;
        
        layout(location = 0) out vec4 fragColor;
        layout(location = 1) out vec4 pickOut;
        
        void main() {
            // Simple diffuse lighting
            vec3 light_dir = normalize(vec3(0.5, 1.0, 0.3));
            float ndotl = max(dot(normalize(v_normal), light_dir), 0.0);
            float ambient = 0.3;
            float diffuse = 0.7 * ndotl;
            
            fragColor = vec4(u_color.rgb * (ambient + diffuse), u_color.a);
            pickOut = u_id_rgba;
        }
        """
        
        self.pipelines["lit_color"] = PipelineGPU(
            program=self.ctx.program(vertex_shader=lit_vs, fragment_shader=lit_fs)
        )
        
        # Pick-only shader (writes only to pick attachment)
        pick_fs = """
        #version 330
        
        uniform vec4 u_id_rgba;
        
        layout(location = 0) out vec4 fragColor;
        layout(location = 1) out vec4 pickOut;
        
        void main() {
            fragColor = vec4(0.0);
            pickOut = u_id_rgba;
        }
        """
        
        self.pipelines["pick_id"] = PipelineGPU(
            program=self.ctx.program(vertex_shader=lit_vs, fragment_shader=pick_fs)
        )
        
        # Instanced lit color shader
        instanced_vs = """
        #version 330
        
        in vec3 in_pos;
        in vec3 in_nrm;
        
        // Per-instance attributes (from instance buffer)
        in mat4 in_world;  // Instance world matrix
        in vec4 in_inst_color;  // Instance color
        
        uniform mat4 u_view_proj;
        
        out vec3 v_normal;
        out vec4 v_color;
        
        void main() {
            vec4 world_pos = in_world * vec4(in_pos, 1.0);
            gl_Position = u_view_proj * world_pos;
            
            // Transform normal (assuming uniform scale)
            mat3 normal_mat = mat3(in_world);
            v_normal = normal_mat * in_nrm;
            
            v_color = in_inst_color;
        }
        """
        
        instanced_fs = """
        #version 330
        
        in vec3 v_normal;
        in vec4 v_color;
        
        uniform vec4 u_id_rgba;  // For picking (same for all instances in batch)
        
        layout(location = 0) out vec4 fragColor;
        layout(location = 1) out vec4 pickOut;
        
        void main() {
            // Simple diffuse lighting
            vec3 light_dir = normalize(vec3(0.5, 1.0, 0.3));
            float ndotl = max(dot(normalize(v_normal), light_dir), 0.0);
            float ambient = 0.3;
            float diffuse = 0.7 * ndotl;
            
            fragColor = vec4(v_color.rgb * (ambient + diffuse), v_color.a);
            pickOut = u_id_rgba;  // TODO: Per-instance ID
        }
        """
        
        self.pipelines["lit_color_instanced"] = PipelineGPU(
            program=self.ctx.program(vertex_shader=instanced_vs, fragment_shader=instanced_fs)
        )
    
    # -------------------------------------------------------------------------
    # Meshes
    # -------------------------------------------------------------------------
    
    def _create_default_meshes(self):
        """Create built-in meshes."""
        self._create_cube_mesh()
    
    def _create_cube_mesh(self):
        """Create a unit cube mesh."""
        positions = []
        normals = []
        indices = []
        
        def add_face(p0, p1, p2, p3, n):
            base = len(positions)
            positions.extend([p0, p1, p2, p3])
            normals.extend([n, n, n, n])
            indices.extend([base + 0, base + 1, base + 2, base + 0, base + 2, base + 3])
        
        s = 0.5  # Half-size
        
        # Front (+Z)
        add_face((-s, -s, s), (s, -s, s), (s, s, s), (-s, s, s), (0, 0, 1))
        # Back (-Z)
        add_face((s, -s, -s), (-s, -s, -s), (-s, s, -s), (s, s, -s), (0, 0, -1))
        # Right (+X)
        add_face((s, -s, s), (s, -s, -s), (s, s, -s), (s, s, s), (1, 0, 0))
        # Left (-X)
        add_face((-s, -s, -s), (-s, -s, s), (-s, s, s), (-s, s, -s), (-1, 0, 0))
        # Top (+Y)
        add_face((-s, s, s), (s, s, s), (s, s, -s), (-s, s, -s), (0, 1, 0))
        # Bottom (-Y)
        add_face((-s, -s, -s), (s, -s, -s), (s, -s, s), (-s, -s, s), (0, -1, 0))
        
        # Interleave position and normal
        v = np.array(positions, dtype=np.float32)
        n = np.array(normals, dtype=np.float32)
        interleaved = np.hstack([v, n]).astype(np.float32)
        idx = np.array(indices, dtype=np.uint32)
        
        vbo = self.ctx.buffer(interleaved.tobytes())
        ibo = self.ctx.buffer(idx.tobytes())
        
        lit_prog = self.pipelines["lit_color"].program
        pick_prog = self.pipelines["pick_id"].program
        
        vao_lit = self.ctx.vertex_array(
            lit_prog,
            [(vbo, "3f 3f", "in_pos", "in_nrm")],
            index_buffer=ibo,
        )
        
        vao_pick = self.ctx.vertex_array(
            pick_prog,
            [(vbo, "3f 3f", "in_pos", "in_nrm")],
            index_buffer=ibo,
        )
        
        self.meshes["cube"] = MeshGPU(
            vao_lit=vao_lit,
            vao_pick=vao_pick,
            vbo=vbo,
            ibo=ibo,
            index_count=len(indices),
            vertex_count=len(positions),
        )
    
    # -------------------------------------------------------------------------
    # Blitter (for compositing)
    # -------------------------------------------------------------------------
    
    def _create_blitter(self):
        """Create resources for blitting viewport textures to screen."""
        
        blit_vs = """
        #version 330
        
        in vec2 in_pos;
        in vec2 in_uv;
        
        out vec2 v_uv;
        
        uniform vec2 u_window;
        uniform vec4 u_rect;  // x, y, w, h in pixels (top-left origin)
        
        void main() {
            // Transform quad position to pixel rect
            vec2 px = vec2(
                u_rect.x + in_pos.x * u_rect.z,
                u_rect.y + in_pos.y * u_rect.w
            );
            
            // Convert to NDC
            vec2 ndc = vec2(
                (px.x / u_window.x) * 2.0 - 1.0,
                1.0 - (px.y / u_window.y) * 2.0  // Flip Y
            );
            
            gl_Position = vec4(ndc, 0.0, 1.0);
            v_uv = in_uv;
        }
        """
        
        blit_fs = """
        #version 330
        
        in vec2 v_uv;
        
        uniform sampler2D u_tex;
        
        out vec4 fragColor;
        
        void main() {
            fragColor = texture(u_tex, v_uv);
        }
        """
        
        self._blit_prog = self.ctx.program(vertex_shader=blit_vs, fragment_shader=blit_fs)
        
        # Fullscreen quad (0,0) to (1,1)
        quad = np.array([
            # pos_x, pos_y, uv_x, uv_y
            0, 0, 0, 0,
            1, 0, 1, 0,
            1, 1, 1, 1,
            0, 0, 0, 0,
            1, 1, 1, 1,
            0, 1, 0, 1,
        ], dtype=np.float32)
        
        self._blit_vbo = self.ctx.buffer(quad.tobytes())
        self._blit_vao = self.ctx.vertex_array(
            self._blit_prog,
            [(self._blit_vbo, "2f 2f", "in_pos", "in_uv")],
        )
    
    # -------------------------------------------------------------------------
    # Accessors
    # -------------------------------------------------------------------------
    
    def get_mesh(self, mesh_id: str) -> MeshGPU:
        """Get a mesh by ID. Raises KeyError if not found."""
        return self.meshes[mesh_id]
    
    def get_pipeline(self, pipeline_id: str) -> PipelineGPU:
        """Get a pipeline by ID. Raises KeyError if not found."""
        return self.pipelines[pipeline_id]
    
    def has_mesh(self, mesh_id: str) -> bool:
        return mesh_id in self.meshes
    
    def has_pipeline(self, pipeline_id: str) -> bool:
        return pipeline_id in self.pipelines
    
    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------
    
    def cleanup(self):
        """Release all GPU resources."""
        for mesh in self.meshes.values():
            mesh.release()
        self.meshes.clear()
        
        for pipeline in self.pipelines.values():
            pipeline.release()
        self.pipelines.clear()
        
        for tex in self.textures.values():
            tex.release()
        self.textures.clear()
        
        if self._blit_prog:
            self._blit_prog.release()
        if self._blit_vao:
            self._blit_vao.release()
        if self._blit_vbo:
            self._blit_vbo.release()
