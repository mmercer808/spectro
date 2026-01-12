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
    vbo: moderngl.Buffer
    ibo: moderngl.Buffer
    index_count: int
    vertex_count: int
    # VAOs are created per-pipeline, stored separately
    
    def release(self):
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
    """
    
    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        self.meshes: Dict[str, MeshGPU] = {}
        self.pipelines: Dict[str, PipelineGPU] = {}
        self.textures: Dict[str, moderngl.Texture] = {}
        
        # VAO cache: (mesh_id, pipeline_id, instance_buffer_id) -> VAO
        self._vao_cache: Dict[Tuple, moderngl.VertexArray] = {}
        
        # Compositor resources
        self._blit_prog: Optional[moderngl.Program] = None
        self._blit_vao: Optional[moderngl.VertexArray] = None
        self._blit_vbo: Optional[moderngl.Buffer] = None
    
    def bootstrap_defaults(self):
        """Create default resources."""
        self._create_default_pipelines()
        self._create_default_meshes()
        self._create_blitter()
    
    # -------------------------------------------------------------------------
    # Pipelines
    # -------------------------------------------------------------------------
    
    def _create_default_pipelines(self):
        """Create built-in shader pipelines."""
        
        # --- Standard lit shader (single object) ---
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
        
        # --- Instanced shader (uses GPUBuffer layout) ---
        instanced_vs = """
        #version 330
        
        // Per-vertex attributes (from mesh)
        in vec3 in_pos;
        in vec3 in_nrm;
        
        // Per-instance attributes (from GPUBuffer, matches INSTANCE_DTYPE)
        in vec3 inst_position;
        in vec3 inst_scale;
        in vec4 inst_rotation;  // quaternion: w, x, y, z
        in vec4 inst_color;
        in float inst_value;
        in uint inst_flags;
        
        uniform mat4 u_view;
        uniform mat4 u_proj;
        
        out vec3 v_normal;
        out vec4 v_color;
        out float v_value;
        flat out uint v_flags;
        
        // Rotate vector by quaternion
        vec3 quat_rotate(vec4 q, vec3 v) {
            vec3 t = 2.0 * cross(q.xyz, v);
            return v + q.w * t + cross(q.xyz, t);
        }
        
        void main() {
            // Check visibility flag (bit 0)
            if ((inst_flags & 1u) == 0u) {
                // Move vertex off-screen to discard
                gl_Position = vec4(0.0, 0.0, -1000.0, 1.0);
                return;
            }
            
            // Apply scale
            vec3 scaled = in_pos * inst_scale;
            
            // Apply rotation (quaternion)
            vec3 rotated = quat_rotate(inst_rotation, scaled);
            
            // Apply translation
            vec3 world_pos = rotated + inst_position;
            
            // Transform to clip space
            vec4 view_pos = u_view * vec4(world_pos, 1.0);
            gl_Position = u_proj * view_pos;
            
            // Transform normal
            v_normal = quat_rotate(inst_rotation, in_nrm);
            
            v_color = inst_color;
            v_value = inst_value;
            v_flags = inst_flags;
        }
        """
        
        instanced_fs = """
        #version 330
        
        in vec3 v_normal;
        in vec4 v_color;
        in float v_value;
        flat in uint v_flags;
        
        uniform int u_color_mode;  // 0=solid, 1=value_ramp
        
        layout(location = 0) out vec4 fragColor;
        layout(location = 1) out vec4 pickOut;
        
        // Simple value to color ramp
        vec3 value_to_color(float v) {
            // Blue -> Cyan -> Green -> Yellow -> Red
            float t = clamp(v, 0.0, 1.0);
            if (t < 0.25) {
                return mix(vec3(0.0, 0.0, 1.0), vec3(0.0, 1.0, 1.0), t * 4.0);
            } else if (t < 0.5) {
                return mix(vec3(0.0, 1.0, 1.0), vec3(0.0, 1.0, 0.0), (t - 0.25) * 4.0);
            } else if (t < 0.75) {
                return mix(vec3(0.0, 1.0, 0.0), vec3(1.0, 1.0, 0.0), (t - 0.5) * 4.0);
            } else {
                return mix(vec3(1.0, 1.0, 0.0), vec3(1.0, 0.0, 0.0), (t - 0.75) * 4.0);
            }
        }
        
        void main() {
            vec3 light_dir = normalize(vec3(0.5, 1.0, 0.3));
            float ndotl = max(dot(normalize(v_normal), light_dir), 0.0);
            float ambient = 0.3;
            float diffuse = 0.7 * ndotl;
            
            vec3 base_color;
            if (u_color_mode == 1) {
                base_color = value_to_color(v_value);
            } else {
                base_color = v_color.rgb;
            }
            
            // Highlight if selected (bit 1)
            if ((v_flags & 2u) != 0u) {
                base_color = mix(base_color, vec3(1.0, 0.8, 0.2), 0.5);
            }
            
            fragColor = vec4(base_color * (ambient + diffuse), v_color.a);
            
            // TODO: Per-instance pick ID
            pickOut = vec4(0.0, 0.0, 0.0, 1.0);
        }
        """
        
        self.pipelines["instanced"] = PipelineGPU(
            program=self.ctx.program(vertex_shader=instanced_vs, fragment_shader=instanced_fs)
        )
        
        # Keep old name for compatibility
        self.pipelines["lit_color_instanced"] = self.pipelines["instanced"]
    
    # -------------------------------------------------------------------------
    # Meshes
    # -------------------------------------------------------------------------
    
    def _create_default_meshes(self):
        """Create built-in meshes."""
        self._create_cube_mesh()
        self._create_sphere_mesh()
    
    def _create_cube_mesh(self):
        """Create a unit cube mesh."""
        positions = []
        normals = []
        indices = []
        
        def add_face(p0, p1, p2, p3, n):
            base = len(positions)
            positions.extend([p0, p1, p2, p3])
            normals.extend([n, n, n, n])
            indices.extend([base, base + 1, base + 2, base, base + 2, base + 3])
        
        s = 0.5
        add_face((-s, -s, s), (s, -s, s), (s, s, s), (-s, s, s), (0, 0, 1))
        add_face((s, -s, -s), (-s, -s, -s), (-s, s, -s), (s, s, -s), (0, 0, -1))
        add_face((s, -s, s), (s, -s, -s), (s, s, -s), (s, s, s), (1, 0, 0))
        add_face((-s, -s, -s), (-s, -s, s), (-s, s, s), (-s, s, -s), (-1, 0, 0))
        add_face((-s, s, s), (s, s, s), (s, s, -s), (-s, s, -s), (0, 1, 0))
        add_face((-s, -s, -s), (s, -s, -s), (s, -s, s), (-s, -s, s), (0, -1, 0))
        
        v = np.array(positions, dtype=np.float32)
        n = np.array(normals, dtype=np.float32)
        interleaved = np.hstack([v, n]).astype(np.float32)
        idx = np.array(indices, dtype=np.uint32)
        
        vbo = self.ctx.buffer(interleaved.tobytes())
        ibo = self.ctx.buffer(idx.tobytes())
        
        self.meshes["cube"] = MeshGPU(
            vbo=vbo,
            ibo=ibo,
            index_count=len(indices),
            vertex_count=len(positions),
        )
    
    def _create_sphere_mesh(self, segments: int = 16, rings: int = 12):
        """Create a unit sphere mesh."""
        positions = []
        normals = []
        indices = []
        
        for j in range(rings + 1):
            phi = np.pi * j / rings
            for i in range(segments + 1):
                theta = 2 * np.pi * i / segments
                
                x = np.sin(phi) * np.cos(theta)
                y = np.cos(phi)
                z = np.sin(phi) * np.sin(theta)
                
                positions.append((x * 0.5, y * 0.5, z * 0.5))
                normals.append((x, y, z))
        
        for j in range(rings):
            for i in range(segments):
                p0 = j * (segments + 1) + i
                p1 = p0 + 1
                p2 = p0 + (segments + 1)
                p3 = p2 + 1
                
                indices.extend([p0, p2, p1])
                indices.extend([p1, p2, p3])
        
        v = np.array(positions, dtype=np.float32)
        n = np.array(normals, dtype=np.float32)
        interleaved = np.hstack([v, n]).astype(np.float32)
        idx = np.array(indices, dtype=np.uint32)
        
        vbo = self.ctx.buffer(interleaved.tobytes())
        ibo = self.ctx.buffer(idx.tobytes())
        
        self.meshes["sphere"] = MeshGPU(
            vbo=vbo,
            ibo=ibo,
            index_count=len(indices),
            vertex_count=len(positions),
        )
    
    # -------------------------------------------------------------------------
    # VAO Management
    # -------------------------------------------------------------------------
    
    def get_or_create_vao(
        self,
        mesh_id: str,
        pipeline_id: str,
        instance_buffer: Optional[moderngl.Buffer] = None,
        instance_format: Optional[str] = None,
        instance_attribs: Optional[List[str]] = None,
    ) -> moderngl.VertexArray:
        """
        Get or create a VAO for the given mesh/pipeline/instance buffer combination.
        """
        # Build cache key
        buf_id = id(instance_buffer) if instance_buffer else None
        key = (mesh_id, pipeline_id, buf_id)
        
        if key in self._vao_cache:
            return self._vao_cache[key]
        
        mesh = self.meshes[mesh_id]
        pipeline = self.pipelines[pipeline_id]
        prog = pipeline.program
        
        # Build VAO content
        content = [
            (mesh.vbo, '3f 3f', 'in_pos', 'in_nrm'),
        ]
        
        # Add instance buffer if provided
        if instance_buffer and instance_format and instance_attribs:
            content.append((
                instance_buffer,
                instance_format + ' /i',  # /i = per-instance
                *instance_attribs,
            ))
        
        vao = self.ctx.vertex_array(
            prog,
            content,
            index_buffer=mesh.ibo,
        )
        
        self._vao_cache[key] = vao
        return vao
    
    def invalidate_vao_cache(self):
        """Clear VAO cache (e.g., when buffers are recreated)."""
        for vao in self._vao_cache.values():
            vao.release()
        self._vao_cache.clear()
    
    # -------------------------------------------------------------------------
    # Blitter
    # -------------------------------------------------------------------------
    
    def _create_blitter(self):
        """Create resources for blitting viewport textures to screen."""
        
        blit_vs = """
        #version 330
        
        in vec2 in_pos;
        in vec2 in_uv;
        
        out vec2 v_uv;
        
        uniform vec2 u_window;
        uniform vec4 u_rect;
        
        void main() {
            vec2 px = vec2(
                u_rect.x + in_pos.x * u_rect.z,
                u_rect.y + in_pos.y * u_rect.w
            );
            
            vec2 ndc = vec2(
                (px.x / u_window.x) * 2.0 - 1.0,
                1.0 - (px.y / u_window.y) * 2.0
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
        
        quad = np.array([
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
            [(self._blit_vbo, '2f 2f', 'in_pos', 'in_uv')],
        )
    
    # -------------------------------------------------------------------------
    # Accessors
    # -------------------------------------------------------------------------
    
    def get_mesh(self, mesh_id: str) -> MeshGPU:
        return self.meshes[mesh_id]
    
    def get_pipeline(self, pipeline_id: str) -> PipelineGPU:
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
        self.invalidate_vao_cache()
        
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
