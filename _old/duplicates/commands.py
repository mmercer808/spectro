"""
Command List System

Pure-data command representation for thread-safe rendering.
CommandLists contain NO GL objects - only stable IDs, keys, and numpy arrays.

This is critical: workers produce CommandLists on any thread,
the GL thread consumes them. If CommandLists held GL objects,
we'd have threading bugs.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


# =============================================================================
# Basic Commands
# =============================================================================

@dataclass(frozen=True)
class CmdSetViewport:
    """Set the GL viewport rectangle."""
    x: int
    y: int
    w: int
    h: int


@dataclass(frozen=True)
class CmdSetScissor:
    """Set scissor test rectangle."""
    x: int
    y: int
    w: int
    h: int
    enabled: bool = True


@dataclass(frozen=True)
class CmdClear:
    """Clear color and/or depth buffers."""
    r: float
    g: float
    b: float
    a: float
    depth: bool = True
    clear_pick: bool = False


# =============================================================================
# Draw Commands
# =============================================================================

@dataclass(frozen=True)
class CmdDrawMesh:
    """
    Draw a single mesh with given uniforms.
    
    This is the simple path - one draw call per item.
    Good for small scenes or items that can't be batched.
    """
    pipeline_id: str
    mesh_id: str
    uniforms: Dict[str, Any]
    entity_id: Optional[int] = None


@dataclass(frozen=True)
class CmdDrawMeshInstanced:
    """
    Draw multiple instances of a mesh in one call.
    
    OLD VERSION - carries instance data in the command.
    Kept for compatibility during transition.
    """
    pipeline_id: str
    mesh_id: str
    instance_count: int
    instance_buffer_key: str
    instance_data: np.ndarray  # Packed instance data (uploaded each frame)
    view_proj: np.ndarray
    entity_ids: Tuple[int, ...]
    camera_index: int = 0


@dataclass(frozen=True)
class CmdDrawPointCloud:
    """
    Draw instances from a persistent GPUBuffer.
    
    NEW VERSION - references a buffer range, no data copying.
    The buffer is already on GPU, just need to bind and draw.
    """
    pipeline_id: str
    mesh_id: str
    buffer_key: str           # Key to look up GPUBuffer in registry
    start: int                # First instance index
    count: int                # Number of instances to draw
    view_matrix: np.ndarray   # 4x4 view matrix
    proj_matrix: np.ndarray   # 4x4 projection matrix
    camera_index: int = 0


# =============================================================================
# Render Pass Commands
# =============================================================================

@dataclass(frozen=True)
class CmdBeginPass:
    """Begin a named render pass."""
    name: str
    clear_color: Optional[Tuple[float, float, float, float]] = None
    clear_depth: bool = False


@dataclass(frozen=True)
class CmdEndPass:
    """End the current render pass."""
    name: str


# =============================================================================
# Future Commands
# =============================================================================

@dataclass(frozen=True)
class CmdSetRenderTarget:
    """Switch to a different render target."""
    target_key: str


@dataclass(frozen=True)
class CmdBlit:
    """Blit one texture to another."""
    src_key: str
    dst_key: str
    shader_id: Optional[str] = None


# =============================================================================
# Command Type Union
# =============================================================================

Command = Union[
    CmdSetViewport,
    CmdSetScissor,
    CmdClear,
    CmdDrawMesh,
    CmdDrawMeshInstanced,
    CmdDrawPointCloud,
    CmdBeginPass,
    CmdEndPass,
    CmdSetRenderTarget,
    CmdBlit,
]


# =============================================================================
# Command List
# =============================================================================

@dataclass
class CommandList:
    """
    An ordered list of rendering commands.
    
    IMPORTANT: CommandLists must contain NO GL objects!
    """
    commands: List[Command] = field(default_factory=list)
    source_snapshot_id: Optional[int] = None
    camera_index: Optional[int] = None
    
    def add(self, cmd: Command):
        """Append a command to the list."""
        self.commands.append(cmd)
    
    def extend(self, cmds: List[Command]):
        """Append multiple commands."""
        self.commands.extend(cmds)
    
    def __len__(self) -> int:
        return len(self.commands)
    
    def __iter__(self):
        return iter(self.commands)
    
    def validate(self) -> List[str]:
        """Check that commandlist contains no GL objects."""
        errors = []
        
        for i, cmd in enumerate(self.commands):
            if hasattr(cmd, '__dict__'):
                for key, val in cmd.__dict__.items():
                    if _looks_like_gl_object(val):
                        errors.append(
                            f"Command {i} ({type(cmd).__name__}): "
                            f"field '{key}' appears to be GL object"
                        )
            
            if isinstance(cmd, CmdDrawMesh):
                for key, val in cmd.uniforms.items():
                    if _looks_like_gl_object(val):
                        errors.append(
                            f"Command {i} ({type(cmd).__name__}): "
                            f"uniform '{key}' appears to be GL object"
                        )
        
        return errors
    
    def get_stats(self) -> Dict[str, int]:
        """Get command count by type."""
        stats: Dict[str, int] = {}
        for cmd in self.commands:
            name = type(cmd).__name__
            stats[name] = stats.get(name, 0) + 1
        return stats
    
    def get_draw_count(self) -> int:
        """Count draw commands."""
        return sum(
            1 for c in self.commands 
            if isinstance(c, (CmdDrawMesh, CmdDrawMeshInstanced, CmdDrawPointCloud))
        )
    
    def get_instance_count(self) -> int:
        """Count total instances to be drawn."""
        count = 0
        for cmd in self.commands:
            if isinstance(cmd, CmdDrawMesh):
                count += 1
            elif isinstance(cmd, CmdDrawMeshInstanced):
                count += cmd.instance_count
            elif isinstance(cmd, CmdDrawPointCloud):
                count += cmd.count
        return count


def _looks_like_gl_object(val: Any) -> bool:
    """Heuristic check if a value looks like a GL object."""
    if val is None:
        return False
    
    type_name = type(val).__name__.lower()
    module = getattr(type(val), '__module__', '') or ''
    
    if 'moderngl' in module:
        return True
    
    gl_patterns = [
        'buffer', 'texture', 'shader', 'program', 'vao', 'fbo',
        'framebuffer', 'renderbuffer', 'vertexarray'
    ]
    if any(p in type_name for p in gl_patterns):
        return True
    
    return False


# =============================================================================
# Command List Merging
# =============================================================================

def merge_command_lists(
    cmdlists: List[CommandList],
    prepend_clear: Optional[CmdClear] = None
) -> CommandList:
    """Merge multiple command lists into one."""
    merged = CommandList()
    
    if prepend_clear:
        merged.add(prepend_clear)
    
    for cl in cmdlists:
        merged.commands.extend(cl.commands)
        
        if merged.source_snapshot_id is None and cl.source_snapshot_id is not None:
            merged.source_snapshot_id = cl.source_snapshot_id
    
    return merged
