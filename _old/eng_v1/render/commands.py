from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass(frozen=True)
class CmdSetViewport:
    x: int
    y: int
    w: int
    h: int

@dataclass(frozen=True)
class CmdSetScissor:
    x: int
    y: int
    w: int
    h: int
    enabled: bool = True

@dataclass(frozen=True)
class CmdClear:
    r: float
    g: float
    b: float
    a: float
    depth: bool = True
    clear_pick: bool = False

@dataclass(frozen=True)
class CmdDrawMesh:
    pipeline_id: str
    mesh_id: str
    uniforms: Dict[str, Any]
    entity_id: Optional[int] = None

Command = CmdSetViewport | CmdSetScissor | CmdClear | CmdDrawMesh

@dataclass
class CommandList:
    commands: List[Command] = field(default_factory=list)
    def add(self, cmd: Command):
        self.commands.append(cmd)
