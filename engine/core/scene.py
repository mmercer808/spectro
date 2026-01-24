# engine/core/scene.py
"""
Scene - The unified 3D coordinate space.

Conceptual space where:
- X axis = Time (beats)
- Y axis = Frequency (Hz) or Pitch
- Z axis = Intensity / Velocity / Amplitude
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Iterator, Tuple
from enum import Enum, auto
import json
import uuid

from .math3d import Vec3
from .signal import SignalBridge, SIGNAL_ENTITY_ADDED, SIGNAL_ENTITY_REMOVED, SIGNAL_ENTITY_CHANGED


# =============================================================================
# Entity System
# =============================================================================

class EntityType(Enum):
    GENERIC = auto()
    AUDIO_CLIP = auto()
    MIDI_EVENT = auto()
    MARKER = auto()
    REGION = auto()
    AUTOMATION = auto()


@dataclass
class Entity:
    """Base entity in the unified space."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    entity_type: EntityType = EntityType.GENERIC
    name: str = ""
    position: Vec3 = field(default_factory=Vec3)
    extent: Vec3 = field(default_factory=lambda: Vec3(1.0, 1.0, 1.0))
    color: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    visible: bool = True
    selected: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def beat(self) -> float:
        return self.position.x
    
    @beat.setter
    def beat(self, value: float):
        self.position.x = value
    
    @property
    def end_beat(self) -> float:
        return self.position.x + self.extent.x
    
    @property
    def duration_beats(self) -> float:
        return self.extent.x
    
    @duration_beats.setter
    def duration_beats(self, value: float):
        self.extent.x = value
    
    @property
    def frequency(self) -> float:
        return self.position.y
    
    @frequency.setter
    def frequency(self, value: float):
        self.position.y = value
    
    @property
    def intensity(self) -> float:
        return self.position.z
    
    @intensity.setter
    def intensity(self, value: float):
        self.position.z = value
    
    def contains_beat(self, beat: float) -> bool:
        return self.beat <= beat < self.end_beat
    
    def overlaps_range(self, start_beat: float, end_beat: float) -> bool:
        return self.beat < end_beat and self.end_beat > start_beat
    
    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'type': self.entity_type.name,
            'name': self.name,
            'position': self.position.to_tuple(),
            'extent': self.extent.to_tuple(),
            'color': self.color,
            'visible': self.visible,
            'metadata': self.metadata,
        }
    
    @staticmethod
    def from_dict(data: dict) -> Entity:
        return Entity(
            id=data.get('id', str(uuid.uuid4())),
            entity_type=EntityType[data.get('type', 'GENERIC')],
            name=data.get('name', ''),
            position=Vec3.from_tuple(data.get('position', (0, 0, 0))),
            extent=Vec3.from_tuple(data.get('extent', (1, 1, 1))),
            color=tuple(data.get('color', (1, 1, 1, 1))),
            visible=data.get('visible', True),
            metadata=data.get('metadata', {}),
        )


# =============================================================================
# Specialized Entities
# =============================================================================

@dataclass
class AudioClipEntity(Entity):
    entity_type: EntityType = EntityType.AUDIO_CLIP
    file_path: str = ""
    sample_rate: int = 44100
    channels: int = 2
    source_start_sample: int = 0
    source_end_sample: int = 0
    
    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update({
            'file_path': self.file_path,
            'sample_rate': self.sample_rate,
            'channels': self.channels,
            'source_start_sample': self.source_start_sample,
            'source_end_sample': self.source_end_sample,
        })
        return d


@dataclass
class MidiEventEntity(Entity):
    entity_type: EntityType = EntityType.MIDI_EVENT
    note: int = 60
    velocity: int = 100
    channel: int = 0
    
    def __post_init__(self):
        self.position.z = self.velocity / 127.0
    
    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update({
            'note': self.note,
            'velocity': self.velocity,
            'channel': self.channel,
        })
        return d


@dataclass
class MarkerEntity(Entity):
    entity_type: EntityType = EntityType.MARKER
    marker_type: str = "generic"
    label: str = ""
    
    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update({
            'marker_type': self.marker_type,
            'label': self.label,
        })
        return d


# =============================================================================
# Scene
# =============================================================================

@dataclass
class SceneBounds:
    time_min: float = 0.0
    time_max: float = 1000.0
    freq_min: float = 20.0
    freq_max: float = 20000.0
    intensity_min: float = 0.0
    intensity_max: float = 1.0


class Scene:
    """The unified 3D coordinate space."""
    
    def __init__(self, bridge: SignalBridge = None):
        self._bridge = bridge
        self._entities: Dict[str, Entity] = {}
        self._entities_by_type: Dict[EntityType, List[str]] = {}
        self.bounds = SceneBounds()
        self._selected_ids: List[str] = []
        self._dirty = False
    
    def bind_bridge(self, bridge: SignalBridge):
        self._bridge = bridge
    
    def add(self, entity: Entity) -> Entity:
        self._entities[entity.id] = entity
        
        if entity.entity_type not in self._entities_by_type:
            self._entities_by_type[entity.entity_type] = []
        self._entities_by_type[entity.entity_type].append(entity.id)
        
        self._dirty = True
        
        if self._bridge:
            self._bridge.emit(SIGNAL_ENTITY_ADDED, entity)
        
        return entity
    
    def remove(self, entity_id: str) -> Optional[Entity]:
        entity = self._entities.pop(entity_id, None)
        if entity:
            type_list = self._entities_by_type.get(entity.entity_type, [])
            if entity_id in type_list:
                type_list.remove(entity_id)
            
            if entity_id in self._selected_ids:
                self._selected_ids.remove(entity_id)
            
            self._dirty = True
            
            if self._bridge:
                self._bridge.emit(SIGNAL_ENTITY_REMOVED, entity_id)
        
        return entity
    
    def get(self, entity_id: str) -> Optional[Entity]:
        return self._entities.get(entity_id)
    
    def update(self, entity: Entity):
        if entity.id in self._entities:
            self._dirty = True
            if self._bridge:
                self._bridge.emit(SIGNAL_ENTITY_CHANGED, entity)
    
    def clear(self):
        self._entities.clear()
        self._entities_by_type.clear()
        self._selected_ids.clear()
        self._dirty = True
    
    def all(self) -> Iterator[Entity]:
        return iter(self._entities.values())
    
    def by_type(self, entity_type: EntityType) -> Iterator[Entity]:
        ids = self._entities_by_type.get(entity_type, [])
        for eid in ids:
            entity = self._entities.get(eid)
            if entity:
                yield entity
    
    def in_time_range(self, start_beat: float, end_beat: float) -> Iterator[Entity]:
        for entity in self._entities.values():
            if entity.overlaps_range(start_beat, end_beat):
                yield entity
    
    def at_beat(self, beat: float) -> Iterator[Entity]:
        for entity in self._entities.values():
            if entity.contains_beat(beat):
                yield entity
    
    def in_box(self, min_pos: Vec3, max_pos: Vec3) -> Iterator[Entity]:
        for entity in self._entities.values():
            p = entity.position
            if (min_pos.x <= p.x <= max_pos.x and
                min_pos.y <= p.y <= max_pos.y and
                min_pos.z <= p.z <= max_pos.z):
                yield entity
    
    def visible(self) -> Iterator[Entity]:
        for entity in self._entities.values():
            if entity.visible:
                yield entity
    
    @property
    def count(self) -> int:
        return len(self._entities)
    
    def select(self, entity_id: str, add_to_selection: bool = False):
        if not add_to_selection:
            for eid in self._selected_ids:
                entity = self._entities.get(eid)
                if entity:
                    entity.selected = False
            self._selected_ids.clear()
        
        entity = self._entities.get(entity_id)
        if entity:
            entity.selected = True
            if entity_id not in self._selected_ids:
                self._selected_ids.append(entity_id)
    
    def deselect(self, entity_id: str):
        entity = self._entities.get(entity_id)
        if entity:
            entity.selected = False
        if entity_id in self._selected_ids:
            self._selected_ids.remove(entity_id)
    
    def deselect_all(self):
        for eid in self._selected_ids:
            entity = self._entities.get(eid)
            if entity:
                entity.selected = False
        self._selected_ids.clear()
    
    def selected(self) -> Iterator[Entity]:
        for eid in self._selected_ids:
            entity = self._entities.get(eid)
            if entity:
                yield entity
    
    @property
    def selection_count(self) -> int:
        return len(self._selected_ids)
    
    def to_dict(self) -> dict:
        return {
            'bounds': {
                'time_min': self.bounds.time_min,
                'time_max': self.bounds.time_max,
                'freq_min': self.bounds.freq_min,
                'freq_max': self.bounds.freq_max,
                'intensity_min': self.bounds.intensity_min,
                'intensity_max': self.bounds.intensity_max,
            },
            'entities': [e.to_dict() for e in self._entities.values()],
        }
    
    def from_dict(self, data: dict):
        self.clear()
        
        bounds = data.get('bounds', {})
        self.bounds.time_min = bounds.get('time_min', 0.0)
        self.bounds.time_max = bounds.get('time_max', 1000.0)
        self.bounds.freq_min = bounds.get('freq_min', 20.0)
        self.bounds.freq_max = bounds.get('freq_max', 20000.0)
        self.bounds.intensity_min = bounds.get('intensity_min', 0.0)
        self.bounds.intensity_max = bounds.get('intensity_max', 1.0)
        
        for entity_data in data.get('entities', []):
            entity = Entity.from_dict(entity_data)
            self.add(entity)
    
    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def load(self, path: str):
        with open(path, 'r') as f:
            data = json.load(f)
        self.from_dict(data)
