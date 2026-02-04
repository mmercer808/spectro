"""
spectro_ui_glue_stubs_onefile.py

Purpose
-------
You said you already have code for viewports/UI, but you want:

- proxy/stub classes for "everything" (model/viewports/commands/events/devices)
- the glue classes with a systematic basis (event -> model -> viewport projection)
- wired up to the buffer+engine system we built (audio + MIDI ring buffers + ObserverHub)

This file is intentionally designed to be dropped into a project and then *connected*
to your existing implementations by subclassing / filling a few TODO methods.

Philosophy / Thought process (kept close to code)
-------------------------------------------------
1) Events are facts; UI is a projection.
   - Events update the model.
   - Viewports project model entities into UI components.
   - UI interactions become commands -> events.

2) Time unification:
   - MIDI and audio are stamped in absolute sample indices (engine.clock samples).
   - Viewports convert sample time <-> screen x using a Camera mapping.

3) Wiring:
   - ObserverHub receives device events and routes them into Engine buffers (already done).
   - We add SequencerGlue that:
        a) listens to ObserverHub topics ("midi", "audio_in"/"audio_out"/"system_audio")
        b) turns raw events into domain events (NoteAdded/NoteMoved/etc.)
        c) applies them to a TimelineModel
        d) emits change-sets to Viewports, which update UI components.

4) Priority routing (your “drop into a set and sort to front” idea):
   - A PriorityEventRouter owns handlers sorted by priority.
   - Handlers can "consume" an event to stop propagation.
   - This keeps the system deterministic and fast to extend.

Signature
---------
SIGNATURE = "Spectro GlueKit — GPT-5.2 Thinking (events-first, timeline-truth, viewport-projections)"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol, Tuple, Union
import threading
import time
import uuid

# This file expects the Engine + ObserverHub + MidiEvent types from the bufferkit.
# If your project structure differs, adjust these imports.
try:
    from spectro_bufferkit_onefile import Engine, ObserverHub, MidiEvent, AudioBlock, TimedEvent
except Exception:
    # Fallback: let you import manually in your project.
    Engine = Any
    ObserverHub = Any
    MidiEvent = Any
    AudioBlock = Any
    TimedEvent = Any


SIGNATURE = "Spectro GlueKit — GPT-5.2 Thinking (events-first, timeline-truth, viewport-projections)"


# =============================================================================
# 1) Domain Events (facts that mutate the model)
# =============================================================================

@dataclass(frozen=True)
class DomainEvent:
    """Base class for domain events that mutate the sequencer model."""
    when_samples: int
    kind: str
    payload: Dict[str, Any] = field(default_factory=dict)

# Core sequencer events you will likely want
def NoteAdded(when_samples: int, note_id: str, start: int, end: int, pitch: int, vel: int, channel: int, track_id: str) -> DomainEvent:
    return DomainEvent(when_samples, "note.added", dict(note_id=note_id, start=start, end=end, pitch=pitch, vel=vel, channel=channel, track_id=track_id))

def NoteMoved(when_samples: int, note_id: str, start: int, end: int, pitch: int) -> DomainEvent:
    return DomainEvent(when_samples, "note.moved", dict(note_id=note_id, start=start, end=end, pitch=pitch))

def NoteDeleted(when_samples: int, note_id: str) -> DomainEvent:
    return DomainEvent(when_samples, "note.deleted", dict(note_id=note_id))

def MarkerAdded(when_samples: int, marker_id: str, label: str) -> DomainEvent:
    return DomainEvent(when_samples, "marker.added", dict(marker_id=marker_id, label=label))

def TempoChanged(when_samples: int, bpm: float) -> DomainEvent:
    return DomainEvent(when_samples, "tempo.changed", dict(bpm=float(bpm)))


# =============================================================================
# 2) Model proxy (entity store + apply(event) + diff)
# =============================================================================

@dataclass
class ChangeSet:
    """Minimal change-set a viewport can consume efficiently."""
    added: List[str] = field(default_factory=list)
    updated: List[str] = field(default_factory=list)
    removed: List[str] = field(default_factory=list)

@dataclass
class NoteEntity:
    id: str
    start: int
    end: int
    pitch: int
    vel: int
    channel: int
    track_id: str

@dataclass
class MarkerEntity:
    id: str
    when: int
    label: str

class TimelineModelProxy:
    """
    Proxy/stub model.
    Replace or subclass this to integrate your existing song/sequence model.

    Contract:
      - apply(event) returns a ChangeSet describing entity ids that changed.
      - get_entity(entity_id) returns note/marker/etc.
      - query_range(s0, s1) returns entity ids in a time range (optional for viewport rebuilds)
    """
    def __init__(self):
        self._lock = threading.RLock()
        self.notes: Dict[str, NoteEntity] = {}
        self.markers: Dict[str, MarkerEntity] = {}
        self.bpm: float = 120.0

    def apply(self, ev: DomainEvent) -> ChangeSet:
        with self._lock:
            cs = ChangeSet()
            k = ev.kind
            p = ev.payload

            if k == "note.added":
                nid = p["note_id"]
                self.notes[nid] = NoteEntity(
                    id=nid, start=int(p["start"]), end=int(p["end"]),
                    pitch=int(p["pitch"]), vel=int(p["vel"]),
                    channel=int(p["channel"]), track_id=str(p["track_id"])
                )
                cs.added.append(nid)
                return cs

            if k == "note.moved":
                nid = p["note_id"]
                n = self.notes.get(nid)
                if n:
                    n.start = int(p["start"])
                    n.end = int(p["end"])
                    n.pitch = int(p["pitch"])
                    cs.updated.append(nid)
                return cs

            if k == "note.deleted":
                nid = p["note_id"]
                if nid in self.notes:
                    del self.notes[nid]
                    cs.removed.append(nid)
                return cs

            if k == "marker.added":
                mid = p["marker_id"]
                self.markers[mid] = MarkerEntity(id=mid, when=int(ev.when_samples), label=str(p.get("label", "")))
                cs.added.append(mid)
                return cs

            if k == "tempo.changed":
                self.bpm = float(p["bpm"])
                # tempo affects projection; you might broadcast a viewport "rebuild"
                cs.updated.append("__tempo__")
                return cs

            # Unknown event: no change
            return cs

    def get_entity(self, entity_id: str) -> Optional[Any]:
        with self._lock:
            if entity_id in self.notes:
                return self.notes[entity_id]
            if entity_id in self.markers:
                return self.markers[entity_id]
            return None

    def query_range(self, s0: int, s1: int) -> List[str]:
        """Naive range query; replace with your spatial/time index later."""
        s0 = int(s0); s1 = int(s1)
        out: List[str] = []
        with self._lock:
            for nid, n in self.notes.items():
                if n.end >= s0 and n.start < s1:
                    out.append(nid)
            for mid, m in self.markers.items():
                if s0 <= m.when < s1:
                    out.append(mid)
        return out


# =============================================================================
# 3) Viewport proxies: project model entities into UI components
# =============================================================================

@dataclass
class CameraProxy:
    """
    Convert sample-time <-> screen x. Replace with your real camera/viewport mapping.

    The default mapping assumes:
      x = (samples - origin_samples) * pixels_per_sample
    """
    origin_samples: int = 0
    pixels_per_sample: float = 0.001  # tune per viewport zoom
    y_pitch_origin: int = 60
    pixels_per_pitch: float = 6.0

    def x_from_samples(self, s: int) -> float:
        return (int(s) - int(self.origin_samples)) * float(self.pixels_per_sample)

    def samples_from_x(self, x: float) -> int:
        return int(round(float(x) / float(self.pixels_per_sample))) + int(self.origin_samples)

    def y_from_pitch(self, pitch: int) -> float:
        return (int(self.y_pitch_origin) - int(pitch)) * float(self.pixels_per_pitch)

    def pitch_from_y(self, y: float) -> int:
        return int(self.y_pitch_origin - (float(y) / float(self.pixels_per_pitch)))


@dataclass
class UIComponentProxy:
    """
    Proxy/stub UI component node.
    Your UI likely has its own Widget/Node class — this just captures the linking requirements.

    Required fields:
      - id: stable UI node id
      - entity_id: model entity id that this component represents
      - kind: "note", "marker", etc.
      - geom: any geometry you need (rect, points, etc.)
    """
    id: str
    entity_id: str
    kind: str
    geom: Dict[str, Any] = field(default_factory=dict)
    style: Dict[str, Any] = field(default_factory=dict)


class ViewportProxy(Protocol):
    """
    Contract a viewport must satisfy for the glue to wire it up.

    Minimal:
      - camera: provides mapping functions
      - upsert_component(component): create or update UI component
      - remove_component(component_id): remove UI component
      - resolve_component_for_entity(entity_id) -> component_id or None
      - request_redraw(): schedule repaint
    """
    camera: CameraProxy
    def upsert_component(self, comp: UIComponentProxy) -> None: ...
    def remove_component(self, comp_id: str) -> None: ...
    def resolve_component_for_entity(self, entity_id: str) -> Optional[str]: ...
    def request_redraw(self) -> None: ...


class ViewportAdapter:
    """
    A concrete base you can subclass to hook into your existing UI system.
    If you're using PySide6 / QtGraphics / ModernGL overlays, override the 4 methods.
    """
    def __init__(self, name: str = "viewport"):
        self.name = name
        self.camera = CameraProxy()
        self._entity_to_comp: Dict[str, str] = {}
        self._comps: Dict[str, UIComponentProxy] = {}
        self._lock = threading.RLock()

    def upsert_component(self, comp: UIComponentProxy) -> None:
        with self._lock:
            self._comps[comp.id] = comp
            self._entity_to_comp[comp.entity_id] = comp.id

    def remove_component(self, comp_id: str) -> None:
        with self._lock:
            comp = self._comps.pop(comp_id, None)
            if comp:
                self._entity_to_comp.pop(comp.entity_id, None)

    def resolve_component_for_entity(self, entity_id: str) -> Optional[str]:
        with self._lock:
            return self._entity_to_comp.get(entity_id)

    def request_redraw(self) -> None:
        # TODO: connect to your UI invalidation / paint schedule
        pass


# =============================================================================
# 4) Priority router (your “sort to front” idea)
# =============================================================================

@dataclass(order=True)
class _HandlerEntry:
    sort_key: Tuple[int, int]          # (-priority, order)
    handler_id: str = field(compare=False)
    fn: Callable[[Any], bool] = field(compare=False)  # returns consumed?
    topic: str = field(compare=False)

class PriorityEventRouter:
    """
    Deterministic, priority-sorted event router.
    Handlers can be registered on topics and may "consume" events to stop propagation.
    """
    def __init__(self):
        self._lock = threading.RLock()
        self._order = 0
        self._handlers: Dict[str, _HandlerEntry] = {}          # by id
        self._topics: Dict[str, List[_HandlerEntry]] = {}      # sorted lists

    def add(self, topic: str, fn: Callable[[Any], bool], priority: int = 0, handler_id: Optional[str] = None) -> str:
        with self._lock:
            self._order += 1
            hid = handler_id or f"h_{uuid.uuid4().hex[:8]}"
            entry = _HandlerEntry(sort_key=(-int(priority), int(self._order)), handler_id=hid, fn=fn, topic=str(topic))
            self._handlers[hid] = entry
            lst = self._topics.setdefault(str(topic), [])
            lst.append(entry)
            lst.sort()
            return hid

    def remove(self, handler_id: str) -> None:
        with self._lock:
            entry = self._handlers.pop(handler_id, None)
            if not entry:
                return
            lst = self._topics.get(entry.topic, [])
            self._topics[entry.topic] = [e for e in lst if e.handler_id != handler_id]

    def dispatch(self, topic: str, payload: Any) -> None:
        with self._lock:
            handlers = list(self._topics.get(str(topic), []))
        for h in handlers:
            try:
                consumed = bool(h.fn(payload))
                if consumed:
                    return
            except Exception:
                # Swallow handler errors to keep realtime loop stable; log in your app.
                continue


# =============================================================================
# 5) Translators: Device/MIDI -> Domain Events
# =============================================================================

class MidiToSequencerObserver:
    """
    Translate raw MidiEvent bytes into DomainEvent(s) for the sequencer model.

    This is intentionally a stub:
    - You probably want a per-channel note-state table
    - You might want quantize, recording modes, arming per track, etc.

    Provided:
    - Basic NoteOn -> NoteAdded placeholder with a default duration
    """
    def __init__(self, track_id: str = "track0", default_note_len_samples: int = 22050):
        self.track_id = str(track_id)
        self.default_note_len = int(default_note_len_samples)
        self._held: Dict[Tuple[int, int], Tuple[str, int, int, int]] = {}  # (chan,pitch)->(note_id,start,vel,chan)
        self._lock = threading.Lock()

    @staticmethod
    def _parse_midi(data: bytes) -> Tuple[int, int, int, int]:
        """
        Very small MIDI parser:
          returns (status, channel, data1, data2)
        """
        b = list(data)
        status = b[0] if b else 0
        channel = status & 0x0F
        msg = status & 0xF0
        d1 = b[1] if len(b) > 1 else 0
        d2 = b[2] if len(b) > 2 else 0
        return msg, channel, d1, d2

    def on_midi(self, me: MidiEvent) -> List[DomainEvent]:
        when = int(me.when_samples)
        msg, ch, d1, d2 = self._parse_midi(me.data)

        out: List[DomainEvent] = []

        NOTE_ON = 0x90
        NOTE_OFF = 0x80

        if msg == NOTE_ON and d2 > 0:
            pitch = int(d1)
            vel = int(d2)
            note_id = f"n_{uuid.uuid4().hex[:10]}"
            start = when
            end = when + self.default_note_len
            with self._lock:
                self._held[(ch, pitch)] = (note_id, start, vel, ch)
            out.append(NoteAdded(when, note_id, start, end, pitch, vel, ch, self.track_id))
            return out

        # treat NOTE_ON vel=0 as NOTE_OFF
        if msg == NOTE_OFF or (msg == NOTE_ON and d2 == 0):
            pitch = int(d1)
            with self._lock:
                held = self._held.pop((ch, pitch), None)
            if held:
                note_id, start, vel, ch2 = held
                out.append(NoteMoved(when, note_id, start=start, end=when, pitch=pitch))
            return out

        return out


# =============================================================================
# 6) Sequencer glue: ObserverHub -> Router -> Model -> Viewports
# =============================================================================

class SequencerGlue:
    """
    The thing you asked for: systematic glue that wires your existing UI + devices to the sequencer.

    Components:
      - engine: provides unified timebase + buffers
      - router: priority handler system (fast extension point)
      - model: timeline model proxy (replace with your own)
      - viewports: list of viewports to update

    Wiring:
      - Subscribes to ObserverHub topics.
      - On MIDI: translate to DomainEvents -> apply -> ChangeSet -> update viewports
      - On audio blocks: optional (waveform UI components etc.)
    """
    def __init__(self,
                 engine: Engine,
                 model: Optional[TimelineModelProxy] = None,
                 viewports: Optional[List[ViewportProxy]] = None):
        self.engine = engine
        self.model = model or TimelineModelProxy()
        self.viewports: List[ViewportProxy] = viewports or []
        self.router = PriorityEventRouter()
        self.midi_translate = MidiToSequencerObserver(track_id="track0")
        self._lock = threading.RLock()

        # Default routing handlers (priority can be tuned)
        self.router.add("midi", self._handle_midi, priority=100, handler_id="h_midi_to_model")
        self.router.add("domain", self._handle_domain_event, priority=90, handler_id="h_domain_apply")

        # Optional: audio block hooks for waveform UI
        self.router.add("audio_in", self._handle_audio_block, priority=10, handler_id="h_audio_in")
        self.router.add("audio_out", self._handle_audio_block, priority=10, handler_id="h_audio_out")
        self.router.add("system_audio", self._handle_audio_block, priority=10, handler_id="h_sys_audio")

        # Subscribe to ObserverHub
        hub = ObserverHub.instance()
        hub.subscribe("midi", lambda ev: self.router.dispatch("midi", ev))
        hub.subscribe("audio_in", lambda blk: self.router.dispatch("audio_in", blk))
        hub.subscribe("audio_out", lambda blk: self.router.dispatch("audio_out", blk))
        hub.subscribe("system_audio", lambda blk: self.router.dispatch("system_audio", blk))

    def add_viewport(self, vp: ViewportProxy) -> None:
        with self._lock:
            self.viewports.append(vp)

    # --------------------------
    # Router handlers
    # --------------------------
    def _handle_midi(self, me: MidiEvent) -> bool:
        # Translate raw MIDI into domain events
        domain_events = self.midi_translate.on_midi(me)
        for de in domain_events:
            self.router.dispatch("domain", de)
        return True  # consume MIDI at this stage by default

    def _handle_domain_event(self, de: DomainEvent) -> bool:
        cs = self.model.apply(de)
        self._apply_changeset_to_viewports(cs)
        return True

    def _handle_audio_block(self, blk: AudioBlock) -> bool:
        # Stub: if you want waveform UI nodes, you can store references or update textures here.
        # Keeping this as a no-op keeps realtime stable.
        return False  # don't consume; allow other handlers

    # --------------------------
    # Viewport update logic (projection)
    # --------------------------
    def _apply_changeset_to_viewports(self, cs: ChangeSet) -> None:
        # Each viewport maintains its own UI node cache; we just upsert/remove as needed.
        with self._lock:
            vps = list(self.viewports)

        # tempo changed: easiest is ask viewports to rebuild visible range; stubbed as redraw
        if "__tempo__" in cs.updated:
            for vp in vps:
                vp.request_redraw()

        # Apply removals first to avoid stale references
        for entity_id in cs.removed:
            for vp in vps:
                comp_id = vp.resolve_component_for_entity(entity_id)
                if comp_id:
                    vp.remove_component(comp_id)
                    vp.request_redraw()

        # Added + updated: re-project entity -> UI component
        for entity_id in cs.added + cs.updated:
            if entity_id == "__tempo__":
                continue
            ent = self.model.get_entity(entity_id)
            if ent is None:
                continue
            for vp in vps:
                comp = self._project_entity_to_component(ent, vp)
                vp.upsert_component(comp)
                vp.request_redraw()

    def _project_entity_to_component(self, ent: Any, vp: ViewportProxy) -> UIComponentProxy:
        """
        Projection rule:
          - NoteEntity -> rectangle in piano roll space
          - MarkerEntity -> vertical line/label

        Replace this projection with your real UI geometry contracts.
        """
        cam = vp.camera
        if hasattr(ent, "pitch"):  # NoteEntity-ish
            x0 = cam.x_from_samples(ent.start)
            x1 = cam.x_from_samples(ent.end)
            y = cam.y_from_pitch(ent.pitch)
            h = cam.pixels_per_pitch * 0.9
            comp_id = vp.resolve_component_for_entity(ent.id) or f"ui_{ent.id}"
            return UIComponentProxy(
                id=comp_id,
                entity_id=ent.id,
                kind="note",
                geom={"x": x0, "y": y, "w": max(2.0, x1 - x0), "h": h},
                style={"alpha": 0.9, "track_id": ent.track_id}
            )

        # MarkerEntity-ish
        if hasattr(ent, "when") and hasattr(ent, "label"):
            x = cam.x_from_samples(ent.when)
            comp_id = vp.resolve_component_for_entity(ent.id) or f"ui_{ent.id}"
            return UIComponentProxy(
                id=comp_id,
                entity_id=ent.id,
                kind="marker",
                geom={"x": x, "y": 0.0, "w": 2.0, "h": 9999.0, "label": ent.label},
                style={"alpha": 0.6}
            )

        return UIComponentProxy(id=f"ui_{uuid.uuid4().hex[:10]}", entity_id=str(getattr(ent, "id", "unknown")), kind="unknown")


# =============================================================================
# 7) Wiring helper (one function to bind your existing app)
# =============================================================================

@dataclass
class WiredSystem:
    engine: Engine
    glue: SequencerGlue
    model: TimelineModelProxy
    viewports: List[ViewportProxy]

def wire_up_system(engine: Engine,
                   viewports: Optional[List[ViewportProxy]] = None,
                   model: Optional[TimelineModelProxy] = None) -> WiredSystem:
    """
    One-call wiring:
      sys = wire_up_system(engine, [piano_roll, timeline, waveform])
    """
    glue = SequencerGlue(engine=engine, model=model, viewports=viewports or [])
    return WiredSystem(engine=engine, glue=glue, model=glue.model, viewports=glue.viewports)


# =============================================================================
# 8) Minimal test bench (headless)
# =============================================================================

def _demo_headless():
    """
    Demonstrates wiring without a GUI:
      - Create Engine
      - Create a ViewportAdapter (in-memory)
      - Wire glue
      - Simulate MIDI injection through ObserverHub
    """
    eng = Engine(sample_rate=44100, channels=2).enable_sim_audio().start(mix_tick_hz=400.0)
    vp = ViewportAdapter("piano")
    sys = wire_up_system(eng, [vp])

    # Subscribe to see midi events
    ObserverHub.instance().subscribe("midi", lambda ev: print("midi:", ev))
    ObserverHub.instance().subscribe("domain", lambda ev: print("domain:", ev))  # won't fire by default (router uses topic "domain" internally)

    # Manually inject midi bytes as if from a device: NoteOn 60 vel 100, then NoteOff
    eng.sync()
    now = eng.clock.get()
    ObserverHub.instance().on_midi_bytes(eng, bytes([0x90, 60, 100]), when_samples=now, port="demo")
    ObserverHub.instance().on_midi_bytes(eng, bytes([0x80, 60, 0]), when_samples=now + 22050, port="demo")

    # Check viewport has UI comps
    print("viewport components:", list(getattr(vp, "_comps", {}).values())[:2])

    eng.stop()


if __name__ == "__main__":
    _demo_headless()
