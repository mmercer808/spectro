"""
GridOverlay - Invisible Qt widget layer for gesture capture.

The overlay sits on top of the shader-rendered grid, capturing gestures
and translating them into grid operations. The shader draws, Qt handles interaction.

Architecture:
    ┌─────────────────────────────────────┐
    │     GridOverlay (transparent Qt)    │  ← Captures gestures
    │  ┌───┬───┬───┬───┐                  │
    │  │ A │ B │ C │ D │  GestureCells    │
    │  ├───┼───┼───┼───┤                  │
    │  │ E │ F │ G │ H │                  │
    │  └───┴───┴───┴───┘                  │
    └─────────────────────────────────────┘
                    ↓
            GestureRouter → SignalBridge → SequenceTensor
                    ↑
    ┌─────────────────────────────────────┐
    │     Shader (visible rendering)      │  ← Draws from tensor
    └─────────────────────────────────────┘
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Optional, Callable, List, Dict, Set, Tuple
from enum import Enum, auto

from PySide6.QtWidgets import QWidget, QGridLayout
from PySide6.QtCore import Qt, Signal, QPoint, QPointF, QTimer
from PySide6.QtGui import QMouseEvent, QKeyEvent, QPainter, QColor

from sequence_tensor import SequenceTensor, Channel


class GestureType(Enum):
    """Types of gestures recognized by the overlay."""
    TAP = auto()              # Quick click
    LONG_PRESS = auto()       # Hold
    DRAG_HORIZONTAL = auto()  # Drag left/right (timing adjust)
    DRAG_VERTICAL = auto()    # Drag up/down (velocity adjust)
    DRAG_DIAGONAL = auto()    # Drag to another cell (move event)
    DOUBLE_TAP = auto()       # Quick double click
    RIGHT_CLICK = auto()      # Context menu
    SCROLL = auto()           # Mouse wheel


@dataclass
class Gesture:
    """A recognized gesture with context."""
    type: GestureType
    row: int
    col: int
    delta: QPointF = field(default_factory=lambda: QPointF(0, 0))
    duration: float = 0.0
    modifiers: int = 0  # Qt.KeyboardModifier flags
    
    # For drag gestures
    start_row: int = -1
    start_col: int = -1
    
    @property
    def shift_held(self) -> bool:
        return bool(self.modifiers & Qt.ShiftModifier)
    
    @property
    def ctrl_held(self) -> bool:
        return bool(self.modifiers & Qt.ControlModifier)
    
    @property
    def alt_held(self) -> bool:
        return bool(self.modifiers & Qt.AltModifier)


class GestureCell(QWidget):
    """
    Invisible gesture surface for one grid position.
    
    Captures mouse events and interprets them as musical gestures.
    Emits signals that the GestureRouter consumes.
    """
    
    # Signals
    gesture_detected = Signal(Gesture)
    hover_entered = Signal(int, int)  # row, col
    hover_left = Signal(int, int)
    
    # Configuration
    DRAG_THRESHOLD = 5  # pixels
    LONG_PRESS_MS = 300
    DOUBLE_TAP_MS = 300
    
    def __init__(self, row: int, col: int, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.row = row
        self.col = col
        
        # Make transparent but capture events
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        self.setMouseTracking(True)
        
        # Gesture state
        self._press_pos: Optional[QPointF] = None
        self._press_time: float = 0.0
        self._last_tap_time: float = 0.0
        self._is_dragging = False
        self._drag_start_cell: Optional[Tuple[int, int]] = None
        
        # Long press timer
        self._long_press_timer = QTimer(self)
        self._long_press_timer.setSingleShot(True)
        self._long_press_timer.timeout.connect(self._on_long_press)
    
    def paintEvent(self, event):
        """Optionally draw debug overlay."""
        # Uncomment for debugging:
        # painter = QPainter(self)
        # painter.setPen(QColor(255, 0, 0, 30))
        # painter.drawRect(self.rect().adjusted(0, 0, -1, -1))
        pass
    
    def enterEvent(self, event):
        self.hover_entered.emit(self.row, self.col)
    
    def leaveEvent(self, event):
        self.hover_left.emit(self.row, self.col)
        self._cancel_gesture()
    
    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self._press_pos = event.position()
            self._press_time = time.perf_counter()
            self._is_dragging = False
            self._drag_start_cell = (self.row, self.col)
            
            # Start long press timer
            self._long_press_timer.start(self.LONG_PRESS_MS)
            
        elif event.button() == Qt.RightButton:
            gesture = Gesture(
                type=GestureType.RIGHT_CLICK,
                row=self.row,
                col=self.col,
                modifiers=int(event.modifiers())
            )
            self.gesture_detected.emit(gesture)
    
    def mouseMoveEvent(self, event: QMouseEvent):
        if self._press_pos is None:
            return
        
        delta = event.position() - self._press_pos
        
        if delta.manhattanLength() > self.DRAG_THRESHOLD:
            self._is_dragging = True
            self._long_press_timer.stop()
    
    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() != Qt.LeftButton or self._press_pos is None:
            return
        
        self._long_press_timer.stop()
        
        delta = event.position() - self._press_pos
        duration = time.perf_counter() - self._press_time
        modifiers = int(event.modifiers())
        
        # Determine gesture type
        if delta.manhattanLength() < self.DRAG_THRESHOLD:
            # Not a drag - check for double tap
            now = time.perf_counter()
            if now - self._last_tap_time < self.DOUBLE_TAP_MS / 1000:
                gesture_type = GestureType.DOUBLE_TAP
            elif duration >= self.LONG_PRESS_MS / 1000:
                gesture_type = GestureType.LONG_PRESS
            else:
                gesture_type = GestureType.TAP
            self._last_tap_time = now
        else:
            # Drag gesture
            if abs(delta.x()) > abs(delta.y()) * 2:
                gesture_type = GestureType.DRAG_HORIZONTAL
            elif abs(delta.y()) > abs(delta.x()) * 2:
                gesture_type = GestureType.DRAG_VERTICAL
            else:
                gesture_type = GestureType.DRAG_DIAGONAL
        
        gesture = Gesture(
            type=gesture_type,
            row=self.row,
            col=self.col,
            delta=delta,
            duration=duration,
            modifiers=modifiers,
            start_row=self._drag_start_cell[0] if self._drag_start_cell else self.row,
            start_col=self._drag_start_cell[1] if self._drag_start_cell else self.col
        )
        
        self.gesture_detected.emit(gesture)
        self._reset_state()
    
    def wheelEvent(self, event):
        delta = QPointF(event.angleDelta().x(), event.angleDelta().y())
        gesture = Gesture(
            type=GestureType.SCROLL,
            row=self.row,
            col=self.col,
            delta=delta,
            modifiers=int(event.modifiers())
        )
        self.gesture_detected.emit(gesture)
    
    def _on_long_press(self):
        """Called when long press timer fires."""
        if self._press_pos is not None and not self._is_dragging:
            gesture = Gesture(
                type=GestureType.LONG_PRESS,
                row=self.row,
                col=self.col,
                duration=self.LONG_PRESS_MS / 1000
            )
            self.gesture_detected.emit(gesture)
    
    def _cancel_gesture(self):
        self._long_press_timer.stop()
        self._reset_state()
    
    def _reset_state(self):
        self._press_pos = None
        self._is_dragging = False
        self._drag_start_cell = None


class GestureRouter:
    """
    Routes gestures to appropriate handlers.
    
    Translates UI gestures into grid operations:
        TAP → Toggle cell
        DOUBLE_TAP → Edit detail
        DRAG_VERTICAL → Adjust velocity
        DRAG_HORIZONTAL → Adjust timing
        LONG_PRESS → Open menu
        etc.
    """
    
    def __init__(self, grid: SequenceTensor, signal_bridge=None):
        self.grid = grid
        self.signal_bridge = signal_bridge
        
        # Custom handlers (can be overridden)
        self.handlers: Dict[GestureType, Callable[[Gesture], None]] = {
            GestureType.TAP: self._handle_tap,
            GestureType.DOUBLE_TAP: self._handle_double_tap,
            GestureType.LONG_PRESS: self._handle_long_press,
            GestureType.DRAG_VERTICAL: self._handle_drag_vertical,
            GestureType.DRAG_HORIZONTAL: self._handle_drag_horizontal,
            GestureType.DRAG_DIAGONAL: self._handle_drag_diagonal,
            GestureType.RIGHT_CLICK: self._handle_right_click,
            GestureType.SCROLL: self._handle_scroll,
        }
        
        # Callback for external notification
        self.on_cell_changed: Optional[Callable[[int, int], None]] = None
        self.on_velocity_changed: Optional[Callable[[int, int, float], None]] = None
        self.on_timing_changed: Optional[Callable[[int, int, float], None]] = None
        self.on_detail_requested: Optional[Callable[[int, int], None]] = None
    
    def handle(self, gesture: Gesture):
        """Route gesture to appropriate handler."""
        handler = self.handlers.get(gesture.type)
        if handler:
            handler(gesture)
    
    def _emit(self, signal_name: str, **kwargs):
        """Emit signal through bridge if available."""
        if self.signal_bridge:
            self.signal_bridge.emit(signal_name, **kwargs)
    
    def _handle_tap(self, gesture: Gesture):
        """Toggle cell on tap."""
        row, col = gesture.row, gesture.col
        
        if gesture.shift_held:
            # Shift+click: extend selection or special behavior
            pass
        else:
            # Normal click: toggle
            new_state = self.grid.toggle(row, col)
            
            # Set default velocity if activated
            if new_state:
                self.grid.set_velocity(row, col, 0.8)
            
            self._emit('seq.cell_toggled', row=row, col=col, active=new_state)
            
            if self.on_cell_changed:
                self.on_cell_changed(row, col)
    
    def _handle_double_tap(self, gesture: Gesture):
        """Open detail editor on double tap."""
        row, col = gesture.row, gesture.col
        
        self._emit('seq.edit_detail', row=row, col=col)
        
        if self.on_detail_requested:
            self.on_detail_requested(row, col)
    
    def _handle_long_press(self, gesture: Gesture):
        """Open context menu or subdivision editor."""
        row, col = gesture.row, gesture.col
        
        # Toggle subdivision mode
        current_subdiv = self.grid.get_subdivision_level(row, col)
        new_subdiv = 4 if current_subdiv == 1 else 1
        self.grid.set_subdivision(row, col, new_subdiv)
        
        self._emit('seq.subdivision_changed', row=row, col=col, level=new_subdiv)
    
    def _handle_drag_vertical(self, gesture: Gesture):
        """Adjust velocity with vertical drag."""
        row, col = gesture.row, gesture.col
        
        # Drag up = louder, drag down = softer
        delta_v = -gesture.delta.y() / 100.0
        current = self.grid.get_velocity(row, col)
        new_velocity = max(0.0, min(1.0, current + delta_v))
        
        self.grid.set_velocity(row, col, new_velocity)
        
        self._emit('seq.velocity_changed', row=row, col=col, velocity=new_velocity)
        
        if self.on_velocity_changed:
            self.on_velocity_changed(row, col, new_velocity)
    
    def _handle_drag_horizontal(self, gesture: Gesture):
        """Adjust micro-timing with horizontal drag."""
        row, col = gesture.row, gesture.col
        
        # Drag right = late, drag left = early
        delta_t = gesture.delta.x() / 50.0
        current = self.grid.get_timing(row, col)
        new_timing = max(-1.0, min(1.0, current + delta_t))
        
        self.grid.set_timing(row, col, new_timing)
        
        self._emit('seq.timing_changed', row=row, col=col, timing=new_timing)
        
        if self.on_timing_changed:
            self.on_timing_changed(row, col, new_timing)
    
    def _handle_drag_diagonal(self, gesture: Gesture):
        """Move event to different cell."""
        from_row, from_col = gesture.start_row, gesture.start_col
        to_row, to_col = gesture.row, gesture.col
        
        if (from_row, from_col) != (to_row, to_col):
            # Check if source is active
            if self.grid.is_active(from_row, from_col):
                # Copy properties
                velocity = self.grid.get_velocity(from_row, from_col)
                timing = self.grid.get_timing(from_row, from_col)
                
                # Move
                self.grid.set_active(from_row, from_col, False)
                self.grid.set_active(to_row, to_col, True)
                self.grid.set_velocity(to_row, to_col, velocity)
                self.grid.set_timing(to_row, to_col, timing)
                
                self._emit('seq.event_moved',
                          from_row=from_row, from_col=from_col,
                          to_row=to_row, to_col=to_col)
    
    def _handle_right_click(self, gesture: Gesture):
        """Open context menu."""
        row, col = gesture.row, gesture.col
        self._emit('seq.context_menu', row=row, col=col)
    
    def _handle_scroll(self, gesture: Gesture):
        """Handle scroll wheel - could zoom or adjust subdivision."""
        row, col = gesture.row, gesture.col
        
        if gesture.ctrl_held:
            # Ctrl+scroll: adjust subdivision
            current = self.grid.get_subdivision_level(row, col)
            delta = 1 if gesture.delta.y() > 0 else -1
            new_level = max(1, min(16, current + delta))
            
            if self.grid.is_active(row, col):
                self.grid.set_subdivision(row, col, new_level)
                self._emit('seq.subdivision_changed', row=row, col=col, level=new_level)


class GridOverlay(QWidget):
    """
    Main overlay widget containing the gesture cell grid.
    
    Usage:
        overlay = GridOverlay(grid=sequence_tensor, rows=8, cols=16)
        overlay.setParent(panel_widget)  # Overlay the shader panel
        overlay.show()
    """
    
    # Signals for external listeners
    cell_changed = Signal(int, int)  # row, col
    cell_hovered = Signal(int, int)  # row, col (-1, -1 for none)
    
    def __init__(
        self,
        grid: SequenceTensor,
        rows: int = 8,
        cols: int = 16,
        signal_bridge=None,
        parent: Optional[QWidget] = None
    ):
        super().__init__(parent)
        
        self.grid = grid
        self.rows = rows
        self.cols = cols
        
        # Create router
        self.router = GestureRouter(grid, signal_bridge)
        self.router.on_cell_changed = lambda r, c: self.cell_changed.emit(r, c)
        
        # Make transparent
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setStyleSheet("background: transparent;")
        
        # Create cell grid
        self._cells: List[List[GestureCell]] = []
        self._setup_cells()
        
        # Hover tracking
        self._hovered_cell: Optional[Tuple[int, int]] = None
    
    def _setup_cells(self):
        """Create the grid of gesture cells."""
        layout = QGridLayout(self)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        
        for row in range(self.rows):
            row_cells = []
            for col in range(self.cols):
                cell = GestureCell(row, col, self)
                cell.gesture_detected.connect(self._on_gesture)
                cell.hover_entered.connect(self._on_hover_enter)
                cell.hover_left.connect(self._on_hover_leave)
                
                layout.addWidget(cell, row, col)
                row_cells.append(cell)
            
            self._cells.append(row_cells)
    
    def _on_gesture(self, gesture: Gesture):
        """Route gesture through router."""
        self.router.handle(gesture)
    
    def _on_hover_enter(self, row: int, col: int):
        self._hovered_cell = (row, col)
        self.cell_hovered.emit(row, col)
    
    def _on_hover_leave(self, row: int, col: int):
        if self._hovered_cell == (row, col):
            self._hovered_cell = None
            self.cell_hovered.emit(-1, -1)
    
    def get_cell(self, row: int, col: int) -> Optional[GestureCell]:
        """Get cell widget at position."""
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return self._cells[row][col]
        return None
    
    def set_grid(self, grid: SequenceTensor):
        """Update the underlying grid."""
        self.grid = grid
        self.router.grid = grid
    
    def resize_grid(self, rows: int, cols: int):
        """Resize the overlay grid (requires rebuilding cells)."""
        if rows == self.rows and cols == self.cols:
            return
        
        # Clear existing
        for row_cells in self._cells:
            for cell in row_cells:
                cell.deleteLater()
        self._cells.clear()
        
        # Rebuild
        self.rows = rows
        self.cols = cols
        self._setup_cells()


# =============================================================================
# SHADER BRIDGE
# =============================================================================

class GridShaderBridge:
    """
    Synchronizes SequenceTensor with GPU textures/SSBOs.
    
    Handles:
    - Uploading tensor state to GPU texture
    - Downloading shader events from SSBO
    - Managing dirty regions for efficient updates
    """
    
    def __init__(self, grid: SequenceTensor, ctx):
        """
        Args:
            grid: The SequenceTensor to sync
            ctx: ModernGL context
        """
        self.grid = grid
        self.ctx = ctx
        
        # Create texture for grid state
        # RGBA float texture: (cols, rows) with 4 components
        self.texture = ctx.texture(
            size=(grid.cols, grid.rows),
            components=4,
            dtype='f4'
        )
        self.texture.filter = (ctx.NEAREST, ctx.NEAREST)
        
        # Create SSBO for event queue (if needed)
        # Layout: [write_head (4 bytes)] [events (capacity * 24 bytes)]
        self.event_capacity = 64
        self.event_buffer = ctx.buffer(reserve=4 + self.event_capacity * 24)
        
        # Initial upload
        self.upload()
    
    def upload(self):
        """Upload tensor state to GPU texture."""
        data = self.grid.to_rgba_bytes()
        self.texture.write(data)
        self.grid.clear_dirty()
    
    def upload_if_dirty(self):
        """Upload only if tensor has changed."""
        if self.grid.is_dirty():
            self.upload()
    
    def download_events(self) -> List[dict]:
        """
        Download and parse events from shader SSBO.
        
        Returns list of event dicts with keys:
            type, row, col, data0, data1, timestamp
        """
        data = self.event_buffer.read()
        
        # Parse write head
        import struct
        write_head = struct.unpack('i', data[:4])[0]
        
        if write_head == 0:
            return []
        
        events = []
        offset = 4
        
        for i in range(min(write_head, self.event_capacity)):
            event_data = struct.unpack('6f', data[offset:offset + 24])
            events.append({
                'type': int(event_data[0]),
                'row': int(event_data[1]),
                'col': int(event_data[2]),
                'data0': event_data[3],
                'data1': event_data[4],
                'timestamp': event_data[5]
            })
            offset += 24
        
        # Clear the buffer (reset write head)
        self.event_buffer.write(struct.pack('i', 0))
        
        return events
    
    def bind_texture(self, location: int = 0):
        """Bind grid texture to texture unit."""
        self.texture.use(location)
    
    def bind_event_buffer(self, binding: int = 0):
        """Bind event SSBO to binding point."""
        self.event_buffer.bind_to_storage_buffer(binding)
    
    def resize(self, rows: int, cols: int):
        """Resize texture for new grid dimensions."""
        if rows != self.grid.rows or cols != self.grid.cols:
            self.texture.release()
            self.texture = self.ctx.texture(
                size=(cols, rows),
                components=4,
                dtype='f4'
            )
            self.texture.filter = (self.ctx.NEAREST, self.ctx.NEAREST)
