"""
Drum Sequencer Demo - Full Pipeline Integration

Demonstrates the complete flow:
    Onset Detection → Timeline → Tensor → Shader → Overlay → Events

Run with:
    python sequencer_demo.py

Keys:
    1-8: Toggle cells in current column
    SPACE: Play/Pause
    R: Reset playhead
    C: Clear grid
    L: Load example pattern
    S: Apply swing
"""

import sys
import time
import numpy as np

# Qt
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout
from PySide6.QtCore import QTimer, Qt
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtGui import QSurfaceFormat

# ModernGL
import moderngl

# Our modules
from sequence_tensor import SequenceTensor, RhythmSet
from grid_overlay import GridOverlay, GestureRouter
from grid_renderer import GridRenderer, GridColors


class SequencerGLWidget(QOpenGLWidget):
    """OpenGL widget that renders the sequencer grid."""
    
    def __init__(self, parent=None):
        # Request OpenGL 3.3 core
        fmt = QSurfaceFormat()
        fmt.setVersion(3, 3)
        fmt.setProfile(QSurfaceFormat.CoreProfile)
        QSurfaceFormat.setDefaultFormat(fmt)
        
        super().__init__(parent)
        
        # Will be initialized in initializeGL
        self.ctx = None
        self.grid = None
        self.renderer = None
        self.overlay = None
        
        # Transport
        self.playing = False
        self.playhead = 0.0
        self.bpm = 120.0
        self.last_time = time.perf_counter()
        
        # Row labels (drum names)
        self.row_names = [
            "Kick", "Snare", "HH Closed", "HH Open",
            "Tom Hi", "Tom Mid", "Tom Low", "Clap"
        ]
        
        # For audio feedback (print for now)
        self.sample_map = {
            0: "kick", 1: "snare", 2: "hihat_c", 3: "hihat_o",
            4: "tom_hi", 5: "tom_mid", 6: "tom_lo", 7: "clap"
        }
    
    def initializeGL(self):
        """Initialize OpenGL context and create renderer."""
        self.ctx = moderngl.create_context()
        
        # Create grid (8 rows, 16 columns)
        self.grid = SequenceTensor(rows=8, cols=16)
        
        # Create renderer
        self.renderer = GridRenderer(self.ctx, self.grid)
        
        # Create overlay (transparent Qt layer for gestures)
        self.overlay = GridOverlay(
            grid=self.grid,
            rows=8,
            cols=16,
            parent=self
        )
        self.overlay.setGeometry(self.rect())
        
        # Connect overlay signals
        self.overlay.cell_changed.connect(self._on_cell_changed)
        self.overlay.cell_hovered.connect(self._on_cell_hovered)
        
        # Load a default pattern
        self._load_example_pattern()
        
        print("OpenGL initialized")
        print(f"  Renderer: {self.ctx.info['GL_RENDERER']}")
        print(f"  Version: {self.ctx.info['GL_VERSION']}")
    
    def resizeGL(self, w, h):
        """Handle resize."""
        self.ctx.viewport = (0, 0, w, h)
        if self.overlay:
            self.overlay.setGeometry(self.rect())
    
    def paintGL(self):
        """Render frame."""
        # Update timing
        now = time.perf_counter()
        dt = now - self.last_time
        self.last_time = now
        
        # Update playhead if playing
        if self.playing:
            beats_per_second = self.bpm / 60.0
            self.playhead += dt * beats_per_second
            
            # Loop at 16 beats (pattern length)
            if self.playhead >= self.grid.cols:
                self.playhead -= self.grid.cols
        
        # Update renderer
        self.renderer.set_playhead(self.playhead, now)
        
        # Check for triggered events
        events = self.renderer.poll_events()
        for event in events:
            self._on_trigger(event)
        
        # Clear and render
        self.ctx.clear(0.08, 0.09, 0.11, 1.0)
        
        w, h = self.width(), self.height()
        self.renderer.render(
            x=0, y=0,
            width=w, height=h,
            window_size=(w, h)
        )
    
    def _on_cell_changed(self, row: int, col: int):
        """Called when overlay detects a cell toggle."""
        state = "ON" if self.grid.is_active(row, col) else "OFF"
        print(f"[EDIT] {self.row_names[row]} @ beat {col+1}: {state}")
        self.update()
    
    def _on_cell_hovered(self, row: int, col: int):
        """Called when mouse hovers over a cell."""
        if row >= 0 and col >= 0:
            # Could update status bar here
            pass
    
    def _on_trigger(self, event):
        """Called when playhead triggers a cell."""
        row = event.row
        col = event.col
        velocity = event.velocity
        sample = self.sample_map.get(row, "?")
        print(f"[TRIGGER] {self.row_names[row]} ({sample}) @ beat {col+1}, vel={velocity:.2f}")
    
    def _load_example_pattern(self):
        """Load a simple drum pattern."""
        # Clear
        self.grid.clear()
        
        # Kick on 1 and 3 (beats 0, 8)
        for col in [0, 8]:
            self.grid.set_active(0, col, True)
            self.grid.set_velocity(0, col, 1.0)
        
        # Snare on 2 and 4 (beats 4, 12)
        for col in [4, 12]:
            self.grid.set_active(1, col, True)
            self.grid.set_velocity(1, col, 0.9)
        
        # Hi-hat on every beat
        for col in range(16):
            self.grid.set_active(2, col, True)
            # Accent on quarter notes
            vel = 0.9 if col % 4 == 0 else 0.6
            self.grid.set_velocity(2, col, vel)
        
        # Clap with snare
        for col in [4, 12]:
            self.grid.set_active(7, col, True)
            self.grid.set_velocity(7, col, 0.7)
        
        print("[PATTERN] Loaded example drum pattern")
    
    def toggle_play(self):
        """Toggle playback."""
        self.playing = not self.playing
        state = "PLAYING" if self.playing else "STOPPED"
        print(f"[TRANSPORT] {state}")
    
    def reset_playhead(self):
        """Reset playhead to start."""
        self.playhead = 0.0
        print("[TRANSPORT] Reset to beat 1")
    
    def clear_grid(self):
        """Clear all cells."""
        self.grid.clear()
        print("[PATTERN] Grid cleared")
        self.update()
    
    def apply_swing(self, amount: float = 0.3):
        """Apply swing timing to off-beats."""
        for row in range(self.grid.rows):
            for col in range(self.grid.cols):
                if col % 2 == 1:  # Off-beats
                    self.grid.set_timing(row, col, amount)
        print(f"[PATTERN] Applied swing: {amount:.0%}")
        self.update()
    
    def keyPressEvent(self, event):
        """Handle keyboard input."""
        key = event.key()
        
        # Number keys 1-8: toggle cells at current column
        if Qt.Key_1 <= key <= Qt.Key_8:
            row = key - Qt.Key_1
            col = int(self.playhead) % self.grid.cols
            self.grid.toggle(row, col)
            if self.grid.is_active(row, col):
                self.grid.set_velocity(row, col, 0.8)
            self.update()
        
        elif key == Qt.Key_Space:
            self.toggle_play()
        
        elif key == Qt.Key_R:
            self.reset_playhead()
        
        elif key == Qt.Key_C:
            self.clear_grid()
        
        elif key == Qt.Key_L:
            self._load_example_pattern()
            self.update()
        
        elif key == Qt.Key_S:
            self.apply_swing()
        
        elif key == Qt.Key_Left:
            self.playhead = max(0, self.playhead - 1)
            self.update()
        
        elif key == Qt.Key_Right:
            self.playhead = (self.playhead + 1) % self.grid.cols
            self.update()
        
        elif key == Qt.Key_Escape:
            self.window().close()


class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SPECTRO Drum Sequencer")
        self.setMinimumSize(1024, 400)
        
        # Create central widget
        central = QWidget()
        self.setCentralWidget(central)
        
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create sequencer widget
        self.sequencer = SequencerGLWidget()
        layout.addWidget(self.sequencer)
        
        # Set focus so keyboard works
        self.sequencer.setFocus()
        
        # Start render timer (60 FPS)
        self.timer = QTimer()
        self.timer.timeout.connect(self.sequencer.update)
        self.timer.start(16)  # ~60 FPS
        
        self._print_help()
    
    def _print_help(self):
        print("\n" + "=" * 50)
        print("SPECTRO DRUM SEQUENCER")
        print("=" * 50)
        print("KEYBOARD:")
        print("  1-8     : Toggle row at current beat")
        print("  SPACE   : Play/Pause")
        print("  R       : Reset to beat 1")
        print("  C       : Clear pattern")
        print("  L       : Load example pattern")
        print("  S       : Apply swing")
        print("  ← →     : Step playhead")
        print("  ESC     : Quit")
        print("")
        print("MOUSE:")
        print("  Click   : Toggle cell")
        print("  Drag ↕  : Adjust velocity")
        print("  Drag ↔  : Adjust timing")
        print("=" * 50 + "\n")


def main():
    app = QApplication(sys.argv)
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
