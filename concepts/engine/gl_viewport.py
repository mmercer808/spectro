from PyQt5.QtWidgets import QOpenGLWidget
from PyQt5.QtCore import Qt, pyqtSignal, QSize
from PyQt5.QtGui import QSurfaceFormat
from .render_engine import RenderEngine
from .scene_system import SceneManager

class GLViewport(QOpenGLWidget):
    """Bridge between PyQt and our custom rendering engine"""
    
    # Signal to notify when render engine is ready
    render_engine_ready = pyqtSignal(bool)  # True if successful, False if failed
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.render_engine = None
        self.scene_manager = None
        self.initialization_attempted = False
        self.initialization_successful = False
        self.setFocusPolicy(Qt.StrongFocus)  # Allow keyboard focus
        
        # Set minimum size to ensure we have a valid context
        self.setMinimumSize(QSize(100, 100))
        
        # Ensure we have a valid format
        format = QSurfaceFormat()
        format.setVersion(3, 3)
        format.setProfile(QSurfaceFormat.CoreProfile)
        format.setSamples(4)
        self.setFormat(format)
        
    def initializeGL(self):
        """Initialize OpenGL context and our custom render engine"""
        if self.initialization_attempted:
            return
            
        # Verify context is valid
        if not self.isValid():
            print("OpenGL context is not valid")
            self.render_engine_ready.emit(False)
            return
            
        # Verify we have a valid size
        if self.width() <= 0 or self.height() <= 0:
            print("Invalid viewport size")
            self.render_engine_ready.emit(False)
            return
            
        self.initialization_attempted = True
        try:
            # Log OpenGL context info
            format = self.format()
            print(f"OpenGL Version: {format.majorVersion()}.{format.minorVersion()}")
            print(f"OpenGL Profile: {'Core' if format.profile() == QSurfaceFormat.CoreProfile else 'Compatibility'}")
            
            self.render_engine = RenderEngine(self.width(), self.height())
            self.initialization_successful = self.render_engine.initialize()
            
            # Initialize scene manager
            if self.initialization_successful:
                self.scene_manager = SceneManager(self.render_engine, self.width(), self.height())
            
            if not self.initialization_successful:
                print("Failed to initialize render engine")
                self.render_engine = None
        except Exception as e:
            print(f"Error initializing render engine: {str(e)}")
            self.initialization_successful = False
            self.render_engine = None
            
        # Emit signal with initialization status
        self.render_engine_ready.emit(self.initialization_successful)
        
    def showEvent(self, event):
        """Handle widget show event
        
        Args:
            event: QShowEvent containing show information
        """
        super().showEvent(event)
        # Log when widget becomes visible
        print(f"GLViewport shown with size: {self.width()}x{self.height()}")
        
    def resizeEvent(self, event):
        """Handle widget resize event
        
        Args:
            event: QResizeEvent containing resize information
        """
        super().resizeEvent(event)
        # Log size changes
        print(f"GLViewport resized to: {self.width()}x{self.height()}")
        
    def resizeGL(self, width, height):
        """Handle viewport resize events
        
        Args:
            width: New viewport width
            height: New viewport height
        """
        if self.render_engine and self.initialization_successful:
            self.render_engine.resize(width, height)
        
    def paintGL(self):
        """Render a frame when requested by Qt
        This is the main bridge between Qt's render loop and our engine
        """
        if self.render_engine and self.initialization_successful:
            self.render_engine.render_frame()
            
            # Render current scene if available
            if self.scene_manager:
                self.scene_manager.render_current_scene()
    
    def get_render_engine(self):
        """Provide access to the render engine for scene management
        
        Returns:
            RenderEngine: The current render engine instance, or None if not initialized
        """
        return self.render_engine if self.initialization_successful else None
    
    def get_scene_manager(self):
        """Provide access to the scene manager
        
        Returns:
            SceneManager: The current scene manager instance, or None if not initialized
        """
        return self.scene_manager if self.initialization_successful else None
        
    def is_initialized(self):
        """Check if the render engine has been successfully initialized
        
        Returns:
            bool: True if render engine is initialized and ready
        """
        return (self.initialization_successful and 
                self.render_engine is not None and 
                self.isValid() and 
                self.width() > 0 and 
                self.height() > 0)
        
    def keyPressEvent(self, event):
        """Handle keyboard events
        
        Args:
            event: QKeyEvent containing key information
        """
        # For now, just pass through to parent
        super().keyPressEvent(event)
        
    def keyReleaseEvent(self, event):
        """Handle keyboard release events
        
        Args:
            event: QKeyEvent containing key information
        """
        # For now, just pass through to parent
        super().keyReleaseEvent(event) 