import numpy as np
import math

# Mock OpenGL if not available
try:
    import OpenGL.GL as gl
    from OpenGL.GL import shaders
    OPENGL_AVAILABLE = True
except ImportError:
    print("OpenGL not available, using mock implementation")
    OPENGL_AVAILABLE = False
    
    # Create mock classes and functions
    class MockGL:
        def __getattr__(self, name):
            return lambda *args, **kwargs: None
    
    gl = MockGL()
    
    class MockShaders:
        def compileShader(self, *args, **kwargs):
            return 0
        
        def compileProgram(self, *args, **kwargs):
            return 0
    
    shaders = MockShaders()

class Renderable:
    """Base class for any object that can be rendered"""
    
    def __init__(self, object_id):
        self.object_id = object_id
        self.visible = True
        self.initialized = False
    
    def initialize(self):
        """Initialize any OpenGL resources needed by this object"""
        self.initialized = True
        return True
    
    def render(self):
        """Render the object using OpenGL"""
        pass
    
    def cleanup(self):
        """Clean up any OpenGL resources used by this object"""
        pass
    
    def update(self, dt):
        """Update the object's state
        
        Args:
            dt: Time elapsed since last update in seconds
        """
        pass


class RenderEngine:
    """Core OpenGL rendering engine, decoupled from Qt"""
    
    def __init__(self, width=800, height=600):
        """Initialize the rendering engine
        
        Args:
            width: Initial viewport width
            height: Initial viewport height
        """
        self.width = width
        self.height = height
        self.renderables = {}  # Dict of renderable objects by ID
        self.initialized = False
        self.default_shader = None
        
        # Basic view transformation
        self.view_matrix = np.identity(4, dtype=np.float32)
        self.proj_matrix = np.identity(4, dtype=np.float32)
    
    def initialize(self):
        """Initialize the OpenGL context and resources"""
        if self.initialized:
            return True
            
        try:
            if OPENGL_AVAILABLE:
                # Print OpenGL info for debugging
                print(f"OpenGL Version: {gl.glGetString(gl.GL_VERSION).decode('utf-8')}")
                print(f"GLSL Version: {gl.glGetString(gl.GL_SHADING_LANGUAGE_VERSION).decode('utf-8')}")
                print(f"Vendor: {gl.glGetString(gl.GL_VENDOR).decode('utf-8')}")
                print(f"Renderer: {gl.glGetString(gl.GL_RENDERER).decode('utf-8')}")
                
                # Basic OpenGL setup
                gl.glClearColor(0.1, 0.1, 0.1, 1.0)
                gl.glEnable(gl.GL_BLEND)
                gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
                
                # Create default shader program
                self._create_default_shader()
            else:
                print("Using mock OpenGL implementation")
            
            # Set up orthographic projection for 2D rendering
            self._update_projection()
            
            self.initialized = True
            return True
        
        except Exception as e:
            print(f"Failed to initialize render engine: {e}")
            return False
    
    def _create_default_shader(self):
        """Create a basic shader program for 2D rendering"""
        if not OPENGL_AVAILABLE:
            return
            
        vertex_shader = """
        #version 330 core
        layout(location = 0) in vec3 position;
        layout(location = 1) in vec4 color;
        
        out vec4 vertexColor;
        
        void main() {
            gl_Position = vec4(position, 1.0);
            vertexColor = color;
        }
        """
        
        fragment_shader = """
        #version 330 core
        in vec4 vertexColor;
        out vec4 fragColor;
        
        void main() {
            fragColor = vertexColor;
        }
        """
        
        try:
            vertex = shaders.compileShader(vertex_shader, gl.GL_VERTEX_SHADER)
            fragment = shaders.compileShader(fragment_shader, gl.GL_FRAGMENT_SHADER)
            self.default_shader = shaders.compileProgram(vertex, fragment)
            print(f"✅ Default shader compiled successfully: {self.default_shader}")
        except Exception as e:
            print(f"❌ Failed to compile default shader: {e}")
            self.default_shader = None
    
    def _update_projection(self):
        """Update the projection matrix based on viewport dimensions"""
        # Orthographic projection for 2D rendering
        # Maps (0,0) to top-left and (width,height) to bottom-right
        left, right = 0, self.width
        bottom, top = self.height, 0  # Inverted Y for top-left origin
        near, far = -1, 1
        
        self.proj_matrix = np.array([
            [2/(right-left), 0, 0, -(right+left)/(right-left)],
            [0, 2/(top-bottom), 0, -(top+bottom)/(top-bottom)],
            [0, 0, -2/(far-near), -(far+near)/(far-near)],
            [0, 0, 0, 1]
        ], dtype=np.float32)
    
    def resize(self, width, height):
        """Handle viewport resize
        
        Args:
            width: New viewport width
            height: New viewport height
        """
        self.width = width
        self.height = height
        if OPENGL_AVAILABLE:
            gl.glViewport(0, 0, width, height)
        self._update_projection()
    
    def add_renderable(self, renderable):
        """Add a renderable object to the scene
        
        Args:
            renderable: Object implementing the Renderable interface
        """
        # Give the renderable access to the render engine for shader access
        renderable.render_engine = self
        
        if not renderable.initialized:
            renderable.initialize()
        
        self.renderables[renderable.object_id] = renderable
    
    def remove_renderable(self, object_id):
        """Remove a renderable from the scene by ID
        
        Args:
            object_id: ID of the renderable to remove
        """
        if object_id in self.renderables:
            self.renderables[object_id].cleanup()
            del self.renderables[object_id]
    
    def get_renderable(self, object_id):
        """Get a renderable by ID
        
        Args:
            object_id: ID of the renderable to get
            
        Returns:
            The renderable object or None if not found
        """
        return self.renderables.get(object_id)
    
    def render_frame(self):
        """Render a complete frame with all scene objects"""
        if not self.initialized:
            return
        
        # Clear the screen
        if OPENGL_AVAILABLE:
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        
        # Try shader-based rendering first, fallback to basic rendering
        try:
            # Render all visible objects
            print(f"🔍 Rendering {len(self.renderables)} objects with shaders...")
            for obj_id, obj in self.renderables.items():
                if obj.visible:
                    try:
                        obj.render()
                    except Exception as e:
                        print(f"❌ Error rendering {obj_id}: {e}")
                        # Fallback to basic rendering for this object
                        self._render_basic(obj)
        except Exception as e:
            print(f"❌ Shader rendering failed, using basic rendering: {e}")
            # Fallback to basic rendering for all objects
            for obj_id, obj in self.renderables.items():
                if obj.visible:
                    self._render_basic(obj)
    
    def _render_basic(self, obj):
        """Basic rendering fallback using simple VBO approach"""
        if not OPENGL_AVAILABLE:
            return
            
        try:
            # Use a very simple approach that should work with Core Profile
            if hasattr(obj, 'x') and hasattr(obj, 'y') and hasattr(obj, 'width') and hasattr(obj, 'height'):
                # Create simple vertex data (just positions, no colors)
                vertices = np.array([
                    obj.x, obj.y, 0.0,
                    obj.x + obj.width, obj.y, 0.0,
                    obj.x, obj.y + obj.height, 0.0,
                    obj.x + obj.width, obj.y, 0.0,
                    obj.x + obj.width, obj.y + obj.height, 0.0,
                    obj.x, obj.y + obj.height, 0.0,
                ], dtype=np.float32)
                
                # Create VBO
                vbo = gl.glGenBuffers(1)
                gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
                gl.glBufferData(gl.GL_ARRAY_BUFFER, vertices.nbytes, vertices, gl.GL_STATIC_DRAW)
                
                # Use a very simple shader or fixed function
                gl.glUseProgram(0)  # Use fixed function pipeline
                
                # Set color if available
                if hasattr(obj, 'color'):
                    gl.glColor4f(*obj.color)
                else:
                    gl.glColor4f(1.0, 1.0, 1.0, 1.0)
                
                # Enable vertex array
                gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
                gl.glVertexPointer(3, gl.GL_FLOAT, 0, None)
                
                # Draw
                gl.glDrawArrays(gl.GL_TRIANGLES, 0, 6)
                
                # Clean up
                gl.glDisableClientState(gl.GL_VERTEX_ARRAY)
                gl.glDeleteBuffers(1, [vbo])
                gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
                
                print(f"✅ Basic VBO rendering for {obj.object_id}")
                
        except Exception as e:
            print(f"❌ Basic VBO rendering failed for {obj.object_id}: {e}")
            # Last resort: just print that we can't render
            print(f"⚠️  Cannot render {obj.object_id} - skipping")
    
    def update(self, dt):
        """Update all renderables
        
        Args:
            dt: Time elapsed since last update in seconds
        """
        for obj_id, obj in self.renderables.items():
            obj.update(dt)
    
    def cleanup(self):
        """Clean up all rendering resources"""
        for obj_id, obj in list(self.renderables.items()):
            obj.cleanup()
        
        self.renderables.clear()
        
        if OPENGL_AVAILABLE and self.default_shader:
            gl.glDeleteProgram(self.default_shader)


# Concrete renderable implementations

class Rectangle(Renderable):
    """A simple rectangular renderable object"""
    
    def __init__(self, object_id, x, y, width, height, color=(1.0, 1.0, 1.0, 1.0)):
        """Initialize a rectangle
        
        Args:
            object_id: Unique identifier
            x, y: Position (top-left corner)
            width, height: Dimensions
            color: RGBA color tuple
        """
        super().__init__(object_id)
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.color = color
        self.vao = None
        self.vbo = None
    
    def initialize(self):
        """Initialize OpenGL resources for this rectangle"""
        if not OPENGL_AVAILABLE:
            return super().initialize()
            
        # For the simple VBO approach, we don't need to pre-create VAOs
        # The render method creates VBOs on-the-fly
        # Just mark as initialized
        return super().initialize()
    
    def render(self):
        """Render the rectangle"""
        if not self.initialized or not OPENGL_AVAILABLE:
            return
            
        # Use the same simple VBO approach as the fallback method
        try:
            # Create simple vertex data (just positions, no colors)
            vertices = np.array([
                self.x, self.y, 0.0,
                self.x + self.width, self.y, 0.0,
                self.x, self.y + self.height, 0.0,
                self.x + self.width, self.y, 0.0,
                self.x + self.width, self.y + self.height, 0.0,
                self.x, self.y + self.height, 0.0,
            ], dtype=np.float32)
            
            # Create VBO
            vbo = gl.glGenBuffers(1)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
            gl.glBufferData(gl.GL_ARRAY_BUFFER, vertices.nbytes, vertices, gl.GL_STATIC_DRAW)
            
            # Use fixed function pipeline
            gl.glUseProgram(0)
            
            # Set color
            gl.glColor4f(*self.color)
            
            # Enable vertex array
            gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
            gl.glVertexPointer(3, gl.GL_FLOAT, 0, None)
            
            # Draw
            gl.glDrawArrays(gl.GL_TRIANGLES, 0, 6)
            
            # Clean up
            gl.glDisableClientState(gl.GL_VERTEX_ARRAY)
            gl.glDeleteBuffers(1, [vbo])
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
            
        except Exception as e:
            print(f"❌ Rectangle rendering failed for {self.object_id}: {e}")
            # Last resort: just print that we can't render
            print(f"⚠️  Cannot render rectangle {self.object_id} - skipping")
    
    def cleanup(self):
        """Clean up OpenGL resources"""
        if not OPENGL_AVAILABLE:
            return super().cleanup()
            
        if self.vbo:
            gl.glDeleteBuffers(1, [self.vbo])
            self.vbo = None
        
        if self.vao:
            gl.glDeleteVertexArrays(1, [self.vao])
            self.vao = None
        
        super().cleanup()


class Grid(Renderable):
    """A grid of lines for the sequencer view"""
    
    def __init__(self, object_id, x, y, width, height, 
                 rows=16, cols=16, 
                 major_color=(0.5, 0.5, 0.5, 1.0),
                 minor_color=(0.3, 0.3, 0.3, 1.0)):
        """Initialize a grid
        
        Args:
            object_id: Unique identifier
            x, y: Position (top-left corner)
            width, height: Dimensions
            rows, cols: Number of rows and columns
            major_color: Color for major grid lines
            minor_color: Color for minor grid lines
        """
        super().__init__(object_id)
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.rows = rows
        self.cols = cols
        self.major_color = major_color
        self.minor_color = minor_color
        self.vao = None
        self.vbo = None
    
    def initialize(self):
        """Initialize OpenGL resources for this grid"""
        if not OPENGL_AVAILABLE:
            return super().initialize()
            
        # For the simple VBO approach, we don't need to pre-create VAOs
        # The render method creates VBOs on-the-fly
        # Just mark as initialized
        return super().initialize()
    
    def render(self):
        """Render the grid"""
        if not self.initialized or not OPENGL_AVAILABLE:
            return
            
        # Use simple VBO-based rendering for grid lines
        try:
            # Calculate number of vertices needed
            num_vertices = 2 * (self.rows + 1 + self.cols + 1)
            vertices = np.zeros(num_vertices * 3, dtype=np.float32)  # 3 floats per vertex (position only)
            
            # Fill vertex array with horizontal lines
            idx = 0
            row_height = self.height / self.rows
            for i in range(self.rows + 1):
                y_pos = self.y + i * row_height
                
                # Left vertex
                vertices[idx:idx+3] = [self.x, y_pos, 0.0]
                idx += 3
                
                # Right vertex
                vertices[idx:idx+3] = [self.x + self.width, y_pos, 0.0]
                idx += 3
            
            # Fill vertex array with vertical lines
            col_width = self.width / self.cols
            for i in range(self.cols + 1):
                x_pos = self.x + i * col_width
                
                # Bottom vertex
                vertices[idx:idx+3] = [x_pos, self.y, 0.0]
                idx += 3
                
                # Top vertex
                vertices[idx:idx+3] = [x_pos, self.y + self.height, 0.0]
                idx += 3
            
            # Create VBO
            vbo = gl.glGenBuffers(1)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
            gl.glBufferData(gl.GL_ARRAY_BUFFER, vertices.nbytes, vertices, gl.GL_STATIC_DRAW)
            
            # Use fixed function pipeline
            gl.glUseProgram(0)
            
            # Set color (use major color for all lines for simplicity)
            gl.glColor4f(*self.major_color)
            
            # Enable vertex array
            gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
            gl.glVertexPointer(3, gl.GL_FLOAT, 0, None)
            
            # Draw
            gl.glDrawArrays(gl.GL_LINES, 0, num_vertices)
            
            # Clean up
            gl.glDisableClientState(gl.GL_VERTEX_ARRAY)
            gl.glDeleteBuffers(1, [vbo])
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
            
        except Exception as e:
            print(f"❌ Grid rendering failed for {self.object_id}: {e}")
            # Last resort: just print that we can't render
            print(f"⚠️  Cannot render grid {self.object_id} - skipping")
    
    def cleanup(self):
        """Clean up OpenGL resources"""
        if not OPENGL_AVAILABLE:
            return super().cleanup()
            
        if self.vbo:
            gl.glDeleteBuffers(1, [self.vbo])
            self.vbo = None
        
        if self.vao:
            gl.glDeleteVertexArrays(1, [self.vao])
            self.vao = None
        
        super().cleanup()


class Text(Renderable):
    """Text rendering placeholder
    
    Note: In a full implementation, this would use texture-based
    text rendering with a font atlas. For this prototype, we're
    keeping it as a placeholder.
    """
    
    def __init__(self, object_id, x, y, text, color=(1.0, 1.0, 1.0, 1.0), size=12):
        """Initialize text object
        
        Args:
            object_id: Unique identifier
            x, y: Position
            text: Text to display
            color: RGBA color tuple
            size: Font size
        """
        super().__init__(object_id)
        self.x = x
        self.y = y
        self.text = text
        self.color = color
        self.size = size
        
        # Calculate width and height for rendering
        # For now, use a simple estimation based on text length and font size
        self.width = len(text) * size * 0.6  # Approximate character width
        self.height = size  # Font size as height
    
    def initialize(self):
        """Initialize text rendering resources"""
        # In a full implementation, this would load fonts and create textures
        return super().initialize()
    
    def render(self):
        """Render the text as a colored rectangle placeholder"""
        if not self.initialized or not OPENGL_AVAILABLE:
            return
            
        # Use the same simple VBO approach as the fallback method
        try:
            # Create simple vertex data (just positions, no colors)
            vertices = np.array([
                self.x, self.y, 0.0,
                self.x + self.width, self.y, 0.0,
                self.x, self.y + self.height, 0.0,
                self.x + self.width, self.y, 0.0,
                self.x + self.width, self.y + self.height, 0.0,
                self.x, self.y + self.height, 0.0,
            ], dtype=np.float32)
            
            # Create VBO
            vbo = gl.glGenBuffers(1)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
            gl.glBufferData(gl.GL_ARRAY_BUFFER, vertices.nbytes, vertices, gl.GL_STATIC_DRAW)
            
            # Use fixed function pipeline
            gl.glUseProgram(0)
            
            # Set color
            gl.glColor4f(*self.color)
            
            # Enable vertex array
            gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
            gl.glVertexPointer(3, gl.GL_FLOAT, 0, None)
            
            # Draw
            gl.glDrawArrays(gl.GL_TRIANGLES, 0, 6)
            
            # Clean up
            gl.glDisableClientState(gl.GL_VERTEX_ARRAY)
            gl.glDeleteBuffers(1, [vbo])
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
            
        except Exception as e:
            print(f"❌ Text rendering failed for {self.object_id}: {e}")
            # Last resort: just print that we can't render
            print(f"⚠️  Cannot render text {self.object_id} - skipping")
    
    def set_text(self, text):
        """Update the text content
        
        Args:
            text: New text to display
        """
        self.text = text 