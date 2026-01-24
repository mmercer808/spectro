"""
Scene and SceneManager System for OpenGL Rendering
Converts SVG layouts to OpenGL shader code for complex UI rendering
"""

import numpy as np
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import os
import math

# Import the existing render engine components
from .render_engine import RenderEngine, Rectangle, Text

class PrimitiveType(Enum):
    RECTANGLE = "rectangle"
    CIRCLE = "circle"
    TEXT = "text"
    POLYGON = "polygon"
    GRADIENT_RECT = "gradient_rect"

@dataclass
class ScenePrimitive:
    """A primitive shape that can be rendered in OpenGL"""
    primitive_id: str
    primitive_type: PrimitiveType
    x: float = 0.0
    y: float = 0.0
    width: float = 0.0
    height: float = 0.0
    color: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    border_color: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    border_width: float = 0.0
    corner_radius: float = 0.0
    text_content: str = ""
    font_size: float = 12.0
    gradient_colors: List[Tuple[float, float, float, float]] = None
    gradient_direction: str = "vertical"  # "vertical", "horizontal", "radial"
    visible: bool = True
    z_index: int = 0
    
    def __post_init__(self):
        if self.gradient_colors is None:
            self.gradient_colors = [self.color]

class Scene:
    """A scene containing primitives that can be rendered together"""
    
    def __init__(self, scene_id: str, render_engine: RenderEngine, width: int = 800, height: int = 600):
        """Initialize a scene
        
        Args:
            scene_id: Unique identifier for this scene
            render_engine: The render engine to use for rendering
            width: Scene width in pixels
            height: Scene height in pixels
        """
        self.scene_id = scene_id
        self.render_engine = render_engine
        self.width = width
        self.height = height
        self.primitives: List[ScenePrimitive] = []
        self.renderable_objects: Dict[str, Any] = {}  # Map primitive_id to renderable objects
        self.initialized = False
        self.background_color = (0.1, 0.1, 0.1, 1.0)
    
    def add_primitive(self, primitive: ScenePrimitive):
        """Add a primitive to the scene
        
        Args:
            primitive: The primitive to add
        """
        self.primitives.append(primitive)
        # Sort by z-index for proper rendering order
        self.primitives.sort(key=lambda p: p.z_index)
    
    def remove_primitive(self, primitive_id: str):
        """Remove a primitive from the scene
        
        Args:
            primitive_id: ID of the primitive to remove
        """
        # Remove from render engine
        if primitive_id in self.renderable_objects:
            renderable = self.renderable_objects[primitive_id]
            self.render_engine.remove_renderable(renderable.object_id)
            del self.renderable_objects[primitive_id]
        
        # Remove from primitives list
        self.primitives = [p for p in self.primitives if p.primitive_id != primitive_id]
    
    def get_primitive(self, primitive_id: str) -> Optional[ScenePrimitive]:
        """Get a primitive by ID
        
        Args:
            primitive_id: ID of the primitive
            
        Returns:
            ScenePrimitive or None if not found
        """
        for primitive in self.primitives:
            if primitive.primitive_id == primitive_id:
                return primitive
        return None
    
    def update_primitive_color(self, primitive_id: str, color: Tuple[float, float, float, float]):
        """Update the color of a primitive
        
        Args:
            primitive_id: ID of the primitive
            color: New RGBA color
        """
        primitive = self.get_primitive(primitive_id)
        if primitive:
            primitive.color = color
            # Update the renderable object if it exists
            if primitive_id in self.renderable_objects:
                renderable = self.renderable_objects[primitive_id]
                if hasattr(renderable, 'color'):
                    renderable.color = color
    
    def update_primitive_visibility(self, primitive_id: str, visible: bool):
        """Update the visibility of a primitive
        
        Args:
            primitive_id: ID of the primitive
            visible: Whether the primitive should be visible
        """
        primitive = self.get_primitive(primitive_id)
        if primitive:
            primitive.visible = visible
            # Update the renderable object if it exists
            if primitive_id in self.renderable_objects:
                renderable = self.renderable_objects[primitive_id]
                renderable.visible = visible
    
    def initialize(self):
        """Initialize the scene by creating renderable objects"""
        if self.initialized:
            return True
        
        try:
            # Create renderable objects for all primitives
            for primitive in self.primitives:
                if primitive.visible:
                    renderable = self._create_renderable_for_primitive(primitive)
                    if renderable:
                        self.renderable_objects[primitive.primitive_id] = renderable
                        self.render_engine.add_renderable(renderable)
            
            self.initialized = True
            return True
            
        except Exception as e:
            print(f"Error initializing scene {self.scene_id}: {e}")
            return False
    
    def _create_renderable_for_primitive(self, primitive: ScenePrimitive):
        """Create a renderable object for a primitive
        
        Args:
            primitive: The primitive to create a renderable for
            
        Returns:
            Renderable object or None if failed
        """
        try:
            if primitive.primitive_type == PrimitiveType.RECTANGLE:
                # Convert color to tuple if needed
                color = primitive.color
                if hasattr(color, 'redF'):  # Check if it's a QColor
                    color = (color.redF(), color.greenF(), color.blueF(), color.alphaF())
                
                return Rectangle(
                    primitive.primitive_id,
                    primitive.x, primitive.y,
                    primitive.width, primitive.height,
                    color=color
                )
            
            elif primitive.primitive_type == PrimitiveType.TEXT:
                # Convert color to tuple if needed
                color = primitive.color
                if hasattr(color, 'redF'):  # Check if it's a QColor
                    color = (color.redF(), color.greenF(), color.blueF(), color.alphaF())
                
                return Text(
                    primitive.primitive_id,
                    primitive.x, primitive.y,
                    primitive.text_content,
                    color=color,
                    size=int(primitive.font_size)
                )
            
            elif primitive.primitive_type == PrimitiveType.CIRCLE:
                # For now, create a rectangle for circles (can be enhanced later)
                color = primitive.color
                if hasattr(color, 'redF'):  # Check if it's a QColor
                    color = (color.redF(), color.greenF(), color.blueF(), color.alphaF())
                
                return Rectangle(
                    primitive.primitive_id,
                    primitive.x, primitive.y,
                    primitive.width, primitive.height,
                    color=color
                )
            
            elif primitive.primitive_type == PrimitiveType.GRADIENT_RECT:
                # For now, use the first gradient color (can be enhanced later)
                color = primitive.gradient_colors[0] if primitive.gradient_colors else primitive.color
                if hasattr(color, 'redF'):  # Check if it's a QColor
                    color = (color.redF(), color.greenF(), color.blueF(), color.alphaF())
                
                return Rectangle(
                    primitive.primitive_id,
                    primitive.x, primitive.y,
                    primitive.width, primitive.height,
                    color=color
                )
            
            else:
                print(f"Unsupported primitive type: {primitive.primitive_type}")
                return None
                
        except Exception as e:
            print(f"Error creating renderable for primitive {primitive.primitive_id}: {e}")
            return None
    
    def render(self):
        """Render the scene (delegates to render engine)"""
        # The render engine handles the actual rendering
        # This method is kept for compatibility but doesn't need to do anything
        pass
    
    def cleanup(self):
        """Clean up scene resources"""
        # Remove all renderable objects from the render engine
        for primitive_id, renderable in self.renderable_objects.items():
            self.render_engine.remove_renderable(renderable.object_id)
        
        self.renderable_objects.clear()
        self.initialized = False

class SceneManager:
    """Manages multiple scenes and handles scene switching"""
    
    def __init__(self, render_engine: RenderEngine, viewport_width: int = 800, viewport_height: int = 600):
        """Initialize the scene manager
        
        Args:
            render_engine: The render engine to use for rendering
            viewport_width: Viewport width in pixels
            viewport_height: Viewport height in pixels
        """
        self.render_engine = render_engine
        self.scenes: Dict[str, Scene] = {}
        self.current_scene_id: Optional[str] = None
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self.primitive_counter = 0  # Counter for generating unique IDs
    
    def create_scene(self, scene_id: str, width: int = 800, height: int = 600) -> Scene:
        """Create a new scene
        
        Args:
            scene_id: Unique identifier for the scene
            width: Scene width
            height: Scene height
            
        Returns:
            The created scene
        """
        scene = Scene(scene_id, self.render_engine, width, height)
        self.scenes[scene_id] = scene
        return scene
    
    def get_scene(self, scene_id: str) -> Optional[Scene]:
        """Get a scene by ID
        
        Args:
            scene_id: Scene ID
            
        Returns:
            Scene or None if not found
        """
        return self.scenes.get(scene_id)
    
    def set_current_scene(self, scene_id: str) -> bool:
        """Set the current active scene
        
        Args:
            scene_id: ID of the scene to activate
            
        Returns:
            True if successful, False if scene not found
        """
        if scene_id in self.scenes:
            # Clean up current scene if it exists
            if self.current_scene_id and self.current_scene_id in self.scenes:
                current_scene = self.scenes[self.current_scene_id]
                current_scene.cleanup()
            
            # Initialize and set new scene
            scene = self.scenes[scene_id]
            if scene.initialize():
                self.current_scene_id = scene_id
                return True
            else:
                print(f"Failed to initialize scene: {scene_id}")
                return False
        return False
    
    def get_current_scene(self) -> Optional[Scene]:
        """Get the current active scene
        
        Returns:
            Current scene or None if no scene is active
        """
        if self.current_scene_id:
            return self.scenes.get(self.current_scene_id)
        return None
    
    def render_current_scene(self):
        """Render the current active scene"""
        # The render engine handles rendering, so this method doesn't need to do anything
        # The scene's renderable objects are already added to the render engine
        pass
    
    def load_scene_from_svg(self, scene_id: str, svg_path: str) -> bool:
        """Load a scene from an SVG file
        
        Args:
            scene_id: ID for the new scene
            svg_path: Path to the SVG file
            
        Returns:
            True if successful, False otherwise
        """
        if not os.path.exists(svg_path):
            print(f"SVG file not found: {svg_path}")
            return False
        
        try:
            with open(svg_path, 'r', encoding='utf-8') as f:
                svg_content = f.read()
            
            return self.create_scene_from_svg_content(scene_id, svg_content)
            
        except Exception as e:
            print(f"Error loading SVG file: {e}")
            return False
    
    def create_scene_from_svg_content(self, scene_id: str, svg_content: str) -> bool:
        """Create a scene from SVG content
        
        Args:
            scene_id: ID for the new scene
            svg_content: SVG XML content
            
        Returns:
            True if successful, False otherwise
        """
        try:
            root = ET.fromstring(svg_content)
            
            # Parse viewBox
            viewbox = root.get('viewBox', '0 0 800 600')
            viewbox_parts = viewbox.split()
            width = int(float(viewbox_parts[2])) if len(viewbox_parts) >= 3 else 800
            height = int(float(viewbox_parts[3])) if len(viewbox_parts) >= 4 else 600
            
            # Create scene
            scene = self.create_scene(scene_id, width, height)
            
            # Parse SVG elements
            self._parse_svg_elements(root, scene)
            
            return True
            
        except Exception as e:
            print(f"Error creating scene from SVG: {e}")
            return False
    
    def _parse_svg_elements(self, element: ET.Element, scene: Scene, parent_id: str = ""):
        """Recursively parse SVG elements and add them to the scene
        
        Args:
            element: XML element to parse
            scene: Scene to add primitives to
            parent_id: ID of parent element
        """
        tag = element.tag.split('}')[-1]  # Remove namespace if present
        
        # Skip non-renderable elements
        if tag in ['defs', 'linearGradient', 'stop']:
            for child in element:
                self._parse_svg_elements(child, scene, parent_id)
            return
        
        # Create primitive based on element type
        primitive = None
        
        if tag == 'rect':
            primitive = self._create_rectangle_primitive(element)
        elif tag == 'circle':
            primitive = self._create_circle_primitive(element)
        elif tag == 'text':
            primitive = self._create_text_primitive(element)
        elif tag == 'polygon':
            primitive = self._create_polygon_primitive(element)
        
        if primitive:
            scene.add_primitive(primitive)
        
        # Parse children
        for child in element:
            child_tag = child.tag.split('}')[-1]
            if child_tag not in ['defs', 'linearGradient', 'stop']:
                self._parse_svg_elements(child, scene, parent_id)
    
    def _create_rectangle_primitive(self, element: ET.Element) -> Optional[ScenePrimitive]:
        """Create a rectangle primitive from SVG element
        
        Args:
            element: SVG rect element
            
        Returns:
            ScenePrimitive or None
        """
        try:
            x = float(element.get('x', 0))
            y = float(element.get('y', 0))
            width = float(element.get('width', 0))
            height = float(element.get('height', 0))
            fill = element.get('fill', '#ffffff')
            stroke = element.get('stroke', 'none')
            stroke_width = float(element.get('stroke-width', 0))
            rx = float(element.get('rx', 0))
            
            # Convert colors
            fill_color = self._parse_color(fill)
            stroke_color = self._parse_color(stroke)
            
            primitive_id = element.get('id', f"rect_{self.primitive_counter}")
            self.primitive_counter += 1
            
            return ScenePrimitive(
                primitive_id=primitive_id,
                primitive_type=PrimitiveType.RECTANGLE,
                x=x, y=y, width=width, height=height,
                color=fill_color,
                border_color=stroke_color,
                border_width=stroke_width,
                corner_radius=rx
            )
            
        except Exception as e:
            print(f"Error creating rectangle primitive: {e}")
            return None
    
    def _create_circle_primitive(self, element: ET.Element) -> Optional[ScenePrimitive]:
        """Create a circle primitive from SVG element
        
        Args:
            element: SVG circle element
            
        Returns:
            ScenePrimitive or None
        """
        try:
            cx = float(element.get('cx', 0))
            cy = float(element.get('cy', 0))
            r = float(element.get('r', 0))
            fill = element.get('fill', '#ffffff')
            
            fill_color = self._parse_color(fill)
            primitive_id = element.get('id', f"circle_{self.primitive_counter}")
            self.primitive_counter += 1
            
            return ScenePrimitive(
                primitive_id=primitive_id,
                primitive_type=PrimitiveType.CIRCLE,
                x=cx - r, y=cy - r, width=r * 2, height=r * 2,
                color=fill_color
            )
            
        except Exception as e:
            print(f"Error creating circle primitive: {e}")
            return None
    
    def _create_text_primitive(self, element: ET.Element) -> Optional[ScenePrimitive]:
        """Create a text primitive from SVG element
        
        Args:
            element: SVG text element
            
        Returns:
            ScenePrimitive or None
        """
        try:
            x = float(element.get('x', 0))
            y = float(element.get('y', 0))
            text_content = element.text or ""
            fill = element.get('fill', '#ffffff')
            font_size = float(element.get('font-size', 12))
            
            fill_color = self._parse_color(fill)
            primitive_id = element.get('id', f"text_{self.primitive_counter}")
            self.primitive_counter += 1
            
            return ScenePrimitive(
                primitive_id=primitive_id,
                primitive_type=PrimitiveType.TEXT,
                x=x, y=y, width=len(text_content) * font_size * 0.6, height=font_size,
                color=fill_color,
                text_content=text_content,
                font_size=font_size
            )
            
        except Exception as e:
            print(f"Error creating text primitive: {e}")
            return None
    
    def _create_polygon_primitive(self, element: ET.Element) -> Optional[ScenePrimitive]:
        """Create a polygon primitive from SVG element
        
        Args:
            element: SVG polygon element
            
        Returns:
            ScenePrimitive or None
        """
        try:
            points_str = element.get('points', '')
            fill = element.get('fill', '#ffffff')
            
            if not points_str:
                return None
            
            # Parse points
            points = []
            point_pairs = points_str.strip().split()
            for pair in point_pairs:
                if ',' in pair:
                    x, y = pair.split(',')
                    points.append((float(x), float(y)))
            
            if not points:
                return None
            
            # Calculate bounding box
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            
            x = min(xs)
            y = min(ys)
            width = max(xs) - x
            height = max(ys) - y
            
            fill_color = self._parse_color(fill)
            primitive_id = element.get('id', f"polygon_{self.primitive_counter}")
            self.primitive_counter += 1
            
            return ScenePrimitive(
                primitive_id=primitive_id,
                primitive_type=PrimitiveType.RECTANGLE,  # Simplified for now
                x=x, y=y, width=width, height=height,
                color=fill_color
            )
            
        except Exception as e:
            print(f"Error creating polygon primitive: {e}")
            return None
    
    def _parse_color(self, color_str: str) -> Tuple[float, float, float, float]:
        """Parse SVG color string to RGBA tuple
        
        Args:
            color_str: SVG color string
            
        Returns:
            RGBA tuple (0.0-1.0)
        """
        if not color_str or color_str == 'none':
            return (0.0, 0.0, 0.0, 0.0)
        
        # Handle hex colors
        if color_str.startswith('#'):
            hex_color = color_str[1:]
            if len(hex_color) == 3:
                hex_color = ''.join([c*2 for c in hex_color])
            
            r = int(hex_color[0:2], 16) / 255.0
            g = int(hex_color[2:4], 16) / 255.0
            b = int(hex_color[4:6], 16) / 255.0
            a = 1.0
            
            if len(hex_color) == 8:
                a = int(hex_color[6:8], 16) / 255.0
            
            return (r, g, b, a)
        
        # Handle named colors
        color_map = {
            'black': (0.0, 0.0, 0.0, 1.0),
            'white': (1.0, 1.0, 1.0, 1.0),
            'red': (1.0, 0.0, 0.0, 1.0),
            'green': (0.0, 1.0, 0.0, 1.0),
            'blue': (0.0, 0.0, 1.0, 1.0),
            'yellow': (1.0, 1.0, 0.0, 1.0),
            'cyan': (0.0, 1.0, 1.0, 1.0),
            'magenta': (1.0, 0.0, 1.0, 1.0),
            'gray': (0.5, 0.5, 0.5, 1.0),
            'lightgray': (0.8, 0.8, 0.8, 1.0),
            'darkgray': (0.3, 0.3, 0.3, 1.0),
        }
        
        return color_map.get(color_str.lower(), (1.0, 1.0, 1.0, 1.0))
    
    def resize_viewport(self, width: int, height: int):
        """Resize the viewport
        
        Args:
            width: New viewport width
            height: New viewport height
        """
        self.viewport_width = width
        self.viewport_height = height
    
    def cleanup(self):
        """Clean up all scenes"""
        for scene in self.scenes.values():
            scene.cleanup()
        self.scenes.clear()
        self.current_scene_id = None 