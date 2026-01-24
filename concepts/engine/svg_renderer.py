"""
SVG Layout Renderer for OpenGL
Integrates SVG parsing with OpenGL rendering for UI layouts
"""

import xml.etree.ElementTree as ET
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import os
import re
from dataclasses import dataclass
from enum import Enum

from engine.render_engine import Renderable, Rectangle, Text
from utils.stylesheet_system import StyleSheet

class SvgElementType(Enum):
    RECT = "rect"
    CIRCLE = "circle"
    ELLIPSE = "ellipse"
    PATH = "path"
    TEXT = "text"
    GROUP = "g"
    POLYGON = "polygon"
    LINE = "line"

@dataclass
class SvgElement:
    """Represents a parsed SVG element"""
    element_id: str
    element_type: SvgElementType
    x: float = 0.0
    y: float = 0.0
    width: float = 0.0
    height: float = 0.0
    fill: str = "#ffffff"
    stroke: str = "none"
    stroke_width: float = 0.0
    text_content: str = ""
    font_family: str = "Arial"
    font_size: float = 12.0
    rx: float = 0.0  # For rounded rectangles
    ry: float = 0.0
    cx: float = 0.0  # For circles
    cy: float = 0.0
    r: float = 0.0
    points: List[Tuple[float, float]] = None
    transform: str = ""
    opacity: float = 1.0
    visible: bool = True
    parent_id: str = ""
    children: List[str] = None
    
    def __post_init__(self):
        if self.children is None:
            self.children = []
        if self.points is None:
            self.points = []

class SvgLayoutRenderer:
    """Renders SVG layouts as OpenGL components"""
    
    def __init__(self, render_engine, stylesheet: StyleSheet = None):
        """Initialize SVG layout renderer
        
        Args:
            render_engine: OpenGL render engine
            stylesheet: Optional stylesheet for color management
        """
        self.render_engine = render_engine
        self.stylesheet = stylesheet
        self.elements: Dict[str, SvgElement] = {}
        self.renderables: Dict[str, Renderable] = {}
        self.viewport_width = 800
        self.viewport_height = 600
        self.initialized = False
    
    def load_svg_layout(self, svg_path: str) -> bool:
        """Load and parse SVG layout file
        
        Args:
            svg_path: Path to SVG file
            
        Returns:
            bool: True if loading was successful
        """
        if not os.path.exists(svg_path):
            print(f"SVG file not found: {svg_path}")
            return False
        
        try:
            with open(svg_path, 'r', encoding='utf-8') as f:
                svg_content = f.read()
            
            return self.parse_svg_content(svg_content)
            
        except Exception as e:
            print(f"Error loading SVG file: {e}")
            return False
    
    def parse_svg_content(self, svg_content: str) -> bool:
        """Parse SVG content and create elements
        
        Args:
            svg_content: SVG XML content
            
        Returns:
            bool: True if parsing was successful
        """
        try:
            root = ET.fromstring(svg_content)
            
            # Parse viewBox
            viewbox = root.get('viewBox', '0 0 800 600')
            viewbox_parts = viewbox.split()
            if len(viewbox_parts) >= 4:
                self.viewport_width = float(viewbox_parts[2])
                self.viewport_height = float(viewbox_parts[3])
            
            # Clear existing elements
            self.elements.clear()
            
            # Parse all elements
            self._parse_element(root, "")
            
            return True
            
        except Exception as e:
            print(f"Error parsing SVG content: {e}")
            return False
    
    def _parse_element(self, element: ET.Element, parent_id: str = ""):
        """Recursively parse SVG element and its children
        
        Args:
            element: XML element to parse
            parent_id: ID of parent element
        """
        tag = element.tag.split('}')[-1]  # Remove namespace if present
        
        # Skip defs and other non-renderable elements
        if tag in ['defs', 'linearGradient', 'stop']:
            # Still parse children of defs
            for child in element:
                self._parse_element(child, parent_id)
            return
        
        # Create element ID
        element_id = element.get('id')
        if not element_id:
            element_id = f"{tag}_{len(self.elements)}"
        
        # Create SVG element based on type
        svg_element = None
        
        if tag == 'rect':
            svg_element = self._parse_rect_element(element, element_id)
        elif tag == 'circle':
            svg_element = self._parse_circle_element(element, element_id)
        elif tag == 'text':
            svg_element = self._parse_text_element(element, element_id)
        elif tag == 'polygon':
            svg_element = self._parse_polygon_element(element, element_id)
        elif tag == 'g':
            svg_element = self._parse_group_element(element, element_id)
        
        if svg_element:
            svg_element.parent_id = parent_id
            self.elements[element_id] = svg_element
            
            # Parse children
            for child in element:
                child_tag = child.tag.split('}')[-1]
                if child_tag not in ['defs', 'linearGradient', 'stop']:
                    self._parse_element(child, element_id)
                    child_id = child.get('id', f"child_{len(svg_element.children)}")
                    svg_element.children.append(child_id)
        
        # Also parse children even if this element wasn't created as a renderable
        # (like for the root svg element)
        for child in element:
            child_tag = child.tag.split('}')[-1]
            if child_tag not in ['defs', 'linearGradient', 'stop']:
                self._parse_element(child, parent_id)
    
    def _parse_rect_element(self, element: ET.Element, element_id: str) -> SvgElement:
        """Parse rectangle element
        
        Args:
            element: XML element
            element_id: Element ID
            
        Returns:
            SvgElement: Parsed rectangle element
        """
        return SvgElement(
            element_id=element_id,
            element_type=SvgElementType.RECT,
            x=float(element.get('x', 0)),
            y=float(element.get('y', 0)),
            width=float(element.get('width', 0)),
            height=float(element.get('height', 0)),
            fill=element.get('fill', '#ffffff'),
            stroke=element.get('stroke', 'none'),
            stroke_width=float(element.get('stroke-width', 0)),
            rx=float(element.get('rx', 0)),
            ry=float(element.get('ry', 0)),
            opacity=float(element.get('opacity', 1.0))
        )
    
    def _parse_circle_element(self, element: ET.Element, element_id: str) -> SvgElement:
        """Parse circle element
        
        Args:
            element: XML element
            element_id: Element ID
            
        Returns:
            SvgElement: Parsed circle element
        """
        return SvgElement(
            element_id=element_id,
            element_type=SvgElementType.CIRCLE,
            cx=float(element.get('cx', 0)),
            cy=float(element.get('cy', 0)),
            r=float(element.get('r', 0)),
            fill=element.get('fill', '#ffffff'),
            stroke=element.get('stroke', 'none'),
            stroke_width=float(element.get('stroke-width', 0)),
            opacity=float(element.get('opacity', 1.0))
        )
    
    def _parse_text_element(self, element: ET.Element, element_id: str) -> SvgElement:
        """Parse text element
        
        Args:
            element: XML element
            element_id: Element ID
            
        Returns:
            SvgElement: Parsed text element
        """
        # Get text content
        text_content = element.text or ""
        for child in element:
            if child.tag.split('}')[-1] == 'tspan':
                text_content += child.text or ""
        
        return SvgElement(
            element_id=element_id,
            element_type=SvgElementType.TEXT,
            x=float(element.get('x', 0)),
            y=float(element.get('y', 0)),
            text_content=text_content,
            fill=element.get('fill', '#ffffff'),
            font_family=element.get('font-family', 'Arial'),
            font_size=float(element.get('font-size', 12)),
            opacity=float(element.get('opacity', 1.0))
        )
    
    def _parse_polygon_element(self, element: ET.Element, element_id: str) -> SvgElement:
        """Parse polygon element
        
        Args:
            element: XML element
            element_id: Element ID
            
        Returns:
            SvgElement: Parsed polygon element
        """
        points_str = element.get('points', '')
        points = []
        
        # Parse points string "x1,y1 x2,y2 x3,y3..."
        if points_str:
            point_pairs = points_str.strip().split()
            for pair in point_pairs:
                if ',' in pair:
                    x, y = pair.split(',')
                    points.append((float(x), float(y)))
        
        return SvgElement(
            element_id=element_id,
            element_type=SvgElementType.POLYGON,
            points=points,
            fill=element.get('fill', '#ffffff'),
            stroke=element.get('stroke', 'none'),
            stroke_width=float(element.get('stroke-width', 0)),
            opacity=float(element.get('opacity', 1.0))
        )
    
    def _parse_group_element(self, element: ET.Element, element_id: str) -> SvgElement:
        """Parse group element
        
        Args:
            element: XML element
            element_id: Element ID
            
        Returns:
            SvgElement: Parsed group element
        """
        return SvgElement(
            element_id=element_id,
            element_type=SvgElementType.GROUP,
            opacity=float(element.get('opacity', 1.0))
        )
    
    def create_renderables(self):
        """Create OpenGL renderables from parsed SVG elements"""
        # Clear existing renderables
        for renderable in self.renderables.values():
            if hasattr(self.render_engine, 'remove_renderable'):
                self.render_engine.remove_renderable(renderable.object_id)
        self.renderables.clear()
        
        # Create renderables for each element
        for element_id, element in self.elements.items():
            if not element.visible:
                continue
                
            renderable = self._create_renderable_for_element(element)
            if renderable:
                self.renderables[element_id] = renderable
                if hasattr(self.render_engine, 'add_renderable'):
                    self.render_engine.add_renderable(renderable)
    
    def _create_renderable_for_element(self, element: SvgElement) -> Optional[Renderable]:
        """Create OpenGL renderable for SVG element
        
        Args:
            element: SVG element
            
        Returns:
            Renderable: OpenGL renderable or None if not supported
        """
        if element.element_type == SvgElementType.RECT:
            return self._create_rectangle_renderable(element)
        elif element.element_type == SvgElementType.CIRCLE:
            return self._create_circle_renderable(element)
        elif element.element_type == SvgElementType.TEXT:
            return self._create_text_renderable(element)
        elif element.element_type == SvgElementType.POLYGON:
            return self._create_polygon_renderable(element)
        
        return None
    
    def _create_rectangle_renderable(self, element: SvgElement) -> Rectangle:
        """Create rectangle renderable
        
        Args:
            element: SVG rectangle element
            
        Returns:
            Rectangle: OpenGL rectangle renderable
        """
        color = self._parse_color(element.fill)
        
        return Rectangle(
            object_id=element.element_id,
            x=element.x,
            y=element.y,
            width=element.width,
            height=element.height,
            color=color
        )
    
    def _create_circle_renderable(self, element: SvgElement) -> Rectangle:
        """Create circle renderable (as rectangle for now)
        
        Args:
            element: SVG circle element
            
        Returns:
            Rectangle: OpenGL rectangle renderable approximating circle
        """
        color = self._parse_color(element.fill)
        diameter = element.r * 2
        
        return Rectangle(
            object_id=element.element_id,
            x=element.cx - element.r,
            y=element.cy - element.r,
            width=diameter,
            height=diameter,
            color=color
        )
    
    def _create_text_renderable(self, element: SvgElement) -> Text:
        """Create text renderable
        
        Args:
            element: SVG text element
            
        Returns:
            Text: OpenGL text renderable
        """
        color = self._parse_color(element.fill)
        
        return Text(
            object_id=element.element_id,
            x=element.x,
            y=element.y,
            text=element.text_content,
            color=color,
            size=int(element.font_size)
        )
    
    def _create_polygon_renderable(self, element: SvgElement) -> Rectangle:
        """Create polygon renderable (as rectangle for now)
        
        Args:
            element: SVG polygon element
            
        Returns:
            Rectangle: OpenGL rectangle renderable approximating polygon
        """
        if not element.points:
            return None
        
        # Calculate bounding box
        xs = [p[0] for p in element.points]
        ys = [p[1] for p in element.points]
        
        x = min(xs)
        y = min(ys)
        width = max(xs) - x
        height = max(ys) - y
        
        color = self._parse_color(element.fill)
        
        return Rectangle(
            object_id=element.element_id,
            x=x,
            y=y,
            width=width,
            height=height,
            color=color
        )
    
    def _parse_color(self, color_str: str) -> Tuple[float, float, float, float]:
        """Parse SVG color string to RGBA tuple
        
        Args:
            color_str: SVG color string
            
        Returns:
            Tuple[float, float, float, float]: RGBA values (0.0-1.0)
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
    
    def get_element_by_id(self, element_id: str) -> Optional[SvgElement]:
        """Get SVG element by ID
        
        Args:
            element_id: Element ID
            
        Returns:
            SvgElement: Element or None if not found
        """
        return self.elements.get(element_id)
    
    def get_renderable_by_id(self, element_id: str) -> Optional[Renderable]:
        """Get renderable by element ID
        
        Args:
            element_id: Element ID
            
        Returns:
            Renderable: Renderable or None if not found
        """
        return self.renderables.get(element_id)
    
    def update_element_color(self, element_id: str, color: Tuple[float, float, float, float]):
        """Update element color
        
        Args:
            element_id: Element ID
            color: New RGBA color
        """
        element = self.elements.get(element_id)
        renderable = self.renderables.get(element_id)
        
        if element and renderable:
            element.fill = f"#{int(color[0]*255):02x}{int(color[1]*255):02x}{int(color[2]*255):02x}"
            renderable.color = color
    
    def update_element_visibility(self, element_id: str, visible: bool):
        """Update element visibility
        
        Args:
            element_id: Element ID
            visible: Whether element should be visible
        """
        element = self.elements.get(element_id)
        renderable = self.renderables.get(element_id)
        
        if element and renderable:
            element.visible = visible
            renderable.visible = visible
    
    def cleanup(self):
        """Clean up resources"""
        for renderable in self.renderables.values():
            if hasattr(renderable, 'cleanup'):
                renderable.cleanup()
        
        self.renderables.clear()
        self.elements.clear() 