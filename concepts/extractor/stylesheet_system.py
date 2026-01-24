import xml.etree.ElementTree as ET
import os
import re
from PyQt5.QtGui import QColor
from PyQt5.QtCore import Qt, QObject, pyqtSignal

class StyleSheet(QObject):
    """Manages styles parsed from SVG files"""
    
    style_changed = pyqtSignal(str)  # Signal emitted when styles are changed
    
    def __init__(self):
        """Initialize the stylesheet system"""
        super().__init__()
        self.styles = {}
        self.gradients = {}
        self.color_mappings = {
            # Default color mappings for components
            "background": "#1a1a1a",
            "header": "#121212",
            "panel": "#212121",
            "button": "#0080ff",
            "button_text": "#ffffff",
            "pad": "#333333",
            "pad_active": "#00cc00",
            "pad_border": "#444444",
            "pad_active_border": "#00ff00",
            "text": "#ffffff",
            "grid_line_major": "#444444",
            "grid_line_minor": "#333333"
        }
        
        # Component-specific style mappings
        self.component_styles = {}
    
    def load_svg(self, svg_path):
        """Load styles from an SVG file
        
        Args:
            svg_path: Path to the SVG file
        
        Returns:
            bool: True if loading was successful
        """
        if not os.path.exists(svg_path):
            print(f"SVG file not found: {svg_path}")
            return False
            
        try:
            tree = ET.parse(svg_path)
            root = tree.getroot()
            
            # Extract styles from different elements
            self._extract_colors(root)
            self._extract_gradients(root)
            self._extract_component_styles(root)
            
            # Get view name from file to use as namespace
            view_name = os.path.basename(svg_path).split('.')[0]
            self.styles[view_name] = {
                "colors": self.color_mappings.copy(),
                "gradients": self.gradients.copy(),
                "components": self.component_styles.copy()
            }
            
            self.style_changed.emit(view_name)
            return True
            
        except Exception as e:
            print(f"Error loading SVG file: {e}")
            return False
    
    def _extract_colors(self, root):
        """Extract color information from SVG
        
        Args:
            root: Root XML element
        """
        # Extract background color
        bg_elements = root.findall(".//rect[@width='800'][@height='600']")
        if bg_elements:
            bg_color = bg_elements[0].get("fill")
            if bg_color:
                self.color_mappings["background"] = bg_color
        
        # Extract header color
        header_elements = root.findall(".//rect[@y='0'][@width='800'][@height='40']")
        if header_elements:
            header_color = header_elements[0].get("fill")
            if header_color:
                self.color_mappings["header"] = header_color
        
        # Extract button colors
        button_elements = root.findall(".//rect[@fill='#0080ff']")
        if button_elements:
            for element in button_elements:
                button_color = element.get("fill")
                if button_color:
                    self.color_mappings["button"] = button_color
                    # Try to find associated text
                    x = float(element.get("x", "0"))
                    y = float(element.get("y", "0"))
                    width = float(element.get("width", "0"))
                    height = float(element.get("height", "0"))
                    
                    # Find text elements within button bounds
                    for text_elem in root.findall(".//text"):
                        text_x = float(text_elem.get("x", "0"))
                        text_y = float(text_elem.get("y", "0"))
                        if (x <= text_x <= x + width) and (y <= text_y <= y + height):
                            text_color = text_elem.get("fill")
                            if text_color:
                                self.color_mappings["button_text"] = text_color
                            break
        
        # Extract pad colors
        pad_elements = root.findall(".//rect[@rx='5'][@stroke='#444444']")
        if pad_elements:
            for element in pad_elements:
                pad_color = element.get("fill")
                if pad_color and pad_color != "none":
                    self.color_mappings["pad"] = pad_color
                    pad_border = element.get("stroke")
                    if pad_border:
                        self.color_mappings["pad_border"] = pad_border
        
        # Extract active pad colors
        active_pad_elements = root.findall(".//rect[@fill='#00cc00']")
        if active_pad_elements:
            for element in active_pad_elements:
                active_color = element.get("fill")
                if active_color:
                    self.color_mappings["pad_active"] = active_color
                    active_border = element.get("stroke")
                    if active_border:
                        self.color_mappings["pad_active_border"] = active_border
    
    def _extract_gradients(self, root):
        """Extract gradient definitions from SVG
        
        Args:
            root: Root XML element
        """
        defs = root.findall(".//defs")
        if not defs:
            return
            
        for def_elem in defs:
            # Find all linearGradient elements
            for gradient in def_elem.findall(".//linearGradient"):
                gradient_id = gradient.get("id")
                if not gradient_id:
                    continue
                
                stops = []
                for stop in gradient.findall(".//stop"):
                    offset = stop.get("offset", "0%")
                    style = stop.get("style", "")
                    
                    # Parse color and opacity from style
                    color = "#000000"
                    opacity = 1.0
                    
                    color_match = re.search(r"stop-color:(#[0-9a-fA-F]+)", style)
                    if color_match:
                        color = color_match.group(1)
                    
                    opacity_match = re.search(r"stop-opacity:([0-9.]+)", style)
                    if opacity_match:
                        opacity = float(opacity_match.group(1))
                    
                    stops.append({
                        "offset": offset,
                        "color": color,
                        "opacity": opacity
                    })
                
                self.gradients[gradient_id] = {
                    "stops": stops,
                    "x1": gradient.get("x1", "0%"),
                    "y1": gradient.get("y1", "0%"),
                    "x2": gradient.get("x2", "100%"),
                    "y2": gradient.get("y2", "0%")
                }
    
    def _extract_component_styles(self, root):
        """Extract component-specific styles from SVG
        
        Args:
            root: Root XML element
        """
        # Extract pad grid layout for MIDI config view
        pad_grid = root.findall(".//rect[@fill='#2a2a2a']")
        if pad_grid:
            # Get the pad grid area dimensions
            for grid in pad_grid:
                x = float(grid.get("x", "0"))
                y = float(grid.get("y", "0"))
                width = float(grid.get("width", "0"))
                height = float(grid.get("height", "0"))
                
                # For the first grid (pads), store the layout
                self.component_styles["pad_grid"] = {
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height
                }
                break
        
        # Extract sequencer grid layout
        sequencer_grid = root.findall(".//rect[@id='sequencer_grid']") or root.findall(".//rect[@x='10'][@y='100']")
        if sequencer_grid:
            grid = sequencer_grid[0]
            self.component_styles["sequencer_grid"] = {
                "x": float(grid.get("x", "0")),
                "y": float(grid.get("y", "0")),
                "width": float(grid.get("width", "0")),
                "height": float(grid.get("height", "0"))
            }
            
        # Extract instrument colors for sequencer
        for instrument in ["kick", "snare", "hihat", "crash", "ride"]:
            gradient_id = f"{instrument}-gradient"
            gradient_elements = root.findall(f".//linearGradient[@id='{gradient_id}']")
            if gradient_elements:
                # Gradient already extracted in _extract_gradients
                self.color_mappings[f"{instrument}_color"] = f"url(#{gradient_id})"
            else:
                # Set default colors if gradients not found
                default_colors = {
                    "kick": "#ff4b2b",
                    "snare": "#38ef7d",
                    "hihat": "#00f2fe",
                    "crash": "#fa709a",
                    "ride": "#8e2de2"
                }
                self.color_mappings[f"{instrument}_color"] = default_colors.get(instrument, "#ffffff")
    
    def get_color(self, color_name, view_name=None):
        """Get a color by name
        
        Args:
            color_name: Name of the color to retrieve
            view_name: Optional view namespace to get view-specific color
            
        Returns:
            QColor: The requested color
        """
        color_value = None
        
        if view_name and view_name in self.styles:
            color_value = self.styles[view_name]["colors"].get(color_name)
        
        if not color_value:
            color_value = self.color_mappings.get(color_name, "#ffffff")
        
        # Handle URL references
        if isinstance(color_value, str) and color_value.startswith("url(#"):
            gradient_id = color_value[5:-1]  # Remove url(# and )
            # For simplicity, return the first color in the gradient
            if view_name and view_name in self.styles:
                gradient = self.styles[view_name]["gradients"].get(gradient_id)
                if gradient and gradient["stops"]:
                    color_value = gradient["stops"][0]["color"]
        
        # Convert to QColor
        return QColor(color_value)
    
    def get_gradient(self, gradient_id, view_name=None):
        """Get a gradient by ID
        
        Args:
            gradient_id: ID of the gradient to retrieve
            view_name: Optional view namespace
            
        Returns:
            dict: The gradient definition or None if not found
        """
        if view_name and view_name in self.styles:
            return self.styles[view_name]["gradients"].get(gradient_id)
        
        return self.gradients.get(gradient_id)
    
    def get_component_style(self, component_name, view_name=None):
        """Get component-specific style
        
        Args:
            component_name: Name of the component
            view_name: Optional view namespace
            
        Returns:
            dict: The component style or empty dict if not found
        """
        if view_name and view_name in self.styles:
            return self.styles[view_name]["components"].get(component_name, {})
        
        return self.component_styles.get(component_name, {}) 