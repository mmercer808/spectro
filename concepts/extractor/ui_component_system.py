import uuid
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QBrush, QLinearGradient

from engine.entity_system import Entity, Component
from engine.render_engine import Rectangle, Text, Renderable
from utils.stylesheet_system import StyleSheet

class UIComponent(Component):
    """Base class for UI components that can be styled"""
    
    def __init__(self, name="ui", view_name=None):
        """Initialize UI component
        
        Args:
            name: Component name
            view_name: Optional view namespace for styling
        """
        super().__init__(name)
        self.view_name = view_name
        self.visible = True
        self.position = (0, 0)
        self.size = (100, 50)
        self.renderables = []  # List of renderable objects
        self.children = []  # List of child entities
        self.stylesheet = None  # Reference to stylesheet system
        self.parent = None  # Parent UI component
    
    def set_position(self, x, y):
        """Set position of this component
        
        Args:
            x: X coordinate
            y: Y coordinate
        """
        old_x, old_y = self.position
        self.position = (x, y)
        
        # Adjust renderables
        for renderable in self.renderables:
            renderable.x += (x - old_x)
            renderable.y += (y - old_y)
    
    def set_size(self, width, height):
        """Set size of this component
        
        Args:
            width: Width of the component
            height: Height of the component
        """
        self.size = (width, height)
        self._update_layout()
    
    def set_stylesheet(self, stylesheet):
        """Set stylesheet for this component
        
        Args:
            stylesheet: StyleSheet instance
        """
        self.stylesheet = stylesheet
        self._apply_styles()
    
    def _apply_styles(self):
        """Apply styles from stylesheet to this component"""
        # To be implemented by subclasses
        pass
    
    def _update_layout(self):
        """Update layout based on current size and position"""
        # To be implemented by subclasses
        pass
    
    def add_child(self, child_entity, child_component=None):
        """Add a child UI entity to this component
        
        Args:
            child_entity: Entity to add as child
            child_component: Optional UIComponent to set as child
            
        Returns:
            The child entity
        """
        self.children.append(child_entity)
        
        # Set parent-child relationship if component is provided
        if child_component and isinstance(child_component, UIComponent):
            child_component.parent = self
        
        return child_entity
    
    def remove_child(self, child_entity):
        """Remove a child entity
        
        Args:
            child_entity: Entity to remove
            
        Returns:
            True if the child was removed, False otherwise
        """
        if child_entity in self.children:
            self.children.remove(child_entity)
            
            # Clear parent relationship
            ui_component = None
            for component in child_entity.components.values():
                if isinstance(component, UIComponent):
                    ui_component = component
                    break
                    
            if ui_component and ui_component.parent == self:
                ui_component.parent = None
                
            return True
            
        return False
    
    def handle_event(self, event_type, emitter_id, event_data):
        """Handle events directed at this component
        
        Args:
            event_type: Type of event
            emitter_id: ID of the emitting entity
            event_data: Event data payload
        """
        # UI event handling will be implemented by subclasses
        pass
    
    def update(self, dt):
        """Update component logic
        
        Args:
            dt: Time elapsed since last update in seconds
        """
        pass
    
    def cleanup(self):
        """Clean up component resources"""
        # Remove all renderables
        for renderable in self.renderables:
            if self.entity and hasattr(self.entity, 'render_engine'):
                self.entity.render_engine.remove_renderable(renderable.object_id)
        
        self.renderables.clear()
        
        # Clear children
        self.children.clear()
        
        super().cleanup()


class PanelComponent(UIComponent):
    """A rectangular panel UI component"""
    
    def __init__(self, view_name=None, color=None, border_color=None, border_width=0, corner_radius=0):
        """Initialize panel component
        
        Args:
            view_name: Optional view namespace for styling
            color: Background color (QColor, tuple, or hex string)
            border_color: Border color (QColor, tuple, or hex string)
            border_width: Border width in pixels
            corner_radius: Corner radius in pixels
        """
        super().__init__("panel", view_name)
        
        # Convert string colors to QColor objects
        if isinstance(color, str):
            self.color = QColor(color)
        else:
            self.color = color or QColor("#333333")
            
        if isinstance(border_color, str):
            self.border_color = QColor(border_color)
        else:
            self.border_color = border_color or QColor("#444444")
            
        self.border_width = border_width
        self.corner_radius = corner_radius
        self.background = None
    
    def initialize(self):
        """Initialize panel after being added to an entity"""
        super().initialize()
        
        # Create background rectangle
        x, y = self.position
        width, height = self.size
        
        # Convert QColor to tuple if needed
        bg_color = self.color
        if isinstance(bg_color, QColor):
            bg_color = (bg_color.redF(), bg_color.greenF(), bg_color.blueF(), bg_color.alphaF())
        
        # Create renderable
        self.background = Rectangle(
            f"panel_{self.entity.entity_id}",
            x, y, width, height,
            color=bg_color
        )
        
        # Add to render engine
        if hasattr(self.entity, 'render_engine'):
            self.entity.render_engine.add_renderable(self.background)
        
        self.renderables.append(self.background)
    
    def _apply_styles(self):
        """Apply styles from stylesheet to this panel"""
        if not self.stylesheet:
            return
            
        # Get panel color
        panel_color = self.stylesheet.get_color("panel", self.view_name)
        self.color = panel_color
        
        # Apply to background if it exists
        if self.background:
            self.background.color = (
                panel_color.redF(),
                panel_color.greenF(),
                panel_color.blueF(),
                panel_color.alphaF()
            )
    
    def _update_layout(self):
        """Update panel layout based on size and position"""
        if self.background:
            x, y = self.position
            width, height = self.size
            self.background.x = x
            self.background.y = y
            self.background.width = width
            self.background.height = height


class ButtonComponent(UIComponent):
    """A button UI component"""
    
    def __init__(self, text="Button", view_name=None, color=None, text_color=None):
        """Initialize button component
        
        Args:
            text: Button text
            view_name: Optional view namespace for styling
            color: Button color (QColor, tuple, or hex string)
            text_color: Text color (QColor, tuple, or hex string)
        """
        super().__init__("button", view_name)
        self.text = text
        
        # Convert string colors to QColor objects
        if isinstance(color, str):
            self.color = QColor(color)
        else:
            self.color = color or QColor("#0080ff")
            
        if isinstance(text_color, str):
            self.text_color = QColor(text_color)
        else:
            self.text_color = text_color or QColor("#ffffff")
            
        self.background = None
        self.text_renderable = None
        self.pressed = False
        self.hover = False
        self.size = (120, 30)  # Default button size
    
    def initialize(self):
        """Initialize button after being added to an entity"""
        super().initialize()
        
        x, y = self.position
        width, height = self.size
        
        # Convert QColor to tuple if needed
        bg_color = self.color
        if isinstance(bg_color, QColor):
            bg_color = (bg_color.redF(), bg_color.greenF(), bg_color.blueF(), bg_color.alphaF())
        
        # Create background rectangle
        self.background = Rectangle(
            f"button_bg_{self.entity.entity_id}",
            x, y, width, height,
            color=bg_color
        )
        
        # Create text renderable (if implemented)
        # For now we'll just use a placeholder since Text isn't fully implemented
        self.text_renderable = Text(
            f"button_text_{self.entity.entity_id}",
            x + width/2, y + height/2,
            self.text
        )
        
        # Add to render engine
        if hasattr(self.entity, 'render_engine'):
            self.entity.render_engine.add_renderable(self.background)
            self.entity.render_engine.add_renderable(self.text_renderable)
        
        self.renderables.append(self.background)
        self.renderables.append(self.text_renderable)
    
    def _apply_styles(self):
        """Apply styles from stylesheet to this button"""
        if not self.stylesheet:
            return
            
        # Get button colors
        button_color = self.stylesheet.get_color("button", self.view_name)
        text_color = self.stylesheet.get_color("button_text", self.view_name)
        
        self.color = button_color
        self.text_color = text_color
        
        # Apply to renderables if they exist
        if self.background:
            self.background.color = (
                button_color.redF(),
                button_color.greenF(),
                button_color.blueF(),
                button_color.alphaF()
            )
        
        if self.text_renderable:
            self.text_renderable.color = (
                text_color.redF(),
                text_color.greenF(),
                text_color.blueF(),
                text_color.alphaF()
            )
    
    def _update_layout(self):
        """Update button layout based on size and position"""
        if not self.background or not self.text_renderable:
            return
            
        x, y = self.position
        width, height = self.size
        
        # Update background
        self.background.x = x
        self.background.y = y
        self.background.width = width
        self.background.height = height
        
        # Update text position (centered)
        self.text_renderable.x = x + width/2
        self.text_renderable.y = y + height/2
    
    def handle_event(self, event_type, emitter_id, event_data):
        """Handle events directed at this button
        
        Args:
            event_type: Type of event
            emitter_id: ID of the emitting entity
            event_data: Event data payload
        """
        if event_type == "mouse_press":
            # Check if press is within button bounds
            mouse_x = event_data.get("x", 0)
            mouse_y = event_data.get("y", 0)
            
            if self._is_point_inside(mouse_x, mouse_y):
                self.pressed = True
                # Darken the button when pressed
                if self.background:
                    original_color = self.color
                    if isinstance(original_color, QColor):
                        darker = original_color.darker(120)
                        self.background.color = (
                            darker.redF(),
                            darker.greenF(),
                            darker.blueF(),
                            darker.alphaF()
                        )
        
        elif event_type == "mouse_release":
            if self.pressed:
                self.pressed = False
                # Restore original color
                self._apply_styles()
                
                # Check if release is within button bounds
                mouse_x = event_data.get("x", 0)
                mouse_y = event_data.get("y", 0)
                
                if self._is_point_inside(mouse_x, mouse_y):
                    # Button click completed, emit click event
                    if self.entity:
                        self.entity.emit("button_click", {"button": self.entity.entity_id})
    
    def _is_point_inside(self, x, y):
        """Check if a point is inside this button
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            bool: True if point is inside button
        """
        button_x, button_y = self.position
        button_width, button_height = self.size
        
        return (button_x <= x <= button_x + button_width and
                button_y <= y <= button_y + button_height)


class PadComponent(UIComponent):
    """A drum pad UI component"""
    
    def __init__(self, pad_id, view_name=None, color=None, active_color=None, key_binding=None):
        """Initialize pad component
        
        Args:
            pad_id: ID for this pad (1-16)
            view_name: Optional view namespace for styling
            color: Pad color (QColor, tuple, or hex string)
            active_color: Active pad color (QColor, tuple, or hex string)
            key_binding: Keyboard key that triggers this pad
        """
        super().__init__("pad", view_name)
        self.pad_id = pad_id
        
        # Convert string colors to QColor objects
        if isinstance(color, str):
            self.color = QColor(color)
        else:
            self.color = color or QColor("#333333")
            
        if isinstance(active_color, str):
            self.active_color = QColor(active_color)
        else:
            self.active_color = active_color or QColor("#00cc00")
            
        self.border_color = QColor("#444444")
        self.active_border_color = QColor("#00ff00")
        self.key_binding = key_binding
        self.background = None
        self.text_renderable = None
        self.key_text = None
        self.size = (80, 80)  # Default pad size
        self.active = False
        self.activation_time = 0
        self.deactivation_time = 0.2  # Time in seconds pad stays lit
    
    def initialize(self):
        """Initialize pad after being added to an entity"""
        super().initialize()
        
        x, y = self.position
        width, height = self.size
        
        # Convert QColor to tuple if needed
        pad_color = self.color
        if isinstance(pad_color, QColor):
            pad_color = (pad_color.redF(), pad_color.greenF(), pad_color.blueF(), pad_color.alphaF())
        
        # Create background rectangle
        self.background = Rectangle(
            f"pad_bg_{self.entity.entity_id}",
            x, y, width, height,
            color=pad_color
        )
        
        # Create text renderables
        self.text_renderable = Text(
            f"pad_text_{self.entity.entity_id}",
            x + width/2, y + height/2 - 10,
            f"Pad {self.pad_id}"
        )
        
        if self.key_binding:
            self.key_text = Text(
                f"pad_key_{self.entity.entity_id}",
                x + width/2, y + height/2 + 15,
                self.key_binding.upper()
            )
        
        # Add to render engine
        if hasattr(self.entity, 'render_engine'):
            self.entity.render_engine.add_renderable(self.background)
            self.entity.render_engine.add_renderable(self.text_renderable)
            if self.key_text:
                self.entity.render_engine.add_renderable(self.key_text)
        
        self.renderables.append(self.background)
        self.renderables.append(self.text_renderable)
        if self.key_text:
            self.renderables.append(self.key_text)
    
    def _apply_styles(self):
        """Apply styles from stylesheet to this pad"""
        if not self.stylesheet:
            return
            
        # Get pad colors
        pad_color = self.stylesheet.get_color("pad", self.view_name)
        active_color = self.stylesheet.get_color("pad_active", self.view_name)
        border_color = self.stylesheet.get_color("pad_border", self.view_name)
        active_border_color = self.stylesheet.get_color("pad_active_border", self.view_name)
        
        self.color = pad_color
        self.active_color = active_color
        self.border_color = border_color
        self.active_border_color = active_border_color
        
        # Apply current state
        self._update_visual_state()
    
    def _update_layout(self):
        """Update pad layout based on size and position"""
        if not self.background:
            return
            
        x, y = self.position
        width, height = self.size
        
        # Update background
        self.background.x = x
        self.background.y = y
        self.background.width = width
        self.background.height = height
        
        # Update text positions
        if self.text_renderable:
            self.text_renderable.x = x + width/2
            self.text_renderable.y = y + height/2 - 10
        
        if self.key_text:
            self.key_text.x = x + width/2
            self.key_text.y = y + height/2 + 15
    
    def set_active(self, active):
        """Set pad active state
        
        Args:
            active: Whether pad should be active (lit up)
        """
        self.active = active
        if active:
            self.activation_time = 0
        
        self._update_visual_state()
    
    def _update_visual_state(self):
        """Update visual appearance based on current state"""
        if not self.background:
            return
            
        if self.active:
            color = self.active_color
        else:
            color = self.color
            
        if isinstance(color, QColor):
            self.background.color = (
                color.redF(),
                color.greenF(),
                color.blueF(),
                color.alphaF()
            )
    
    def update(self, dt):
        """Update pad state
        
        Args:
            dt: Time elapsed since last update in seconds
        """
        if self.active:
            self.activation_time += dt
            if self.activation_time >= self.deactivation_time:
                self.set_active(False)
    
    def handle_event(self, event_type, emitter_id, event_data):
        """Handle events directed at this pad
        
        Args:
            event_type: Type of event
            emitter_id: ID of the emitting entity
            event_data: Event data payload
        """
        if event_type == "pad_trigger" and event_data.get("pad_id") == self.pad_id:
            self.set_active(True)
            
        elif event_type == "mouse_press":
            # Check if press is within pad bounds
            mouse_x = event_data.get("x", 0)
            mouse_y = event_data.get("y", 0)
            
            if self._is_point_inside(mouse_x, mouse_y):
                self.set_active(True)
                
                # Emit pad trigger event
                if self.entity:
                    self.entity.emit("pad_trigger", {
                        "pad_id": self.pad_id, 
                        "velocity": 100
                    })
    
    def _is_point_inside(self, x, y):
        """Check if a point is inside this pad
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            bool: True if point is inside pad
        """
        pad_x, pad_y = self.position
        pad_width, pad_height = self.size
        
        return (pad_x <= x <= pad_x + pad_width and
                pad_y <= y <= pad_y + pad_height)


class GridComponent(UIComponent):
    """A grid component for sequencer view"""
    
    def __init__(self, rows=8, cols=16, view_name=None):
        """Initialize grid component
        
        Args:
            rows: Number of rows
            cols: Number of columns
            view_name: Optional view namespace for styling
        """
        super().__init__("grid", view_name)
        self.rows = rows
        self.cols = cols
        self.size = (600, 300)  # Default grid size
        self.grid_lines = []
        self.background = None
        self.major_line_color = QColor("#444444")
        self.minor_line_color = QColor("#333333")
        self.background_color = QColor("#212121")
        self.line_width_major = 2
        self.line_width_minor = 1
    
    def initialize(self):
        """Initialize grid after being added to an entity"""
        super().initialize()
        
        x, y = self.position
        width, height = self.size
        
        # Create background
        bg_color = self.background_color
        if isinstance(bg_color, QColor):
            bg_color = (bg_color.redF(), bg_color.greenF(), bg_color.blueF(), bg_color.alphaF())
        
        self.background = Rectangle(
            f"grid_bg_{self.entity.entity_id}",
            x, y, width, height,
            color=bg_color
        )
        
        if hasattr(self.entity, 'render_engine'):
            self.entity.render_engine.add_renderable(self.background)
        
        self.renderables.append(self.background)
        
        # Create grid from engine.render_engine.Grid if available
        # Otherwise we'll create it manually in the next version
    
    def _apply_styles(self):
        """Apply styles from stylesheet to this grid"""
        if not self.stylesheet:
            return
            
        # Get grid colors
        panel_color = self.stylesheet.get_color("panel", self.view_name)
        major_line_color = self.stylesheet.get_color("grid_line_major", self.view_name)
        minor_line_color = self.stylesheet.get_color("grid_line_minor", self.view_name)
        
        self.background_color = panel_color
        self.major_line_color = major_line_color
        self.minor_line_color = minor_line_color
        
        # Apply to background
        if self.background:
            self.background.color = (
                panel_color.redF(),
                panel_color.greenF(),
                panel_color.blueF(),
                panel_color.alphaF()
            )
    
    def _update_layout(self):
        """Update grid layout based on size and position"""
        if self.background:
            x, y = self.position
            width, height = self.size
            self.background.x = x
            self.background.y = y
            self.background.width = width
            self.background.height = height


# Helper functions for UI creation

def create_ui_entity(entity_system, render_engine, entity_id=None):
    """Create a new entity for UI components
    
    Args:
        entity_system: EntityEventSystem instance
        render_engine: RenderEngine instance
        entity_id: Optional entity ID
        
    Returns:
        Entity: The created entity
    """
    if not entity_id:
        entity_id = f"ui_{uuid.uuid4().hex[:8]}"
        
    entity = Entity(entity_id)
    entity.render_engine = render_engine
    entity_system.add_entity(entity)
    
    return entity


def create_ui_panel(entity_system, render_engine, position=(0, 0), size=(100, 100), 
                   color=None, view_name=None, entity_id=None):
    """Create a panel entity
    
    Args:
        entity_system: EntityEventSystem instance
        render_engine: RenderEngine instance
        position: (x, y) position tuple
        size: (width, height) size tuple
        color: Panel color
        view_name: View namespace for styling
        entity_id: Optional entity ID
        
    Returns:
        tuple: (Entity, PanelComponent)
    """
    entity = create_ui_entity(entity_system, render_engine, entity_id)
    
    panel = PanelComponent(view_name, color)
    panel.position = position
    panel.size = size
    
    entity.add_component(panel)
    
    return entity, panel


def create_ui_button(entity_system, render_engine, text="Button", position=(0, 0), 
                    size=None, color=None, view_name=None, entity_id=None):
    """Create a button entity
    
    Args:
        entity_system: EntityEventSystem instance
        render_engine: RenderEngine instance
        text: Button text
        position: (x, y) position tuple
        size: (width, height) size tuple or None for default
        color: Button color
        view_name: View namespace for styling
        entity_id: Optional entity ID
        
    Returns:
        tuple: (Entity, ButtonComponent)
    """
    entity = create_ui_entity(entity_system, render_engine, entity_id)
    
    button = ButtonComponent(text, view_name, color)
    button.position = position
    if size:
        button.size = size
    
    entity.add_component(button)
    
    return entity, button


def create_ui_pad(entity_system, render_engine, pad_id, position=(0, 0), size=None,
                 color=None, active_color=None, key_binding=None, view_name=None, entity_id=None):
    """Create a pad entity
    
    Args:
        entity_system: EntityEventSystem instance
        render_engine: RenderEngine instance
        pad_id: Pad ID (1-16)
        position: (x, y) position tuple
        size: (width, height) size tuple or None for default
        color: Pad color
        active_color: Active pad color
        key_binding: Keyboard key binding
        view_name: View namespace for styling
        entity_id: Optional entity ID
        
    Returns:
        tuple: (Entity, PadComponent)
    """
    entity = create_ui_entity(entity_system, render_engine, entity_id)
    
    pad = PadComponent(pad_id, view_name, color, active_color, key_binding)
    pad.position = position
    if size:
        pad.size = size
    
    entity.add_component(pad)
    
    # Register for pad trigger events
    entity_system.register_handler(entity.entity_id, "pad_trigger")
    
    return entity, pad 