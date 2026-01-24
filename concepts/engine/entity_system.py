from PyQt5.QtCore import QObject, pyqtSignal

class Component:
    """Base class for entity components"""
    
    def __init__(self, name):
        """Initialize component
        
        Args:
            name: Unique name for this component
        """
        self.name = name
        self.entity = None  # Will be set when added to an entity
    
    def initialize(self):
        """Initialize component after being added to an entity"""
        pass
    
    def update(self, dt):
        """Update component logic
        
        Args:
            dt: Time elapsed since last update in seconds
        """
        pass
    
    def cleanup(self):
        """Clean up component resources"""
        pass
    
    def handle_event(self, event_type, emitter_id, event_data):
        """Handle events directed at this component
        
        Args:
            event_type: Type of event
            emitter_id: ID of the emitting entity
            event_data: Event data payload
        """
        pass


class Entity(QObject):
    """A game entity with components"""
    
    # Signal for events emitted by this entity
    entity_event = pyqtSignal(str, str, object)  # event_type, entity_id, event_data
    
    def __init__(self, entity_id):
        """Initialize entity
        
        Args:
            entity_id: Unique identifier for this entity
        """
        super().__init__()
        self.entity_id = entity_id
        self.components = {}  # name -> component mapping
        self.tags = set()  # Set of string tags for categorization
        self.active = True
    
    def add_component(self, component):
        """Add a component to this entity
        
        Args:
            component: Component instance to add
            
        Returns:
            The added component for chaining
        """
        if component.name in self.components:
            # Replace existing component
            self.components[component.name].entity = None
        
        self.components[component.name] = component
        component.entity = self
        component.initialize()
        return component
    
    def get_component(self, name):
        """Get a component by name
        
        Args:
            name: Name of the component to get
            
        Returns:
            Component instance or None if not found
        """
        return self.components.get(name)
    
    def remove_component(self, name):
        """Remove a component by name
        
        Args:
            name: Name of the component to remove
            
        Returns:
            The removed component or None if not found
        """
        if name in self.components:
            component = self.components[name]
            component.cleanup()
            component.entity = None
            del self.components[name]
            return component
        return None
    
    def has_component(self, name):
        """Check if entity has a component
        
        Args:
            name: Name of the component to check
            
        Returns:
            True if entity has component, False otherwise
        """
        return name in self.components
    
    def add_tag(self, tag):
        """Add a tag to this entity
        
        Args:
            tag: String tag to add
        """
        self.tags.add(tag)
    
    def remove_tag(self, tag):
        """Remove a tag from this entity
        
        Args:
            tag: String tag to remove
        """
        self.tags.discard(tag)
    
    def has_tag(self, tag):
        """Check if entity has a tag
        
        Args:
            tag: String tag to check
            
        Returns:
            True if entity has tag, False otherwise
        """
        return tag in self.tags
    
    def update(self, dt):
        """Update all components
        
        Args:
            dt: Time elapsed since last update in seconds
        """
        if not self.active:
            return
            
        for component in self.components.values():
            component.update(dt)
    
    def cleanup(self):
        """Clean up entity resources"""
        for component in list(self.components.values()):
            component.cleanup()
        self.components.clear()
    
    def emit(self, event_type, event_data):
        """Emit an event to the entity system
        
        Args:
            event_type: Type of event being emitted
            event_data: Data payload for the event
            
        Returns:
            None
        """
        self.entity_event.emit(event_type, self.entity_id, event_data)


class EntityEventSystem(QObject):
    """System for entity event communication"""
    
    def __init__(self):
        """Initialize the entity event system"""
        super().__init__()
        self.entities = {}  # id -> entity mapping
        self.emitters = {}  # event_type -> [entity_id] mapping
        self.handlers = {}  # event_type -> [entity_id] mapping
    
    def add_entity(self, entity):
        """Add an entity to the system
        
        Args:
            entity: Entity instance to add
        """
        self.entities[entity.entity_id] = entity
        # Connect entity's event signal to our event handler
        entity.entity_event.connect(self.emit_event)
    
    def remove_entity(self, entity_id):
        """Remove an entity from the system
        
        Args:
            entity_id: ID of the entity to remove
            
        Returns:
            The removed entity or None if not found
        """
        if entity_id in self.entities:
            entity = self.entities[entity_id]
            
            # Disconnect entity's event signal
            entity.entity_event.disconnect(self.emit_event)
            
            # Remove entity from emitters and handlers
            for event_type, emitter_ids in list(self.emitters.items()):
                if entity_id in emitter_ids:
                    emitter_ids.remove(entity_id)
            
            for event_type, handler_ids in list(self.handlers.items()):
                if entity_id in handler_ids:
                    handler_ids.remove(entity_id)
            
            del self.entities[entity_id]
            return entity
        return None
    
    def get_entity(self, entity_id):
        """Get an entity by ID
        
        Args:
            entity_id: ID of the entity to get
            
        Returns:
            Entity instance or None if not found
        """
        return self.entities.get(entity_id)
    
    def register_emitter(self, entity_id, event_type):
        """Register an entity as an event emitter
        
        Args:
            entity_id: ID of the emitting entity
            event_type: Type of event it emits
        """
        if event_type not in self.emitters:
            self.emitters[event_type] = set()
        
        self.emitters[event_type].add(entity_id)
    
    def register_handler(self, entity_id, event_type):
        """Register an entity as an event handler
        
        Args:
            entity_id: ID of the handling entity
            event_type: Type of event it handles
        """
        if event_type not in self.handlers:
            self.handlers[event_type] = set()
        
        self.handlers[event_type].add(entity_id)
    
    def unregister_emitter(self, entity_id, event_type):
        """Unregister an entity as an event emitter
        
        Args:
            entity_id: ID of the emitting entity
            event_type: Type of event it emits
        """
        if event_type in self.emitters:
            self.emitters[event_type].discard(entity_id)
    
    def unregister_handler(self, entity_id, event_type):
        """Unregister an entity as an event handler
        
        Args:
            entity_id: ID of the handling entity
            event_type: Type of event it handles
        """
        if event_type in self.handlers:
            self.handlers[event_type].discard(entity_id)
    
    def emit_event(self, event_type, emitter_id, event_data):
        """Emit an event to all registered handlers
        
        Args:
            event_type: Type of the event
            emitter_id: ID of the emitting entity
            event_data: Data payload for the event
        """
        if event_type not in self.handlers:
            return
            
        for handler_id in self.handlers[event_type]:
            handler = self.entities.get(handler_id)
            if handler and handler.active:
                # Find components that can handle this event
                for component in handler.components.values():
                    if hasattr(component, 'handle_event'):
                        component.handle_event(event_type, emitter_id, event_data)
    
    def update_all(self, dt):
        """Update all entities
        
        Args:
            dt: Time elapsed since last update in seconds
        """
        for entity in self.entities.values():
            entity.update(dt) 