#!/usr/bin/env python3
"""
Functional API Interface for Serializable Context Library

This module provides a clean, functional interface to the library components.
"""

import asyncio
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

from matts.context_system import SerializableContextLibrary, SerializableExecutionContext
from matts.signal_system import SignalLine, SignalPayload, SignalType, SignalPriority
from matts.generator_system import AdvancedGeneratorComposer
from matts.graph_system import BasicRelationshipGraph, RelationshipType

# =============================================================================
# GLOBAL LIBRARY INSTANCE
# =============================================================================

_global_library = None
_global_signal_line = None

def initialize_library() -> SerializableContextLibrary:
    """Initialize the global library instance."""
    global _global_library
    if _global_library is None:
        _global_library = SerializableContextLibrary()
    return _global_library

async def initialize_signal_line(line_id: str = "default") -> SignalLine:
    """Initialize the global signal line."""
    global _global_signal_line
    if _global_signal_line is None:
        from matts.signal_system import create_signal_line
        _global_signal_line = await create_signal_line(line_id)
    return _global_signal_line

# =============================================================================
# CONTEXT MANAGEMENT FUNCTIONS
# =============================================================================

def create_context(context_id: str = None) -> SerializableExecutionContext:
    """Create new serializable execution context."""
    library = initialize_library()
    return library.create_context(context_id)

# Ensure create_signal_line is properly exported
from matts.signal_system import create_signal_line  # Make sure this import works
def get_context(context_id: str) -> Optional[SerializableExecutionContext]:
    """Get context by ID."""
    library = initialize_library()
    return library.get_context(context_id)

def destroy_context(context_id: str) -> bool:
    """Destroy context and clean up resources."""
    library = initialize_library()
    return library.destroy_context(context_id)

async def bind_callback_with_dependencies(context: SerializableExecutionContext, 
                                         callback_source: str, callback_id: str,
                                         node_id: str = None) -> str:
    """Bind callback with automatic dependency detection."""
    return await context.bind_callback_with_auto_dependency_detection(
        callback_source, callback_id, node_id
    )

async def create_context_snapshot(context: SerializableExecutionContext) -> str:
    """Create full context snapshot."""
    return await context.create_full_context_snapshot()

def serialize_context(context: SerializableExecutionContext) -> Dict[str, Any]:
    """Serialize complete context."""
    return context.serialize_complete_context()

def deserialize_context(serialized_data: Dict[str, Any]) -> SerializableExecutionContext:
    """Deserialize complete context."""
    return SerializableExecutionContext.deserialize_complete_context(serialized_data)

# =============================================================================
# SIGNAL OPERATIONS
# =============================================================================

async def emit_signal(signal_type: Union[SignalType, str], 
                     source_context_id: str = "",
                     target_context_id: str = "", 
                     data: Dict[str, Any] = None,
                     priority: SignalPriority = SignalPriority.NORMAL,
                     include_context: bool = False) -> Dict[str, Any]:
    """Emit signal through global signal line."""
    signal_line = await initialize_signal_line()
    
    signal = SignalPayload(
        signal_type=signal_type if isinstance(signal_type, SignalType) else SignalType.CUSTOM,
        source_id=source_context_id,
        target_id=target_context_id,
        priority=priority,
        data=data or {}
    )
    
    # If including context, we need to use the enhanced Signal class
    if include_context and source_context_id:
        from matts.context_serialization import Signal as EnhancedSignal
        library = initialize_library()
        source_context = library.get_context(source_context_id)
        if source_context:
            enhanced_signal = EnhancedSignal(
                signal_type=signal_type if isinstance(signal_type, SignalType) else SignalType.CUSTOM,
                source_context_id=source_context_id,
                target_context_id=target_context_id,
                priority=priority,
                data=data or {}
            )
            enhanced_signal.attach_serialized_context(source_context)
            
            # Use enhanced signal bus
            from matts.context_serialization import HighPerformanceSignalBus
            signal_bus = HighPerformanceSignalBus()
            return await signal_bus.emit_signal(enhanced_signal)
    
    return await signal_line.emit(signal)

async def transmit_context(source_context_id: str, target_context_id: str = "",
                          additional_data: Dict[str, Any] = None) -> Dict[str, Any]:
    """Transmit context via signal."""
    return await emit_signal(
        SignalType.CONTEXT_UPDATE,
        source_context_id=source_context_id,
        target_context_id=target_context_id,
        data=additional_data,
        priority=SignalPriority.HIGH,
        include_context=True
    )

async def register_signal_callback(callback, 
                                  signal_filters: List[Union[SignalType, str]] = None,
                                  priority_filters: List[SignalPriority] = None) -> str:
    """Register a callback for signal handling."""
    signal_line = await initialize_signal_line()
    
    from matts.signal_system import ObserverPriority
    return await signal_line.register_callback(
        callback,
        priority=ObserverPriority.NORMAL,
        signal_filters=signal_filters,
        priority_filters=priority_filters
    )

# =============================================================================
# PERFORMANCE OPERATIONS
# =============================================================================

def run_garbage_collection() -> Dict[str, int]:
    """Run garbage collection on all contexts."""
    library = initialize_library()
    return library.run_garbage_collection()

def get_performance_stats() -> Dict[str, Any]:
    """Get comprehensive performance statistics."""
    library = initialize_library()
    stats = library.get_library_stats()
    
    # Add signal line stats if available
    global _global_signal_line
    if _global_signal_line:
        stats['signal_line'] = _global_signal_line.get_line_stats()
    
    return stats

async def get_signal_stats() -> Dict[str, Any]:
    """Get signal line statistics."""
    signal_line = await initialize_signal_line()
    return {
        'line_stats': signal_line.get_line_stats(),
        'observer_stats': await signal_line.get_observer_stats()
    }

# =============================================================================
# GENERATOR COMPOSITION FUNCTIONS
# =============================================================================

def create_generator_composer(context_id: str) -> AdvancedGeneratorComposer:
    """Create generator composer for context."""
    library = initialize_library()
    context = library.get_context(context_id)
    if not context:
        raise ValueError(f"Context {context_id} not found")
    return AdvancedGeneratorComposer(context)

def register_generator_factory(context_id: str, name: str, factory_func):
    """Register a generator factory with a context's composer."""
    composer = create_generator_composer(context_id)
    composer.register_generator_factory(name, factory_func)
    return composer

# =============================================================================
# RELATIONSHIP MANAGEMENT
# =============================================================================

def add_relationship(source_id: str, target_id: str, 
                    relationship_type: RelationshipType,
                    properties: Dict[str, Any] = None) -> str:
    """Add relationship to global graph."""
    library = initialize_library()
    
    # Get or create relationship graph
    if not hasattr(library, 'relationship_graph'):
        library.relationship_graph = BasicRelationshipGraph()
    
    from matts.graph_system import GraphEdge
    edge = GraphEdge(
        source_id=source_id,
        target_id=target_id,
        relationship_type=relationship_type,
        properties=properties or {}
    )
    
    library.relationship_graph.add_edge(edge)
    return edge.edge_id

def find_related(node_id: str, relationship_type: RelationshipType = None,
                max_depth: int = 3):
    """Find nodes related to given node."""
    library = initialize_library()
    
    if not hasattr(library, 'relationship_graph'):
        return []
    
    return library.relationship_graph.find_related_nodes(
        node_id, relationship_type, max_depth
    )

def get_relationship_path(from_node: str, to_node: str):
    """Get relationship path between two nodes."""
    library = initialize_library()
    
    if not hasattr(library, 'relationship_graph'):
        return None
    
    return library.relationship_graph.get_relationship_path(from_node, to_node)

# =============================================================================
# LIVE CODE SYSTEM INTEGRATION
# =============================================================================

def create_live_code_system(trusted_mode: bool = False):
    """Create live code system for runtime code injection."""
    from matts.live_code_system import CompleteLiveCodeSystem
    return CompleteLiveCodeSystem(trusted_mode)

def create_callback_system(trusted_mode: bool = False):
    """Create callback system with live code injection."""
    from matts.live_code_system import LiveCodeCallbackSystem
    return LiveCodeCallbackSystem(trusted_mode)

def serialize_source_code(source_code: str, name: str = "", 
                         code_type: str = "function", trusted: bool = False):
    """Serialize source code for transmission."""
    from matts.live_code_system import SourceCodeSerializer
    serializer = SourceCodeSerializer()
    return serializer.serialize_source_code(source_code, name, code_type, trusted_source=trusted)

def serialize_function(func, trusted: bool = False):
    """Serialize an existing function."""
    from matts.live_code_system import SourceCodeSerializer
    serializer = SourceCodeSerializer()
    return serializer.serialize_function(func, trusted)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

async def pause_callback_chain(context_id: str, node_id: str = None):
    """Pause callback chain execution."""
    library = initialize_library()
    context = library.get_context(context_id)
    if context:
        target_node = context._find_node_by_id(node_id) if node_id else context.context_chain
        if target_node and hasattr(context, 'iterator_observer'):
            return await context.iterator_observer.pause_callback_chain(
                target_node.node_id, {"action": "pause"}
            )

async def resume_callback_chain(context_id: str, node_id: str = None):
    """Resume paused callback chain execution."""
    library = initialize_library()
    context = library.get_context(context_id)
    if context:
        target_node = context._find_node_by_id(node_id) if node_id else context.context_chain
        if target_node and hasattr(context, 'iterator_observer'):
            return await context.iterator_observer.resume_callback_chain(target_node.node_id)

def add_context_observer(context_id: str, observer, node_id: str = None, inherit: bool = True):
    """Add observer to context node."""
    library = initialize_library()
    context = library.get_context(context_id)
    if context:
        target_node = context._find_node_by_id(node_id) if node_id else context.context_chain
        if target_node:
            target_node.add_observer(observer, inherit)

def hot_swap_context_version(context_id: str, node_id: str, version_id: str, 
                            changes: Dict[str, Any] = None) -> bool:
    """Create and swap to hot-swap version."""
    library = initialize_library()
    context = library.get_context(context_id)
    if context:
        target_node = context._find_node_by_id(node_id)
        if target_node:
            if changes:
                target_node.create_hot_swap_version(version_id, changes)
            return target_node.hot_swap_to_version(version_id)
    return False

async def update_context_data(context_id: str, updates: Dict[str, Any], node_id: str = None):
    """Update context data for a specific node."""
    library = initialize_library()
    context = library.get_context(context_id)
    if context:
        target_node = context._find_node_by_id(node_id) if node_id else context.context_chain
        if target_node:
            await target_node.update_context_data(updates)

def get_context_data(context_id: str, node_id: str = None) -> Dict[str, Any]:
    """Get context data from a specific node."""
    library = initialize_library()
    context = library.get_context(context_id)
    if context:
        target_node = context._find_node_by_id(node_id) if node_id else context.context_chain
        if target_node:
            return target_node.context_data.copy()
    return {}

def get_context_snapshots(context_id: str, node_id: str = None) -> List[str]:
    """Get list of snapshot IDs for a context node."""
    library = initialize_library()
    context = library.get_context(context_id)
    if context:
        target_node = context._find_node_by_id(node_id) if node_id else context.context_chain
        if target_node and hasattr(target_node, 'snapshots'):
            return [snapshot.snapshot_id for snapshot in target_node.snapshots]
    return []

# =============================================================================
# BATCH OPERATIONS
# =============================================================================

async def create_multiple_contexts(count: int, prefix: str = "context") -> List[str]:
    """Create multiple contexts efficiently."""
    contexts = []
    for i in range(count):
        context = create_context(f"{prefix}_{i}")
        contexts.append(context.context_id)
    return contexts

async def transmit_to_multiple_contexts(source_context_id: str, 
                                      target_context_ids: List[str],
                                      data: Dict[str, Any] = None) -> Dict[str, Any]:
    """Transmit context to multiple targets."""
    results = {}
    
    for target_id in target_context_ids:
        try:
            result = await transmit_context(source_context_id, target_id, data)
            results[target_id] = result
        except Exception as e:
            results[target_id] = {'error': str(e)}
    
    return results

def batch_hot_swap(context_ids: List[str], version_changes: Dict[str, Dict[str, Any]]) -> Dict[str, bool]:
    """Perform hot-swap on multiple contexts."""
    results = {}
    
    for context_id in context_ids:
        changes = version_changes.get(context_id, {})
        if changes:
            version_id = f"batch_swap_{datetime.now().timestamp()}"
            success = hot_swap_context_version(context_id, None, version_id, changes)
            results[context_id] = success
        else:
            results[context_id] = False
    
    return results

# =============================================================================
# DEBUGGING AND MONITORING
# =============================================================================

def get_detailed_context_info(context_id: str) -> Dict[str, Any]:
    """Get detailed information about a context."""
    library = initialize_library()
    context = library.get_context(context_id)
    
    if not context:
        return {'error': f'Context {context_id} not found'}
    
    info = {
        'context_id': context.context_id,
        'created_at': context.created_at.isoformat() if hasattr(context, 'created_at') else None,
        'current_snapshot_id': getattr(context, 'current_snapshot_id', None),
        'serialization_metadata_count': len(getattr(context, 'serialization_metadata', {})),
        'global_environment_size': len(getattr(context, 'global_environment', {})),
        'scoped_access_rules': getattr(context, 'scoped_access_rules', {}),
    }
    
    # Add node information if available
    if hasattr(context, 'context_chain') and context.context_chain:
        info['root_node_id'] = context.context_chain.node_id
        info['root_node_state'] = context.context_chain.state.name
        info['root_node_data_keys'] = list(context.context_chain.context_data.keys())
        info['child_node_count'] = len(context.context_chain._child_node_ids)
        
        # Add observer information
        info['observer_count'] = len(context.context_chain.observers)
        info['inherited_observer_count'] = len(context.context_chain.inherited_observers)
    
    return info

def list_all_contexts() -> List[Dict[str, Any]]:
    """List all contexts with basic information."""
    library = initialize_library()
    
    contexts_info = []
    for context_id, context in library.contexts.items():
        basic_info = {
            'context_id': context_id,
            'created_at': context.created_at.isoformat() if hasattr(context, 'created_at') else None,
            'has_root_node': hasattr(context, 'context_chain') and context.context_chain is not None,
            'serialization_metadata_count': len(getattr(context, 'serialization_metadata', {}))
        }
        contexts_info.append(basic_info)
    
    return contexts_info

async def health_check() -> Dict[str, Any]:
    """Perform comprehensive health check on the library."""
    library = initialize_library()
    
    health = {
        'library_status': 'healthy',
        'context_count': len(library.contexts),
        'garbage_collection_needed': False,
        'signal_line_status': 'unknown',
        'performance_metrics': get_performance_stats(),
        'timestamp': datetime.now().isoformat()
    }
    
    # Check if garbage collection is needed
    try:
        gc_results = run_garbage_collection()
        health['garbage_collection_results'] = gc_results
        health['garbage_collection_needed'] = gc_results['total_collected'] > 0
    except Exception as e:
        health['garbage_collection_error'] = str(e)
    
    # Check signal line status
    try:
        global _global_signal_line
        if _global_signal_line:
            health['signal_line_status'] = 'running' if _global_signal_line._running else 'stopped'
            health['signal_line_stats'] = _global_signal_line.get_line_stats()
        else:
            health['signal_line_status'] = 'not_initialized'
    except Exception as e:
        health['signal_line_error'] = str(e)
    
    return health

# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def quick_start(context_id: str = "main", signal_line_id: str = "main") -> tuple:
    """Quick start: Initialize library, create context and signal line."""
    library = initialize_library()
    signal_line = await initialize_signal_line(signal_line_id)
    context = library.create_context(context_id)
    
    return library, signal_line, context

async def demo_setup() -> Dict[str, Any]:
    """Setup for running demonstrations."""
    library, signal_line, context = await quick_start("demo_context", "demo_signals")
    
    # Add some sample data
    await update_context_data(context.context_id, {
        'demo_mode': True,
        'setup_time': datetime.now().isoformat(),
        'sample_data': {'value': 42, 'message': 'Demo setup complete'}
    })
    
    return {
        'library': library,
        'signal_line': signal_line,
        'context': context,
        'context_id': context.context_id,
        'signal_line_id': signal_line.line_id
    }