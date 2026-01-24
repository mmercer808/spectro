#!/usr/bin/env python3
"""
Serializable Context Library
===========================

A comprehensive library for serializable execution contexts with:
- Signal system integration with context transmission
- Live code serialization and runtime injection
- Advanced generator composition patterns
- Performance-optimized hot-swapping
- Complete portability across systems

Author: Serializable Context Library Team
Version: 1.0.0
"""

# Core functional interface - most commonly used
from matts.functional_interface import (
    # Context Management
    create_context, get_context, destroy_context,
    bind_callback_with_dependencies, create_context_snapshot,
    serialize_context, deserialize_context,
    
    # Signal Operations
    emit_signal, transmit_context, register_signal_callback,
    
    # Performance Operations
    run_garbage_collection, get_performance_stats, get_signal_stats,
    
    # Generator Composition
    create_generator_composer, register_generator_factory,
    
    # Relationship Management
    add_relationship, find_related, get_relationship_path,
    
    # Live Code System
    create_live_code_system, create_callback_system,
    serialize_source_code, serialize_function,
    
    # Utility Functions
    pause_callback_chain, resume_callback_chain,
    add_context_observer, hot_swap_context_version,
    update_context_data, get_context_data, get_context_snapshots,
    
    # Batch Operations
    create_multiple_contexts, transmit_to_multiple_contexts,
    batch_hot_swap,
    
    # Debugging and Monitoring
    get_detailed_context_info, list_all_contexts, health_check,
    
    # Convenience Functions
    quick_start, demo_setup
)

# Core signal system
from .signal_system import (
    SignalLine, SignalPayload, SignalType, SignalPriority,
    Observer, ObserverPriority, CallbackObserver,
    create_signal_line, signal_handler,
    
    # Additional signal system components
    CircuitState, ObserverStats
)

# Core context system
from .context_system import (
    SerializableExecutionContext, ContextChainNode, ContextState,
    ContextObserver, SerializableContextLibrary,
    
    # Additional context system components
    SignalAwareContextObserver, CompositeObserver, CallbackIteratorObserver,
    ContextSnapshot, CircularReferenceDetector, ContextGarbageCollector
)

# Live code system
from .live_code_system import (
    CompleteLiveCodeSystem, SerializedSourceCode, RuntimeSourceEditor,
    BytecodeExecutionEngine, LiveCodeCallbackSystem,
    
    # Additional live code components
    CodeSerializationMethod, CompleteSerializedCode, SourceCodeSerializer,
    ContextAwareDeserializer, RuntimeCodeCache, SecurityError, DeserializationError
)

# Context serialization
from .context_serialization import (
    SerializedContextMetadata, Signal, ContextAwareSignalObserver,
    HighPerformanceSignalBus, FastDependencyBundler, OptimizedSerializer,
    
    # Additional serialization components
    SerializableExecutionContextWithPortability
)

# Generator system
from .generator_system import (
    AdvancedGeneratorComposer, GeneratorCompositionPattern, GeneratorBranch,
    GeneratorCompositionEngine,
    
    # Additional generator components
    GeneratorStateBranch, create_data_generator_factory,
    create_transformer_generator_factory, create_filter_generator_factory,
    create_aggregator_generator_factory
)

# Graph system
from .graph_system import (
    BasicRelationshipGraph, GraphNode, GraphEdge, RelationshipType
)

# Library metadata
__version__ = "1.0.0"
__author__ = "Serializable Context Library Team"
__description__ = "Complete serializable execution context library with signal integration"
__license__ = "MIT"

# Core exports for common usage - these are the main API
__all__ = [
    # === MAIN LIBRARY CLASS ===
    'SerializableContextLibrary',
    
    # === CORE CLASSES ===
    'SerializableExecutionContext', 'ContextChainNode', 'Signal',
    'SignalLine', 'SignalPayload', 'Observer',
    'AdvancedGeneratorComposer', 'BasicRelationshipGraph',
    
    # === ENUMS ===
    'SignalType', 'SignalPriority', 'ContextState', 
    'ObserverPriority', 'RelationshipType', 'GeneratorCompositionPattern',
    'CodeSerializationMethod', 'CircuitState',
    
    # === FUNCTIONAL API (MOST COMMONLY USED) ===
    # Context Management
    'create_context', 'get_context', 'destroy_context',
    'serialize_context', 'deserialize_context',
    
    # Signal Operations
    'emit_signal', 'transmit_context', 'create_signal_line',
    
    # Live Code System
    'create_live_code_system', 'create_callback_system',
    'serialize_source_code', 'serialize_function',
    
    # Generator Composition
    'create_generator_composer',
    
    # Relationship Management
    'add_relationship', 'find_related',
    
    # Performance and Utilities
    'run_garbage_collection', 'get_performance_stats', 'health_check',
    'quick_start',
    
    # === LIVE CODE SYSTEM ===
    'CompleteLiveCodeSystem', 'LiveCodeCallbackSystem',
    'SerializedSourceCode', 'BytecodeExecutionEngine',
    
    # === ADVANCED COMPONENTS ===
    'ContextObserver', 'ContextSnapshot', 'GeneratorBranch',
    'GraphNode', 'GraphEdge', 'SerializedContextMetadata',
    
    # === FACTORY FUNCTIONS ===
    'create_data_generator_factory', 'create_transformer_generator_factory',
    'create_filter_generator_factory', 'create_aggregator_generator_factory',
    
    # === EXCEPTIONS ===
    'SecurityError', 'DeserializationError'
]

# Initialize global library instance for convenience
_global_library = None

def initialize_library() -> SerializableContextLibrary:
    """Initialize the global library instance."""
    global _global_library
    if _global_library is None:
        _global_library = SerializableContextLibrary()
    return _global_library

def get_library() -> SerializableContextLibrary:
    """Get the global library instance."""
    return initialize_library()

# Convenience function for quick setup
async def quick_library_start(library_id: str = "default") -> tuple:
    """Quick start: Initialize library and create signal line."""
    library = initialize_library()
    signal_line = await create_signal_line(f"{library_id}_signals")
    return library, signal_line

# Version check function
def check_version() -> dict:
    """Get version and compatibility information."""
    import sys
    return {
        'library_version': __version__,
        'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        'python_compatible': sys.version_info >= (3, 8),
        'features': {
            'async_support': True,
            'live_code_injection': True,
            'hot_swapping': True,
            'context_serialization': True,
            'signal_system': True,
            'generator_composition': True,
            'relationship_graphs': True,
            'distributed_computing': True
        }
    }

# Help function
def library_help():
    """Print library help and usage information."""
    print(f"🚀 Serializable Context Library v{__version__}")
    print("=" * 50)
    print()
    print("📚 Quick Start:")
    print("  from matts import create_context, emit_signal, quick_start")
    print("  library, signal_line, context = await quick_start()")
    print()
    print("🔧 Core Functions:")
    print("  • create_context() - Create serializable execution context")
    print("  • emit_signal() - Send signals between contexts")
    print("  • transmit_context() - Send context data via signals")
    print("  • serialize_context() - Serialize context for storage/transmission")
    print("  • create_live_code_system() - Enable live code injection")
    print()
    print("🎮 Examples:")
    print("  from matts.examples import run_complete_library_demo")
    print("  await run_complete_library_demo()")
    print()
    print("📖 Documentation:")
    print("  • Check individual module docstrings")
    print("  • Run examples for real-world usage patterns")
    print("  • Use health_check() for system status")

# Auto-initialization on import (optional)
def _auto_init():
    """Auto-initialize library on import if desired."""
    # Uncomment to auto-initialize:
    # initialize_library()
    pass

_auto_init()

# Demo function for quick testing
async def run_quick_demo():
    """Run a quick demonstration of library features."""
    print("🚀 Quick Library Demo")
    print("-" * 20)
    
    # Quick start
    library, signal_line, context = await quick_start("demo")
    print(f"✅ Created context: {context.context_id}")
    
    # Update context data
    await update_context_data(context.context_id, {
        'demo_data': 'Hello, Serializable Contexts!',
        'timestamp': str(datetime.now())
    })
    
    # Emit signal
    result = await emit_signal(
        SignalType.CUSTOM,
        source_context_id=context.context_id,
        data={'message': 'Demo signal sent!'}
    )
    print(f"✅ Signal emitted: {result}")
    
    # Health check
    health = await health_check()
    print(f"✅ System health: {health.get('library_status', 'unknown')}")
    
    return {
        'library': library,
        'signal_line': signal_line,
        'context': context,
        'demo_completed': True
    }

if __name__ == "__main__":
    # If someone runs the package directly, show help
    library_help()
    print("\nTo run examples:")
    print("  python -m matts.examples.demo_system")
    print("  # or")
    print("  from matts.examples import run_complete_library_demo")
    print("  import asyncio")
    print("  asyncio.run(run_complete_library_demo())")