#!/usr/bin/env python3
"""
Context Serialization for Signal Transmission

This module handles serializing complete execution contexts and passing them
through signals with compression and performance optimization.
"""

import ast
import marshal
import pickle
import dill
import base64
import hashlib
import uuid
import sys
import inspect
import types
import importlib
import threading
import weakref
import traceback
import copy
import gc
import time
import zlib
import json
import os
import platform
from typing import Dict, Any, List, Optional, Callable, Union, Tuple, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum, auto
from collections import defaultdict, deque
from abc import ABC, abstractmethod
import functools
import asyncio
from concurrent.futures import ThreadPoolExecutor

from matts.signal_system import SignalType, SignalPriority, Observer, SignalPayload

# =============================================================================
# SERIALIZED CONTEXT METADATA FOR SIGNALS
# =============================================================================

@dataclass
class SerializedContextMetadata:
    """Metadata containing serialized execution context for signal transmission."""
    context_id: str
    serialized_context: str  # Base64 encoded complete context
    context_snapshot_id: str
    dependencies: Dict[str, Any]
    
    # Compression for performance
    compression_type: str = "zlib"
    original_size: int = 0
    compressed_size: int = 0
    
    # Portability information
    python_version: str = field(default_factory=lambda: f"{sys.version_info.major}.{sys.version_info.minor}")
    platform_info: Dict[str, str] = field(default_factory=dict)
    bundled_modules: List[str] = field(default_factory=list)
    
    def compress_context(self, context_data: bytes) -> str:
        """Compress context data for efficient transmission."""
        self.original_size = len(context_data)
        
        if self.compression_type == "zlib":
            compressed = zlib.compress(context_data, level=6)
        else:
            compressed = context_data
        
        self.compressed_size = len(compressed)
        return base64.b64encode(compressed).decode('utf-8')
    
    def decompress_context(self, compressed_data: str) -> bytes:
        """Decompress context data."""
        data = base64.b64decode(compressed_data.encode('utf-8'))
        
        if self.compression_type == "zlib":
            return zlib.decompress(data)
        else:
            return data


@dataclass  
class Signal:
    """Enhanced signal with serialized context metadata."""
    signal_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    signal_type: SignalType = SignalType.CUSTOM
    priority: SignalPriority = SignalPriority.NORMAL
    source_context_id: str = ""
    target_context_id: str = ""
    
    # Core signal data
    data: Dict[str, Any] = field(default_factory=dict)
    
    # ⭐ CRITICAL: Serialized context in metadata
    context_metadata: Optional[SerializedContextMetadata] = None
    
    # Signal lifecycle
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    processed: bool = False
    
    # Performance tracking
    processing_time: float = 0.0
    transmission_size: int = 0
    
    def attach_serialized_context(self, context: 'SerializableExecutionContext'):
        """Attach serialized context to signal metadata."""
        # Serialize the complete context
        serialized = context.serialize_complete_context()
        context_bytes = pickle.dumps(serialized)
        
        # Create metadata
        self.context_metadata = SerializedContextMetadata(
            context_id=context.context_id,
            serialized_context="",  # Will be set below
            context_snapshot_id=context.current_snapshot_id or "",
            dependencies={}  # Empty dependencies for now
        )
        
        # Compress and encode
        self.context_metadata.serialized_context = self.context_metadata.compress_context(context_bytes)
        self.transmission_size = self.context_metadata.compressed_size
        
        # Add platform info for portability
        self.context_metadata.platform_info = {
            'system': platform.system(),
            'python_version': platform.python_version(),
            'architecture': platform.architecture()[0]
        }
    
    def extract_serialized_context(self) -> Optional['SerializableExecutionContext']:
        """Extract and deserialize context from signal metadata."""
        if not self.context_metadata:
            return None
        
        try:
            # Import here to avoid circular imports
            from matts.context_system import SerializableExecutionContext
            
            # Decompress context data
            context_bytes = self.context_metadata.decompress_context(
                self.context_metadata.serialized_context
            )
            
            # Deserialize context
            serialized_data = pickle.loads(context_bytes)
            return SerializableExecutionContext.deserialize_complete_context(serialized_data)
            
        except Exception as e:
            print(f"Failed to extract context from signal: {e}")
            return None


# =============================================================================
# CONTEXT-AWARE SIGNAL OBSERVER
# =============================================================================

class ContextAwareSignalObserver(Observer):
    """Signal observer that can extract and use serialized contexts."""
    
    def __init__(self, signal_filters: List[SignalType] = None):
        super().__init__()
        self.signal_filters = signal_filters or []
        self.extracted_contexts: Dict[str, 'SerializableExecutionContext'] = {}
    
    async def on_signal(self, signal: SignalPayload) -> bool:
        """Process signal and extract context if present."""
        if self.signal_filters and signal.signal_type not in self.signal_filters:
            return False
        
        # Check if this is our enhanced Signal type with context metadata
        if hasattr(signal, 'context_metadata') and signal.context_metadata:
            context = signal.extract_serialized_context()
            if context:
                self.extracted_contexts[context.context_id] = context
                await self.on_context_extracted(signal, context)
                return True
        
        return await self.on_signal_without_context(signal)
    
    async def on_context_extracted(self, signal: 'Signal', context: 'SerializableExecutionContext'):
        """Handle signal with extracted context."""
        print(f"Extracted context {context.context_id} from signal {signal.signal_id}")
    
    async def on_signal_without_context(self, signal: SignalPayload) -> bool:
        """Handle signal without context."""
        return False


# =============================================================================
# PERFORMANCE-OPTIMIZED DEPENDENCY BUNDLER
# =============================================================================

class FastDependencyBundler:
    """High-performance dependency bundler with caching."""
    
    def __init__(self):
        self.dependency_cache: Dict[str, Set[str]] = {}
        self.bundle_cache: Dict[str, Dict[str, Any]] = {}
        self.module_source_cache: Dict[str, str] = {}
        
    def detect_dependencies_cached(self, source_code: str, cache_key: str = None) -> Set[str]:
        """Fast dependency detection with caching."""
        if not cache_key:
            cache_key = hashlib.md5(source_code.encode()).hexdigest()
        
        if cache_key in self.dependency_cache:
            return self.dependency_cache[cache_key]
        
        dependencies = self._extract_dependencies_fast(source_code)
        self.dependency_cache[cache_key] = dependencies
        return dependencies
    
    def _extract_dependencies_fast(self, source_code: str) -> Set[str]:
        """Fast dependency extraction using AST."""
        dependencies = set()
        
        try:
            tree = ast.parse(source_code)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    dependencies.update(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    dependencies.add(node.module)
                    
        except SyntaxError:
            # Fallback to regex for malformed code
            import re
            import_pattern = r'(?:from\s+(\S+)\s+)?import\s+([^#\n]+)'
            matches = re.findall(import_pattern, source_code)
            for from_module, imports in matches:
                if from_module:
                    dependencies.add(from_module)
                else:
                    import_names = [name.strip().split(' as ')[0] for name in imports.split(',')]
                    dependencies.update(import_names)
        
        return dependencies
    
    def create_portable_bundle(self, dependencies: Set[str]) -> Dict[str, Any]:
        """Create portable dependency bundle for cross-system compatibility."""
        bundle_key = '_'.join(sorted(dependencies))
        
        if bundle_key in self.bundle_cache:
            return self.bundle_cache[bundle_key]
        
        bundle = {
            'dependencies': {},
            'source_modules': {},
            'binary_modules': {},
            'bundle_metadata': {
                'created_at': datetime.now().isoformat(),
                'python_version': sys.version,
                'platform': sys.platform
            }
        }
        
        for dep_name in dependencies:
            try:
                module = importlib.import_module(dep_name)
                
                dep_info = {
                    'name': dep_name,
                    'version': getattr(module, '__version__', 'unknown'),
                    'file': getattr(module, '__file__', None),
                    'is_builtin': dep_name in sys.builtin_module_names,
                    'package': getattr(module, '__package__', None)
                }
                
                # Try to bundle source for non-stdlib modules
                if (not self._is_stdlib_module(dep_name) and 
                    not dep_info['is_builtin'] and 
                    dep_info['file']):
                    
                    try:
                        with open(dep_info['file'], 'r', encoding='utf-8') as f:
                            source = f.read()
                            bundle['source_modules'][dep_name] = source
                            dep_info['bundled_source'] = True
                    except:
                        dep_info['bundled_source'] = False
                
                bundle['dependencies'][dep_name] = dep_info
                
            except ImportError:
                bundle['dependencies'][dep_name] = {
                    'name': dep_name,
                    'missing': True,
                    'needs_installation': True
                }
        
        self.bundle_cache[bundle_key] = bundle
        return bundle
    
    def _is_stdlib_module(self, module_name: str) -> bool:
        """Fast stdlib module detection."""
        stdlib_prefixes = {
            'os', 'sys', 'json', 'pickle', 'base64', 'hashlib', 'uuid',
            'time', 'datetime', 'threading', 'asyncio', 'functools',
            'itertools', 'collections', 'dataclasses', 'enum', 'abc',
            'typing', 'inspect', 'traceback', 'copy', 'gc', 're',
            'math', 'random', 'string', 'io', 'pathlib', 'urllib',
            'http', 'email', 'xml', 'html', 'sqlite3', 'logging'
        }
        return module_name.split('.')[0] in stdlib_prefixes


# =============================================================================
# OPTIMIZED SERIALIZER
# =============================================================================

class OptimizedSerializer:
    """High-performance serializer with hot-swapping support."""
    
    def __init__(self):
        self.serialization_cache: Dict[str, bytes] = {}
        self.hot_swap_cache: Dict[str, Any] = {}
        self.compression_cache: Dict[str, bytes] = {}
        
        # Performance tracking
        self.serialization_times: List[float] = []
        self.cache_hit_rate = 0.0
        self.cache_requests = 0
        self.cache_hits = 0
    
    def serialize_fast(self, obj: Any, cache_key: str = None, 
                      use_compression: bool = True) -> bytes:
        """Fast serialization with caching and compression."""
        start_time = time.time()
        
        # Check cache first
        if cache_key:
            self.cache_requests += 1
            if cache_key in self.serialization_cache:
                self.cache_hits += 1
                self.cache_hit_rate = self.cache_hits / self.cache_requests
                return self.serialization_cache[cache_key]
        
        # Serialize using fastest method available
        try:
            serialized = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        except:
            try:
                serialized = dill.dumps(obj)
            except Exception as e:
                serialized = pickle.dumps(str(obj))
        
        # Compress if requested and beneficial
        if use_compression and len(serialized) > 1024:
            compressed = zlib.compress(serialized, level=1)
            if len(compressed) < len(serialized) * 0.9:
                serialized = b'COMPRESSED:' + compressed
        
        # Cache result
        if cache_key:
            self.serialization_cache[cache_key] = serialized
        
        # Track performance
        serialization_time = time.time() - start_time
        self.serialization_times.append(serialization_time)
        if len(self.serialization_times) > 1000:
            self.serialization_times = self.serialization_times[-500:]
        
        return serialized
    
    def deserialize_fast(self, data: bytes, cache_key: str = None) -> Any:
        """Fast deserialization with decompression."""
        # Handle compression
        if data.startswith(b'COMPRESSED:'):
            data = zlib.decompress(data[11:])
        
        # Deserialize
        try:
            return pickle.loads(data)
        except:
            try:
                return dill.loads(data)  
            except Exception as e:
                return f"Deserialization failed: {e}"
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get serialization performance metrics."""
        if not self.serialization_times:
            return {'average_time': 0, 'cache_hit_rate': 0}
        
        return {
            'average_serialization_time': sum(self.serialization_times) / len(self.serialization_times),
            'cache_hit_rate': self.cache_hit_rate,
            'cache_size': len(self.serialization_cache),
            'total_serializations': len(self.serialization_times)
        }


# =============================================================================
# HIGH-PERFORMANCE SIGNAL BUS WITH CONTEXT SUPPORT
# =============================================================================

class HighPerformanceSignalBus:
    """High-performance signal bus with context serialization support."""
    
    def __init__(self, max_workers: int = 4):
        self.observers: Dict[SignalType, List[Observer]] = defaultdict(list)
        self.signal_queue: Dict[SignalPriority, deque] = {
            priority: deque() for priority in SignalPriority
        }
        
        # Performance optimization
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.processing_stats = {
            'signals_processed': 0,
            'contexts_transmitted': 0,
            'total_transmission_size': 0,
            'average_processing_time': 0.0
        }
        
        self.processing_lock = threading.RLock()
    
    def register_observer(self, signal_type: SignalType, observer: Observer):
        """Register observer for specific signal type."""
        self.observers[signal_type].append(observer)
    
    async def emit_signal(self, signal: Signal) -> Dict[str, Any]:
        """Emit signal with performance tracking."""
        start_time = time.time()
        
        with self.processing_lock:
            self.signal_queue[signal.priority].append(signal)
        
        result = await self._process_signal(signal)
        
        processing_time = time.time() - start_time
        signal.processing_time = processing_time
        
        with self.processing_lock:
            self.processing_stats['signals_processed'] += 1
            if signal.context_metadata:
                self.processing_stats['contexts_transmitted'] += 1
                self.processing_stats['total_transmission_size'] += signal.transmission_size
            
            total_signals = self.processing_stats['signals_processed']
            current_avg = self.processing_stats['average_processing_time']
            self.processing_stats['average_processing_time'] = (
                (current_avg * (total_signals - 1) + processing_time) / total_signals
            )
        
        return result
    
    async def _process_signal(self, signal: Signal) -> Dict[str, Any]:
        """Process signal through registered observers."""
        signal_observers = self.observers.get(signal.signal_type, [])
        
        if not signal_observers:
            return {'handled': False, 'observers_notified': 0}
        
        tasks = []
        for observer in signal_observers:
            # Convert our Signal to SignalPayload for compatibility
            signal_payload = SignalPayload(
                signal_id=signal.signal_id,
                signal_type=signal.signal_type,
                source_id=signal.source_context_id,
                target_id=signal.target_context_id,
                priority=signal.priority,
                data=signal.data,
                metadata=asdict(signal.context_metadata) if signal.context_metadata else {}
            )
            task = asyncio.create_task(observer.on_signal(signal_payload))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        handled_count = sum(1 for result in results if result is True)
        signal.processed = handled_count > 0
        
        return {
            'handled': signal.processed,
            'observers_notified': len(signal_observers),
            'successful_handlers': handled_count
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get signal bus performance statistics."""
        with self.processing_lock:
            return self.processing_stats.copy()


# =============================================================================
# SERIALIZABLE EXECUTION CONTEXT WITH ENHANCED PORTABILITY
# =============================================================================

class SerializableExecutionContextWithPortability:
    """Enhanced execution context with complete portability features."""
    
    def __init__(self, context_id: str = None):
        self.context_id = context_id or str(uuid.uuid4())
        self.created_at = datetime.now()
        
        # Core components
        self.dependency_bundler = FastDependencyBundler()
        self.serializer = OptimizedSerializer()
        self.signal_bus = HighPerformanceSignalBus()
        
        # Context state
        self.current_snapshot_id: Optional[str] = None
        self.hot_swap_enabled = True
        
        # Portability
        self.execution_thread_id = threading.get_ident()
        self.python_interpreter_state: Dict[str, Any] = {}
        
        # Node management
        self.all_nodes: Dict[str, Any] = {}  # Will be populated by context system
        self.root_node = None  # Will be set by context system
        self.current_node = None  # Will be set by context system
        
    def bundle_python_interpreter_state(self) -> Dict[str, Any]:
        """Bundle essential Python interpreter state for portability."""
        return {
            'thread_id': self.execution_thread_id,
            'sys_path': sys.path.copy(),
            'sys_modules_keys': list(sys.modules.keys()),
            'current_working_directory': os.getcwd() if hasattr(os, 'getcwd') else None,
            'environment_variables': dict(os.environ) if hasattr(os, 'environ') else {},
            'python_version': sys.version,
            'platform_info': {
                'system': platform.system(),
                'machine': platform.machine(),
                'processor': platform.processor(),
                'architecture': platform.architecture()[0]
            }
        }
    
    def get_dependency_bundle(self) -> Dict[str, Any]:
        """Get bundled dependencies for portability."""
        all_dependencies = set()
        
        # Collect dependencies from all nodes that have source code
        for node in self.all_nodes.values():
            if hasattr(node, 'execution_environment') and 'source_code' in node.execution_environment:
                source = node.execution_environment['source_code']
                deps = self.dependency_bundler.detect_dependencies_cached(source, node.node_id)
                all_dependencies.update(deps)
        
        return self.dependency_bundler.create_portable_bundle(all_dependencies)
    
    def serialize_for_transmission(self) -> Dict[str, Any]:
        """Serialize complete context for signal transmission."""
        return {
            'context_id': self.context_id,
            'created_at': self.created_at.isoformat(),
            'execution_thread_id': self.execution_thread_id,
            
            # Core context data
            'root_node': self.root_node.get_portable_execution_context() if self.root_node else None,
            'all_nodes': {
                node_id: (node.get_portable_execution_context() if hasattr(node, 'get_portable_execution_context') else str(node))
                for node_id, node in self.all_nodes.items()
            },
            
            # Portability data
            'python_interpreter_state': self.bundle_python_interpreter_state(),
            'dependency_bundle': self.get_dependency_bundle(),
            
            # Performance data
            'current_snapshot_id': self.current_snapshot_id,
            'serialization_metadata': {
                'serialized_at': datetime.now().isoformat(),
                'serializer_performance': self.serializer.get_performance_metrics(),
                'signal_bus_performance': self.signal_bus.get_performance_stats()
            }
        }
    
    async def emit_context_signal(self, signal_type: SignalType, data: Dict[str, Any] = None,
                                 include_serialized_context: bool = True) -> Dict[str, Any]:
        """Emit signal with optional serialized context."""
        signal = Signal(
            signal_type=signal_type,
            source_context_id=self.context_id,
            data=data or {}
        )
        
        if include_serialized_context:
            signal.attach_serialized_context(self)
        
        return await self.signal_bus.emit_signal(signal)