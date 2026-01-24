#!/usr/bin/env python3
"""
Serializable Execution Context System

Core architecture for serializable execution contexts with:
- Context chains: Pure state management (with observer pattern)
- Callback chains: Pure execution logic  
- Signal system: Communication coordination
- Observer integration and inheritance
- Hot-swapping and performance optimization
"""

import ast
import dis
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
import os
import platform
from typing import Dict, Any, List, Optional, Callable, Union, Tuple, Generator, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum, auto
from collections import defaultdict, deque
from abc import ABC, abstractmethod
import functools
import asyncio

from matts.signal_system import SignalType, SignalPriority, Observer, SignalPayload

# =============================================================================
# CORE ENUMS AND STATE MANAGEMENT
# =============================================================================

class ContextState(Enum):
    """Execution context states."""
    CREATED = auto()
    INITIALIZING = auto()
    ACTIVE = auto()
    SUSPENDED = auto()
    SERIALIZING = auto()
    DESERIALIZING = auto()
    HOT_SWAPPING = auto()
    COMPLETED = auto()
    ERROR = auto()
    DESTROYED = auto()


# =============================================================================
# DEPENDENCY AUTO-DETECTION & BUNDLING
# =============================================================================

class DependencyBundler:
    """Auto-detects and bundles dependencies for serializable contexts."""
    
    def __init__(self):
        self.detected_dependencies: Dict[str, Set[str]] = {}
        self.bundled_modules: Dict[str, bytes] = {}
        self.import_graph: Dict[str, List[str]] = {}
        
    def detect_dependencies(self, source_code: str) -> Set[str]:
        """Auto-detect all dependencies from source code."""
        dependencies = set()
        
        try:
            tree = ast.parse(source_code)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        dependencies.add(alias.name)
                        
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        dependencies.add(node.module)
                        
                elif isinstance(node, ast.Call):
                    # Detect dynamic imports like importlib.import_module
                    if (isinstance(node.func, ast.Attribute) and 
                        node.func.attr == 'import_module'):
                        if node.args and isinstance(node.args[0], ast.Str):
                            dependencies.add(node.args[0].s)
                            
        except Exception as e:
            print(f"Dependency detection failed: {e}")
            
        return dependencies
    
    def bundle_dependencies(self, dependencies: Set[str]) -> Dict[str, Any]:
        """Bundle dependencies for serialization."""
        bundled = {}
        
        for dep_name in dependencies:
            try:
                # Try to import and get module info
                module = importlib.import_module(dep_name)
                
                bundled[dep_name] = {
                    'name': dep_name,
                    'file': getattr(module, '__file__', None),
                    'version': getattr(module, '__version__', 'unknown'),
                    'package': getattr(module, '__package__', None),
                    'is_builtin': dep_name in sys.builtin_module_names,
                    'is_stdlib': self._is_stdlib_module(dep_name)
                }
                
                # For non-stdlib modules, we might need to bundle source
                if not bundled[dep_name]['is_stdlib'] and not bundled[dep_name]['is_builtin']:
                    bundled[dep_name]['needs_bundling'] = True
                    
            except ImportError:
                bundled[dep_name] = {
                    'name': dep_name,
                    'missing': True,
                    'needs_bundling': True
                }
                
        return bundled
    
    def _is_stdlib_module(self, module_name: str) -> bool:
        """Check if module is part of standard library."""
        stdlib_modules = {
            'os', 'sys', 'json', 'pickle', 'base64', 'hashlib', 'uuid',
            'time', 'datetime', 'threading', 'asyncio', 'functools',
            'itertools', 'collections', 'dataclasses', 'enum', 'abc',
            'typing', 'inspect', 'traceback', 'copy', 'gc', 're'
        }
        return module_name.split('.')[0] in stdlib_modules


# =============================================================================
# OBSERVER PATTERN INTEGRATION
# =============================================================================

class ContextObserver(ABC):
    """Abstract observer for context changes."""
    
    @abstractmethod
    async def on_context_state_change(self, context_id: str, old_state: ContextState, 
                                     new_state: ContextState, context_data: Dict[str, Any]):
        """Called when context state changes."""
        pass
    
    @abstractmethod
    async def on_context_data_change(self, context_id: str, changed_keys: Set[str], 
                                   context_data: Dict[str, Any]):
        """Called when context data changes."""
        pass


class SignalAwareContextObserver(ContextObserver):
    """Context observer that also listens to signals."""
    
    def __init__(self, signal_filters: List[str] = None):
        self.signal_filters = signal_filters or []
        self.context_validity_rules: Dict[str, Callable] = {}
    
    def add_validity_rule(self, rule_name: str, rule_func: Callable[[Dict[str, Any]], bool]):
        """Add rule to check if context state is valid."""
        self.context_validity_rules[rule_name] = rule_func
    
    async def on_signal_received(self, signal_type: str, signal_data: Dict[str, Any], 
                               context_id: str):
        """Handle incoming signals and validate context state."""
        if not self.signal_filters or signal_type in self.signal_filters:
            # Check context validity rules
            for rule_name, rule_func in self.context_validity_rules.items():
                try:
                    is_valid = rule_func(signal_data)
                    if not is_valid:
                        print(f"Context {context_id} validity rule '{rule_name}' failed")
                        # Could trigger context state change or correction
                except Exception as e:
                    print(f"Context validity rule '{rule_name}' error: {e}")
    
    async def on_context_state_change(self, context_id: str, old_state: ContextState, 
                                     new_state: ContextState, context_data: Dict[str, Any]):
        """Default implementation - can be overridden."""
        print(f"Context {context_id} state: {old_state.name} -> {new_state.name}")
    
    async def on_context_data_change(self, context_id: str, changed_keys: Set[str], 
                                   context_data: Dict[str, Any]):
        """Default implementation - can be overridden."""
        print(f"Context {context_id} data changed: {changed_keys}")


class CompositeObserver:
    """Composite observer that simplifies inheritance patterns."""
    
    def __init__(self):
        self.child_observers: List[ContextObserver] = []
        self.observer_filters: Dict[str, Callable] = {}
    
    def add_observer(self, observer: ContextObserver, filter_func: Callable = None):
        """Add child observer with optional filter."""
        self.child_observers.append(observer)
        if filter_func:
            observer_id = getattr(observer, 'observer_id', str(id(observer)))
            self.observer_filters[observer_id] = filter_func
    
    async def notify_all(self, event_type: str, *args, **kwargs):
        """Notify all child observers with filtering."""
        tasks = []
        
        for observer in self.child_observers:
            observer_id = getattr(observer, 'observer_id', str(id(observer)))
            filter_func = self.observer_filters.get(observer_id)
            
            # Apply filter if exists
            if not filter_func or filter_func(event_type, *args, **kwargs):
                if hasattr(observer, f'on_{event_type}'):
                    method = getattr(observer, f'on_{event_type}')
                    if asyncio.iscoroutinefunction(method):
                        tasks.append(method(*args, **kwargs))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


class CallbackIteratorObserver(ContextObserver):
    """Special observer for pausing/resuming callback chains."""
    
    def __init__(self):
        self.paused_chains: Dict[str, Dict[str, Any]] = {}
        self.execution_checkpoints: Dict[str, List[Dict[str, Any]]] = {}
        
    async def pause_callback_chain(self, context_id: str, checkpoint_data: Dict[str, Any]):
        """Pause callback chain execution at current point."""
        self.paused_chains[context_id] = {
            'paused_at': datetime.now(),
            'checkpoint_data': checkpoint_data,
            'resume_state': 'paused'
        }
        
        if context_id not in self.execution_checkpoints:
            self.execution_checkpoints[context_id] = []
        
        self.execution_checkpoints[context_id].append(checkpoint_data)
    
    async def resume_callback_chain(self, context_id: str) -> Optional[Dict[str, Any]]:
        """Resume paused callback chain."""
        if context_id in self.paused_chains:
            resume_data = self.paused_chains[context_id]['checkpoint_data']
            del self.paused_chains[context_id]
            return resume_data
        return None
    
    async def on_context_state_change(self, context_id: str, old_state: ContextState, 
                                     new_state: ContextState, context_data: Dict[str, Any]):
        """Handle context state changes for pause/resume logic."""
        if new_state == ContextState.SUSPENDED:
            await self.pause_callback_chain(context_id, context_data)
        elif old_state == ContextState.SUSPENDED and new_state == ContextState.ACTIVE:
            await self.resume_callback_chain(context_id)
    
    async def on_context_data_change(self, context_id: str, changed_keys: Set[str], 
                                   context_data: Dict[str, Any]):
        """Monitor for pause/resume triggers in data changes."""
        if 'execution_control' in changed_keys:
            control = context_data.get('execution_control', {})
            if control.get('action') == 'pause':
                await self.pause_callback_chain(context_id, context_data)
            elif control.get('action') == 'resume':
                await self.resume_callback_chain(context_id)


# =============================================================================
# FULL STATE SNAPSHOT SYSTEM
# =============================================================================

@dataclass
class ContextSnapshot:
    """Full state snapshot of execution context."""
    snapshot_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    context_id: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Full state capture
    context_data: Dict[str, Any] = field(default_factory=dict)
    local_variables: Dict[str, Any] = field(default_factory=dict)
    closure_variables: Dict[str, Any] = field(default_factory=dict)
    generator_states: Dict[str, Any] = field(default_factory=dict)
    
    # Execution state
    call_stack_info: List[Dict[str, Any]] = field(default_factory=list)
    execution_position: Dict[str, Any] = field(default_factory=dict)
    
    # Dependency information
    bundled_dependencies: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    snapshot_type: str = "full"  # full, incremental, event-triggered
    parent_snapshot_id: Optional[str] = None
    
    def calculate_diff(self, other_snapshot: 'ContextSnapshot') -> Dict[str, Any]:
        """Calculate incremental diff between snapshots."""
        diff = {
            'snapshot_id': self.snapshot_id,
            'parent_snapshot_id': other_snapshot.snapshot_id,
            'timestamp': self.timestamp,
            'changes': {}
        }
        
        # Data changes
        for key, value in self.context_data.items():
            if key not in other_snapshot.context_data or other_snapshot.context_data[key] != value:
                diff['changes'][f'context_data.{key}'] = {
                    'old': other_snapshot.context_data.get(key),
                    'new': value
                }
        
        # Variable changes
        for key, value in self.local_variables.items():
            if key not in other_snapshot.local_variables or other_snapshot.local_variables[key] != value:
                diff['changes'][f'local_variables.{key}'] = {
                    'old': other_snapshot.local_variables.get(key),
                    'new': value
                }
        
        return diff
    
    def serialize_snapshot(self) -> Dict[str, Any]:
        """Serialize snapshot for transmission/storage."""
        return {
            'snapshot_id': self.snapshot_id,
            'context_id': self.context_id,
            'timestamp': self.timestamp.isoformat(),
            'context_data': self._serialize_data(self.context_data),
            'local_variables': self._serialize_data(self.local_variables),
            'closure_variables': self._serialize_data(self.closure_variables),
            'generator_states': self.generator_states,
            'call_stack_info': self.call_stack_info,
            'execution_position': self.execution_position,
            'bundled_dependencies': self.bundled_dependencies,
            'snapshot_type': self.snapshot_type,
            'parent_snapshot_id': self.parent_snapshot_id
        }
    
    def _serialize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize data with full closure capture."""
        serialized = {}
        
        for key, value in data.items():
            try:
                if callable(value):
                    # Serialize callable with full closure
                    serialized[key] = self._serialize_callable(value)
                elif isinstance(value, types.GeneratorType):
                    # Serialize generator state
                    serialized[key] = self._serialize_generator(value)
                else:
                    # Try pickle first, then dill, then string representation
                    try:
                        pickle.dumps(value)
                        serialized[key] = {'type': 'pickle', 'data': base64.b64encode(pickle.dumps(value)).decode()}
                    except:
                        try:
                            serialized[key] = {'type': 'dill', 'data': base64.b64encode(dill.dumps(value)).decode()}
                        except:
                            serialized[key] = {'type': 'string', 'data': str(value)}
                            
            except Exception as e:
                serialized[key] = {'type': 'error', 'data': f'Serialization failed: {e}'}
        
        return serialized
    
    def _serialize_callable(self, func: Callable) -> Dict[str, Any]:
        """Serialize callable with full closure capture."""
        try:
            closure_data = {}
            
            if func.__closure__:
                closure_names = func.__code__.co_freevars
                for i, cell in enumerate(func.__closure__):
                    try:
                        var_name = closure_names[i]
                        closure_data[var_name] = cell.cell_contents
                    except (IndexError, ValueError):
                        pass
            
            return {
                'type': 'callable',
                'name': func.__name__,
                'source': inspect.getsource(func) if inspect.getsource(func) else None,
                'closure_data': closure_data,
                'code_object': base64.b64encode(marshal.dumps(func.__code__)).decode(),
                'defaults': func.__defaults__,
                'kwdefaults': func.__kwdefaults__
            }
            
        except Exception as e:
            return {'type': 'callable_error', 'error': str(e)}
    
    def _serialize_generator(self, gen: Generator) -> Dict[str, Any]:
        """Serialize generator state for full restoration."""
        try:
            if gen.gi_frame is None:
                return {'type': 'generator', 'exhausted': True}
            
            return {
                'type': 'generator',
                'exhausted': False,
                'locals': dict(gen.gi_frame.f_locals),
                'instruction_pointer': gen.gi_frame.f_lasti,
                'code_object': base64.b64encode(marshal.dumps(gen.gi_code)).decode(),
                'gi_running': gen.gi_running,
                'gi_yieldfrom': str(gen.gi_yieldfrom) if gen.gi_yieldfrom else None
            }
            
        except Exception as e:
            return {'type': 'generator_error', 'error': str(e)}


# =============================================================================
# CIRCULAR REFERENCE DETECTION
# =============================================================================

class CircularReferenceDetector:
    """Detects circular references in context chains."""
    
    def __init__(self):
        self.visited_nodes: Set[str] = set()
        self.visiting_stack: List[str] = []
    
    def detect_circular_reference(self, node_id: str, get_children_func: Callable) -> Optional[List[str]]:
        """Detect circular reference starting from node_id."""
        if node_id in self.visiting_stack:
            # Found circular reference
            cycle_start = self.visiting_stack.index(node_id)
            return self.visiting_stack[cycle_start:] + [node_id]
        
        if node_id in self.visited_nodes:
            return None
        
        self.visiting_stack.append(node_id)
        
        try:
            children = get_children_func(node_id)
            for child_id in children:
                cycle = self.detect_circular_reference(child_id, get_children_func)
                if cycle:
                    return cycle
        finally:
            self.visiting_stack.pop()
            self.visited_nodes.add(node_id)
        
        return None


# =============================================================================
# CONTEXT CHAIN WITH OBSERVER INHERITANCE
# =============================================================================

class ContextChainNode:
    """Node in context chain with observer pattern integration."""
    
    def __init__(self, node_id: str = None):
        self.node_id = node_id or str(uuid.uuid4())
        self.context_data: Dict[str, Any] = {}
        self.state = ContextState.CREATED
        
        # Observer management
        self.observers: List[ContextObserver] = []
        self.inherited_observers: List[ContextObserver] = []
        
        # Chain structure with weak references
        self.parent_node: Optional[weakref.ref] = None
        self.child_nodes: List[weakref.ref] = []
        self._child_node_ids: List[str] = []  # For serialization
        
        # Callback bindings
        self.bound_callbacks: Dict[str, str] = {}  # callback_id -> code_id
        
        # Snapshot management
        self.snapshots: List[ContextSnapshot] = []
        self.current_snapshot: Optional[ContextSnapshot] = None
        
        # Hot-swapping support
        self.hot_swap_versions: Dict[str, Dict[str, Any]] = {}
        self.current_version = "v1"
        
        # Fast context switching state
        self.cached_execution_state: Dict[str, Any] = {}
        self.last_activity = datetime.now()
        
        # Performance optimization
        self.serialization_dirty = True
        
    def add_observer(self, observer: ContextObserver, inherit_to_children: bool = True):
        """Add observer to this node."""
        self.observers.append(observer)
        
        if inherit_to_children:
            # Add to all child nodes as inherited observer
            for child_ref in self.child_nodes:
                child = child_ref()
                if child:
                    child.inherited_observers.append(observer)
                    child._propagate_inherited_observer(observer)
    
    def _propagate_inherited_observer(self, observer: ContextObserver):
        """Propagate inherited observer down the chain."""
        for child_ref in self.child_nodes:
            child = child_ref()
            if child:
                child.inherited_observers.append(observer)
                child._propagate_inherited_observer(observer)
    
    async def set_state(self, new_state: ContextState):
        """Change state and notify observers."""
        old_state = self.state
        self.state = new_state
        self.last_activity = datetime.now()
        
        # Notify all observers (own + inherited)
        all_observers = self.observers + self.inherited_observers
        
        for observer in all_observers:
            try:
                await observer.on_context_state_change(
                    self.node_id, old_state, new_state, self.context_data
                )
            except Exception as e:
                print(f"Observer notification failed: {e}")
    
    async def update_context_data(self, updates: Dict[str, Any]):
        """Update context data and notify observers."""
        changed_keys = set()
        
        for key, value in updates.items():
            if key not in self.context_data or self.context_data[key] != value:
                changed_keys.add(key)
                self.context_data[key] = value
        
        if changed_keys:
            self.last_activity = datetime.now()
            self.serialization_dirty = True
            
            all_observers = self.observers + self.inherited_observers
            
            for observer in all_observers:
                try:
                    await observer.on_context_data_change(
                        self.node_id, changed_keys, self.context_data
                    )
                except Exception as e:
                    print(f"Observer notification failed: {e}")
    
    def add_child_node(self, child: 'ContextChainNode'):
        """Add child node with observer inheritance and circular reference detection."""
        detector = CircularReferenceDetector()
        
        def get_children(node_id: str) -> List[str]:
            if node_id == self.node_id:
                return self._child_node_ids + [child.node_id]
            elif node_id == child.node_id:
                return []
            else:
                return []
        
        cycle = detector.detect_circular_reference(self.node_id, get_children)
        if cycle:
            raise ValueError(f"Circular reference detected: {' -> '.join(cycle)}")
        
        # Add child with weak reference
        child.parent_node = weakref.ref(self)
        self.child_nodes.append(weakref.ref(child))
        self._child_node_ids.append(child.node_id)
        self.serialization_dirty = True
        self.last_activity = datetime.now()
        
        # Inherit observers
        for observer in self.observers + self.inherited_observers:
            child.inherited_observers.append(observer)
    
    def create_hot_swap_version(self, version_id: str, changes: Dict[str, Any]) -> str:
        """Create hot-swappable version of this node."""
        baseline = {
            'context_data': copy.deepcopy(self.context_data),
            'state': self.state
        }
        
        new_version = copy.deepcopy(baseline)
        new_version['context_data'].update(changes.get('context_data', {}))
        
        self.hot_swap_versions[version_id] = new_version
        return version_id
    
    def hot_swap_to_version(self, version_id: str) -> bool:
        """Hot-swap to specific version with minimal downtime."""
        if version_id not in self.hot_swap_versions:
            return False
        
        old_state = self.state
        self.state = ContextState.HOT_SWAPPING
        
        try:
            version_data = self.hot_swap_versions[version_id]
            self.context_data = version_data['context_data']
            self.current_version = version_id
            self.serialization_dirty = True
            self.last_activity = datetime.now()
            
            self.state = old_state
            return True
            
        except Exception as e:
            print(f"Hot swap failed: {e}")
            self.state = old_state
            return False
    
    def create_snapshot(self, snapshot_type: str = "full") -> str:
        """Create context snapshot."""
        snapshot = ContextSnapshot(
            context_id=self.node_id,
            context_data=copy.deepcopy(self.context_data),
            snapshot_type=snapshot_type
        )
        
        # Capture execution state from current frame
        frame = inspect.currentframe()
        if frame:
            try:
                snapshot.local_variables = dict(frame.f_locals)
                snapshot.call_stack_info = self._capture_call_stack(frame)
            finally:
                del frame
        
        self.snapshots.append(snapshot)
        self.current_snapshot = snapshot
        
        return snapshot.snapshot_id
    
    def _capture_call_stack(self, frame) -> List[Dict[str, Any]]:
        """Capture call stack information."""
        stack_info = []
        current_frame = frame
        
        while current_frame and len(stack_info) < 20:  # Limit depth
            try:
                stack_info.append({
                    'function_name': current_frame.f_code.co_name,
                    'filename': current_frame.f_code.co_filename,
                    'line_number': current_frame.f_lineno,
                    'local_vars': {k: str(v) for k, v in current_frame.f_locals.items() 
                                 if not k.startswith('_')}
                })
                current_frame = current_frame.f_back
            except:
                break
        
        return stack_info
    
    def get_portable_execution_context(self) -> Dict[str, Any]:
        """Get execution context for cross-system portability."""
        return {
            'node_id': self.node_id,
            'context_data': self.context_data,
            'bound_callbacks': self.bound_callbacks,
            'current_version': self.current_version,
            'hot_swap_versions': self.hot_swap_versions,
            'child_node_ids': self._child_node_ids,
            'parent_node_id': self.parent_node().node_id if self.parent_node and self.parent_node() else None,
            'state': self.state.name,
            'last_activity': self.last_activity.isoformat()
        }


# =============================================================================
# SERIALIZABLE EXECUTION CONTEXT - THE CORE
# =============================================================================

class SerializableExecutionContext:
    """The main serializable execution context."""
    
    def __init__(self, context_id: str = None):
        self.context_id = context_id or str(uuid.uuid4())
        
        # Core components
        self.dependency_bundler = DependencyBundler()
        self.context_chain = self._create_root_chain_node()
        self.iterator_observer = CallbackIteratorObserver()
        
        # Global environment with scoped access
        self.global_environment: Dict[str, Any] = {}
        self.scoped_access_rules: Dict[str, List[str]] = {}
        
        # Serialization state
        self.serialization_metadata: Dict[str, Any] = {}
        self.is_serializing = False
        
        # Fast context switching
        self.context_switch_cache: Dict[str, Any] = {}
        
        # Performance state
        self.current_snapshot_id: Optional[str] = None
        self.created_at = datetime.now()
        
        # Add iterator observer to root node
        self.context_chain.add_observer(self.iterator_observer)
    
    def _create_root_chain_node(self) -> ContextChainNode:
        """Create root context chain node."""
        root = ContextChainNode(f"{self.context_id}_root")
        
        # Add signal-aware observer
        signal_observer = SignalAwareContextObserver(['context_update', 'state_change'])
        root.add_observer(signal_observer)
        
        return root
    
    def add_scoped_access_rule(self, context_node_id: str, allowed_modules: List[str]):
        """Add scoped access rule for environment isolation."""
        self.scoped_access_rules[context_node_id] = allowed_modules
    
    def get_scoped_environment(self, context_node_id: str) -> Dict[str, Any]:
        """Get scoped environment for specific context node."""
        allowed_modules = self.scoped_access_rules.get(context_node_id, [])
        
        scoped_env = {}
        for module_name in allowed_modules:
            if module_name in self.global_environment:
                scoped_env[module_name] = self.global_environment[module_name]
        
        return scoped_env
    
    async def bind_callback_with_auto_dependency_detection(self, callback_source: str, 
                                                          callback_id: str,
                                                          node_id: str = None) -> str:
        """Bind callback with automatic dependency detection and bundling."""
        target_node_id = node_id or self.context_chain.node_id
        target_node = self._find_node_by_id(target_node_id)
        
        if not target_node:
            raise ValueError(f"Node {target_node_id} not found")
        
        # Auto-detect dependencies
        dependencies = self.dependency_bundler.detect_dependencies(callback_source)
        bundled_deps = self.dependency_bundler.bundle_dependencies(dependencies)
        
        # Store in global environment
        for dep_name, dep_info in bundled_deps.items():
            if not dep_info.get('missing', False):
                try:
                    module = importlib.import_module(dep_name)
                    self.global_environment[dep_name] = module
                except ImportError:
                    print(f"Could not load dependency: {dep_name}")
        
        # Create code ID and bind
        code_id = str(uuid.uuid4())
        target_node.bound_callbacks[callback_id] = code_id
        
        # Store callback source with metadata
        self.serialization_metadata[code_id] = {
            'source': callback_source,
            'dependencies': bundled_deps,
            'callback_id': callback_id,
            'node_id': target_node_id
        }
        
        return code_id
    
    def _find_node_by_id(self, node_id: str) -> Optional[ContextChainNode]:
        """Find node by ID recursively."""
        def search_node(node: ContextChainNode) -> Optional[ContextChainNode]:
            if node.node_id == node_id:
                return node
            
            for child_ref in node.child_nodes:
                child = child_ref()
                if child:
                    result = search_node(child)
                    if result:
                        return result
            
            return None
        
        return search_node(self.context_chain)
    
    async def create_full_context_snapshot(self) -> str:
        """Create full context snapshot for serialization."""
        # Create snapshot at root level
        snapshot_id = self.context_chain.create_snapshot("full")
        
        # Enhance with serialization metadata
        snapshot = self.context_chain.current_snapshot
        snapshot.bundled_dependencies = {
            code_id: meta['dependencies'] 
            for code_id, meta in self.serialization_metadata.items()
        }
        
        self.current_snapshot_id = snapshot_id
        return snapshot_id
    
    def serialize_complete_context(self) -> Dict[str, Any]:
        """Serialize complete execution context for transmission."""
        self.is_serializing = True
        
        try:
            # Create final snapshot
            snapshot_id = self.context_chain.create_snapshot("serialization")
            
            serialized = {
                'context_id': self.context_id,
                'created_at': self.created_at.isoformat(),
                'timestamp': datetime.now().isoformat(),
                'root_node': self._serialize_node_tree(self.context_chain),
                'global_environment_keys': list(self.global_environment.keys()),
                'scoped_access_rules': self.scoped_access_rules,
                'serialization_metadata': self.serialization_metadata,
                'current_snapshot': self.context_chain.current_snapshot.serialize_snapshot() if self.context_chain.current_snapshot else None,
                'context_switch_cache': self.context_switch_cache,
                'current_snapshot_id': self.current_snapshot_id
            }
            
            return serialized
            
        finally:
            self.is_serializing = False
    
    def _serialize_node_tree(self, node: ContextChainNode) -> Dict[str, Any]:
        """Recursively serialize node tree."""
        return {
            'node_id': node.node_id,
            'context_data': node.context_data,
            'state': node.state.name,
            'bound_callbacks': node.bound_callbacks,
            'current_snapshot_id': node.current_snapshot.snapshot_id if node.current_snapshot else None,
            'current_version': node.current_version,
            'hot_swap_versions': node.hot_swap_versions,
            'child_node_ids': node._child_node_ids,
            'last_activity': node.last_activity.isoformat(),
            'child_nodes': [
                self._serialize_node_tree(child_ref()) 
                for child_ref in node.child_nodes 
                if child_ref() is not None
            ]
        }
    
    @classmethod
    def deserialize_complete_context(cls, serialized_data: Dict[str, Any]) -> 'SerializableExecutionContext':
        """Deserialize complete execution context."""
        context = cls(serialized_data['context_id'])
        context.created_at = datetime.fromisoformat(serialized_data['created_at'])
        
        # Restore serialization metadata
        context.serialization_metadata = serialized_data['serialization_metadata']
        context.scoped_access_rules = serialized_data['scoped_access_rules']
        context.context_switch_cache = serialized_data['context_switch_cache']
        context.current_snapshot_id = serialized_data.get('current_snapshot_id')
        
        # Restore global environment (would need to re-import modules)
        for module_name in serialized_data['global_environment_keys']:
            try:
                context.global_environment[module_name] = importlib.import_module(module_name)
            except ImportError:
                print(f"Could not restore module: {module_name}")
        
        # Restore node tree
        context.context_chain = context._deserialize_node_tree(serialized_data['root_node'])
        
        return context
    
    def _deserialize_node_tree(self, node_data: Dict[str, Any]) -> ContextChainNode:
        """Recursively deserialize node tree."""
        node = ContextChainNode(node_data['node_id'])
        node.context_data = node_data['context_data']
        node.state = ContextState[node_data['state']]
        node.bound_callbacks = node_data['bound_callbacks']
        node.current_version = node_data.get('current_version', 'v1')
        node.hot_swap_versions = node_data.get('hot_swap_versions', {})
        node._child_node_ids = node_data.get('child_node_ids', [])
        node.last_activity = datetime.fromisoformat(node_data['last_activity'])
        
        # Deserialize child nodes
        for child_data in node_data.get('child_nodes', []):
            child_node = self._deserialize_node_tree(child_data)
            child_node.parent_node = weakref.ref(node)
            node.child_nodes.append(weakref.ref(child_node))
        
        return node


# =============================================================================
# GARBAGE COLLECTION SYSTEM
# =============================================================================

class ContextGarbageCollector:
    """Garbage collector for context chains."""
    
    def __init__(self):
        self.collection_threshold = 100
        self.last_collection = datetime.now()
        self.collection_interval = 300
        
    def should_collect(self, context: SerializableExecutionContext) -> bool:
        """Determine if garbage collection should run."""
        # Count all nodes in the context tree
        node_count = self._count_nodes(context.context_chain)
        time_since_last = (datetime.now() - self.last_collection).total_seconds()
        
        return (node_count > self.collection_threshold or 
                time_since_last > self.collection_interval)
    
    def _count_nodes(self, node: ContextChainNode) -> int:
        """Count all nodes in the tree."""
        count = 1  # Count this node
        for child_ref in node.child_nodes:
            child = child_ref()
            if child:
                count += self._count_nodes(child)
        return count
    
    def collect_unused_nodes(self, context: SerializableExecutionContext) -> int:
        """Collect unused nodes from context."""
        collected_count = 0
        nodes_to_remove = []
        
        # Find unused nodes
        self._find_unused_nodes(context.context_chain, context, nodes_to_remove)
        
        # Remove unused nodes
        for node in nodes_to_remove:
            self._remove_node(node)
            collected_count += 1
        
        self.last_collection = datetime.now()
        gc.collect()
        
        return collected_count
    
    def _find_unused_nodes(self, node: ContextChainNode, context: SerializableExecutionContext, 
                          nodes_to_remove: List[ContextChainNode]):
        """Recursively find unused nodes."""
        # Check children first
        for child_ref in node.child_nodes.copy():
            child = child_ref()
            if child:
                if self._is_node_unused(child, context):
                    nodes_to_remove.append(child)
                else:
                    self._find_unused_nodes(child, context, nodes_to_remove)
    
    def _is_node_unused(self, node: ContextChainNode, context: SerializableExecutionContext) -> bool:
        """Check if node is unused and can be collected."""
        # Don't collect root node
        if node == context.context_chain:
            return False
        
        # Don't collect nodes with active observers
        if node.observers:
            return False
        
        # Don't collect nodes with recent activity
        time_since_activity = (datetime.now() - node.last_activity).total_seconds()
        if time_since_activity < 300:  # 5 minutes
            return False
        
        # Check if node has living children
        alive_children = [ref() for ref in node.child_nodes if ref() is not None]
        if alive_children:
            return False
        
        # Check if node has bound callbacks
        if node.bound_callbacks:
            return False
        
        return True
    
    def _remove_node(self, node: ContextChainNode):
        """Remove node from its parent."""
        if node.parent_node and node.parent_node():
            parent = node.parent_node()
            # Remove from parent's child list
            parent.child_nodes = [ref for ref in parent.child_nodes if ref() != node]
            parent._child_node_ids = [id for id in parent._child_node_ids if id != node.node_id]


# =============================================================================
# MAIN LIBRARY CLASS
# =============================================================================

class SerializableContextLibrary:
    """Main library interface for serializable contexts."""
    
    def __init__(self):
        self.contexts: Dict[str, SerializableExecutionContext] = {}
        self.garbage_collector = ContextGarbageCollector()
        
        # Library metadata
        self.created_at = datetime.now()
        
    def create_context(self, context_id: str = None) -> SerializableExecutionContext:
        """Create new serializable execution context."""
        context = SerializableExecutionContext(context_id)
        self.contexts[context.context_id] = context
        return context
    
    def get_context(self, context_id: str) -> Optional[SerializableExecutionContext]:
        """Get context by ID."""
        return self.contexts.get(context_id)
    
    def destroy_context(self, context_id: str) -> bool:
        """Destroy context and clean up resources."""
        if context_id not in self.contexts:
            return False
        
        del self.contexts[context_id]
        return True
    
    def run_garbage_collection(self) -> Dict[str, int]:
        """Run garbage collection on all contexts."""
        total_collected = 0
        collection_results = {}
        
        for context_id, context in self.contexts.items():
            if self.garbage_collector.should_collect(context):
                collected = self.garbage_collector.collect_unused_nodes(context)
                collection_results[context_id] = collected
                total_collected += collected
        
        return {
            'total_collected': total_collected,
            'per_context': collection_results
        }
    
    def get_library_stats(self) -> Dict[str, Any]:
        """Get comprehensive library statistics."""
        return {
            'contexts': {
                'total_contexts': len(self.contexts),
                'context_ids': list(self.contexts.keys())
            },
            'garbage_collection': {
                'last_collection': self.garbage_collector.last_collection.isoformat(),
                'collection_threshold': self.garbage_collector.collection_threshold
            },
            'created_at': self.created_at.isoformat()
        }
