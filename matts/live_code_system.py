#!/usr/bin/env python3
"""
Live Source Code Serialization & Runtime Decorator Injection System

This system allows serializing Python source code into "blobs" that can be:
- Passed through signal metadata
- Cached for runtime swapping
- Dynamically reconstructed with context objects
- Applied as decorators to callbacks on-the-fly

DANGEROUS BUT POWERFUL: This executes arbitrary code from serialized sources.
Only use in trusted environments or with proper sandboxing.

Key Components:
1. Source Code Serialization (multiple formats)
2. Context-Aware Deserialization
3. Runtime Code Injection Cache
4. Signal Metadata Code Passing
5. Dynamic Decorator Swapping
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
from typing import Dict, Any, List, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import defaultdict
import functools
import traceback
import threading
import weakref

# =============================================================================
# SERIALIZED SOURCE CODE CONTAINER
# =============================================================================

class CodeSerializationMethod(Enum):
    """Different methods for serializing Python code."""
    SOURCE_TEXT = "source_text"        # Raw Python source code
    AST_TREE = "ast_tree"             # Abstract Syntax Tree
    BYTECODE_MARSHAL = "bytecode_marshal"  # Marshaled bytecode
    COMPILED_CODE = "compiled_code"    # Compiled code objects
    PICKLE_FUNCTION = "pickle_function"  # Pickled function objects
    DILL_ENHANCED = "dill_enhanced"    # Dill serialized (handles more cases)

@dataclass
class SerializedSourceCode:
    """Container for serialized Python source code."""
    code_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    code_type: str = "function"  # "function", "decorator_generator", "class", "module"
    
    # Multiple serialization formats (redundancy for robustness)
    source_code: Optional[str] = None
    ast_blob: Optional[str] = None  # Base64 encoded AST
    bytecode_blob: Optional[str] = None  # Base64 encoded bytecode
    pickled_object: Optional[str] = None  # Base64 encoded pickled object
    
    # Context requirements
    required_imports: List[str] = field(default_factory=list)
    required_context_keys: List[str] = field(default_factory=list)
    
    # Security and validation
    code_hash: str = field(default="")
    trusted_source: bool = False
    sandbox_safe: bool = False
    
    # Execution metadata
    serialized_at: datetime = field(default_factory=datetime.now)
    python_version: str = field(default_factory=lambda: f"{sys.version_info.major}.{sys.version_info.minor}")
    
    def calculate_hash(self) -> str:
        """Calculate hash of the source code for integrity checking."""
        if self.source_code:
            return hashlib.sha256(self.source_code.encode()).hexdigest()
        return ""
    
    def to_message_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for message passing."""
        return {
            'code_id': self.code_id,
            'name': self.name,
            'code_type': self.code_type,
            'source_code': self.source_code,
            'ast_blob': self.ast_blob,
            'bytecode_blob': self.bytecode_blob,
            'pickled_object': self.pickled_object,
            'required_imports': self.required_imports,
            'required_context_keys': self.required_context_keys,
            'code_hash': self.code_hash,
            'trusted_source': self.trusted_source,
            'sandbox_safe': self.sandbox_safe,
            'serialized_at': self.serialized_at.isoformat(),
            'python_version': self.python_version
        }
    
    @classmethod
    def from_message_dict(cls, data: Dict[str, Any]) -> 'SerializedSourceCode':
        """Reconstruct from message dictionary."""
        data = data.copy()
        data['serialized_at'] = datetime.fromisoformat(data['serialized_at'])
        return cls(**data)


# =============================================================================
# COMPLETE SERIALIZED CODE WITH AST TREE
# =============================================================================

@dataclass
class CompleteSerializedCode:
    """Complete serialized code with ALL formats including AST."""
    code_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    code_type: str = "function"
    
    # ALL serialization formats
    source_code: Optional[str] = None
    ast_tree_blob: Optional[str] = None      # ⭐ AST tree serialized
    bytecode_blob: Optional[str] = None      # ⭐ Raw bytecode for execution
    compiled_code_blob: Optional[str] = None # ⭐ Compiled code objects
    pickled_object: Optional[str] = None
    
    # Enhanced metadata with AST analysis
    ast_metadata: Dict[str, Any] = field(default_factory=dict)
    bytecode_metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Context chain requirements
    context_chain_requirements: List[str] = field(default_factory=list)
    required_imports: List[str] = field(default_factory=list)
    
    # Security and validation
    code_hash: str = field(default="")
    ast_hash: str = field(default="")
    bytecode_hash: str = field(default="")
    trusted_source: bool = False
    
    # Runtime modification tracking
    modification_history: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_metadata_dict(self) -> Dict[str, Any]:
        """Convert to metadata dictionary with AST tree included."""
        return {
            'code_id': self.code_id,
            'name': self.name,
            'code_type': self.code_type,
            'source_code': self.source_code,
            'ast_tree_blob': self.ast_tree_blob,        # ⭐ AST in metadata
            'bytecode_blob': self.bytecode_blob,        # ⭐ Bytecode in metadata
            'compiled_code_blob': self.compiled_code_blob,
            'pickled_object': self.pickled_object,
            'ast_metadata': self.ast_metadata,
            'bytecode_metadata': self.bytecode_metadata,
            'context_chain_requirements': self.context_chain_requirements,
            'required_imports': self.required_imports,
            'code_hash': self.code_hash,
            'ast_hash': self.ast_hash,
            'bytecode_hash': self.bytecode_hash,
            'trusted_source': self.trusted_source,
            'modification_history': self.modification_history
        }


# =============================================================================
# SOURCE CODE SERIALIZER
# =============================================================================

class SourceCodeSerializer:
    """Serializes Python code in multiple formats for robust transmission."""
    
    def __init__(self):
        self.serialization_methods = {
            CodeSerializationMethod.SOURCE_TEXT: self._serialize_source_text,
            CodeSerializationMethod.AST_TREE: self._serialize_ast,
            CodeSerializationMethod.BYTECODE_MARSHAL: self._serialize_bytecode,
            CodeSerializationMethod.PICKLE_FUNCTION: self._serialize_pickle,
            CodeSerializationMethod.DILL_ENHANCED: self._serialize_dill
        }
    
    def serialize_source_code(self, source_code: str, name: str = "", 
                            code_type: str = "function",
                            methods: List[CodeSerializationMethod] = None,
                            trusted_source: bool = False) -> SerializedSourceCode:
        """Serialize source code using multiple methods."""
        
        if methods is None:
            methods = [
                CodeSerializationMethod.SOURCE_TEXT,
                CodeSerializationMethod.AST_TREE,
                CodeSerializationMethod.PICKLE_FUNCTION
            ]
        
        serialized = SerializedSourceCode(
            name=name,
            code_type=code_type,
            source_code=source_code,
            trusted_source=trusted_source
        )
        
        # Calculate hash for integrity
        serialized.code_hash = serialized.calculate_hash()
        
        # Extract required imports and context
        serialized.required_imports = self._extract_imports(source_code)
        serialized.required_context_keys = self._extract_context_requirements(source_code)
        
        # Apply serialization methods
        for method in methods:
            try:
                serializer = self.serialization_methods.get(method)
                if serializer:
                    serializer(source_code, serialized)
            except Exception as e:
                print(f"Failed to serialize with method {method}: {e}")
        
        return serialized
    
    def serialize_function(self, func: Callable, trusted_source: bool = False) -> SerializedSourceCode:
        """Serialize an existing function object."""
        
        # Get source code
        try:
            source_code = inspect.getsource(func)
        except:
            source_code = f"# Could not extract source for {func.__name__}\npass"
        
        serialized = self.serialize_source_code(
            source_code,
            name=func.__name__,
            code_type="function",
            trusted_source=trusted_source
        )
        
        # Try to pickle the function directly
        try:
            pickled = pickle.dumps(func)
            serialized.pickled_object = base64.b64encode(pickled).decode('utf-8')
        except:
            try:
                pickled = dill.dumps(func)
                serialized.pickled_object = base64.b64encode(pickled).decode('utf-8')  
            except Exception as e:
                print(f"Could not pickle function {func.__name__}: {e}")
        
        return serialized
    
    def create_complete_serialized_code(self, source_code: str, name: str, 
                                       code_type: str, trusted_source: bool = False) -> CompleteSerializedCode:
        """Create complete serialized code with all formats."""
        serialized = CompleteSerializedCode(
            name=name,
            code_type=code_type,
            source_code=source_code,
            trusted_source=trusted_source
        )
        
        try:
            # Create AST tree blob ⭐
            ast_tree = ast.parse(source_code)
            serialized.ast_metadata = self._analyze_ast(ast_tree)
            ast_bytes = pickle.dumps(ast_tree)
            serialized.ast_tree_blob = base64.b64encode(ast_bytes).decode('utf-8')
            serialized.ast_hash = hashlib.sha256(ast_bytes).hexdigest()
            
            # Create bytecode blob ⭐
            compiled = compile(ast_tree, f'<{name}>', 'exec')
            bytecode_bytes = marshal.dumps(compiled)
            serialized.bytecode_blob = base64.b64encode(bytecode_bytes).decode('utf-8')
            serialized.bytecode_hash = hashlib.sha256(bytecode_bytes).hexdigest()
            
            # Create compiled code blob ⭐
            compiled_bytes = pickle.dumps(compiled)
            serialized.compiled_code_blob = base64.b64encode(compiled_bytes).decode('utf-8')
            
            # Calculate hashes
            serialized.code_hash = hashlib.sha256(source_code.encode()).hexdigest()
            
            # Extract requirements
            serialized.required_imports = self._extract_imports(source_code)
            serialized.context_chain_requirements = self._extract_context_requirements(source_code)
            
        except Exception as e:
            print(f"Failed to create complete serialized code: {e}")
        
        return serialized
    
    def _serialize_source_text(self, source_code: str, container: SerializedSourceCode):
        """Serialize as raw source text (already done in main method)."""
        pass  # Source code already stored
    
    def _serialize_ast(self, source_code: str, container: SerializedSourceCode):
        """Serialize as AST."""
        try:
            ast_tree = ast.parse(source_code)
            ast_bytes = pickle.dumps(ast_tree)
            container.ast_blob = base64.b64encode(ast_bytes).decode('utf-8')
        except Exception as e:
            print(f"AST serialization failed: {e}")
    
    def _serialize_bytecode(self, source_code: str, container: SerializedSourceCode):
        """Serialize as bytecode."""
        try:
            compiled = compile(source_code, '<serialized>', 'exec')
            bytecode_bytes = marshal.dumps(compiled)
            container.bytecode_blob = base64.b64encode(bytecode_bytes).decode('utf-8')
        except Exception as e:
            print(f"Bytecode serialization failed: {e}")
    
    def _serialize_pickle(self, source_code: str, container: SerializedSourceCode):
        """Serialize by compiling and pickling."""
        try:
            # Compile the source into a code object
            compiled = compile(source_code, '<serialized>', 'exec')
            pickled = pickle.dumps(compiled)
            container.pickled_object = base64.b64encode(pickled).decode('utf-8')
        except Exception as e:
            print(f"Pickle serialization failed: {e}")
    
    def _serialize_dill(self, source_code: str, container: SerializedSourceCode):
        """Serialize using dill for complex objects."""
        try:
            # Create a namespace and execute the code
            namespace = {}
            exec(source_code, namespace)
            
            # Find the main function/class
            for name, obj in namespace.items():
                if not name.startswith('__'):
                    pickled = dill.dumps(obj)
                    container.pickled_object = base64.b64encode(pickled).decode('utf-8')
                    break
        except Exception as e:
            print(f"Dill serialization failed: {e}")
    
    def _extract_imports(self, source_code: str) -> List[str]:
        """Extract import statements from source code."""
        imports = []
        try:
            tree = ast.parse(source_code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)
        except:
            pass
        return imports
    
    def _extract_context_requirements(self, source_code: str) -> List[str]:
        """Extract context variable requirements from source code."""
        # Simple heuristic: look for variables that might be context
        context_hints = []
        
        if 'context' in source_code:
            context_hints.append('context')
        if 'execution_context' in source_code:
            context_hints.append('execution_context')
        
        # Look for context.get() calls
        try:
            tree = ast.parse(source_code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if (isinstance(node.func, ast.Attribute) and 
                        node.func.attr == 'get' and
                        isinstance(node.func.value, ast.Name) and
                        'context' in node.func.value.id):
                        
                        # Extract the key being accessed
                        if node.args and isinstance(node.args[0], ast.Str):
                            context_hints.append(node.args[0].s)
        except:
            pass
        
        return list(set(context_hints))
    
    def _analyze_ast(self, ast_tree) -> Dict[str, Any]:
        """Analyze AST for metadata."""
        analysis = {
            'function_names': [],
            'class_names': [],
            'imported_modules': [],
            'variable_names': [],
            'complexity_score': 0
        }
        
        try:
            for node in ast.walk(ast_tree):
                if isinstance(node, ast.FunctionDef):
                    analysis['function_names'].append(node.name)
                elif isinstance(node, ast.ClassDef):
                    analysis['class_names'].append(node.name)
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        analysis['imported_modules'].append(alias.name)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    analysis['imported_modules'].append(node.module)
                elif isinstance(node, ast.Name):
                    analysis['variable_names'].append(node.id)
                
                # Simple complexity scoring
                if isinstance(node, (ast.If, ast.For, ast.While, ast.Try)):
                    analysis['complexity_score'] += 1
        
        except Exception as e:
            analysis['analysis_error'] = str(e)
        
        return analysis


# =============================================================================
# CONTEXT-AWARE DESERIALIZER
# =============================================================================

class ContextAwareDeserializer:
    """Deserializes code with context injection."""
    
    def __init__(self, trusted_mode: bool = False):
        self.trusted_mode = trusted_mode
        self.execution_namespace = {}
        self.context_providers: Dict[str, Callable] = {}
    
    def register_context_provider(self, key: str, provider: Callable):
        """Register a function that provides context values."""
        self.context_providers[key] = provider
    
    def deserialize_and_reconstruct(self, serialized: SerializedSourceCode,
                                  runtime_context: Dict[str, Any] = None) -> Any:
        """Deserialize code and reconstruct with context."""
        
        if not self.trusted_mode and not serialized.trusted_source:
            raise SecurityError("Untrusted code in non-trusted mode")
        
        # Prepare execution context
        execution_context = self._prepare_execution_context(serialized, runtime_context)
        
        # Try different deserialization methods in order of preference
        methods = [
            self._deserialize_from_pickle,
            self._deserialize_from_source,
            self._deserialize_from_ast,
            self._deserialize_from_bytecode,
        ]
        
        last_error = None
        for method in methods:
            try:
                result = method(serialized, execution_context)
                if result is not None:
                    return result
            except Exception as e:
                last_error = e
                continue
        
        raise DeserializationError(f"All deserialization methods failed. Last error: {last_error}")
    
    def _prepare_execution_context(self, serialized: SerializedSourceCode,
                                 runtime_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Prepare the execution context for code reconstruction."""
        
        context = {
            # Standard Python builtins
            '__builtins__': __builtins__,
            
            # Runtime context
            'runtime_context': runtime_context or {},
            
            # Execution metadata
            'code_id': serialized.code_id,
            'serialized_at': serialized.serialized_at,
        }
        
        # Add required imports
        for import_name in serialized.required_imports:
            try:
                context[import_name] = importlib.import_module(import_name)
            except ImportError as e:
                print(f"Could not import required module {import_name}: {e}")
        
        # Add context providers
        for key in serialized.required_context_keys:
            if key in self.context_providers:
                try:
                    context[key] = self.context_providers[key]()
                except Exception as e:
                    print(f"Context provider for {key} failed: {e}")
            elif runtime_context and key in runtime_context:
                context[key] = runtime_context[key]
        
        return context
    
    def _deserialize_from_pickle(self, serialized: SerializedSourceCode,
                               context: Dict[str, Any]) -> Any:
        """Deserialize from pickled object."""
        if not serialized.pickled_object:
            return None
        
        try:
            pickled_bytes = base64.b64decode(serialized.pickled_object.encode('utf-8'))
            obj = pickle.loads(pickled_bytes)
            
            # If it's a code object, execute it in context
            if isinstance(obj, types.CodeType):
                exec(obj, context)
                # Return the main function/class from context
                for name, value in context.items():
                    if not name.startswith('__') and callable(value):
                        return value
            else:
                return obj
                
        except:
            # Try dill
            try:
                pickled_bytes = base64.b64decode(serialized.pickled_object.encode('utf-8'))
                return dill.loads(pickled_bytes)
            except Exception as e:
                raise DeserializationError(f"Pickle deserialization failed: {e}")
        
        return None
    
    def _deserialize_from_source(self, serialized: SerializedSourceCode,
                               context: Dict[str, Any]) -> Any:
        """Deserialize from source code."""
        if not serialized.source_code:
            return None
        
        try:
            # Execute the source code in the prepared context
            exec(serialized.source_code, context)
            
            # Find and return the main function/class
            for name, value in context.items():
                if (not name.startswith('__') and 
                    not name in ['runtime_context', 'code_id', 'serialized_at'] and
                    callable(value)):
                    return value
            
            # If no callable found, return the entire namespace (minus builtins)
            result = {k: v for k, v in context.items() 
                     if not k.startswith('__') and k != 'runtime_context'}
            return result
            
        except Exception as e:
            raise DeserializationError(f"Source execution failed: {e}")
    
    def _deserialize_from_ast(self, serialized: SerializedSourceCode,
                            context: Dict[str, Any]) -> Any:
        """Deserialize from AST."""
        if not serialized.ast_blob:
            return None
        
        try:
            ast_bytes = base64.b64decode(serialized.ast_blob.encode('utf-8'))
            ast_tree = pickle.loads(ast_bytes)
            
            # Compile and execute the AST
            compiled = compile(ast_tree, '<deserialized>', 'exec')
            exec(compiled, context)
            
            # Return main function/class
            for name, value in context.items():
                if not name.startswith('__') and callable(value):
                    return value
                    
        except Exception as e:
            raise DeserializationError(f"AST deserialization failed: {e}")
        
        return None
    
    def _deserialize_from_bytecode(self, serialized: SerializedSourceCode,
                                 context: Dict[str, Any]) -> Any:
        """Deserialize from bytecode."""
        if not serialized.bytecode_blob:
            return None
        
        try:
            bytecode_bytes = base64.b64decode(serialized.bytecode_blob.encode('utf-8'))
            code_obj = marshal.loads(bytecode_bytes)
            
            # Execute the code object
            exec(code_obj, context)
            
            # Return main function/class
            for name, value in context.items():
                if not name.startswith('__') and callable(value):
                    return value
                    
        except Exception as e:
            raise DeserializationError(f"Bytecode deserialization failed: {e}")
        
        return None


# =============================================================================
# BYTECODE EXECUTION ENGINE
# =============================================================================

class BytecodeExecutionEngine:
    """Engine for loading and executing bytecode at runtime."""
    
    def __init__(self):
        self.loaded_code_objects: Dict[str, types.CodeType] = {}
        self.execution_namespace: Dict[str, Any] = {}
        self.bytecode_cache: Dict[str, bytes] = {}
        
        # Security tracking
        self.execution_log: List[Dict[str, Any]] = []
        self.trusted_code_hashes: set = set()
    
    def load_bytecode_from_blob(self, code_id: str, bytecode_blob: str, 
                               trusted: bool = False) -> bool:
        """Load bytecode from base64 blob."""
        try:
            # Decode bytecode
            bytecode_bytes = base64.b64decode(bytecode_blob.encode('utf-8'))
            self.bytecode_cache[code_id] = bytecode_bytes
            
            # Unmarshal to code object
            code_object = marshal.loads(bytecode_bytes)
            self.loaded_code_objects[code_id] = code_object
            
            # Track security
            if trusted:
                code_hash = hashlib.sha256(bytecode_bytes).hexdigest()
                self.trusted_code_hashes.add(code_hash)
            
            self._log_bytecode_load(code_id, trusted)
            return True
            
        except Exception as e:
            print(f"Failed to load bytecode {code_id}: {e}")
            return False
    
    def execute_bytecode(self, code_id: str, execution_context: Dict[str, Any] = None,
                        inject_into_namespace: bool = True) -> Any:
        """Execute loaded bytecode."""
        if code_id not in self.loaded_code_objects:
            raise ValueError(f"Bytecode {code_id} not loaded")
        
        code_object = self.loaded_code_objects[code_id]
        
        # Prepare execution namespace
        exec_namespace = self.execution_namespace.copy()
        if execution_context:
            exec_namespace.update(execution_context)
        
        # Execute the bytecode
        try:
            exec(code_object, exec_namespace)
            
            # Extract result (look for functions/classes defined)
            result = None
            for name, value in exec_namespace.items():
                if (not name.startswith('__') and 
                    name not in self.execution_namespace and
                    callable(value)):
                    result = value
                    
                    # Optionally inject into global namespace
                    if inject_into_namespace:
                        self.execution_namespace[name] = value
                    break
            
            self._log_bytecode_execution(code_id, True)
            return result
            
        except Exception as e:
            self._log_bytecode_execution(code_id, False, str(e))
            raise
    
    def compile_and_load_source(self, code_id: str, source_code: str,
                               filename: str = "<dynamic>") -> bool:
        """Compile source to bytecode and load it."""
        try:
            # Compile source to bytecode
            compiled = compile(source_code, filename, 'exec')
            bytecode_bytes = marshal.dumps(compiled)
            
            # Store and load
            self.bytecode_cache[code_id] = bytecode_bytes
            self.loaded_code_objects[code_id] = compiled
            
            return True
            
        except Exception as e:
            print(f"Failed to compile and load {code_id}: {e}")
            return False
    
    def _log_bytecode_load(self, code_id: str, trusted: bool):
        """Log bytecode loading."""
        self.execution_log.append({
            'action': 'load',
            'code_id': code_id,
            'trusted': trusted,
            'timestamp': datetime.now().isoformat()
        })
    
    def _log_bytecode_execution(self, code_id: str, success: bool, error: str = None):
        """Log bytecode execution."""
        self.execution_log.append({
            'action': 'execute',
            'code_id': code_id,
            'success': success,
            'error': error,
            'timestamp': datetime.now().isoformat()
        })
    
    def get_loaded_code_info(self) -> Dict[str, Any]:
        """Get information about loaded code."""
        return {
            'loaded_objects': list(self.loaded_code_objects.keys()),
            'cached_bytecode': list(self.bytecode_cache.keys()),
            'namespace_size': len(self.execution_namespace),
            'execution_log_size': len(self.execution_log),
            'trusted_hashes': len(self.trusted_code_hashes)
        }


# =============================================================================
# RUNTIME SOURCE CODE EDITOR & EXECUTOR
# =============================================================================

class RuntimeSourceEditor:
    """Edit and execute Python source code during runtime."""
    
    def __init__(self):
        self.source_versions: Dict[str, List[str]] = {}  # code_id -> versions
        self.compiled_versions: Dict[str, List[types.CodeType]] = {}
        self.execution_engine = BytecodeExecutionEngine()
        
        # Live editing state
        self.active_editors: Dict[str, Dict[str, Any]] = {}
        self.modification_callbacks: List[Callable] = []
    
    def create_editable_source(self, code_id: str, initial_source: str) -> str:
        """Create editable source code."""
        if code_id not in self.source_versions:
            self.source_versions[code_id] = []
            self.compiled_versions[code_id] = []
        
        # Add initial version
        version_id = f"{code_id}_v{len(self.source_versions[code_id])}"
        self.source_versions[code_id].append(initial_source)
        
        # Compile initial version
        self.execution_engine.compile_and_load_source(version_id, initial_source)
        
        return version_id
    
    def edit_source_runtime(self, code_id: str, modifications: Dict[str, Any]) -> str:
        """Edit source code at runtime and create new version."""
        if code_id not in self.source_versions or not self.source_versions[code_id]:
            raise ValueError(f"No source code found for {code_id}")
        
        # Get current source
        current_source = self.source_versions[code_id][-1]
        
        # Apply modifications
        modified_source = self._apply_source_modifications(current_source, modifications)
        
        # Create new version
        new_version_id = f"{code_id}_v{len(self.source_versions[code_id])}"
        self.source_versions[code_id].append(modified_source)
        
        # Compile new version
        success = self.execution_engine.compile_and_load_source(new_version_id, modified_source)
        
        if success:
            # Notify modification callbacks
            for callback in self.modification_callbacks:
                try:
                    callback(code_id, new_version_id, modified_source)
                except:
                    pass
        
        return new_version_id if success else None
    
    def _apply_source_modifications(self, source: str, modifications: Dict[str, Any]) -> str:
        """Apply modifications to source code."""
        modified = source
        
        # Line replacements
        if 'replace_lines' in modifications:
            lines = modified.split('\n')
            for line_num, new_content in modifications['replace_lines'].items():
                if 0 <= line_num < len(lines):
                    lines[line_num] = new_content
            modified = '\n'.join(lines)
        
        # Text replacements
        if 'replace_text' in modifications:
            for old_text, new_text in modifications['replace_text'].items():
                modified = modified.replace(old_text, new_text)
        
        # Function insertions
        if 'insert_functions' in modifications:
            for func_source in modifications['insert_functions']:
                modified += '\n\n' + func_source
        
        # Decorator additions
        if 'add_decorators' in modifications:
            for func_name, decorators in modifications['add_decorators'].items():
                # Find function definition and add decorators
                lines = modified.split('\n')
                for i, line in enumerate(lines):
                    if f'def {func_name}(' in line:
                        # Insert decorators before function
                        for decorator in reversed(decorators):
                            lines.insert(i, f'@{decorator}')
                        break
                modified = '\n'.join(lines)
        
        return modified
    
    def get_source_diff(self, code_id: str, version_a: int = -2, version_b: int = -1) -> str:
        """Get diff between two versions of source code."""
        if code_id not in self.source_versions:
            return "No source versions found"
        
        versions = self.source_versions[code_id]
        if len(versions) < 2:
            return "Not enough versions for diff"
        
        source_a = versions[version_a]
        source_b = versions[version_b]
        
        # Simple line-by-line diff
        lines_a = source_a.split('\n')
        lines_b = source_b.split('\n')
        
        diff_lines = []
        max_lines = max(len(lines_a), len(lines_b))
        
        for i in range(max_lines):
            line_a = lines_a[i] if i < len(lines_a) else ""
            line_b = lines_b[i] if i < len(lines_b) else ""
            
            if line_a != line_b:
                diff_lines.append(f"- {line_a}")
                diff_lines.append(f"+ {line_b}")
            else:
                diff_lines.append(f"  {line_a}")
        
        return '\n'.join(diff_lines)


# =============================================================================
# RUNTIME CODE INJECTION CACHE
# =============================================================================

class RuntimeCodeCache:
    """Cache system for runtime code injection and decorator swapping."""
    
    def __init__(self, trusted_mode: bool = False):
        self.trusted_mode = trusted_mode
        
        # Cache storage
        self.code_cache: Dict[str, Any] = {}  # code_id -> executable object
        self.decorator_generators: Dict[str, Callable] = {}  # name -> generator function
        self.active_decorators: Dict[str, List[str]] = {}  # callback_id -> decorator_ids
        
        # Serialization/Deserialization
        self.serializer = SourceCodeSerializer()
        self.deserializer = ContextAwareDeserializer(trusted_mode)
        
        # Thread safety
        self._lock = threading.RLock()
    
    def inject_source_code(self, source_code: str, name: str, code_type: str = "function",
                          context: Dict[str, Any] = None) -> str:
        """Inject raw source code into cache."""
        
        with self._lock:
            # Serialize the source code
            serialized = self.serializer.serialize_source_code(
                source_code, name, code_type, trusted_source=self.trusted_mode
            )
            
            # Deserialize and cache
            executable = self.deserializer.deserialize_and_reconstruct(serialized, context)
            self.code_cache[serialized.code_id] = executable
            
            # If it's a decorator generator, add to special registry
            if code_type == "decorator_generator":
                self.decorator_generators[name] = executable
            
            return serialized.code_id
    
    def inject_serialized_code(self, serialized: SerializedSourceCode,
                             context: Dict[str, Any] = None) -> str:
        """Inject already serialized code into cache."""
        
        with self._lock:
            # Deserialize and cache
            executable = self.deserializer.deserialize_and_reconstruct(serialized, context)
            self.code_cache[serialized.code_id] = executable
            
            # If it's a decorator generator, add to special registry
            if serialized.code_type == "decorator_generator":
                self.decorator_generators[serialized.name] = executable
            
            return serialized.code_id
    
    def get_decorator_generator(self, name: str) -> Optional[Callable]:
        """Get a decorator generator by name."""
        with self._lock:
            return self.decorator_generators.get(name)
    
    def create_decorator_from_generator(self, generator_name: str, 
                                      config: Dict[str, Any] = None,
                                      context: Dict[str, Any] = None) -> Optional[Callable]:
        """Create a decorator using a cached generator."""
        
        generator = self.get_decorator_generator(generator_name)
        if not generator:
            return None
        
        try:
            # Call the generator with config and context
            return generator(config or {}, context or {})
        except Exception as e:
            print(f"Failed to create decorator from generator {generator_name}: {e}")
            return None
    
    def apply_decorator_to_callback(self, callback_id: str, callback: Callable,
                                  decorator_generator_name: str,
                                  decorator_config: Dict[str, Any] = None,
                                  context: Dict[str, Any] = None) -> Tuple[str, Callable]:
        """Apply a decorator to a callback and track it."""
        
        with self._lock:
            # Create decorator
            decorator = self.create_decorator_from_generator(
                decorator_generator_name, decorator_config, context
            )
            
            if not decorator:
                raise ValueError(f"No decorator generator found: {decorator_generator_name}")
            
            # Apply decorator
            decorated_callback = decorator(callback)
            
            # Generate decorated callback ID
            decorated_id = f"{callback_id}_decorated_{uuid.uuid4()}"
            
            # Track active decorators
            if callback_id not in self.active_decorators:
                self.active_decorators[callback_id] = []
            self.active_decorators[callback_id].append(decorated_id)
            
            return decorated_id, decorated_callback
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                'total_cached_objects': len(self.code_cache),
                'decorator_generators': len(self.decorator_generators),
                'active_decorators': len(self.active_decorators),
                'generator_names': list(self.decorator_generators.keys()),
                'trusted_mode': self.trusted_mode
            }


# =============================================================================
# INTEGRATED LIVE CODE SYSTEM WITH EVERYTHING
# =============================================================================

class CompleteLiveCodeSystem:
    """The complete live code system with all components."""
    
    def __init__(self, trusted_mode: bool = False):
        self.trusted_mode = trusted_mode
        
        # Core components
        self.bytecode_engine = BytecodeExecutionEngine()
        self.source_editor = RuntimeSourceEditor()
        self.runtime_cache = RuntimeCodeCache(trusted_mode)
        
        # Serialization
        self.serializer = SourceCodeSerializer()
        
        # Runtime state
        self.runtime_modifications: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, Any] = defaultdict(list)
    
    def inject_complete_serialized_code(self, serialized_code: CompleteSerializedCode,
                                       load_bytecode: bool = True) -> bool:
        """Inject complete serialized code with all formats."""
        code_id = serialized_code.code_id
        
        # Load bytecode if available
        if load_bytecode and serialized_code.bytecode_blob:
            success = self.bytecode_engine.load_bytecode_from_blob(
                code_id, 
                serialized_code.bytecode_blob,
                serialized_code.trusted_source
            )
            if not success:
                return False
        
        # Create editable source if available
        if serialized_code.source_code:
            self.source_editor.create_editable_source(code_id, serialized_code.source_code)
        
        return True
    
    def create_complete_serialized_code(self, source_code: str, name: str, 
                                       code_type: str = "function") -> CompleteSerializedCode:
        """Create complete serialized code with all formats."""
        return self.serializer.create_complete_serialized_code(
            source_code, name, code_type, self.trusted_mode
        )
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        return {
            'bytecode_engine': self.bytecode_engine.get_loaded_code_info(),
            'runtime_cache': self.runtime_cache.get_cache_stats(),
            'runtime_modifications': len(self.runtime_modifications),
            'trusted_mode': self.trusted_mode
        }


# =============================================================================
# INTEGRATED CALLBACK SYSTEM WITH LIVE CODE INJECTION
# =============================================================================

class LiveCodeCallbackSystem:
    """Callback system with live code injection and decorator swapping."""
    
    def __init__(self, trusted_mode: bool = False):
        self.callbacks: Dict[str, Callable] = {}
        self.code_cache = RuntimeCodeCache(trusted_mode)
        
        # Track decorated callbacks
        self.decorated_callbacks: Dict[str, Callable] = {}
        self.callback_metadata: Dict[str, Dict[str, Any]] = {}
    
    def register_callback(self, callback_id: str, callback: Callable):
        """Register a callback."""
        self.callbacks[callback_id] = callback
        self.callback_metadata[callback_id] = {
            'registered_at': datetime.now(),
            'original_callback': callback,
            'applied_decorators': []
        }
    
    def execute_callback(self, callback_id: str, *args, **kwargs) -> Any:
        """Execute callback (original or decorated version)."""
        
        # Check for decorated versions first
        metadata = self.callback_metadata.get(callback_id, {})
        applied_decorators = metadata.get('applied_decorators', [])
        
        if applied_decorators:
            # Use the most recently applied decorator
            latest_decorator = applied_decorators[-1]
            decorated_id = latest_decorator['decorated_id']
            
            if decorated_id in self.decorated_callbacks:
                return self.decorated_callbacks[decorated_id](*args, **kwargs)
        
        # Fallback to original callback
        if callback_id in self.callbacks:
            return self.callbacks[callback_id](*args, **kwargs)
        
        raise ValueError(f"No callback found: {callback_id}")
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        return {
            'callbacks_registered': len(self.callbacks),
            'decorated_callbacks': len(self.decorated_callbacks),
            'code_cache_stats': self.code_cache.get_cache_stats(),
            'callback_metadata': {
                callback_id: {
                    'registered_at': metadata['registered_at'].isoformat(),
                    'decorators_applied': len(metadata['applied_decorators'])
                }
                for callback_id, metadata in self.callback_metadata.items()
            }
        }


# =============================================================================
# CUSTOM EXCEPTIONS
# =============================================================================

class SecurityError(Exception):
    """Raised when attempting to execute untrusted code in trusted mode."""
    pass

class DeserializationError(Exception):
    """Raised when code deserialization fails."""
    pass
                