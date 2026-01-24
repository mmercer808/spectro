#!/usr/bin/env python3
"""
Advanced Signal System Library

A production-ready, high-performance signal system with:
- Concurrent observer processing (no head-of-line blocking)
- Circuit breaker pattern for resilience
- Priority-based dispatching
- Worker pools per observer
- Comprehensive error handling and monitoring
- Memory-efficient design
- Thread-safe operations
- Deadlock prevention
"""

import asyncio
import threading
import time
import uuid
import weakref
import json
import logging
from abc import ABC, abstractmethod
from typing import (
    Dict, Any, List, Optional, Set, Callable, AsyncGenerator, 
    Union, TypeVar, Generic, Protocol, Awaitable, Tuple
)
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum, IntEnum
from concurrent.futures import ThreadPoolExecutor
import inspect
import sys

# Configure logging
logger = logging.getLogger(__name__)

# Type definitions
T = TypeVar('T')
CallbackType = Union[Callable, Callable[..., Awaitable]]

# Version info
__version__ = "1.0.0"

# =============================================================================
# CORE ENUMS AND DATA STRUCTURES
# =============================================================================

class SignalPriority(IntEnum):
    """Priority levels for signal processing."""
    CRITICAL = 0    # System-critical signals (errors, shutdowns)
    HIGH = 1        # High-priority user requests  
    NORMAL = 2      # Standard operations
    LOW = 3         # Background tasks, analytics
    BULK = 4        # Batch processing, cleanup


class SignalType(Enum):
    """Standard signal types."""
    # Streaming signals
    STREAM_START = "stream_start"
    STREAM_CHUNK = "stream_chunk"
    STREAM_END = "stream_end" 
    STREAM_ERROR = "stream_error"
    
    # Query signals
    QUERY_START = "query_start"
    QUERY_PROGRESS = "query_progress"
    QUERY_COMPLETE = "query_complete"
    QUERY_FAILED = "query_failed"
    
    # Context signals
    CONTEXT_UPDATE = "context_update"
    CONTEXT_CREATED = "context_created"  
    CONTEXT_DESTROYED = "context_destroyed"
    CONTEXT_SERIALIZED = "context_serialized"
    
    # System signals
    SYSTEM_STARTUP = "system_startup"
    SYSTEM_SHUTDOWN = "system_shutdown"
    SYSTEM_ERROR = "system_error"
    
    # Observer lifecycle
    OBSERVER_REGISTERED = "observer_registered"
    OBSERVER_UNREGISTERED = "observer_unregistered"
    
    # Custom signals
    CUSTOM = "custom"


class ObserverPriority(Enum):
    """Observer execution priority."""
    CRITICAL = "critical"    # Must complete before continuing
    NORMAL = "normal"        # Concurrent with timeout
    BACKGROUND = "background" # Fire-and-forget


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject calls
    HALF_OPEN = "half_open" # Testing if recovered


@dataclass
class SignalPayload:
    """Core signal payload with rich metadata."""
    signal_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    signal_type: Union[SignalType, str] = SignalType.CUSTOM
    source_id: str = ""
    target_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    priority: SignalPriority = SignalPriority.NORMAL
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict) 
    correlation_id: Optional[str] = None
    parent_signal_id: Optional[str] = None
    timeout: float = 30.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        result['signal_type'] = (
            self.signal_type.value if isinstance(self.signal_type, SignalType) 
            else str(self.signal_type)
        )
        result['priority'] = self.priority.value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SignalPayload':
        """Create from dictionary."""
        data = data.copy()
        data['timestamp'] = datetime.fromisoformat(data.get('timestamp', datetime.now().isoformat()))
        
        # Handle signal_type
        signal_type_val = data.get('signal_type', SignalType.CUSTOM.value)
        try:
            data['signal_type'] = SignalType(signal_type_val)
        except ValueError:
            data['signal_type'] = signal_type_val  # Keep as string for custom types
        
        data['priority'] = SignalPriority(data.get('priority', SignalPriority.NORMAL.value))
        return cls(**data)


@dataclass
class ObserverStats:
    """Observer performance statistics."""
    observer_id: str
    registration_time: datetime
    last_activity: Optional[datetime] = None
    signals_processed: int = 0
    signals_failed: int = 0
    total_processing_time: float = 0.0
    avg_processing_time: float = 0.0
    circuit_breaker_trips: int = 0
    current_circuit_state: CircuitState = CircuitState.CLOSED
    
    def update_processing_time(self, processing_time: float):
        """Update processing time statistics."""
        self.total_processing_time += processing_time
        self.signals_processed += 1
        self.avg_processing_time = self.total_processing_time / self.signals_processed
        self.last_activity = datetime.now()


# =============================================================================
# OBSERVER SYSTEM
# =============================================================================

class Observer(ABC):
    """Abstract base class for signal observers."""
    
    def __init__(self, 
                 observer_id: str = None,
                 priority: ObserverPriority = ObserverPriority.NORMAL,
                 timeout: float = 10.0):
        self.observer_id = observer_id or f"observer_{uuid.uuid4()}"
        self.priority = priority
        self.timeout = timeout
        
        # Filtering
        self.signal_filters: Set[Union[SignalType, str]] = set()
        self.priority_filters: Set[SignalPriority] = set()
        self.source_filters: Set[str] = set()
        
        # State
        self.active = True
        self.stats = ObserverStats(
            observer_id=self.observer_id,
            registration_time=datetime.now()
        )
    
    @abstractmethod
    async def on_signal(self, signal: SignalPayload) -> Any:
        """Handle incoming signal. Must be implemented by subclasses."""
        pass
    
    def can_handle_signal(self, signal: SignalPayload) -> bool:
        """Check if observer can handle this signal."""
        if not self.active:
            return False
        
        # Check signal type filters
        if self.signal_filters:
            signal_type = (
                signal.signal_type.value if isinstance(signal.signal_type, SignalType)
                else signal.signal_type
            )
            if not any(
                (isinstance(f, SignalType) and f == signal.signal_type) or
                (isinstance(f, str) and f == signal_type)
                for f in self.signal_filters
            ):
                return False
        
        # Check priority filters
        if self.priority_filters and signal.priority not in self.priority_filters:
            return False
        
        # Check source filters  
        if self.source_filters and signal.source_id not in self.source_filters:
            return False
        
        return True
    
    def add_signal_filter(self, signal_type: Union[SignalType, str]):
        """Add signal type filter."""
        self.signal_filters.add(signal_type)
    
    def add_priority_filter(self, priority: SignalPriority):
        """Add priority filter."""
        self.priority_filters.add(priority)
    
    def add_source_filter(self, source_id: str):
        """Add source filter."""
        self.source_filters.add(source_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get observer statistics."""
        return {
            'observer_id': self.observer_id,
            'priority': self.priority.value,
            'active': self.active,
            'registration_time': self.stats.registration_time.isoformat(),
            'last_activity': (
                self.stats.last_activity.isoformat() 
                if self.stats.last_activity else None
            ),
            'signals_processed': self.stats.signals_processed,
            'signals_failed': self.stats.signals_failed,
            'avg_processing_time': self.stats.avg_processing_time,
            'circuit_breaker_trips': self.stats.circuit_breaker_trips,
            'current_circuit_state': self.stats.current_circuit_state.value,
            'signal_filters': [
                f.value if isinstance(f, SignalType) else f 
                for f in self.signal_filters
            ],
            'priority_filters': [p.value for p in self.priority_filters],
            'source_filters': list(self.source_filters)
        }


class CallbackObserver(Observer):
    """Observer that wraps callback functions."""
    
    def __init__(self, 
                 callback: CallbackType, 
                 observer_id: str = None,
                 priority: ObserverPriority = ObserverPriority.NORMAL):
        super().__init__(observer_id, priority)
        self.callback = callback
        self.is_async = inspect.iscoroutinefunction(callback)
    
    async def on_signal(self, signal: SignalPayload) -> Any:
        """Execute callback with signal."""
        if self.is_async:
            return await self.callback(signal)
        else:
            # Run sync callback in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.callback, signal)


# =============================================================================
# CIRCUIT BREAKER SYSTEM
# =============================================================================

class CircuitBreaker:
    """Circuit breaker to protect against misbehaving observers."""
    
    def __init__(self, 
                 failure_threshold: int = 5,
                 timeout_threshold: float = 30.0,
                 recovery_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.timeout_threshold = timeout_threshold
        self.recovery_timeout = recovery_timeout
        self.observer_states: Dict[str, Dict[str, Any]] = {}
        self.observer_locks: Dict[str, asyncio.Lock] = {}
        self._global_lock = asyncio.Lock()
    
    async def _get_observer_lock(self, observer_id: str) -> asyncio.Lock:
        """Get or create lock for specific observer."""
        if observer_id not in self.observer_locks:
            async with self._global_lock:
                if observer_id not in self.observer_locks:
                    self.observer_locks[observer_id] = asyncio.Lock()
        return self.observer_locks[observer_id]
    
    async def call_observer(self, observer: Observer, signal: SignalPayload) -> Tuple[bool, Any]:
        """Call observer through circuit breaker with per-observer locking."""
        observer_id = observer.observer_id
        
        # Get per-observer lock to prevent race conditions
        observer_lock = await self._get_observer_lock(observer_id)
        
        # Initialize state if needed (under per-observer lock)
        async with observer_lock:
            if observer_id not in self.observer_states:
                self.observer_states[observer_id] = {
                    'failures': 0,
                    'last_failure': None,
                    'state': CircuitState.CLOSED,
                    'last_success': datetime.now()
                }
            
            state = self.observer_states[observer_id]
            
            # Check circuit state
            if state['state'] == CircuitState.OPEN:
                if (datetime.now() - state['last_failure']).seconds < self.recovery_timeout:
                    logger.debug(f"Circuit breaker OPEN for {observer_id}")
                    return False, None
                else:
                    state['state'] = CircuitState.HALF_OPEN
                    logger.info(f"Circuit breaker HALF_OPEN for {observer_id}")
        
        # Attempt to call observer (OUTSIDE the lock)
        start_time = time.time()
        try:
            result = await asyncio.wait_for(
                observer.on_signal(signal),
                timeout=min(observer.timeout, self.timeout_threshold)
            )
            
            processing_time = time.time() - start_time
            
            # Success - update stats and reset circuit
            observer.stats.update_processing_time(processing_time)
            
            async with observer_lock:
                state['failures'] = 0
                state['state'] = CircuitState.CLOSED
                state['last_success'] = datetime.now()
            
            return True, result
            
        except Exception as e:
            processing_time = time.time() - start_time
            observer.stats.signals_failed += 1
            
            # Handle circuit breaker logic (under per-observer lock)
            async with observer_lock:
                state['failures'] += 1
                state['last_failure'] = datetime.now()
                
                if state['failures'] >= self.failure_threshold:
                    state['state'] = CircuitState.OPEN
                    observer.stats.circuit_breaker_trips += 1
                    observer.stats.current_circuit_state = CircuitState.OPEN
                    logger.warning(
                        f"Circuit breaker OPENED for {observer_id} "
                        f"after {state['failures']} failures: {e}"
                    )
            
            logger.error(f"Observer {observer_id} failed: {e}")
            return False, None


# =============================================================================
# OBSERVER WORKER POOL
# =============================================================================

class ObserverWorkerPool:
    """Manages concurrent execution of observers with per-observer limits."""
    
    def __init__(self, max_concurrent_per_observer: int = 10):
        self.max_concurrent = max_concurrent_per_observer
        self.observer_semaphores: Dict[str, asyncio.Semaphore] = {}
        self.observer_locks: Dict[str, asyncio.Lock] = {}
        self._pool_lock = asyncio.Lock()
    
    async def get_observer_semaphore(self, observer_id: str) -> asyncio.Semaphore:
        """Get or create semaphore for observer."""
        if observer_id not in self.observer_semaphores:
            async with self._pool_lock:
                if observer_id not in self.observer_semaphores:
                    self.observer_semaphores[observer_id] = asyncio.Semaphore(
                        self.max_concurrent
                    )
        return self.observer_semaphores[observer_id]
    
    async def notify_observer(self, 
                            observer: Observer, 
                            signal: SignalPayload,
                            circuit_breaker: CircuitBreaker) -> Tuple[bool, Any]:
        """Notify observer through worker pool with concurrency control."""
        semaphore = await self.get_observer_semaphore(observer.observer_id)
        
        async with semaphore:
            return await circuit_breaker.call_observer(observer, signal)


# =============================================================================
# PRIORITY DISPATCHER
# =============================================================================

class PriorityDispatcher:
    """Dispatches signals to observers based on priority levels."""
    
    def __init__(self, worker_pool: ObserverWorkerPool, circuit_breaker: CircuitBreaker):
        self.worker_pool = worker_pool
        self.circuit_breaker = circuit_breaker
        
        # Organize observers by priority
        self.critical_observers: Set[Observer] = set()
        self.normal_observers: Set[Observer] = set()
        self.background_observers: Set[Observer] = set()
        
        self._observers_lock = asyncio.Lock()
    
    async def register_observer(self, observer: Observer):
        """Register observer in appropriate priority queue."""
        async with self._observers_lock:
            # Remove from all sets first (in case priority changed)
            self.critical_observers.discard(observer)
            self.normal_observers.discard(observer)
            self.background_observers.discard(observer)
            
            # Add to appropriate set
            if observer.priority == ObserverPriority.CRITICAL:
                self.critical_observers.add(observer)
            elif observer.priority == ObserverPriority.BACKGROUND:
                self.background_observers.add(observer)
            else:
                self.normal_observers.add(observer)
    
    async def unregister_observer(self, observer: Observer):
        """Remove observer from all priority queues."""
        async with self._observers_lock:
            self.critical_observers.discard(observer)
            self.normal_observers.discard(observer)
            self.background_observers.discard(observer)
    
    async def dispatch_signal(self, signal: SignalPayload) -> Dict[str, int]:
        """Dispatch signal with priority-based execution."""
        results = {
            'critical_notified': 0,
            'normal_notified': 0,
            'background_notified': 0,
            'total_notified': 0
        }
        
        # Get snapshot of observers to avoid lock during notification
        async with self._observers_lock:
            critical_observers = [
                obs for obs in self.critical_observers 
                if obs.can_handle_signal(signal)
            ]
            normal_observers = [
                obs for obs in self.normal_observers
                if obs.can_handle_signal(signal)
            ]
            background_observers = [
                obs for obs in self.background_observers
                if obs.can_handle_signal(signal)
            ]
        
        # Critical observers: Wait for completion
        if critical_observers:
            tasks = [
                self.worker_pool.notify_observer(obs, signal, self.circuit_breaker)
                for obs in critical_observers
            ]
            
            try:
                critical_results = await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=signal.timeout
                )
                results['critical_notified'] = sum(
                    1 for result in critical_results 
                    if isinstance(result, tuple) and result[0]
                )
            except asyncio.TimeoutError:
                logger.warning(f"Critical observer timeout for signal {signal.signal_id}")
        
        # Normal observers: Concurrent processing
        if normal_observers:
            tasks = [
                self.worker_pool.notify_observer(obs, signal, self.circuit_breaker)
                for obs in normal_observers
            ]
            
            # Process without blocking main thread
            async def process_normal_observers():
                try:
                    normal_results = await asyncio.gather(*tasks, return_exceptions=True)
                    results['normal_notified'] = sum(
                        1 for result in normal_results
                        if isinstance(result, tuple) and result[0]
                    )
                except Exception as e:
                    logger.error(f"Error processing normal observers: {e}")
            
            asyncio.create_task(process_normal_observers())
            results['normal_notified'] = len(normal_observers)  # Estimate
        
        # Background observers: Fire-and-forget
        for observer in background_observers:
            asyncio.create_task(
                self.worker_pool.notify_observer(observer, signal, self.circuit_breaker)
            )
        
        results['background_notified'] = len(background_observers)
        results['total_notified'] = (
            results['critical_notified'] + 
            results['normal_notified'] + 
            results['background_notified']
        )
        
        return results


# =============================================================================  
# MAIN SIGNAL LINE
# =============================================================================

class SignalLine:
    """High-performance, concurrent signal line with resilience features."""
    
    def __init__(self, 
                 line_id: str = None,
                 max_observers: int = 1000,
                 max_concurrent_per_observer: int = 10,
                 circuit_breaker_config: Dict[str, Any] = None):
        
        self.line_id = line_id or f"signal_line_{uuid.uuid4()}"
        self.max_observers = max_observers
        
        # Core components
        self.circuit_breaker = CircuitBreaker(**(circuit_breaker_config or {}))
        self.worker_pool = ObserverWorkerPool(max_concurrent_per_observer)
        self.priority_dispatcher = PriorityDispatcher(self.worker_pool, self.circuit_breaker)
        
        # Observer management
        self._observers: Dict[str, Observer] = {}
        self._observers_lock = asyncio.Lock()
        
        # System state
        self._running = False
        self._shutdown_event = asyncio.Event()
        
        # Statistics
        self.signals_emitted = 0
        self.signals_processed = 0
        self.creation_time = datetime.now()
        self.last_signal_time: Optional[datetime] = None
        
        logger.info(f"SignalLine {self.line_id} created")
    
    async def start(self):
        """Start the signal line."""
        if self._running:
            return
        
        self._running = True
        self._shutdown_event.clear()
        
        # Emit startup signal
        startup_signal = SignalPayload(
            signal_type=SignalType.SYSTEM_STARTUP,
            source_id=self.line_id,
            priority=SignalPriority.HIGH
        )
        
        asyncio.create_task(self._emit_internal(startup_signal))
        logger.info(f"SignalLine {self.line_id} started")
    
    async def stop(self, timeout: float = 30.0):
        """Stop the signal line gracefully."""
        if not self._running:
            return
        
        logger.info(f"SignalLine {self.line_id} stopping...")
        
        # Emit shutdown signal
        shutdown_signal = SignalPayload(
            signal_type=SignalType.SYSTEM_SHUTDOWN,
            source_id=self.line_id,
            priority=SignalPriority.CRITICAL
        )
        
        await self._emit_internal(shutdown_signal)
        
        # Mark as stopped
        self._running = False
        self._shutdown_event.set()
        
        logger.info(f"SignalLine {self.line_id} stopped")
    
    async def register_observer(self, observer: Observer) -> bool:
        """Register an observer."""
        async with self._observers_lock:
            if len(self._observers) >= self.max_observers:
                logger.warning(f"Max observers ({self.max_observers}) reached")
                return False
            
            if observer.observer_id in self._observers:
                logger.warning(f"Observer {observer.observer_id} already registered")
                return False
            
            self._observers[observer.observer_id] = observer
        
        # Register with priority dispatcher
        await self.priority_dispatcher.register_observer(observer)
        
        # Emit registration signal
        reg_signal = SignalPayload(
            signal_type=SignalType.OBSERVER_REGISTERED,
            source_id=self.line_id,
            data={'observer_id': observer.observer_id},
            priority=SignalPriority.LOW
        )
        
        asyncio.create_task(self._emit_internal(reg_signal))
        
        logger.info(f"Observer {observer.observer_id} registered")
        return True
    
    async def unregister_observer(self, observer_id: str) -> bool:
        """Unregister an observer."""
        observer = None
        
        async with self._observers_lock:
            if observer_id not in self._observers:
                return False
            observer = self._observers.pop(observer_id)
        
        # Unregister from priority dispatcher
        await self.priority_dispatcher.unregister_observer(observer)
        
        # Emit unregistration signal
        unreg_signal = SignalPayload(
            signal_type=SignalType.OBSERVER_UNREGISTERED,
            source_id=self.line_id,
            data={'observer_id': observer_id},
            priority=SignalPriority.LOW
        )
        
        asyncio.create_task(self._emit_internal(unreg_signal))
        
        logger.info(f"Observer {observer_id} unregistered")
        return True
    
    async def register_callback(self, 
                               callback: CallbackType,
                               observer_id: str = None,
                               priority: ObserverPriority = ObserverPriority.NORMAL,
                               signal_filters: List[Union[SignalType, str]] = None,
                               priority_filters: List[SignalPriority] = None,
                               source_filters: List[str] = None) -> str:
        """Register a callback as an observer."""
        
        observer = CallbackObserver(callback, observer_id, priority)
        
        # Apply filters
        if signal_filters:
            for signal_type in signal_filters:
                observer.add_signal_filter(signal_type)
        
        if priority_filters:
            for priority_filter in priority_filters:
                observer.add_priority_filter(priority_filter)
        
        if source_filters:
            for source in source_filters:
                observer.add_source_filter(source)
        
        await self.register_observer(observer)
        return observer.observer_id
    
    async def emit(self, signal: SignalPayload) -> Dict[str, int]:
        """Emit a signal to observers."""
        if not self._running:
            logger.warning(f"Cannot emit signal - SignalLine {self.line_id} not running")
            return {'total_notified': 0}
        
        return await self._emit_internal(signal)
    
    async def _emit_internal(self, signal: SignalPayload) -> Dict[str, int]:
        """Internal emit method."""
        self.signals_emitted += 1
        self.last_signal_time = datetime.now()
        
        # Use priority dispatcher for high-performance concurrent processing
        results = await self.priority_dispatcher.dispatch_signal(signal)
        
        self.signals_processed += results['total_notified']
        
        return results
    
    async def get_observer_stats(self) -> Dict[str, Any]:
        """Get statistics for all observers."""
        async with self._observers_lock:
            observers_snapshot = dict(self._observers)
        
        return {
            observer_id: observer.get_stats()
            for observer_id, observer in observers_snapshot.items()
        }
    
    def get_line_stats(self) -> Dict[str, Any]:
        """Get signal line statistics."""
        return {
            'line_id': self.line_id,
            'running': self._running,
            'creation_time': self.creation_time.isoformat(),
            'last_signal_time': (
                self.last_signal_time.isoformat() 
                if self.last_signal_time else None
            ),
            'observer_count': len(self._observers),
            'max_observers': self.max_observers,
            'signals_emitted': self.signals_emitted,
            'signals_processed': self.signals_processed
        }


# =============================================================================
# CONVENIENCE FUNCTIONS AND UTILITIES
# =============================================================================

async def create_signal_line(line_id: str = None, **kwargs) -> SignalLine:
    """Create and start a signal line."""
    line = SignalLine(line_id, **kwargs)
    await line.start()
    return line


def signal_handler(signal_types: List[Union[SignalType, str]] = None,
                  priority: ObserverPriority = ObserverPriority.NORMAL):
    """Decorator to create observer from function."""
    def decorator(func: Callable):
        class DecoratedObserver(Observer):
            def __init__(self):
                super().__init__(priority=priority)
                if signal_types:
                    for signal_type in signal_types:
                        self.add_signal_filter(signal_type)
            
            async def on_signal(self, signal: SignalPayload):
                if inspect.iscoroutinefunction(func):
                    return await func(signal)
                else:
                    return func(signal)
        
        return DecoratedObserver()
    
    return decorator


# =============================================================================
# EXAMPLES AND TESTING
# =============================================================================

class ExampleObserver(Observer):
    """Example observer implementation."""
    
    def __init__(self, name: str = "example"):
        super().__init__(f"example_{name}")
        self.name = name
    
    async def on_signal(self, signal: SignalPayload):
        logger.info(f"Observer {self.name} received: {signal.signal_type}")
        # Simulate some processing
        await asyncio.sleep(0.01)
        return f"Processed by {self.name}"


async def run_example():
    """Example usage of the signal system."""
    # Create signal line
    signal_line = await create_signal_line("example_app")
    
    # Create observers
    fast_observer = ExampleObserver("fast")
    slow_observer = ExampleObserver("slow") 
    
    # Register observers
    await signal_line.register_observer(fast_observer)
    await signal_line.register_observer(slow_observer)
    
    # Register callback observer
    await signal_line.register_callback(
        lambda signal: print(f"Callback got: {signal.signal_type}"),
        signal_filters=[SignalType.CUSTOM]
    )
    
    # Emit signals
    signal = SignalPayload(
        signal_type=SignalType.CUSTOM,
        source_id="example_emitter",
        data={"message": "Hello, World!"}
    )
    
    results = await signal_line.emit(signal)
    print(f"Signal emitted to {results['total_notified']} observers")
    
    # Get statistics
    stats = signal_line.get_line_stats()
    print(f"Line stats: {stats}")
    
    observer_stats = await signal_line.get_observer_stats()
    print(f"Observer stats: {observer_stats}")
    
    # Cleanup
    await signal_line.stop()


if __name__ == "__main__":
    # Run example
    asyncio.run(run_example())