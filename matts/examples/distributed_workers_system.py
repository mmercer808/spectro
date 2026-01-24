#!/usr/bin/env python3
"""
Distributed Worker System Example

Demonstrates using serializable contexts for distributed computing where
worker contexts can be serialized, transmitted across network boundaries,
executed on remote machines, and have their results transmitted back.
"""

import asyncio
import uuid
import json
import hashlib
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import time
import random

# matts.functional_interface
from matts import (
    create_context, get_context, transmit_context, emit_signal,
    update_context_data, get_context_data, serialize_context, deserialize_context,
    create_live_code_system, create_callback_system, serialize_source_code
)
# matts.signal_system
from matts import SignalType, SignalPriority
# matts.live_code_system
from matts import SerializedSourceCode

# =============================================================================
# WORKER SYSTEM ENUMS AND DATA STRUCTURES
# =============================================================================

class WorkerStatus(Enum):
    """Worker execution status."""
    IDLE = "idle"
    WORKING = "working"
    COMPLETED = "completed"
    FAILED = "failed"
    OFFLINE = "offline"

class TaskPriority(Enum):
    """Task execution priority."""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BATCH = 5

@dataclass
class WorkerNode:
    """Represents a worker node in the distributed system."""
    worker_id: str
    node_name: str
    capabilities: List[str] = field(default_factory=list)
    status: WorkerStatus = WorkerStatus.IDLE
    current_task_id: Optional[str] = None
    last_heartbeat: datetime = field(default_factory=datetime.now)
    processed_tasks: int = 0
    failed_tasks: int = 0
    total_processing_time: float = 0.0
    context_id: Optional[str] = None

@dataclass
class DistributedTask:
    """Represents a task to be executed in the distributed system."""
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_type: str = "generic"
    priority: TaskPriority = TaskPriority.NORMAL
    required_capabilities: List[str] = field(default_factory=list)
    input_data: Dict[str, Any] = field(default_factory=dict)
    serialized_code: Optional[SerializedSourceCode] = None
    estimated_duration: float = 60.0  # seconds
    max_retries: int = 3
    timeout: float = 300.0  # seconds
    created_at: datetime = field(default_factory=datetime.now)
    assigned_worker: Optional[str] = None
    status: str = "pending"
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    attempts: int = 0

@dataclass
class TaskResult:
    """Result of task execution."""
    task_id: str
    worker_id: str
    success: bool
    result_data: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    execution_time: float = 0.0
    completion_time: datetime = field(default_factory=datetime.now)
    context_snapshot_id: Optional[str] = None

# =============================================================================
# DISTRIBUTED WORKER SYSTEM
# =============================================================================

class DistributedWorkerSystem:
    """
    Complete distributed worker system using serializable contexts.
    
    Features:
    - Worker registration and heartbeat monitoring
    - Task distribution with capability matching
    - Serializable task execution contexts
    - Live code injection for dynamic task definitions
    - Cross-network context transmission
    - Fault tolerance and task redistribution
    - Real-time monitoring and statistics
    """
    
    def __init__(self, system_id: str = None):
        self.system_id = system_id or f"distributed_system_{uuid.uuid4()}"
        
        # Core contexts
        self.coordinator_context_id: Optional[str] = None
        self.worker_contexts: Dict[str, str] = {}  # worker_id -> context_id
        self.task_contexts: Dict[str, str] = {}    # task_id -> context_id
        
        # System state
        self.workers: Dict[str, WorkerNode] = {}
        self.pending_tasks: Dict[str, DistributedTask] = {}
        self.active_tasks: Dict[str, DistributedTask] = {}
        self.completed_tasks: Dict[str, DistributedTask] = {}
        
        # Live code system for dynamic task injection
        self.live_code_system = None
        self.callback_system = None
        
        # Monitoring
        self.heartbeat_interval = 30.0  # seconds
        self.task_timeout_check_interval = 60.0  # seconds
        self.monitoring_active = False
        
        # Statistics
        self.created_at = datetime.now()
        self.total_tasks_processed = 0
        self.total_processing_time = 0.0
    
    async def initialize_system(self, trusted_mode: bool = True) -> Dict[str, str]:
        """Initialize the distributed worker system."""
        print(f"🔧 Initializing Distributed Worker System: {self.system_id}")
        
        # Create coordinator context
        coordinator_context = create_context(f"{self.system_id}_coordinator")
        self.coordinator_context_id = coordinator_context.context_id
        
        # Initialize coordinator state
        await update_context_data(self.coordinator_context_id, {
            'system_id': self.system_id,
            'created_at': self.created_at.isoformat(),
            'worker_registry': {},
            'task_queue': [],
            'system_stats': {
                'workers_registered': 0,
                'tasks_completed': 0,
                'average_task_time': 0.0,
                'system_uptime': 0.0
            },
            'load_balancing': {
                'strategy': 'capability_based',
                'max_tasks_per_worker': 5,
                'task_distribution_algorithm': 'round_robin'
            }
        })
        
        # Initialize live code systems
        self.live_code_system = create_live_code_system(trusted_mode)
        self.callback_system = create_callback_system(trusted_mode)
        
        # Start monitoring
        self.monitoring_active = True
        asyncio.create_task(self._heartbeat_monitor())
        asyncio.create_task(self._task_timeout_monitor())
        
        print(f"✅ System initialized with coordinator context: {self.coordinator_context_id}")
        
        return {
            'coordinator_context_id': self.coordinator_context_id,
            'system_id': self.system_id,
            'trusted_mode': trusted_mode
        }
    
    async def register_worker(self, worker_id: str, node_name: str, 
                            capabilities: List[str]) -> WorkerNode:
        """Register a new worker node."""
        print(f"👷 Registering worker: {node_name} ({worker_id})")
        
        # Create worker context
        worker_context = create_context(f"{self.system_id}_worker_{worker_id}")
        self.worker_contexts[worker_id] = worker_context.context_id
        
        # Create worker node
        worker = WorkerNode(
            worker_id=worker_id,
            node_name=node_name,
            capabilities=capabilities,
            context_id=worker_context.context_id
        )
        
        # Initialize worker context
        await update_context_data(worker_context.context_id, {
            'worker_info': {
                'worker_id': worker_id,
                'node_name': node_name,
                'capabilities': capabilities,
                'registration_time': datetime.now().isoformat()
            },
            'execution_environment': {
                'python_version': '3.8+',
                'available_memory': '4GB',
                'cpu_cores': 4,
                'network_bandwidth': '1Gbps'
            },
            'current_status': {
                'status': WorkerStatus.IDLE.value,
                'current_task': None,
                'last_heartbeat': datetime.now().isoformat(),
                'task_queue': []
            },
            'performance_metrics': {
                'tasks_completed': 0,
                'tasks_failed': 0,
                'average_task_time': 0.0,
                'total_uptime': 0.0
            }
        })
        
        self.workers[worker_id] = worker
        
        # Update coordinator registry
        await update_context_data(self.coordinator_context_id, {
            f'worker_registry.{worker_id}': {
                'node_name': node_name,
                'capabilities': capabilities,
                'registered_at': datetime.now().isoformat(),
                'context_id': worker_context.context_id
            }
        })
        
        # Emit worker registration signal
        await emit_signal(
            SignalType.CUSTOM,
            source_context_id=worker_context.context_id,
            data={
                'event_type': 'worker_registered',
                'worker_id': worker_id,
                'capabilities': capabilities
            }
        )
        
        print(f"✅ Worker registered with context: {worker_context.context_id}")
        return worker
    
    async def submit_task(self, task: DistributedTask) -> str:
        """Submit a task for distributed execution."""
        print(f"📋 Submitting task: {task.task_id} (type: {task.task_type})")
        
        # Create task context
        task_context = create_context(f"{self.system_id}_task_{task.task_id}")
        self.task_contexts[task.task_id] = task_context.context_id
        
        # Initialize task context with all task data
        await update_context_data(task_context.context_id, {
            'task_info': {
                'task_id': task.task_id,
                'task_type': task.task_type,
                'priority': task.priority.value,
                'required_capabilities': task.required_capabilities,
                'created_at': task.created_at.isoformat(),
                'estimated_duration': task.estimated_duration
            },
            'input_data': task.input_data,
            'execution_state': {
                'status': 'pending',
                'assigned_worker': None,
                'attempts': 0,
                'error_history': []
            },
            'serialized_code': (
                task.serialized_code.to_message_dict() 
                if task.serialized_code else None
            ),
            'result': None
        })
        
        # Add to pending tasks
        self.pending_tasks[task.task_id] = task
        
        # Try to assign immediately
        await self._try_assign_task(task.task_id)
        
        # Emit task submission signal
        await emit_signal(
            SignalType.CUSTOM,
            source_context_id=task_context.context_id,
            data={
                'event_type': 'task_submitted',
                'task_id': task.task_id,
                'task_type': task.task_type,
                'priority': task.priority.value
            },
            priority=SignalPriority.HIGH
        )
        
        return task_context.context_id
    
    async def submit_code_task(self, task_type: str, input_data: Dict[str, Any],
                             code_source: str, priority: TaskPriority = TaskPriority.NORMAL,
                             required_capabilities: List[str] = None) -> str:
        """Submit a task with live code for execution."""
        print(f"💻 Submitting code task: {task_type}")
        
        # Serialize the code
        serialized_code = serialize_source_code(
            code_source, 
            name=f"{task_type}_execution",
            code_type="function",
            trusted=True
        )
        
        # Create task
        task = DistributedTask(
            task_type=task_type,
            priority=priority,
            required_capabilities=required_capabilities or [],
            input_data=input_data,
            serialized_code=serialized_code
        )
        
        return await self.submit_task(task)
    
    async def execute_task_on_worker(self, task_id: str, worker_id: str) -> TaskResult:
        """Execute a task on a specific worker (simulated execution)."""
        if task_id not in self.active_tasks:
            raise ValueError(f"Task {task_id} not found in active tasks")
        
        if worker_id not in self.workers:
            raise ValueError(f"Worker {worker_id} not found")
        
        task = self.active_tasks[task_id]
        worker = self.workers[worker_id]
        
        print(f"⚡ Executing task {task_id} on worker {worker_id}")
        
        start_time = time.time()
        
        try:
            # Update worker status
            worker.status = WorkerStatus.WORKING
            worker.current_task_id = task_id
            
            await update_context_data(worker.context_id, {
                'current_status.status': WorkerStatus.WORKING.value,
                'current_status.current_task': task_id
            })
            
            # Transmit task context to worker for execution
            task_context_id = self.task_contexts[task_id]
            transmission_result = await transmit_context(task_context_id, worker.context_id)
            
            # Simulate task execution
            execution_result = await self._simulate_task_execution(task, worker)
            
            execution_time = time.time() - start_time
            
            # Create task result
            result = TaskResult(
                task_id=task_id,
                worker_id=worker_id,
                success=execution_result['success'],
                result_data=execution_result.get('data', {}),
                error_message=execution_result.get('error'),
                execution_time=execution_time
            )
            
            # Update task context with result
            await update_context_data(task_context_id, {
                'result': {
                    'success': result.success,
                    'data': result.result_data,
                    'error': result.error_message,
                    'execution_time': execution_time,
                    'completed_at': result.completion_time.isoformat()
                },
                'execution_state.status': 'completed' if result.success else 'failed'
            })
            
            # Update worker statistics
            worker.processed_tasks += 1
            worker.total_processing_time += execution_time
            worker.status = WorkerStatus.IDLE
            worker.current_task_id = None
            
            await update_context_data(worker.context_id, {
                'current_status.status': WorkerStatus.IDLE.value,
                'current_status.current_task': None,
                'performance_metrics.tasks_completed': worker.processed_tasks,
                'performance_metrics.total_processing_time': worker.total_processing_time
            })
            
            # Move task to completed
            if result.success:
                task.status = "completed"
                task.result = result.result_data
                self.completed_tasks[task_id] = self.active_tasks.pop(task_id)
                self.total_tasks_processed += 1
                self.total_processing_time += execution_time
            else:
                task.status = "failed"
                task.error_message = result.error_message
                task.attempts += 1
                
                # Retry logic
                if task.attempts < task.max_retries:
                    print(f"🔄 Retrying task {task_id} (attempt {task.attempts + 1})")
                    self.pending_tasks[task_id] = self.active_tasks.pop(task_id)
                    asyncio.create_task(self._try_assign_task(task_id))
                else:
                    self.completed_tasks[task_id] = self.active_tasks.pop(task_id)
                    worker.failed_tasks += 1
            
            # Emit task completion signal
            await emit_signal(
                SignalType.CUSTOM,
                source_context_id=worker.context_id,
                data={
                    'event_type': 'task_completed',
                    'task_id': task_id,
                    'worker_id': worker_id,
                    'success': result.success,
                    'execution_time': execution_time
                },
                priority=SignalPriority.HIGH
            )
            
            print(f"✅ Task {task_id} completed on worker {worker_id}: {result.success}")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            error_result = TaskResult(
                task_id=task_id,
                worker_id=worker_id,
                success=False,
                error_message=str(e),
                execution_time=execution_time
            )
            
            # Update worker status
            worker.status = WorkerStatus.IDLE
            worker.current_task_id = None
            worker.failed_tasks += 1
            
            print(f"❌ Task {task_id} failed on worker {worker_id}: {e}")
            return error_result
    
    async def simulate_distributed_workload(self, num_tasks: int = 10, 
                                          task_types: List[str] = None) -> Dict[str, Any]:
        """Simulate a distributed workload with multiple tasks."""
        print(f"🚀 Simulating distributed workload: {num_tasks} tasks")
        
        task_types = task_types or ['data_processing', 'ml_training', 'image_processing', 'batch_analysis']
        
        submitted_tasks = []
        
        # Submit multiple tasks
        for i in range(num_tasks):
            task_type = random.choice(task_types)
            
            # Create sample task code
            task_code = f"""
def execute_task(input_data, context):
    '''Execute {task_type} task'''
    import time
    import random
    
    # Simulate processing time
    processing_time = random.uniform(1, 5)
    time.sleep(processing_time)
    
    # Simulate different types of processing
    if input_data.get('task_type') == 'data_processing':
        result = {{
            'processed_items': input_data.get('items', 0) * 2,
            'processing_time': processing_time,
            'algorithm_used': 'optimized_v2'
        }}
    elif input_data.get('task_type') == 'ml_training':
        result = {{
            'model_accuracy': random.uniform(0.8, 0.95),
            'training_epochs': input_data.get('epochs', 10),
            'loss': random.uniform(0.1, 0.3)
        }}
    else:
        result = {{
            'status': 'completed',
            'output_size': random.randint(100, 1000),
            'quality_score': random.uniform(0.7, 1.0)
        }}
    
    return result
"""
            
            # Submit task
            task_context_id = await self.submit_code_task(
                task_type=task_type,
                input_data={
                    'task_type': task_type,
                    'task_index': i,
                    'items': random.randint(100, 1000),
                    'epochs': random.randint(5, 20)
                },
                code_source=task_code,
                priority=random.choice(list(TaskPriority)),
                required_capabilities=[task_type]
            )
            
            submitted_tasks.append(task_context_id)
        
        # Wait for all tasks to complete (with timeout)
        max_wait_time = 300  # 5 minutes
        start_wait = time.time()
        
        while (self.pending_tasks or self.active_tasks) and (time.time() - start_wait) < max_wait_time:
            await asyncio.sleep(1)
            
            # Process any pending assignments
            for task_id in list(self.pending_tasks.keys()):
                await self._try_assign_task(task_id)
        
        # Collect results
        completed_count = len(self.completed_tasks)
        failed_count = sum(1 for task in self.completed_tasks.values() if task.status == 'failed')
        
        result_summary = {
            'total_submitted': num_tasks,
            'completed': completed_count,
            'failed': failed_count,
            'still_pending': len(self.pending_tasks),
            'still_active': len(self.active_tasks),
            'average_execution_time': (
                self.total_processing_time / max(self.total_tasks_processed, 1)
            ),
            'submitted_task_contexts': submitted_tasks
        }
        
        print(f"📊 Workload simulation complete: {result_summary}")
        return result_summary
    
    async def _try_assign_task(self, task_id: str):
        """Try to assign a pending task to an available worker."""
        if task_id not in self.pending_tasks:
            return
        
        task = self.pending_tasks[task_id]
        
        # Find suitable workers
        suitable_workers = []
        for worker_id, worker in self.workers.items():
            if (worker.status == WorkerStatus.IDLE and
                all(cap in worker.capabilities for cap in task.required_capabilities)):
                suitable_workers.append(worker)
        
        if suitable_workers:
            # Simple assignment: choose worker with least processed tasks
            chosen_worker = min(suitable_workers, key=lambda w: w.processed_tasks)
            
            # Move task to active
            self.active_tasks[task_id] = self.pending_tasks.pop(task_id)
            task.assigned_worker = chosen_worker.worker_id
            task.status = "active"
            
            # Execute task
            asyncio.create_task(self.execute_task_on_worker(task_id, chosen_worker.worker_id))
    
    async def _simulate_task_execution(self, task: DistributedTask, worker: WorkerNode) -> Dict[str, Any]:
        """Simulate task execution (in real system, this would involve actual code execution)."""
        # Simulate processing time
        processing_time = min(task.estimated_duration, random.uniform(1, 10))
        await asyncio.sleep(processing_time / 10)  # Speed up for demo
        
        # Simulate success/failure
        success_rate = 0.9  # 90% success rate
        success = random.random() < success_rate
        
        if success:
            # Generate mock result based on task type
            if task.task_type == 'data_processing':
                result_data = {
                    'processed_items': task.input_data.get('items', 0) * 2,
                    'algorithm_used': 'distributed_v1',
                    'worker_node': worker.node_name
                }
            elif task.task_type == 'ml_training':
                result_data = {
                    'model_accuracy': random.uniform(0.8, 0.95),
                    'training_epochs': task.input_data.get('epochs', 10),
                    'worker_node': worker.node_name
                }
            else:
                result_data = {
                    'status': 'completed',
                    'output_size': random.randint(100, 1000),
                    'worker_node': worker.node_name
                }
            
            return {'success': True, 'data': result_data}
        else:
            return {'success': False, 'error': 'Simulated execution failure'}
    
    async def _heartbeat_monitor(self):
        """Monitor worker heartbeats."""
        while self.monitoring_active:
            current_time = datetime.now()
            
            for worker_id, worker in self.workers.items():
                # Update heartbeat
                worker.last_heartbeat = current_time
                
                await update_context_data(worker.context_id, {
                    'current_status.last_heartbeat': current_time.isoformat()
                })
            
            await asyncio.sleep(self.heartbeat_interval)
    
    async def _task_timeout_monitor(self):
        """Monitor task timeouts and handle stuck tasks."""
        while self.monitoring_active:
            current_time = datetime.now()
            
            for task_id, task in list(self.active_tasks.items()):
                task_age = (current_time - task.created_at).total_seconds()
                
                if task_age > task.timeout:
                    print(f"⏰ Task {task_id} timed out after {task_age} seconds")
                    
                    # Move to failed
                    task.status = "timeout"
                    task.error_message = f"Task timed out after {task_age} seconds"
                    self.completed_tasks[task_id] = self.active_tasks.pop(task_id)
                    
                    # Free up worker
                    if task.assigned_worker and task.assigned_worker in self.workers:
                        worker = self.workers[task.assigned_worker]
                        worker.status = WorkerStatus.IDLE
                        worker.current_task_id = None
            
            await asyncio.sleep(self.task_timeout_check_interval)
    
    async def get_system_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        uptime = (datetime.now() - self.created_at).total_seconds()
        
        # Worker statistics
        worker_stats = {}
        for worker_id, worker in self.workers.items():
            worker_stats[worker_id] = {
                'node_name': worker.node_name,
                'status': worker.status.value,
                'processed_tasks': worker.processed_tasks,
                'failed_tasks': worker.failed_tasks,
                'average_task_time': (
                    worker.total_processing_time / max(worker.processed_tasks, 1)
                ),
                'capabilities': worker.capabilities
            }
        
        return {
            'system_id': self.system_id,
            'uptime_seconds': uptime,
            'total_workers': len(self.workers),
            'active_workers': sum(1 for w in self.workers.values() if w.status != WorkerStatus.OFFLINE),
            'pending_tasks': len(self.pending_tasks),
            'active_tasks': len(self.active_tasks),
            'completed_tasks': len(self.completed_tasks),
            'total_tasks_processed': self.total_tasks_processed,
            'average_task_processing_time': (
                self.total_processing_time / max(self.total_tasks_processed, 1)
            ),
            'task_success_rate': (
                (self.total_tasks_processed - sum(
                    1 for t in self.completed_tasks.values() if t.status == 'failed'
                )) / max(self.total_tasks_processed, 1)
            ),
            'worker_stats': worker_stats,
            'contexts': {
                'coordinator_context': self.coordinator_context_id,
                'worker_contexts': len(self.worker_contexts),
                'task_contexts': len(self.task_contexts)
            }
        }
    
    async def shutdown_system(self):
        """Gracefully shutdown the distributed system."""
        print(f"🛑 Shutting down distributed system: {self.system_id}")
        
        self.monitoring_active = False
        
        # Wait for active tasks to complete (with timeout)
        timeout = 60  # 1 minute
        start_time = time.time()
        
        while self.active_tasks and (time.time() - start_time) < timeout:
            await asyncio.sleep(1)
        
        # Mark remaining tasks as cancelled
        for task_id, task in self.active_tasks.items():
            task.status = "cancelled"
            task.error_message = "System shutdown"
            self.completed_tasks[task_id] = task
        
        self.active_tasks.clear()
        
        # Update all worker statuses to offline
        for worker in self.workers.values():
            worker.status = WorkerStatus.OFFLINE
        
        print("✅ Distributed system shutdown complete")

# =============================================================================
# DEMO FUNCTIONS
# =============================================================================

async def demo_distributed_worker_system():
    """Demonstrate the distributed worker system."""
    print("\n🔧 Distributed Worker System Demo")
    print("=" * 50)
    
    # Create distributed system
    system = DistributedWorkerSystem("demo_cluster")
    
    # Initialize system
    await system.initialize_system(trusted_mode=True)
    
    # Register workers with different capabilities
    await system.register_worker("worker_1", "DataProcessor-Node-1", 
                                 ["data_processing", "batch_analysis"])
    await system.register_worker("worker_2", "MLTrainer-Node-2", 
                                 ["ml_training", "data_processing"])
    await system.register_worker("worker_3", "ImageProcessor-Node-3", 
                                 ["image_processing", "batch_analysis"])
    await system.register_worker("worker_4", "GeneralWorker-Node-4", 
                                 ["data_processing", "ml_training", "image_processing"])
    
    print(f"\n📋 Registered {len(system.workers)} workers")
    
    # Submit individual tasks
    print("\n📤 Submitting individual tasks:")
    
    # Data processing task
    data_task_code = """
def process_data(input_data, context):
    import time
    time.sleep(2)  # Simulate processing
    
    items = input_data.get('items', [])
    processed = [item * 2 for item in items]
    
    return {
        'original_count': len(items),
        'processed_items': processed,
        'processing_node': context.get('worker_info', {}).get('node_name', 'unknown')
    }
"""
    
    data_task_id = await system.submit_code_task(
        task_type="data_processing",
        input_data={'items': [1, 2, 3, 4, 5]},
        code_source=data_task_code,
        priority=TaskPriority.HIGH,
        required_capabilities=["data_processing"]
    )
    print(f"Submitted data processing task: {data_task_id}")
    
    # ML training task
    ml_task_code = """
def train_model(input_data, context):
    import time
    import random
    
    epochs = input_data.get('epochs', 10)
    
    # Simulate training
    for epoch in range(epochs):
        time.sleep(0.1)  # Speed up for demo
    
    return {
        'model_type': input_data.get('model_type', 'neural_network'),
        'epochs_completed': epochs,
        'final_accuracy': random.uniform(0.85, 0.95),
        'training_node': context.get('worker_info', {}).get('node_name', 'unknown')
    }
"""
    
    ml_task_id = await system.submit_code_task(
        task_type="ml_training",
        input_data={'model_type': 'cnn', 'epochs': 5},
        code_source=ml_task_code,
        priority=TaskPriority.NORMAL,
        required_capabilities=["ml_training"]
    )
    print(f"Submitted ML training task: {ml_task_id}")
    
    # Wait a bit for individual tasks
    await asyncio.sleep(5)
    
    # Simulate distributed workload
    print("\n🚀 Simulating distributed workload:")
    workload_result = await system.simulate_distributed_workload(
        num_tasks=20,
        task_types=["data_processing", "ml_training", "image_processing", "batch_analysis"]
    )
    
    print(f"\nWorkload Results:")
    for key, value in workload_result.items():
        print(f"  {key}: {value}")
    
    # Get system statistics
    print("\n📊 System Statistics:")
    stats = await system.get_system_statistics()
    for key, value in stats.items():
        if isinstance(value, dict) and key == 'worker_stats':
            print(f"  {key}:")
            for worker_id, worker_stat in value.items():
                print(f"    {worker_id}: {worker_stat}")
        elif isinstance(value, dict):
            print(f"  {key}: {dict(value)}")
        else:
            print(f"  {key}: {value}")
    
    # Shutdown system
    await system.shutdown_system()
    
    return system

if __name__ == "__main__":
    # Add parent directory to Python path when running directly
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    
    asyncio.run(demo_distributed_worker_system())