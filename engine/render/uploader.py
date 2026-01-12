"""
Upload Queue with Resource Versioning

This module provides thread-safe upload of CPU data to GPU resources.
Workers push upload jobs; the GL thread drains them with a budget.

Key principles:
1. Workers NEVER touch GL objects - they only create UploadJobs with CPU data
2. UploadJobs contain raw bytes/numpy arrays - pure data, no GL references
3. Version guards prevent stale async jobs from clobbering newer data
4. Budget-based draining prevents frame hitches from large uploads
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, Callable, List
from enum import Enum, auto
from collections import deque
import threading
import time
import numpy as np


class UploadKind(Enum):
    """Types of GPU resources that can be uploaded."""
    MESH_VERTICES = auto()
    MESH_INDICES = auto()
    INSTANCE_BUFFER = auto()
    TEXTURE_2D = auto()
    TEXTURE_CUBE = auto()
    UNIFORM_BUFFER = auto()


class UploadPriority(Enum):
    """Upload priority levels. Higher = processed first."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    IMMEDIATE = 3  # Process this frame no matter what


@dataclass
class UploadJob:
    """
    A pending GPU upload request.
    
    Contains ONLY CPU data - no GL objects, no context references.
    This is critical for thread safety.
    """
    kind: UploadKind
    resource_key: str
    payload: bytes  # Raw bytes to upload
    version: int    # For version guards
    priority: UploadPriority = UploadPriority.NORMAL
    
    # Metadata for specific upload types
    width: int = 0
    height: int = 0
    components: int = 4
    dtype: str = "f4"  # numpy dtype string
    
    # Timing info
    created_at: float = field(default_factory=time.perf_counter)
    
    @property
    def nbytes(self) -> int:
        return len(self.payload)
    
    def __lt__(self, other: "UploadJob") -> bool:
        """For priority queue ordering (higher priority first)."""
        return self.priority.value > other.priority.value


@dataclass
class ResourceVersion:
    """Tracks version of a GPU resource."""
    current_version: int = 0
    pending_version: int = 0
    last_upload_time: float = 0.0


class UploadQueue:
    """
    Thread-safe queue for GPU upload jobs.
    
    Usage pattern:
    - Workers call push() to enqueue upload jobs
    - GL thread calls drain() each frame with a byte budget
    - Version guards prevent stale uploads
    """
    
    def __init__(self, max_queue_size: int = 1000):
        self._queue: deque[UploadJob] = deque()
        self._lock = threading.Lock()
        self._max_size = max_queue_size
        
        # Version tracking per resource
        self._versions: Dict[str, ResourceVersion] = {}
        self._version_lock = threading.Lock()
        
        # Stats
        self._total_bytes_queued = 0
        self._total_jobs_queued = 0
        self._dropped_stale = 0
    
    def push(self, job: UploadJob) -> bool:
        """
        Enqueue an upload job. Thread-safe.
        
        Returns True if queued, False if dropped (queue full or version stale).
        """
        # Check version first
        with self._version_lock:
            ver = self._versions.setdefault(
                job.resource_key, 
                ResourceVersion()
            )
            if job.version < ver.current_version:
                # Stale job - a newer version already uploaded
                self._dropped_stale += 1
                return False
            
            # Update pending version
            ver.pending_version = max(ver.pending_version, job.version)
        
        with self._lock:
            if len(self._queue) >= self._max_size:
                return False
            
            self._queue.append(job)
            self._total_bytes_queued += job.nbytes
            self._total_jobs_queued += 1
            return True
    
    def drain(
        self, 
        byte_budget: int,
        processor: Callable[[UploadJob], bool]
    ) -> DrainResult:
        """
        Process queued jobs up to byte_budget.
        
        Args:
            byte_budget: Maximum bytes to upload this frame
            processor: Callback that performs actual GL upload.
                       Returns True on success, False to requeue.
        
        Returns:
            DrainResult with stats about what was processed.
        
        MUST be called from GL thread only!
        """
        processed_count = 0
        processed_bytes = 0
        skipped_stale = 0
        remaining_budget = byte_budget
        
        # Sort by priority (process high priority first)
        with self._lock:
            jobs_to_process = list(self._queue)
            self._queue.clear()
        
        jobs_to_process.sort(reverse=True)  # Uses __lt__ for priority
        
        requeue: List[UploadJob] = []
        
        for job in jobs_to_process:
            # Version check
            with self._version_lock:
                ver = self._versions.get(job.resource_key)
                if ver and job.version < ver.current_version:
                    skipped_stale += 1
                    continue
            
            # Budget check (IMMEDIATE priority bypasses)
            if job.priority != UploadPriority.IMMEDIATE:
                if job.nbytes > remaining_budget:
                    requeue.append(job)
                    continue
            
            # Process the upload
            success = processor(job)
            
            if success:
                processed_count += 1
                processed_bytes += job.nbytes
                remaining_budget -= job.nbytes
                
                # Update version tracking
                with self._version_lock:
                    ver = self._versions.get(job.resource_key)
                    if ver:
                        ver.current_version = max(ver.current_version, job.version)
                        ver.last_upload_time = time.perf_counter()
            else:
                # Failed - requeue for next frame
                requeue.append(job)
        
        # Requeue unprocessed jobs
        with self._lock:
            for job in requeue:
                self._queue.appendleft(job)
        
        return DrainResult(
            processed_count=processed_count,
            processed_bytes=processed_bytes,
            skipped_stale=skipped_stale,
            remaining_in_queue=len(requeue),
        )
    
    def get_resource_version(self, key: str) -> int:
        """Get current version of a resource (for version checks)."""
        with self._version_lock:
            ver = self._versions.get(key)
            return ver.current_version if ver else 0
    
    def bump_version(self, key: str) -> int:
        """
        Allocate a new version number for a resource.
        Call this when preparing an upload to get the version to use.
        """
        with self._version_lock:
            ver = self._versions.setdefault(key, ResourceVersion())
            ver.pending_version += 1
            return ver.pending_version
    
    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        with self._lock:
            queue_size = len(self._queue)
            queue_bytes = sum(j.nbytes for j in self._queue)
        
        return {
            "queue_size": queue_size,
            "queue_bytes": queue_bytes,
            "total_bytes_queued": self._total_bytes_queued,
            "total_jobs_queued": self._total_jobs_queued,
            "dropped_stale": self._dropped_stale,
        }
    
    def clear(self):
        """Clear all pending jobs (e.g., on context loss)."""
        with self._lock:
            self._queue.clear()


@dataclass
class DrainResult:
    """Result of draining the upload queue."""
    processed_count: int
    processed_bytes: int
    skipped_stale: int
    remaining_in_queue: int


# Convenience functions for creating common upload jobs

def create_mesh_upload(
    key: str,
    vertices: np.ndarray,
    indices: Optional[np.ndarray],
    version: int,
    priority: UploadPriority = UploadPriority.NORMAL
) -> List[UploadJob]:
    """Create upload jobs for a mesh (vertices + optional indices)."""
    jobs = []
    
    jobs.append(UploadJob(
        kind=UploadKind.MESH_VERTICES,
        resource_key=f"{key}:vertices",
        payload=vertices.astype(np.float32).tobytes(),
        version=version,
        priority=priority,
    ))
    
    if indices is not None:
        jobs.append(UploadJob(
            kind=UploadKind.MESH_INDICES,
            resource_key=f"{key}:indices",
            payload=indices.astype(np.uint32).tobytes(),
            version=version,
            priority=priority,
        ))
    
    return jobs


def create_instance_buffer_upload(
    key: str,
    instance_data: np.ndarray,
    version: int,
    priority: UploadPriority = UploadPriority.HIGH
) -> UploadJob:
    """Create upload job for instance buffer (transforms, colors, etc.)."""
    return UploadJob(
        kind=UploadKind.INSTANCE_BUFFER,
        resource_key=key,
        payload=instance_data.astype(np.float32).tobytes(),
        version=version,
        priority=priority,
    )


def create_texture_upload(
    key: str,
    pixels: np.ndarray,
    width: int,
    height: int,
    components: int,
    version: int,
    priority: UploadPriority = UploadPriority.NORMAL
) -> UploadJob:
    """Create upload job for a 2D texture."""
    return UploadJob(
        kind=UploadKind.TEXTURE_2D,
        resource_key=key,
        payload=pixels.tobytes(),
        version=version,
        priority=priority,
        width=width,
        height=height,
        components=components,
    )
