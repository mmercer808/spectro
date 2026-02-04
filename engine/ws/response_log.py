"""
Response log: append-only text space for WS responses.

Hold the space; keep appending. Main thread drains a queue (fed by WS client) and appends to the log so the user can see what happens.
"""

from __future__ import annotations
import threading
from typing import List
from collections import deque


class ResponseLog:
    """
    Thread-safe append-only log for WS responses.
    Hold the space; keep appending. Optional max_lines to trim old entries.
    """

    def __init__(self, max_lines: int = 500):
        self._lines: List[str] = []
        self._lock = threading.Lock()
        self._queue: deque = deque()
        self.max_lines = max_lines

    def append(self, line: str) -> None:
        """Append one line (thread-safe)."""
        with self._lock:
            self._lines.append(line)
            if self.max_lines > 0 and len(self._lines) > self.max_lines:
                self._lines = self._lines[-self.max_lines :]

    def push_from_queue(self, line: str) -> None:
        """Push from another thread; will be appended when drain() is called (or append directly)."""
        self._queue.append(line)

    def drain(self) -> int:
        """Drain the queue and append each item. Call from main thread. Returns count drained."""
        n = 0
        while self._queue:
            try:
                line = self._queue.popleft()
            except IndexError:
                break
            self.append(line)
            n += 1
        return n

    def get_lines(self) -> List[str]:
        """Return a copy of current lines (for display)."""
        with self._lock:
            return list(self._lines)

    def clear(self) -> None:
        """Clear the log (optional; breaks append-only if used)."""
        with self._lock:
            self._lines.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._lines)
