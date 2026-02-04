"""
Dark scheduler: background queue that sends project edits over WS.

Edits are enqueued; a background thread (or the WS client thread) pops and sends so the main loop is not blocked.
"""

from __future__ import annotations
import threading
import time
from typing import Optional, Deque
from collections import deque

from .client import WSClient
from .response_log import ResponseLog  # noqa: F401 - for type


class DarkScheduler:
    """
    Queue of edit payloads; background thread sends them via WSClient.
    WS responses are pushed to the client's response queue; main thread drains to ResponseLog.
    """

    def __init__(self, client: WSClient, response_log: ResponseLog):
        self.client = client
        self.response_log = response_log
        self._queue: Deque[dict] = deque()
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def start(self) -> None:
        """Start background sender thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self.response_log.append("[dark] scheduler started")

    def stop(self) -> None:
        """Stop background thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        self.response_log.append("[dark] scheduler stopped")

    def enqueue_edit(self, project_id: str, payload: dict) -> None:
        """Enqueue an edit to be sent over WS. Non-blocking."""
        with self._lock:
            self._queue.append({"type": "edit", "project_id": project_id, "payload": payload})
        self.response_log.push_from_queue(f"[dark] enqueued edit for project {project_id}")

    def _run(self) -> None:
        """Background: pop edits and send via client."""
        while self._running:
            with self._lock:
                if not self._queue:
                    item = None
                else:
                    item = self._queue.popleft()
            if item is not None and self.client.is_connected:
                self.client.send(item)
            time.sleep(0.05)
