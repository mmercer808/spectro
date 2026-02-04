"""
WebSocket client and dark scheduling for saving project edits.

- WSClient: connect to accepted WS server; send edits; push responses to a queue.
- DarkScheduler: background queue that sends edit payloads over WS.
- ResponseLog: append-only text space; drain WS response queue and append so user can see what happens.
"""

from .response_log import ResponseLog
from .client import WSClient
from .dark_scheduler import DarkScheduler

__all__ = ["ResponseLog", "WSClient", "DarkScheduler"]
