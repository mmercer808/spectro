"""
WebSocket client: connect to accepted WS server; send edits; push responses to ResponseLog queue.
"""

from __future__ import annotations
import threading
import json
from typing import Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .response_log import ResponseLog

try:
    import websocket
    HAS_WEBSOCKET = True
except ImportError:
    HAS_WEBSOCKET = False


class WSClient:
    """
    Sync WebSocket client. Runs receive loop in a background thread.
    Incoming messages are pushed to ResponseLog via push_from_queue; main thread drains to see them.
    """

    def __init__(self, response_log: Optional["ResponseLog"] = None):
        self.response_log = response_log
        self._ws: Optional[Any] = None
        self._url: Optional[str] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._connected = False
        self._lock = threading.Lock()

    def connect(self, url: str) -> bool:
        """Connect to WS server. Returns True if started (or already connected)."""
        if not HAS_WEBSOCKET:
            self._push_response("[WS] websocket-client not installed; pip install websocket-client")
            return False
        with self._lock:
            if self._running:
                return True
            self._url = url
            self._running = True
            self._push_response(f"[WS] connecting to {url}")
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return True

    def _push_response(self, line: str) -> None:
        if self.response_log is not None:
            self.response_log.push_from_queue(line)

    def disconnect(self) -> None:
        """Stop receive loop and close connection."""
        self._running = False
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._push_response("[WS] disconnected")

    def send(self, data: dict | str) -> bool:
        """Send edit or other payload. Thread-safe. Returns True if sent."""
        if not self._ws:
            self._push_response("[WS] not connected; cannot send")
            return False
        try:
            if isinstance(data, dict):
                payload = json.dumps(data)
            else:
                payload = data
            with self._lock:
                if self._ws:
                    self._ws.send(payload)
                    self._push_response(f"[WS] sent: {payload[:80]}...")
                    return True
        except Exception as e:
            self._push_response(f"[WS] send error: {e}")
        return False

    def _run(self) -> None:
        """Background thread: connect and receive loop."""
        if not HAS_WEBSOCKET or not self._url:
            return
        try:
            self._ws = websocket.WebSocketApp(
                self._url,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
                on_open=self._on_open,
            )
            self._ws.run_forever()
        except Exception as e:
            self._push_response(f"[WS] error: {e}")
        finally:
            self._ws = None

    def _on_message(self, ws: Any, message: str) -> None:
        self._push_response(f"[WS] recv: {message[:200]}")

    def _on_error(self, ws: Any, error: Exception) -> None:
        self._push_response(f"[WS] error: {error}")


    def _on_open(self, ws: Any) -> None:
        self._connected = True
        self._push_response("[WS] connected")

    def _on_close(self, ws: Any, close_status_code: Optional[int], close_msg: Optional[str]) -> None:
        self._connected = False
        self._push_response(f"[WS] closed {close_status_code} {close_msg or ''}")

    @property
    def is_connected(self) -> bool:
        return self._connected and self._running
