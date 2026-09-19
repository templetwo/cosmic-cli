"""In-process MissionBus. Subscriber failures never raise into the agent loop."""

from __future__ import annotations

import copy
import logging
import threading
from typing import Callable, List

logger = logging.getLogger(__name__)

EventCallback = Callable[[dict], None]
Unsubscribe = Callable[[], None]


class LocalMissionBus:
    """Process-local pub/sub. Each subscriber receives its own deepcopy."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subs: List[EventCallback] = []

    def subscribe(self, fn: EventCallback) -> Unsubscribe:
        with self._lock:
            self._subs.append(fn)

        def unsubscribe() -> None:
            self.unsubscribe(fn)

        return unsubscribe

    def unsubscribe(self, fn: EventCallback) -> None:
        """Idempotent. Missing subscribers are a no-op."""
        with self._lock:
            try:
                self._subs.remove(fn)
            except ValueError:
                return

    def publish(self, event: dict) -> None:
        snapshot = copy.deepcopy(event)
        with self._lock:
            subs = list(self._subs)
        for fn in subs:
            try:
                fn(copy.deepcopy(snapshot))
            except Exception:
                logger.exception("bus subscriber")
