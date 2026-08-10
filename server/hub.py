"""Fan-out of subtitle events from the pipeline thread to WebSocket clients.

The pipeline runs on a plain thread and the sockets live on the asyncio loop, so every publish
crosses that boundary via `call_soon_threadsafe`. Subscribers each get their own bounded queue:
a browser tab that stops reading is dropped from rather than allowed to stall the pipeline.
"""

from __future__ import annotations

import asyncio
import logging

log = logging.getLogger("polyminutes.hub")

QUEUE_SIZE = 256


class Hub:
    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._subscribers: list[asyncio.Queue] = []

    def bind(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=QUEUE_SIZE)
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        if q in self._subscribers:
            self._subscribers.remove(q)

    def publish(self, event: dict) -> None:
        """Called from the pipeline thread."""
        if self._loop is None:
            return
        self._loop.call_soon_threadsafe(self._dispatch, event)

    def _dispatch(self, event: dict) -> None:
        for q in list(self._subscribers):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                log.warning("subscriber queue full, dropping event")
