"""Telemetry recorder capturing metrics, latency, and step counts."""
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


class TraceRecorder:
    """In-memory event history for one sequential execution attempt."""

    def __init__(
        self,
        *,
        attempt_id: str,
        environment_id: str,
    ) -> None:
        self.attempt_id = attempt_id
        self.environment_id = environment_id
        self._events: list[dict[str, Any]] = []

    def record(
        self,
        *,
        event_type: str,
        payload: dict[str, Any],
        tool_call_id: str | None = None,
    ) -> None:
        event = {
            "event_id": str(uuid4()),
            "attempt_id": self.attempt_id,
            "environment_id": self.environment_id,
            "sequence_number": len(self._events) + 1,
            "event_type": event_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tool_call_id": tool_call_id,
            "payload": deepcopy(payload),
        }

        self._events.append(event)

    def get_events(self) -> list[dict[str, Any]]:
        return deepcopy(self._events)