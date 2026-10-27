"""Tool gateway enforcing isolation, logging, validation, and permissions."""
from typing import Any
from uuid import uuid4

from pydantic import ValidationError

from app.environments.support.environment import SupportEnvironment
from app.telemetry.recorder import TraceRecorder
from app.tools.schemas import GetOrderArguments


class ToolGateway:
    def __init__(
        self,
        *,
        environment: SupportEnvironment,
        recorder: TraceRecorder,
    ) -> None:
        if recorder.environment_id != environment.environment_id:
            raise ValueError("Recorder and environment must match.")

        self._environment = environment
        self._recorder = recorder

    def get_order(self, arguments: dict[str, Any]) -> dict[str, Any]:
        tool_call_id = str(uuid4())

        self._recorder.record(
            event_type="tool_requested",
            tool_call_id=tool_call_id,
            payload={
                "tool_name": "get_order",
                "arguments": arguments,
            },
        )

        try:
            validated = GetOrderArguments.model_validate(arguments)
        except ValidationError:
            return self._reject(
                tool_call_id,
                code="INVALID_ARGUMENTS",
                message="Provide only a non-blank string order_id.",
            )

        try:
            order = self._environment.get_order(validated.order_id)
        except PermissionError:
            return self._reject(
                tool_call_id,
                code="ACCESS_DENIED",
                message="Order not found or access denied.",
            )
        except Exception:
            self._recorder.record(
                event_type="tool_failed",
                tool_call_id=tool_call_id,
                payload={
                    "tool_name": "get_order",
                    "error": {
                        "code": "INTERNAL_ERROR",
                        "message": "Unexpected tool execution failure.",
                    },
                },
            )
            raise

        self._recorder.record(
            event_type="tool_finished",
            tool_call_id=tool_call_id,
            payload={
                "tool_name": "get_order",
                "result": order,
            },
        )

        return {
            "ok": True,
            "tool_call_id": tool_call_id,
            "result": order,
        }

    def _reject(
        self,
        tool_call_id: str,
        *,
        code: str,
        message: str,
    ) -> dict[str, Any]:
        error = {"code": code, "message": message}

        self._recorder.record(
            event_type="tool_rejected",
            tool_call_id=tool_call_id,
            payload={
                "tool_name": "get_order",
                "error": error,
            },
        )

        return {
            "ok": False,
            "tool_call_id": tool_call_id,
            "error": error,
        }