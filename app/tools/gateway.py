from time import perf_counter
from typing import Any, Callable
from uuid import uuid4

from pydantic import ValidationError

from app.environments.support.environment import SupportEnvironment
from app.execution.budget import ExecutionBudget, ExecutionLimitExceeded
from app.telemetry.recorder import TraceRecorder
from app.tools.schemas import (
    GetOrderArguments,
    GetRefundStatusArguments,
    GetReturnStatusArguments,
    IssueRefundArguments,
    RequestRefundApprovalArguments,
    UpdateTicketArguments,
)


class ToolGateway:
    def __init__(
        self,
        *,
        environment: SupportEnvironment,
        recorder: TraceRecorder,
        budget: ExecutionBudget | None = None,
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        if recorder.environment_id != environment.environment_id:
            raise ValueError("Recorder and environment must match.")

        self._environment = environment
        self._recorder = recorder
        self._budget = budget
        self._clock = clock if budget is None else budget._clock

        # Only these tools can be called by an agent.
        self._tools = {
            "get_order": (
                GetOrderArguments,
                environment.get_order,
            ),
            "get_return_status": (
                GetReturnStatusArguments,
                environment.get_return_status,
            ),
            "get_refund_status": (
                GetRefundStatusArguments,
                environment.get_refund_status,
            ),
            "request_refund_approval": (
                RequestRefundApprovalArguments,
                environment.request_refund_approval,
            ),
            "issue_refund": (
                IssueRefundArguments,
                environment.issue_refund,
            ),
            "update_ticket": (
                UpdateTicketArguments,
                environment.update_ticket,
            ),
        }

    @property
    def recorder(self) -> TraceRecorder:
        return self._recorder

    @property
    def budget(self) -> ExecutionBudget | None:
        return self._budget

    @property
    def clock(self) -> Callable[[], float]:
        return self._clock

    def get_order(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Keep the existing interface while callers adopt call()."""
        return self.call("get_order", arguments)

    def call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        tool_call_id = str(uuid4())

        self._recorder.record(
            event_type="tool_requested",
            tool_call_id=tool_call_id,
            payload={
                "tool_name": tool_name,
                "arguments": arguments,
            },
        )

        if self._budget is not None:
            try:
                self._budget.check_and_consume_tool_call()
            except ExecutionLimitExceeded as exc:
                self._recorder.record(
                    event_type="tool_limit_exceeded",
                    tool_call_id=tool_call_id,
                    payload={
                        "tool_name": tool_name,
                        "limit_type": exc.limit_type,
                        "error": {
                            "code": exc.code,
                            "message": exc.message,
                        },
                    },
                )
                raise

        if tool_name not in self._tools:
            return self._reject(
                tool_name,
                tool_call_id,
                code="UNKNOWN_TOOL",
                message="Tool is not available.",
            )

        schema, handler = self._tools[tool_name]

        try:
            validated = schema.model_validate(arguments)
        except ValidationError:
            return self._reject(
                tool_name,
                tool_call_id,
                code="INVALID_ARGUMENTS",
                message="Arguments do not match the tool schema.",
            )

        before = self._environment.snapshot()
        started = self._clock()

        # Keep this try block limited to tool execution.
        try:
            result = handler(**validated.model_dump())
        except PermissionError:
            return self._reject(
                tool_name,
                tool_call_id,
                code="ACCESS_DENIED",
                message=(
                    "Order not found or access denied."
                    if tool_name != "update_ticket"
                    else "Ticket not found or access denied."
                ),
            )
        except ValueError:
            return self._reject(
                tool_name,
                tool_call_id,
                code="PRECONDITION_FAILED",
                message="Tool preconditions were not satisfied.",
            )
        except Exception:
            self._recorder.record(
                event_type="tool_failed",
                tool_call_id=tool_call_id,
                payload={
                    "tool_name": tool_name,
                    "error": {
                        "code": "INTERNAL_ERROR",
                        "message": "Unexpected tool execution failure.",
                    },
                },
            )
            raise

        duration_ms = (self._clock() - started) * 1000
        after = self._environment.snapshot()

        if after != before:
            self._recorder.record(
                event_type="environment_changed",
                tool_call_id=tool_call_id,
                payload={
                    "tool_name": tool_name,
                    "before": before,
                    "after": after,
                },
            )

        self._recorder.record(
            event_type="tool_finished",
            tool_call_id=tool_call_id,
            payload={
                "tool_name": tool_name,
                "result": result,
                "duration_ms": duration_ms,
            },
        )

        if self._budget is not None:
            try:
                self._budget.check_deadline()
            except ExecutionLimitExceeded as exc:
                self._recorder.record(
                    event_type="tool_limit_exceeded",
                    tool_call_id=tool_call_id,
                    payload={
                        "tool_name": tool_name,
                        "limit_type": exc.limit_type,
                        "error": {
                            "code": exc.code,
                            "message": exc.message,
                        },
                    },
                )
                raise

        return {
            "ok": True,
            "tool_call_id": tool_call_id,
            "result": result,
        }

    def _reject(
        self,
        tool_name: str,
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
                "tool_name": tool_name,
                "error": error,
            },
        )

        if self._budget is not None:
            try:
                self._budget.check_deadline()
            except ExecutionLimitExceeded as exc:
                self._recorder.record(
                    event_type="tool_limit_exceeded",
                    tool_call_id=tool_call_id,
                    payload={
                        "tool_name": tool_name,
                        "limit_type": exc.limit_type,
                        "error": {
                            "code": exc.code,
                            "message": exc.message,
                        },
                    },
                )
                raise

        return {
            "ok": False,
            "tool_call_id": tool_call_id,
            "error": error,
        }