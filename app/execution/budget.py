"""Execution budget component enforcing tool-call counts and cooperative deadlines."""
import math
from time import perf_counter
from typing import Callable


class ExecutionLimitExceeded(Exception):
    """Raised when an execution limit (tool calls or deadline) is exceeded.

    Attributes:
        limit_type: Type of limit exceeded ('tool_calls' or 'deadline').
        code: Stable, safe machine-readable error code.
        message: Human-readable safe explanation.
    """

    def __init__(self, limit_type: str, code: str, message: str) -> None:
        super().__init__(message)
        self.limit_type = limit_type
        self.code = code
        self.message = message


class ExecutionBudget:
    """Manages and enforces tool-call and cooperative deadline budgets.

    Enforces:
    1. Maximum tool-call requests (up to N allowed, N+1 blocked).
    2. Cooperative attempt deadline evaluated monotonically.
    """

    def __init__(
        self,
        *,
        max_tool_calls: int,
        timeout_seconds: float | int,
        max_model_turns: int | None = None,
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        if (
            isinstance(max_tool_calls, bool)
            or not isinstance(max_tool_calls, int)
            or max_tool_calls <= 0
        ):
            raise ValueError("max_tool_calls must be a positive integer.")

        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be a positive finite number.")

        if max_model_turns is not None:
            if (
                isinstance(max_model_turns, bool)
                or not isinstance(max_model_turns, int)
                or max_model_turns <= 0
            ):
                raise ValueError("max_model_turns must be a positive integer.")

        self.max_tool_calls: int = max_tool_calls
        self.timeout_seconds: float = float(timeout_seconds)
        self.max_model_turns: int | None = max_model_turns
        self._clock: Callable[[], float] = clock

        self._start_time: float | None = None
        self._deadline: float | None = None
        self._calls_requested: int = 0
        self._model_turns_requested: int = 0
        self._is_exhausted: bool = False
        self._exhaustion_reason: ExecutionLimitExceeded | None = None

    @property
    def is_exhausted(self) -> bool:
        return self._is_exhausted

    @property
    def exhaustion_reason(self) -> ExecutionLimitExceeded | None:
        return self._exhaustion_reason

    @property
    def calls_requested(self) -> int:
        return self._calls_requested

    @property
    def model_turns_requested(self) -> int:
        return self._model_turns_requested

    def remaining_time(self) -> float:
        """Return the remaining attempt time in seconds, or timeout_seconds if unstarted."""
        if self._deadline is None:
            return self.timeout_seconds
        return max(0.0, self._deadline - self._clock())

    def start(self) -> None:
        """Start the attempt execution budget timer."""
        self._start_time = self._clock()
        self._deadline = self._start_time + self.timeout_seconds

    def check_deadline(self) -> None:
        """Check if the deadline has expired, marking exhaustion if exceeded."""
        if self._is_exhausted and self._exhaustion_reason is not None:
            raise self._exhaustion_reason

        if self._deadline is not None:
            now = self._clock()
            if now >= self._deadline:
                self._is_exhausted = True
                self._exhaustion_reason = ExecutionLimitExceeded(
                    limit_type="deadline",
                    code="DEADLINE_EXCEEDED",
                    message=f"Execution deadline of {self.timeout_seconds} seconds exceeded.",
                )
                raise self._exhaustion_reason

    def check_and_consume_tool_call(self) -> None:
        """Check deadline and call limit, consuming one call slot.

        Permits up to max_tool_calls requests. The (max_tool_calls + 1)th request
        raises ExecutionLimitExceeded.
        """
        if self._is_exhausted and self._exhaustion_reason is not None:
            raise self._exhaustion_reason

        self.check_deadline()

        self._calls_requested += 1
        if self._calls_requested > self.max_tool_calls:
            self._is_exhausted = True
            self._exhaustion_reason = ExecutionLimitExceeded(
                limit_type="tool_calls",
                code="TOOL_CALL_LIMIT_EXCEEDED",
                message=f"Maximum tool-call limit of {self.max_tool_calls} exceeded.",
            )
            raise self._exhaustion_reason

    def check_and_consume_model_turn(self) -> None:
        """Check deadline and model-turn limit, consuming one turn slot.

        Permits up to max_model_turns requests. The (max_model_turns + 1)th request
        raises ExecutionLimitExceeded.
        """
        if self._is_exhausted and self._exhaustion_reason is not None:
            raise self._exhaustion_reason

        self.check_deadline()

        self._model_turns_requested += 1
        if (
            self.max_model_turns is not None
            and self._model_turns_requested > self.max_model_turns
        ):
            self._is_exhausted = True
            self._exhaustion_reason = ExecutionLimitExceeded(
                limit_type="model_turns",
                code="MODEL_TURN_LIMIT_EXCEEDED",
                message=f"Maximum model-turn limit of {self.max_model_turns} exceeded.",
            )
            raise self._exhaustion_reason

    def check_limits(self) -> None:
        """Check if budget has been exhausted or deadline expired after agent return."""
        if self._is_exhausted and self._exhaustion_reason is not None:
            raise self._exhaustion_reason
        self.check_deadline()

