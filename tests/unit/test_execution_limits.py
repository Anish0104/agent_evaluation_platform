"""Deterministic tests for execution limits: tool-call counts and cooperative deadlines."""
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from app.agents.scripted import ScriptedRefundAgent
from app.environments.support.environment import SupportEnvironment
from app.execution.budget import ExecutionBudget, ExecutionLimitExceeded
from app.execution.runner import run_attempt
from app.telemetry.recorder import TraceRecorder
from app.tools.gateway import ToolGateway
from scripts.compare_agents import compare_agents, main as compare_cli_main
from scripts.run_demo import main as run_demo_cli_main

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCENARIO_PATH = PROJECT_ROOT / "datasets/support/dev/refund_eligible_001.json"


class FakeClock:
    def __init__(self, current: float = 1000.0) -> None:
        self.current = current

    def __call__(self) -> float:
        return self.current

    def advance(self, seconds: float) -> None:
        self.current += seconds


@pytest.fixture
def task_data() -> dict[str, Any]:
    return json.loads(SCENARIO_PATH.read_text(encoding="utf-8"))


# --- 1. Budget Unit & Validation Tests ---


def test_budget_rejects_invalid_configurations() -> None:
    # max_tool_calls invalid
    with pytest.raises(ValueError, match="max_tool_calls"):
        ExecutionBudget(max_tool_calls=0, timeout_seconds=10.0)
    with pytest.raises(ValueError, match="max_tool_calls"):
        ExecutionBudget(max_tool_calls=-5, timeout_seconds=10.0)
    with pytest.raises(ValueError, match="max_tool_calls"):
        ExecutionBudget(max_tool_calls=True, timeout_seconds=10.0)  # bool excluded
    with pytest.raises(ValueError, match="max_tool_calls"):
        ExecutionBudget(max_tool_calls="5", timeout_seconds=10.0)  # type: ignore

    # timeout_seconds invalid
    with pytest.raises(ValueError, match="timeout_seconds"):
        ExecutionBudget(max_tool_calls=5, timeout_seconds=0)
    with pytest.raises(ValueError, match="timeout_seconds"):
        ExecutionBudget(max_tool_calls=5, timeout_seconds=-1.0)
    with pytest.raises(ValueError, match="timeout_seconds"):
        ExecutionBudget(max_tool_calls=5, timeout_seconds=True)  # bool excluded
    with pytest.raises(ValueError, match="timeout_seconds"):
        ExecutionBudget(max_tool_calls=5, timeout_seconds=float("nan"))
    with pytest.raises(ValueError, match="timeout_seconds"):
        ExecutionBudget(max_tool_calls=5, timeout_seconds=float("inf"))


def test_runner_rejects_missing_or_invalid_scenario_limits(task_data: dict[str, Any]) -> None:
    bad_task = deepcopy(task_data)
    del bad_task["evaluation_spec"]["limits"]
    with pytest.raises(ValueError, match="limits"):
        run_attempt(task=bad_task, order_id="ord_1042")

    bad_task2 = deepcopy(task_data)
    bad_task2["evaluation_spec"]["limits"]["max_tool_calls"] = 0
    with pytest.raises(ValueError, match="max_tool_calls"):
        run_attempt(task=bad_task2, order_id="ord_1042")


# --- 2. Tool-Call Budget Enforcement ---


def test_exactly_n_requests_allowed_and_n_plus_one_blocked(task_data: dict[str, Any]) -> None:
    task = deepcopy(task_data)
    # Standard ScriptedRefundAgent requires 6 tool calls; restrict to 2
    task["evaluation_spec"]["limits"]["max_tool_calls"] = 2

    clock = FakeClock()
    report = run_attempt(task=task, order_id="ord_1042", clock=clock)

    assert report["execution_status"] == "limit_exceeded"
    assert report["metrics"]["tool_call_count"] == 3  # Exactly 2 allowed + 1 blocked request
    assert report["termination"]["limit_type"] == "tool_calls"
    assert report["termination"]["code"] == "TOOL_CALL_LIMIT_EXCEEDED"

    events = report["events"]
    requested = [e for e in events if e.get("event_type") == "tool_requested"]
    finished = [e for e in events if e.get("event_type") == "tool_finished"]
    tool_limit = [e for e in events if e.get("event_type") == "tool_limit_exceeded"]
    attempt_limit = [e for e in events if e.get("event_type") == "attempt_limit_exceeded"]

    assert len(requested) == 3
    assert len(finished) == 2
    assert len(tool_limit) == 1
    assert len(attempt_limit) == 1

    # The 3rd request was recorded but blocked before handler execution
    assert tool_limit[0]["tool_call_id"] == requested[2]["tool_call_id"]
    assert tool_limit[0]["payload"]["limit_type"] == "tool_calls"

    # Evaluators were skipped
    assert report["evaluations"]["outcome"]["status"] == "not_evaluated"
    assert report["evaluations"]["trajectory"]["status"] == "not_evaluated"


def test_invalid_argument_and_unknown_tool_requests_consume_budget(task_data: dict[str, Any]) -> None:
    task = deepcopy(task_data)
    task["evaluation_spec"]["limits"]["max_tool_calls"] = 2

    class BadCallsAgent:
        def run(self, gateway: Any, order_id: str, ticket_id: str) -> dict[str, Any]:
            # Call 1: invalid arguments
            gateway.call("get_order", {"invalid_arg": 123})
            # Call 2: unknown tool
            gateway.call("unknown_tool", {})
            # Call 3: should be blocked
            gateway.call("get_order", {"order_id": order_id})
            return {"status": "completed"}

    report = run_attempt(task=task, order_id="ord_1042", agent=BadCallsAgent())

    assert report["execution_status"] == "limit_exceeded"
    assert report["metrics"]["tool_call_count"] == 3
    events = report["events"]
    rejections = [e for e in events if e.get("event_type") == "tool_rejected"]
    assert len(rejections) == 2
    assert rejections[0]["payload"]["error"]["code"] == "INVALID_ARGUMENTS"
    assert rejections[1]["payload"]["error"]["code"] == "UNKNOWN_TOOL"


def test_rejected_authorization_requests_consume_budget(task_data: dict[str, Any]) -> None:
    task = deepcopy(task_data)
    task["evaluation_spec"]["limits"]["max_tool_calls"] = 1

    class AuthAndMoreAgent:
        def run(self, gateway: Any, order_id: str, ticket_id: str) -> dict[str, Any]:
            # Call 1: access denied (wrong order id)
            gateway.call("get_order", {"order_id": "ord_WRONG"})
            # Call 2: should be blocked
            gateway.call("get_order", {"order_id": order_id})
            return {"status": "completed"}

    report = run_attempt(task=task, order_id="ord_1042", agent=AuthAndMoreAgent())

    assert report["execution_status"] == "limit_exceeded"
    assert report["metrics"]["tool_call_count"] == 2
    events = report["events"]
    rejections = [e for e in events if e.get("event_type") == "tool_rejected"]
    assert len(rejections) == 1
    assert rejections[0]["payload"]["error"]["code"] == "ACCESS_DENIED"


def test_exhaustion_remains_sticky_if_agent_catches_exception(task_data: dict[str, Any]) -> None:
    task = deepcopy(task_data)
    task["evaluation_spec"]["limits"]["max_tool_calls"] = 1

    class CatchingAgent:
        def run(self, gateway: Any, order_id: str, ticket_id: str) -> dict[str, Any]:
            # Call 1 succeeds
            gateway.call("get_order", {"order_id": order_id})
            # Call 2 blocked (caught)
            try:
                gateway.call("get_order", {"order_id": order_id})
            except ExecutionLimitExceeded:
                pass
            # Call 3 blocked (caught)
            try:
                gateway.call("get_order", {"order_id": order_id})
            except ExecutionLimitExceeded:
                pass
            return {"status": "completed", "final_response": "caught all limits"}

    report = run_attempt(task=task, order_id="ord_1042", agent=CatchingAgent())

    # Sticky exhaustion ensures status is still limit_exceeded
    assert report["execution_status"] == "limit_exceeded"
    assert report["agent_result"] == {
        "status": "completed",
        "final_response": "caught all limits",
    }
    assert report["metrics"]["tool_call_count"] == 3
    assert report["evaluations"]["outcome"]["status"] == "not_evaluated"


# --- 3. Cooperative Deadline Enforcement ---


def test_expired_deadline_before_call_prevents_execution(task_data: dict[str, Any]) -> None:
    task = deepcopy(task_data)
    task["evaluation_spec"]["limits"]["timeout_seconds"] = 10.0

    clock = FakeClock(current=100.0)

    class DelayedCallAgent:
        def run(self, gateway: Any, order_id: str, ticket_id: str) -> dict[str, Any]:
            clock.advance(15.0)  # Advance past 10s deadline before calling tool
            gateway.call("get_order", {"order_id": order_id})
            return {"status": "completed"}

    report = run_attempt(task=task, order_id="ord_1042", agent=DelayedCallAgent(), clock=clock)

    assert report["execution_status"] == "limit_exceeded"
    assert report["termination"]["limit_type"] == "deadline"
    assert report["termination"]["code"] == "DEADLINE_EXCEEDED"
    # Call was recorded as requested and exceeded, but handler never executed
    events = report["events"]
    assert len([e for e in events if e.get("event_type") == "tool_requested"]) == 1
    assert len([e for e in events if e.get("event_type") == "tool_finished"]) == 0
    assert len([e for e in events if e.get("event_type") == "tool_limit_exceeded"]) == 1


def test_deadline_equality_triggers_exhaustion(task_data: dict[str, Any]) -> None:
    task = deepcopy(task_data)
    task["evaluation_spec"]["limits"]["timeout_seconds"] = 10.0

    clock = FakeClock(current=100.0)

    class ExactDeadlineAgent:
        def run(self, gateway: Any, order_id: str, ticket_id: str) -> dict[str, Any]:
            clock.advance(10.0)  # Exactly at deadline
            gateway.call("get_order", {"order_id": order_id})
            return {"status": "completed"}

    report = run_attempt(task=task, order_id="ord_1042", agent=ExactDeadlineAgent(), clock=clock)
    assert report["execution_status"] == "limit_exceeded"
    assert report["termination"]["limit_type"] == "deadline"


def test_handler_advancing_clock_past_deadline_preserves_effect_and_terminates(
    task_data: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    task = deepcopy(task_data)
    task["evaluation_spec"]["limits"]["timeout_seconds"] = 5.0

    clock = FakeClock(current=100.0)

    # Wrap get_order to advance clock by 10s during execution
    orig_get_order = SupportEnvironment.get_order

    def slow_get_order(self: SupportEnvironment, order_id: str) -> dict[str, Any]:
        clock.advance(10.0)
        return orig_get_order(self, order_id)

    monkeypatch.setattr(SupportEnvironment, "get_order", slow_get_order)

    report = run_attempt(task=task, order_id="ord_1042", clock=clock)

    assert report["execution_status"] == "limit_exceeded"
    assert report["termination"]["limit_type"] == "deadline"

    events = report["events"]
    # Verify tool_finished was recorded with actual result and duration
    finished = [e for e in events if e.get("event_type") == "tool_finished"]
    assert len(finished) == 1
    assert finished[0]["payload"]["result"]["id"] == "ord_1042"

    # Then tool_limit_exceeded was recorded
    tool_limit = [e for e in events if e.get("event_type") == "tool_limit_exceeded"]
    assert len(tool_limit) == 1
    assert tool_limit[0]["payload"]["limit_type"] == "deadline"


def test_agent_advancing_clock_without_tools_caught_after_return(task_data: dict[str, Any]) -> None:
    task = deepcopy(task_data)
    task["evaluation_spec"]["limits"]["timeout_seconds"] = 5.0

    clock = FakeClock(current=100.0)

    class ClockAdvancingAgent:
        def run(self, gateway: Any, order_id: str, ticket_id: str) -> dict[str, Any]:
            clock.advance(10.0)  # No tools, but time passed
            return {"status": "completed", "final_response": "slept"}

    report = run_attempt(task=task, order_id="ord_1042", agent=ClockAdvancingAgent(), clock=clock)

    assert report["execution_status"] == "limit_exceeded"
    assert report["agent_result"] == {"status": "completed", "final_response": "slept"}
    assert report["termination"]["limit_type"] == "deadline"
    assert report["evaluations"]["outcome"]["status"] == "not_evaluated"


# --- 4. Unexpected Runtime Errors Remain "error" ---


def test_unexpected_runtime_error_retains_error_status(task_data: dict[str, Any]) -> None:
    task = deepcopy(task_data)

    class CrashingAgent:
        def run(self, gateway: Any, order_id: str, ticket_id: str) -> dict[str, Any]:
            raise RuntimeError("Unexpected boom!")

    report = run_attempt(task=task, order_id="ord_1042", agent=CrashingAgent())

    assert report["execution_status"] == "error"
    assert report["error"]["code"] == "INTERNAL_ERROR"
    assert report["error"]["stage"] == "agent_execution"
    events = report["events"]
    assert any(e.get("event_type") == "attempt_failed" for e in events)


# --- 5. CLI Limit-Termination Compatibility ---


def test_run_demo_cli_exits_nonzero_on_limit_exceeded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    task = deepcopy(json.loads(SCENARIO_PATH.read_text(encoding="utf-8")))
    task["evaluation_spec"]["limits"]["max_tool_calls"] = 1  # will exceed
    task_file = tmp_path / "low_limit_task.json"
    task_file.write_text(json.dumps(task), encoding="utf-8")

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_demo.py",
            "--task",
            str(task_file),
            "--order-id",
            "ord_1042",
        ],
    )

    with pytest.raises(SystemExit) as exc:
        run_demo_cli_main()

    assert exc.value.code != 0
    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert report["execution_status"] == "limit_exceeded"
    assert report["termination"]["limit_type"] == "tool_calls"


def test_compare_agents_cli_exits_nonzero_on_limit_exceeded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    task = deepcopy(json.loads(SCENARIO_PATH.read_text(encoding="utf-8")))
    task["evaluation_spec"]["limits"]["max_tool_calls"] = 1  # will exceed
    task_file = tmp_path / "low_limit_task.json"
    task_file.write_text(json.dumps(task), encoding="utf-8")
    output_dir = tmp_path / "artifacts"

    monkeypatch.setattr(
        "sys.argv",
        [
            "compare_agents.py",
            "--task",
            str(task_file),
            "--order-id",
            "ord_1042",
            "--output-dir",
            str(output_dir),
        ],
    )

    with pytest.raises(SystemExit) as exc:
        compare_cli_main()

    assert exc.value.code != 0
    captured = capsys.readouterr()
    assert "=== Agent Comparison Summary ===" in captured.out
    assert "NOT_EVALUATED" in captured.out

    # Reports were saved despite limit termination
    saved_dirs = list(output_dir.iterdir())
    assert len(saved_dirs) == 1
    assert (saved_dirs[0] / "good_agent.json").exists()
    assert (saved_dirs[0] / "comparison.json").exists()

