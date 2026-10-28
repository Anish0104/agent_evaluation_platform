"""Unit tests for the local in-memory single-attempt runner."""
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from app.execution.runner import run_attempt
from app.tools.gateway import ToolGateway

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASK_PATH = PROJECT_ROOT / "datasets/support/dev/refund_eligible_001.json"


@pytest.fixture
def task_data() -> dict[str, Any]:
    return json.loads(TASK_PATH.read_text(encoding="utf-8"))


def test_eligible_scenario_completes_and_passes_evaluators(task_data: dict[str, Any]) -> None:
    report = run_attempt(task=task_data, order_id="ord_1042")

    assert report["execution_status"] == "completed"
    assert report["agent_result"] == {
        "status": "completed",
        "final_response": "Confirmed refund of 249.99 USD.",
    }

    evals = report["evaluations"]
    assert evals["outcome"]["status"] == "evaluated"
    assert evals["outcome"]["passed"] is True
    assert evals["outcome"]["failed_checks"] == []

    assert evals["trajectory"]["status"] == "evaluated"
    assert evals["trajectory"]["passed"] is True
    assert evals["trajectory"]["failed_checks"] == []

    assert evals["policy"]["status"] == "not_evaluated"
    assert evals["policy"]["reason"] == "Evaluator not implemented."

    assert evals["response"]["status"] == "not_evaluated"
    assert evals["response"]["reason"] == "Evaluator not implemented."

    assert report["error"] is None


def test_tool_call_count_tracks_requested_tools(task_data: dict[str, Any]) -> None:
    report = run_attempt(task=task_data, order_id="ord_1042")

    assert report["metrics"]["tool_call_count"] == 6
    requested_events = [
        e for e in report["events"] if e.get("event_type") == "tool_requested"
    ]
    assert len(requested_events) == 6


def test_independent_runs_maintain_isolated_state(task_data: dict[str, Any]) -> None:
    run1 = run_attempt(task=task_data, order_id="ord_1042")
    run2 = run_attempt(task=task_data, order_id="ord_1042")

    assert run1["attempt_id"] != run2["attempt_id"]
    assert run1["environment_id"] != run2["environment_id"]

    # Initial state of the second run is completely pristine
    assert run2["initial_state"]["orders"][0]["refunded_amount_minor"] == 0
    assert run2["initial_state"]["refunds"] == []


def test_runner_does_not_mutate_input_task(task_data: dict[str, Any]) -> None:
    task_copy = deepcopy(task_data)
    run_attempt(task=task_data, order_id="ord_1042")

    assert task_data == task_copy


def test_reviewer_rejection_completes_execution_with_evaluator_failures(
    task_data: dict[str, Any],
) -> None:
    task_copy = deepcopy(task_data)
    task_copy["environment_fixture"]["reviewer_behavior"]["decision"] = "reject"

    report = run_attempt(task=task_copy, order_id="ord_1042")

    # Execution itself completed normally without error
    assert report["execution_status"] == "completed"
    assert report["agent_result"] == {
        "status": "stopped",
        "final_response": "Refund approval was not obtained.",
    }

    # State reflects no refund
    assert report["final_state"]["orders"][0]["refunded_amount_minor"] == 0
    assert len(report["final_state"]["refunds"]) == 0

    # Business evaluators fail due to unachieved goal
    assert report["evaluations"]["outcome"]["passed"] is False
    assert report["evaluations"]["trajectory"]["passed"] is False


def test_injected_agent_raising_exception_produces_error_status(
    task_data: dict[str, Any],
) -> None:
    class FailingAgent:
        def run(self, *, gateway: Any, order_id: str, ticket_id: str) -> dict[str, Any]:
            raise RuntimeError("secret private database connection string")

    report = run_attempt(task=task_data, order_id="ord_1042", agent=FailingAgent())

    assert report["execution_status"] == "error"
    assert report["error"] == {
        "stage": "agent_execution",
        "code": "INTERNAL_ERROR",
        "message": "Unexpected error during agent_execution.",
    }

    # Evaluators should not have run
    assert report["evaluations"]["outcome"]["status"] == "not_evaluated"
    assert report["evaluations"]["trajectory"]["status"] == "not_evaluated"

    # Raw exception message must not leak anywhere in report or events
    report_json = json.dumps(report)
    assert "secret private database" not in report_json

    # Terminal event must be attempt_failed
    assert report["events"][-1]["event_type"] == "attempt_failed"


def test_evaluator_exception_is_classified_as_execution_error(
    task_data: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken_evaluate_outcome(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("private evaluator stack trace")

    monkeypatch.setattr(
        "app.execution.runner.evaluate_outcome",
        broken_evaluate_outcome,
    )

    report = run_attempt(task=task_data, order_id="ord_1042")

    assert report["execution_status"] == "error"
    assert report["error"] == {
        "stage": "evaluation",
        "code": "INTERNAL_ERROR",
        "message": "Unexpected error during evaluation.",
    }
    assert report["evaluations"]["outcome"]["status"] == "error"
    assert report["evaluations"]["outcome"]["reason"] == "Evaluator execution failed."

    # Must not leak private message
    report_json = json.dumps(report)
    assert "private evaluator stack trace" not in report_json

    assert report["events"][-1]["event_type"] == "attempt_failed"


def test_report_is_json_serializable(task_data: dict[str, Any]) -> None:
    report = run_attempt(task=task_data, order_id="ord_1042")

    serialized = json.dumps(report)
    parsed = json.loads(serialized)
    assert parsed == report


def test_report_includes_terminal_event(task_data: dict[str, Any]) -> None:
    # Successful run ends with attempt_finished
    success_report = run_attempt(task=task_data, order_id="ord_1042")
    assert success_report["events"][-1]["event_type"] == "attempt_finished"

    # Failed run ends with attempt_failed
    class BrokenAgent:
        def run(self, **kwargs: Any) -> Any:
            raise ValueError("boom")

    error_report = run_attempt(task=task_data, order_id="ord_1042", agent=BrokenAgent())
    assert error_report["events"][-1]["event_type"] == "attempt_failed"


def test_injected_agent_receives_only_intended_arguments(
    task_data: dict[str, Any],
) -> None:
    class SpyAgent:
        def __init__(self) -> None:
            self.captured_kwargs: dict[str, Any] = {}

        def run(self, **kwargs: Any) -> dict[str, Any]:
            self.captured_kwargs = kwargs
            return {"status": "completed", "final_response": "ok"}

    spy = SpyAgent()
    run_attempt(task=task_data, order_id="ord_1042", agent=spy)

    assert set(spy.captured_kwargs.keys()) == {"gateway", "order_id", "ticket_id"}
    assert spy.captured_kwargs["order_id"] == "ord_1042"
    assert spy.captured_kwargs["ticket_id"] == "ticket_001"
    assert isinstance(spy.captured_kwargs["gateway"], ToolGateway)


def test_cli_demo_succeeds_with_valid_arguments(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from scripts.run_demo import main

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_demo.py",
            "--task",
            str(TASK_PATH),
            "--order-id",
            "ord_1042",
        ],
    )

    main()

    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert report["execution_status"] == "completed"
    assert report["agent_result"]["status"] == "completed"
    assert report["metrics"]["tool_call_count"] == 6


def test_cli_demo_exits_nonzero_on_missing_task_file(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from scripts.run_demo import main

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_demo.py",
            "--task",
            "nonexistent_task.json",
            "--order-id",
            "ord_1042",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code != 0
    captured = capsys.readouterr()
    assert "Task file not found" in captured.err

