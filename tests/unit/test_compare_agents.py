"""Unit tests for SkipReturnInspectionAgent and the comparison CLI."""
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from app.agents.scripted import ScriptedRefundAgent, SkipReturnInspectionAgent
from app.environments.support.environment import SupportEnvironment
from app.execution.runner import run_attempt
from app.telemetry.recorder import TraceRecorder
from app.tools.gateway import ToolGateway
from scripts.compare_agents import compare_agents, main as cli_main

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASK_PATH = PROJECT_ROOT / "datasets/support/dev/refund_eligible_001.json"


@pytest.fixture
def scenario_task() -> dict[str, Any]:
    return json.loads(TASK_PATH.read_text(encoding="utf-8"))


def build_execution(task: dict[str, Any]) -> tuple[ToolGateway, TraceRecorder, SupportEnvironment]:
    environment = SupportEnvironment(
        fixture=task["environment_fixture"],
        authenticated_customer_id=task["agent_input"]["authenticated_customer_id"],
    )
    recorder = TraceRecorder(
        attempt_id="attempt_compare_test",
        environment_id=environment.environment_id,
    )
    gateway = ToolGateway(
        environment=environment,
        recorder=recorder,
    )
    return gateway, recorder, environment


def test_faulty_agent_five_tool_sequence_and_state(scenario_task: dict[str, Any]) -> None:
    gateway, recorder, environment = build_execution(scenario_task)

    agent = SkipReturnInspectionAgent()
    result = agent.run(
        gateway=gateway,
        order_id="ord_1042",
        ticket_id="ticket_001",
    )

    assert result == {
        "status": "completed",
        "final_response": "Confirmed refund of 249.99 USD.",
    }

    # Verify exactly five tools called, with no get_return_status
    requested_tools = [
        e["payload"]["tool_name"]
        for e in recorder.get_events()
        if e["event_type"] == "tool_requested"
    ]
    assert requested_tools == [
        "get_order",
        "request_refund_approval",
        "issue_refund",
        "get_refund_status",
        "update_ticket",
    ]
    assert "get_return_status" not in requested_tools

    # State checks: one refund of 24999 USD cents issued and ticket resolved
    state = environment.snapshot()
    order = next(o for o in state["orders"] if o["id"] == "ord_1042")
    assert order["refunded_amount_minor"] == 24999
    assert len(state["refunds"]) == 1
    assert state["refunds"][0]["amount_minor"] == 24999
    assert state["refunds"][0]["currency"] == "USD"
    ticket = next(t for t in state["tickets"] if t["id"] == "ticket_001")
    assert ticket["status"] == "resolved"


def test_faulty_agent_stops_on_reviewer_rejection(scenario_task: dict[str, Any]) -> None:
    scenario_task["environment_fixture"]["reviewer_behavior"]["decision"] = "reject"
    gateway, recorder, environment = build_execution(scenario_task)

    agent = SkipReturnInspectionAgent()
    result = agent.run(
        gateway=gateway,
        order_id="ord_1042",
        ticket_id="ticket_001",
    )

    assert result == {
        "status": "stopped",
        "final_response": "Refund approval was not obtained.",
    }

    requested_tools = [
        e["payload"]["tool_name"]
        for e in recorder.get_events()
        if e["event_type"] == "tool_requested"
    ]
    assert requested_tools == ["get_order", "request_refund_approval"]

    state = environment.snapshot()
    assert state["orders"][0]["refunded_amount_minor"] == 0
    assert len(state["refunds"]) == 0
    assert state["tickets"][0]["status"] == "open"


def test_faulty_agent_evaluation_outcome_passes_and_trajectory_fails(
    scenario_task: dict[str, Any],
) -> None:
    report = run_attempt(
        task=scenario_task,
        order_id="ord_1042",
        agent=SkipReturnInspectionAgent(),
    )

    assert report["execution_status"] == "completed"
    assert report["agent_result"]["status"] == "completed"

    evals = report["evaluations"]
    # Outcome passes because environment state has resolved refund
    assert evals["outcome"]["status"] == "evaluated"
    assert evals["outcome"]["passed"] is True

    # Trajectory fails strictly on inspect_return_before_refund
    assert evals["trajectory"]["status"] == "evaluated"
    assert evals["trajectory"]["passed"] is False
    assert evals["trajectory"]["failed_checks"] == ["inspect_return_before_refund"]

    # Metrics
    assert report["metrics"]["tool_call_count"] == 5
    assert report["metrics"]["token_usage"] is None
    assert report["metrics"]["estimated_cost_usd"] is None
    assert report["metrics"]["usage_status"] == "not_applicable"


def test_good_agent_retains_six_tool_sequence_and_passes_evaluations(
    scenario_task: dict[str, Any],
) -> None:
    report = run_attempt(
        task=scenario_task,
        order_id="ord_1042",
        agent=ScriptedRefundAgent(),
    )

    assert report["execution_status"] == "completed"
    assert report["evaluations"]["outcome"]["passed"] is True
    assert report["evaluations"]["trajectory"]["passed"] is True
    assert report["evaluations"]["trajectory"]["failed_checks"] == []
    assert report["metrics"]["tool_call_count"] == 6


def test_compare_agents_artifacts_and_consistency(
    tmp_path: Path,
) -> None:
    original_task_bytes = TASK_PATH.read_bytes()

    target_dir, comparison = compare_agents(
        task_path=TASK_PATH,
        order_id="ord_1042",
        output_dir=tmp_path,
    )

    assert target_dir.exists()
    assert target_dir.is_dir()

    # Task file remains unchanged
    assert TASK_PATH.read_bytes() == original_task_bytes

    good_file = target_dir / "good_agent.json"
    faulty_file = target_dir / "skip_return_inspection_agent.json"
    comp_file = target_dir / "comparison.json"

    assert good_file.exists()
    assert faulty_file.exists()
    assert comp_file.exists()

    good_report = json.loads(good_file.read_text(encoding="utf-8"))
    faulty_report = json.loads(faulty_file.read_text(encoding="utf-8"))
    saved_comparison = json.loads(comp_file.read_text(encoding="utf-8"))

    # Agreement with in-memory comparison
    assert saved_comparison == comparison
    assert saved_comparison["schema_version"] == "1.0.0"

    # Distinct attempt and environment IDs across runs
    assert good_report["attempt_id"] != faulty_report["attempt_id"]
    assert good_report["environment_id"] != faulty_report["environment_id"]
    assert good_report["attempt_id"] == saved_comparison["agents"]["scripted_refund_v1"]["attempt_id"]
    assert faulty_report["attempt_id"] == saved_comparison["agents"]["skip_return_inspection_v1"]["attempt_id"]

    # Saved traces contain actual tool sequence differences without synthetic trace alteration
    good_tools = [
        e["payload"]["tool_name"]
        for e in good_report["events"]
        if e["event_type"] == "tool_requested"
    ]
    faulty_tools = [
        e["payload"]["tool_name"]
        for e in faulty_report["events"]
        if e["event_type"] == "tool_requested"
    ]
    assert good_tools == [
        "get_order",
        "get_return_status",
        "request_refund_approval",
        "issue_refund",
        "get_refund_status",
        "update_ticket",
    ]
    assert faulty_tools == [
        "get_order",
        "request_refund_approval",
        "issue_refund",
        "get_refund_status",
        "update_ticket",
    ]

    # Evaluator agreement
    good_summary = saved_comparison["agents"]["scripted_refund_v1"]
    faulty_summary = saved_comparison["agents"]["skip_return_inspection_v1"]

    assert good_summary["evaluations"]["outcome"]["passed"] is True
    assert good_summary["evaluations"]["trajectory"]["passed"] is True
    assert good_summary["tool_call_count"] == 6

    assert faulty_summary["evaluations"]["outcome"]["passed"] is True
    assert faulty_summary["evaluations"]["trajectory"]["passed"] is False
    assert faulty_summary["evaluations"]["trajectory"]["failed_checks"] == ["inspect_return_before_refund"]
    assert faulty_summary["tool_call_count"] == 5


def test_repeated_comparisons_create_separate_output_directories(
    tmp_path: Path,
) -> None:
    dir1, comp1 = compare_agents(
        task_path=TASK_PATH,
        order_id="ord_1042",
        output_dir=tmp_path,
    )
    dir2, comp2 = compare_agents(
        task_path=TASK_PATH,
        order_id="ord_1042",
        output_dir=tmp_path,
    )

    assert dir1 != dir2
    assert dir1.exists()
    assert dir2.exists()
    assert comp1["comparison_id"] != comp2["comparison_id"]


def test_compare_agents_cli_zero_exit_on_faulty_agent_evaluation_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "compare_agents.py",
            "--task",
            str(TASK_PATH),
            "--order-id",
            "ord_1042",
            "--output-dir",
            str(tmp_path),
        ],
    )

    # Calling main must not exit with non-zero
    cli_main()

    captured = capsys.readouterr()
    assert "=== Agent Comparison Summary ===" in captured.out
    assert "scripted_refund_v1 (Good Agent)" in captured.out
    assert "skip_return_inspection_v1 (Faulty Agent)" in captured.out
    assert "inspect_return_before_refund" in captured.out


def test_compare_agents_cli_nonzero_exit_on_missing_task_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "compare_agents.py",
            "--task",
            "nonexistent_task.json",
            "--order-id",
            "ord_1042",
            "--output-dir",
            str(tmp_path),
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        cli_main()

    assert exc_info.value.code != 0
    captured = capsys.readouterr()
    assert "Task file not found" in captured.err

