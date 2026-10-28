import json
from pathlib import Path

import pytest

from app.agents.scripted import ScriptedRefundAgent
from app.environments.support.environment import SupportEnvironment
from app.telemetry.recorder import TraceRecorder
from app.tools.gateway import ToolGateway
from app.evaluators.outcome import evaluate_outcome
from app.evaluators.trajectory import evaluate_trajectory

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASK_PATH = (
    PROJECT_ROOT
    / "datasets/support/dev/refund_eligible_001.json"
)


@pytest.fixture
def scenario_task():
    return json.loads(TASK_PATH.read_text(encoding="utf-8"))


def build_execution(task):
    environment = SupportEnvironment(
        fixture=task["environment_fixture"],
        authenticated_customer_id=(
            task["agent_input"]["authenticated_customer_id"]
        ),
    )
    recorder = TraceRecorder(
        attempt_id="attempt_scripted_test",
        environment_id=environment.environment_id,
    )
    gateway = ToolGateway(
        environment=environment,
        recorder=recorder,
    )
    return gateway, recorder, environment


def requested_tools(recorder):
    return [
        event["payload"]["tool_name"]
        for event in recorder.get_events()
        if event["event_type"] == "tool_requested"
    ]


def test_agent_completes_eligible_refund(scenario_task):
    gateway, recorder, environment = build_execution(scenario_task)

    result = ScriptedRefundAgent().run(
        gateway=gateway,
        order_id="ord_1042",
        ticket_id="ticket_001",
    )

    assert result == {
        "status": "completed",
        "final_response": "Confirmed refund of 249.99 USD.",
    }

    assert requested_tools(recorder) == [
        "get_order",
        "get_return_status",
        "request_refund_approval",
        "issue_refund",
        "get_refund_status",
        "update_ticket",
    ]

    state = environment.snapshot()

    order = next(
        item for item in state["orders"]
        if item["id"] == "ord_1042"
    )
    assert order["refunded_amount_minor"] == 24999

    assert len(state["refunds"]) == 1
    refund = state["refunds"][0]
    assert refund["order_id"] == "ord_1042"
    assert refund["amount_minor"] == 24999
    assert refund["currency"] == "USD"
    assert refund["status"] == "completed"

    ticket = next(
        item for item in state["tickets"]
        if item["id"] == "ticket_001"
    )
    assert ticket["status"] == "resolved"

    # The original scenario fixture remains unchanged.
    fixture = scenario_task["environment_fixture"]
    assert fixture["refunds"] == []
    assert fixture["orders"][0]["refunded_amount_minor"] == 0


def test_agent_stops_when_approval_is_rejected(scenario_task):
    # Modify this test's in-memory fixture, not the dataset file.
    scenario_task["environment_fixture"]["reviewer_behavior"][
        "decision"
    ] = "reject"

    gateway, recorder, environment = build_execution(scenario_task)
    before = environment.snapshot()

    result = ScriptedRefundAgent().run(
        gateway=gateway,
        order_id="ord_1042",
        ticket_id="ticket_001",
    )

    assert result == {
        "status": "stopped",
        "final_response": "Refund approval was not obtained.",
    }

    # The agent must stop before issuing a refund or resolving the ticket.
    assert requested_tools(recorder) == [
        "get_order",
        "get_return_status",
        "request_refund_approval",
    ]

    state = environment.snapshot()

    assert state["refunds"] == []
    assert state["orders"] == before["orders"]
    assert state["tickets"] == before["tickets"]
    assert len(state["approvals"]) == 1
    assert state["approvals"][0]["status"] == "rejected"

    # A reviewer rejection is a successful tool execution.
    approval_finished = [
        event
        for event in recorder.get_events()
        if event["event_type"] == "tool_finished"
        and event["payload"]["tool_name"] == "request_refund_approval"
    ]
    assert len(approval_finished) == 1
    assert approval_finished[0]["payload"]["result"]["status"] == "rejected"

def test_completed_agent_run_passes_outcome_evaluation(scenario_task):
    gateway, _, environment = build_execution(scenario_task)

    ScriptedRefundAgent().run(
        gateway=gateway,
        order_id="ord_1042",
        ticket_id="ticket_001",
    )

    evaluation = evaluate_outcome(
        final_state=environment.snapshot(),
        evaluation_spec=scenario_task["evaluation_spec"],
    )

    assert evaluation["passed"] is True
    assert evaluation["failed_checks"] == []
    assert all(evaluation["checks"].values())

def test_completed_agent_run_passes_trajectory_evaluation(scenario_task):
    gateway, recorder, _ = build_execution(scenario_task)

    ScriptedRefundAgent().run(
        gateway=gateway,
        order_id="ord_1042",
        ticket_id="ticket_001",
    )

    result = evaluate_trajectory(
        events=recorder.get_events(),
        evaluation_spec=scenario_task["evaluation_spec"],
    )

    assert result["passed"] is True
    assert result["failed_checks"] == []
    assert len(result["checks"]) == 4


def test_trajectory_fails_when_return_inspection_is_missing(scenario_task):
    gateway, recorder, environment = build_execution(scenario_task)

    ScriptedRefundAgent().run(
        gateway=gateway,
        order_id="ord_1042",
        ticket_id="ticket_001",
    )

    original_events = recorder.get_events()

    return_call_ids = {
        event["tool_call_id"]
        for event in original_events
        if event["event_type"] == "tool_requested"
        and event["payload"]["tool_name"] == "get_return_status"
    }
    assert len(return_call_ids) > 0

    altered_events = [
        event
        for event in original_events
        if event["tool_call_id"] not in return_call_ids
    ]

    trajectory_result = evaluate_trajectory(
        events=altered_events,
        evaluation_spec=scenario_task["evaluation_spec"],
    )

    assert trajectory_result["passed"] is False
    assert trajectory_result["failed_checks"] == ["inspect_return_before_refund"]

    outcome_result = evaluate_outcome(
        final_state=environment.snapshot(),
        evaluation_spec=scenario_task["evaluation_spec"],
    )
    assert outcome_result["passed"] is True

    assert recorder.get_events() == original_events


def test_trajectory_fails_when_return_inspection_is_missing(scenario_task):
    gateway, recorder, environment = build_execution(scenario_task)

    ScriptedRefundAgent().run(
        gateway=gateway,
        order_id="ord_1042",
        ticket_id="ticket_001",
    )

    original_events = recorder.get_events()

    return_call_ids = {
        event["tool_call_id"]
        for event in original_events
        if event["event_type"] == "tool_requested"
        and event["payload"]["tool_name"] == "get_return_status"
    }
    assert len(return_call_ids) > 0

    altered_events = [
        event
        for event in original_events
        if event["tool_call_id"] not in return_call_ids
    ]

    trajectory_result = evaluate_trajectory(
        events=altered_events,
        evaluation_spec=scenario_task["evaluation_spec"],
    )

    assert trajectory_result["passed"] is False
    assert trajectory_result["failed_checks"] == ["inspect_return_before_refund"]

    outcome_result = evaluate_outcome(
        final_state=environment.snapshot(),
        evaluation_spec=scenario_task["evaluation_spec"],
    )
    assert outcome_result["passed"] is True

    assert recorder.get_events() == original_events
