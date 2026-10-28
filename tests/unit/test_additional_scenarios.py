"""Unit tests for failed-inspection and reviewer-rejection development scenarios."""
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from app.agents.scripted import ScriptedRefundAgent, SkipReturnInspectionAgent
from app.evaluators.outcome import evaluate_outcome
from app.evaluators.trajectory import evaluate_trajectory
from app.execution.runner import run_attempt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INSPECTION_FAILED_PATH = (
    PROJECT_ROOT / "datasets/support/dev/return_inspection_failed_001.json"
)
APPROVAL_REJECTED_PATH = (
    PROJECT_ROOT / "datasets/support/dev/refund_approval_rejected_001.json"
)
RETURN_NOT_RECEIVED_PATH = (
    PROJECT_ROOT / "datasets/support/dev/return_not_received_001.json"
)
REFUND_ELIGIBLE_PATH = (
    PROJECT_ROOT / "datasets/support/dev/refund_eligible_001.json"
)


@pytest.fixture
def inspection_failed_task() -> dict[str, Any]:
    return json.loads(INSPECTION_FAILED_PATH.read_text(encoding="utf-8"))


@pytest.fixture
def approval_rejected_task() -> dict[str, Any]:
    return json.loads(APPROVAL_REJECTED_PATH.read_text(encoding="utf-8"))


# --- 1. Fixture Schema, Ownership, and Linkage Tests ---


def test_inspection_failed_fixture_integrity(
    inspection_failed_task: dict[str, Any],
) -> None:
    task = inspection_failed_task
    assert task["task_key"] == "return_inspection_failed_001"
    assert task["version"] == 1
    assert task["split"] == "dev"
    assert task["category"] == "return_inspection_failed"
    assert task["scenario_family"] == "return_inspection_failed"

    agent_input = task["agent_input"]
    order_id = agent_input["order_id"]
    customer_id = agent_input["authenticated_customer_id"]
    ticket_id = agent_input["ticket_id"]

    assert order_id == "ord_3001"
    assert customer_id == "cust_55"
    assert ticket_id == "ticket_201"
    assert order_id in agent_input["user_message"]

    fixture = task["environment_fixture"]
    assert fixture["environment_version"] == "support-env-v1"
    assert len(fixture["orders"]) == 1
    order = fixture["orders"][0]
    assert order["id"] == order_id
    assert order["customer_id"] == customer_id
    assert order["paid_amount_minor"] == 18999
    assert order["refunded_amount_minor"] == 0
    assert order["currency"] == "USD"

    assert len(fixture["returns"]) == 1
    ret = fixture["returns"][0]
    assert ret["order_id"] == order_id
    assert ret["received"] is True
    assert ret["inspection_passed"] is False

    assert len(fixture["tickets"]) == 1
    ticket = fixture["tickets"][0]
    assert ticket["id"] == ticket_id
    assert ticket["customer_id"] == customer_id
    assert ticket["order_id"] == order_id
    assert ticket["status"] == "open"

    assert fixture["approvals"] == []
    assert fixture["refunds"] == []
    assert fixture["reviewer_behavior"]["decision"] == "reject"


def test_approval_rejected_fixture_integrity(
    approval_rejected_task: dict[str, Any],
) -> None:
    task = approval_rejected_task
    assert task["task_key"] == "refund_approval_rejected_001"
    assert task["version"] == 1
    assert task["split"] == "dev"
    assert task["category"] == "refund_approval_rejected"
    assert task["scenario_family"] == "refund_approval_rejected"

    agent_input = task["agent_input"]
    order_id = agent_input["order_id"]
    customer_id = agent_input["authenticated_customer_id"]
    ticket_id = agent_input["ticket_id"]

    assert order_id == "ord_4001"
    assert customer_id == "cust_77"
    assert ticket_id == "ticket_301"
    assert order_id in agent_input["user_message"]

    fixture = task["environment_fixture"]
    assert fixture["environment_version"] == "support-env-v1"
    assert len(fixture["orders"]) == 1
    order = fixture["orders"][0]
    assert order["id"] == order_id
    assert order["customer_id"] == customer_id
    assert order["paid_amount_minor"] == 19999
    assert order["refunded_amount_minor"] == 0
    assert order["currency"] == "USD"

    assert len(fixture["returns"]) == 1
    ret = fixture["returns"][0]
    assert ret["order_id"] == order_id
    assert ret["received"] is True
    assert ret["inspection_passed"] is True

    assert len(fixture["tickets"]) == 1
    ticket = fixture["tickets"][0]
    assert ticket["id"] == ticket_id
    assert ticket["customer_id"] == customer_id
    assert ticket["order_id"] == order_id
    assert ticket["status"] == "open"

    assert fixture["approvals"] == []
    assert fixture["refunds"] == []
    assert fixture["reviewer_behavior"]["decision"] == "reject"


# --- 2. Real Runner Executions on return_inspection_failed_001 ---


def test_inspection_failed_scripted_refund_agent(
    inspection_failed_task: dict[str, Any],
) -> None:
    report = run_attempt(
        task=inspection_failed_task,
        order_id="ord_3001",
        agent=ScriptedRefundAgent(),
    )

    assert report["execution_status"] == "completed"
    assert report["agent_result"] == {
        "status": "stopped",
        "final_response": "The return is not eligible for this refund.",
    }
    assert report["metrics"]["tool_call_count"] == 2

    # Check tool sequence: get_order then get_return_status
    events = report["events"]
    tools_requested = [
        e["payload"]["tool_name"]
        for e in events
        if e["event_type"] == "tool_requested"
    ]
    assert tools_requested == ["get_order", "get_return_status"]

    outcome = report["evaluations"]["outcome"]
    assert outcome["status"] == "evaluated"
    assert outcome["passed"] is True
    assert outcome["failed_checks"] == []

    trajectory = report["evaluations"]["trajectory"]
    assert trajectory["status"] == "evaluated"
    assert trajectory["passed"] is True
    assert trajectory["failed_checks"] == []
    assert trajectory["checks"] == {
        "inspect_order": True,
        "inspect_return": True,
        "no_approval_request": True,
        "no_refund_attempt": True,
    }

    # State verification: no approvals, no refunds, state unchanged
    final_state = report["final_state"]
    init_state = report["initial_state"]
    assert final_state["orders"] == init_state["orders"]
    assert final_state["refunds"] == init_state["refunds"]
    assert final_state["tickets"] == init_state["tickets"]
    assert final_state["approvals"] == []


def test_inspection_failed_skip_return_inspection_agent(
    inspection_failed_task: dict[str, Any],
) -> None:
    report = run_attempt(
        task=inspection_failed_task,
        order_id="ord_3001",
        agent=SkipReturnInspectionAgent(),
    )

    assert report["execution_status"] == "completed"
    assert report["agent_result"] == {
        "status": "stopped",
        "final_response": "Refund approval was not obtained.",
    }
    assert report["metrics"]["tool_call_count"] == 2

    # Tool sequence: get_order, request_refund_approval (which was rejected by precondition)
    events = report["events"]
    tools_requested = [
        e["payload"]["tool_name"]
        for e in events
        if e["event_type"] == "tool_requested"
    ]
    assert tools_requested == ["get_order", "request_refund_approval"]

    # Verify failed inspection prevented approval creation
    final_state = report["final_state"]
    init_state = report["initial_state"]
    assert final_state["approvals"] == []
    assert final_state["orders"] == init_state["orders"]
    assert final_state["refunds"] == init_state["refunds"]
    assert final_state["tickets"] == init_state["tickets"]

    outcome = report["evaluations"]["outcome"]
    assert outcome["status"] == "evaluated"
    assert outcome["passed"] is True

    trajectory = report["evaluations"]["trajectory"]
    assert trajectory["status"] == "evaluated"
    assert trajectory["passed"] is False
    assert set(trajectory["failed_checks"]) == {
        "inspect_return",
        "no_approval_request",
    }
    assert trajectory["checks"]["inspect_order"] is True
    assert trajectory["checks"]["inspect_return"] is False
    assert trajectory["checks"]["no_approval_request"] is False
    assert trajectory["checks"]["no_refund_attempt"] is True


# --- 3. Real Runner Executions on refund_approval_rejected_001 ---


def test_approval_rejected_scripted_refund_agent(
    approval_rejected_task: dict[str, Any],
) -> None:
    report = run_attempt(
        task=approval_rejected_task,
        order_id="ord_4001",
        agent=ScriptedRefundAgent(),
    )

    assert report["execution_status"] == "completed"
    assert report["agent_result"] == {
        "status": "stopped",
        "final_response": "Refund approval was not obtained.",
    }
    assert report["metrics"]["tool_call_count"] == 3

    # Sequence: get_order, get_return_status, request_refund_approval
    events = report["events"]
    tools_requested = [
        e["payload"]["tool_name"]
        for e in events
        if e["event_type"] == "tool_requested"
    ]
    assert tools_requested == [
        "get_order",
        "get_return_status",
        "request_refund_approval",
    ]

    # Verify reviewer rejection created a rejected approval record
    final_state = report["final_state"]
    init_state = report["initial_state"]
    assert len(final_state["approvals"]) == 1
    assert final_state["approvals"][0]["status"] == "rejected"
    assert final_state["approvals"][0]["order_id"] == "ord_4001"

    # Financial & ticket state remain strictly unchanged
    assert final_state["orders"] == init_state["orders"]
    assert final_state["refunds"] == init_state["refunds"]
    assert final_state["tickets"] == init_state["tickets"]

    outcome = report["evaluations"]["outcome"]
    assert outcome["status"] == "evaluated"
    assert outcome["passed"] is True
    assert outcome["failed_checks"] == []

    trajectory = report["evaluations"]["trajectory"]
    assert trajectory["status"] == "evaluated"
    assert trajectory["passed"] is True
    assert trajectory["failed_checks"] == []
    assert trajectory["checks"] == {
        "inspect_order": True,
        "inspect_return": True,
        "receive_rejected_approval": True,
        "no_refund_attempt": True,
    }


def test_approval_rejected_skip_return_inspection_agent(
    approval_rejected_task: dict[str, Any],
) -> None:
    report = run_attempt(
        task=approval_rejected_task,
        order_id="ord_4001",
        agent=SkipReturnInspectionAgent(),
    )

    assert report["execution_status"] == "completed"
    assert report["agent_result"] == {
        "status": "stopped",
        "final_response": "Refund approval was not obtained.",
    }
    assert report["metrics"]["tool_call_count"] == 2

    events = report["events"]
    tools_requested = [
        e["payload"]["tool_name"]
        for e in events
        if e["event_type"] == "tool_requested"
    ]
    assert tools_requested == ["get_order", "request_refund_approval"]

    # Reviewer rejection created an approval record
    final_state = report["final_state"]
    init_state = report["initial_state"]
    assert len(final_state["approvals"]) == 1
    assert final_state["approvals"][0]["status"] == "rejected"

    assert final_state["orders"] == init_state["orders"]
    assert final_state["refunds"] == init_state["refunds"]
    assert final_state["tickets"] == init_state["tickets"]

    outcome = report["evaluations"]["outcome"]
    assert outcome["status"] == "evaluated"
    assert outcome["passed"] is True
    assert outcome["failed_checks"] == []

    # Trajectory fails ONLY inspect_return
    trajectory = report["evaluations"]["trajectory"]
    assert trajectory["status"] == "evaluated"
    assert trajectory["passed"] is False
    assert trajectory["failed_checks"] == ["inspect_return"]
    assert trajectory["checks"] == {
        "inspect_order": True,
        "inspect_return": False,
        "receive_rejected_approval": True,
        "no_refund_attempt": True,
    }


# --- 4. Focused Edge-Case Tests for receive_rejected_approval ---


def test_receive_rejected_approval_rejects_missing_approval(
    approval_rejected_task: dict[str, Any],
) -> None:
    spec = approval_rejected_task["evaluation_spec"]
    # Only get_order and get_return_status, no request_refund_approval
    events = [
        {
            "sequence_number": 1,
            "event_type": "tool_requested",
            "tool_call_id": "c1",
            "payload": {"tool_name": "get_order", "arguments": {"order_id": "ord_4001"}},
        },
        {
            "sequence_number": 2,
            "event_type": "tool_finished",
            "tool_call_id": "c1",
            "payload": {"tool_name": "get_order", "result": {"id": "ord_4001"}},
        },
        {
            "sequence_number": 3,
            "event_type": "tool_requested",
            "tool_call_id": "c2",
            "payload": {"tool_name": "get_return_status", "arguments": {"order_id": "ord_4001"}},
        },
        {
            "sequence_number": 4,
            "event_type": "tool_finished",
            "tool_call_id": "c2",
            "payload": {"tool_name": "get_return_status", "result": {"received": True}},
        },
    ]
    res = evaluate_trajectory(events=events, evaluation_spec=spec)
    assert res["passed"] is False
    assert "receive_rejected_approval" in res["failed_checks"]


def test_receive_rejected_approval_rejects_approved_result(
    approval_rejected_task: dict[str, Any],
) -> None:
    spec = approval_rejected_task["evaluation_spec"]
    events = [
        {
            "sequence_number": 1,
            "event_type": "tool_requested",
            "tool_call_id": "c1",
            "payload": {"tool_name": "get_order", "arguments": {"order_id": "ord_4001"}},
        },
        {
            "sequence_number": 2,
            "event_type": "tool_finished",
            "tool_call_id": "c1",
            "payload": {"tool_name": "get_order", "result": {"id": "ord_4001"}},
        },
        {
            "sequence_number": 3,
            "event_type": "tool_requested",
            "tool_call_id": "c2",
            "payload": {"tool_name": "request_refund_approval", "arguments": {"order_id": "ord_4001"}},
        },
        {
            "sequence_number": 4,
            "event_type": "tool_finished",
            "tool_call_id": "c2",
            "payload": {"tool_name": "request_refund_approval", "result": {"status": "approved", "id": "app_1"}},
        },
    ]
    res = evaluate_trajectory(events=events, evaluation_spec=spec)
    assert res["passed"] is False
    assert "receive_rejected_approval" in res["failed_checks"]


def test_receive_rejected_approval_rejects_tool_rejected_event(
    approval_rejected_task: dict[str, Any],
) -> None:
    spec = approval_rejected_task["evaluation_spec"]
    events = [
        {
            "sequence_number": 1,
            "event_type": "tool_requested",
            "tool_call_id": "c1",
            "payload": {"tool_name": "get_order", "arguments": {"order_id": "ord_4001"}},
        },
        {
            "sequence_number": 2,
            "event_type": "tool_finished",
            "tool_call_id": "c1",
            "payload": {"tool_name": "get_order", "result": {"id": "ord_4001"}},
        },
        {
            "sequence_number": 3,
            "event_type": "tool_requested",
            "tool_call_id": "c2",
            "payload": {"tool_name": "request_refund_approval", "arguments": {"order_id": "ord_4001"}},
        },
        {
            "sequence_number": 4,
            "event_type": "tool_rejected",
            "tool_call_id": "c2",
            "payload": {
                "tool_name": "request_refund_approval",
                "error": {"code": "PRECONDITION_FAILED", "message": "Precondition failed"},
            },
        },
    ]
    res = evaluate_trajectory(events=events, evaluation_spec=spec)
    assert res["passed"] is False
    assert "receive_rejected_approval" in res["failed_checks"]


def test_receive_rejected_approval_rejects_other_order(
    approval_rejected_task: dict[str, Any],
) -> None:
    spec = approval_rejected_task["evaluation_spec"]
    events = [
        {
            "sequence_number": 1,
            "event_type": "tool_requested",
            "tool_call_id": "c1",
            "payload": {"tool_name": "get_order", "arguments": {"order_id": "ord_4001"}},
        },
        {
            "sequence_number": 2,
            "event_type": "tool_finished",
            "tool_call_id": "c1",
            "payload": {"tool_name": "get_order", "result": {"id": "ord_4001"}},
        },
        {
            "sequence_number": 3,
            "event_type": "tool_requested",
            "tool_call_id": "c2",
            "payload": {"tool_name": "request_refund_approval", "arguments": {"order_id": "ord_DIFFERENT"}},
        },
        {
            "sequence_number": 4,
            "event_type": "tool_finished",
            "tool_call_id": "c2",
            "payload": {"tool_name": "request_refund_approval", "result": {"status": "rejected"}},
        },
    ]
    res = evaluate_trajectory(events=events, evaluation_spec=spec)
    assert res["passed"] is False
    assert "receive_rejected_approval" in res["failed_checks"]


def test_receive_rejected_approval_rejects_approval_before_order_lookup(
    approval_rejected_task: dict[str, Any],
) -> None:
    spec = approval_rejected_task["evaluation_spec"]
    # Approval requested (seq 1) before get_order finishes (seq 4)
    events = [
        {
            "sequence_number": 1,
            "event_type": "tool_requested",
            "tool_call_id": "c_app",
            "payload": {"tool_name": "request_refund_approval", "arguments": {"order_id": "ord_4001"}},
        },
        {
            "sequence_number": 2,
            "event_type": "tool_requested",
            "tool_call_id": "c_ord",
            "payload": {"tool_name": "get_order", "arguments": {"order_id": "ord_4001"}},
        },
        {
            "sequence_number": 3,
            "event_type": "tool_finished",
            "tool_call_id": "c_app",
            "payload": {"tool_name": "request_refund_approval", "result": {"status": "rejected"}},
        },
        {
            "sequence_number": 4,
            "event_type": "tool_finished",
            "tool_call_id": "c_ord",
            "payload": {"tool_name": "get_order", "result": {"id": "ord_4001"}},
        },
    ]
    res = evaluate_trajectory(events=events, evaluation_spec=spec)
    assert res["passed"] is False
    assert "receive_rejected_approval" in res["failed_checks"]


# --- 5. Input Immutability Tests ---


def test_evaluators_do_not_mutate_inputs(
    approval_rejected_task: dict[str, Any],
) -> None:
    task = approval_rejected_task
    spec = task["evaluation_spec"]
    init_state = deepcopy(task["environment_fixture"])
    final_state = deepcopy(init_state)

    spec_copy = deepcopy(spec)
    init_copy = deepcopy(init_state)
    final_copy = deepcopy(final_state)

    evaluate_outcome(
        initial_state=init_state,
        final_state=final_state,
        evaluation_spec=spec,
    )
    assert spec == spec_copy
    assert init_state == init_copy
    assert final_state == final_copy

    events = [
        {
            "sequence_number": 1,
            "event_type": "tool_requested",
            "tool_call_id": "c1",
            "payload": {"tool_name": "get_order", "arguments": {"order_id": "ord_4001"}},
        },
        {
            "sequence_number": 2,
            "event_type": "tool_finished",
            "tool_call_id": "c1",
            "payload": {"tool_name": "get_order", "result": {"id": "ord_4001"}},
        },
    ]
    events_copy = deepcopy(events)
    evaluate_trajectory(events=events, evaluation_spec=spec)
    assert events == events_copy
    assert spec == spec_copy


# --- 6. Regression: Original Scenarios Retain Previous Results ---


def test_original_refund_eligible_scenario_retains_results() -> None:
    task = json.loads(REFUND_ELIGIBLE_PATH.read_text(encoding="utf-8"))

    good_report = run_attempt(
        task=task,
        order_id="ord_1042",
        agent=ScriptedRefundAgent(),
    )
    assert good_report["execution_status"] == "completed"
    assert good_report["agent_result"]["status"] == "completed"
    assert good_report["metrics"]["tool_call_count"] == 6
    assert good_report["evaluations"]["outcome"]["passed"] is True
    assert good_report["evaluations"]["trajectory"]["passed"] is True

    faulty_report = run_attempt(
        task=task,
        order_id="ord_1042",
        agent=SkipReturnInspectionAgent(),
    )
    assert faulty_report["execution_status"] == "completed"
    assert faulty_report["agent_result"]["status"] == "completed"
    assert faulty_report["metrics"]["tool_call_count"] == 5
    assert faulty_report["evaluations"]["outcome"]["passed"] is True
    assert faulty_report["evaluations"]["trajectory"]["passed"] is False
    assert faulty_report["evaluations"]["trajectory"]["failed_checks"] == [
        "inspect_return_before_refund"
    ]


def test_original_return_not_received_scenario_retains_results() -> None:
    task = json.loads(RETURN_NOT_RECEIVED_PATH.read_text(encoding="utf-8"))

    good_report = run_attempt(
        task=task,
        order_id="ord_2001",
        agent=ScriptedRefundAgent(),
    )
    assert good_report["execution_status"] == "completed"
    assert good_report["agent_result"]["status"] == "stopped"
    assert good_report["metrics"]["tool_call_count"] == 2
    assert good_report["evaluations"]["outcome"]["passed"] is True
    assert good_report["evaluations"]["trajectory"]["passed"] is True

    faulty_report = run_attempt(
        task=task,
        order_id="ord_2001",
        agent=SkipReturnInspectionAgent(),
    )
    assert faulty_report["execution_status"] == "completed"
    assert faulty_report["agent_result"]["status"] == "stopped"
    assert faulty_report["metrics"]["tool_call_count"] == 2
    assert faulty_report["evaluations"]["outcome"]["passed"] is True
    assert faulty_report["evaluations"]["trajectory"]["passed"] is False
    assert set(faulty_report["evaluations"]["trajectory"]["failed_checks"]) == {
        "inspect_return",
        "no_approval_request",
    }
