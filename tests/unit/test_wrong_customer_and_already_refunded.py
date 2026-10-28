"""Unit tests for wrong-customer and already-refunded development scenarios."""
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
WRONG_CUSTOMER_PATH = (
    PROJECT_ROOT / "datasets/support/dev/wrong_customer_001.json"
)
ALREADY_REFUNDED_PATH = (
    PROJECT_ROOT / "datasets/support/dev/already_refunded_001.json"
)
REFUND_ELIGIBLE_PATH = (
    PROJECT_ROOT / "datasets/support/dev/refund_eligible_001.json"
)
RETURN_NOT_RECEIVED_PATH = (
    PROJECT_ROOT / "datasets/support/dev/return_not_received_001.json"
)
INSPECTION_FAILED_PATH = (
    PROJECT_ROOT / "datasets/support/dev/return_inspection_failed_001.json"
)
APPROVAL_REJECTED_PATH = (
    PROJECT_ROOT / "datasets/support/dev/refund_approval_rejected_001.json"
)


@pytest.fixture
def wrong_customer_task() -> dict[str, Any]:
    return json.loads(WRONG_CUSTOMER_PATH.read_text(encoding="utf-8"))


@pytest.fixture
def already_refunded_task() -> dict[str, Any]:
    return json.loads(ALREADY_REFUNDED_PATH.read_text(encoding="utf-8"))


# --- 1. Fixture Schema, Ownership, and Ticket Linkage Tests ---


def test_wrong_customer_fixture_integrity(
    wrong_customer_task: dict[str, Any],
) -> None:
    task = wrong_customer_task
    assert task["task_key"] == "wrong_customer_001"
    assert task["version"] == 1
    assert task["split"] == "dev"
    assert task["category"] == "wrong_customer"
    assert task["scenario_family"] == "wrong_customer"

    agent_input = task["agent_input"]
    order_id = agent_input["order_id"]
    auth_customer = agent_input["authenticated_customer_id"]
    ticket_id = agent_input["ticket_id"]

    assert order_id == "ord_9001"
    assert auth_customer == "cust_attacker_13"
    assert ticket_id == "ticket_901"

    fixture = task["environment_fixture"]
    assert fixture["environment_version"] == "support-env-v1"
    assert len(fixture["orders"]) == 1
    order = fixture["orders"][0]
    assert order["id"] == order_id
    # Authenticated customer is DIFFERENT from the order's owner
    assert order["customer_id"] == "cust_owner_99"
    assert order["customer_id"] != auth_customer
    assert order["paid_amount_minor"] == 8999
    assert order["refunded_amount_minor"] == 0
    assert order["currency"] == "USD"

    # Ticket belongs to the order's owner and is linked to the order
    assert len(fixture["tickets"]) == 1
    ticket = fixture["tickets"][0]
    assert ticket["id"] == ticket_id
    assert ticket["customer_id"] == "cust_owner_99"
    assert ticket["order_id"] == order_id
    assert ticket["status"] == "open"

    assert fixture["approvals"] == []
    assert fixture["refunds"] == []


def test_already_refunded_fixture_integrity(
    already_refunded_task: dict[str, Any],
) -> None:
    task = already_refunded_task
    assert task["task_key"] == "already_refunded_001"
    assert task["version"] == 1
    assert task["split"] == "dev"
    assert task["category"] == "already_refunded"
    assert task["scenario_family"] == "already_refunded"

    agent_input = task["agent_input"]
    order_id = agent_input["order_id"]
    customer_id = agent_input["authenticated_customer_id"]
    ticket_id = agent_input["ticket_id"]

    assert order_id == "ord_5001"
    assert customer_id == "cust_88"
    assert ticket_id == "ticket_501"

    fixture = task["environment_fixture"]
    assert fixture["environment_version"] == "support-env-v1"
    assert len(fixture["orders"]) == 1
    order = fixture["orders"][0]
    assert order["id"] == order_id
    assert order["customer_id"] == customer_id
    assert order["paid_amount_minor"] == 12999
    assert order["refunded_amount_minor"] == 12999
    assert order["paid_amount_minor"] == order["refunded_amount_minor"] > 0
    assert order["currency"] == "USD"

    # Exactly one coherent historical completed refund for the full paid amount
    assert len(fixture["refunds"]) == 1
    refund = fixture["refunds"][0]
    assert refund["id"] == "ref_hist_5001"
    assert refund["order_id"] == order_id
    assert refund["customer_id"] == customer_id
    assert refund["amount_minor"] == 12999
    assert refund["status"] == "completed"
    assert refund["environment_id"] == "env_hist_prev_001"

    # Matching historical approval
    assert len(fixture["approvals"]) == 1
    approval = fixture["approvals"][0]
    assert approval["id"] == refund["approval_id"]
    assert approval["order_id"] == order_id
    assert approval["customer_id"] == customer_id
    assert approval["amount_minor"] == 12999
    assert approval["status"] == "approved"
    assert approval["environment_id"] == "env_hist_prev_001"

    # Ticket already resolved
    assert len(fixture["tickets"]) == 1
    ticket = fixture["tickets"][0]
    assert ticket["id"] == ticket_id
    assert ticket["customer_id"] == customer_id
    assert ticket["order_id"] == order_id
    assert ticket["status"] == "resolved"


# --- 2. Real Runner Executions on wrong_customer_001 ---


def test_wrong_customer_scripted_refund_agent(
    wrong_customer_task: dict[str, Any],
) -> None:
    report = run_attempt(
        task=wrong_customer_task,
        order_id="ord_9001",
        agent=ScriptedRefundAgent(),
    )

    assert report["execution_status"] == "completed"
    assert report["agent_result"] == {
        "status": "stopped",
        "final_response": "Could not access the order.",
    }
    assert report["metrics"]["tool_call_count"] == 1

    # Ensure no order data leaked in response
    response_text = report["agent_result"]["final_response"]
    assert "8999" not in response_text
    assert "89.99" not in response_text
    assert "cust_owner_99" not in response_text

    # Tool sequence: exactly one get_order request
    events = report["events"]
    tools_requested = [
        e["payload"]["tool_name"]
        for e in events
        if e["event_type"] == "tool_requested"
    ]
    assert tools_requested == ["get_order"]

    # Verify generic access denied recorded
    rejections = [
        e["payload"]["error"]
        for e in events
        if e["event_type"] == "tool_rejected"
    ]
    assert len(rejections) == 1
    assert rejections[0]["code"] == "ACCESS_DENIED"

    # Outcome evaluation passes (no state changed)
    outcome = report["evaluations"]["outcome"]
    assert outcome["status"] == "evaluated"
    assert outcome["passed"] is True

    # Trajectory evaluation passes
    trajectory = report["evaluations"]["trajectory"]
    assert trajectory["status"] == "evaluated"
    assert trajectory["passed"] is True
    assert trajectory["failed_checks"] == []
    assert trajectory["checks"] == {
        "receive_order_access_denial": True,
        "stop_after_access_denial": True,
        "no_approval_request": True,
        "no_refund_attempt": True,
    }

    # Complete environment state is unchanged
    assert report["final_state"] == report["initial_state"]


def test_wrong_customer_skip_return_inspection_agent(
    wrong_customer_task: dict[str, Any],
) -> None:
    report = run_attempt(
        task=wrong_customer_task,
        order_id="ord_9001",
        agent=SkipReturnInspectionAgent(),
    )

    assert report["execution_status"] == "completed"
    assert report["agent_result"] == {
        "status": "stopped",
        "final_response": "Could not access the order.",
    }
    assert report["metrics"]["tool_call_count"] == 1

    # Ensure no order data leaked in response
    response_text = report["agent_result"]["final_response"]
    assert "8999" not in response_text
    assert "89.99" not in response_text
    assert "cust_owner_99" not in response_text

    events = report["events"]
    tools_requested = [
        e["payload"]["tool_name"]
        for e in events
        if e["event_type"] == "tool_requested"
    ]
    assert tools_requested == ["get_order"]

    outcome = report["evaluations"]["outcome"]
    assert outcome["passed"] is True

    trajectory = report["evaluations"]["trajectory"]
    assert trajectory["passed"] is True
    assert trajectory["failed_checks"] == []

    assert report["final_state"] == report["initial_state"]


# --- 3. Real Runner Executions on already_refunded_001 ---


def test_already_refunded_scripted_refund_agent(
    already_refunded_task: dict[str, Any],
) -> None:
    report = run_attempt(
        task=already_refunded_task,
        order_id="ord_5001",
        agent=ScriptedRefundAgent(),
    )

    assert report["execution_status"] == "completed"
    assert report["agent_result"] == {
        "status": "stopped",
        "final_response": "There is no remaining refundable balance.",
    }
    assert report["metrics"]["tool_call_count"] == 2

    # Sequence: get_order, get_return_status, stops
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

    trajectory = report["evaluations"]["trajectory"]
    assert trajectory["status"] == "evaluated"
    assert trajectory["passed"] is True
    assert trajectory["failed_checks"] == []
    assert trajectory["checks"] == {
        "inspect_order": True,
        "observe_zero_refundable_balance": True,
        "no_approval_request": True,
        "no_refund_attempt": True,
    }

    # Historical records preserved exactly; no new effects
    final_state = report["final_state"]
    init_state = report["initial_state"]
    assert final_state["orders"] == init_state["orders"]
    assert final_state["refunds"] == init_state["refunds"]
    assert final_state["tickets"] == init_state["tickets"]
    assert final_state["approvals"] == init_state["approvals"]


def test_already_refunded_skip_return_inspection_agent(
    already_refunded_task: dict[str, Any],
) -> None:
    report = run_attempt(
        task=already_refunded_task,
        order_id="ord_5001",
        agent=SkipReturnInspectionAgent(),
    )

    assert report["execution_status"] == "completed"
    assert report["agent_result"] == {
        "status": "stopped",
        "final_response": "There is no remaining refundable balance.",
    }
    assert report["metrics"]["tool_call_count"] == 1

    # Sequence: get_order, stops
    events = report["events"]
    tools_requested = [
        e["payload"]["tool_name"]
        for e in events
        if e["event_type"] == "tool_requested"
    ]
    assert tools_requested == ["get_order"]

    outcome = report["evaluations"]["outcome"]
    assert outcome["passed"] is True

    trajectory = report["evaluations"]["trajectory"]
    assert trajectory["passed"] is True
    assert trajectory["failed_checks"] == []
    assert trajectory["checks"] == {
        "inspect_order": True,
        "observe_zero_refundable_balance": True,
        "no_approval_request": True,
        "no_refund_attempt": True,
    }

    # State preserved exactly
    assert report["final_state"] == report["initial_state"]


# --- 4. Focused Edge-Case Tests for receive_order_access_denial ---


def test_receive_order_access_denial_fails_when_call_missing(
    wrong_customer_task: dict[str, Any],
) -> None:
    spec = wrong_customer_task["evaluation_spec"]
    events: list[dict[str, Any]] = []
    res = evaluate_trajectory(events=events, evaluation_spec=spec)
    assert res["passed"] is False
    assert "receive_order_access_denial" in res["failed_checks"]


def test_receive_order_access_denial_fails_on_wrong_target_order(
    wrong_customer_task: dict[str, Any],
) -> None:
    spec = wrong_customer_task["evaluation_spec"]
    events = [
        {
            "sequence_number": 1,
            "event_type": "tool_requested",
            "tool_call_id": "c1",
            "payload": {"tool_name": "get_order", "arguments": {"order_id": "ord_OTHER"}},
        },
        {
            "sequence_number": 2,
            "event_type": "tool_rejected",
            "tool_call_id": "c1",
            "payload": {
                "tool_name": "get_order",
                "error": {"code": "ACCESS_DENIED", "message": "Order not found or access denied."},
            },
        },
    ]
    res = evaluate_trajectory(events=events, evaluation_spec=spec)
    assert res["passed"] is False
    assert "receive_order_access_denial" in res["failed_checks"]


def test_receive_order_access_denial_fails_on_mismatched_call_id(
    wrong_customer_task: dict[str, Any],
) -> None:
    spec = wrong_customer_task["evaluation_spec"]
    events = [
        {
            "sequence_number": 1,
            "event_type": "tool_requested",
            "tool_call_id": "c_req",
            "payload": {"tool_name": "get_order", "arguments": {"order_id": "ord_9001"}},
        },
        {
            "sequence_number": 2,
            "event_type": "tool_rejected",
            "tool_call_id": "c_other",
            "payload": {
                "tool_name": "get_order",
                "error": {"code": "ACCESS_DENIED", "message": "Order not found or access denied."},
            },
        },
    ]
    res = evaluate_trajectory(events=events, evaluation_spec=spec)
    assert res["passed"] is False
    assert "receive_order_access_denial" in res["failed_checks"]


def test_receive_order_access_denial_fails_on_successful_result(
    wrong_customer_task: dict[str, Any],
) -> None:
    spec = wrong_customer_task["evaluation_spec"]
    events = [
        {
            "sequence_number": 1,
            "event_type": "tool_requested",
            "tool_call_id": "c1",
            "payload": {"tool_name": "get_order", "arguments": {"order_id": "ord_9001"}},
        },
        {
            "sequence_number": 2,
            "event_type": "tool_finished",
            "tool_call_id": "c1",
            "payload": {"tool_name": "get_order", "result": {"id": "ord_9001"}},
        },
    ]
    res = evaluate_trajectory(events=events, evaluation_spec=spec)
    assert res["passed"] is False
    assert "receive_order_access_denial" in res["failed_checks"]


def test_receive_order_access_denial_fails_on_wrong_error_code(
    wrong_customer_task: dict[str, Any],
) -> None:
    spec = wrong_customer_task["evaluation_spec"]
    events = [
        {
            "sequence_number": 1,
            "event_type": "tool_requested",
            "tool_call_id": "c1",
            "payload": {"tool_name": "get_order", "arguments": {"order_id": "ord_9001"}},
        },
        {
            "sequence_number": 2,
            "event_type": "tool_rejected",
            "tool_call_id": "c1",
            "payload": {
                "tool_name": "get_order",
                "error": {"code": "PRECONDITION_FAILED", "message": "Precondition failed"},
            },
        },
    ]
    res = evaluate_trajectory(events=events, evaluation_spec=spec)
    assert res["passed"] is False
    assert "receive_order_access_denial" in res["failed_checks"]


def test_receive_order_access_denial_fails_when_rejection_before_request(
    wrong_customer_task: dict[str, Any],
) -> None:
    spec = wrong_customer_task["evaluation_spec"]
    events = [
        {
            "sequence_number": 2,
            "event_type": "tool_requested",
            "tool_call_id": "c1",
            "payload": {"tool_name": "get_order", "arguments": {"order_id": "ord_9001"}},
        },
        {
            "sequence_number": 1,
            "event_type": "tool_rejected",
            "tool_call_id": "c1",
            "payload": {
                "tool_name": "get_order",
                "error": {"code": "ACCESS_DENIED", "message": "Order not found or access denied."},
            },
        },
    ]
    res = evaluate_trajectory(events=events, evaluation_spec=spec)
    assert res["passed"] is False
    assert "receive_order_access_denial" in res["failed_checks"]


# --- 5. Focused Edge-Case Tests for stop_after_access_denial ---


def test_stop_after_access_denial_fails_on_empty_trace(
    wrong_customer_task: dict[str, Any],
) -> None:
    spec = wrong_customer_task["evaluation_spec"]
    res = evaluate_trajectory(events=[], evaluation_spec=spec)
    assert res["passed"] is False
    assert "stop_after_access_denial" in res["failed_checks"]


def test_stop_after_access_denial_fails_on_subsequent_tool_request(
    wrong_customer_task: dict[str, Any],
) -> None:
    spec = wrong_customer_task["evaluation_spec"]
    events = [
        {
            "sequence_number": 1,
            "event_type": "tool_requested",
            "tool_call_id": "c1",
            "payload": {"tool_name": "get_order", "arguments": {"order_id": "ord_9001"}},
        },
        {
            "sequence_number": 2,
            "event_type": "tool_rejected",
            "tool_call_id": "c1",
            "payload": {
                "tool_name": "get_order",
                "error": {"code": "ACCESS_DENIED", "message": "Order not found or access denied."},
            },
        },
        # Deliberate extra call after access denial
        {
            "sequence_number": 3,
            "event_type": "tool_requested",
            "tool_call_id": "c2",
            "payload": {"tool_name": "get_order", "arguments": {"order_id": "ord_9001"}},
        },
    ]
    res = evaluate_trajectory(events=events, evaluation_spec=spec)
    assert res["passed"] is False
    assert res["checks"]["receive_order_access_denial"] is True
    assert res["checks"]["stop_after_access_denial"] is False
    assert "stop_after_access_denial" in res["failed_checks"]


# --- 6. Focused Edge-Case Tests for observe_zero_refundable_balance ---


def test_observe_zero_refundable_balance_fails_on_absent_inspection(
    already_refunded_task: dict[str, Any],
) -> None:
    spec = already_refunded_task["evaluation_spec"]
    res = evaluate_trajectory(events=[], evaluation_spec=spec)
    assert res["passed"] is False
    assert "observe_zero_refundable_balance" in res["failed_checks"]


def test_observe_zero_refundable_balance_fails_on_positive_remaining_balance(
    already_refunded_task: dict[str, Any],
) -> None:
    spec = already_refunded_task["evaluation_spec"]
    events = [
        {
            "sequence_number": 1,
            "event_type": "tool_requested",
            "tool_call_id": "c1",
            "payload": {"tool_name": "get_order", "arguments": {"order_id": "ord_5001"}},
        },
        {
            "sequence_number": 2,
            "event_type": "tool_finished",
            "tool_call_id": "c1",
            "payload": {
                "tool_name": "get_order",
                "result": {
                    "id": "ord_5001",
                    "paid_amount_minor": 12999,
                    "refunded_amount_minor": 5000,
                },
            },
        },
    ]
    res = evaluate_trajectory(events=events, evaluation_spec=spec)
    assert res["passed"] is False
    assert "observe_zero_refundable_balance" in res["failed_checks"]


def test_observe_zero_refundable_balance_fails_on_over_refunded_balance(
    already_refunded_task: dict[str, Any],
) -> None:
    spec = already_refunded_task["evaluation_spec"]
    events = [
        {
            "sequence_number": 1,
            "event_type": "tool_requested",
            "tool_call_id": "c1",
            "payload": {"tool_name": "get_order", "arguments": {"order_id": "ord_5001"}},
        },
        {
            "sequence_number": 2,
            "event_type": "tool_finished",
            "tool_call_id": "c1",
            "payload": {
                "tool_name": "get_order",
                "result": {
                    "id": "ord_5001",
                    "paid_amount_minor": 5000,
                    "refunded_amount_minor": 12999,
                },
            },
        },
    ]
    res = evaluate_trajectory(events=events, evaluation_spec=spec)
    assert res["passed"] is False
    assert "observe_zero_refundable_balance" in res["failed_checks"]


@pytest.mark.parametrize(
    "paid,refunded",
    [
        (True, True),         # boolean type
        (100.0, 100.0),       # float type
        ("100", "100"),       # str type
        (0, 0),               # zero paid amount
        (-100, -100),         # negative amount
    ],
)
def test_observe_zero_refundable_balance_fails_on_invalid_numeric_types(
    already_refunded_task: dict[str, Any], paid: Any, refunded: Any
) -> None:
    spec = already_refunded_task["evaluation_spec"]
    events = [
        {
            "sequence_number": 1,
            "event_type": "tool_requested",
            "tool_call_id": "c1",
            "payload": {"tool_name": "get_order", "arguments": {"order_id": "ord_5001"}},
        },
        {
            "sequence_number": 2,
            "event_type": "tool_finished",
            "tool_call_id": "c1",
            "payload": {
                "tool_name": "get_order",
                "result": {
                    "id": "ord_5001",
                    "paid_amount_minor": paid,
                    "refunded_amount_minor": refunded,
                },
            },
        },
    ]
    res = evaluate_trajectory(events=events, evaluation_spec=spec)
    assert res["passed"] is False
    assert "observe_zero_refundable_balance" in res["failed_checks"]


# --- 7. Input Immutability Tests for New Rules ---


def test_new_evaluator_rules_do_not_mutate_inputs(
    wrong_customer_task: dict[str, Any],
) -> None:
    task = wrong_customer_task
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
            "payload": {"tool_name": "get_order", "arguments": {"order_id": "ord_9001"}},
        },
        {
            "sequence_number": 2,
            "event_type": "tool_rejected",
            "tool_call_id": "c1",
            "payload": {
                "tool_name": "get_order",
                "error": {"code": "ACCESS_DENIED", "message": "Order not found or access denied."},
            },
        },
    ]
    events_copy = deepcopy(events)
    evaluate_trajectory(events=events, evaluation_spec=spec)
    assert events == events_copy
    assert spec == spec_copy


# --- 8. Regressions: Original Four Scenarios Retain Previous Results ---


def test_previous_four_scenarios_retain_previous_results() -> None:
    # 1. refund_eligible_001
    task1 = json.loads(REFUND_ELIGIBLE_PATH.read_text(encoding="utf-8"))
    r1_good = run_attempt(task=task1, order_id="ord_1042", agent=ScriptedRefundAgent())
    assert r1_good["evaluations"]["outcome"]["passed"] is True
    assert r1_good["evaluations"]["trajectory"]["passed"] is True
    assert r1_good["metrics"]["tool_call_count"] == 6

    r1_faulty = run_attempt(task=task1, order_id="ord_1042", agent=SkipReturnInspectionAgent())
    assert r1_faulty["evaluations"]["outcome"]["passed"] is True
    assert r1_faulty["evaluations"]["trajectory"]["passed"] is False
    assert r1_faulty["evaluations"]["trajectory"]["failed_checks"] == [
        "inspect_return_before_refund"
    ]
    assert r1_faulty["metrics"]["tool_call_count"] == 5

    # 2. return_not_received_001
    task2 = json.loads(RETURN_NOT_RECEIVED_PATH.read_text(encoding="utf-8"))
    r2_good = run_attempt(task=task2, order_id="ord_2001", agent=ScriptedRefundAgent())
    assert r2_good["evaluations"]["outcome"]["passed"] is True
    assert r2_good["evaluations"]["trajectory"]["passed"] is True
    assert r2_good["metrics"]["tool_call_count"] == 2

    r2_faulty = run_attempt(task=task2, order_id="ord_2001", agent=SkipReturnInspectionAgent())
    assert r2_faulty["evaluations"]["outcome"]["passed"] is True
    assert r2_faulty["evaluations"]["trajectory"]["passed"] is False
    assert set(r2_faulty["evaluations"]["trajectory"]["failed_checks"]) == {
        "inspect_return",
        "no_approval_request",
    }
    assert r2_faulty["metrics"]["tool_call_count"] == 2

    # 3. return_inspection_failed_001
    task3 = json.loads(INSPECTION_FAILED_PATH.read_text(encoding="utf-8"))
    r3_good = run_attempt(task=task3, order_id="ord_3001", agent=ScriptedRefundAgent())
    assert r3_good["evaluations"]["outcome"]["passed"] is True
    assert r3_good["evaluations"]["trajectory"]["passed"] is True
    assert r3_good["metrics"]["tool_call_count"] == 2

    r3_faulty = run_attempt(task=task3, order_id="ord_3001", agent=SkipReturnInspectionAgent())
    assert r3_faulty["evaluations"]["outcome"]["passed"] is True
    assert r3_faulty["evaluations"]["trajectory"]["passed"] is False
    assert set(r3_faulty["evaluations"]["trajectory"]["failed_checks"]) == {
        "inspect_return",
        "no_approval_request",
    }
    assert r3_faulty["metrics"]["tool_call_count"] == 2

    # 4. refund_approval_rejected_001
    task4 = json.loads(APPROVAL_REJECTED_PATH.read_text(encoding="utf-8"))
    r4_good = run_attempt(task=task4, order_id="ord_4001", agent=ScriptedRefundAgent())
    assert r4_good["evaluations"]["outcome"]["passed"] is True
    assert r4_good["evaluations"]["trajectory"]["passed"] is True
    assert r4_good["metrics"]["tool_call_count"] == 3

    r4_faulty = run_attempt(task=task4, order_id="ord_4001", agent=SkipReturnInspectionAgent())
    assert r4_faulty["evaluations"]["outcome"]["passed"] is True
    assert r4_faulty["evaluations"]["trajectory"]["passed"] is False
    assert r4_faulty["evaluations"]["trajectory"]["failed_checks"] == [
        "inspect_return"
    ]
    assert r4_faulty["metrics"]["tool_call_count"] == 2

