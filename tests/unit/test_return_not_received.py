"""Tests for the return_not_received_001 scenario and support-evals-v2 contract."""
import json
from copy import deepcopy
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from app.agents.scripted import ScriptedRefundAgent, SkipReturnInspectionAgent
from app.environments.support.environment import SupportEnvironment
from app.evaluators.outcome import evaluate_outcome
from app.evaluators.trajectory import evaluate_trajectory
from app.execution.runner import run_attempt
from app.telemetry.recorder import TraceRecorder
from app.tools.gateway import ToolGateway

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCENARIO_PATH = PROJECT_ROOT / "datasets/support/dev/return_not_received_001.json"


@pytest.fixture
def scenario_task() -> dict[str, Any]:
    return json.loads(SCENARIO_PATH.read_text(encoding="utf-8"))


# --- 1. Scenario Schema & Integrity Tests ---


def test_scenario_file_exists_and_conforms_to_schema(scenario_task: dict[str, Any]) -> None:
    assert scenario_task["task_key"] == "return_not_received_001"
    assert scenario_task["version"] == 1
    assert scenario_task["category"] == "return_not_received"
    assert scenario_task["scenario_family"] == "return_not_received"
    assert scenario_task["split"] == "dev"

    agent_input = scenario_task["agent_input"]
    assert agent_input["order_id"] == "ord_2001"
    assert agent_input["authenticated_customer_id"] == "cust_42"
    assert agent_input["ticket_id"] == "ticket_101"
    assert "ord_2001" in agent_input["user_message"]

    fixture = scenario_task["environment_fixture"]
    assert fixture["environment_version"] == "support-env-v1"
    assert len(fixture["orders"]) == 1
    order = fixture["orders"][0]
    assert order["id"] == "ord_2001"
    assert order["customer_id"] == "cust_42"
    assert order["paid_amount_minor"] == 14999
    assert order["refunded_amount_minor"] == 0
    assert order["currency"] == "USD"

    assert len(fixture["returns"]) == 1
    ret = fixture["returns"][0]
    assert ret["order_id"] == "ord_2001"
    assert ret["received"] is False
    assert ret["inspection_passed"] is False

    assert len(fixture["tickets"]) == 1
    ticket = fixture["tickets"][0]
    assert ticket["id"] == "ticket_101"
    assert ticket["customer_id"] == "cust_42"
    assert ticket["order_id"] == "ord_2001"
    assert ticket["status"] == "open"

    assert fixture["reviewer_behavior"]["decision"] == "reject"

    eval_spec = scenario_task["evaluation_spec"]
    assert eval_spec["evaluator_version"] == "support-evals-v2"
    assert eval_spec["expected_outcome"] == {
        "kind": "no_refund",
        "order_id": "ord_2001",
        "ticket_id": "ticket_101",
    }
    assert eval_spec["required_behavior"] == [
        "inspect_order",
        "inspect_return",
        "no_approval_request",
        "no_refund_attempt",
    ]


# --- 2. Outcome Evaluator Tests for support-evals-v2 (kind="no_refund") ---


def test_outcome_evaluator_v2_passes_on_unmodified_state(scenario_task: dict[str, Any]) -> None:
    initial_state = deepcopy(scenario_task["environment_fixture"])
    final_state = deepcopy(initial_state)
    spec = deepcopy(scenario_task["evaluation_spec"])

    result = evaluate_outcome(
        initial_state=initial_state,
        final_state=final_state,
        evaluation_spec=spec,
    )

    assert result["passed"] is True
    assert result["failed_checks"] == []
    assert result["checks"] == {
        "target_order_exists_uniquely": True,
        "target_ticket_exists_uniquely": True,
        "orders_unchanged": True,
        "refund_records_unchanged": True,
        "tickets_unchanged": True,
        "order_linked_to_ticket": True,
    }


def test_outcome_evaluator_v2_preserves_input_immutability(scenario_task: dict[str, Any]) -> None:
    initial_state = deepcopy(scenario_task["environment_fixture"])
    final_state = deepcopy(initial_state)
    spec = deepcopy(scenario_task["evaluation_spec"])

    init_copy = deepcopy(initial_state)
    final_copy = deepcopy(final_state)
    spec_copy = deepcopy(spec)

    evaluate_outcome(
        initial_state=initial_state,
        final_state=final_state,
        evaluation_spec=spec,
    )

    assert initial_state == init_copy
    assert final_state == final_copy
    assert spec == spec_copy


def test_outcome_evaluator_v2_requires_initial_state(scenario_task: dict[str, Any]) -> None:
    final_state = deepcopy(scenario_task["environment_fixture"])
    spec = deepcopy(scenario_task["evaluation_spec"])

    with pytest.raises(ValueError, match="initial_state is required"):
        evaluate_outcome(
            final_state=final_state,
            evaluation_spec=spec,
        )


def test_outcome_evaluator_v2_rejects_unsupported_kind(scenario_task: dict[str, Any]) -> None:
    state = deepcopy(scenario_task["environment_fixture"])
    spec = deepcopy(scenario_task["evaluation_spec"])
    spec["expected_outcome"]["kind"] = "unsupported_kind"

    with pytest.raises(ValueError, match="Unsupported expected_outcome kind"):
        evaluate_outcome(
            initial_state=state,
            final_state=state,
            evaluation_spec=spec,
        )


@pytest.mark.parametrize(
    "tampering,expected_failed_check",
    [
        ("order_refunded_changed", "orders_unchanged"),
        ("refund_record_added", "refund_records_unchanged"),
        ("ticket_status_resolved", "tickets_unchanged"),
        ("ticket_order_id_changed", "order_linked_to_ticket"),
        ("missing_target_order", "target_order_exists_uniquely"),
        ("duplicate_target_order", "target_order_exists_uniquely"),
        ("missing_target_ticket", "target_ticket_exists_uniquely"),
        ("duplicate_target_ticket", "target_ticket_exists_uniquely"),
    ],
)
def test_outcome_evaluator_v2_detects_violations(
    scenario_task: dict[str, Any], tampering: str, expected_failed_check: str
) -> None:
    initial_state = deepcopy(scenario_task["environment_fixture"])
    final_state = deepcopy(initial_state)
    spec = deepcopy(scenario_task["evaluation_spec"])

    if tampering == "order_refunded_changed":
        final_state["orders"][0]["refunded_amount_minor"] = 1000
    elif tampering == "refund_record_added":
        final_state["refunds"].append(
            {
                "id": "ref_bogus",
                "order_id": "ord_2001",
                "amount_minor": 14999,
                "currency": "USD",
                "status": "completed",
            }
        )
    elif tampering == "ticket_status_resolved":
        final_state["tickets"][0]["status"] = "resolved"
    elif tampering == "ticket_order_id_changed":
        final_state["tickets"][0]["order_id"] = "ord_other"
        # also update initial so tickets_unchanged passes, isolating order_linked_to_ticket
        initial_state["tickets"][0]["order_id"] = "ord_other"
    elif tampering == "missing_target_order":
        final_state["orders"].clear()
    elif tampering == "duplicate_target_order":
        final_state["orders"].append(deepcopy(final_state["orders"][0]))
    elif tampering == "missing_target_ticket":
        final_state["tickets"].clear()
    elif tampering == "duplicate_target_ticket":
        final_state["tickets"].append(deepcopy(final_state["tickets"][0]))

    result = evaluate_outcome(
        initial_state=initial_state,
        final_state=final_state,
        evaluation_spec=spec,
    )

    assert result["passed"] is False
    assert expected_failed_check in result["failed_checks"]


# --- 3. Trajectory Evaluator Tests for New Rules ---


def test_trajectory_evaluator_v2_passes_on_valid_trace(scenario_task: dict[str, Any]) -> None:
    spec = scenario_task["evaluation_spec"]
    events = [
        {
            "sequence_number": 1,
            "event_type": "tool_requested",
            "tool_call_id": "call_1",
            "payload": {"tool_name": "get_order", "arguments": {"order_id": "ord_2001"}},
        },
        {
            "sequence_number": 2,
            "event_type": "tool_finished",
            "tool_call_id": "call_1",
            "payload": {"tool_name": "get_order", "result": {"id": "ord_2001"}},
        },
        {
            "sequence_number": 3,
            "event_type": "tool_requested",
            "tool_call_id": "call_2",
            "payload": {"tool_name": "get_return_status", "arguments": {"order_id": "ord_2001"}},
        },
        {
            "sequence_number": 4,
            "event_type": "tool_finished",
            "tool_call_id": "call_2",
            "payload": {"tool_name": "get_return_status", "result": {"received": False}},
        },
    ]

    events_copy = deepcopy(events)
    spec_copy = deepcopy(spec)

    result = evaluate_trajectory(events=events, evaluation_spec=spec)

    assert result["passed"] is True
    assert result["failed_checks"] == []
    assert result["checks"] == {
        "inspect_order": True,
        "inspect_return": True,
        "no_approval_request": True,
        "no_refund_attempt": True,
    }
    assert events == events_copy
    assert spec == spec_copy


def test_trajectory_evaluator_v2_detects_order_inspection_missing(scenario_task: dict[str, Any]) -> None:
    spec = scenario_task["evaluation_spec"]
    events = [
        {
            "sequence_number": 1,
            "event_type": "tool_requested",
            "tool_call_id": "call_2",
            "payload": {"tool_name": "get_return_status", "arguments": {"order_id": "ord_2001"}},
        },
        {
            "sequence_number": 2,
            "event_type": "tool_finished",
            "tool_call_id": "call_2",
            "payload": {"tool_name": "get_return_status", "result": {"received": False}},
        },
    ]

    result = evaluate_trajectory(events=events, evaluation_spec=spec)
    assert result["passed"] is False
    assert "inspect_order" in result["failed_checks"]
    assert "inspect_return" in result["failed_checks"]  # cannot inspect return before inspecting order


def test_trajectory_evaluator_v2_detects_return_inspection_before_order(scenario_task: dict[str, Any]) -> None:
    spec = scenario_task["evaluation_spec"]
    # Return inspected before get_order finishes
    events = [
        {
            "sequence_number": 1,
            "event_type": "tool_requested",
            "tool_call_id": "call_ret",
            "payload": {"tool_name": "get_return_status", "arguments": {"order_id": "ord_2001"}},
        },
        {
            "sequence_number": 2,
            "event_type": "tool_finished",
            "tool_call_id": "call_ret",
            "payload": {"tool_name": "get_return_status", "result": {"received": False}},
        },
        {
            "sequence_number": 3,
            "event_type": "tool_requested",
            "tool_call_id": "call_ord",
            "payload": {"tool_name": "get_order", "arguments": {"order_id": "ord_2001"}},
        },
        {
            "sequence_number": 4,
            "event_type": "tool_finished",
            "tool_call_id": "call_ord",
            "payload": {"tool_name": "get_order", "result": {"id": "ord_2001"}},
        },
    ]

    result = evaluate_trajectory(events=events, evaluation_spec=spec)
    assert result["passed"] is False
    assert result["checks"]["inspect_order"] is True
    assert result["checks"]["inspect_return"] is False
    assert "inspect_return" in result["failed_checks"]


def test_trajectory_evaluator_v2_detects_forbidden_approval_attempt(scenario_task: dict[str, Any]) -> None:
    spec = scenario_task["evaluation_spec"]
    events = [
        {
            "sequence_number": 1,
            "event_type": "tool_requested",
            "tool_call_id": "call_1",
            "payload": {"tool_name": "get_order", "arguments": {"order_id": "ord_2001"}},
        },
        {
            "sequence_number": 2,
            "event_type": "tool_finished",
            "tool_call_id": "call_1",
            "payload": {"tool_name": "get_order", "result": {"id": "ord_2001"}},
        },
        {
            "sequence_number": 3,
            "event_type": "tool_requested",
            "tool_call_id": "call_2",
            "payload": {"tool_name": "get_return_status", "arguments": {"order_id": "ord_2001"}},
        },
        {
            "sequence_number": 4,
            "event_type": "tool_finished",
            "tool_call_id": "call_2",
            "payload": {"tool_name": "get_return_status", "result": {"received": False}},
        },
        # Deliberate attempt even if rejected
        {
            "sequence_number": 5,
            "event_type": "tool_requested",
            "tool_call_id": "call_3",
            "payload": {"tool_name": "request_refund_approval", "arguments": {"order_id": "ord_2001"}},
        },
        {
            "sequence_number": 6,
            "event_type": "tool_rejected",
            "tool_call_id": "call_3",
            "payload": {"tool_name": "request_refund_approval", "error": {"code": "PRECONDITION_FAILED"}},
        },
    ]

    result = evaluate_trajectory(events=events, evaluation_spec=spec)
    assert result["passed"] is False
    assert "no_approval_request" in result["failed_checks"]


def test_trajectory_evaluator_v2_detects_forbidden_refund_attempt(scenario_task: dict[str, Any]) -> None:
    spec = scenario_task["evaluation_spec"]
    events = [
        {
            "sequence_number": 1,
            "event_type": "tool_requested",
            "tool_call_id": "call_1",
            "payload": {"tool_name": "get_order", "arguments": {"order_id": "ord_2001"}},
        },
        {
            "sequence_number": 2,
            "event_type": "tool_finished",
            "tool_call_id": "call_1",
            "payload": {"tool_name": "get_order", "result": {"id": "ord_2001"}},
        },
        {
            "sequence_number": 3,
            "event_type": "tool_requested",
            "tool_call_id": "call_2",
            "payload": {"tool_name": "issue_refund", "arguments": {"order_id": "ord_2001"}},
        },
        {
            "sequence_number": 4,
            "event_type": "tool_rejected",
            "tool_call_id": "call_2",
            "payload": {"tool_name": "issue_refund", "error": {"code": "PRECONDITION_FAILED"}},
        },
    ]

    result = evaluate_trajectory(events=events, evaluation_spec=spec)
    assert result["passed"] is False
    assert "no_refund_attempt" in result["failed_checks"]


# --- 4. End-to-End Runner Tests on return_not_received_001 ---


def test_scripted_refund_agent_passes_all_evaluations_on_refusal_scenario(
    scenario_task: dict[str, Any],
) -> None:
    report = run_attempt(
        task=scenario_task,
        order_id="ord_2001",
        agent=ScriptedRefundAgent(),
    )

    assert report["execution_status"] == "completed"
    assert report["agent_result"] == {
        "status": "stopped",
        "final_response": "The return is not eligible for this refund.",
    }
    assert report["metrics"]["tool_call_count"] == 2

    # Outcome evaluation
    outcome = report["evaluations"]["outcome"]
    assert outcome["status"] == "evaluated"
    assert outcome["passed"] is True
    assert outcome["failed_checks"] == []

    # Trajectory evaluation
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

    # Snapshot validation: financial and ticket states strictly preserved
    final_state = report["final_state"]
    init_state = report["initial_state"]
    assert final_state["orders"] == init_state["orders"]
    assert final_state["refunds"] == init_state["refunds"]
    assert final_state["tickets"] == init_state["tickets"]


def test_faulty_agent_passes_outcome_but_fails_trajectory_on_refusal_scenario(
    scenario_task: dict[str, Any],
) -> None:
    report = run_attempt(
        task=scenario_task,
        order_id="ord_2001",
        agent=SkipReturnInspectionAgent(),
    )

    assert report["execution_status"] == "completed"
    assert report["agent_result"] == {
        "status": "stopped",
        "final_response": "Refund approval was not obtained.",
    }
    assert report["metrics"]["tool_call_count"] == 2

    # Outcome passes because environment rejected approval, preserving state
    outcome = report["evaluations"]["outcome"]
    assert outcome["status"] == "evaluated"
    assert outcome["passed"] is True
    assert outcome["failed_checks"] == []

    # Trajectory FAILS because it skipped get_return_status and attempted approval
    trajectory = report["evaluations"]["trajectory"]
    assert trajectory["status"] == "evaluated"
    assert trajectory["passed"] is False
    assert set(trajectory["failed_checks"]) == {"inspect_return", "no_approval_request"}
    assert trajectory["checks"]["inspect_order"] is True
    assert trajectory["checks"]["inspect_return"] is False
    assert trajectory["checks"]["no_approval_request"] is False
    assert trajectory["checks"]["no_refund_attempt"] is True

    # State still preserved
    final_state = report["final_state"]
    init_state = report["initial_state"]
    assert final_state["orders"] == init_state["orders"]
    assert final_state["refunds"] == init_state["refunds"]
    assert final_state["tickets"] == init_state["tickets"]

