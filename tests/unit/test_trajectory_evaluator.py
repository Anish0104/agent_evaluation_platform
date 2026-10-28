import json
from copy import deepcopy
from pathlib import Path

import pytest

from app.agents.scripted import ScriptedRefundAgent
from app.environments.support.environment import SupportEnvironment
from app.evaluators.trajectory import evaluate_trajectory
from app.telemetry.recorder import TraceRecorder
from app.tools.gateway import ToolGateway

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASK_PATH = PROJECT_ROOT / "datasets/support/dev/refund_eligible_001.json"


@pytest.fixture
def execution_trace():
    """Run ScriptedRefundAgent on the dev scenario and return its trace and spec."""
    task = json.loads(TASK_PATH.read_text(encoding="utf-8"))

    environment = SupportEnvironment(
        fixture=task["environment_fixture"],
        authenticated_customer_id=(
            task["agent_input"]["authenticated_customer_id"]
        ),
    )
    recorder = TraceRecorder(
        attempt_id="attempt_trajectory_test",
        environment_id=environment.environment_id,
    )
    gateway = ToolGateway(
        environment=environment,
        recorder=recorder,
    )

    ScriptedRefundAgent().run(
        gateway=gateway,
        order_id="ord_1042",
        ticket_id="ticket_001",
    )

    return recorder.get_events(), deepcopy(task["evaluation_spec"])


def test_valid_trace_passes_all_checks(execution_trace):
    # A standard successful run must satisfy all required trajectory checks.
    events, spec = execution_trace

    result = evaluate_trajectory(events=events, evaluation_spec=spec)

    assert result["passed"] is True
    assert result["failed_checks"] == []
    assert len(result["checks"]) == 4
    assert all(result["checks"].values())


def test_missing_order_inspection_fails(execution_trace):
    # Removing get_order events must fail only inspect_order_before_refund.
    events, spec = execution_trace

    order_call_ids = {
        e["tool_call_id"]
        for e in events
        if e["event_type"] == "tool_requested"
        and e["payload"]["tool_name"] == "get_order"
    }
    assert len(order_call_ids) > 0

    altered_events = [
        e for e in events if e["tool_call_id"] not in order_call_ids
    ]
    result = evaluate_trajectory(events=altered_events, evaluation_spec=spec)

    assert result["passed"] is False
    assert result["failed_checks"] == ["inspect_order_before_refund"]


def test_missing_return_inspection_fails(execution_trace):
    # Removing get_return_status events must fail only inspect_return_before_refund.
    events, spec = execution_trace

    return_call_ids = {
        e["tool_call_id"]
        for e in events
        if e["event_type"] == "tool_requested"
        and e["payload"]["tool_name"] == "get_return_status"
    }
    assert len(return_call_ids) > 0

    altered_events = [
        e for e in events if e["tool_call_id"] not in return_call_ids
    ]
    result = evaluate_trajectory(events=altered_events, evaluation_spec=spec)

    assert result["passed"] is False
    assert result["failed_checks"] == ["inspect_return_before_refund"]


def test_mismatched_approval_fails(execution_trace):
    # Changing the approval_id on issue_refund must fail obtain_approval_before_refund.
    events, spec = execution_trace
    altered_events = deepcopy(events)

    found = False
    for event in altered_events:
        if (
            event["event_type"] == "tool_requested"
            and event["payload"]["tool_name"] == "issue_refund"
        ):
            event["payload"]["arguments"]["approval_id"] = "approval_nonexistent"
            found = True
    assert found

    result = evaluate_trajectory(events=altered_events, evaluation_spec=spec)

    assert result["passed"] is False
    assert result["failed_checks"] == ["obtain_approval_before_refund"]


def test_ticket_resolved_too_early_fails(execution_trace):
    # Resolving the ticket before issuing the refund must fail resolve_ticket_after_confirmed_refund.
    events, spec = execution_trace
    altered_events = deepcopy(events)

    update_ticket_call_id = next(
        e["tool_call_id"]
        for e in altered_events
        if e["event_type"] == "tool_requested"
        and e["payload"]["tool_name"] == "update_ticket"
    )
    issue_refund_call_id = next(
        e["tool_call_id"]
        for e in altered_events
        if e["event_type"] == "tool_requested"
        and e["payload"]["tool_name"] == "issue_refund"
    )

    ticket_events = [
        e for e in altered_events if e["tool_call_id"] == update_ticket_call_id
    ]
    non_ticket_events = [
        e for e in altered_events if e["tool_call_id"] != update_ticket_call_id
    ]

    refund_first_index = next(
        i
        for i, e in enumerate(non_ticket_events)
        if e["tool_call_id"] == issue_refund_call_id
    )

    # Place the update_ticket group immediately before the issue_refund group
    reordered_events = (
        non_ticket_events[:refund_first_index]
        + ticket_events
        + non_ticket_events[refund_first_index:]
    )

    # Renumber sequence numbers starting at 1 to reflect chronological shift
    for i, e in enumerate(reordered_events, start=1):
        e["sequence_number"] = i

    result = evaluate_trajectory(events=reordered_events, evaluation_spec=spec)

    assert result["passed"] is False
    assert result["failed_checks"] == ["resolve_ticket_after_confirmed_refund"]


def test_empty_trace_fails_all_checks(execution_trace):
    # An empty trace must fail all four checks rather than vacously passing with all([]).
    _, spec = execution_trace

    result = evaluate_trajectory(events=[], evaluation_spec=spec)

    assert result["passed"] is False
    assert result["failed_checks"] == [
        "inspect_order_before_refund",
        "inspect_return_before_refund",
        "obtain_approval_before_refund",
        "resolve_ticket_after_confirmed_refund",
    ]


def test_unsupported_required_rule_raises_value_error(execution_trace):
    # Unknown rule names in evaluation_spec must raise ValueError.
    events, spec = execution_trace
    altered_spec = deepcopy(spec)
    altered_spec["required_behavior"].append("unsupported_custom_rule")

    with pytest.raises(ValueError, match="Unsupported trajectory rules"):
        evaluate_trajectory(events=events, evaluation_spec=altered_spec)


def test_input_immutability(execution_trace):
    # evaluate_trajectory must leave its input events and spec untouched.
    events, spec = execution_trace
    original_events = deepcopy(events)
    original_spec = deepcopy(spec)

    evaluate_trajectory(events=events, evaluation_spec=spec)

    assert events == original_events
    assert spec == original_spec


def test_sequence_numbers_determine_chronology(execution_trace):
    # Reversing list order must not affect evaluation when sequence numbers are intact.
    events, spec = execution_trace
    reversed_events = list(reversed(deepcopy(events)))

    result = evaluate_trajectory(events=reversed_events, evaluation_spec=spec)

    assert result["passed"] is True
    assert result["failed_checks"] == []


def test_rejected_lookup_does_not_count_as_inspection(execution_trace):
    # A rejected lookup does not satisfy the requirement to inspect order details.
    events, spec = execution_trace
    altered_events = deepcopy(events)

    order_call_id = next(
        e["tool_call_id"]
        for e in altered_events
        if e["event_type"] == "tool_requested"
        and e["payload"]["tool_name"] == "get_order"
    )

    for i, event in enumerate(altered_events):
        if (
            event["tool_call_id"] == order_call_id
            and event["event_type"] == "tool_finished"
        ):
            altered_events[i] = {
                "event_id": event["event_id"],
                "attempt_id": event["attempt_id"],
                "environment_id": event["environment_id"],
                "sequence_number": event["sequence_number"],
                "event_type": "tool_rejected",
                "timestamp": event["timestamp"],
                "tool_call_id": event["tool_call_id"],
                "payload": {
                    "error": {
                        "code": "ACCESS_DENIED",
                        "message": "Order not found or access denied.",
                    }
                },
            }
            break

    result = evaluate_trajectory(events=altered_events, evaluation_spec=spec)

    assert result["passed"] is False
    assert result["failed_checks"] == ["inspect_order_before_refund"]

