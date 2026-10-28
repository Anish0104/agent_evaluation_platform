import json
from pathlib import Path

import pytest

from app.environments.support.environment import SupportEnvironment
from app.telemetry.recorder import TraceRecorder
from app.tools.gateway import ToolGateway


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASK_PATH = (
    PROJECT_ROOT
    / "datasets/support/dev/refund_eligible_001.json"
)


@pytest.fixture
def gateway_setup():
    task = json.loads(TASK_PATH.read_text(encoding="utf-8"))

    environment = SupportEnvironment(
        fixture=task["environment_fixture"],
        authenticated_customer_id=(
            task["agent_input"]["authenticated_customer_id"]
        ),
    )

    recorder = TraceRecorder(
        attempt_id="attempt_test",
        environment_id=environment.environment_id,
    )

    gateway = ToolGateway(
        environment=environment,
        recorder=recorder,
    )

    return gateway, recorder, environment


def test_successful_lookup_records_request_and_result(gateway_setup):
    gateway, recorder, environment = gateway_setup
    before = environment.snapshot()

    response = gateway.get_order({"order_id": "ord_1042"})

    assert response["ok"] is True
    assert response["result"]["id"] == "ord_1042"

    events = recorder.get_events()

    assert [event["event_type"] for event in events] == [
        "tool_requested",
        "tool_finished",
    ]
    assert events[0]["payload"]["arguments"] == {
        "order_id": "ord_1042"
    }
    assert events[1]["payload"]["result"] == response["result"]

    assert all(
        event["tool_call_id"] == response["tool_call_id"]
        for event in events
    )

    assert environment.snapshot() == before


def test_invalid_arguments_are_rejected_before_execution(
    gateway_setup, monkeypatch
):
    _, recorder, environment = gateway_setup

    def unexpected_execution(*args, **kwargs):
        pytest.fail("Invalid arguments must not reach the environment.")

    monkeypatch.setattr(
        environment, "get_order", unexpected_execution
    )

    # Construct after patching so the registry captures the patched method.
    gateway = ToolGateway(
        environment=environment,
        recorder=recorder,
    )

    response = gateway.get_order({"order_id": 123})

    assert response["ok"] is False
    assert response["error"]["code"] == "INVALID_ARGUMENTS"

    events = recorder.get_events()

    assert [event["event_type"] for event in events] == [
        "tool_requested",
        "tool_rejected",
    ]
    assert events[0]["payload"]["arguments"] == {"order_id": 123}
    assert events[1]["payload"]["error"] == response["error"]
    assert all(
        event["tool_call_id"] == response["tool_call_id"]
        for event in events
    )


def test_missing_order_returns_safe_access_error(gateway_setup):
    gateway, recorder, environment = gateway_setup
    before = environment.snapshot()

    response = gateway.get_order({"order_id": "missing_order"})

    assert response["ok"] is False
    assert response["error"] == {
        "code": "ACCESS_DENIED",
        "message": "Order not found or access denied.",
    }

    events = recorder.get_events()

    assert [event["event_type"] for event in events] == [
        "tool_requested",
        "tool_rejected",
    ]
    assert events[1]["payload"]["error"] == response["error"]
    assert environment.snapshot() == before


def test_mismatched_recorder_is_rejected(gateway_setup):
    _, _, environment = gateway_setup

    wrong_recorder = TraceRecorder(
        attempt_id="attempt_other",
        environment_id="different_environment",
    )

    with pytest.raises(
        ValueError,
        match="Recorder and environment must match",
    ):
        ToolGateway(
            environment=environment,
            recorder=wrong_recorder,
        )


def test_unexpected_error_is_recorded_and_reraised(
    gateway_setup, monkeypatch
):
    _, recorder, environment = gateway_setup

    def broken_get_order(order_id):
        raise RuntimeError("private internal details")

    monkeypatch.setattr(
        environment, "get_order", broken_get_order
    )

    # Construct after patching so the registry captures the patched method.
    gateway = ToolGateway(
        environment=environment,
        recorder=recorder,
    )

    with pytest.raises(
        RuntimeError,
        match="private internal details",
    ):
        gateway.get_order({"order_id": "ord_1042"})

    events = recorder.get_events()

    assert [event["event_type"] for event in events] == [
        "tool_requested",
        "tool_failed",
    ]
    assert events[1]["tool_call_id"] == events[0]["tool_call_id"]
    assert events[1]["payload"]["error"] == {
        "code": "INTERNAL_ERROR",
        "message": "Unexpected tool execution failure.",
    }
    assert "private internal details" not in str(events)


@pytest.mark.parametrize(
    "tool_name",
    ["snapshot", "__init__", "delete_everything"],
)
def test_unavailable_tools_are_rejected(gateway_setup, tool_name):
    gateway, recorder, environment = gateway_setup
    before = environment.snapshot()

    response = gateway.call(tool_name, {})

    assert response["ok"] is False
    assert response["error"]["code"] == "UNKNOWN_TOOL"
    assert "result" not in response

    events = recorder.get_events()

    assert [event["event_type"] for event in events] == [
        "tool_requested",
        "tool_rejected",
    ]
    assert events[0]["payload"]["tool_name"] == tool_name
    assert events[1]["payload"]["error"] == response["error"]
    assert all(
        event["tool_call_id"] == response["tool_call_id"]
        for event in events
    )

    assert environment.snapshot() == before

def test_full_refund_workflow_and_safe_retry(gateway_setup):
    gateway, recorder, environment = gateway_setup

    # 1. Inspect the order.
    order_response = gateway.call(
        "get_order",
        {"order_id": "ord_1042"},
    )
    assert order_response["ok"] is True

    order = order_response["result"]
    amount = order["paid_amount_minor"] - order["refunded_amount_minor"]
    assert amount == 24999

    # 2. Verify the returned item was received and inspected.
    return_response = gateway.call(
        "get_return_status",
        {"order_id": "ord_1042"},
    )
    assert return_response["ok"] is True
    assert return_response["result"]["received"] is True
    assert return_response["result"]["inspection_passed"] is True

    # 3. Obtain approval.
    approval_response = gateway.call(
        "request_refund_approval",
        {
            "order_id": "ord_1042",
            "amount_minor": amount,
            "reason": "Returned item received and inspection passed.",
        },
    )
    assert approval_response["ok"] is True
    assert approval_response["result"]["status"] == "approved"

    # 4. Issue the approved refund.
    refund_arguments = {
        "order_id": "ord_1042",
        "amount_minor": amount,
        "approval_id": approval_response["result"]["id"],
    }

    refund_response = gateway.call("issue_refund", refund_arguments)
    assert refund_response["ok"] is True

    after_refund = environment.snapshot()

    # 5. Retry the same request. It must return the original refund.
    retry_response = gateway.call("issue_refund", refund_arguments)

    assert retry_response["ok"] is True
    assert retry_response["result"] == refund_response["result"]
    assert retry_response["tool_call_id"] != refund_response["tool_call_id"]
    assert environment.snapshot() == after_refund

    retry_events = [
        event
        for event in recorder.get_events()
        if event["tool_call_id"] == retry_response["tool_call_id"]
    ]
    assert [event["event_type"] for event in retry_events] == [
        "tool_requested",
        "tool_finished",
    ]

    # 6. Confirm the financial outcome.
    status_response = gateway.call(
        "get_refund_status",
        {"order_id": "ord_1042"},
    )
    assert status_response["ok"] is True
    assert status_response["result"]["refunded_amount_minor"] == 24999
    assert status_response["result"]["remaining_refundable_amount_minor"] == 0

    # 7. Resolve the ticket after confirming the refund.
    ticket_response = gateway.call(
        "update_ticket",
        {
            "ticket_id": "ticket_001",
            "status": "resolved",
            "note": "Confirmed refund of $249.99 USD.",
        },
    )
    assert ticket_response["ok"] is True
    assert ticket_response["result"]["status"] == "resolved"

    # Check actual state, independently of the tool responses.
    final_state = environment.snapshot()
    assert len(final_state["refunds"]) == 1

    final_order = next(
        order
        for order in final_state["orders"]
        if order["id"] == "ord_1042"
    )
    assert final_order["refunded_amount_minor"] == 24999

    final_ticket = next(
        ticket
        for ticket in final_state["tickets"]
        if ticket["id"] == "ticket_001"
    )
    assert final_ticket["status"] == "resolved"

    # Only approval, the first refund, and ticket update changed state.
    changes = [
        event
        for event in recorder.get_events()
        if event["event_type"] == "environment_changed"
    ]
    assert [event["payload"]["tool_name"] for event in changes] == [
        "request_refund_approval",
        "issue_refund",
        "update_ticket",
    ]

def test_refund_without_approval_is_rejected(gateway_setup):
    gateway, recorder, environment = gateway_setup
    before = environment.snapshot()

    response = gateway.call(
        "issue_refund",
        {
            "order_id": "ord_1042",
            "amount_minor": 24999,
            "approval_id": "nonexistent_approval",
        },
    )

    assert response["ok"] is False
    assert response["error"]["code"] == "PRECONDITION_FAILED"
    assert environment.snapshot() == before

    events = recorder.get_events()

    assert [event["event_type"] for event in events] == [
        "tool_requested",
        "tool_rejected",
    ]
    assert events[0]["payload"]["tool_name"] == "issue_refund"
    assert events[1]["payload"]["error"] == response["error"]
    assert all(
        event["tool_call_id"] == response["tool_call_id"]
        for event in events
    )