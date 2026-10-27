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

    # Both events belong to the same tool call.
    assert all(
        event["tool_call_id"] == response["tool_call_id"]
        for event in events
    )

    # Reading an order must not change store state.
    assert environment.snapshot() == before


def test_invalid_arguments_are_rejected_before_execution(
    gateway_setup, monkeypatch
):
    gateway, recorder, environment = gateway_setup

    def unexpected_execution(*args, **kwargs):
        pytest.fail("Invalid arguments must not reach the environment.")

    monkeypatch.setattr(
        environment, "get_order", unexpected_execution
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
    gateway, recorder, environment = gateway_setup

    def broken_get_order(order_id):
        raise RuntimeError("private internal details")

    monkeypatch.setattr(
        environment, "get_order", broken_get_order
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