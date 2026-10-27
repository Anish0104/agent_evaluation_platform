from datetime import datetime, timedelta
from uuid import UUID

from app.telemetry.recorder import TraceRecorder


def make_recorder():
    return TraceRecorder(
        attempt_id="attempt_test",
        environment_id="env_test",
    )


def test_event_contains_execution_context():
    recorder = make_recorder()

    recorder.record(
        event_type="tool_requested",
        tool_call_id="call_001",
        payload={"tool_name": "get_order"},
    )

    event = recorder.get_events()[0]

    assert event["attempt_id"] == "attempt_test"
    assert event["environment_id"] == "env_test"
    assert event["event_type"] == "tool_requested"
    assert event["tool_call_id"] == "call_001"
    assert event["payload"] == {"tool_name": "get_order"}

    # Parsing raises an error if the ID is not a valid UUID.
    assert UUID(event["event_id"]).version == 4

    timestamp = datetime.fromisoformat(event["timestamp"])
    assert timestamp.utcoffset() == timedelta(0)


def test_events_have_ordered_sequences_and_unique_ids():
    recorder = make_recorder()

    for event_type in ("attempt_started", "tool_requested", "tool_finished"):
        recorder.record(event_type=event_type, payload={})

    events = recorder.get_events()

    assert [event["sequence_number"] for event in events] == [1, 2, 3]
    assert [event["event_type"] for event in events] == [
        "attempt_started",
        "tool_requested",
        "tool_finished",
    ]
    assert len({event["event_id"] for event in events}) == 3


def test_original_payload_changes_do_not_rewrite_history():
    recorder = make_recorder()
    payload = {"arguments": {"order_id": "ord_1042"}}

    recorder.record(
        event_type="tool_requested",
        payload=payload,
    )

    payload["arguments"]["order_id"] = "changed_order"

    event = recorder.get_events()[0]
    assert event["payload"]["arguments"]["order_id"] == "ord_1042"


def test_returned_events_cannot_modify_history():
    recorder = make_recorder()
    recorder.record(
        event_type="tool_requested",
        payload={"arguments": {"order_id": "ord_1042"}},
    )

    events = recorder.get_events()
    events[0]["payload"]["arguments"]["order_id"] = "changed_order"
    events.clear()

    stored_events = recorder.get_events()

    assert len(stored_events) == 1
    assert stored_events[0]["payload"]["arguments"]["order_id"] == "ord_1042"


def test_recorders_have_independent_histories():
    first = make_recorder()
    second = make_recorder()

    first.record(event_type="attempt_started", payload={})

    assert len(first.get_events()) == 1
    assert second.get_events() == []

    second.record(event_type="attempt_started", payload={})

    assert second.get_events()[0]["sequence_number"] == 1


def test_non_tool_event_has_no_tool_call_id():
    recorder = make_recorder()

    recorder.record(event_type="attempt_started", payload={})

    assert recorder.get_events()[0]["tool_call_id"] is None