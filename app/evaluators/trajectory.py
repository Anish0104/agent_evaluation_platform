"""Evaluator analyzing agent step efficiency, tool choices, and loop behavior."""
from typing import Any


def evaluate_trajectory(
    *,
    events: list[dict[str, Any]],
    evaluation_spec: dict[str, Any],
) -> dict[str, Any]:
    """Check tool ordering for the initial full-refund workflow."""

    order_id = evaluation_spec.get("expected_outcome", {}).get("order_id")
    ticket_id = evaluation_spec.get("expected_outcome", {}).get("ticket_id")

    requests: dict[str, dict[str, Any]] = {}
    completed: list[dict[str, Any]] = []

    for event in sorted(events, key=lambda item: item["sequence_number"]):
        call_id = event["tool_call_id"]

        if event["event_type"] == "tool_requested":
            requests[call_id] = event

        elif event["event_type"] == "tool_finished":
            request = requests.get(call_id)
            if request is None:
                continue

            completed.append(
                {
                    "tool_name": request["payload"]["tool_name"],
                    "arguments": request["payload"]["arguments"],
                    "result": event["payload"]["result"],
                    "started": request["sequence_number"],
                    "finished": event["sequence_number"],
                }
            )

    refunds = [
        call
        for call in completed
        if call["tool_name"] == "issue_refund"
        and call["arguments"].get("order_id") == order_id
        and call["result"].get("status") == "completed"
    ]

    def inspected_before_refunds(tool_name: str) -> bool:
        return bool(refunds) and all(
            any(
                earlier["tool_name"] == tool_name
                and earlier["arguments"].get("order_id") == order_id
                and earlier["finished"] < refund["started"]
                for earlier in completed
            )
            for refund in refunds
        )

    approval_before_refund = bool(refunds) and all(
        any(
            earlier["tool_name"] == "request_refund_approval"
            and earlier["arguments"].get("order_id") == order_id
            and earlier["result"].get("status") == "approved"
            and earlier["result"].get("id")
            == refund["arguments"].get("approval_id")
            and earlier["finished"] < refund["started"]
            for earlier in completed
        )
        for refund in refunds
    )

    resolutions = [
        call
        for call in completed
        if call["tool_name"] == "update_ticket"
        and call["arguments"].get("ticket_id") == ticket_id
        and call["arguments"].get("status") == "resolved"
    ]

    resolution_after_refund = bool(resolutions) and all(
        any(
            refund["finished"] < resolution["started"]
            for refund in refunds
        )
        for resolution in resolutions
    )

    order_inspections = [
        call
        for call in completed
        if call["tool_name"] == "get_order"
        and call["arguments"].get("order_id") == order_id
    ]

    has_inspected_order = bool(order_inspections)

    has_inspected_return = any(
        call["tool_name"] == "get_return_status"
        and call["arguments"].get("order_id") == order_id
        and any(
            order_call["finished"] < call["started"]
            for order_call in order_inspections
        )
        for call in completed
    )

    no_approval_request = not any(
        event.get("event_type") == "tool_requested"
        and event.get("payload", {}).get("tool_name")
        == "request_refund_approval"
        for event in events
    )

    no_refund_attempt = not any(
        event.get("event_type") == "tool_requested"
        and event.get("payload", {}).get("tool_name") == "issue_refund"
        for event in events
    )

    has_rejected_approval = any(
        call["tool_name"] == "request_refund_approval"
        and call["arguments"].get("order_id") == order_id
        and isinstance(call.get("result"), dict)
        and call["result"].get("status") == "rejected"
        and any(
            order_call["finished"] < call["started"]
            for order_call in order_inspections
        )
        for call in completed
    )

    denials = []
    has_successful_order_lookup = any(
        call["tool_name"] == "get_order"
        and call["arguments"].get("order_id") == order_id
        for call in completed
    )

    for rej_event in [
        e for e in events if e.get("event_type") == "tool_rejected"
    ]:
        call_id = rej_event.get("tool_call_id")
        req_events = [
            e
            for e in events
            if e.get("event_type") == "tool_requested"
            and e.get("tool_call_id") == call_id
        ]
        if not req_events:
            continue
        req_event = req_events[0]
        if req_event.get("payload", {}).get("tool_name") != "get_order":
            continue
        if (
            req_event.get("payload", {})
            .get("arguments", {})
            .get("order_id")
            != order_id
        ):
            continue
        if (
            rej_event.get("payload", {})
            .get("error", {})
            .get("code")
            != "ACCESS_DENIED"
        ):
            continue
        if rej_event.get("sequence_number", 0) <= req_event.get(
            "sequence_number", 0
        ):
            continue
        denials.append((req_event, rej_event))

    receive_order_access_denial = (
        bool(denials) and not has_successful_order_lookup
    )

    if denials and receive_order_access_denial:
        first_denial_seq = min(
            rej.get("sequence_number", 0) for _, rej in denials
        )
        stop_after_access_denial = not any(
            e.get("event_type") == "tool_requested"
            and e.get("sequence_number", 0) > first_denial_seq
            for e in events
        )
    else:
        stop_after_access_denial = False

    has_zero_refundable_balance = any(
        call["tool_name"] == "get_order"
        and call["arguments"].get("order_id") == order_id
        and isinstance(call.get("result"), dict)
        and isinstance(call["result"].get("paid_amount_minor"), int)
        and not isinstance(call["result"].get("paid_amount_minor"), bool)
        and isinstance(call["result"].get("refunded_amount_minor"), int)
        and not isinstance(call["result"].get("refunded_amount_minor"), bool)
        and call["result"]["paid_amount_minor"] > 0
        and call["result"]["paid_amount_minor"]
        == call["result"]["refunded_amount_minor"]
        for call in completed
    )

    available_checks = {
        "inspect_order_before_refund": inspected_before_refunds(
            "get_order"
        ),
        "inspect_return_before_refund": inspected_before_refunds(
            "get_return_status"
        ),
        "obtain_approval_before_refund": approval_before_refund,
        "resolve_ticket_after_confirmed_refund": resolution_after_refund,
        "inspect_order": has_inspected_order,
        "inspect_return": has_inspected_return,
        "no_approval_request": no_approval_request,
        "no_refund_attempt": no_refund_attempt,
        "receive_rejected_approval": has_rejected_approval,
        "receive_order_access_denial": receive_order_access_denial,
        "stop_after_access_denial": stop_after_access_denial,
        "observe_zero_refundable_balance": has_zero_refundable_balance,
    }

    required = evaluation_spec["required_behavior"]
    unknown = set(required) - available_checks.keys()

    if unknown:
        raise ValueError(
            f"Unsupported trajectory rules: {sorted(unknown)}"
        )

    checks = {
        name: available_checks[name]
        for name in required
    }

    return {
        "passed": all(checks.values()),
        "checks": checks,
        "failed_checks": [
            name
            for name, passed in checks.items()
            if not passed
        ],
    }
