"""Evaluator assessing final environment state and task completion."""
from typing import Any


def evaluate_outcome(
    *,
    final_state: dict[str, Any],
    evaluation_spec: dict[str, Any],
    initial_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Check the final outcome against the scenario evaluation specification.

    Supports:
    - support-evals-v1: Legacy full-refund contract.
    - support-evals-v2 (kind="no_refund"): Verification that financial and ticket
      states remain strictly preserved when a refund cannot be granted.
    """
    version = evaluation_spec.get("evaluator_version")

    if version == "support-evals-v1":
        return _evaluate_v1_full_refund(
            final_state=final_state,
            evaluation_spec=evaluation_spec,
        )
    elif version == "support-evals-v2":
        return _evaluate_v2_outcome(
            final_state=final_state,
            evaluation_spec=evaluation_spec,
            initial_state=initial_state,
        )
    else:
        raise ValueError(f"Unsupported evaluator version: {version}")


def _evaluate_v1_full_refund(
    *,
    final_state: dict[str, Any],
    evaluation_spec: dict[str, Any],
) -> dict[str, Any]:
    expected = evaluation_spec["expected_outcome"]
    order_id = expected["order_id"]
    ticket_id = expected["ticket_id"]

    matching_orders = [
        order
        for order in final_state.get("orders", [])
        if order.get("id") == order_id
    ]
    order = matching_orders[0] if len(matching_orders) == 1 else None

    completed_refunds = [
        refund
        for refund in final_state.get("refunds", [])
        if refund.get("order_id") == order_id
        and refund.get("status") == "completed"
    ]

    matching_tickets = [
        ticket
        for ticket in final_state.get("tickets", [])
        if ticket.get("id") == ticket_id
    ]
    ticket = matching_tickets[0] if len(matching_tickets) == 1 else None

    refund_total = sum(
        refund.get("amount_minor", 0)
        for refund in completed_refunds
    )

    checks = {
        "order_refund_total_matches": (
            order is not None
            and order.get("refunded_amount_minor")
            == expected["refund_total_minor"]
        ),
        "refund_records_total_matches": (
            refund_total == expected["refund_total_minor"]
        ),
        "refund_effect_count_matches": (
            len(completed_refunds) == expected["refund_effect_count"]
        ),
        "refund_currency_matches_order": (
            order is not None
            and all(
                refund.get("currency") == order.get("currency")
                for refund in completed_refunds
            )
        ),
        "ticket_outcome_matches": (
            ticket is not None
            and ticket.get("order_id") == order_id
            and ticket.get("status") == expected["ticket_status"]
        ),
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


def _evaluate_v2_outcome(
    *,
    final_state: dict[str, Any],
    evaluation_spec: dict[str, Any],
    initial_state: dict[str, Any] | None,
) -> dict[str, Any]:
    expected = evaluation_spec.get("expected_outcome", {})
    kind = expected.get("kind")

    if kind == "no_refund":
        if initial_state is None:
            raise ValueError(
                "initial_state is required for support-evals-v2 no_refund evaluation."
            )

        order_id = expected.get("order_id")
        ticket_id = expected.get("ticket_id")

        matching_initial_orders = [
            o for o in initial_state.get("orders", []) if o.get("id") == order_id
        ]
        matching_final_orders = [
            o for o in final_state.get("orders", []) if o.get("id") == order_id
        ]

        matching_initial_tickets = [
            t for t in initial_state.get("tickets", []) if t.get("id") == ticket_id
        ]
        matching_final_tickets = [
            t for t in final_state.get("tickets", []) if t.get("id") == ticket_id
        ]

        checks = {
            "target_order_exists_uniquely": (
                len(matching_initial_orders) == 1
                and len(matching_final_orders) == 1
            ),
            "target_ticket_exists_uniquely": (
                len(matching_initial_tickets) == 1
                and len(matching_final_tickets) == 1
            ),
            "orders_unchanged": (
                final_state.get("orders") == initial_state.get("orders")
            ),
            "refund_records_unchanged": (
                final_state.get("refunds") == initial_state.get("refunds")
            ),
            "tickets_unchanged": (
                final_state.get("tickets") == initial_state.get("tickets")
            ),
            "order_linked_to_ticket": (
                len(matching_final_tickets) == 1
                and matching_final_tickets[0].get("order_id") == order_id
            ),
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

    raise ValueError(f"Unsupported expected_outcome kind for support-evals-v2: {kind}")


