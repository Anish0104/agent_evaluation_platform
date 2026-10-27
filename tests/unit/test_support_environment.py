from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from app.environments.support.environment import SupportEnvironment


@pytest.fixture
def scenario_task() -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[2]
    scenario_path = (
        project_root / "datasets" / "support" / "dev" / "refund_eligible_001.json"
    )
    with scenario_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_support_environment(
    task: dict[str, Any],
    *,
    authenticated_customer_id: str | None = None,
) -> SupportEnvironment:
    return SupportEnvironment(
        fixture=task["environment_fixture"],
        authenticated_customer_id=(
            authenticated_customer_id
            if authenticated_customer_id is not None
            else task["agent_input"]["authenticated_customer_id"]
        ),
    )


# ==============================================================================
# Preserved Initial 7 Tests
# ==============================================================================


def test_unique_environment_ids(scenario_task: dict[str, Any]) -> None:
    env1 = build_support_environment(scenario_task)
    env2 = build_support_environment(scenario_task)

    assert env1.environment_id != env2.environment_id


def test_fixture_mutation_after_construction_does_not_affect_environment(
    scenario_task: dict[str, Any],
) -> None:
    env = build_support_environment(scenario_task)

    scenario_task["environment_fixture"]["orders"][0]["paid_amount_minor"] = 99999
    scenario_task["environment_fixture"]["orders"][0]["refunded_amount_minor"] = 12345

    order = env.get_order("ord_1042")
    assert order["paid_amount_minor"] == 24999
    assert order["refunded_amount_minor"] == 0


def test_get_order_mutation_does_not_affect_internal_state(
    scenario_task: dict[str, Any],
) -> None:
    env = build_support_environment(scenario_task)

    order = env.get_order("ord_1042")
    assert order["paid_amount_minor"] == 24999
    assert order["refunded_amount_minor"] == 0

    order["paid_amount_minor"] = 99999
    order["refunded_amount_minor"] = 50000

    fresh_order = env.get_order("ord_1042")
    assert fresh_order["paid_amount_minor"] == 24999
    assert fresh_order["refunded_amount_minor"] == 0


def test_snapshot_mutation_does_not_affect_internal_state(
    scenario_task: dict[str, Any],
) -> None:
    env = build_support_environment(scenario_task)

    state_snapshot = env.snapshot()
    assert state_snapshot["orders"][0]["paid_amount_minor"] == 24999
    assert state_snapshot["orders"][0]["refunded_amount_minor"] == 0

    state_snapshot["orders"][0]["paid_amount_minor"] = 99999
    state_snapshot["orders"][0]["refunded_amount_minor"] = 50000

    order = env.get_order("ord_1042")
    assert order["paid_amount_minor"] == 24999
    assert order["refunded_amount_minor"] == 0

    fresh_snapshot = env.snapshot()
    assert fresh_snapshot["orders"][0]["paid_amount_minor"] == 24999
    assert fresh_snapshot["orders"][0]["refunded_amount_minor"] == 0


def test_mutating_internal_state_does_not_affect_other_environment_or_fixture(
    scenario_task: dict[str, Any],
) -> None:
    env1 = build_support_environment(scenario_task)
    env2 = build_support_environment(scenario_task)

    env1._state["orders"][0]["paid_amount_minor"] = 99999
    env1._state["orders"][0]["refunded_amount_minor"] = 50000

    env2_order = env2.get_order("ord_1042")
    assert env2_order["paid_amount_minor"] == 24999
    assert env2_order["refunded_amount_minor"] == 0

    fixture_order = scenario_task["environment_fixture"]["orders"][0]
    assert fixture_order["paid_amount_minor"] == 24999
    assert fixture_order["refunded_amount_minor"] == 0


def test_different_authenticated_customer_cannot_access_order(
    scenario_task: dict[str, Any],
) -> None:
    env = build_support_environment(
        scenario_task,
        authenticated_customer_id="cust_different",
    )

    with pytest.raises(PermissionError) as exc_info:
        env.get_order("ord_1042")

    assert str(exc_info.value) == "Order not found or access denied."


def test_nonexistent_order_is_rejected(
    scenario_task: dict[str, Any],
) -> None:
    env = build_support_environment(scenario_task)

    with pytest.raises(PermissionError) as exc_info:
        env.get_order("ord_nonexistent")

    assert str(exc_info.value) == "Order not found or access denied."


# ==============================================================================
# Returns, Approvals, and Refunds Tests
# ==============================================================================


def test_authorized_return_lookup_and_deep_copy(
    scenario_task: dict[str, Any],
) -> None:
    env = build_support_environment(scenario_task)

    ret = env.get_return_status("ord_1042")
    assert ret == {
        "order_id": "ord_1042",
        "received": True,
        "inspection_passed": True,
    }

    # Mutating returned record must not affect internal state
    ret["received"] = False
    ret["inspection_passed"] = False

    fresh_ret = env.get_return_status("ord_1042")
    assert fresh_ret["received"] is True
    assert fresh_ret["inspection_passed"] is True
    assert env.snapshot()["returns"][0]["received"] is True


def test_missing_return_record_returns_default(
    scenario_task: dict[str, Any],
) -> None:
    scenario_task["environment_fixture"]["returns"] = []
    env = build_support_environment(scenario_task)

    ret = env.get_return_status("ord_1042")
    assert ret == {
        "order_id": "ord_1042",
        "received": False,
        "inspection_passed": False,
    }

    ret["received"] = True
    fresh_ret = env.get_return_status("ord_1042")
    assert fresh_ret["received"] is False


def test_cross_customer_and_nonexistent_return_lookup(
    scenario_task: dict[str, Any],
) -> None:
    env_diff_customer = build_support_environment(
        scenario_task,
        authenticated_customer_id="cust_other",
    )
    with pytest.raises(PermissionError) as exc_info:
        env_diff_customer.get_return_status("ord_1042")
    assert str(exc_info.value) == "Order not found or access denied."

    env = build_support_environment(scenario_task)
    with pytest.raises(PermissionError) as exc_info:
        env.get_return_status("ord_nonexistent")
    assert str(exc_info.value) == "Order not found or access denied."


def test_eligible_approval_without_refund_effect(
    scenario_task: dict[str, Any],
) -> None:
    env = build_support_environment(scenario_task)

    appr = env.request_refund_approval(
        order_id="ord_1042",
        amount_minor=24999,
        reason="Customer returned merchandise in original packaging.",
    )

    assert appr["status"] == "approved"
    assert appr["order_id"] == "ord_1042"
    assert appr["customer_id"] == "cust_12"
    assert appr["amount_minor"] == 24999
    assert appr["currency"] == "USD"
    assert appr["environment_id"] == env.environment_id
    assert "reviewer_behavior" not in appr

    # Check resulting state: no refund effect should have occurred
    snap = env.snapshot()
    assert snap["orders"][0]["refunded_amount_minor"] == 0
    assert len(snap["refunds"]) == 0
    assert len(snap["approvals"]) == 1
    assert snap["approvals"][0]["id"] == appr["id"]


def test_reviewer_rejection(
    scenario_task: dict[str, Any],
) -> None:
    env = build_support_environment(scenario_task)

    # Allowed amount is 24999 in fixture, so requesting 10000 results in rejection
    appr = env.request_refund_approval(
        order_id="ord_1042",
        amount_minor=10000,
        reason="Partial refund requested",
    )
    assert appr["status"] == "rejected"
    assert appr["amount_minor"] == 10000
    assert "reviewer_behavior" not in appr

    # Reviewer behavior explicitly rejecting
    scenario_task["environment_fixture"]["reviewer_behavior"] = {
        "decision": "reject",
    }
    env2 = build_support_environment(scenario_task)
    appr2 = env2.request_refund_approval(
        order_id="ord_1042",
        amount_minor=24999,
        reason="Full refund requested",
    )
    assert appr2["status"] == "rejected"

    # State checks: 0 refunded amount, 0 refunds created
    assert env.snapshot()["orders"][0]["refunded_amount_minor"] == 0
    assert len(env.snapshot()["refunds"]) == 0
    assert env2.snapshot()["orders"][0]["refunded_amount_minor"] == 0
    assert len(env2.snapshot()["refunds"]) == 0


@pytest.mark.parametrize(
    "return_variation",
    [
        {"received": False, "inspection_passed": True},
        {"received": True, "inspection_passed": False},
        {"received": False, "inspection_passed": False},
        None,  # No return record at all
    ],
)
def test_return_not_received_or_inspection_failed(
    scenario_task: dict[str, Any],
    return_variation: dict[str, Any] | None,
) -> None:
    if return_variation is None:
        scenario_task["environment_fixture"]["returns"] = []
    else:
        record = deepcopy(return_variation)
        record["order_id"] = "ord_1042"
        scenario_task["environment_fixture"]["returns"] = [record]

    env = build_support_environment(scenario_task)
    snap_before = env.snapshot()

    # Approval should be rejected with ValueError
    with pytest.raises(ValueError) as exc_info:
        env.request_refund_approval(
            order_id="ord_1042",
            amount_minor=24999,
            reason="Return not fully checked",
        )
    assert "return" in str(exc_info.value).lower()
    assert env.snapshot() == snap_before

    # Direct issue_refund with a pre-existing approved record should also be blocked
    approval_record = {
        "id": "appr_preexisting",
        "environment_id": env.environment_id,
        "customer_id": "cust_12",
        "order_id": "ord_1042",
        "amount_minor": 24999,
        "currency": "USD",
        "reason": "Pre-approved",
        "status": "approved",
    }
    env._state.setdefault("approvals", []).append(approval_record)
    snap_before_issue = env.snapshot()

    with pytest.raises(ValueError) as exc_info:
        env.issue_refund(
            order_id="ord_1042",
            amount_minor=24999,
            approval_id="appr_preexisting",
        )
    assert "return" in str(exc_info.value).lower()
    assert env.snapshot() == snap_before_issue


@pytest.mark.parametrize(
    "invalid_amount",
    [0, -100, 100.5, "24999", True, False],
)
def test_invalid_amounts_rejected(
    scenario_task: dict[str, Any],
    invalid_amount: Any,
) -> None:
    env = build_support_environment(scenario_task)
    snap_before = env.snapshot()

    # Test request_refund_approval
    with pytest.raises(ValueError):
        env.request_refund_approval(
            order_id="ord_1042",
            amount_minor=invalid_amount,
            reason="Testing invalid amount",
        )
    assert env.snapshot() == snap_before

    # Test issue_refund
    with pytest.raises(ValueError):
        env.issue_refund(
            order_id="ord_1042",
            amount_minor=invalid_amount,
            approval_id="some_approval_id",
        )
    assert env.snapshot() == snap_before


@pytest.mark.parametrize(
    "invalid_reason",
    ["", "   ", "\t\n  "],
)
def test_empty_or_whitespace_reason_rejected(
    scenario_task: dict[str, Any],
    invalid_reason: str,
) -> None:
    env = build_support_environment(scenario_task)
    snap_before = env.snapshot()

    with pytest.raises(ValueError):
        env.request_refund_approval(
            order_id="ord_1042",
            amount_minor=24999,
            reason=invalid_reason,
        )
    assert env.snapshot() == snap_before


def test_approval_amount_above_remaining_balance_rejected(
    scenario_task: dict[str, Any],
) -> None:
    env = build_support_environment(scenario_task)
    snap_before = env.snapshot()

    with pytest.raises(ValueError) as exc_info:
        env.request_refund_approval(
            order_id="ord_1042",
            amount_minor=25000,  # Paid is 24999
            reason="Exceeding paid balance",
        )
    assert "balance" in str(exc_info.value).lower()
    assert env.snapshot() == snap_before


def test_successful_refund_with_correct_effect(
    scenario_task: dict[str, Any],
) -> None:
    env = build_support_environment(scenario_task)

    appr = env.request_refund_approval(
        order_id="ord_1042",
        amount_minor=24999,
        reason="Customer return inspected",
    )
    ref = env.issue_refund(
        order_id="ord_1042",
        amount_minor=24999,
        approval_id=appr["id"],
    )

    assert ref["status"] == "completed"
    assert ref["amount_minor"] == 24999
    assert ref["currency"] == "USD"
    assert ref["order_id"] == "ord_1042"
    assert ref["customer_id"] == "cust_12"
    assert ref["approval_id"] == appr["id"]
    assert ref["environment_id"] == env.environment_id

    # Check resulting state
    snap = env.snapshot()
    assert snap["orders"][0]["refunded_amount_minor"] == 24999
    assert len(snap["refunds"]) == 1
    assert snap["refunds"][0] == ref


def test_missing_or_rejected_approval(
    scenario_task: dict[str, Any],
) -> None:
    env = build_support_environment(scenario_task)
    snap_before = env.snapshot()

    # Missing approval
    with pytest.raises(ValueError):
        env.issue_refund(
            order_id="ord_1042",
            amount_minor=24999,
            approval_id="nonexistent_approval",
        )
    assert env.snapshot() == snap_before

    # Rejected approval
    rejected_appr = env.request_refund_approval(
        order_id="ord_1042",
        amount_minor=10000,
        reason="Partial refund",
    )
    assert rejected_appr["status"] == "rejected"
    snap_before_rejected_issue = env.snapshot()

    with pytest.raises(ValueError):
        env.issue_refund(
            order_id="ord_1042",
            amount_minor=10000,
            approval_id=rejected_appr["id"],
        )
    assert env.snapshot() == snap_before_rejected_issue
    assert env.snapshot()["orders"][0]["refunded_amount_minor"] == 0
    assert len(env.snapshot()["refunds"]) == 0


def test_approval_used_for_different_amount_or_order(
    scenario_task: dict[str, Any],
) -> None:
    # Add a second order for the same customer
    scenario_task["environment_fixture"]["orders"].append(
        {
            "id": "ord_1043",
            "customer_id": "cust_12",
            "paid_amount_minor": 24999,
            "refunded_amount_minor": 0,
            "currency": "USD",
        }
    )
    scenario_task["environment_fixture"]["returns"].append(
        {
            "order_id": "ord_1043",
            "received": True,
            "inspection_passed": True,
        }
    )

    env = build_support_environment(scenario_task)
    appr = env.request_refund_approval(
        order_id="ord_1042",
        amount_minor=24999,
        reason="Approval for ord_1042",
    )

    # Different amount
    snap_before = env.snapshot()
    with pytest.raises(ValueError):
        env.issue_refund(
            order_id="ord_1042",
            amount_minor=10000,
            approval_id=appr["id"],
        )
    assert env.snapshot() == snap_before

    # Different order
    with pytest.raises(ValueError):
        env.issue_refund(
            order_id="ord_1043",
            amount_minor=24999,
            approval_id=appr["id"],
        )
    assert env.snapshot() == snap_before


def test_approval_from_another_environment_rejected(
    scenario_task: dict[str, Any],
) -> None:
    env1 = build_support_environment(scenario_task)
    env2 = build_support_environment(scenario_task)

    appr1 = env1.request_refund_approval(
        order_id="ord_1042",
        amount_minor=24999,
        reason="Approved in env1",
    )

    # Attempting to use env1's approval in env2
    snap2_before = env2.snapshot()
    with pytest.raises(ValueError):
        env2.issue_refund(
            order_id="ord_1042",
            amount_minor=24999,
            approval_id=appr1["id"],
        )
    assert env2.snapshot() == snap2_before

    # Even if an approval record with a foreign environment_id exists in env2._state
    foreign_approval = deepcopy(appr1)
    foreign_approval["id"] = "foreign_appr_id"
    foreign_approval["environment_id"] = "foreign_env_id"
    env2._state.setdefault("approvals", []).append(foreign_approval)

    snap2_with_foreign = env2.snapshot()
    with pytest.raises(ValueError):
        env2.issue_refund(
            order_id="ord_1042",
            amount_minor=24999,
            approval_id="foreign_appr_id",
        )
    assert env2.snapshot() == snap2_with_foreign
    assert env2.snapshot()["orders"][0]["refunded_amount_minor"] == 0
    assert len(env2.snapshot()["refunds"]) == 0


def test_identical_refund_retry_idempotency(
    scenario_task: dict[str, Any],
) -> None:
    env = build_support_environment(scenario_task)

    appr = env.request_refund_approval(
        order_id="ord_1042",
        amount_minor=24999,
        reason="Approved",
    )
    ref1 = env.issue_refund(
        order_id="ord_1042",
        amount_minor=24999,
        approval_id=appr["id"],
    )
    ref2 = env.issue_refund(
        order_id="ord_1042",
        amount_minor=24999,
        approval_id=appr["id"],
    )

    assert ref1["id"] == ref2["id"]
    assert ref1 == ref2

    # Verify exactly one financial effect occurred
    snap = env.snapshot()
    assert snap["orders"][0]["refunded_amount_minor"] == 24999
    assert len(snap["refunds"]) == 1


def test_changed_argument_retry_rejected(
    scenario_task: dict[str, Any],
) -> None:
    env = build_support_environment(scenario_task)

    appr = env.request_refund_approval(
        order_id="ord_1042",
        amount_minor=24999,
        reason="Approved",
    )
    env.issue_refund(
        order_id="ord_1042",
        amount_minor=24999,
        approval_id=appr["id"],
    )

    snap_after_refund = env.snapshot()

    # Retry with different amount
    with pytest.raises(ValueError):
        env.issue_refund(
            order_id="ord_1042",
            amount_minor=10000,
            approval_id=appr["id"],
        )

    assert env.snapshot() == snap_after_refund
    assert env.snapshot()["orders"][0]["refunded_amount_minor"] == 24999
    assert len(env.snapshot()["refunds"]) == 1


def test_two_approvals_cannot_over_refund(
    scenario_task: dict[str, Any],
) -> None:
    env = build_support_environment(scenario_task)

    # Manually configure two approvals that together exceed paid_amount_minor (24999)
    appr1 = {
        "id": "appr_part_1",
        "environment_id": env.environment_id,
        "customer_id": "cust_12",
        "order_id": "ord_1042",
        "amount_minor": 15000,
        "currency": "USD",
        "reason": "Part 1",
        "status": "approved",
    }
    appr2 = {
        "id": "appr_part_2",
        "environment_id": env.environment_id,
        "customer_id": "cust_12",
        "order_id": "ord_1042",
        "amount_minor": 15000,
        "currency": "USD",
        "reason": "Part 2",
        "status": "approved",
    }
    env._state.setdefault("approvals", []).extend([appr1, appr2])

    # First refund succeeds
    ref1 = env.issue_refund(
        order_id="ord_1042",
        amount_minor=15000,
        approval_id="appr_part_1",
    )
    assert ref1["amount_minor"] == 15000
    assert env.get_order("ord_1042")["refunded_amount_minor"] == 15000

    # Second refund exceeds remaining balance (24999 - 15000 = 9999 < 15000)
    snap_before_second = env.snapshot()
    with pytest.raises(ValueError) as exc_info:
        env.issue_refund(
            order_id="ord_1042",
            amount_minor=15000,
            approval_id="appr_part_2",
        )
    assert "balance" in str(exc_info.value).lower()

    # State remains bounded at 15000 with 1 refund record
    assert env.snapshot() == snap_before_second
    assert env.get_order("ord_1042")["refunded_amount_minor"] == 15000
    assert len(env.snapshot()["refunds"]) == 1


def test_failed_refund_calls_leave_snapshot_unchanged(
    scenario_task: dict[str, Any],
) -> None:
    env = build_support_environment(scenario_task)
    initial_snapshot = env.snapshot()

    # Unauthorized order
    with pytest.raises(PermissionError):
        env.issue_refund("ord_nonexistent", 24999, "some_appr")
    assert env.snapshot() == initial_snapshot

    # Missing approval
    with pytest.raises(ValueError):
        env.issue_refund("ord_1042", 24999, "missing_appr")
    assert env.snapshot() == initial_snapshot

    # Invalid amount
    with pytest.raises(ValueError):
        env.issue_refund("ord_1042", -50, "some_appr")
    assert env.snapshot() == initial_snapshot


def test_mutating_returned_approval_and_refund_records_cannot_change_state(
    scenario_task: dict[str, Any],
) -> None:
    env = build_support_environment(scenario_task)

    appr = env.request_refund_approval(
        order_id="ord_1042",
        amount_minor=24999,
        reason="Valid return",
    )
    appr["status"] = "tampered"
    appr["amount_minor"] = 999999

    # Internal state of approval is untouched
    stored_appr = env.snapshot()["approvals"][0]
    assert stored_appr["status"] == "approved"
    assert stored_appr["amount_minor"] == 24999

    ref = env.issue_refund(
        order_id="ord_1042",
        amount_minor=24999,
        approval_id=stored_appr["id"],
    )
    ref["status"] = "tampered"
    ref["amount_minor"] = 999999

    # Internal state of refund is untouched
    stored_ref = env.snapshot()["refunds"][0]
    assert stored_ref["status"] == "completed"
    assert stored_ref["amount_minor"] == 24999
    assert env.snapshot()["orders"][0]["refunded_amount_minor"] == 24999


def test_refund_execution_isolation_across_environments_and_fixture(
    scenario_task: dict[str, Any],
) -> None:
    env1 = build_support_environment(scenario_task)
    env2 = build_support_environment(scenario_task)

    appr = env1.request_refund_approval(
        order_id="ord_1042",
        amount_minor=24999,
        reason="Valid return",
    )
    env1.issue_refund(
        order_id="ord_1042",
        amount_minor=24999,
        approval_id=appr["id"],
    )

    # env1 reflects the completed refund
    assert env1.snapshot()["orders"][0]["refunded_amount_minor"] == 24999
    assert len(env1.snapshot()["refunds"]) == 1

    # env2 is completely unaffected
    assert env2.snapshot()["orders"][0]["refunded_amount_minor"] == 0
    assert len(env2.snapshot()["refunds"]) == 0
    assert len(env2.snapshot()["approvals"]) == 0

    # scenario_task original fixture is completely unaffected
    assert (
        scenario_task["environment_fixture"]["orders"][0]["refunded_amount_minor"]
        == 0
    )
    assert len(scenario_task["environment_fixture"]["refunds"]) == 0
    assert len(scenario_task["environment_fixture"]["approvals"]) == 0


# ==============================================================================
# Refund Status and Ticket Updates Tests
# ==============================================================================


def test_get_refund_status_initial_zero_refund(
    scenario_task: dict[str, Any],
) -> None:
    env = build_support_environment(scenario_task)

    status = env.get_refund_status("ord_1042")
    assert status == {
        "order_id": "ord_1042",
        "currency": "USD",
        "paid_amount_minor": 24999,
        "refunded_amount_minor": 0,
        "remaining_refundable_amount_minor": 24999,
        "refunds": [],
    }
    assert "approvals" not in status
    assert "reviewer_behavior" not in status


def test_get_refund_status_after_full_refund(
    scenario_task: dict[str, Any],
) -> None:
    env = build_support_environment(scenario_task)

    appr = env.request_refund_approval(
        order_id="ord_1042",
        amount_minor=24999,
        reason="Customer returned merchandise",
    )
    ref = env.issue_refund(
        order_id="ord_1042",
        amount_minor=24999,
        approval_id=appr["id"],
    )

    status = env.get_refund_status("ord_1042")
    assert status["paid_amount_minor"] == 24999
    assert status["refunded_amount_minor"] == 24999
    assert status["remaining_refundable_amount_minor"] == 0
    assert len(status["refunds"]) == 1
    assert status["refunds"][0] == ref


def test_get_refund_status_after_partial_refund(
    scenario_task: dict[str, Any],
) -> None:
    env = build_support_environment(scenario_task)

    # Pre-configure partial approval
    appr = {
        "id": "appr_part",
        "environment_id": env.environment_id,
        "customer_id": "cust_12",
        "order_id": "ord_1042",
        "amount_minor": 10000,
        "currency": "USD",
        "reason": "Partial concession",
        "status": "approved",
    }
    env._state.setdefault("approvals", []).append(appr)
    ref = env.issue_refund("ord_1042", 10000, "appr_part")

    status = env.get_refund_status("ord_1042")
    assert status["paid_amount_minor"] == 24999
    assert status["refunded_amount_minor"] == 10000
    assert status["remaining_refundable_amount_minor"] == 14999
    assert len(status["refunds"]) == 1
    assert status["refunds"][0] == ref


def test_get_refund_status_cross_customer_and_missing_order(
    scenario_task: dict[str, Any],
) -> None:
    env_diff = build_support_environment(
        scenario_task,
        authenticated_customer_id="cust_other",
    )
    with pytest.raises(PermissionError) as exc_info:
        env_diff.get_refund_status("ord_1042")
    assert str(exc_info.value) == "Order not found or access denied."

    env = build_support_environment(scenario_task)
    with pytest.raises(PermissionError) as exc_info:
        env.get_refund_status("ord_nonexistent")
    assert str(exc_info.value) == "Order not found or access denied."


def test_get_refund_status_excludes_other_orders(
    scenario_task: dict[str, Any],
) -> None:
    # Add second order owned by same customer
    scenario_task["environment_fixture"]["orders"].append(
        {
            "id": "ord_1043",
            "customer_id": "cust_12",
            "paid_amount_minor": 5000,
            "refunded_amount_minor": 5000,
            "currency": "USD",
        }
    )
    env = build_support_environment(scenario_task)

    # Refund for ord_1042
    appr_1042 = env.request_refund_approval("ord_1042", 24999, "Full refund")
    ref_1042 = env.issue_refund("ord_1042", 24999, appr_1042["id"])

    # Simulate existing refund for ord_1043
    ref_1043 = {
        "id": "ref_other_order",
        "environment_id": env.environment_id,
        "customer_id": "cust_12",
        "order_id": "ord_1043",
        "amount_minor": 5000,
        "currency": "USD",
        "approval_id": "appr_other",
        "status": "completed",
    }
    env._state.setdefault("refunds", []).append(ref_1043)

    status_1042 = env.get_refund_status("ord_1042")
    assert status_1042["order_id"] == "ord_1042"
    assert len(status_1042["refunds"]) == 1
    assert status_1042["refunds"][0]["id"] == ref_1042["id"]
    assert status_1042["refunded_amount_minor"] == 24999

    status_1043 = env.get_refund_status("ord_1043")
    assert status_1043["order_id"] == "ord_1043"
    assert len(status_1043["refunds"]) == 1
    assert status_1043["refunds"][0]["id"] == ref_1043["id"]


def test_get_refund_status_nested_mutation_isolation(
    scenario_task: dict[str, Any],
) -> None:
    env = build_support_environment(scenario_task)
    appr = env.request_refund_approval("ord_1042", 24999, "Full refund")
    env.issue_refund("ord_1042", 24999, appr["id"])

    status = env.get_refund_status("ord_1042")
    status["refunds"][0]["amount_minor"] = 999999
    status["refunds"].append({"fake": "injected"})

    fresh_status = env.get_refund_status("ord_1042")
    assert len(fresh_status["refunds"]) == 1
    assert fresh_status["refunds"][0]["amount_minor"] == 24999
    assert env.snapshot()["refunds"][0]["amount_minor"] == 24999


def test_update_ticket_authorized_open_and_pending(
    scenario_task: dict[str, Any],
) -> None:
    env = build_support_environment(scenario_task)

    res1 = env.update_ticket(
        ticket_id="ticket_001",
        status="pending",
        note="Waiting for customer response.",
    )
    assert res1["id"] == "ticket_001"
    assert res1["status"] == "pending"
    assert res1["notes"] == ["Waiting for customer response."]

    res2 = env.update_ticket(
        ticket_id="ticket_001",
        status="open",
        note="Customer provided return tracking number.",
    )
    assert res2["id"] == "ticket_001"
    assert res2["status"] == "open"
    assert res2["notes"] == [
        "Waiting for customer response.",
        "Customer provided return tracking number.",
    ]

    snap = env.snapshot()
    assert snap["tickets"][0]["status"] == "open"
    assert len(snap["tickets"][0]["notes"]) == 2


def test_update_ticket_missing_and_cross_customer(
    scenario_task: dict[str, Any],
) -> None:
    env = build_support_environment(scenario_task)

    # Missing ticket
    with pytest.raises(PermissionError) as exc_info:
        env.update_ticket("ticket_nonexistent", "open", "Note")
    assert str(exc_info.value) == "Ticket not found or access denied."

    # Cross-customer
    env_other = build_support_environment(
        scenario_task,
        authenticated_customer_id="cust_other",
    )
    with pytest.raises(PermissionError) as exc_info:
        env_other.update_ticket("ticket_001", "open", "Note")
    assert str(exc_info.value) == "Ticket not found or access denied."


def test_update_ticket_cross_customer_linked_order(
    scenario_task: dict[str, Any],
) -> None:
    # Ticket owned by cust_12, but linked order owned by cust_other
    scenario_task["environment_fixture"]["orders"].append(
        {
            "id": "ord_other_cust",
            "customer_id": "cust_other",
            "paid_amount_minor": 1000,
            "refunded_amount_minor": 0,
            "currency": "USD",
        }
    )
    scenario_task["environment_fixture"]["tickets"].append(
        {
            "id": "ticket_cross_order",
            "customer_id": "cust_12",
            "order_id": "ord_other_cust",
            "status": "open",
            "notes": [],
        }
    )

    env = build_support_environment(scenario_task)
    with pytest.raises(PermissionError) as exc_info:
        env.update_ticket("ticket_cross_order", "open", "Note")
    assert str(exc_info.value) == "Ticket not found or access denied."

    # Ticket with nonexistent linked order
    scenario_task["environment_fixture"]["tickets"].append(
        {
            "id": "ticket_missing_order",
            "customer_id": "cust_12",
            "order_id": "ord_does_not_exist",
            "status": "open",
            "notes": [],
        }
    )
    env2 = build_support_environment(scenario_task)
    with pytest.raises(PermissionError) as exc_info:
        env2.update_ticket("ticket_missing_order", "open", "Note")
    assert str(exc_info.value) == "Ticket not found or access denied."


@pytest.mark.parametrize(
    "bad_status",
    ["closed", "done", "", "   ", 123, None],
)
def test_update_ticket_invalid_status_rejected(
    scenario_task: dict[str, Any],
    bad_status: Any,
) -> None:
    env = build_support_environment(scenario_task)
    snap_before = env.snapshot()

    with pytest.raises(ValueError):
        env.update_ticket("ticket_001", bad_status, "Valid note")
    assert env.snapshot() == snap_before


@pytest.mark.parametrize(
    "bad_note",
    ["", "   ", "\t\n  ", 123, None],
)
def test_update_ticket_invalid_note_rejected(
    scenario_task: dict[str, Any],
    bad_note: Any,
) -> None:
    env = build_support_environment(scenario_task)
    snap_before = env.snapshot()

    with pytest.raises(ValueError):
        env.update_ticket("ticket_001", "open", bad_note)
    assert env.snapshot() == snap_before


@pytest.mark.parametrize(
    "bad_ticket_id",
    ["", "   ", "\t\n", 123, None],
)
def test_update_ticket_invalid_ticket_id_rejected(
    scenario_task: dict[str, Any],
    bad_ticket_id: Any,
) -> None:
    env = build_support_environment(scenario_task)
    snap_before = env.snapshot()

    with pytest.raises(ValueError):
        env.update_ticket(bad_ticket_id, "open", "Valid note")
    assert env.snapshot() == snap_before


def test_update_ticket_resolution_before_refund_rejected(
    scenario_task: dict[str, Any],
) -> None:
    env = build_support_environment(scenario_task)
    snap_before = env.snapshot()

    with pytest.raises(ValueError) as exc_info:
        env.update_ticket("ticket_001", "resolved", "Closing ticket prematurely")
    assert "refund" in str(exc_info.value).lower()
    assert env.snapshot() == snap_before


def test_update_ticket_resolution_after_approval_only_rejected(
    scenario_task: dict[str, Any],
) -> None:
    env = build_support_environment(scenario_task)
    env.request_refund_approval("ord_1042", 24999, "Approved only")
    snap_before = env.snapshot()

    with pytest.raises(ValueError) as exc_info:
        env.update_ticket(
            "ticket_001",
            "resolved",
            "Closing ticket after approval only",
        )
    assert "refund" in str(exc_info.value).lower()
    assert env.snapshot() == snap_before
    assert env.snapshot()["tickets"][0]["status"] == "open"


def test_update_ticket_resolution_after_partial_refund_rejected(
    scenario_task: dict[str, Any],
) -> None:
    env = build_support_environment(scenario_task)

    appr = {
        "id": "appr_partial",
        "environment_id": env.environment_id,
        "customer_id": "cust_12",
        "order_id": "ord_1042",
        "amount_minor": 10000,
        "currency": "USD",
        "reason": "Part",
        "status": "approved",
    }
    env._state.setdefault("approvals", []).append(appr)
    env.issue_refund("ord_1042", 10000, "appr_partial")

    snap_before = env.snapshot()
    with pytest.raises(ValueError) as exc_info:
        env.update_ticket(
            "ticket_001",
            "resolved",
            "Attempting resolution with partial refund",
        )
    assert "balance" in str(exc_info.value).lower()
    assert env.snapshot() == snap_before
    assert env.snapshot()["tickets"][0]["status"] == "open"


def test_update_ticket_resolution_after_full_refund_succeeds(
    scenario_task: dict[str, Any],
) -> None:
    env = build_support_environment(scenario_task)

    appr = env.request_refund_approval("ord_1042", 24999, "Return inspected")
    env.issue_refund("ord_1042", 24999, appr["id"])

    res = env.update_ticket(
        "ticket_001",
        "resolved",
        "Full refund processed and confirmed.",
    )
    assert res["status"] == "resolved"
    assert "Full refund processed and confirmed." in res["notes"]
    assert env.snapshot()["tickets"][0]["status"] == "resolved"


def test_update_ticket_failures_leave_snapshot_unchanged(
    scenario_task: dict[str, Any],
) -> None:
    env = build_support_environment(scenario_task)
    initial_snap = env.snapshot()

    with pytest.raises(PermissionError):
        env.update_ticket("ticket_missing", "open", "Note")
    assert env.snapshot() == initial_snap

    with pytest.raises(ValueError):
        env.update_ticket("ticket_001", "invalid_status", "Note")
    assert env.snapshot() == initial_snap

    with pytest.raises(ValueError):
        env.update_ticket("ticket_001", "open", "   ")
    assert env.snapshot() == initial_snap

    with pytest.raises(ValueError):
        env.update_ticket("ticket_001", "resolved", "Premature")
    assert env.snapshot() == initial_snap


def test_update_ticket_successful_update_affects_only_intended_ticket(
    scenario_task: dict[str, Any],
) -> None:
    scenario_task["environment_fixture"]["tickets"].append(
        {
            "id": "ticket_002",
            "customer_id": "cust_12",
            "order_id": "ord_1042",
            "status": "open",
            "notes": ["Initial note for ticket 2"],
        }
    )
    env = build_support_environment(scenario_task)

    env.update_ticket("ticket_001", "pending", "Pending note for ticket 001")

    snap = env.snapshot()
    assert snap["tickets"][0]["id"] == "ticket_001"
    assert snap["tickets"][0]["status"] == "pending"

    # Ticket 2 unaffected
    assert snap["tickets"][1]["id"] == "ticket_002"
    assert snap["tickets"][1]["status"] == "open"
    assert snap["tickets"][1]["notes"] == ["Initial note for ticket 2"]

    # Orders, approvals, refunds unaffected
    assert snap["orders"] == scenario_task["environment_fixture"]["orders"]
    assert snap["approvals"] == []
    assert snap["refunds"] == []


def test_update_ticket_returned_ticket_mutation_isolation(
    scenario_task: dict[str, Any],
) -> None:
    env = build_support_environment(scenario_task)

    res = env.update_ticket("ticket_001", "pending", "Original note")
    res["status"] = "tampered"
    res["notes"].append("Tampered note")

    snap = env.snapshot()
    assert snap["tickets"][0]["status"] == "pending"
    assert snap["tickets"][0]["notes"] == ["Original note"]


def test_update_ticket_isolation_across_environments(
    scenario_task: dict[str, Any],
) -> None:
    env1 = build_support_environment(scenario_task)
    env2 = build_support_environment(scenario_task)

    env1.update_ticket("ticket_001", "pending", "Updated in env1")

    assert env1.snapshot()["tickets"][0]["status"] == "pending"
    assert env1.snapshot()["tickets"][0]["notes"] == ["Updated in env1"]

    # env2 unaffected
    assert env2.snapshot()["tickets"][0]["status"] == "open"
    assert env2.snapshot()["tickets"][0]["notes"] == []

    # fixture unaffected
    assert (
        scenario_task["environment_fixture"]["tickets"][0]["status"]
        == "open"
    )
    assert scenario_task["environment_fixture"]["tickets"][0]["notes"] == []
