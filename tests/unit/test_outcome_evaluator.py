import json
from copy import deepcopy
from pathlib import Path

import pytest

from app.agents.scripted import ScriptedRefundAgent
from app.environments.support.environment import SupportEnvironment
from app.evaluators.outcome import evaluate_outcome
from app.telemetry.recorder import TraceRecorder
from app.tools.gateway import ToolGateway


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASK_PATH = (
    PROJECT_ROOT
    / "datasets/support/dev/refund_eligible_001.json"
)


@pytest.fixture
def completed_case():
    task = json.loads(TASK_PATH.read_text(encoding="utf-8"))

    environment = SupportEnvironment(
        fixture=task["environment_fixture"],
        authenticated_customer_id=(
            task["agent_input"]["authenticated_customer_id"]
        ),
    )
    recorder = TraceRecorder(
        attempt_id="attempt_outcome_test",
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

    return environment.snapshot(), task["evaluation_spec"]


def test_correct_outcome_passes_without_mutating_inputs(completed_case):
    state, spec = completed_case
    original_state = deepcopy(state)
    original_spec = deepcopy(spec)

    result = evaluate_outcome(
        final_state=state,
        evaluation_spec=spec,
    )

    assert result["passed"] is True
    assert result["failed_checks"] == []
    assert state == original_state
    assert spec == original_spec


@pytest.mark.parametrize(
    "failure",
    [
        "wrong_order_balance",
        "missing_refund_record",
        "duplicate_refund_effect",
        "wrong_currency",
        "unresolved_ticket",
    ],
)
def test_incorrect_outcomes_fail(completed_case, failure):
    state, spec = completed_case

    order = next(
        item for item in state["orders"]
        if item["id"] == "ord_1042"
    )
    ticket = next(
        item for item in state["tickets"]
        if item["id"] == "ticket_001"
    )

    if failure == "wrong_order_balance":
        order["refunded_amount_minor"] = 0
        expected_failure = "order_refund_total_matches"

    elif failure == "missing_refund_record":
        state["refunds"].clear()
        expected_failure = "refund_records_total_matches"

    elif failure == "duplicate_refund_effect":
        # Keep the correct total, but split it into two financial effects.
        first = state["refunds"][0]
        second = deepcopy(first)
        second["id"] = "refund_duplicate_test"

        first["amount_minor"] = 10000
        second["amount_minor"] = 14999
        state["refunds"].append(second)

        expected_failure = "refund_effect_count_matches"

    elif failure == "wrong_currency":
        state["refunds"][0]["currency"] = "EUR"
        expected_failure = "refund_currency_matches_order"

    else:
        ticket["status"] = "open"
        expected_failure = "ticket_outcome_matches"

    result = evaluate_outcome(
        final_state=state,
        evaluation_spec=spec,
    )

    assert result["passed"] is False
    assert result["checks"][expected_failure] is False
    assert expected_failure in result["failed_checks"]

    if failure == "duplicate_refund_effect":
        assert result["checks"]["order_refund_total_matches"] is True
        assert result["checks"]["refund_records_total_matches"] is True