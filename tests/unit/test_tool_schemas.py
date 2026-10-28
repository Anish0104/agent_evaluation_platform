import pytest
from pydantic import ValidationError

from app.tools.schemas import GetOrderArguments, UpdateTicketArguments

import pytest
from pydantic import ValidationError

from app.tools.schemas import (
    GetOrderArguments,
    GetRefundStatusArguments,
    GetReturnStatusArguments,
    IssueRefundArguments,
    RequestRefundApprovalArguments,
    UpdateTicketArguments,
)


def test_valid_order_id_is_accepted():
    arguments = GetOrderArguments.model_validate(
        {"order_id": "ord_1042"}
    )

    assert arguments.order_id == "ord_1042"


@pytest.mark.parametrize(
    "arguments",
    [
        {"order_id": 123},
        {"order_id": ""},
        {"order_id": "   "},
        {"order_id": "ord_1042", "admin": True},
        {},
    ],
)
def test_invalid_get_order_arguments_are_rejected(arguments):
    with pytest.raises(ValidationError):
        GetOrderArguments.model_validate(arguments)

        from app.tools.schemas import (
    GetRefundStatusArguments,
    GetReturnStatusArguments,
    IssueRefundArguments,
    RequestRefundApprovalArguments,
    UpdateTicketArguments,
)


@pytest.mark.parametrize(
    "schema",
    [GetReturnStatusArguments, GetRefundStatusArguments],
)
def test_order_lookup_schemas(schema):
    arguments = schema.model_validate({"order_id": "ord_1042"})
    assert arguments.order_id == "ord_1042"

    with pytest.raises(ValidationError):
        schema.model_validate({"order_id": "   "})


@pytest.mark.parametrize(
    "schema, extra",
    [
        (
            RequestRefundApprovalArguments,
            {"reason": "Return received and inspected."},
        ),
        (
            IssueRefundArguments,
            {"approval_id": "approval_001"},
        ),
    ],
)
def test_refund_schemas_accept_integer_cents(schema, extra):
    arguments = schema.model_validate(
        {
            "order_id": "ord_1042",
            "amount_minor": 24999,
            **extra,
        }
    )

    assert arguments.amount_minor == 24999


@pytest.mark.parametrize(
    "schema, extra",
    [
        (
            RequestRefundApprovalArguments,
            {"reason": "Return received and inspected."},
        ),
        (
            IssueRefundArguments,
            {"approval_id": "approval_001"},
        ),
    ],
)
@pytest.mark.parametrize(
    "amount",
    [0, -1, 249.99, "24999", True, False],
)
def test_refund_schemas_reject_invalid_amounts(schema, extra, amount):
    with pytest.raises(ValidationError):
        schema.model_validate(
            {
                "order_id": "ord_1042",
                "amount_minor": amount,
                **extra,
            }
        )


@pytest.mark.parametrize(
    "schema, extra",
    [
        (RequestRefundApprovalArguments, {"reason": "   "}),
        (IssueRefundArguments, {"approval_id": ""}),
    ],
)
def test_refund_schemas_reject_blank_required_text(schema, extra):
    with pytest.raises(ValidationError):
        schema.model_validate(
            {
                "order_id": "ord_1042",
                "amount_minor": 24999,
                **extra,
            }
        )


@pytest.mark.parametrize("status", ["open", "pending", "resolved"])
def test_ticket_schema_accepts_supported_statuses(status):
    arguments = UpdateTicketArguments.model_validate(
        {
            "ticket_id": "ticket_001",
            "status": status,
            "note": "Customer updated.",
        }
    )

    assert arguments.status == status


@pytest.mark.parametrize(
    "changes",
    [
        {"status": "closed"},
        {"ticket_id": "   "},
        {"note": ""},
        {"authenticated_customer_id": "another_customer"},
    ],
)
def test_ticket_schema_rejects_invalid_arguments(changes):
    arguments = {
        "ticket_id": "ticket_001",
        "status": "pending",
        "note": "Waiting for approval.",
    }
    arguments.update(changes)

    with pytest.raises(ValidationError):
        UpdateTicketArguments.model_validate(arguments)