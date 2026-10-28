from typing import Annotated, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
)


def reject_blank(value: str) -> str:
    if not value.strip():
        raise ValueError("Value must not be blank.")
    return value


NonBlankString = Annotated[
    str,
    Field(min_length=1),
    AfterValidator(reject_blank),
]

PositiveAmount = Annotated[int, Field(gt=0)]


class ToolArguments(BaseModel):
    """Shared validation rules for agent-provided arguments."""

    model_config = ConfigDict(
        strict=True,
        extra="forbid",
    )


class GetOrderArguments(ToolArguments):
    order_id: NonBlankString


class GetReturnStatusArguments(ToolArguments):
    order_id: NonBlankString


class GetRefundStatusArguments(ToolArguments):
    order_id: NonBlankString


class RequestRefundApprovalArguments(ToolArguments):
    order_id: NonBlankString
    amount_minor: PositiveAmount
    reason: NonBlankString


class IssueRefundArguments(ToolArguments):
    order_id: NonBlankString
    amount_minor: PositiveAmount
    approval_id: NonBlankString


class UpdateTicketArguments(ToolArguments):
    ticket_id: NonBlankString
    status: Literal["open", "pending", "resolved"]
    note: NonBlankString