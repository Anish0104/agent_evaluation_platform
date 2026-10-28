"""Public tool definitions and schemas for LLM tool calling."""
from copy import deepcopy
from typing import Any

from app.tools.schemas import (
    GetOrderArguments,
    GetRefundStatusArguments,
    GetReturnStatusArguments,
    IssueRefundArguments,
    RequestRefundApprovalArguments,
    UpdateTicketArguments,
)

_TOOL_DESCRIPTIONS = {
    "get_order": "Retrieve order details including customer ownership, amounts, currency, and line items.",
    "get_return_status": "Retrieve return details for an order, including whether the return was received and passed inspection.",
    "get_refund_status": "Retrieve existing refunds and current refund balance for an order.",
    "request_refund_approval": (
        "Request authorization from a reviewer to issue a refund. Required before issuing any refund. "
        "The amount_minor argument must be integer cents (e.g. 24999 represents $249.99 USD)."
    ),
    "issue_refund": (
        "Execute an authorized refund for an order. Requires a valid approval_id from request_refund_approval. "
        "The amount_minor argument must be integer cents (e.g. 24999 represents $249.99 USD)."
    ),
    "update_ticket": "Update customer support ticket status and append an internal note.",
}

_PARAM_DESCRIPTIONS = {
    "order_id": "Identifier of the order (e.g. 'ord_1042').",
    "amount_minor": "Refund amount in integer cents (e.g., 24999 for $249.99 USD).",
    "reason": "Clear reason explaining why the refund is justified.",
    "approval_id": "The approval identifier obtained from request_refund_approval.",
    "ticket_id": "Identifier of the support ticket (e.g. 'ticket_001').",
    "status": "New status of the ticket ('open', 'pending', or 'resolved').",
    "note": "Summary note or explanation for the ticket update.",
}

_TOOL_SCHEMAS = {
    "get_order": GetOrderArguments,
    "get_return_status": GetReturnStatusArguments,
    "get_refund_status": GetRefundStatusArguments,
    "request_refund_approval": RequestRefundApprovalArguments,
    "issue_refund": IssueRefundArguments,
    "update_ticket": UpdateTicketArguments,
}


def get_support_tool_definitions() -> list[dict[str, Any]]:
    """Return OpenAI/Ollama-compatible tool calling definitions derived from Pydantic schemas."""
    definitions: list[dict[str, Any]] = []

    for name, schema_cls in _TOOL_SCHEMAS.items():
        json_schema = schema_cls.model_json_schema()
        parameters: dict[str, Any] = {
            "type": "object",
            "properties": {},
            "required": json_schema.get("required", []),
        }

        for prop_name, prop_val in json_schema.get("properties", {}).items():
            param_spec = deepcopy(prop_val)
            if prop_name in _PARAM_DESCRIPTIONS:
                param_spec["description"] = _PARAM_DESCRIPTIONS[prop_name]
            parameters["properties"][prop_name] = param_spec

        definitions.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": _TOOL_DESCRIPTIONS[name],
                    "parameters": parameters,
                },
            }
        )

    return definitions

