"""Deterministic scripted fake agent for baseline testing and verification."""
from typing import Any

from app.tools.gateway import ToolGateway


class ScriptedRefundAgent:
    """Deterministic agent for the initial full-refund workflow."""

    def run(
        self,
        *,
        gateway: ToolGateway,
        order_id: str,
        ticket_id: str,
    ) -> dict[str, Any]:
        def call(tool_name: str, arguments: dict[str, Any]):
            response = gateway.call(tool_name, arguments)

            if not response["ok"]:
                return None

            return response["result"]

        order = call("get_order", {"order_id": order_id})
        if order is None:
            return self._stop("Could not access the order.")

        return_status = call(
            "get_return_status",
            {"order_id": order_id},
        )
        if return_status is None:
            return self._stop("Could not check the return.")

        if (
            return_status["received"] is not True
            or return_status["inspection_passed"] is not True
        ):
            return self._stop("The return is not eligible for this refund.")

        amount = (
            order["paid_amount_minor"]
            - order["refunded_amount_minor"]
        )
        if amount <= 0:
            return self._stop("There is no remaining refundable balance.")

        approval = call(
            "request_refund_approval",
            {
                "order_id": order_id,
                "amount_minor": amount,
                "reason": "Return received and inspection passed.",
            },
        )
        if approval is None or approval["status"] != "approved":
            return self._stop("Refund approval was not obtained.")

        refund = call(
            "issue_refund",
            {
                "order_id": order_id,
                "amount_minor": amount,
                "approval_id": approval["id"],
            },
        )
        if refund is None:
            return self._stop("The refund could not be confirmed.")

        status = call("get_refund_status", {"order_id": order_id})
        if status is None:
            return self._stop("Could not verify the final refund status.")

        if status["remaining_refundable_amount_minor"] != 0:
            return self._stop("The full refund is not yet confirmed.")

        # Integer arithmetic preserves exact cents.
        amount_text = f"{amount // 100}.{amount % 100:02d}"
        message = (
            f"Confirmed refund of {amount_text} {order['currency']}."
        )

        ticket = call(
            "update_ticket",
            {
                "ticket_id": ticket_id,
                "status": "resolved",
                "note": message,
            },
        )
        if ticket is None:
            return self._stop(
                f"{message} The support ticket could not be resolved."
            )

        return {
            "status": "completed",
            "final_response": message,
        }

    @staticmethod
    def _stop(message: str) -> dict[str, Any]:
        return {
            "status": "stopped",
            "final_response": message,
        }


class SkipReturnInspectionAgent:
    """Scripted agent that deliberately omits return inspection before refunding."""

    def run(
        self,
        *,
        gateway: ToolGateway,
        order_id: str,
        ticket_id: str,
    ) -> dict[str, Any]:
        def call(tool_name: str, arguments: dict[str, Any]):
            response = gateway.call(tool_name, arguments)

            if not response["ok"]:
                return None

            return response["result"]

        order = call("get_order", {"order_id": order_id})
        if order is None:
            return self._stop("Could not access the order.")

        # Deliberately omit get_return_status check

        amount = (
            order["paid_amount_minor"]
            - order["refunded_amount_minor"]
        )
        if amount <= 0:
            return self._stop("There is no remaining refundable balance.")

        approval = call(
            "request_refund_approval",
            {
                "order_id": order_id,
                "amount_minor": amount,
                "reason": "Return received and inspection passed.",
            },
        )
        if approval is None or approval["status"] != "approved":
            return self._stop("Refund approval was not obtained.")

        refund = call(
            "issue_refund",
            {
                "order_id": order_id,
                "amount_minor": amount,
                "approval_id": approval["id"],
            },
        )
        if refund is None:
            return self._stop("The refund could not be confirmed.")

        status = call("get_refund_status", {"order_id": order_id})
        if status is None:
            return self._stop("Could not verify the final refund status.")

        if status["remaining_refundable_amount_minor"] != 0:
            return self._stop("The full refund is not yet confirmed.")

        # Integer arithmetic preserves exact cents.
        amount_text = f"{amount // 100}.{amount % 100:02d}"
        message = (
            f"Confirmed refund of {amount_text} {order['currency']}."
        )

        ticket = call(
            "update_ticket",
            {
                "ticket_id": ticket_id,
                "status": "resolved",
                "note": message,
            },
        )
        if ticket is None:
            return self._stop(
                f"{message} The support ticket could not be resolved."
            )

        return {
            "status": "completed",
            "final_response": message,
        }

    @staticmethod
    def _stop(message: str) -> dict[str, Any]:
        return {
            "status": "stopped",
            "final_response": message,
        }
