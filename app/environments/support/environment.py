from copy import deepcopy
from typing import Any
from uuid import uuid4


class SupportEnvironment:
    """Isolated simulated environment for customer support scenarios."""

    def __init__(
        self,
        *,
        fixture: dict[str, Any],
        authenticated_customer_id: str,
    ) -> None:
        self.environment_id: str = str(uuid4())
        self.authenticated_customer_id: str = authenticated_customer_id
        self._state: dict[str, Any] = deepcopy(fixture)

    def snapshot(self) -> dict[str, Any]:
        """Return a deep copy of the entire internal simulated state.

        Note:
            This method is intended for trusted persistence, checkpointing,
            and evaluation code. It must not be exposed as an agent tool.
        """
        return deepcopy(self._state)

    def _get_authorized_order_record(self, order_id: str) -> dict[str, Any]:
        """Internal helper to find an authorized order record.

        Raises:
            PermissionError: If the order is not found or does not belong to the
                authenticated customer.
        """
        for order in self._state.get("orders", []):
            if order.get("id") == order_id:
                if order.get("customer_id") == self.authenticated_customer_id:
                    return order
                raise PermissionError("Order not found or access denied.")
        raise PermissionError("Order not found or access denied.")

    def get_order(self, order_id: str) -> dict[str, Any]:
        """Retrieve an order for the authenticated customer.

        Args:
            order_id: The unique identifier of the order.

        Returns:
            A deep copy of the authorized order dictionary.

        Raises:
            PermissionError: If the order is not found or does not belong to the
                authenticated customer.
        """
        return deepcopy(self._get_authorized_order_record(order_id))

    def get_return_status(self, order_id: str) -> dict[str, Any]:
        """Retrieve return status for an authorized order.

        Args:
            order_id: The unique identifier of the order.

        Returns:
            A deep copy of the matching return record, or a default unreceived
            status dict if no return record exists for an authorized order.

        Raises:
            PermissionError: If the order is not found or does not belong to the
                authenticated customer.
        """
        self._get_authorized_order_record(order_id)
        for ret in self._state.get("returns", []):
            if ret.get("order_id") == order_id:
                return deepcopy(ret)
        return {
            "order_id": order_id,
            "received": False,
            "inspection_passed": False,
        }

    def request_refund_approval(
        self,
        order_id: str,
        amount_minor: int,
        reason: str,
    ) -> dict[str, Any]:
        """Request supervisor approval for a refund.

        Args:
            order_id: The unique identifier of the order.
            amount_minor: Amount to refund in USD cents (positive integer).
            reason: Non-empty description of the refund reason.

        Returns:
            A deep copy of the recorded approval decision record.

        Raises:
            PermissionError: If the order is not found or access is denied.
            ValueError: For invalid arguments or unmet business preconditions.
        """
        order = self._get_authorized_order_record(order_id)

        if isinstance(amount_minor, bool) or not isinstance(amount_minor, int) or amount_minor <= 0:
            raise ValueError("Amount must be a positive integer.")

        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("Reason must be a non-empty, non-whitespace string.")

        matching_return = None
        for ret in self._state.get("returns", []):
            if ret.get("order_id") == order_id:
                matching_return = ret
                break

        if (
            matching_return is None
            or matching_return.get("received") is not True
            or matching_return.get("inspection_passed") is not True
        ):
            raise ValueError("Return has not been received or inspection has not passed.")

        remaining_balance = order.get("paid_amount_minor", 0) - order.get("refunded_amount_minor", 0)
        if amount_minor > remaining_balance:
            raise ValueError("Requested refund amount exceeds remaining refundable balance.")

        reviewer = self._state.get("reviewer_behavior", {})
        is_approved = (
            reviewer.get("decision") == "approve"
            and reviewer.get("allowed_order_id") == order_id
            and reviewer.get("allowed_amount_minor") == amount_minor
        )
        status = "approved" if is_approved else "rejected"

        record = {
            "id": str(uuid4()),
            "environment_id": self.environment_id,
            "customer_id": self.authenticated_customer_id,
            "order_id": order_id,
            "amount_minor": amount_minor,
            "currency": order.get("currency", "USD"),
            "reason": reason,
            "status": status,
        }

        self._state.setdefault("approvals", []).append(record)
        return deepcopy(record)

    def issue_refund(
        self,
        order_id: str,
        amount_minor: int,
        approval_id: str,
    ) -> dict[str, Any]:
        """Issue a refund against a valid supervisor approval.

        Note:
            Execution is sequential in-memory for this prototype and does not provide
            thread-safe or distributed idempotency.

        Args:
            order_id: The unique identifier of the order.
            amount_minor: Amount to refund in USD cents (positive integer).
            approval_id: ID of the pre-approved refund request.

        Returns:
            A deep copy of the completed refund record.

        Raises:
            PermissionError: If the order is not found or access is denied.
            ValueError: If inputs or approvals are invalid, mismatched, or preconditions fail.
        """
        order = self._get_authorized_order_record(order_id)

        if isinstance(amount_minor, bool) or not isinstance(amount_minor, int) or amount_minor <= 0:
            raise ValueError("Amount must be a positive integer.")

        if not isinstance(approval_id, str) or not approval_id.strip():
            raise ValueError("Approval ID must be a non-empty string.")

        matching_approval = None
        for appr in self._state.get("approvals", []):
            if appr.get("id") == approval_id:
                matching_approval = appr
                break

        if matching_approval is None:
            raise ValueError("Approval not found.")

        if (
            matching_approval.get("status") != "approved"
            or matching_approval.get("environment_id") != self.environment_id
            or matching_approval.get("customer_id") != self.authenticated_customer_id
            or matching_approval.get("order_id") != order_id
            or matching_approval.get("amount_minor") != amount_minor
            or matching_approval.get("currency") != order.get("currency", "USD")
        ):
            raise ValueError("Approval does not match or is not approved.")

        # Idempotency check: return existing refund if already processed
        for ref in self._state.get("refunds", []):
            if ref.get("approval_id") == approval_id:
                return deepcopy(ref)

        # For a new refund, recheck preconditions before modifying state
        matching_return = None
        for ret in self._state.get("returns", []):
            if ret.get("order_id") == order_id:
                matching_return = ret
                break

        if (
            matching_return is None
            or matching_return.get("received") is not True
            or matching_return.get("inspection_passed") is not True
        ):
            raise ValueError("Return has not been received or inspection has not passed.")

        remaining_balance = order.get("paid_amount_minor", 0) - order.get("refunded_amount_minor", 0)
        if amount_minor > remaining_balance:
            raise ValueError("Refund amount exceeds remaining refundable balance.")

        new_refund = {
            "id": str(uuid4()),
            "environment_id": self.environment_id,
            "customer_id": self.authenticated_customer_id,
            "order_id": order_id,
            "amount_minor": amount_minor,
            "currency": order.get("currency", "USD"),
            "approval_id": approval_id,
            "status": "completed",
        }

        self._state.setdefault("refunds", []).append(new_refund)
        order["refunded_amount_minor"] = order.get("refunded_amount_minor", 0) + amount_minor

        return deepcopy(new_refund)

    def get_refund_status(self, order_id: str) -> dict[str, Any]:
        """Retrieve refund status and history for an authorized order.

        Args:
            order_id: The unique identifier of the order.

        Returns:
            A dictionary containing order refund summary and independent copies
            of associated refund records.

        Raises:
            PermissionError: If the order is not found or does not belong to the
                authenticated customer.
        """
        order = self._get_authorized_order_record(order_id)
        paid = order.get("paid_amount_minor", 0)
        refunded = order.get("refunded_amount_minor", 0)
        remaining = paid - refunded

        order_refunds = [
            deepcopy(r)
            for r in self._state.get("refunds", [])
            if r.get("order_id") == order_id
        ]

        return {
            "order_id": order_id,
            "currency": order.get("currency", "USD"),
            "paid_amount_minor": paid,
            "refunded_amount_minor": refunded,
            "remaining_refundable_amount_minor": remaining,
            "refunds": order_refunds,
        }

    def update_ticket(
        self,
        ticket_id: str,
        status: str,
        note: str,
    ) -> dict[str, Any]:
        """Update a customer support ticket status and append a note.

        Note:
            This method is not idempotent: repeating a valid call will append
            another note to the ticket history.

            The resolution rule enforced here (requiring at least one completed refund
            and zero remaining balance on the linked order) is specific to this initial
            full-refund workflow scenario. Future scenarios (e.g. inquiry lookup,
            policy refusal, escalation) will require distinct workflow-specific
            resolution rules. Not all customer support tickets require refunds.

        Args:
            ticket_id: Unique identifier of the ticket.
            status: Target status ('open', 'pending', or 'resolved').
            note: Non-empty description or comment to append to the ticket.

        Returns:
            A deep copy of the updated ticket dictionary.

        Raises:
            PermissionError: If the ticket is not found, unauthorized, or its linked
                order is inaccessible to the authenticated customer.
            ValueError: For invalid input arguments or unmet resolution preconditions.
        """
        if not isinstance(ticket_id, str) or not ticket_id.strip():
            raise ValueError("Ticket ID must be a non-empty string.")

        if not isinstance(status, str) or status not in {"open", "pending", "resolved"}:
            raise ValueError("Invalid status. Allowed statuses: 'open', 'pending', 'resolved'.")

        if not isinstance(note, str) or not note.strip():
            raise ValueError("Note must be a non-empty, non-whitespace string.")

        matching_ticket = None
        for ticket in self._state.get("tickets", []):
            if ticket.get("id") == ticket_id:
                matching_ticket = ticket
                break

        if matching_ticket is None:
            raise PermissionError("Ticket not found or access denied.")

        if matching_ticket.get("customer_id") != self.authenticated_customer_id:
            raise PermissionError("Ticket not found or access denied.")

        linked_order_id = matching_ticket.get("order_id")
        linked_order_record = None
        if linked_order_id:
            for order in self._state.get("orders", []):
                if order.get("id") == linked_order_id:
                    if order.get("customer_id") == self.authenticated_customer_id:
                        linked_order_record = order
                    break
            if linked_order_record is None:
                raise PermissionError("Ticket not found or access denied.")

        if status == "resolved":
            if linked_order_record is None:
                raise ValueError("Ticket resolution requires an accessible linked order.")

            completed_refunds = [
                r
                for r in self._state.get("refunds", [])
                if r.get("order_id") == linked_order_id and r.get("status") == "completed"
            ]
            if not completed_refunds:
                raise ValueError(
                    "Ticket resolution requires at least one completed refund for the linked order."
                )

            remaining_balance = (
                linked_order_record.get("paid_amount_minor", 0)
                - linked_order_record.get("refunded_amount_minor", 0)
            )
            if remaining_balance != 0:
                raise ValueError(
                    "Ticket resolution requires zero remaining refundable balance on the linked order."
                )

        matching_ticket["status"] = status
        matching_ticket.setdefault("notes", []).append(note)

        return deepcopy(matching_ticket)
