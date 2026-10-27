"""Database seeding script for default datasets, agents, and scenarios."""
import hashlib
import json
from pathlib import Path

from sqlalchemy import select

from app.db.models import Dataset, DatasetVersion, TaskVersion
from app.db.session import SessionLocal


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TASK_PATH = (
    PROJECT_ROOT
    / "datasets/support/dev/refund_eligible_001.json"
)

DATASET_NAME = "electronics-support"
DATASET_VERSION = 1


def load_task() -> dict:
    task = json.loads(TASK_PATH.read_text(encoding="utf-8"))

    required_fields = {
        "task_key",
        "version",
        "category",
        "scenario_family",
        "split",
        "agent_input",
        "environment_fixture",
        "evaluation_spec",
    }

    if set(task) != required_fields:
        raise ValueError("Task fields do not match the expected structure.")

    if type(task["version"]) is not int or task["version"] < 1:
        raise ValueError("Task version must be a positive integer.")

    if task["split"] != "dev":
        raise ValueError("This seed task must belong to the dev split.")

    # Checks specific to our first $249.99 eligible-refund scenario.
    fixture = task["environment_fixture"]
    order = fixture["orders"][0]
    expected = task["evaluation_spec"]["expected_outcome"]
    reviewer = fixture["reviewer_behavior"]

    amounts = [
        order["paid_amount_minor"],
        reviewer["allowed_amount_minor"],
        expected["refund_total_minor"],
    ]

    if any(type(amount) is not int for amount in amounts):
        raise ValueError("Amounts must be integer cents.")

    if amounts != [24999, 24999, 24999]:
        raise ValueError("Expected a consistent refund amount of 24999 cents.")

    if order["currency"] != "USD":
        raise ValueError("Expected USD.")

    if (
        type(order["refunded_amount_minor"]) is not int
        or order["refunded_amount_minor"] != 0
    ):
        raise ValueError("This scenario must start with zero refunded cents.")

    if not (
        order["id"]
        == reviewer["allowed_order_id"]
        == expected["order_id"]
    ):
        raise ValueError("Order IDs must match.")

    if order["customer_id"] != task["agent_input"]["authenticated_customer_id"]:
        raise ValueError("The eligible-refund customer must own the order.")

    return task


def main() -> None:
    task = load_task()

    # Hash the full release content, not whitespace or JSON key order.
    release = {
        "dataset_name": DATASET_NAME,
        "version": DATASET_VERSION,
        "tasks": [task],
    }
    canonical_json = json.dumps(
        release,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    content_hash = hashlib.sha256(
        canonical_json.encode("utf-8")
    ).hexdigest()

    # Commit all records together; roll back if any operation fails.
    with SessionLocal.begin() as session:
        dataset = session.scalar(
            select(Dataset).where(Dataset.name == DATASET_NAME)
        )

        if dataset is None:
            dataset = Dataset(
                name=DATASET_NAME,
                description="Synthetic electronics-store support scenarios.",
            )
            session.add(dataset)
            session.flush()

        existing_version = session.scalar(
            select(DatasetVersion).where(
                DatasetVersion.dataset_id == dataset.id,
                DatasetVersion.version == DATASET_VERSION,
            )
        )

        if existing_version is not None:
            if existing_version.content_hash != content_hash:
                raise ValueError(
                    "Dataset version already exists with different content. "
                    "Create a new dataset version instead of overwriting it."
                )

            print("Already imported: electronics-support v1. No changes made.")
            return

        dataset_version = DatasetVersion(
            dataset_id=dataset.id,
            version=DATASET_VERSION,
            content_hash=content_hash,
            manifest={
                "task_count": 1,
                "task_keys": [task["task_key"]],
                "split_counts": {"dev": 1, "validation": 0, "test": 0},
                "hash_algorithm": "sha256",
                "canonicalization": "sorted-compact-json-v1",
            },
        )
        session.add(dataset_version)
        session.flush()

        session.add(
            TaskVersion(
                dataset_version_id=dataset_version.id,
                **task,
            )
        )

    print("Imported electronics-support v1: 1 task.")
    print(f"Content hash: {content_hash}")


if __name__ == "__main__":
    main()