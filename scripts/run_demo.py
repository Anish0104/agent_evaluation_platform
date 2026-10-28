"""CLI demo to execute a single agent attempt on a task and print its report."""
import argparse
import json
import sys
from pathlib import Path

from app.execution.runner import run_attempt


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Execute a single in-memory attempt of ScriptedRefundAgent against "
            "a task scenario and output the evaluation report as JSON.\n\n"
            "NOTE: Execution status 'completed' indicates the agent run and evaluators "
            "finished without infrastructure crashes; it does not imply evaluation checks passed."
        )
    )
    parser.add_argument(
        "--task",
        type=Path,
        required=True,
        help="Path to the JSON task definition file.",
    )
    parser.add_argument(
        "--order-id",
        type=str,
        required=True,
        help="Explicit order ID to process (must not be derived from expected outcomes).",
    )

    args = parser.parse_args()

    if not args.task.exists():
        sys.stderr.write(f"Error: Task file not found: {args.task}\n")
        sys.exit(1)

    try:
        task_data = json.loads(args.task.read_text(encoding="utf-8"))
    except Exception as exc:
        sys.stderr.write(f"Error reading task JSON: {exc}\n")
        sys.exit(1)

    try:
        report = run_attempt(
            task=task_data,
            order_id=args.order_id,
        )
    except Exception as exc:
        sys.stderr.write(f"Fatal runner setup error: {exc}\n")
        sys.exit(1)

    json.dump(report, sys.stdout, indent=2)
    sys.stdout.write("\n")

    if report.get("execution_status") in ("error", "limit_exceeded"):
        sys.exit(1)


if __name__ == "__main__":
    main()

