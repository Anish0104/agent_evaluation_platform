"""CLI script to run and compare ScriptedRefundAgent and SkipReturnInspectionAgent."""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.agents.scripted import ScriptedRefundAgent, SkipReturnInspectionAgent
from app.execution.runner import run_attempt


def build_agent_summary(report: dict[str, Any], report_filename: str, agent_id: str) -> dict[str, Any]:
    evals = report.get("evaluations", {})
    agent_result = report.get("agent_result")
    agent_status = agent_result.get("status") if isinstance(agent_result, dict) else None

    return {
        "agent_id": agent_id,
        "report_file": report_filename,
        "attempt_id": report.get("attempt_id"),
        "execution_status": report.get("execution_status"),
        "agent_status": agent_status,
        "duration_ms": report.get("duration_ms"),
        "tool_call_count": report.get("metrics", {}).get("tool_call_count"),
        "evaluations": {
            "outcome": {
                "status": evals.get("outcome", {}).get("status"),
                "passed": evals.get("outcome", {}).get("passed"),
                "failed_checks": evals.get("outcome", {}).get("failed_checks", []),
            },
            "trajectory": {
                "status": evals.get("trajectory", {}).get("status"),
                "passed": evals.get("trajectory", {}).get("passed"),
                "failed_checks": evals.get("trajectory", {}).get("failed_checks", []),
            },
            "policy": {
                "status": evals.get("policy", {}).get("status"),
            },
            "response": {
                "status": evals.get("response", {}).get("status"),
            },
        },
    }


def compare_agents(
    *,
    task_path: Path,
    order_id: str,
    output_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    if not task_path.exists():
        raise FileNotFoundError(f"Task file not found: {task_path}")

    task_bytes = task_path.read_bytes()
    task_file_sha256 = hashlib.sha256(task_bytes).hexdigest()
    task_data = json.loads(task_bytes.decode("utf-8"))

    # Execute both agents independently with fresh instances
    good_agent = ScriptedRefundAgent()
    good_report = run_attempt(
        task=task_data,
        order_id=order_id,
        agent=good_agent,
    )

    faulty_agent = SkipReturnInspectionAgent()
    faulty_report = run_attempt(
        task=task_data,
        order_id=order_id,
        agent=faulty_agent,
    )

    comparison_id = f"comp_{uuid4().hex[:12]}"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    target_dir = output_dir / f"{timestamp}_{comparison_id}"
    target_dir.mkdir(parents=True, exist_ok=True)

    good_filename = "good_agent.json"
    faulty_filename = "skip_return_inspection_agent.json"
    comparison_filename = "comparison.json"

    comparison_data: dict[str, Any] = {
        "schema_version": "1.0.0",
        "comparison_id": comparison_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": {
            "task_key": task_data.get("task_key"),
            "task_version": task_data.get("version", 1),
            "task_file_sha256": task_file_sha256,
        },
        "agents": {
            "scripted_refund_v1": build_agent_summary(good_report, good_filename, "scripted_refund_v1"),
            "skip_return_inspection_v1": build_agent_summary(faulty_report, faulty_filename, "skip_return_inspection_v1"),
        },
    }

    # Save artifacts unconditionally so reports remain available for inspection
    (target_dir / good_filename).write_text(json.dumps(good_report, indent=2), encoding="utf-8")
    (target_dir / faulty_filename).write_text(json.dumps(faulty_report, indent=2), encoding="utf-8")
    (target_dir / comparison_filename).write_text(json.dumps(comparison_data, indent=2), encoding="utf-8")

    return target_dir, comparison_data


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a reproducible comparison between ScriptedRefundAgent (good) and "
            "SkipReturnInspectionAgent (faulty) on a scenario task.\n\n"
            "NOTE: Ordinary business evaluation failures (such as the faulty agent failing "
            "trajectory checks) are valid experimental findings and will result in exit code 0. "
            "A non-zero exit code indicates invalid arguments, limit termination, or infrastructure errors."
        )
    )
    parser.add_argument(
        "--task",
        type=Path,
        required=True,
        help="Path to the scenario task JSON file.",
    )
    parser.add_argument(
        "--order-id",
        type=str,
        required=True,
        help="Explicit order ID to evaluate.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/comparisons"),
        help="Base directory to save comparison artifacts (default: artifacts/comparisons).",
    )

    args = parser.parse_args()

    try:
        target_dir, comparison = compare_agents(
            task_path=args.task,
            order_id=args.order_id,
            output_dir=args.output_dir,
        )
    except FileNotFoundError as exc:
        sys.stderr.write(f"Error: {exc}\n")
        sys.exit(1)
    except Exception as exc:
        sys.stderr.write(f"Execution Error: {exc}\n")
        sys.exit(1)

    agents = comparison["agents"]
    good = agents["scripted_refund_v1"]
    faulty = agents["skip_return_inspection_v1"]

    def format_eval(eval_data: dict[str, Any]) -> str:
        status = eval_data.get("status")
        if status == "evaluated":
            return "PASSED" if eval_data.get("passed") else "FAILED"
        if status == "not_evaluated":
            return "NOT_EVALUATED"
        return "ERROR"

    good_outcome = format_eval(good["evaluations"]["outcome"])
    faulty_outcome = format_eval(faulty["evaluations"]["outcome"])

    good_traj = format_eval(good["evaluations"]["trajectory"])
    faulty_traj = format_eval(faulty["evaluations"]["trajectory"])
    faulty_traj_failed = faulty["evaluations"]["trajectory"].get("failed_checks", [])
    faulty_traj_detail = f" (failed checks: {', '.join(faulty_traj_failed)})" if faulty_traj_failed else ""

    print("=== Agent Comparison Summary ===")
    print(f"Comparison ID: {comparison['comparison_id']}")
    print(f"Task: {comparison['task']['task_key']} (version {comparison['task']['task_version']})")
    print(f"Output Directory: {target_dir}")
    print()
    print("Agent: scripted_refund_v1 (Good Agent)")
    print(f"  Execution:  {good['execution_status']}")
    print(f"  Agent:      {good['agent_status']}")
    print(f"  Outcome:    {good_outcome}")
    print(f"  Trajectory: {good_traj}")
    print(f"  Tool Calls: {good['tool_call_count']}")
    print(f"  Duration:   {good['duration_ms']:.2f} ms")
    print()
    print("Agent: skip_return_inspection_v1 (Faulty Agent)")
    print(f"  Execution:  {faulty['execution_status']}")
    print(f"  Agent:      {faulty['agent_status']}")
    print(f"  Outcome:    {faulty_outcome}")
    print(f"  Trajectory: {faulty_traj}{faulty_traj_detail}")
    print(f"  Tool Calls: {faulty['tool_call_count']}")
    print(f"  Duration:   {faulty['duration_ms']:.2f} ms")

    if (
        good.get("execution_status") in ("limit_exceeded", "error")
        or faulty.get("execution_status") in ("limit_exceeded", "error")
    ):
        sys.exit(1)


if __name__ == "__main__":
    main()

