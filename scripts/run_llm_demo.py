"""CLI runner for executing an evaluation attempt with the Ollama LLM agent."""
import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.agents.ollama import OllamaSupportAgent
from app.execution.runner import run_attempt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an evaluation attempt using a local Ollama LLM agent."
    )
    parser.add_argument(
        "--task",
        type=str,
        default="datasets/support/dev/refund_eligible_001.json",
        help="Path to scenario task JSON file.",
    )
    parser.add_argument(
        "--order-id",
        type=str,
        default="ord_1042",
        help="Explicit order ID for the agent to process.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen3:4b",
        help="Ollama model tag (default: qwen3:4b).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/llm_runs",
        help="Base directory for saving run artifacts (default: artifacts/llm_runs).",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://127.0.0.1:11434",
        help="Ollama server URL (default: http://127.0.0.1:11434).",
    )
    return parser.parse_args()


def print_summary(report: dict[str, Any], report_path: Path) -> None:
    print("\n" + "=" * 60)
    print("OLLAMA AGENT EVALUATION REPORT SUMMARY")
    print("=" * 60)
    print(f"Task Key:         {report.get('task_key')} (v{report.get('task_version')})")
    print(f"Attempt ID:       {report.get('attempt_id')}")
    print(f"Execution Status: {report.get('execution_status')}")
    print(f"Duration:         {report.get('duration_ms')} ms")

    if report.get("execution_status") == "limit_exceeded":
        term = report.get("termination") or {}
        print(f"Limit Exceeded:   {term.get('limit_type')} ({term.get('code')}): {term.get('message')}")
    elif report.get("execution_status") == "error":
        err = report.get("error") or {}
        print(f"Execution Error:  [{err.get('code')}] {err.get('message')}")

    metrics = report.get("metrics") or {}
    print("-" * 60)
    print("METRICS & TELEMETRY")
    print(f"Tool Calls:       {metrics.get('tool_call_count')}")
    print(f"Model Turns:      {metrics.get('model_turn_count')}")
    print(f"Usage Status:     {metrics.get('usage_status')}")

    token_usage = metrics.get("token_usage")
    if token_usage:
        print(
            f"Tokens:           prompt={token_usage.get('prompt_tokens')}, "
            f"completion={token_usage.get('completion_tokens')}, "
            f"total={token_usage.get('total_tokens')}"
        )
    else:
        print("Tokens:           unavailable")

    model_meta = metrics.get("model_metadata") or {}
    if model_meta:
        print(f"Model:            {model_meta.get('model')} (digest: {model_meta.get('model_digest') or 'unknown'})")
        print(f"Ollama Version:   {model_meta.get('server_version') or 'unknown'}")
        print(f"Prompt Version:   {model_meta.get('prompt_version')} (hash: {model_meta.get('prompt_hash')[:12]}...)")

    evals = report.get("evaluations") or {}
    print("-" * 60)
    print("INDEPENDENT EVALUATIONS")
    for eval_name in ("outcome", "trajectory", "policy", "response"):
        ev = evals.get(eval_name, {})
        status = ev.get("status", "unknown")
        if status == "evaluated":
            passed = ev.get("passed")
            result_str = "PASSED" if passed else "FAILED"
            failed_checks = ev.get("failed_checks") or []
            print(f"- {eval_name.capitalize():12}: {result_str} (checks: {len(ev.get('checks', {}))})")
            if failed_checks:
                for fc in failed_checks:
                    print(f"    * FAILED CHECK: {fc}")
        elif status == "not_evaluated":
            print(f"- {eval_name.capitalize():12}: NOT EVALUATED ({ev.get('reason')})")
        else:
            print(f"- {eval_name.capitalize():12}: {status.upper()}")

    print("-" * 60)
    print(f"Report saved to:  {report_path}")
    print("=" * 60 + "\n")


def main() -> int:
    args = parse_args()
    task_file = Path(args.task)
    if not task_file.exists():
        print(f"Error: Scenario task file not found at {task_file}", file=sys.stderr)
        return 1

    try:
        task_data = json.loads(task_file.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"Error reading scenario JSON: {exc}", file=sys.stderr)
        return 1

    agent = OllamaSupportAgent(
        model=args.model,
        base_url=args.base_url,
    )

    try:
        report = run_attempt(
            task=task_data,
            order_id=args.order_id,
            agent=agent,
        )
    except Exception as exc:
        print(f"Fatal runner error: {exc}", file=sys.stderr)
        return 1

    # Save artifact
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    unique_run_dir = Path(args.output_dir) / f"{timestamp}_{uuid4().hex[:8]}"
    unique_run_dir.mkdir(parents=True, exist_ok=True)
    report_file = unique_run_dir / "report.json"
    report_file.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print_summary(report, report_file)

    if report.get("execution_status") in ("error", "limit_exceeded"):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

