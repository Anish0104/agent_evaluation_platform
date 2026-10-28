"""Local in-memory single-attempt runner for AI Agent Evaluation Platform."""
from copy import deepcopy
from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Callable
from uuid import uuid4

from app.agents.ollama import OllamaAgentError
from app.environments.support.environment import SupportEnvironment
from app.evaluators.outcome import evaluate_outcome
from app.evaluators.trajectory import evaluate_trajectory
from app.execution.budget import ExecutionBudget, ExecutionLimitExceeded
from app.telemetry.recorder import TraceRecorder
from app.tools.gateway import ToolGateway


def run_attempt(
    *,
    task: dict[str, Any],
    order_id: str,
    agent: Any = None,
    clock: Callable[[], float] = perf_counter,
) -> dict[str, Any]:
    """Execute a single evaluation attempt in an isolated in-memory environment.

    Args:
        task: Full scenario task definition dictionary.
        order_id: Explicit order ID for the agent to process.
        agent: Optional agent instance implementing run(gateway=..., order_id=..., ticket_id=...).
               Defaults to ScriptedRefundAgent.
        clock: Optional monotonic clock function (defaults to perf_counter).

    Returns:
        JSON-serializable report containing attempt metadata, execution status,
        independent evaluator results, metrics, initial/final states, and event trace.
    """
    # Boundary: Setup must fail clearly if prerequisite fields are absent
    attempt_id = str(uuid4())
    agent_input = task["agent_input"]
    ticket_id = agent_input["ticket_id"]
    authenticated_customer_id = agent_input["authenticated_customer_id"]
    fixture = task["environment_fixture"]

    # Validate limits configuration before execution
    limits = task.get("evaluation_spec", {}).get("limits")
    if not isinstance(limits, dict):
        raise ValueError("Scenario evaluation_spec.limits must be a dictionary.")

    if "max_tool_calls" not in limits or "timeout_seconds" not in limits:
        raise ValueError(
            "Missing required limits in evaluation_spec: max_tool_calls and timeout_seconds."
        )

    budget = ExecutionBudget(
        max_tool_calls=limits["max_tool_calls"],
        timeout_seconds=limits["timeout_seconds"],
        max_model_turns=limits.get("max_model_turns"),
        clock=clock,
    )

    if agent is None:
        from app.agents.scripted import ScriptedRefundAgent

        agent = ScriptedRefundAgent()

    user_message = agent_input.get("user_message")
    if hasattr(agent, "set_user_message") and callable(agent.set_user_message):
        agent.set_user_message(user_message)
    elif hasattr(agent, "user_message") and getattr(agent, "user_message") is None:
        setattr(agent, "user_message", user_message)

    environment = SupportEnvironment(
        fixture=fixture,
        authenticated_customer_id=authenticated_customer_id,
    )
    recorder = TraceRecorder(
        attempt_id=attempt_id,
        environment_id=environment.environment_id,
    )
    gateway = ToolGateway(
        environment=environment,
        recorder=recorder,
        budget=budget,
        clock=clock,
    )

    initial_state = environment.snapshot()
    final_state: dict[str, Any] | None = None
    agent_result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    termination: dict[str, Any] | None = None
    execution_status = "completed"

    evaluations: dict[str, Any] = {
        "outcome": {
            "status": "not_evaluated",
            "reason": "Evaluator not run.",
        },
        "trajectory": {
            "status": "not_evaluated",
            "reason": "Evaluator not run.",
        },
        "policy": {
            "status": "not_evaluated",
            "reason": "Evaluator not implemented.",
        },
        "response": {
            "status": "not_evaluated",
            "reason": "Evaluator not implemented.",
        },
    }

    # Start attempt budget immediately before recording attempt_started and invoking agent
    budget.start()
    started_at = datetime.now(timezone.utc).isoformat()
    start_time = clock()

    recorder.record(
        event_type="attempt_started",
        payload={
            "task_key": task.get("task_key"),
            "task_version": task.get("version", 1),
        },
    )

    stage = "agent_execution"
    try:
        agent_result = agent.run(
            gateway=gateway,
            order_id=order_id,
            ticket_id=ticket_id,
        )

        # Enforce budget after agent returns to catch clock advance or caught limit exceptions
        budget.check_limits()

        recorder.record(
            event_type="agent_finished",
            payload={
                "status": agent_result.get("status") if isinstance(agent_result, dict) else None,
                "final_response": agent_result.get("final_response") if isinstance(agent_result, dict) else None,
            },
        )

        final_state = environment.snapshot()

        stage = "evaluation"

        # Independent outcome evaluation
        try:
            outcome_res = evaluate_outcome(
                initial_state=initial_state,
                final_state=final_state,
                evaluation_spec=task["evaluation_spec"],
            )
            evaluations["outcome"] = {
                "status": "evaluated",
                "passed": outcome_res["passed"],
                "checks": outcome_res["checks"],
                "failed_checks": outcome_res["failed_checks"],
            }
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            evaluations["outcome"] = {
                "status": "error",
                "reason": "Evaluator execution failed.",
            }
            raise

        # Independent trajectory evaluation
        try:
            trajectory_res = evaluate_trajectory(
                events=recorder.get_events(),
                evaluation_spec=task["evaluation_spec"],
            )
            evaluations["trajectory"] = {
                "status": "evaluated",
                "passed": trajectory_res["passed"],
                "checks": trajectory_res["checks"],
                "failed_checks": trajectory_res["failed_checks"],
            }
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            evaluations["trajectory"] = {
                "status": "error",
                "reason": "Evaluator execution failed.",
            }
            raise

        recorder.record(
            event_type="attempt_finished",
            payload={},
        )
        execution_status = "completed"

    except (KeyboardInterrupt, SystemExit):
        raise
    except ExecutionLimitExceeded as exc:
        execution_status = "limit_exceeded"
        if final_state is None:
            final_state = environment.snapshot()

        termination = {
            "limit_type": exc.limit_type,
            "code": exc.code,
            "message": exc.message,
        }
        error = {
            "stage": stage,
            "code": exc.code,
            "message": exc.message,
            "limit_type": exc.limit_type,
        }
        recorder.record(
            event_type="attempt_limit_exceeded",
            payload=termination,
        )
        evaluations["outcome"] = {
            "status": "not_evaluated",
            "reason": "Execution limit exceeded.",
        }
        evaluations["trajectory"] = {
            "status": "not_evaluated",
            "reason": "Execution limit exceeded.",
        }

    except OllamaAgentError as exc:
        execution_status = "error"
        if final_state is None:
            final_state = environment.snapshot()

        error = {
            "stage": stage,
            "code": exc.code,
            "message": exc.message,
        }
        recorder.record(
            event_type="attempt_failed",
            payload={
                "code": exc.code,
                "message": exc.message,
            },
        )
        evaluations["outcome"] = {
            "status": "not_evaluated",
            "reason": f"Execution error: {exc.code}.",
        }
        evaluations["trajectory"] = {
            "status": "not_evaluated",
            "reason": f"Execution error: {exc.code}.",
        }

    except Exception:
        execution_status = "error"
        if final_state is None:
            final_state = environment.snapshot()

        error_message = f"Unexpected error during {stage}."
        error = {
            "stage": stage,
            "code": "INTERNAL_ERROR",
            "message": error_message,
        }
        recorder.record(
            event_type="attempt_failed",
            payload={
                "code": "INTERNAL_ERROR",
                "message": error_message,
            },
        )
        if stage == "agent_execution":
            evaluations["outcome"] = {
                "status": "not_evaluated",
                "reason": "Execution error.",
            }
            evaluations["trajectory"] = {
                "status": "not_evaluated",
                "reason": "Execution error.",
            }

    end_time = clock()
    finished_at = datetime.now(timezone.utc).isoformat()
    duration_ms = round((end_time - start_time) * 1000, 2)

    # Collect complete trace after recording terminal attempt event
    events = recorder.get_events()
    tool_call_count = sum(
        1 for e in events if e.get("event_type") == "tool_requested"
    )
    model_turn_count = sum(
        1 for e in events if e.get("event_type") == "model_requested"
    )

    token_usage = None
    estimated_cost_usd = None
    usage_status = "not_applicable"
    model_metadata = None

    if hasattr(agent, "get_metrics") and callable(agent.get_metrics):
        agent_metrics = agent.get_metrics()
        token_usage = agent_metrics.get("token_usage")
        usage_status = agent_metrics.get("usage_status", "unavailable")
        model_metadata = agent_metrics.get("model_metadata")
    elif model_turn_count > 0:
        usage_status = "unavailable"

    return {
        "attempt_id": attempt_id,
        "environment_id": environment.environment_id,
        "task_key": task.get("task_key"),
        "task_version": task.get("version", 1),
        "execution_status": execution_status,
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_ms": duration_ms,
        "agent_result": agent_result,
        "evaluations": evaluations,
        "metrics": {
            "tool_call_count": tool_call_count,
            "model_turn_count": model_turn_count,
            "token_usage": token_usage,
            "estimated_cost_usd": estimated_cost_usd,
            "usage_status": usage_status,
            "model_metadata": model_metadata,
        },
        "limits": {
            "configured": {
                "max_tool_calls": budget.max_tool_calls,
                "timeout_seconds": budget.timeout_seconds,
                "max_model_turns": budget.max_model_turns,
            },
            "enforcement": {
                "type": "cooperative",
                "description": (
                    "Enforces maximum requested tool calls, model turns, and a cooperative deadline. "
                    "The deadline is checked before and after each model request and tool call. "
                    "Cannot pre-emptively interrupt a blocked Python function or hung network call."
                ),
            },
        },
        "termination": termination,
        "initial_state": initial_state,
        "final_state": final_state,
        "events": events,
        "error": error,
    }

