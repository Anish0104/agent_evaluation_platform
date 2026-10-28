"""Mocked unit tests for the Ollama tool-calling agent and execution runner integration."""
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import httpx
import pytest

from app.agents.ollama import (
    OllamaAgentError,
    OllamaInvalidResponseError,
    OllamaSupportAgent,
    OllamaTimeoutError,
    OllamaTruncatedResponseError,
    OllamaUnavailableError,
)
from app.execution.budget import ExecutionBudget, ExecutionLimitExceeded
from app.execution.runner import run_attempt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCENARIO_PATH = PROJECT_ROOT / "datasets/support/dev/refund_eligible_001.json"


@pytest.fixture
def task_data() -> dict[str, Any]:
    return json.loads(SCENARIO_PATH.read_text(encoding="utf-8"))


class FakeClock:
    def __init__(self, current: float = 1000.0) -> None:
        self.current = current

    def __call__(self) -> float:
        return self.current

    def advance(self, seconds: float) -> None:
        self.current += seconds


# 1. Valid multi-turn tool use and final response
def test_valid_multi_turn_tool_use_and_final_response(task_data: dict[str, Any]) -> None:
    turn = 0

    def mock_handler(request: httpx.Request) -> httpx.Response:
        nonlocal turn
        url = str(request.url)
        if "/api/version" in url:
            return httpx.Response(200, json={"version": "0.40.11"})
        if "/api/tags" in url:
            return httpx.Response(200, json={"models": [{"name": "qwen3:4b", "digest": "digest_123"}]})

        body = json.loads(request.content)
        turn += 1

        if turn == 1:
            return httpx.Response(
                200,
                json={
                    "message": {
                        "role": "assistant",
                        "content": "<think>Inspect order first</think>",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "function": {
                                    "name": "get_order",
                                    "arguments": {"order_id": "ord_1042"},
                                },
                            }
                        ],
                    },
                    "prompt_eval_count": 50,
                    "eval_count": 20,
                    "done_reason": "stop",
                },
            )
        elif turn == 2:
            return httpx.Response(
                200,
                json={
                    "message": {
                        "role": "assistant",
                        "content": "<think>Inspect return status</think>",
                        "tool_calls": [
                            {
                                "id": "call_2",
                                "function": {
                                    "name": "get_return_status",
                                    "arguments": {"order_id": "ord_1042"},
                                },
                            }
                        ],
                    },
                    "prompt_eval_count": 75,
                    "eval_count": 22,
                    "done_reason": "stop",
                },
            )
        elif turn == 3:
            return httpx.Response(
                200,
                json={
                    "message": {
                        "role": "assistant",
                        "content": "<think>Request approval</think>",
                        "tool_calls": [
                            {
                                "id": "call_3",
                                "function": {
                                    "name": "request_refund_approval",
                                    "arguments": {
                                        "order_id": "ord_1042",
                                        "amount_minor": 24999,
                                        "reason": "Return passed inspection",
                                    },
                                },
                            }
                        ],
                    },
                    "prompt_eval_count": 100,
                    "eval_count": 30,
                    "done_reason": "stop",
                },
            )
        elif turn == 4:
            # Extract approval_id dynamically from turn 3 tool response
            last_tool_msg = [m for m in body["messages"] if m.get("role") == "tool"][-1]
            approval_res = json.loads(last_tool_msg["content"])
            approval_id = approval_res["id"]
            return httpx.Response(
                200,
                json={
                    "message": {
                        "role": "assistant",
                        "content": "<think>Issue refund</think>",
                        "tool_calls": [
                            {
                                "id": "call_4",
                                "function": {
                                    "name": "issue_refund",
                                    "arguments": {
                                        "order_id": "ord_1042",
                                        "amount_minor": 24999,
                                        "approval_id": approval_id,
                                    },
                                },
                            }
                        ],
                    },
                    "prompt_eval_count": 130,
                    "eval_count": 25,
                    "done_reason": "stop",
                },
            )
        elif turn == 5:
            return httpx.Response(
                200,
                json={
                    "message": {
                        "role": "assistant",
                        "content": "<think>Update ticket</think>",
                        "tool_calls": [
                            {
                                "id": "call_5",
                                "function": {
                                    "name": "update_ticket",
                                    "arguments": {
                                        "ticket_id": "ticket_001",
                                        "status": "resolved",
                                        "note": "Refund processed.",
                                    },
                                },
                            }
                        ],
                    },
                    "prompt_eval_count": 160,
                    "eval_count": 25,
                    "done_reason": "stop",
                },
            )
        else:
            return httpx.Response(
                200,
                json={
                    "message": {
                        "role": "assistant",
                        "content": "<think>Done</think>\n\nRefund of $249.99 for order ord_1042 has been completed.",
                        "tool_calls": [],
                    },
                    "prompt_eval_count": 190,
                    "eval_count": 35,
                    "done_reason": "stop",
                },
            )

    client = httpx.Client(transport=httpx.MockTransport(mock_handler))
    agent = OllamaSupportAgent(client=client)

    report = run_attempt(task=task_data, order_id="ord_1042", agent=agent)

    assert report["execution_status"] == "completed"
    assert report["evaluations"]["outcome"]["passed"] is True
    assert report["evaluations"]["trajectory"]["passed"] is True
    assert report["metrics"]["tool_call_count"] == 5
    assert report["metrics"]["model_turn_count"] == 6
    assert report["metrics"]["usage_status"] == "complete"
    assert report["metrics"]["token_usage"] == {
        "prompt_tokens": 50 + 75 + 100 + 130 + 160 + 190,
        "completion_tokens": 20 + 22 + 30 + 25 + 25 + 35,
        "total_tokens": (50 + 75 + 100 + 130 + 160 + 190) + (20 + 22 + 30 + 25 + 25 + 35),
    }
    assert report["agent_result"]["final_response"] == "Refund of $249.99 for order ord_1042 has been completed."


# 2. Multiple tools in single turn executed sequentially
def test_multiple_tools_in_single_turn_executed_sequentially(task_data: dict[str, Any]) -> None:
    turn = 0
    executed_tools: list[str] = []

    def mock_handler(request: httpx.Request) -> httpx.Response:
        nonlocal turn
        if "/api/chat" not in str(request.url):
            return httpx.Response(200, json={})
        turn += 1
        if turn == 1:
            return httpx.Response(
                200,
                json={
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {"id": "call_1", "function": {"name": "get_order", "arguments": {"order_id": "ord_1042"}}},
                            {"id": "call_2", "function": {"name": "get_return_status", "arguments": {"order_id": "ord_1042"}}},
                        ],
                    },
                    "prompt_eval_count": 20,
                    "eval_count": 15,
                    "done_reason": "stop",
                },
            )
        else:
            body = json.loads(request.content)
            messages = body["messages"]
            tool_msgs = [m for m in messages if m.get("role") == "tool"]
            assert len(tool_msgs) == 2
            assert tool_msgs[0]["tool_call_id"] == "call_1"
            assert tool_msgs[1]["tool_call_id"] == "call_2"
            return httpx.Response(
                200,
                json={
                    "message": {"role": "assistant", "content": "Both tools checked.", "tool_calls": []},
                    "prompt_eval_count": 40,
                    "eval_count": 10,
                    "done_reason": "stop",
                },
            )

    client = httpx.Client(transport=httpx.MockTransport(mock_handler))
    agent = OllamaSupportAgent(client=client)

    report = run_attempt(task=task_data, order_id="ord_1042", agent=agent)

    assert report["execution_status"] == "completed"
    assert report["metrics"]["tool_call_count"] == 2
    assert report["metrics"]["model_turn_count"] == 2


# 3. Unknown tools and invalid arguments handled through gateway
def test_unknown_tools_and_invalid_arguments_handled_through_gateway(task_data: dict[str, Any]) -> None:
    turn = 0

    def mock_handler(request: httpx.Request) -> httpx.Response:
        nonlocal turn
        if "/api/chat" not in str(request.url):
            return httpx.Response(200, json={})
        turn += 1
        if turn == 1:
            return httpx.Response(
                200,
                json={
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {"id": "call_bad_tool", "function": {"name": "unknown_tool", "arguments": {}}},
                            {"id": "call_bad_args", "function": {"name": "get_order", "arguments": {"invalid_key": 123}}},
                        ],
                    },
                    "prompt_eval_count": 20,
                    "eval_count": 15,
                    "done_reason": "stop",
                },
            )
        else:
            body = json.loads(request.content)
            tool_msgs = [m for m in body["messages"] if m.get("role") == "tool"]
            assert len(tool_msgs) == 2
            res1 = json.loads(tool_msgs[0]["content"])
            res2 = json.loads(tool_msgs[1]["content"])
            assert res1["error"]["code"] == "UNKNOWN_TOOL"
            assert res2["error"]["code"] == "INVALID_ARGUMENTS"

            return httpx.Response(
                200,
                json={
                    "message": {"role": "assistant", "content": "I encountered errors.", "tool_calls": []},
                    "prompt_eval_count": 30,
                    "eval_count": 10,
                    "done_reason": "stop",
                },
            )

    client = httpx.Client(transport=httpx.MockTransport(mock_handler))
    agent = OllamaSupportAgent(client=client)

    report = run_attempt(task=task_data, order_id="ord_1042", agent=agent)

    assert report["execution_status"] == "completed"
    assert report["metrics"]["tool_call_count"] == 2


# 4. Gateway rejection returned to model as tool result
def test_gateway_rejection_returned_to_model(task_data: dict[str, Any]) -> None:
    # Use wrong_customer scenario where access is denied
    wrong_cust_path = PROJECT_ROOT / "datasets/support/dev/wrong_customer_001.json"
    task = json.loads(wrong_cust_path.read_text(encoding="utf-8"))

    turn = 0

    def mock_handler(request: httpx.Request) -> httpx.Response:
        nonlocal turn
        if "/api/chat" not in str(request.url):
            return httpx.Response(200, json={})
        turn += 1
        if turn == 1:
            return httpx.Response(
                200,
                json={
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {"id": "c1", "function": {"name": "get_order", "arguments": {"order_id": "ord_9001"}}}
                        ],
                    },
                    "prompt_eval_count": 20,
                    "eval_count": 10,
                    "done_reason": "stop",
                },
            )
        else:
            body = json.loads(request.content)
            tool_msg = [m for m in body["messages"] if m.get("role") == "tool"][0]
            content = json.loads(tool_msg["content"])
            assert content["error"]["code"] == "ACCESS_DENIED"

            return httpx.Response(
                200,
                json={
                    "message": {"role": "assistant", "content": "Access was denied so I must stop.", "tool_calls": []},
                    "prompt_eval_count": 30,
                    "eval_count": 10,
                    "done_reason": "stop",
                },
            )

    client = httpx.Client(transport=httpx.MockTransport(mock_handler))
    agent = OllamaSupportAgent(client=client)

    report = run_attempt(task=task, order_id="ord_9001", agent=agent)

    assert report["execution_status"] == "completed"
    assert report["evaluations"]["outcome"]["passed"] is True
    assert report["evaluations"]["trajectory"]["passed"] is True


# 5. Model turn exhaustion before extra HTTP request
def test_model_turn_exhaustion_before_extra_http_request(task_data: dict[str, Any]) -> None:
    task = deepcopy(task_data)
    task["evaluation_spec"]["limits"]["max_model_turns"] = 2

    requests_received = 0

    def mock_handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests_received
        if "/api/chat" not in str(request.url):
            return httpx.Response(200, json={})
        requests_received += 1
        return httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"id": f"call_{requests_received}", "function": {"name": "get_order", "arguments": {"order_id": "ord_1042"}}}
                    ],
                },
                "prompt_eval_count": 10,
                "eval_count": 10,
                "done_reason": "stop",
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(mock_handler))
    agent = OllamaSupportAgent(client=client)

    report = run_attempt(task=task, order_id="ord_1042", agent=agent)

    assert report["execution_status"] == "limit_exceeded"
    assert report["termination"]["limit_type"] == "model_turns"
    assert report["termination"]["code"] == "MODEL_TURN_LIMIT_EXCEEDED"
    # Exactly 2 requests allowed; 3rd blocked BEFORE sending HTTP request
    assert requests_received == 2
    assert report["metrics"]["model_turn_count"] == 2


# 6. Shared tool-call and deadline enforcement
def test_shared_tool_call_limit_enforcement(task_data: dict[str, Any]) -> None:
    task = deepcopy(task_data)
    task["evaluation_spec"]["limits"]["max_tool_calls"] = 1

    def mock_handler(request: httpx.Request) -> httpx.Response:
        if "/api/chat" not in str(request.url):
            return httpx.Response(200, json={})
        return httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"id": "call_1", "function": {"name": "get_order", "arguments": {"order_id": "ord_1042"}}},
                        {"id": "call_2", "function": {"name": "get_return_status", "arguments": {"order_id": "ord_1042"}}},
                    ],
                },
                "prompt_eval_count": 10,
                "eval_count": 10,
                "done_reason": "stop",
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(mock_handler))
    agent = OllamaSupportAgent(client=client)

    report = run_attempt(task=task, order_id="ord_1042", agent=agent)

    assert report["execution_status"] == "limit_exceeded"
    assert report["termination"]["limit_type"] == "tool_calls"
    assert report["termination"]["code"] == "TOOL_CALL_LIMIT_EXCEEDED"


def test_deadline_enforcement_with_fake_clock(task_data: dict[str, Any]) -> None:
    clock = FakeClock(100.0)

    def mock_handler(request: httpx.Request) -> httpx.Response:
        if "/api/chat" not in str(request.url):
            return httpx.Response(200, json={})
        # Advance clock beyond deadline during model execution
        clock.advance(150.0)
        return httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": "Done late",
                    "tool_calls": [],
                },
                "prompt_eval_count": 10,
                "eval_count": 10,
                "done_reason": "stop",
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(mock_handler))
    agent = OllamaSupportAgent(client=client)

    report = run_attempt(task=task_data, order_id="ord_1042", agent=agent, clock=clock)

    assert report["execution_status"] == "limit_exceeded"
    assert report["termination"]["limit_type"] == "deadline"
    assert report["termination"]["code"] == "DEADLINE_EXCEEDED"


# 7. Provider timeout and connection failure
def test_provider_timeout_and_connection_failure(task_data: dict[str, Any]) -> None:
    def timeout_handler(request: httpx.Request) -> httpx.Response:
        if "/api/chat" in str(request.url):
            raise httpx.ReadTimeout("Request timed out")
        return httpx.Response(200, json={})

    client = httpx.Client(transport=httpx.MockTransport(timeout_handler))
    agent = OllamaSupportAgent(client=client)

    report = run_attempt(task=task_data, order_id="ord_1042", agent=agent)
    assert report["execution_status"] == "error"
    assert report["error"]["code"] == "PROVIDER_TIMEOUT"

    def connect_error_handler(request: httpx.Request) -> httpx.Response:
        if "/api/chat" in str(request.url):
            raise httpx.ConnectError("Connection refused")
        return httpx.Response(200, json={})

    client2 = httpx.Client(transport=httpx.MockTransport(connect_error_handler))
    agent2 = OllamaSupportAgent(client=client2)

    report2 = run_attempt(task=task_data, order_id="ord_1042", agent=agent2)
    assert report2["execution_status"] == "error"
    assert report2["error"]["code"] == "PROVIDER_UNAVAILABLE"


# 8. Malformed, empty, and truncated responses
def test_malformed_empty_and_truncated_responses(task_data: dict[str, Any]) -> None:
    # Non-JSON
    def non_json_handler(request: httpx.Request) -> httpx.Response:
        if "/api/chat" in str(request.url):
            return httpx.Response(200, text="NOT JSON")
        return httpx.Response(200, json={})

    client = httpx.Client(transport=httpx.MockTransport(non_json_handler))
    agent = OllamaSupportAgent(client=client)
    report = run_attempt(task=task_data, order_id="ord_1042", agent=agent)
    assert report["execution_status"] == "error"
    assert report["error"]["code"] == "INVALID_PROVIDER_RESPONSE"

    # Empty content and no tools
    def empty_handler(request: httpx.Request) -> httpx.Response:
        if "/api/chat" in str(request.url):
            return httpx.Response(200, json={"message": {"role": "assistant", "content": ""}, "done_reason": "stop"})
        return httpx.Response(200, json={})

    client2 = httpx.Client(transport=httpx.MockTransport(empty_handler))
    agent2 = OllamaSupportAgent(client=client2)
    report2 = run_attempt(task=task_data, order_id="ord_1042", agent=agent2)
    assert report2["execution_status"] == "error"
    assert report2["error"]["code"] == "INVALID_PROVIDER_RESPONSE"

    # Truncated response (done_reason=length)
    def truncated_handler(request: httpx.Request) -> httpx.Response:
        if "/api/chat" in str(request.url):
            return httpx.Response(
                200,
                json={
                    "message": {"role": "assistant", "content": "I was saying something but..."},
                    "done_reason": "length",
                },
            )
        return httpx.Response(200, json={})

    client3 = httpx.Client(transport=httpx.MockTransport(truncated_handler))
    agent3 = OllamaSupportAgent(client=client3)
    report3 = run_attempt(task=task_data, order_id="ord_1042", agent=agent3)
    assert report3["execution_status"] == "error"
    assert report3["error"]["code"] == "PROVIDER_TRUNCATED"


# 9. Missing and partial token usage
def test_missing_and_partial_token_usage(task_data: dict[str, Any]) -> None:
    turn = 0

    def partial_usage_handler(request: httpx.Request) -> httpx.Response:
        nonlocal turn
        if "/api/chat" not in str(request.url):
            return httpx.Response(200, json={})
        turn += 1
        if turn == 1:
            return httpx.Response(
                200,
                json={
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {"id": "c1", "function": {"name": "get_order", "arguments": {"order_id": "ord_1042"}}}
                        ],
                    },
                    "prompt_eval_count": 50,
                    "eval_count": 25,
                    "done_reason": "stop",
                },
            )
        else:
            return httpx.Response(
                200,
                json={
                    "message": {
                        "role": "assistant",
                        "content": "Done.",
                        "tool_calls": [],
                    },
                    # Omit token usage fields
                    "done_reason": "stop",
                },
            )

    client = httpx.Client(transport=httpx.MockTransport(partial_usage_handler))
    agent = OllamaSupportAgent(client=client)
    report = run_attempt(task=task_data, order_id="ord_1042", agent=agent)

    assert report["execution_status"] == "completed"
    assert report["metrics"]["usage_status"] == "partial"
    assert report["metrics"]["token_usage"] == {
        "prompt_tokens": 50,
        "completion_tokens": 25,
        "total_tokens": 75,
    }


# 10. Correct context and history construction, and no leakage of private context
def test_context_construction_and_no_leakage(task_data: dict[str, Any]) -> None:
    recorded_requests: list[dict[str, Any]] = []

    def mock_handler(request: httpx.Request) -> httpx.Response:
        if "/api/chat" in str(request.url):
            body = json.loads(request.content)
            recorded_requests.append(body)
            return httpx.Response(
                200,
                json={
                    "message": {
                        "role": "assistant",
                        "content": "Finished.",
                        "tool_calls": [],
                    },
                    "prompt_eval_count": 20,
                    "eval_count": 10,
                    "done_reason": "stop",
                },
            )
        return httpx.Response(200, json={})

    client = httpx.Client(transport=httpx.MockTransport(mock_handler))
    agent = OllamaSupportAgent(client=client)

    report = run_attempt(task=task_data, order_id="ord_1042", agent=agent)
    assert report["execution_status"] == "completed"

    assert len(recorded_requests) == 1
    req = recorded_requests[0]
    messages = req["messages"]

    # 1. System prompt is first
    assert messages[0]["role"] == "system"
    assert "You are a customer support agent" in messages[0]["content"]

    # 2. User prompt contains customer message, order_id, ticket_id
    assert messages[1]["role"] == "user"
    assert "ord_1042" in messages[1]["content"]
    assert "ticket_001" in messages[1]["content"]
    assert "I returned order ord_1042" in messages[1]["content"]

    # 3. No leakage of environment_fixture, reviewer_behavior, expected_outcome, or authenticated_customer_id
    req_json = json.dumps(req)
    assert "environment_fixture" not in req_json
    assert "reviewer_behavior" not in req_json
    assert "expected_outcome" not in req_json
    assert "cust_12" not in req_json


# 11. Independent conversation state across attempts
def test_independent_conversation_state_across_attempts(task_data: dict[str, Any]) -> None:
    recorded_requests: list[dict[str, Any]] = []

    def mock_handler(request: httpx.Request) -> httpx.Response:
        if "/api/chat" in str(request.url):
            body = json.loads(request.content)
            recorded_requests.append(body)
            return httpx.Response(
                200,
                json={
                    "message": {
                        "role": "assistant",
                        "content": "Done.",
                        "tool_calls": [],
                    },
                    "prompt_eval_count": 10,
                    "eval_count": 5,
                    "done_reason": "stop",
                },
            )
        return httpx.Response(200, json={})

    client = httpx.Client(transport=httpx.MockTransport(mock_handler))
    agent = OllamaSupportAgent(client=client)

    # Attempt 1
    report1 = run_attempt(task=task_data, order_id="ord_1042", agent=agent)
    assert report1["execution_status"] == "completed"

    # Attempt 2
    report2 = run_attempt(task=task_data, order_id="ord_1042", agent=agent)
    assert report2["execution_status"] == "completed"

    assert len(recorded_requests) == 2
    # Both requests started with fresh 2 messages
    assert len(recorded_requests[0]["messages"]) == 2
    assert len(recorded_requests[1]["messages"]) == 2


# 12. JSON report serialization
def test_json_report_serialization(task_data: dict[str, Any]) -> None:
    def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": "Completed successfully.",
                    "tool_calls": [],
                },
                "prompt_eval_count": 10,
                "eval_count": 5,
                "done_reason": "stop",
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(mock_handler))
    agent = OllamaSupportAgent(client=client)
    report = run_attempt(task=task_data, order_id="ord_1042", agent=agent)

    # Should serialize to JSON without error
    serialized = json.dumps(report)
    deserialized = json.loads(serialized)
    assert deserialized["attempt_id"] == report["attempt_id"]
    assert deserialized["execution_status"] == "completed"


# 13. Regression: Request body serialization adheres to documented API
def test_serialized_request_body_structure(task_data: dict[str, Any]) -> None:
    captured_payloads: list[dict[str, Any]] = []

    def mock_handler(request: httpx.Request) -> httpx.Response:
        if "/api/chat" in str(request.url):
            body = json.loads(request.content)
            captured_payloads.append(body)
            return httpx.Response(
                200,
                json={
                    "message": {"role": "assistant", "content": "Done.", "tool_calls": []},
                    "prompt_eval_count": 10,
                    "eval_count": 5,
                    "done_reason": "stop",
                },
            )
        return httpx.Response(200, json={})

    client = httpx.Client(transport=httpx.MockTransport(mock_handler))
    agent = OllamaSupportAgent(client=client)
    report = run_attempt(task=task_data, order_id="ord_1042", agent=agent)

    assert report["execution_status"] == "completed"
    assert len(captured_payloads) == 1
    req = captured_payloads[0]

    # think=false is at top-level request body
    assert req.get("think") is False
    # think is NOT accidentally inside options
    assert "think" not in req.get("options", {})
    assert req.get("model") == "qwen3:4b"
    assert req.get("stream") is False

    options = req.get("options", {})
    assert options.get("temperature") == 0
    assert options.get("num_ctx") == 8192
    assert options.get("num_predict") == 1024

    # All six tools are included with valid structures
    tools = req.get("tools", [])
    assert len(tools) == 6
    tool_names = {t["function"]["name"] for t in tools}
    assert tool_names == {
        "get_order",
        "get_return_status",
        "get_refund_status",
        "request_refund_approval",
        "issue_refund",
        "update_ticket",
    }


# 14. Regression: Dedicated thinking field excluded from output and traces
def test_dedicated_thinking_field_excluded(task_data: dict[str, Any]) -> None:
    def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": "Final answer without thinking tags.",
                    "thinking": "Sensitive internal reasoning text that must never leak.",
                    "tool_calls": [],
                },
                "prompt_eval_count": 10,
                "eval_count": 5,
                "done_reason": "stop",
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(mock_handler))
    agent = OllamaSupportAgent(client=client)
    report = run_attempt(task=task_data, order_id="ord_1042", agent=agent)

    assert report["execution_status"] == "completed"
    assert report["agent_result"]["final_response"] == "Final answer without thinking tags."

    # Verify sensitive thinking text is nowhere in report or events
    report_json = json.dumps(report)
    assert "Sensitive internal reasoning" not in report_json
    assert "thinking" not in report_json or '"thinking":' not in report_json


# 15. Regression: Truncation with tool calls blocks tool dispatch
def test_truncation_with_tool_calls_blocks_tool_dispatch(task_data: dict[str, Any]) -> None:
    def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "function": {"name": "issue_refund", "arguments": {"order_id": "ord_1042", "amount_minor": 24999}},
                        }
                    ],
                },
                "prompt_eval_count": 100,
                "eval_count": 1024,
                "done_reason": "length",
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(mock_handler))
    agent = OllamaSupportAgent(client=client)
    report = run_attempt(task=task_data, order_id="ord_1042", agent=agent)

    # Must be classified as PROVIDER_TRUNCATED error
    assert report["execution_status"] == "error"
    assert report["error"]["code"] == "PROVIDER_TRUNCATED"
    # No tool was dispatched!
    assert report["metrics"]["tool_call_count"] == 0
    # Usage was preserved even though truncated
    assert report["metrics"]["usage_status"] == "complete"
    assert report["metrics"]["token_usage"]["completion_tokens"] == 1024


# 16. Regression: Unclosed thinking block with done_reason='stop' is classified as invalid
def test_unclosed_thinking_marker_classified_as_invalid(task_data: dict[str, Any]) -> None:
    def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": "<think>Thinking that never closed",
                    "tool_calls": [],
                },
                "prompt_eval_count": 10,
                "eval_count": 10,
                "done_reason": "stop",
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(mock_handler))
    agent = OllamaSupportAgent(client=client)
    report = run_attempt(task=task_data, order_id="ord_1042", agent=agent)

    assert report["execution_status"] == "error"
    assert report["error"]["code"] == "INVALID_PROVIDER_RESPONSE"

