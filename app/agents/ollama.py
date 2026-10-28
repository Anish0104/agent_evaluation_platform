"""Ollama-based LLM agent using local tool-calling endpoint."""
import json
import re
from typing import Any
from uuid import uuid4

import httpx

from app.agents.prompts import (
    PROMPT_HASH_V1,
    PROMPT_VERSION_V1,
    SUPPORT_AGENT_SYSTEM_PROMPT_V1,
)
from app.execution.budget import ExecutionLimitExceeded
from app.tools.definitions import get_support_tool_definitions
from app.tools.gateway import ToolGateway


class OllamaAgentError(Exception):
    """Base error for Ollama agent execution failures."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class OllamaUnavailableError(OllamaAgentError):
    """Raised when the Ollama server is unreachable."""

    def __init__(self, message: str = "Ollama server is unavailable.") -> None:
        super().__init__("PROVIDER_UNAVAILABLE", message)


class OllamaTimeoutError(OllamaAgentError):
    """Raised when an Ollama request times out before the attempt deadline."""

    def __init__(self, message: str = "Ollama request timed out before deadline.") -> None:
        super().__init__("PROVIDER_TIMEOUT", message)


class OllamaInvalidResponseError(OllamaAgentError):
    """Raised when the Ollama response structure is malformed or invalid."""

    def __init__(self, message: str = "Malformed or invalid response from Ollama.") -> None:
        super().__init__("INVALID_PROVIDER_RESPONSE", message)


class OllamaTruncatedResponseError(OllamaAgentError):
    """Raised when provider output is truncated due to length limits."""

    def __init__(self, message: str = "Model generation truncated due to token limit.") -> None:
        super().__init__("PROVIDER_TRUNCATED", message)


class OllamaSupportAgent:
    """Customer support agent powered by a local Ollama model using tool calling."""

    def __init__(
        self,
        *,
        model: str = "qwen3:4b",
        base_url: str = "http://127.0.0.1:11434",
        user_message: str | None = None,
        provider_timeout_seconds: float = 60.0,
        client: httpx.Client | None = None,
        system_prompt: str = SUPPORT_AGENT_SYSTEM_PROMPT_V1,
        prompt_version: str = PROMPT_VERSION_V1,
        prompt_hash: str = PROMPT_HASH_V1,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.user_message = user_message
        self.provider_timeout_seconds = float(provider_timeout_seconds)
        self._external_client = client
        self.system_prompt = system_prompt
        self.prompt_version = prompt_version
        self.prompt_hash = prompt_hash
        self.tool_definitions = get_support_tool_definitions()

        # Telemetry metrics accumulated during execution
        self._model_turns: int = 0
        self._total_prompt_tokens: int = 0
        self._total_completion_tokens: int = 0
        self._has_prompt_tokens: bool = False
        self._has_completion_tokens: bool = False
        self._missing_usage_count: int = 0
        self._server_version: str | None = None
        self._model_digest: str | None = None
        self._metadata_fetched: bool = False

    def set_user_message(self, user_message: str | None) -> None:
        self.user_message = user_message

    def _fetch_metadata_if_needed(self, client: httpx.Client) -> None:
        if self._metadata_fetched:
            return
        self._metadata_fetched = True

        # Fetch server version
        try:
            resp = client.get(f"{self.base_url}/api/version", timeout=3.0)
            if resp.status_code == 200:
                self._server_version = resp.json().get("version")
        except Exception:
            self._server_version = None

        # Fetch model digest
        try:
            resp = client.get(f"{self.base_url}/api/tags", timeout=3.0)
            if resp.status_code == 200:
                models = resp.json().get("models", [])
                for m in models:
                    name = m.get("name", "")
                    if name == self.model or name.startswith(f"{self.model}:"):
                        self._model_digest = m.get("digest")
                        break
        except Exception:
            self._model_digest = None

    def get_metrics(self) -> dict[str, Any]:
        """Return token usage, status, and model metadata for reporting."""
        if self._model_turns == 0:
            usage_status = "unavailable"
            token_usage = None
        elif self._missing_usage_count == 0 and (self._has_prompt_tokens or self._has_completion_tokens):
            usage_status = "complete"
            token_usage = {
                "prompt_tokens": self._total_prompt_tokens,
                "completion_tokens": self._total_completion_tokens,
                "total_tokens": self._total_prompt_tokens + self._total_completion_tokens,
            }
        elif self._has_prompt_tokens or self._has_completion_tokens:
            usage_status = "partial"
            token_usage = {
                "prompt_tokens": self._total_prompt_tokens,
                "completion_tokens": self._total_completion_tokens,
                "total_tokens": self._total_prompt_tokens + self._total_completion_tokens,
            }
        else:
            usage_status = "unavailable"
            token_usage = None

        return {
            "model_turn_count": self._model_turns,
            "token_usage": token_usage,
            "usage_status": usage_status,
            "estimated_cost_usd": None,
            "model_metadata": {
                "provider": "ollama",
                "model": self.model,
                "model_digest": self._model_digest,
                "server_version": self._server_version,
                "settings": {
                    "temperature": 0,
                    "num_ctx": 8192,
                    "num_predict": 1024,
                    "stream": False,
                    "think": False,
                },
                "prompt_version": self.prompt_version,
                "prompt_hash": self.prompt_hash,
            },
        }

    @staticmethod
    def _process_content_and_thinking(
        raw_content: str,
        has_separate_thinking: bool,
    ) -> tuple[str, bool]:
        """Extract clean content and detect incomplete thinking blocks.

        Returns:
            (clean_content, is_incomplete_thinking)
        """
        if not raw_content:
            return "", False

        # If Ollama delivered thinking in a separate field, raw_content is pure content
        if has_separate_thinking:
            if "<think>" in raw_content or "</think>" in raw_content:
                parts = raw_content.split("</think>")
                clean = parts[-1].strip()
                clean = re.sub(r"<think>.*?</think>", "", clean, flags=re.DOTALL).strip()
                is_incomplete = "<think>" in clean
                return ("" if is_incomplete else clean), is_incomplete
            return raw_content.strip(), False

        # If inline thinking markers exist:
        if "</think>" in raw_content:
            parts = raw_content.split("</think>")
            clean = parts[-1].strip()
            clean = re.sub(r"<think>.*?</think>", "", clean, flags=re.DOTALL).strip()
            is_incomplete = "<think>" in clean
            return ("" if is_incomplete else clean), is_incomplete
        elif "<think>" in raw_content:
            # Opened but never closed -> incomplete thinking block
            return "", True
        else:
            return raw_content.strip(), False

    @staticmethod
    def _strip_thinking(content: str) -> str:
        """Strip hidden thinking tags and contents emitted by the model."""
        clean, _ = OllamaSupportAgent._process_content_and_thinking(content, False)
        return clean

    def run(
        self,
        *,
        gateway: ToolGateway,
        order_id: str,
        ticket_id: str,
        user_message: str | None = None,
    ) -> dict[str, Any]:
        """Execute the multi-turn agent tool-calling loop."""
        effective_user_message = user_message or self.user_message or ""
        budget = gateway.budget
        recorder = gateway.recorder

        # Build initial messages strictly adhering to input boundary
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": (
                    f"Customer Message: {effective_user_message}\n"
                    f"Order ID: {order_id}\n"
                    f"Ticket ID: {ticket_id}"
                ),
            },
        ]

        client_to_use = self._external_client or httpx.Client()
        owns_client = self._external_client is None

        try:
            self._fetch_metadata_if_needed(client_to_use)

            while True:
                # 1. Budget enforcement before model request
                if budget is not None:
                    budget.check_and_consume_model_turn()
                    remaining = budget.remaining_time()
                    if remaining <= 0:
                        budget.check_deadline()
                    request_timeout = min(self.provider_timeout_seconds, remaining)
                else:
                    request_timeout = self.provider_timeout_seconds

                self._model_turns += 1
                model_call_id = str(uuid4())

                # 2. Record model_requested event
                recorder.record(
                    event_type="model_requested",
                    model_call_id=model_call_id,
                    payload={
                        "model": self.model,
                        "messages_count": len(messages),
                        "settings": {
                            "temperature": 0,
                            "num_ctx": 8192,
                            "num_predict": 1024,
                            "stream": False,
                            "think": False,
                        },
                    },
                )

                # 3. Dispatch HTTP request to Ollama
                request_payload = {
                    "model": self.model,
                    "stream": False,
                    "think": False,
                    "options": {
                        "temperature": 0,
                        "num_ctx": 8192,
                        "num_predict": 1024,
                    },
                    "messages": messages,
                    "tools": self.tool_definitions,
                }

                started_at = gateway.clock()
                try:
                    resp = client_to_use.post(
                        f"{self.base_url}/api/chat",
                        json=request_payload,
                        timeout=request_timeout,
                    )
                except httpx.TimeoutException:
                    self._missing_usage_count += 1
                    recorder.record(
                        event_type="model_failed",
                        model_call_id=model_call_id,
                        payload={
                            "model": self.model,
                            "error": {
                                "code": "PROVIDER_TIMEOUT",
                                "message": "Provider request timed out.",
                            },
                        },
                    )
                    # Distinguish deadline vs provider timeout
                    if budget is not None:
                        budget.check_deadline()
                    raise OllamaTimeoutError("Ollama request timed out before attempt deadline.")
                except (httpx.ConnectError, httpx.NetworkError) as exc:
                    self._missing_usage_count += 1
                    recorder.record(
                        event_type="model_failed",
                        model_call_id=model_call_id,
                        payload={
                            "model": self.model,
                            "error": {
                                "code": "PROVIDER_UNAVAILABLE",
                                "message": "Provider connection failed.",
                            },
                        },
                    )
                    raise OllamaUnavailableError("Ollama server is unavailable or connection refused.")
                except Exception as exc:
                    self._missing_usage_count += 1
                    recorder.record(
                        event_type="model_failed",
                        model_call_id=model_call_id,
                        payload={
                            "model": self.model,
                            "error": {
                                "code": "PROVIDER_ERROR",
                                "message": "Unexpected provider communication error.",
                            },
                        },
                    )
                    raise OllamaAgentError("PROVIDER_ERROR", "Unexpected provider communication error.")

                duration_ms = round((gateway.clock() - started_at) * 1000, 2)

                if resp.status_code != 200:
                    self._missing_usage_count += 1
                    recorder.record(
                        event_type="model_failed",
                        model_call_id=model_call_id,
                        payload={
                            "model": self.model,
                            "error": {
                                "code": "PROVIDER_ERROR",
                                "message": f"Provider HTTP {resp.status_code}.",
                            },
                        },
                    )
                    raise OllamaAgentError(
                        "PROVIDER_ERROR",
                        f"Ollama server returned error status {resp.status_code}.",
                    )

                try:
                    data = resp.json()
                except Exception:
                    self._missing_usage_count += 1
                    recorder.record(
                        event_type="model_failed",
                        model_call_id=model_call_id,
                        payload={
                            "model": self.model,
                            "error": {
                                "code": "INVALID_PROVIDER_RESPONSE",
                                "message": "Provider response was not valid JSON.",
                            },
                        },
                    )
                    raise OllamaInvalidResponseError("Provider response was not valid JSON.")

                # 4. Extract token usage evidence immediately
                prompt_eval_count = data.get("prompt_eval_count")
                eval_count = data.get("eval_count")
                if isinstance(prompt_eval_count, int) and isinstance(eval_count, int):
                    self._total_prompt_tokens += prompt_eval_count
                    self._total_completion_tokens += eval_count
                    self._has_prompt_tokens = True
                    self._has_completion_tokens = True
                else:
                    self._missing_usage_count += 1

                # 5. Extract message and tool calls
                msg_data = data.get("message")
                if not isinstance(msg_data, dict):
                    recorder.record(
                        event_type="model_failed",
                        model_call_id=model_call_id,
                        payload={
                            "model": self.model,
                            "error": {
                                "code": "INVALID_PROVIDER_RESPONSE",
                                "message": "Missing or invalid 'message' in response.",
                            },
                        },
                    )
                    raise OllamaInvalidResponseError("Missing or invalid 'message' in response.")

                raw_content = msg_data.get("content", "") or ""
                has_separate_thinking = msg_data.get("thinking") is not None
                clean_content, is_incomplete_thinking = self._process_content_and_thinking(
                    raw_content, has_separate_thinking
                )
                tool_calls = msg_data.get("tool_calls") or []

                # Record model_responded
                recorder.record(
                    event_type="model_responded",
                    model_call_id=model_call_id,
                    payload={
                        "model": self.model,
                        "duration_ms": duration_ms,
                        "tool_calls_count": len(tool_calls),
                        "has_content": bool(clean_content),
                        "done_reason": data.get("done_reason"),
                        "usage": {
                            "prompt_eval_count": prompt_eval_count,
                            "eval_count": eval_count,
                        },
                    },
                )

                # 6. Check deadline after response received before dispatching tools
                if budget is not None:
                    budget.check_deadline()

                # 7. Check truncation (blocks tool dispatch if truncated)
                if data.get("done_reason") == "length":
                    raise OllamaTruncatedResponseError(
                        "Model generation truncated due to token limit (done_reason=length)."
                    )

                # If not truncated but thinking markers are incomplete, reject as invalid
                if is_incomplete_thinking:
                    raise OllamaInvalidResponseError(
                        "Model response contains an unclosed thinking block."
                    )

                # 8. If no tool calls, validate final response
                if not tool_calls:
                    if not clean_content:
                        raise OllamaInvalidResponseError(
                            "Model returned empty content without any tool calls."
                        )
                    return {
                        "status": "completed",
                        "final_response": clean_content,
                    }

                # 9. Handle tool calls sequentially
                assistant_message: dict[str, Any] = {
                    "role": "assistant",
                    "content": clean_content,
                    "tool_calls": tool_calls,
                }
                messages.append(assistant_message)

                for tc in tool_calls:
                    if not isinstance(tc, dict):
                        raise OllamaInvalidResponseError("Malformed tool call object.")
                    fn = tc.get("function")
                    if not isinstance(fn, dict):
                        raise OllamaInvalidResponseError("Malformed tool function object.")

                    tool_name = fn.get("name")
                    arguments = fn.get("arguments", {})
                    tc_id = tc.get("id")

                    if not isinstance(tool_name, str) or not isinstance(arguments, dict):
                        raise OllamaInvalidResponseError("Invalid tool name or arguments.")

                    call_result = gateway.call(tool_name, arguments)

                    # Return rejection or success to model as tool response
                    tool_content = (
                        call_result["result"]
                        if call_result.get("ok")
                        else {"error": call_result.get("error")}
                    )

                    tool_message: dict[str, Any] = {
                        "role": "tool",
                        "content": json.dumps(tool_content),
                    }
                    if tc_id:
                        tool_message["tool_call_id"] = tc_id
                    if tool_name:
                        tool_message["name"] = tool_name

                    messages.append(tool_message)

        finally:
            if owns_client:
                client_to_use.close()

