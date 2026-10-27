# AI Agent Evaluation Platform

An evaluation and observability platform for AI agents.

## Overview

The platform runs AI agents against controlled business scenarios, captures their behavior, evaluates task completion and safety, measures latency/token usage/cost, inspects failures through traces, and compares different agent configurations.

The initial business environment is a **mock electronics retailer/customer-support system**.

## Architecture & Capabilities

- **Datasets & Versioned Tasks**: Reproducible evaluation scenarios across dev, validation, and test splits.
- **Agent Configurations & Prompts**: Track models, hyperparams, and prompt templates.
- **Isolated Business Environments**: Safe sandboxed environments (mock customer support).
- **Tool Gateway**: Intercepts, validates, and logs all tool executions before touching business state.
- **Multi-faceted Evaluators**: Outcome, policy, trajectory, and response evaluation.
- **Telemetry & Traces**: Granular event logging, model calls, tool calls, and PII redaction.
- **Execution & Workers**: Leased job execution, timeouts, budget caps, and crash recovery.
- **Reports & Dashboard**: Web dashboard for inspecting runs, traces, and agent comparisons.
- **Benchmarks**: Extensible support for standardized industry benchmarks (e.g. BFCL).

## Project Structure

Refer to `docs/architecture.md` for detailed architectural boundaries and module responsibilities.
