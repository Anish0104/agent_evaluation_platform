# Architecture Overview

This document outlines the core architectural boundaries and design principles of the AI Agent Evaluation Platform.

## Module Boundaries
- `app/api/`: HTTP API routes and request handling.
- `app/schemas/`: Pydantic data transfer objects and validation contracts.
- `app/db/`: SQLAlchemy ORM models, migrations, and database session lifecycle.
- `app/agents/`: Agent implementations (scripted fake and LLM tool loop).
- `app/providers/`: LLM provider abstractions.
- `app/environments/`: Isolated business environments (e.g. mock electronics support).
- `app/tools/`: Tool gateway and specific business tool implementations.
- `app/evaluators/`: Deterministic and LLM evaluators (outcome, policy, trajectory, response).
- `app/execution/`: Background worker, job queue, limits, and lease recovery.
- `app/telemetry/`: Event recorder, trace manager, and redaction logic.
- `app/benchmarks/`: Public benchmark integrations.
- `app/reports/`: Comparison generators and export handlers.
- `app/web/`: Web dashboard templates and static assets.
