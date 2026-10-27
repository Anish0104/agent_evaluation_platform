"""FastAPI application entrypoint and route mounting."""

from fastapi import FastAPI

app = FastAPI(
    title="AI Agent Evaluation Platform",
    description=(
        "Evaluate AI agents for correctness, safety, "
        "reliability, latency, and cost."
    ),
    version="0.1.0",
)


@app.get("/")
def root():
    return {
        "message": "AI Agent Evaluation Platform is running!",
        "status": "healthy",
        "version": "0.1.0",
    }


@app.get("/health")
def health_check():
    return {"status": "healthy"}
