"""
FastAPI application for the Codebase Refactoring RL Environment.

Endpoints (provided by OpenEnv's create_app helper):
    POST /reset   — Reset the environment, get initial observation
    POST /step    — Submit an action, receive observation + reward
    GET  /state   — Retrieve current episode metadata
    GET  /schema  — JSON schemas for Action and Observation
    WS   /ws      — Persistent WebSocket session

Run locally (development):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

Run in Docker:
    docker build -t code-refactor-env .
    docker run -p 8000:8000 code-refactor-env
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required. Install with: uv sync"
    ) from e

try:
    from ..models import CodeRefactorAction, CodeRefactorObservation
    from .rlproj_environment import CodeRefactorEnvironment
except ModuleNotFoundError:
    from models import CodeRefactorAction, CodeRefactorObservation
    from server.rlproj_environment import CodeRefactorEnvironment


# ---------------------------------------------------------------------------
# Create the FastAPI application
# ---------------------------------------------------------------------------

app = create_app(
    CodeRefactorEnvironment,
    CodeRefactorAction,
    CodeRefactorObservation,
    env_name="code-refactor",
    # Increase this number to allow many concurrent agents / WebSocket sessions
    max_concurrent_envs=4,
)


# ---------------------------------------------------------------------------
# Entry point for direct / uv-run execution
# ---------------------------------------------------------------------------

def main(host: str = "0.0.0.0", port: int = 8000):
    """
    Start the uvicorn server.

    Usage:
        uv run --project . server
        uv run --project . server --port 8001
        python -m rlproj.server.app
    """
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Codebase Refactoring RL Environment Server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(host=args.host, port=args.port)
