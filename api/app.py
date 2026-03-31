"""FastAPI app wiring for the workplace environment."""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from ..core.models import WorkplaceAction, WorkplaceObservation
    from ..environment import WorkplaceEnvironment
except ImportError:  # pragma: no cover
    from core.models import WorkplaceAction, WorkplaceObservation
    from environment import WorkplaceEnvironment


app = create_app(
    WorkplaceEnvironment,
    WorkplaceAction,
    WorkplaceObservation,
    env_name="workplace_env",
    max_concurrent_envs=1,
)


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
