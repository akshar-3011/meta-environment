"""FastAPI app wiring for the workplace environment."""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from ..core.config import get_config
    from ..core.logging_config import setup_logging
    from ..core.models import WorkplaceAction, WorkplaceObservation
    from ..environment import WorkplaceEnvironment
except ImportError:  # pragma: no cover
    from core.config import get_config
    from core.logging_config import setup_logging
    from core.models import WorkplaceAction, WorkplaceObservation
    from environment import WorkplaceEnvironment


setup_logging()
CFG = get_config()

app = create_app(
    WorkplaceEnvironment,
    WorkplaceAction,
    WorkplaceObservation,
    env_name="workplace_env",
    max_concurrent_envs=CFG.api.max_concurrent_envs,
)


def main(host: str = CFG.api.host, port: int = CFG.api.server_port):
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=CFG.api.server_port)
    args = parser.parse_args()
    main(port=args.port)
