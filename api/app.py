"""FastAPI app wiring for the workplace environment."""

import inspect
from pathlib import Path

try:
    from openenv.core.env_server.http_server import create_app
except Exception:  # pragma: no cover
    def create_app(*args, **kwargs):
        raise RuntimeError("openenv-core>=0.2.2 is required")

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

try:
    from ..api.middleware import apply_production_middleware
except ImportError:  # pragma: no cover
    from api.middleware import apply_production_middleware

setup_logging()
CFG = get_config()

_sig = inspect.signature(create_app)
_kwargs = dict(
    env_name="workplace_env",
    max_concurrent_envs=CFG.api.max_concurrent_envs,
)
_args = [WorkplaceEnvironment, WorkplaceAction, WorkplaceObservation]
_accepted = set(_sig.parameters.keys())
_kwargs = {k: v for k, v in _kwargs.items() if k in _accepted}
app = create_app(*_args, **_kwargs)

app.title = "Workplace Env — Customer Support Triage"
app.description = "OpenEnv environment for 3-step support workflow"
app.version = "1.0.0"

# Apply production middleware (API key, CORS, rate limiting, /metrics)
apply_production_middleware(app)

from fastapi.responses import HTMLResponse

_UI_TEMPLATE_PATH = Path(__file__).parent / "ui_template.html"

@app.get("/", response_class=HTMLResponse)
async def root():
    try:
        return _UI_TEMPLATE_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        return HTMLResponse("<h1>Meta-Environment</h1><p>UI template not found. Visit <a href='/docs'>/docs</a> for API.</p>")


def main(host: str = CFG.api.host, port: int = CFG.api.server_port):
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=CFG.api.server_port)
    args = parser.parse_args()
    main(port=args.port)
