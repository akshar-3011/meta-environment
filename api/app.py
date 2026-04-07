"""FastAPI app wiring for the workplace environment."""

import inspect

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

from fastapi.responses import HTMLResponse

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html><head><title>Workplace Env</title>
    <style>body{font-family:monospace;background:#0d1117;color:#e6edf3;
    padding:40px;max-width:700px;margin:auto}
    h1{color:#58a6ff}code{background:#161b22;padding:2px 6px;
    border-radius:4px}pre{background:#161b22;padding:16px;
    border-radius:8px;overflow-x:auto}
    .tag{color:#3fb950}.url{color:#f0883e}</style></head>
    <body>
    <h1>🏢 Workplace Env</h1>
    <p>OpenEnv customer-support triage environment.
    3-step episodes: <code>classify → reply → escalate</code></p>
    <h2>Endpoints</h2>
    <pre>POST /reset      Start a new episode
POST /step       Submit an action
GET  /state      Current episode state
GET  /health     Health check
GET  /docs       Interactive API docs (Swagger)</pre>
    <h2>Quick Start</h2>
    <pre>curl -X POST /reset -H "Content-Type: application/json" -d "{}"
curl -X POST /step  -H "Content-Type: application/json" \\
     -d '{"action":{"action_type":"classify","content":"complaint"}}'
    </pre>
    <h2>Tasks</h2>
    <pre><span class="tag">easy-triage</span>   Clear intent, low complexity
<span class="tag">medium-triage</span> Mixed sentiment, requires judgment
<span class="tag">hard-triage</span>   Adversarial: sarcasm, multi-intent, threats</pre>
    <p>Reward range: <code>[0.0, 1.0]</code> per episode</p>
    </body></html>
    """


def main(host: str = CFG.api.host, port: int = CFG.api.server_port):
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=CFG.api.server_port)
    args = parser.parse_args()
    main(port=args.port)
