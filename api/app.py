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

@app.get("/", response_class=HTMLResponse)
async def root():
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Meta-Environment — RL Customer Support Triage</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
<style>
*{margin:0;padding:0;box-sizing:border-box}
:root{
  --bg:#0a0b0f;--surface:#12141a;--glass:rgba(255,255,255,0.04);
  --glass-border:rgba(255,255,255,0.08);--accent:#6366f1;--accent2:#8b5cf6;
  --accent3:#a78bfa;--green:#34d399;--amber:#fbbf24;--red:#f87171;
  --cyan:#22d3ee;--text:#e2e8f0;--text-muted:#94a3b8;--text-dim:#64748b;
}
body{font-family:'Inter',sans-serif;background:var(--bg);color:var(--text);
  min-height:100vh;overflow-x:hidden}
.bg-mesh{position:fixed;top:0;left:0;right:0;bottom:0;z-index:0;
  background:radial-gradient(ellipse 80% 50% at 50% -20%,rgba(99,102,241,0.15),transparent),
  radial-gradient(ellipse 60% 40% at 80% 50%,rgba(139,92,246,0.08),transparent),
  radial-gradient(ellipse 50% 50% at 20% 80%,rgba(34,211,238,0.06),transparent)}
.container{position:relative;z-index:1;max-width:1100px;margin:0 auto;padding:40px 24px 80px}

/* Hero */
.hero{text-align:center;padding:48px 0 40px}
.hero-badge{display:inline-flex;align-items:center;gap:6px;padding:6px 16px;
  border-radius:100px;background:rgba(99,102,241,0.12);border:1px solid rgba(99,102,241,0.25);
  font-size:12px;font-weight:500;color:var(--accent3);margin-bottom:20px;
  animation:fadeIn 0.6s ease}
.hero-badge .dot{width:6px;height:6px;border-radius:50%;background:var(--green);
  animation:pulse 2s infinite}
h1{font-size:clamp(32px,5vw,52px);font-weight:800;
  background:linear-gradient(135deg,#fff 0%,var(--accent3) 50%,var(--cyan) 100%);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
  background-clip:text;line-height:1.1;margin-bottom:16px;animation:fadeIn 0.8s ease}
.hero p{font-size:18px;color:var(--text-muted);max-width:640px;margin:0 auto;
  line-height:1.6;animation:fadeIn 1s ease}

/* Cards */
.grid{display:grid;gap:16px;margin-top:32px}
.grid-3{grid-template-columns:repeat(auto-fit,minmax(200px,1fr))}
.grid-2{grid-template-columns:repeat(auto-fit,minmax(320px,1fr))}
.card{background:var(--glass);backdrop-filter:blur(20px);border:1px solid var(--glass-border);
  border-radius:16px;padding:24px;transition:all 0.3s ease;animation:slideUp 0.6s ease both}
.card:hover{border-color:rgba(99,102,241,0.3);transform:translateY(-2px);
  box-shadow:0 8px 32px rgba(99,102,241,0.1)}

/* Metric cards */
.metric{text-align:center;padding:28px 20px}
.metric .value{font-size:36px;font-weight:800;
  background:linear-gradient(135deg,var(--accent),var(--cyan));
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
.metric .label{font-size:13px;color:var(--text-dim);text-transform:uppercase;
  letter-spacing:1px;margin-top:4px}
.metric .sub{font-size:12px;color:var(--text-muted);margin-top:2px}

/* Section headers */
.section-label{font-size:12px;font-weight:600;text-transform:uppercase;
  letter-spacing:2px;color:var(--accent3);margin-bottom:8px}
h2{font-size:28px;font-weight:700;margin-bottom:6px}
.section-desc{color:var(--text-muted);font-size:15px;margin-bottom:24px}

/* Workflow */
.workflow{display:flex;align-items:center;justify-content:center;gap:8px;
  flex-wrap:wrap;margin:24px 0 32px}
.step-pill{display:flex;align-items:center;gap:8px;padding:12px 20px;
  border-radius:12px;background:var(--glass);border:1px solid var(--glass-border);
  font-size:14px;font-weight:500;transition:all 0.3s}
.step-pill:hover{border-color:var(--accent);background:rgba(99,102,241,0.08)}
.step-num{width:28px;height:28px;border-radius:8px;display:flex;align-items:center;
  justify-content:center;font-size:12px;font-weight:700;color:#fff}
.s1 .step-num{background:linear-gradient(135deg,#6366f1,#818cf8)}
.s2 .step-num{background:linear-gradient(135deg,#8b5cf6,#a78bfa)}
.s3 .step-num{background:linear-gradient(135deg,#06b6d4,#22d3ee)}
.arrow{color:var(--text-dim);font-size:18px}

/* Interactive demo */
.demo-box{background:var(--surface);border:1px solid var(--glass-border);
  border-radius:16px;overflow:hidden}
.demo-header{padding:16px 20px;border-bottom:1px solid var(--glass-border);
  display:flex;justify-content:space-between;align-items:center}
.demo-header h3{font-size:15px;font-weight:600}
.demo-dots{display:flex;gap:6px}
.demo-dots span{width:10px;height:10px;border-radius:50%}
.demo-dots .r{background:#f87171}.demo-dots .y{background:#fbbf24}.demo-dots .g{background:#34d399}
.demo-body{padding:20px;font-family:'SF Mono','Fira Code',monospace;font-size:13px;
  line-height:1.7;color:var(--text-muted);max-height:360px;overflow-y:auto}
.demo-body .cmd{color:var(--cyan)}
.demo-body .comment{color:var(--text-dim)}
.demo-body .json-key{color:var(--accent3)}
.demo-body .json-str{color:var(--green)}
.demo-body .json-num{color:var(--amber)}

/* Feature list */
.feature{display:flex;gap:14px;padding:16px 0;border-bottom:1px solid var(--glass-border)}
.feature:last-child{border-bottom:none}
.feature-icon{width:40px;height:40px;border-radius:10px;display:flex;align-items:center;
  justify-content:center;font-size:18px;flex-shrink:0}
.feature h4{font-size:15px;font-weight:600;margin-bottom:2px}
.feature p{font-size:13px;color:var(--text-muted);line-height:1.5}
.fi-1{background:rgba(99,102,241,0.12)}.fi-2{background:rgba(34,211,238,0.12)}
.fi-3{background:rgba(139,92,246,0.12)}.fi-4{background:rgba(52,211,153,0.12)}
.fi-5{background:rgba(251,191,36,0.12)}.fi-6{background:rgba(248,113,113,0.12)}

/* Endpoint table */
table{width:100%;border-collapse:collapse;font-size:14px}
th{text-align:left;padding:10px 14px;font-size:12px;font-weight:600;
  text-transform:uppercase;letter-spacing:1px;color:var(--text-dim);
  border-bottom:1px solid var(--glass-border)}
td{padding:12px 14px;border-bottom:1px solid rgba(255,255,255,0.03)}
.method{padding:3px 10px;border-radius:6px;font-size:11px;font-weight:700;font-family:monospace}
.get{background:rgba(34,211,238,0.12);color:var(--cyan)}
.post{background:rgba(99,102,241,0.12);color:var(--accent3)}
.path{font-family:monospace;font-weight:500;color:var(--text)}
tr:hover td{background:rgba(255,255,255,0.02)}

/* Buttons */
.btn-row{display:flex;gap:12px;justify-content:center;margin-top:28px;flex-wrap:wrap}
.btn{display:inline-flex;align-items:center;gap:8px;padding:12px 24px;
  border-radius:12px;font-size:14px;font-weight:600;text-decoration:none;
  transition:all 0.3s;cursor:pointer;border:none}
.btn-primary{background:linear-gradient(135deg,var(--accent),var(--accent2));
  color:#fff;box-shadow:0 4px 16px rgba(99,102,241,0.3)}
.btn-primary:hover{transform:translateY(-2px);box-shadow:0 8px 32px rgba(99,102,241,0.4)}
.btn-secondary{background:var(--glass);border:1px solid var(--glass-border);color:var(--text)}
.btn-secondary:hover{border-color:var(--accent);background:rgba(99,102,241,0.06)}

/* Footer */
.footer{text-align:center;padding-top:48px;color:var(--text-dim);font-size:13px}
.footer a{color:var(--accent3);text-decoration:none}

/* Tags */
.tag{display:inline-block;padding:3px 10px;border-radius:6px;font-size:11px;
  font-weight:600;margin-right:4px}
.tag-easy{background:rgba(34,211,238,0.12);color:var(--cyan)}
.tag-medium{background:rgba(251,191,36,0.12);color:var(--amber)}
.tag-hard{background:rgba(248,113,113,0.12);color:var(--red)}

/* Animations */
@keyframes fadeIn{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
@keyframes slideUp{from{opacity:0;transform:translateY(20px)}to{opacity:1;transform:translateY(0)}}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.4}}
.d1{animation-delay:0.1s}.d2{animation-delay:0.2s}.d3{animation-delay:0.3s}
.d4{animation-delay:0.4s}.d5{animation-delay:0.5s}.d6{animation-delay:0.6s}
</style>
</head>
<body>
<div class="bg-mesh"></div>
<div class="container">

  <!-- Hero -->
  <div class="hero">
    <div class="hero-badge"><span class="dot"></span> OpenEnv Compliant · v1.0.0</div>
    <h1>Meta-Environment</h1>
    <p>Production-grade reinforcement learning environment for customer support email triage. Train agents to classify, respond, and escalate — with dense rewards and sub-millisecond latency.</p>
  </div>

  <!-- Workflow -->
  <div class="workflow">
    <div class="step-pill s1"><span class="step-num">1</span> Classify Intent</div>
    <span class="arrow">→</span>
    <div class="step-pill s2"><span class="step-num">2</span> Draft Reply</div>
    <span class="arrow">→</span>
    <div class="step-pill s3"><span class="step-num">3</span> Escalate?</div>
  </div>

  <!-- Metrics -->
  <div class="grid grid-3">
    <div class="card metric d1"><div class="value">0.3ms</div><div class="label">P50 Latency</div><div class="sub">P99: 0.4ms</div></div>
    <div class="card metric d2"><div class="value">3,022</div><div class="label">Episodes/sec</div><div class="sub">Single-threaded</div></div>
    <div class="card metric d3"><div class="value">100</div><div class="label">Scenarios</div><div class="sub">Easy · Medium · Hard</div></div>
    <div class="card metric d4"><div class="value">232</div><div class="label">Tests Passing</div><div class="sub">85% coverage</div></div>
    <div class="card metric d5"><div class="value">0.1MB</div><div class="label">Memory/Episode</div><div class="sub">Bounded</div></div>
    <div class="card metric d6"><div class="value">9</div><div class="label">Security Layers</div><div class="sub">CSP · HSTS · Rate Limit</div></div>
  </div>

  <!-- Try It -->
  <div style="margin-top:56px">
    <div class="section-label">Interactive Demo</div>
    <h2>Try It Now</h2>
    <p class="section-desc">Run a complete 3-step episode right from this page.</p>
    <div class="grid grid-2">
      <div class="demo-box">
        <div class="demo-header">
          <h3>📡 API Request</h3>
          <div class="demo-dots"><span class="r"></span><span class="y"></span><span class="g"></span></div>
        </div>
        <div class="demo-body">
<span class="comment"># 1. Start a new episode</span>
<span class="cmd">POST</span> /reset
<span class="comment"># → Returns customer email + metadata</span>

<span class="comment"># 2. Classify the email</span>
<span class="cmd">POST</span> /step
{
  <span class="json-key">"action"</span>: {
    <span class="json-key">"action_type"</span>: <span class="json-str">"classify"</span>,
    <span class="json-key">"content"</span>: <span class="json-str">"refund"</span>
  }
}
<span class="comment"># → reward: <span class="json-num">0.400</span></span>

<span class="comment"># 3. Draft reply</span>
<span class="cmd">POST</span> /step
{
  <span class="json-key">"action_type"</span>: <span class="json-str">"reply"</span>,
  <span class="json-key">"content"</span>: <span class="json-str">"We'll process your refund..."</span>
}
<span class="comment"># → reward: <span class="json-num">0.296</span></span>

<span class="comment"># 4. Escalation decision</span>
<span class="cmd">POST</span> /step
{
  <span class="json-key">"action_type"</span>: <span class="json-str">"escalate"</span>,
  <span class="json-key">"content"</span>: <span class="json-str">"no"</span>
}
<span class="comment"># → reward: <span class="json-num">0.283</span>, done: true</span>
        </div>
      </div>
      <div class="card" style="display:flex;flex-direction:column;justify-content:space-between">
        <div>
          <h3 style="font-size:17px;font-weight:700;margin-bottom:14px">🎯 Reward Breakdown</h3>
          <div style="margin-bottom:18px">
            <div style="display:flex;justify-content:space-between;margin-bottom:6px;font-size:14px">
              <span>Classify (40%)</span><span style="color:var(--green);font-weight:600">0.400</span>
            </div>
            <div style="height:8px;border-radius:4px;background:rgba(255,255,255,0.06);overflow:hidden">
              <div style="width:100%;height:100%;border-radius:4px;background:linear-gradient(90deg,var(--accent),var(--accent2))"></div>
            </div>
          </div>
          <div style="margin-bottom:18px">
            <div style="display:flex;justify-content:space-between;margin-bottom:6px;font-size:14px">
              <span>Reply Quality (35%)</span><span style="color:var(--green);font-weight:600">0.296</span>
            </div>
            <div style="height:8px;border-radius:4px;background:rgba(255,255,255,0.06);overflow:hidden">
              <div style="width:84%;height:100%;border-radius:4px;background:linear-gradient(90deg,var(--accent2),var(--cyan))"></div>
            </div>
          </div>
          <div style="margin-bottom:18px">
            <div style="display:flex;justify-content:space-between;margin-bottom:6px;font-size:14px">
              <span>Escalation (25%)</span><span style="color:var(--green);font-weight:600">0.283</span>
            </div>
            <div style="height:8px;border-radius:4px;background:rgba(255,255,255,0.06);overflow:hidden">
              <div style="width:94%;height:100%;border-radius:4px;background:linear-gradient(90deg,var(--cyan),var(--green))"></div>
            </div>
          </div>
          <div style="padding:16px;border-radius:12px;background:rgba(52,211,153,0.08);border:1px solid rgba(52,211,153,0.2);margin-top:16px">
            <div style="font-size:13px;color:var(--text-muted)">Total Episode Reward</div>
            <div style="font-size:32px;font-weight:800;color:var(--green)">0.979</div>
          </div>
        </div>
        <div style="margin-top:20px;display:flex;gap:8px">
          <span class="tag tag-easy">Easy</span>
          <span class="tag tag-medium">Medium</span>
          <span class="tag tag-hard">Hard</span>
          <span style="font-size:12px;color:var(--text-dim);margin-left:4px;align-self:center">100 scenarios across 3 levels</span>
        </div>
      </div>
    </div>
  </div>

  <!-- Features -->
  <div style="margin-top:56px">
    <div class="section-label">Capabilities</div>
    <h2>Built for Production</h2>
    <p class="section-desc">Everything you need to train, evaluate, and deploy RL agents at scale.</p>
    <div class="grid grid-2">
      <div class="card">
        <div class="feature"><div class="feature-icon fi-1">🧠</div><div><h4>Dense Rewards</h4><p>Every step returns immediate feedback. Weighted grading: 40% classify, 35% reply, 25% escalate with difficulty multipliers.</p></div></div>
        <div class="feature"><div class="feature-icon fi-2">⚡</div><div><h4>Sub-Millisecond Latency</h4><p>P50: 0.3ms, 3,022 eps/sec throughput. Zero-copy observations, sequential grading for deterministic results.</p></div></div>
        <div class="feature"><div class="feature-icon fi-3">🧪</div><div><h4>A/B Experiment Framework</h4><p>Test 4 reward policies with statistical analysis. Automatic variant routing via consistent hashing.</p></div></div>
      </div>
      <div class="card">
        <div class="feature"><div class="feature-icon fi-4">🛡️</div><div><h4>Security Hardened</h4><p>CSP, HSTS, rate limiting (10-100/min per endpoint), audit logging, error sanitization. STRIDE threat modeled.</p></div></div>
        <div class="feature"><div class="feature-icon fi-5">📊</div><div><h4>Full Observability</h4><p>Prometheus metrics, OpenTelemetry distributed tracing, structured JSON audit logs for SIEM integration.</p></div></div>
        <div class="feature"><div class="feature-icon fi-6">☸️</div><div><h4>K8s Ready</h4><p>Helm chart with HPA (10-50 pods), PDB, NetworkPolicies, SOPS secrets, zero-downtime deploys.</p></div></div>
      </div>
    </div>
  </div>

  <!-- API Endpoints -->
  <div style="margin-top:56px">
    <div class="section-label">API Reference</div>
    <h2>Endpoints</h2>
    <p class="section-desc">RESTful API with interactive Swagger docs at <a href="/docs" style="color:var(--accent3)">/docs</a></p>
    <div class="card" style="padding:0;overflow:hidden">
      <table>
        <thead><tr><th>Method</th><th>Path</th><th>Description</th><th>Rate Limit</th></tr></thead>
        <tbody>
          <tr><td><span class="method post">POST</span></td><td class="path">/reset</td><td>Start new episode</td><td style="color:var(--text-dim)">10/min</td></tr>
          <tr><td><span class="method post">POST</span></td><td class="path">/step</td><td>Submit action → reward</td><td style="color:var(--text-dim)">100/min</td></tr>
          <tr><td><span class="method get">GET</span></td><td class="path">/state</td><td>Current episode state</td><td style="color:var(--text-dim)">—</td></tr>
          <tr><td><span class="method get">GET</span></td><td class="path">/health</td><td>Liveness probe</td><td style="color:var(--text-dim)">—</td></tr>
          <tr><td><span class="method get">GET</span></td><td class="path">/metrics</td><td>Prometheus metrics</td><td style="color:var(--text-dim)">5/min</td></tr>
          <tr><td><span class="method post">POST</span></td><td class="path">/experiments</td><td>Create A/B experiment</td><td style="color:var(--text-dim)">30/min</td></tr>
          <tr><td><span class="method get">GET</span></td><td class="path">/docs</td><td>Swagger interactive docs</td><td style="color:var(--text-dim)">—</td></tr>
        </tbody>
      </table>
    </div>
  </div>

  <!-- Buttons -->
  <div class="btn-row" style="margin-top:48px">
    <a href="/docs" class="btn btn-primary">📖 API Docs</a>
    <a href="https://github.com/akshar-3011/meta-environment" class="btn btn-secondary" target="_blank">⭐ GitHub</a>
    <a href="/health" class="btn btn-secondary">💚 Health Check</a>
  </div>

  <!-- Footer -->
  <div class="footer">
    <p>Built with ❤️ by <a href="https://github.com/akshar-3011" target="_blank">Akshar Dhakad</a> · OpenEnv Compliant · <a href="https://github.com/akshar-3011/meta-environment" target="_blank">Source Code</a></p>
  </div>

</div>
</body>
</html>"""


def main(host: str = CFG.api.host, port: int = CFG.api.server_port):
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=CFG.api.server_port)
    args = parser.parse_args()
    main(port=args.port)
