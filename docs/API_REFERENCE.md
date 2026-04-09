# API Reference

> Complete endpoint documentation for meta-environment v1.0.0.

Base URL: `http://localhost:8000`

---

## Authentication

All state-mutating endpoints require an API key when `API_KEY` is set:

```bash
# Header authentication
curl -H "X-API-Key: your-key" http://localhost:8000/reset

# Bearer token (alternative)
curl -H "Authorization: Bearer your-key" http://localhost:8000/reset
```

Exempt paths: `/health`, `/docs`, `/openapi.json`, `/redoc`, `/metrics`, `/`

---

## Endpoints

### `POST /reset`

Start a new episode. Returns the initial observation with a customer email.

**Request Body:** `{}` (empty JSON object)

**Response:**
```json
{
  "observation": {
    "email": "Dear Support Team, I purchased a laptop last week...",
    "category_options": ["refund", "complaint", "query"],
    "history": [],
    "reward": null,
    "done": false,
    "scenario_difficulty": "easy",
    "urgency": "medium",
    "sentiment": "negative",
    "complexity_score": 2,
    "scenario_metadata": {
      "min_reply_length": 30
    }
  }
}
```

**Rate Limit:** 10/min per IP

---

### `POST /step`

Submit an action for the current episode step.

**Request Body:**
```json
{
  "action": {
    "action_type": "classify",
    "content": "refund"
  }
}
```

| Field | Type | Required | Values |
|---|---|---|---|
| `action_type` | `string` | ✅ | `"classify"`, `"reply"`, `"escalate"` |
| `content` | `string` | ✅ | Category label, reply text, or `"yes"`/`"no"` |

**Response:**
```json
{
  "observation": {
    "email": "Dear Support Team, I purchased a laptop last week...",
    "category_options": ["refund", "complaint", "query"],
    "history": ["classify:refund"],
    "reward": 0.4,
    "done": false,
    "scenario_difficulty": "easy",
    "urgency": "medium",
    "sentiment": "negative",
    "complexity_score": 2,
    "scenario_metadata": {
      "min_reply_length": 30
    }
  }
}
```

**Rate Limit:** 100/min per IP

**Notes:**
- Step 1 must be `classify`, step 2 must be `reply`, step 3 must be `escalate`
- Episode ends (`done: true`) after the escalate step
- Reward range: `[0.0, 1.0]` per step (weighted by step type)

---

### `GET /state`

Get the current episode state.

**Response:**
```json
{
  "step_count": 1,
  "rewards": [0.4],
  "history": ["classify:refund"],
  "done": false,
  "scenario_id": "E1"
}
```

---

### `GET /health`

Liveness probe. Always returns 200 if the server is running.

**Response:**
```json
{
  "status": "ok"
}
```

---

### `GET /metrics`

Prometheus metrics in exposition format.

**Response:** (text/plain)
```
# HELP env_requests_total Total API requests
# TYPE env_requests_total counter
env_requests_total{endpoint="/step",method="POST",status="200"} 42.0
# HELP env_request_latency_seconds Request latency
# TYPE env_request_latency_seconds histogram
env_request_latency_seconds_bucket{endpoint="/step",le="0.01"} 40.0
```

**Rate Limit:** 5/min per IP

---

### `POST /experiments`

Create a new A/B experiment for reward policy testing.

**Request Body:**
```json
{
  "name": "test-equal-weights",
  "policy_type": "equal",
  "traffic_split": 0.2,
  "target_scenarios": ["E1", "E2", "E3"]
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `name` | `string` | ✅ | Experiment name |
| `policy_type` | `string` | ✅ | `"equal"`, `"escalation_first"`, `"reply_quality"` |
| `traffic_split` | `float` | ✅ | Fraction of traffic for variant (0.0–1.0) |
| `target_scenarios` | `string[]` | ❌ | Limit to specific scenarios |

**Response:**
```json
{
  "id": "exp_abc123",
  "name": "test-equal-weights",
  "policy_type": "equal",
  "traffic_split": 0.2,
  "status": "active",
  "created_at": "2026-04-10T00:00:00Z",
  "metrics": {
    "control": {"count": 0, "mean_reward": 0.0},
    "variant": {"count": 0, "mean_reward": 0.0}
  }
}
```

**Constraints:** Maximum 2 concurrent active experiments.

---

### `GET /experiments/{id}`

Get experiment status and metrics.

**Response:**
```json
{
  "id": "exp_abc123",
  "name": "test-equal-weights",
  "status": "active",
  "metrics": {
    "control": {"count": 150, "mean_reward": 0.85, "std_reward": 0.05},
    "variant": {"count": 38, "mean_reward": 0.82, "std_reward": 0.06}
  }
}
```

---

## Error Responses

All errors return JSON with a consistent structure:

```json
{
  "error": "Rate limit exceeded",
  "retry_after": 30,
  "scope": "endpoint:/reset"
}
```

| Status | Meaning | Common Causes |
|---|---|---|
| `401` | Unauthorized | Missing or invalid API key |
| `413` | Payload Too Large | Request body > 1MB |
| `422` | Validation Error | Invalid action type or missing fields |
| `429` | Rate Limited | Too many requests (check `Retry-After` header) |
| `500` | Internal Error | Server error (request_id included for debugging) |

---

## Rate Limits

| Endpoint | Limit | Scope |
|---|---|---|
| `/reset` | 10/min | Per IP |
| `/step` | 100/min | Per IP |
| `/infer` | 30/min | Per IP |
| `/metrics` | 5/min | Per IP |
| Global | 1000/min | Per API key |

Rate limit headers on every response:
- `X-RateLimit-Limit`: Max requests in window
- `X-RateLimit-Remaining`: Remaining requests
- `Retry-After`: Seconds until rate limit resets (on 429)

---

## Response Headers

Every response includes security headers:

| Header | Value |
|---|---|
| `X-Request-ID` | Unique request identifier for tracing |
| `Content-Security-Policy` | `default-src 'self'` |
| `Strict-Transport-Security` | `max-age=31536000; includeSubDomains` |
| `X-Content-Type-Options` | `nosniff` |
| `X-Frame-Options` | `DENY` |
