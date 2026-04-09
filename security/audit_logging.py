"""Security audit logging — structured event log for auth, episodes, and violations.

Emits JSON-formatted events compatible with Splunk/Datadog/CloudWatch SIEM
ingestion. All events include timestamp, request_id, client_ip, and severity.

Log destinations (controlled via AUDIT_LOG_DEST):
  - "file"    → security/audit.log (default)
  - "stdout"  → JSON to stdout (for K8s log aggregation)
  - "both"    → both file and stdout
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


# ─── Configuration ───────────────────────────────────────────────────────────

AUDIT_LOG_DIR = Path(os.getenv("AUDIT_LOG_DIR", str(Path(__file__).parent)))
AUDIT_LOG_FILE = AUDIT_LOG_DIR / "audit.log"
AUDIT_LOG_DEST = os.getenv("AUDIT_LOG_DEST", "both")  # file, stdout, both
AUDIT_LOG_LEVEL = os.getenv("AUDIT_LOG_LEVEL", "INFO")


# ─── Logger Setup ────────────────────────────────────────────────────────────

_logger = logging.getLogger("security.audit")
_logger.setLevel(getattr(logging, AUDIT_LOG_LEVEL.upper(), logging.INFO))
_logger.propagate = False

# JSON formatter for structured logging
class _JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "audit_data"):
            data.update(record.audit_data)
        return json.dumps(data, default=str)


_json_formatter = _JSONFormatter()

if AUDIT_LOG_DEST in ("file", "both"):
    AUDIT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    _file_handler = logging.FileHandler(str(AUDIT_LOG_FILE), encoding="utf-8")
    _file_handler.setFormatter(_json_formatter)
    _logger.addHandler(_file_handler)

if AUDIT_LOG_DEST in ("stdout", "both"):
    _stdout_handler = logging.StreamHandler()
    _stdout_handler.setFormatter(_json_formatter)
    _logger.addHandler(_stdout_handler)


# ─── Event Types ─────────────────────────────────────────────────────────────

class AuditEvent:
    """Structured audit event names."""
    AUTH_SUCCESS = "auth.success"
    AUTH_FAILURE = "auth.failure"
    EPISODE_RESET = "episode.reset"
    EPISODE_STEP = "episode.step"
    RATE_LIMIT_EXCEEDED = "rate_limit.exceeded"
    SECURITY_HEADER_MISSING = "security.header_missing"
    REQUEST_TOO_LARGE = "request.too_large"
    ERROR_SANITIZED = "error.sanitized"


# ─── Core Logging Functions ─────────────────────────────────────────────────

def _emit(
    event: str,
    severity: str,
    client_ip: str = "",
    request_id: str = "",
    api_key_hash: str = "",
    details: Optional[Dict[str, Any]] = None,
) -> None:
    """Emit a structured audit log event."""
    record = _logger.makeRecord(
        name="security.audit",
        level=getattr(logging, severity.upper(), logging.INFO),
        fn="",
        lno=0,
        msg=event,
        args=(),
        exc_info=None,
    )
    record.audit_data = {  # type: ignore[attr-defined]
        "event": event,
        "severity": severity.upper(),
        "client_ip": client_ip,
        "request_id": request_id,
        "api_key_hash": api_key_hash[:8] + "..." if api_key_hash else "",
        "details": details or {},
        "source": "meta-environment",
        "ddsource": "meta-environment",  # Datadog
        "index": "security",  # Splunk
    }
    _logger.handle(record)


# ─── Public API ──────────────────────────────────────────────────────────────

def log_auth_success(
    client_ip: str,
    request_id: str = "",
    api_key: str = "",
    path: str = "",
) -> None:
    """Log successful authentication."""
    import hashlib
    key_hash = hashlib.sha256(api_key.encode()).hexdigest() if api_key else ""
    _emit(
        AuditEvent.AUTH_SUCCESS,
        "INFO",
        client_ip=client_ip,
        request_id=request_id,
        api_key_hash=key_hash,
        details={"path": path},
    )


def log_auth_failure(
    client_ip: str,
    request_id: str = "",
    api_key: str = "",
    path: str = "",
    reason: str = "",
) -> None:
    """Log failed authentication attempt."""
    import hashlib
    key_hash = hashlib.sha256(api_key.encode()).hexdigest() if api_key else ""
    _emit(
        AuditEvent.AUTH_FAILURE,
        "WARNING",
        client_ip=client_ip,
        request_id=request_id,
        api_key_hash=key_hash,
        details={"path": path, "reason": reason},
    )


def log_episode_reset(
    client_ip: str,
    request_id: str = "",
    api_key: str = "",
    scenario_id: str = "",
    episode_id: str = "",
) -> None:
    """Log episode reset (who reset which episode)."""
    import hashlib
    key_hash = hashlib.sha256(api_key.encode()).hexdigest() if api_key else ""
    _emit(
        AuditEvent.EPISODE_RESET,
        "INFO",
        client_ip=client_ip,
        request_id=request_id,
        api_key_hash=key_hash,
        details={"scenario_id": scenario_id, "episode_id": episode_id},
    )


def log_episode_step(
    client_ip: str,
    request_id: str = "",
    action_type: str = "",
    step_count: int = 0,
) -> None:
    """Log episode step action."""
    _emit(
        AuditEvent.EPISODE_STEP,
        "INFO",
        client_ip=client_ip,
        request_id=request_id,
        details={"action_type": action_type, "step_count": step_count},
    )


def log_rate_limit_violation(
    client_ip: str,
    path: str,
    api_key: Optional[str] = None,
    headers: Optional[Dict[str, str]] = None,
) -> None:
    """Log rate limit violation."""
    import hashlib
    key_hash = hashlib.sha256(api_key.encode()).hexdigest() if api_key else ""
    _emit(
        AuditEvent.RATE_LIMIT_EXCEEDED,
        "WARNING",
        client_ip=client_ip,
        api_key_hash=key_hash,
        details={
            "path": path,
            "scope": (headers or {}).get("X-RateLimit-Scope", ""),
            "retry_after": (headers or {}).get("Retry-After", ""),
        },
    )


def log_request_too_large(
    client_ip: str,
    path: str,
    content_length: int,
    max_size: int,
) -> None:
    """Log oversized request rejection."""
    _emit(
        AuditEvent.REQUEST_TOO_LARGE,
        "WARNING",
        client_ip=client_ip,
        details={
            "path": path,
            "content_length": content_length,
            "max_size": max_size,
        },
    )


def log_error_sanitized(
    client_ip: str,
    path: str,
    original_status: int,
    error_type: str = "",
) -> None:
    """Log when a server error was sanitized before returning to client."""
    _emit(
        AuditEvent.ERROR_SANITIZED,
        "ERROR",
        client_ip=client_ip,
        details={
            "path": path,
            "original_status": original_status,
            "error_type": error_type,
        },
    )
