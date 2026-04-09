"""Tests for security hardening: rate limiting, audit logging, headers, error sanitization."""

from __future__ import annotations

import os
import sys
import time
import hashlib

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ─── Rate Limiter Tests ─────────────────────────────────────────────────────

class TestStrictRateLimiter:

    def test_sliding_window_allows_within_limit(self):
        from security.rate_limit_strict import SlidingWindowLimiter
        limiter = SlidingWindowLimiter(max_requests=5, window_seconds=60)
        for _ in range(5):
            allowed, remaining, _ = limiter.check("test_ip")
            assert allowed

    def test_sliding_window_blocks_over_limit(self):
        from security.rate_limit_strict import SlidingWindowLimiter
        limiter = SlidingWindowLimiter(max_requests=3, window_seconds=60)
        for _ in range(3):
            limiter.check("test_ip")
        allowed, remaining, retry_after = limiter.check("test_ip")
        assert not allowed
        assert remaining == 0
        assert retry_after > 0

    def test_remaining_decrements(self):
        from security.rate_limit_strict import SlidingWindowLimiter
        limiter = SlidingWindowLimiter(max_requests=5, window_seconds=60)
        _, r1, _ = limiter.check("ip1")
        _, r2, _ = limiter.check("ip1")
        assert r2 < r1

    def test_different_ips_independent(self):
        from security.rate_limit_strict import SlidingWindowLimiter
        limiter = SlidingWindowLimiter(max_requests=2, window_seconds=60)
        limiter.check("ip1")
        limiter.check("ip1")
        allowed, _, _ = limiter.check("ip2")
        assert allowed  # ip2 should not be affected by ip1

    def test_disabled_when_zero(self):
        from security.rate_limit_strict import SlidingWindowLimiter
        limiter = SlidingWindowLimiter(max_requests=0, window_seconds=60)
        for _ in range(100):
            allowed, _, _ = limiter.check("test_ip")
            assert allowed

    def test_composite_applies_endpoint_limits(self):
        from security.rate_limit_strict import (
            CompositeRateLimiter, EndpointLimit,
        )
        limiter = CompositeRateLimiter(
            endpoint_limits=[EndpointLimit("/reset", 2)],
            global_limit=1000,
        )
        limiter.check("/reset", "ip1")
        limiter.check("/reset", "ip1")
        allowed, headers = limiter.check("/reset", "ip1")
        assert not allowed
        assert "Retry-After" in headers
        assert headers.get("X-RateLimit-Scope") == "endpoint:/reset"

    def test_composite_global_limit(self):
        from security.rate_limit_strict import (
            CompositeRateLimiter, EndpointLimit,
        )
        limiter = CompositeRateLimiter(
            endpoint_limits=[],  # No per-endpoint limits
            global_limit=3,
        )
        for _ in range(3):
            limiter.check("/anything", "ip1")
        allowed, headers = limiter.check("/another", "ip1")
        assert not allowed
        assert headers.get("X-RateLimit-Scope") == "global"

    def test_composite_uses_api_key_for_global(self):
        from security.rate_limit_strict import (
            CompositeRateLimiter, EndpointLimit,
        )
        limiter = CompositeRateLimiter(
            endpoint_limits=[],
            global_limit=2,
        )
        # Same IP but different API keys = separate limits
        limiter.check("/path", "ip1", api_key="key1")
        limiter.check("/path", "ip1", api_key="key1")
        allowed_key1, _ = limiter.check("/path", "ip1", api_key="key1")
        allowed_key2, _ = limiter.check("/path", "ip1", api_key="key2")
        assert not allowed_key1  # key1 exhausted
        assert allowed_key2   # key2 still fresh

    def test_retry_after_is_positive_integer(self):
        from security.rate_limit_strict import SlidingWindowLimiter
        limiter = SlidingWindowLimiter(max_requests=1, window_seconds=30)
        limiter.check("ip1")
        _, _, retry_after = limiter.check("ip1")
        assert retry_after > 0
        assert isinstance(retry_after, (int, float))


# ─── Audit Logging Tests ────────────────────────────────────────────────────

class TestAuditLogging:

    def test_log_auth_success(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AUDIT_LOG_DIR", str(tmp_path))
        monkeypatch.setenv("AUDIT_LOG_DEST", "file")

        # Force reimport to pick up new env
        import importlib
        import security.audit_logging as al
        importlib.reload(al)

        al.log_auth_success("192.168.1.1", "req-123", "test-key", "/step")

        log_file = tmp_path / "audit.log"
        assert log_file.exists()

        import json
        content = log_file.read_text()
        entry = json.loads(content.strip().split("\n")[-1])
        assert entry["event"] == "auth.success"
        assert entry["client_ip"] == "192.168.1.1"
        assert entry["details"]["path"] == "/step"

    def test_log_auth_failure(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AUDIT_LOG_DIR", str(tmp_path))
        monkeypatch.setenv("AUDIT_LOG_DEST", "file")

        import importlib
        import security.audit_logging as al
        importlib.reload(al)

        al.log_auth_failure("10.0.0.1", "req-456", "bad-key", "/reset", "invalid_api_key")

        import json
        content = (tmp_path / "audit.log").read_text()
        entry = json.loads(content.strip().split("\n")[-1])
        assert entry["event"] == "auth.failure"
        assert entry["severity"] == "WARNING"

    def test_log_rate_limit_violation(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AUDIT_LOG_DIR", str(tmp_path))
        monkeypatch.setenv("AUDIT_LOG_DEST", "file")

        import importlib
        import security.audit_logging as al
        importlib.reload(al)

        al.log_rate_limit_violation(
            "10.0.0.5", "/step", "api-key-123",
            {"Retry-After": "30", "X-RateLimit-Scope": "endpoint:/step"},
        )

        import json
        content = (tmp_path / "audit.log").read_text()
        entry = json.loads(content.strip().split("\n")[-1])
        assert entry["event"] == "rate_limit.exceeded"
        assert entry["details"]["scope"] == "endpoint:/step"

    def test_api_key_hashed_not_plaintext(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AUDIT_LOG_DIR", str(tmp_path))
        monkeypatch.setenv("AUDIT_LOG_DEST", "file")

        import importlib
        import security.audit_logging as al
        importlib.reload(al)

        al.log_auth_success("1.2.3.4", "req-1", "my-secret-key", "/test")

        import json
        content = (tmp_path / "audit.log").read_text()
        entry = json.loads(content.strip().split("\n")[-1])
        # Key should be hashed, not plaintext
        assert "my-secret-key" not in entry.get("api_key_hash", "")
        assert entry["api_key_hash"].endswith("...")

    def test_siem_fields_present(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AUDIT_LOG_DIR", str(tmp_path))
        monkeypatch.setenv("AUDIT_LOG_DEST", "file")

        import importlib
        import security.audit_logging as al
        importlib.reload(al)

        al.log_episode_reset("1.2.3.4", "req-1", "", "E1", "ep-001")

        import json
        content = (tmp_path / "audit.log").read_text()
        entry = json.loads(content.strip().split("\n")[-1])
        assert "ddsource" in entry   # Datadog
        assert "index" in entry      # Splunk
        assert entry["source"] == "meta-environment"


# ─── Security Headers Tests ─────────────────────────────────────────────────

# SECURITY_HEADERS and MAX_REQUEST_BODY_SIZE are defined as module-level dicts/ints
# in api/middleware.py. Importing the module triggers Prometheus counter registration
# which collides in the test suite. We test via direct value verification instead.

class TestSecurityHeaders:

    @pytest.fixture(autouse=True)
    def _load_headers(self):
        """Load SECURITY_HEADERS without triggering full middleware import."""
        # Read the file and extract the dict
        import ast
        import re
        middleware_path = os.path.join(
            os.path.dirname(__file__), "..", "api", "middleware.py"
        )
        with open(middleware_path) as f:
            source = f.read()

        # Extract SECURITY_HEADERS dict
        match = re.search(r"SECURITY_HEADERS\s*=\s*(\{[^}]+\})", source)
        assert match, "SECURITY_HEADERS not found in middleware.py"
        self.headers = ast.literal_eval(match.group(1))

        # Extract MAX_REQUEST_BODY_SIZE
        match2 = re.search(r"MAX_REQUEST_BODY_SIZE\s*=\s*(.+?)(?:\s*#|$)", source, re.MULTILINE)
        assert match2, "MAX_REQUEST_BODY_SIZE not found"
        self.max_body_size = eval(match2.group(1).strip())

    def test_all_headers_defined(self):
        required = [
            "Content-Security-Policy",
            "Strict-Transport-Security",
            "X-Content-Type-Options",
            "X-Frame-Options",
            "Referrer-Policy",
        ]
        for header in required:
            assert header in self.headers, f"Missing header: {header}"

    def test_hsts_max_age(self):
        hsts = self.headers["Strict-Transport-Security"]
        assert "max-age=31536000" in hsts
        assert "includeSubDomains" in hsts

    def test_csp_restricts_default(self):
        csp = self.headers["Content-Security-Policy"]
        assert "default-src 'self'" in csp

    def test_xframe_deny(self):
        assert self.headers["X-Frame-Options"] == "DENY"

    def test_nosniff(self):
        assert self.headers["X-Content-Type-Options"] == "nosniff"


# ─── Request Size Limit Tests ───────────────────────────────────────────────

class TestRequestSizeLimit:

    def test_max_body_size_defined(self):
        # 1MB = 1 * 1024 * 1024
        import re
        middleware_path = os.path.join(
            os.path.dirname(__file__), "..", "api", "middleware.py"
        )
        with open(middleware_path) as f:
            source = f.read()
        match = re.search(r"MAX_REQUEST_BODY_SIZE\s*=\s*(.+?)(?:\s*#|$)", source, re.MULTILINE)
        assert match
        value = eval(match.group(1).strip())
        assert value == 1 * 1024 * 1024, f"Expected 1MB, got {value}"
