"""Custom exception hierarchy for production-ready error handling."""

from __future__ import annotations

from typing import Any, Dict, Optional


class WorkplaceEnvError(Exception):
    """Base project exception with machine-readable code/details."""

    def __init__(self, message: str, *, code: str = "WORKPLACE_ENV_ERROR", details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.code = code
        self.details = details or {}


class ConfigurationError(WorkplaceEnvError):
    def __init__(self, message: str, *, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, code="CONFIGURATION_ERROR", details=details)


class InferenceError(WorkplaceEnvError):
    def __init__(self, message: str, *, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, code="INFERENCE_ERROR", details=details)


class GradingError(WorkplaceEnvError):
    def __init__(self, message: str, *, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, code="GRADING_ERROR", details=details)


class PipelineError(WorkplaceEnvError):
    def __init__(self, message: str, *, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, code="PIPELINE_ERROR", details=details)
