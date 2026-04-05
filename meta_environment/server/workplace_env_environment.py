"""Backward-compatible environment export.

Canonical implementation now lives in `environment/workplace_environment.py`.
"""

try:
	from workplace_env.environment.workplace_environment import WorkplaceEnvironment
except ImportError:  # pragma: no cover
	from environment.workplace_environment import WorkplaceEnvironment

__all__ = ["WorkplaceEnvironment"]