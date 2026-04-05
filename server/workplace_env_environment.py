"""Backward-compatible environment export.

Canonical implementation now lives in `environment/workplace_environment.py`.
"""

from workplace_env.environment.workplace_environment import WorkplaceEnvironment

__all__ = ["WorkplaceEnvironment"]