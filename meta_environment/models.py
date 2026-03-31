"""Backward-compatible model exports.

Canonical implementations now live in `core/models/workplace.py`.
"""

from core.models.workplace import GradeResult, WorkplaceAction, WorkplaceObservation

__all__ = ["WorkplaceObservation", "WorkplaceAction", "GradeResult"]