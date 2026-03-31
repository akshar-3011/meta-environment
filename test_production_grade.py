"""Backward-compatible pytest shim.

Canonical tests live in `tests/` and are discovered by pytest.
"""

from tests.test_production_grade import *  # noqa: F401,F403
