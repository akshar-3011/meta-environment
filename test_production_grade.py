"""Backward-compatible test launcher.

Canonical test module now lives in `tests/test_production_grade.py`.
"""

from tests.test_production_grade import run_all_tests


if __name__ == "__main__":
    raise SystemExit(run_all_tests())
