"""Shared pytest setup for workspace-local imports."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_PARENT = Path(__file__).resolve().parents[2]

for p in [str(PROJECT_ROOT), str(PROJECT_PARENT)]:
    if p not in sys.path:
        sys.path.insert(0, p)
