"""Shared constants for the CaviDAC package."""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_VDW_RADIUS: float = 1.5
DEFAULT_GRID_RESOLUTION: float = 0.1
MAX_BOND_DISTANCE: float = 2.0
SPHERE_U_RESOLUTION: int = 30
SPHERE_V_RESOLUTION: int = 15


def get_project_path() -> Path:
    """Return the project root directory."""
    return Path(__file__).parent.parent


def get_data_path() -> Path:
    """Return the path to the data/ directory."""
    return get_project_path() / "data"
