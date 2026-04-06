"""Shared fixtures for CaviDAC tests."""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pytest

from CalixVolApp.utils.paths import get_project_path


@pytest.fixture
def project_path() -> str:
    return str(get_project_path())


@pytest.fixture
def sample_molecule_file(project_path: str) -> str:
    """Path to molecule 1 (smallest test file)."""
    return os.path.join(project_path, "CalixVolApp", "data", "molecules", "txt_calix", "1.txt")


@pytest.fixture
def vdw_radius_file(project_path: str) -> str:
    return os.path.join(project_path, "CalixVolApp", "data", "vdw", "vdw_radius.json")


@pytest.fixture
def vdw_colors_file(project_path: str) -> str:
    return os.path.join(project_path, "CalixVolApp", "data", "vdw", "vdw_colors.json")


@pytest.fixture
def tmp_molecule_file() -> str:
    """Create a temporary molecule file with a simple tetrahedron."""
    content = (
        "C  0.0  0.0  0.0\n"
        "C  2.0  0.0  0.0\n"
        "C  1.0  2.0  0.0\n"
        "C  1.0  1.0  2.0\n"
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(content)
        path = f.name
    yield path
    os.unlink(path)


@pytest.fixture
def tmp_vdw_json() -> str:
    """Create a temporary VDW radius JSON with only C."""
    data = {"C": 1.7, "O": 1.52, "H": 1.2}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        path = f.name
    yield path
    os.unlink(path)
