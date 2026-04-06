"""Tests for ConvexHullVolumeEstimator and MoleculeVolumeCalculator."""

from __future__ import annotations

import numpy as np
import pytest

from cavidac.geometry.volume import (
    ConvexHullVolumeEstimator,
    MoleculeVolumeCalculator,
)
from cavidac.io.reader import MoleculeFileReader
from cavidac.io.vdw_provider import JsonVDWRadiusProvider


class TestConvexHullVolumeEstimator:
    def setup_method(self) -> None:
        self.estimator = ConvexHullVolumeEstimator()

    def test_negative_grid_resolution_raises(self, tmp_vdw_json: str) -> None:
        provider = JsonVDWRadiusProvider(tmp_vdw_json)
        atoms = ["C", "C", "C", "C"]
        coords = np.array([[0, 0, 0], [2, 0, 0], [1, 2, 0], [1, 1, 2]], dtype=float)
        with pytest.raises(ValueError, match="positive number"):
            self.estimator.estimate(atoms, coords, provider, grid_resolution=-0.1)

    def test_zero_grid_resolution_raises(self, tmp_vdw_json: str) -> None:
        provider = JsonVDWRadiusProvider(tmp_vdw_json)
        atoms = ["C", "C", "C", "C"]
        coords = np.array([[0, 0, 0], [2, 0, 0], [1, 2, 0], [1, 1, 2]], dtype=float)
        with pytest.raises(ValueError, match="positive number"):
            self.estimator.estimate(atoms, coords, provider, grid_resolution=0)

    def test_returns_three_volumes(self, tmp_vdw_json: str) -> None:
        provider = JsonVDWRadiusProvider(tmp_vdw_json)
        atoms = ["C", "C", "C", "C"]
        coords = np.array([[0, 0, 0], [5, 0, 0], [2.5, 5, 0], [2.5, 2.5, 5]], dtype=float)
        total, atom_vol, cavity_vol = self.estimator.estimate(
            atoms, coords, provider, grid_resolution=0.5
        )
        assert total > 0
        assert atom_vol >= 0
        assert cavity_vol >= 0

    def test_total_equals_atom_plus_cavity(self, tmp_vdw_json: str) -> None:
        provider = JsonVDWRadiusProvider(tmp_vdw_json)
        atoms = ["C", "C", "C", "C"]
        coords = np.array([[0, 0, 0], [5, 0, 0], [2.5, 5, 0], [2.5, 2.5, 5]], dtype=float)
        total, atom_vol, cavity_vol = self.estimator.estimate(
            atoms, coords, provider, grid_resolution=0.5
        )
        assert total == pytest.approx(atom_vol + cavity_vol, rel=0.05)


class TestMoleculeVolumeCalculator:
    def test_calculate_real_molecule(
        self, sample_molecule_file: str, vdw_radius_file: str
    ) -> None:
        reader = MoleculeFileReader()
        provider = JsonVDWRadiusProvider(vdw_radius_file)
        estimator = ConvexHullVolumeEstimator()
        calculator = MoleculeVolumeCalculator(reader, provider, estimator)

        total, atom_vol, cavity_vol = calculator.calculate(
            sample_molecule_file, grid_resolution=0.5
        )
        assert total > 0
        assert atom_vol > 0
        assert cavity_vol >= 0
