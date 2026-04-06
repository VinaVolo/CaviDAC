"""Tests for spatial classification utilities."""

from __future__ import annotations

import numpy as np
import pytest

from cavidac.geometry.spatial import (
    SpatialClassification,
    _classify_inside_atoms,
    classify_grid_points,
    classify_points,
)


class TestClassifyInsideAtoms:
    def test_point_inside_sphere(self) -> None:
        coords = np.array([[0.0, 0.0, 0.0]])
        radii = np.array([2.0])
        points = np.array([[0.5, 0.0, 0.0]])
        result = _classify_inside_atoms(points, coords, radii, max_radius=2.0)
        assert result[0] is np.True_

    def test_point_outside_sphere(self) -> None:
        coords = np.array([[0.0, 0.0, 0.0]])
        radii = np.array([1.0])
        points = np.array([[5.0, 5.0, 5.0]])
        result = _classify_inside_atoms(points, coords, radii, max_radius=1.0)
        assert result[0] is np.False_

    def test_point_on_boundary(self) -> None:
        coords = np.array([[0.0, 0.0, 0.0]])
        radii = np.array([1.0])
        points = np.array([[1.0, 0.0, 0.0]])
        result = _classify_inside_atoms(points, coords, radii, max_radius=1.0)
        assert result[0] is np.True_


class TestClassifyPoints:
    def test_returns_spatial_classification(self) -> None:
        coords = np.array([
            [0, 0, 0], [5, 0, 0], [2.5, 5, 0], [2.5, 2.5, 5]
        ], dtype=float)
        radii = np.array([1.7, 1.7, 1.7, 1.7])
        result = classify_points(coords, radii, num_points=1000)
        assert isinstance(result, SpatialClassification)
        assert len(result.inside_atoms_mask) == len(result.points_in_hull)

    def test_some_points_in_atoms_some_in_cavity(self) -> None:
        coords = np.array([
            [0, 0, 0], [10, 0, 0], [5, 10, 0], [5, 5, 10]
        ], dtype=float)
        radii = np.array([1.7, 1.7, 1.7, 1.7])
        result = classify_points(coords, radii, num_points=10_000)
        n_in_atoms = np.sum(result.inside_atoms_mask)
        n_in_cavity = np.sum(~result.inside_atoms_mask)
        assert n_in_atoms > 0
        assert n_in_cavity > 0


class TestClassifyGridPoints:
    def test_returns_three_floats(self) -> None:
        coords = np.array([
            [0, 0, 0], [5, 0, 0], [2.5, 5, 0], [2.5, 2.5, 5]
        ], dtype=float)
        radii = np.array([1.7, 1.7, 1.7, 1.7])
        total, atom_vol, cavity_vol = classify_grid_points(coords, radii, grid_resolution=0.5)
        assert total > 0
        assert atom_vol >= 0
        assert cavity_vol >= 0

    def test_volume_conservation(self) -> None:
        coords = np.array([
            [0, 0, 0], [5, 0, 0], [2.5, 5, 0], [2.5, 2.5, 5]
        ], dtype=float)
        radii = np.array([1.7, 1.7, 1.7, 1.7])
        total, atom_vol, cavity_vol = classify_grid_points(coords, radii, grid_resolution=0.5)
        assert total == pytest.approx(atom_vol + cavity_vol, rel=0.05)
