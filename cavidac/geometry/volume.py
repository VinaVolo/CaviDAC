"""Volume estimation interfaces and implementations."""

from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod

from cavidac.io.reader import IMoleculeReader
from cavidac.io.vdw_provider import IVDWRadiusProvider
from cavidac.geometry.spatial import classify_grid_points


class IVolumeEstimator(ABC):
    @abstractmethod
    def estimate(
        self,
        atoms: list[str],
        coordinates: np.ndarray,
        vdw_provider: IVDWRadiusProvider,
        grid_resolution: float = 0.1,
    ) -> tuple[float, float, float]:
        """
        Estimates the volume of a molecule.

        Returns
        -------
        tuple[float, float, float]
            (total_volume, atomic_volume, cavity_volume)

        Raises
        ------
        ValueError
            If grid_resolution is not positive.
        """
        pass


class ConvexHullVolumeEstimator(IVolumeEstimator):
    def estimate(
        self,
        atoms: list[str],
        coordinates: np.ndarray,
        vdw_provider: IVDWRadiusProvider,
        grid_resolution: float = 0.1,
    ) -> tuple[float, float, float]:
        """Estimate volumes using convex hull and grid-based classification."""
        if grid_resolution <= 0:
            raise ValueError("The grid_resolution parameter must be a positive number.")

        radii = np.array([vdw_provider.get_radius(atom) for atom in atoms])
        return classify_grid_points(coordinates, radii, grid_resolution)


class MoleculeVolumeCalculator:
    """Orchestrates reading, VDW lookup, and volume estimation."""

    def __init__(
        self,
        reader: IMoleculeReader,
        vdw_provider: IVDWRadiusProvider,
        estimator: IVolumeEstimator,
    ) -> None:
        self.reader = reader
        self.vdw_provider = vdw_provider
        self.estimator = estimator

    def calculate(
        self, molecule_file_path: str, grid_resolution: float = 0.1
    ) -> tuple[float, float, float]:
        """Calculate volumes for a molecule file."""
        atoms, coords = self.reader.read(molecule_file_path)
        return self.estimator.estimate(atoms, coords, self.vdw_provider, grid_resolution)

    def calculate_with_data(
        self, molecule_file_path: str, grid_resolution: float = 0.1
    ) -> tuple[list[str], np.ndarray, tuple[float, float, float]]:
        """Calculate volumes and return parsed molecule data alongside results."""
        atoms, coords = self.reader.read(molecule_file_path)
        volumes = self.estimator.estimate(atoms, coords, self.vdw_provider, grid_resolution)
        return atoms, coords, volumes
