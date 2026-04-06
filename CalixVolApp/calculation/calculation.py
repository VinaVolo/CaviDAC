"""Molecular volume calculation using convex hull and grid-based estimation."""

from __future__ import annotations

import json
import os

import numpy as np
from abc import ABC, abstractmethod

from CalixVolApp.calculation.spatial import (
    DEFAULT_GRID_RESOLUTION,
    DEFAULT_VDW_RADIUS,
    classify_grid_points,
)
from CalixVolApp.utils.paths import get_project_path


class IMoleculeReader(ABC):
    @abstractmethod
    def read(self, file_path: str) -> tuple[list[str], np.ndarray]:
        """
        Reads atomic symbols and coordinates from a file.

        Parameters
        ----------
        file_path : str
            Path to the file containing atomic symbols and coordinates.

        Returns
        -------
        tuple[list[str], np.ndarray]
            Tuple of atomic symbols and (n_atoms, 3) coordinate array.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        """
        pass


class IVDWRadiusProvider(ABC):
    @abstractmethod
    def get_radius(self, atom: str) -> float:
        """
        Returns the van der Waals radius for the given atom.

        Parameters
        ----------
        atom : str
            Atomic symbol

        Returns
        -------
        float
            Van der Waals radius in Angstroms

        Raises
        ------
        KeyError
            If the atom is not found in the database.
        """
        pass


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

        Parameters
        ----------
        atoms : list[str]
            List of atomic symbols.
        coordinates : np.ndarray
            Array of shape (n_atoms, 3) containing coordinates of atoms.
        vdw_provider : IVDWRadiusProvider
            Provider of van der Waals radii.
        grid_resolution : float, optional
            Resolution of the grid for estimation, by default 0.1

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


class MoleculeFileReader(IMoleculeReader):
    def read(self, file_path: str) -> tuple[list[str], np.ndarray]:
        """
        Reads atomic symbols and coordinates from a file.

        Parameters
        ----------
        file_path : str
            Path to the file containing atomic symbols and coordinates.

        Returns
        -------
        tuple
            Tuple of two elements. The first element is a list of atomic symbols, the second is a 2D numpy array of shape (n_atoms, 3) containing coordinates of atoms.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        atoms, coords = [], []
        with open(file_path, "r") as file:
            for line_num, line in enumerate(file, start=1):
                parts = line.strip().split()
                if not parts:
                    continue
                if len(parts) < 4:
                    raise ValueError(
                        f"Line {line_num} in {file_path} has {len(parts)} columns, "
                        f"expected at least 4 (ELEMENT x y z): {line.strip()!r}"
                    )
                atoms.append(parts[0])
                coords.append([float(x) for x in parts[1:4]])

        return atoms, np.array(coords)


class JsonVDWRadiusProvider(IVDWRadiusProvider):
    def __init__(self, json_file_path: str):
        """
        Initializes the provider from a JSON file.

        Parameters
        ----------
        json_file_path : str
            Path to the JSON file containing van der Waals radii.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        """
        if not os.path.exists(json_file_path):
            raise FileNotFoundError(f"The radius file was not found: {json_file_path}")

        with open(json_file_path, "r") as file:
            self._radii = json.load(file)

    def get_radius(self, atom: str) -> float:
        """
        Returns the van der Waals radius for the given atom.

        Parameters
        ----------
        atom : str
            Atomic symbol

        Returns
        -------
        float
            Van der Waals radius in Angstroms

        Raises
        ------
        KeyError
            If the atom is not found in the database.
        """
        return self._radii.get(atom, DEFAULT_VDW_RADIUS)


class ConvexHullVolumeEstimator(IVolumeEstimator):
    def estimate(
        self,
        atoms: list[str],
        coordinates: np.ndarray,
        vdw_provider: IVDWRadiusProvider,
        grid_resolution: float = 0.1,
    ) -> tuple[float, float, float]:
        """
        Estimates the volume of a molecule.

        Parameters
        ----------
        atoms : list
            List of atomic symbols.
        coordinates : numpy.ndarray
            Array of shape (n_atoms, 3) containing coordinates of atoms in a molecule.
        vdw_provider : IVDWRadiusProvider
            Provider of van der Waals radii.
        grid_resolution : float, optional
            Resolution of the grid for estimation, by default 0.1

        Returns
        -------
        tuple
            Tuple of three floats containing total volume, volume of atoms and volume of cavity.

        Raises
        ------
        ValueError
            If grid_resolution is not positive.
        """
        if grid_resolution <= 0:
            raise ValueError("The grid_resolution parameter must be a positive number.")

        radii = np.array([vdw_provider.get_radius(atom) for atom in atoms])
        return classify_grid_points(coordinates, radii, grid_resolution)


class MoleculeVolumeCalculator:
    def __init__(self, reader: IMoleculeReader, vdw_provider: IVDWRadiusProvider, estimator: IVolumeEstimator):
        """
        Constructor of MoleculeVolumeCalculator

        Parameters
        ----------
        reader : IMoleculeReader
            The reader that reads atomic symbols and coordinates from a file.
        vdw_provider : IVDWRadiusProvider
            The provider that provides Van der Waals radii of atoms.
        estimator : IVolumeEstimator
            The estimator that estimates the volume of a molecule.
        """
        self.reader = reader
        self.vdw_provider = vdw_provider
        self.estimator = estimator

    def calculate(self, molecule_file_path: str, grid_resolution: float = 0.1) -> tuple[float, float, float]:
        """
        Calculates the volume of a molecule.

        Parameters
        ----------
        molecule_file_path : str
            Path to the file containing the molecule.
        grid_resolution : float, optional
            Resolution of the grid for estimation, by default 0.1

        Returns
        -------
        tuple[float, float, float]
            (total_volume, atomic_volume, cavity_volume)
        """
        atoms, coords = self.reader.read(molecule_file_path)
        return self.estimator.estimate(atoms, coords, self.vdw_provider, grid_resolution)

    def calculate_with_data(
        self, molecule_file_path: str, grid_resolution: float = 0.1
    ) -> tuple[list[str], np.ndarray, tuple[float, float, float]]:
        """Calculate volumes and return parsed molecule data alongside results.

        Avoids reading the file twice when both volumes and raw data are needed.

        Returns
        -------
        tuple[list[str], np.ndarray, tuple[float, float, float]]
            (atoms, coordinates, (total_volume, atomic_volume, cavity_volume))
        """
        atoms, coords = self.reader.read(molecule_file_path)
        volumes = self.estimator.estimate(atoms, coords, self.vdw_provider, grid_resolution)
        return atoms, coords, volumes


if __name__ == "__main__":
    vdw_file = os.path.join(get_project_path(), "CalixVolApp", "data", "vdw", "vdw_radius.json")
    molecule_file = os.path.join(get_project_path(), "CalixVolApp", "data", "molecules", "txt_calix", "3.txt")

    reader = MoleculeFileReader()
    vdw_provider = JsonVDWRadiusProvider(vdw_file)
    estimator = ConvexHullVolumeEstimator()
    calculator = MoleculeVolumeCalculator(reader, vdw_provider, estimator)

    total, atom_vol, cavity = calculator.calculate(molecule_file, grid_resolution=0.1)

    print(f"Total volume (convex hull): {total:.2f} Å³")
    print(f"Volume occupied by atoms: {atom_vol:.2f} Å³")
    print(f"Cavity volume: {cavity:.2f} Å³")
